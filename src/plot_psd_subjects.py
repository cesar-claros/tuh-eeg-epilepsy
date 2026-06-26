"""Per-subject channel-averaged PSD traces (diagnostic for noisy subjects).

For each subject, plots its channel-averaged, length-weighted PSD as one faint line
(colored by class), with the class **mean** (solid) and **median** (dashed) overlaid.
This reveals whether a few outlier subjects (broadband / line-noise / high-frequency
tails) dominate the linear cross-subject mean used by ``plot_psd.py`` and
``kernel_psd.py``: if the mean sits well above the median in a band, that band is
carried by a handful of subjects, not the cohort.

Two subject sources:
  - ``--windows_csv``: the training subjects of one run (default).
  - ``--all-recordings`` / ``--sfreq``: the WHOLE corpus (engine descriptions),
    optionally filtered to one native sampling rate. Use ``--sfreq 250`` to hold the
    acquisition bandwidth fixed and check whether the class difference persists within
    a single sampling-rate group (the confound control). Needs the corpus, like
    ``plot_sfreq.py``.

Reads the same PSD sidecars as the other PSD tools, so pass the matching ``--bipolar``
/ ``--notch_freqs`` you used at precompute time. ``--normalize`` shows each subject at
unit power (compare spectral SHAPE, not loudness).

    python src/plot_psd_subjects.py --windows_csv logs/train/runs/<ts>/windows_train.csv \
        --fmax 80 --bipolar --notch_freqs 60 120
    python src/plot_psd_subjects.py --sfreq 250 --fmax 80 --bipolar --notch_freqs 60 120
    # rank the highlighted subjects by power AND roughness simultaneously
    python src/plot_psd_subjects.py --sfreq 250 --rank_by power roughness --highlight_top 10
    # fuse the two metrics by rank instead of summed z-scores (robust to a heavy tail)
    python src/plot_psd_subjects.py --sfreq 250 --rank_by power roughness --combine rrf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}
_EPS = 1e-30


def _psd_suffix(bipolar: bool = False, notch_freqs=None, native: bool = False) -> str:
    """PSD sidecar suffix. MUST match TUHEEGEpilepsy._psd_suffix (the writer)."""
    s = "-psd"
    if bipolar:
        s += "-bipolar"
    if native:
        s += "-native"
    if notch_freqs:
        s += "-notch-" + "-".join(str(int(round(f))) for f in notch_freqs)
    return s + ".npz"


# Per-subject anomaly metrics (higher metric value -> more extreme). power and
# flatness are two-sided outliers (loud OR quiet, peaky OR white), so they are scored
# by |robust z|; roughness and hf are one-sided (high = anomalous).
_METRICS = ("roughness", "power", "flatness", "hf")
_ABS_METRICS = {"power", "flatness"}
_TAG = {"roughness": "rough", "power": "pow", "flatness": "flat", "hf": "hf"}


def _subject_metrics(freqs, psd, hf_cut: float) -> dict:
    """Per-subject PSD anomaly metrics: roughness, power, flatness, hf fraction."""
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    lp = 10.0 * np.log10(psd + _EPS)
    w = max(5, int(round(5.0 / df)))
    w += 1 - (w % 2)
    smooth = np.convolve(lp, np.ones(w) / w, mode="same")
    return {
        "roughness": float(np.sqrt(np.mean((lp - smooth) ** 2))),   # ripple RMS (dB), wiggliness
        "power": float(10.0 * np.log10(psd.sum() * df + _EPS)),     # log total power (the norm)
        "flatness": float(np.exp(np.mean(np.log(psd + _EPS))) / (psd.mean() + _EPS)),  # 0..1 tonality
        "hf": float(psd[freqs >= hf_cut].sum() / (psd.sum() + _EPS)),  # high-frequency power fraction
    }


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score (robust to the very outliers we are ranking)."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + _EPS
    return (x - med) / (1.4826 * mad)


# How the per-metric anomaly scores are fused into one ranking. Each method returns a
# composite where higher = more anomalous, so the caller can argsort descending.
_COMBINE = ("l1", "l2", "borda", "rrf")


def _desc_rank(s: np.ndarray) -> np.ndarray:
    """1-based descending rank of s (rank 1 = largest value = most anomalous). Ties
    broken arbitrarily but consistently; fine for our continuous z-scores."""
    return np.argsort(np.argsort(-s)) + 1


def _combine_scores(scores: dict, metrics, method: str, rrf_k: float = 60.0) -> np.ndarray:
    """Fuse per-metric anomaly scores (each oriented so higher = more anomalous) into a
    single composite, higher = more anomalous.

    scores[m] is the per-subject score array for metric m (|robust z| for two-sided
    metrics, signed robust z for one-sided). Methods:
      - l1   : sum of scores (the original default). Signed, so an unusually-low
               one-sided score partially cancels; a subject must accumulate evidence.
      - l2   : Euclidean norm of the non-negative (anomalous-direction) scores; rewards
               being extreme on any single axis more than l1. Negatives are clipped to 0
               so 'less anomalous than typical' never inflates the magnitude.
      - borda: sum across metrics of the within-metric rank (Borda count); scale-free
               and robust to one metric's heavy tail (a rank is bounded by n).
      - rrf  : reciprocal rank fusion, sum_m 1 / (rrf_k + rank_m); like borda but
               down-weights all but the top of each metric, very robust to outliers.
    """
    mats = [scores[m] for m in metrics]
    if method == "l1":
        return np.sum(mats, axis=0)
    if method == "l2":
        pos = [np.maximum(m, 0.0) for m in mats]
        return np.sqrt(np.sum([m ** 2 for m in pos], axis=0))
    if method == "borda":
        # ascending rank (0..n-1): largest score -> largest points -> most anomalous.
        return np.sum([np.argsort(np.argsort(m)) for m in mats], axis=0).astype(float)
    if method == "rrf":
        return np.sum([1.0 / (rrf_k + _desc_rank(m)) for m in mats], axis=0)
    raise ValueError(f"unknown combine method: {method}")


def _subject_psd(recordings, suffix: str):
    """Length-weighted, channel-averaged mean PSD over a subject's recordings."""
    acc = None
    wsum = 0.0
    freqs = None
    for edf in recordings:
        npz = Path(str(edf).replace(".edf", suffix))
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=False)
        cm = d["psd"].mean(axis=0)  # channel-average (kernels are channel-agnostic)
        n = int(d["n_times"])
        if freqs is None:
            freqs = d["freqs"]
        acc = n * cm if acc is None else acc + n * cm
        wsum += n
    if acc is None:
        return None, None
    return freqs, acc / wsum


def _subjects_from_windows(windows_csv: str) -> dict:
    """{subject: (recordings, class)} from a run's windows_train.csv."""
    df = pd.read_csv(windows_csv)
    return {
        str(subj): (sorted(set(g["path"].astype(str))), int(bool(g["epilepsy"].iloc[0])))
        for subj, g in df.groupby("subject")
    }


def _subjects_from_corpus(data_dir, version, sfreq, exclude_seizures=False, min_duration_s=None) -> dict:
    """{subject: (recordings, class)} from the whole corpus, optionally filtered.

    Reads the engine descriptions; a subject is kept if it has any recording passing
    the filters, and only those recordings are aggregated. ``sfreq`` keeps one native
    rate; ``exclude_seizures`` drops recordings with a seizure annotation;
    ``min_duration_s`` drops recordings shorter than that. Imports the engine lazily so
    the windows-CSV mode stays light.
    """
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: PLC0415

    # add_annotations=True so descriptions carries n_seizure / duration (needed below).
    tuh = TUHEEGEpilepsy(data_dir=data_dir or str(root / "data"), version=version, add_annotations=True)
    df = tuh.descriptions[["path", "subject", "epilepsy", "sfreq", "n_seizure", "duration"]].copy()
    df["epilepsy"] = df["epilepsy"].astype(bool).astype(int)
    if sfreq is not None:
        df = df[np.isclose(df["sfreq"].astype(float), float(sfreq))]
    if exclude_seizures:
        df = df[df["n_seizure"] == 0]
    if min_duration_s is not None:
        df = df[df["duration"].astype(float) >= float(min_duration_s)]
    if df.empty:
        raise SystemExit("No recordings left after the sfreq/seizure/duration filters.")
    return {
        str(subj): (sorted(set(g["path"].astype(str))), int(g["epilepsy"].iloc[0]))
        for subj, g in df.groupby("subject")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows_csv", default=None, help="windows_train.csv from a run (training-subject source).")
    parser.add_argument("--all-recordings", action="store_true", dest="all_recordings",
                        help="Use the WHOLE corpus instead of a windows CSV.")
    parser.add_argument("--sfreq", type=float, default=None,
                        help="Corpus mode: keep only recordings at this native sampling rate (Hz), e.g. 250.")
    parser.add_argument("--data_dir", default=None, help="Corpus mode: parent of the version folder.")
    parser.add_argument("--version", default="v3.0.0", help="Corpus mode: corpus version subfolder.")
    parser.add_argument("--exclude_seizures", action="store_true",
                        help="Corpus mode: drop recordings with a seizure annotation (matters for the means).")
    parser.add_argument("--min_duration_min", type=float, default=None,
                        help="Corpus mode: drop recordings shorter than this many minutes (e.g. 2 = training window).")
    parser.add_argument("--out", default=None, help="Output PNG (default: auto-named).")
    parser.add_argument("--out_dir", default=None, help="Corpus mode: output directory (default: cwd).")
    parser.add_argument("--fmax", type=float, default=None, help="Max frequency to plot (Hz); default full grid.")
    parser.add_argument("--bipolar", action="store_true", help="Read the bipolar sidecars.")
    parser.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Read the notched sidecars.")
    parser.add_argument(
        "--native", action="store_true",
        help="Read the native-rate sidecars (-psd-native...; from precompute_psd.py --native). "
        "Requires --sfreq, since native PSDs share a grid only within one sampling rate.",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Per-subject unit power: compare spectral SHAPE, not loudness.",
    )
    parser.add_argument(
        "--highlight_top", type=int, default=10,
        help="Highlight the N most anomalous subject traces with distinct colors and "
        "their IDs in the legend. 0 disables.",
    )
    parser.add_argument(
        "--rank_by", nargs="+", default=["roughness"], choices=list(_METRICS),
        help="Metric(s) the highlight ranks by; pass >1 to rank by them simultaneously "
        "(sum of robust z-scores), e.g. '--rank_by power roughness' for subjects that are "
        "both energy outliers AND wiggly. roughness=wiggliness, power=energy outlier (loud "
        "OR quiet), flatness=tonality outlier, hf=high-frequency power fraction. The legend "
        "shows each chosen metric's value per subject. Default: roughness.",
    )
    parser.add_argument(
        "--combine", default="l1", choices=list(_COMBINE),
        help="How to fuse >1 --rank_by metric into one ranking (no effect on a single "
        "metric). l1=sum of robust z (default), l2=Euclidean norm of the anomalous-side "
        "scores (rewards any single extreme), borda=sum of within-metric ranks "
        "(scale-free, robust), rrf=reciprocal rank fusion (most robust to a heavy tail).",
    )
    parser.add_argument(
        "--rrf_k", type=float, default=60.0,
        help="Reciprocal rank fusion constant for --combine rrf (larger = flatter "
        "weighting across ranks). Default 60.",
    )
    parser.add_argument(
        "--hf_cut", type=float, default=None,
        help="Frequency (Hz) above which power is 'high-frequency' for the hf metric "
        "(default: grid Nyquist / 2).",
    )
    args = parser.parse_args()
    if args.native and args.sfreq is None:
        raise SystemExit("--native requires --sfreq <rate>: native-rate PSDs share a grid only within one sampling rate.")
    suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)

    corpus_mode = args.all_recordings or args.sfreq is not None
    if corpus_mode:
        subj_recs = _subjects_from_corpus(
            args.data_dir, args.version, args.sfreq,
            exclude_seizures=args.exclude_seizures,
            min_duration_s=None if args.min_duration_min is None else args.min_duration_min * 60.0,
        )
    elif args.windows_csv:
        subj_recs = _subjects_from_windows(args.windows_csv)
    else:
        raise SystemExit("Provide --windows_csv, or --all-recordings / --sfreq for the whole corpus.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = None
    per_class: dict = {0: [], 1: []}  # cls -> list of (subject_id, psd)
    for subj, (recs, cls) in subj_recs.items():
        f, psd = _subject_psd(recs, suffix)
        if f is None:
            continue
        if freqs is None:
            freqs = f
        if args.normalize:
            psd = psd / (psd.sum() + _EPS)
        per_class[cls].append((str(subj), psd))
    if freqs is None:
        raise SystemExit(f"No {suffix} sidecars found; run precompute_psd.py with the matching flags first.")

    # Rank ALL subjects by the chosen metric(s); top-N (global) get highlighted.
    all_subj = [(cls, subj, psd) for cls in (0, 1) for subj, psd in per_class[cls]]
    hf_cut = args.hf_cut if args.hf_cut is not None else float(freqs[-1]) / 2.0
    mv = [_subject_metrics(freqs, psd, hf_cut) for _cls, _subj, psd in all_subj]
    raw = {m: np.array([d[m] for d in mv]) for m in _METRICS}
    z = {m: (np.abs(_robust_z(raw[m])) if m in _ABS_METRICS else _robust_z(raw[m])) for m in _METRICS}
    composite = _combine_scores(z, args.rank_by, args.combine, args.rrf_k) if all_subj else np.array([])
    order = np.argsort(composite)[::-1] if len(composite) else []
    single = args.rank_by[0] if len(args.rank_by) == 1 else None
    hl_rank: dict = {}  # (cls, subj) -> (rank 1-based, color index, label tag)
    for rank, i in enumerate(order[: max(0, args.highlight_top)]):
        cls, subj, _psd = all_subj[i]
        # Single metric: show its raw value. Multiple metrics (e.g. power+roughness):
        # the order is the composite robust z, but show each metric's raw value so the
        # combined ranking is interpretable (which subjects are loud AND wiggly).
        tag = (
            f"{_TAG[single]}={raw[single][i]:.2f}" if single
            else " ".join(f"{_TAG[m]}={raw[m][i]:.2f}" for m in args.rank_by)
        )
        hl_rank[(cls, subj)] = (rank + 1, rank, tag)

    fmask = np.ones_like(freqs, dtype=bool) if args.fmax is None else (freqs <= args.fmax)
    fx = freqs[fmask]
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True, sharey=True)
    for cls in (0, 1):
        ax = axes[cls]
        subj_psd = per_class[cls]
        for subj, psd in subj_psd:  # faint background = non-highlighted subjects
            if (cls, subj) in hl_rank:
                continue
            ax.plot(fx, 10.0 * np.log10(psd[fmask] + _EPS), color=CLASS_COLORS[cls], lw=0.4, alpha=0.16)
        for subj, psd in sorted(  # highlighted, in rank order, distinct colors + IDs
            (sp for sp in subj_psd if (cls, sp[0]) in hl_rank), key=lambda sp: hl_rank[(cls, sp[0])][0]
        ):
            rk, ci, tag = hl_rank[(cls, subj)]
            ax.plot(fx, 10.0 * np.log10(psd[fmask] + _EPS), color=cmap(ci % 10), lw=1.2, alpha=0.95,
                    label=f"#{rk} {subj} ({tag})")
        if subj_psd:  # mean / median over ALL subjects in the class
            stack = np.stack([psd for _subj, psd in subj_psd])
            ax.plot(fx, 10.0 * np.log10(np.mean(stack, axis=0)[fmask] + _EPS), color="k", lw=1.8, label="class mean")
            ax.plot(fx, 10.0 * np.log10(np.median(stack, axis=0)[fmask] + _EPS), color="k", lw=1.3, ls="--", label="class median")
        ax.set_title(f"{CLASS_NAMES[cls]} (n={len(subj_psd)})")
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(0, fx[-1] if len(fx) else None)
        ax.legend(fontsize=6, loc="upper right", ncol=1)
    axes[0].set_ylabel(("relative " if args.normalize else "") + "PSD (dB)")
    scope = f"native {args.sfreq:g} Hz" if args.sfreq is not None else ("whole corpus" if corpus_mode else "training subjects")
    by = "+".join(args.rank_by) + (f" [{args.combine}]" if len(args.rank_by) > 1 else "")
    hl_note = f"; top {len(hl_rank)} by {by} highlighted (#rank ID metric)" if hl_rank else ""
    fig.suptitle(
        f"Per-subject channel-averaged PSD ({scope})"
        + (" [unit-power]" if args.normalize else "")
        + "  -  mean above median = a few subjects carry that band"
        + hl_note
    )
    fig.tight_layout()

    tag = suffix.replace("-psd", "").replace(".npz", "")
    scope_tag = f"-sfreq{args.sfreq:g}" if args.sfreq is not None else ("-allrates" if corpus_mode else "")
    name = f"psd_subjects{tag}{scope_tag}{'-norm' if args.normalize else ''}.png"
    if args.out:
        out = Path(args.out)
    elif corpus_mode:
        out = (Path(args.out_dir) if args.out_dir else Path.cwd()) / name
    else:
        out = Path(args.windows_csv).parent / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  ({len(per_class[0])} no-epilepsy + {len(per_class[1])} epilepsy subjects)")


if __name__ == "__main__":
    main()
