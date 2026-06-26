"""PSD traces of every recording of one subject, with their length-weighted mean.

Drill-down from ``plot_psd_subjects.py``: a subject's trace there is the
length-weighted mean over its recordings; this decomposes that mean into the
individual recordings, so you can see which recording drives an anomalous subject.
For each recording it loads the channel-averaged PSD sidecar and plots it (opacity
scaled by its length weight), with the length-weighted mean overlaid in bold:

    mean(f) = sum_r n_r * psd_r(f) / sum_r n_r       (n_r = recording sample count)

Needs the corpus (engine descriptions) to map subject -> recordings. Match the
sidecar flags (``--bipolar`` / ``--notch_freqs`` / ``--native``); native sidecars
share a grid only within one rate, so ``--sfreq`` is required with ``--native``.

    python src/plot_subject_recordings.py --subject aaaaaanr --sfreq 250 \
        --native --bipolar --notch_freqs 60 120 --fmax 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
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


def _records_for_subject(data_dir, version, subject, sfreq, exclude_seizures=False, min_duration_s=None):
    """(list of edf paths, class) for one subject, optionally filtered."""
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: PLC0415

    tuh = TUHEEGEpilepsy(data_dir=data_dir or str(root / "data"), version=version)
    df = tuh.descriptions
    df = df[df["subject"].astype(str) == str(subject)]
    if df.empty:
        raise SystemExit(f"Subject {subject} not found in the corpus.")
    cls = int(bool(df["epilepsy"].iloc[0]))
    if sfreq is not None:
        df = df[np.isclose(df["sfreq"].astype(float), float(sfreq))]
    if exclude_seizures:
        df = df[df["n_seizure"] == 0]
    if min_duration_s is not None:
        df = df[df["duration"].astype(float) >= float(min_duration_s)]
    if df.empty:
        raise SystemExit(f"Subject {subject} has no recordings after the sfreq/seizure/duration filters.")
    return sorted(str(p) for p in df["path"]), cls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="Subject ID (e.g. aaaaaanr).")
    parser.add_argument("--sfreq", type=float, default=None, help="Restrict to this native sampling rate (Hz).")
    parser.add_argument("--data_dir", default=None, help="Parent of the version folder.")
    parser.add_argument("--version", default="v3.0.0", help="Corpus version subfolder.")
    parser.add_argument("--exclude_seizures", action="store_true",
                        help="Drop this subject's recordings that contain a seizure annotation.")
    parser.add_argument("--min_duration_min", type=float, default=None,
                        help="Drop recordings shorter than this many minutes (e.g. 2 = training window).")
    parser.add_argument("--bipolar", action="store_true", help="Read the bipolar sidecars.")
    parser.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Read the notched sidecars.")
    parser.add_argument("--native", action="store_true", help="Read the native-rate sidecars.")
    parser.add_argument("--fmax", type=float, default=None, help="Max frequency to plot (Hz); default full grid.")
    parser.add_argument("--out", default=None, help="Output PNG (default: ./psd_recordings-<subject>...png).")
    args = parser.parse_args()
    if args.native and args.sfreq is None:
        raise SystemExit("--native requires --sfreq: native-rate sidecars share a grid only within one rate.")
    suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)

    paths, cls = _records_for_subject(
        args.data_dir, args.version, args.subject, args.sfreq,
        exclude_seizures=args.exclude_seizures,
        min_duration_s=None if args.min_duration_min is None else args.min_duration_min * 60.0,
    )

    freqs = None
    recs = []  # (stem, channel-avg psd, n_times, sfreq)
    for p in paths:
        npz = Path(p.replace(".edf", suffix))
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=False)
        if freqs is None:
            freqs = d["freqs"]
        recs.append((Path(p).stem, d["psd"].mean(axis=0), int(d["n_times"]), float(d["sfreq"])))
    if not recs:
        raise SystemExit(f"No {suffix} sidecars for subject {args.subject}; run precompute_psd.py with matching flags.")

    # length-weighted mean = sum(n * psd) / sum(n)  (exactly _subject_psd)
    wsum = sum(n for _s, _c, n, _f in recs)
    mean = sum(n * cm for _s, cm, n, _f in recs) / wsum

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmask = np.ones_like(freqs, dtype=bool) if args.fmax is None else (freqs <= args.fmax)
    fx = freqs[fmask]
    cmap = plt.get_cmap("tab20")
    nmax = max(n for _s, _c, n, _f in recs)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for i, (stem, cm, n, fs) in enumerate(recs):
        alpha = 0.30 + 0.60 * (n / nmax)  # opacity scales with the length weight
        ax.plot(fx, 10.0 * np.log10(cm[fmask] + _EPS), color=cmap(i % 20), lw=0.9, alpha=alpha,
                label=f"{stem}  ({n / fs:.0f}s)")
    ax.plot(fx, 10.0 * np.log10(mean[fmask] + _EPS), color="k", lw=2.4, label="length-weighted mean")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_xlim(0, fx[-1] if len(fx) else None)
    rate = f", native {args.sfreq:g} Hz" if args.sfreq is not None else ""
    ax.set_title(f"subject {args.subject} ({CLASS_NAMES[cls]}{rate}): {len(recs)} recordings, "
                 f"total {wsum / recs[0][3]:.0f}s   -   opacity ~ length weight")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, title="recording (duration)", title_fontsize=7)

    fig.tight_layout()
    tag = suffix.replace("-psd", "").replace(".npz", "")
    scope = f"-sfreq{args.sfreq:g}" if args.sfreq is not None else ""
    out = Path(args.out) if args.out else Path.cwd() / f"psd_recordings-{args.subject}{scope}{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  ({len(recs)} recordings for subject {args.subject})")


if __name__ == "__main__":
    main()
