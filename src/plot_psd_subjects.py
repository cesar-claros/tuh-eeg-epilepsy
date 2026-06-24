"""Per-subject channel-averaged PSD traces (diagnostic for noisy subjects).

For each training subject, plots its channel-averaged, length-weighted PSD as one
faint line (colored by class), with the class **mean** (solid) and **median**
(dashed) overlaid. This reveals whether a few outlier subjects (broadband / line-
noise / high-frequency tails) dominate the linear cross-subject mean used by
``plot_psd.py`` and ``kernel_psd.py``: if the mean sits well above the median in a
band, that band is carried by a handful of subjects, not the cohort.

Reads the same PSD sidecars as the other PSD tools, so pass the matching
``--bipolar`` / ``--notch_freqs`` you used at precompute time. ``--normalize`` shows
each subject at unit power (compare spectral SHAPE, not loudness).

    python src/plot_psd_subjects.py --windows_csv logs/train/runs/<ts>/windows_train.csv \
        --fmax 80 --bipolar --notch_freqs 60 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}
_EPS = 1e-30


def _psd_suffix(bipolar: bool = False, notch_freqs=None) -> str:
    """PSD sidecar suffix. MUST match TUHEEGEpilepsy._psd_suffix (the writer)."""
    s = "-psd"
    if bipolar:
        s += "-bipolar"
    if notch_freqs:
        s += "-notch-" + "-".join(str(int(round(f))) for f in notch_freqs)
    return s + ".npz"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows_csv", required=True, help="windows_train.csv from a run.")
    parser.add_argument("--out", default=None, help="Output PNG (default: <windows_csv dir>/psd_subjects*.png).")
    parser.add_argument("--fmax", type=float, default=None, help="Max frequency to plot (Hz); default full grid.")
    parser.add_argument("--bipolar", action="store_true", help="Read the bipolar sidecars.")
    parser.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Read the notched sidecars.")
    parser.add_argument(
        "--normalize", action="store_true",
        help="Per-subject unit power: compare spectral SHAPE, not loudness.",
    )
    args = parser.parse_args()
    suffix = _psd_suffix(args.bipolar, args.notch_freqs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.windows_csv)
    freqs = None
    per_class: dict = {0: [], 1: []}
    subj_ids: dict = {0: [], 1: []}
    for subj, g in df.groupby("subject"):
        cls = int(bool(g["epilepsy"].iloc[0]))
        f, psd = _subject_psd(sorted(set(g["path"])), suffix)
        if f is None:
            continue
        if freqs is None:
            freqs = f
        if args.normalize:
            psd = psd / (psd.sum() + _EPS)
        per_class[cls].append(psd)
        subj_ids[cls].append(str(subj))
    if freqs is None:
        raise SystemExit(f"No {suffix} sidecars found; run precompute_psd.py with the matching flags first.")

    fmask = np.ones_like(freqs, dtype=bool) if args.fmax is None else (freqs <= args.fmax)
    fx = freqs[fmask]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True, sharey=True)
    for cls in (0, 1):
        ax = axes[cls]
        arrs = per_class[cls]
        for psd in arrs:
            ax.plot(fx, 10.0 * np.log10(psd[fmask] + _EPS), color=CLASS_COLORS[cls], lw=0.4, alpha=0.25)
        if arrs:
            stack = np.stack(arrs)
            mean_db = 10.0 * np.log10(np.mean(stack, axis=0)[fmask] + _EPS)
            med_db = 10.0 * np.log10(np.median(stack, axis=0)[fmask] + _EPS)
            ax.plot(fx, mean_db, color="k", lw=1.8, label="class mean")
            ax.plot(fx, med_db, color="k", lw=1.3, ls="--", label="class median")
        ax.set_title(f"{CLASS_NAMES[cls]} (n={len(arrs)})")
        ax.set_xlabel("frequency (Hz)")
        ax.set_xlim(0, fx[-1] if len(fx) else None)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel(("relative " if args.normalize else "") + "PSD (dB)")
    fig.suptitle(
        "Per-subject channel-averaged PSD"
        + (" (unit-power)" if args.normalize else "")
        + "  -  mean above median = a few subjects carry that band"
    )
    fig.tight_layout()
    tag = suffix.replace("-psd", "").replace(".npz", "")
    name = f"psd_subjects{tag}{'-norm' if args.normalize else ''}.png"
    out = Path(args.out) if args.out else Path(args.windows_csv).parent / name
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  ({len(per_class[0])} no-epilepsy + {len(per_class[1])} epilepsy subjects)")


if __name__ == "__main__":
    main()
