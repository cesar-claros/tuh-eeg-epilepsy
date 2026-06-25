"""Plot the training-set average PSD per channel, split by class.

Reads a run's ``windows_train.csv`` (subject / path / epilepsy), loads the cached
per-recording PSDs (``-psd.npz`` from ``precompute_psd.py``), aggregates to one PSD
per subject (length-weighted mean over the subject's recordings), then averages the
subject PSDs within each class. Draws a grid with one channel per subplot, the
epilepsy and no-epilepsy averages overlaid (PSD in dB). Aggregating to the subject
first keeps recording-heavy subjects from dominating. Run on the machine that holds
the corpus and the ``-psd.npz`` sidecars.

Example
-------
::

    python src/plot_psd.py --windows_csv logs/train/runs/<ts>/windows_train.csv \
        --out logs/train/runs/<ts>/psd_by_channel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}


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


def _load_recording_psd(edf_path: str, suffix: str = "-psd.npz"):
    """Load a recording's PSD sidecar (``suffix``), or None if missing."""
    npz = Path(str(edf_path).replace(".edf", suffix))
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=False)
    return d["freqs"], d["psd"], [str(c) for c in d["channels"]], int(d["n_times"])


def _subject_psd(recordings, suffix: str = "-psd.npz"):
    """Length-weighted mean PSD per channel over one subject's recordings.

    Returns ``(freqs, {channel: psd})``, or ``(None, {})`` if nothing loaded.
    """
    acc: dict = {}
    wsum: dict = {}
    freqs = None
    for edf in recordings:
        out = _load_recording_psd(edf, suffix)
        if out is None:
            continue
        f, psd, chans, n = out
        if freqs is None:
            freqs = f
        for i, c in enumerate(chans):
            acc[c] = acc.get(c, 0.0) + n * psd[i]
            wsum[c] = wsum.get(c, 0.0) + n
    return freqs, {c: acc[c] / wsum[c] for c in acc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows_csv", required=True, help="windows_train.csv from a run."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output PNG (default: <windows_csv dir>/psd_by_channel.png).",
    )
    parser.add_argument(
        "--fmax", type=float, default=None,
        help="Maximum frequency to plot (Hz); default the full grid (Nyquist).",
    )
    parser.add_argument(
        "--ncols", type=int, default=4, help="Columns in the channel grid."
    )
    parser.add_argument(
        "--bipolar", action="store_true",
        help="Read the bipolar PSD sidecars (-psd-bipolar.npz from "
        "precompute_psd.py --bipolar) instead of the referential -psd.npz.",
    )
    parser.add_argument(
        "--notch_freqs", type=float, nargs="+", default=None,
        help="Read the notched sidecars (must match precompute_psd.py --notch_freqs), "
        "e.g. --notch_freqs 60 120.",
    )
    parser.add_argument(
        "--native", action="store_true",
        help="Read the native-rate sidecars (precompute_psd.py --native). Only coherent "
        "when the recordings share one native rate (per-channel aggregation assumes one grid).",
    )
    args = parser.parse_args()
    suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.windows_csv)
    # Each training subject -> its recordings and class label.
    recs_by_subject: dict = {}
    class_by_subject: dict = {}
    for subj, g in df.groupby("subject"):
        recs_by_subject[subj] = sorted(set(g["path"]))
        class_by_subject[subj] = int(bool(g["epilepsy"].iloc[0]))

    # Subject PSDs -> per-class lists of subject PSDs, per channel.
    per_class: dict = {0: {}, 1: {}}
    freqs = None
    for subj, recs in recs_by_subject.items():
        f, subj_psd = _subject_psd(recs, suffix)
        if f is None or not subj_psd:
            continue
        if freqs is None:
            freqs = f
        cls = class_by_subject[subj]
        for c, psd in subj_psd.items():
            per_class[cls].setdefault(c, []).append(psd)
    if freqs is None:
        raise SystemExit(f"No {suffix} sidecars found; run precompute_psd.py with the matching flags first.")

    channels = sorted(set(per_class[0]) | set(per_class[1]))
    fmask = np.ones_like(freqs, dtype=bool) if args.fmax is None else (freqs <= args.fmax)
    fx = freqs[fmask]

    nrows = int(np.ceil(len(channels) / args.ncols))
    fig, axes = plt.subplots(
        nrows, args.ncols, figsize=(3.2 * args.ncols, 2.2 * nrows),
        squeeze=False, sharex=True,
    )
    for k, ch in enumerate(channels):
        ax = axes[k // args.ncols][k % args.ncols]
        for cls in (0, 1):
            arrs = per_class[cls].get(ch)
            if not arrs:
                continue
            mean = np.mean(np.stack(arrs), axis=0)[fmask]
            ax.plot(
                fx, 10.0 * np.log10(mean + 1e-20), color=CLASS_COLORS[cls],
                lw=1.0, label=CLASS_NAMES[cls] if k == 0 else None,
            )
        n0 = len(per_class[0].get(ch, []))
        n1 = len(per_class[1].get(ch, []))
        ax.set_title(f"{ch}  (n={n1}/{n0})", fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(len(channels), nrows * args.ncols):
        axes[k // args.ncols][k % args.ncols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.supxlabel("frequency (Hz)")
    fig.supylabel("PSD (dB)")
    fig.suptitle("Training-set average PSD per channel (epilepsy n / no-epilepsy n)")
    fig.tight_layout()

    # Name the figure after the sidecar variant (e.g. psd_by_channel-bipolar-notch-60-120.png).
    tag = suffix.replace("-psd", "").replace(".npz", "")
    default_name = f"psd_by_channel{tag}.png"
    out = Path(args.out) if args.out else Path(args.windows_csv).parent / default_name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
