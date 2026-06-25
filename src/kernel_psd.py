"""Relate the top classifier HYDRA kernels to the training-set PSD.

For a run, this reads the training subjects' cached PSDs (``-psd.npz`` from
``precompute_psd.py``) and the ``top_classifier_kernels.csv`` (kernels ranked by
classifier ``|coef|``), recomputes each kernel's magnitude response ``|H(f)|`` from
its saved weights / dilation / representation (View B: the first-difference
high-pass is folded into ``diff`` kernels, so every spectrum is the response to the
raw signal x), and writes two figures:

1. ``kernel_sensitivity_<csv>.png`` (Panel A): the channel-averaged, class-split
   **relative** PSD (epilepsy vs no-epilepsy, each normalized to unit total power,
   dB) with the kernels' classifier-weighted aggregate spectral sensitivity
   ``sum_k |coef_k| * |H_k(f)|^2`` overlaid -- where the selected kernels look,
   relative to where the class spectra differ.

2. ``kernel_taps_psd_<csv>.png``: one row per top-N kernel, the kernel's 9 taps on
   the left and, on the right, the **PSD after that kernel is applied to the raw
   signal**, per class. The filtered spectrum is ``|H(f)|^2 * PSD_rel_class(f)``
   (the PSD of the kernel output, by the filter power theorem), renormalized to unit
   power so it shows how the kernel *reshapes* the spectrum; the dashed grey line is
   the input (pre-filter) relative PSD for reference.

PSDs are normalized to unit power PER CLASS first, so comparisons reflect spectral
SHAPE, not the broadband amplitude/loudness difference between cohorts. Kernels are
compared against the channel-averaged PSD because multichannel HYDRA mixes random
channel subsets (the reading is spectral, not spatial).

This is a sensitivity / response view: HYDRA features are competition win-counts,
not filtered power, so ``|H(f)|`` says which frequencies drive a kernel and the
filtered PSD shows what its output spectrum looks like, not the literal feature.

Example
-------
::

    python src/kernel_psd.py \
        --windows_csv logs/train/runs/<ts>/windows_train.csv \
        --kernels_csv logs/train/runs/<ts>/top_classifier_kernels.csv \
        --out_dir logs/train/runs/<ts>
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


def _load_recording_psd(edf_path: str, suffix: str = "-psd.npz"):
    """Load a recording's PSD sidecar (``suffix``) channel-averaged, or None."""
    npz = Path(str(edf_path).replace(".edf", suffix))
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=False)
    # Channel-average immediately: kernels mix random channel subsets, so the
    # relevant input spectrum is the mean over channels (shape, not per-sensor).
    return d["freqs"], d["psd"].mean(axis=0), int(d["n_times"])


def _class_relative_psd(windows_csv: str, suffix: str = "-psd.npz"):
    """Channel-averaged, subject-aggregated, per-class relative PSD.

    Returns ``(freqs, {0: psd_rel, 1: psd_rel}, {0: n_subj, 1: n_subj})`` with each
    class PSD normalized to unit total power over the full frequency grid. ``suffix``
    selects the sidecar (``-psd.npz`` referential, ``-psd-bipolar.npz`` bipolar).
    """
    df = pd.read_csv(windows_csv)
    freqs = None
    per_class_subjects: dict = {0: [], 1: []}
    for _, g in df.groupby("subject"):
        cls = int(bool(g["epilepsy"].iloc[0]))
        acc = None
        wsum = 0.0
        for edf in sorted(set(g["path"])):
            out = _load_recording_psd(edf, suffix)
            if out is None:
                continue
            f, cm, n = out
            if freqs is None:
                freqs = f
            acc = n * cm if acc is None else acc + n * cm
            wsum += n
        if acc is not None and wsum > 0:
            per_class_subjects[cls].append(acc / wsum)
    if freqs is None:
        raise SystemExit("No -psd.npz sidecars found; run precompute_psd.py first.")

    rel: dict = {}
    counts: dict = {}
    for cls in (0, 1):
        subs = per_class_subjects[cls]
        counts[cls] = len(subs)
        if subs:
            mean = np.mean(np.stack(subs), axis=0)
            rel[cls] = mean / mean.sum()  # unit total power per class
    return freqs, rel, counts


def _kernel_response(row: pd.Series, freqs: np.ndarray, sfreq: float) -> np.ndarray:
    """``|H(f)|`` of one kernel on the PSD grid (View B for 'diff').

    Evaluates over the full band (comb replicas included). For 'diff' kernels the
    undilated first-difference high-pass ``2|sin(pi f / sfreq)|`` is applied, so the
    result is the kernel's effective response on the raw signal x.
    """
    w = np.array([row[f"w{i}"] for i in range(9)], dtype=float)
    d = float(row["dilation"])
    j = np.arange(9)
    phase = (2.0 * np.pi * d / sfreq) * np.outer(freqs, j)  # (n_freqs, 9)
    mag = np.abs((w[None, :] * np.exp(-1j * phase)).sum(axis=1))
    if str(row.get("representation", "raw")) == "diff":
        mag = mag * (2.0 * np.abs(np.sin(np.pi * freqs / sfreq)))
    return mag


def _panel_a_figure(kdf, freqs, rel, counts, sfreq, fmax, out_path):
    """Panel A: class relative PSD (dB) + classifier-weighted kernel sensitivity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H2 = np.stack([_kernel_response(r, freqs, sfreq) ** 2 for _, r in kdf.iterrows()])
    weights = kdf["count"].to_numpy(dtype=float) if "count" in kdf.columns else np.ones(len(kdf))
    sens = (weights[:, None] * H2).sum(axis=0)
    sens = sens / sens.max()

    fmask = freqs <= fmax
    fx = freqs[fmask]
    fig, axA = plt.subplots(figsize=(7.5, 3.6))
    for cls in (0, 1):
        axA.plot(
            fx, 10.0 * np.log10(rel[cls][fmask] + _EPS),
            color=CLASS_COLORS[cls], lw=1.4, label=f"{CLASS_NAMES[cls]} (n={counts[cls]})",
        )
    axA.set_xlabel("frequency (Hz)")
    axA.set_ylabel("relative PSD (dB)")
    axA.set_xlim(0, fmax)
    axA.legend(fontsize=8, loc="upper right")

    axS = axA.twinx()
    axS.fill_between(fx, 0, sens[fmask], color="0.5", alpha=0.18)
    axS.plot(fx, sens[fmask], color="0.30", lw=1.4, label=r"kernel sensitivity")
    axS.set_ylim(0, 1.05)
    axS.set_ylabel(r"classifier $\sum_k|c_k|\,|H_k(f)|^2$ (norm.)")
    axS.legend(fontsize=8, loc="upper left")
    axA.set_title("Top classifier kernels: spectral sensitivity vs class PSD", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _kernel_grid_figure(kdf, freqs, rel, sfreq, fmax, out_path, left_mode):
    """5x4 grid, one cell per kernel: a left panel + the filtered PSD (right).

    ``left_mode`` selects the left panel:
      - ``"taps"``     : the kernel's 9 weights (time domain).
      - ``"response"`` : the kernel's effective magnitude response |H(f)| on the raw
                         signal (View B, linear, max-normalized), sharing the right
                         panel's frequency axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(kdf)
    kpr = 2  # kernel cells per row -> 5 rows x 4 columns for 10 kernels
    n_rows = (n + kpr - 1) // kpr
    fmask = freqs <= fmax
    fx = freqs[fmask]
    input_ref = 0.5 * (rel[0] + rel[1])  # class-mean relative PSD (pre-filter reference)

    left_w = 1.0 if left_mode == "taps" else 1.7  # response panel needs room for the comb
    fig = plt.figure(figsize=((5.6 + left_w) * kpr, 1.35 * n_rows + 0.6))
    gs = GridSpec(n_rows, 2 * kpr, width_ratios=[left_w, 2.6] * kpr, wspace=0.34, hspace=0.42)
    rows = list(kdf.iterrows())
    for i, (_, row) in enumerate(rows):
        r, cp = divmod(i, kpr)  # grid row, and which kernel cell within the row
        top = r == 0            # column titles on the first grid row
        show_x = (i + kpr) >= n  # x labels only when no kernel sits directly below
        w = np.array([row[f"w{j}"] for j in range(9)], dtype=float)
        dil = int(row["dilation"])
        rep = str(row.get("representation", "raw"))
        H = _kernel_response(row, freqs, sfreq)  # |H(f)| on the raw signal (View B)

        # --- left: taps (time domain) or magnitude response (frequency domain) ---
        axk = fig.add_subplot(gs[r, cp * 2])
        if left_mode == "taps":
            axk.stem(np.arange(9), w, basefmt=" ", linefmt="0.4", markerfmt="o")
            axk.axhline(0.0, color="0.7", lw=0.6)
            axk.set_xticks([0, 4, 8])
            x_label, left_title = "tap", "kernel taps"
        else:
            Hn = H / (H[fmask].max() + _EPS)  # normalize to the in-band peak
            axk.fill_between(fx, 0, Hn[fmask], color="0.55", alpha=0.20)
            axk.plot(fx, Hn[fmask], color="0.30", lw=1.1)
            axk.set_xlim(0, fmax)
            axk.set_ylim(0, 1.05)
            x_label, left_title = "frequency (Hz)", r"kernel $|H(f)|$ (norm.)"
        axk.tick_params(labelsize=7)
        axk.set_ylabel(f"#{i + 1}", fontsize=8, rotation=0, labelpad=12, va="center")
        if not show_x:
            axk.set_xticklabels([])
        else:
            axk.set_xlabel(x_label, fontsize=8)
        if top:
            axk.set_title(left_title, fontsize=9)

        # --- right: PSD after the kernel is applied to the raw signal ---
        H2 = H ** 2
        axp = fig.add_subplot(gs[r, cp * 2 + 1])
        in_db = 10.0 * np.log10(input_ref[fmask] + _EPS)
        axp.plot(fx, in_db, color="0.6", lw=0.8, ls="--", label="input PSD" if i == 0 else None)
        peak = in_db.max()
        for cls in (0, 1):
            out = H2 * rel[cls]
            out = out / (out.sum() + _EPS)  # renormalize: how the kernel reshapes the spectrum
            out_db = 10.0 * np.log10(out[fmask] + _EPS)
            peak = max(peak, out_db.max())
            axp.plot(fx, out_db, color=CLASS_COLORS[cls], lw=1.1,
                     label=CLASS_NAMES[cls] if i == 0 else None)
        # Clip the floor so the comb nulls (exact zeros -> -inf dB) do not crush the
        # visible shape; ~45 dB of dynamic range shows the reshaping and the teeth.
        axp.set_ylim(peak - 45.0, peak + 3.0)
        axp.set_xlim(0, fmax)
        axp.tick_params(labelsize=7)
        axp.text(0.985, 0.90, f"d={dil}, {rep}", transform=axp.transAxes, ha="right", va="top",
                 fontsize=7, color="0.25")
        if not show_x:
            axp.set_xticklabels([])
        else:
            axp.set_xlabel("frequency (Hz)", fontsize=8)
        if top:
            axp.set_title("PSD after kernel applied to raw signal (relative, dB)", fontsize=9)
        if i == 0:
            axp.legend(fontsize=6.5, loc="lower left", ncol=3, columnspacing=1.0, handlelength=1.2)

    left_name = "taps" if left_mode == "taps" else "frequency response"
    fig.suptitle(f"Top {n} classifier kernels: {left_name} and filtered PSD", fontsize=11, y=0.997)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows_csv", required=True, help="windows_train.csv from a run.")
    parser.add_argument("--kernels_csv", required=True, help="top_classifier_kernels.csv from the run.")
    parser.add_argument("--out_dir", default=None, help="Output dir (default: windows_csv dir).")
    parser.add_argument("--fmax", type=float, default=60.0, help="Max frequency to plot (Hz).")
    parser.add_argument("--top_taps", type=int, default=10, help="Kernels shown in the taps figure.")
    parser.add_argument(
        "--kernel_sfreq", type=float, default=256.0,
        help="Sample rate (Hz) the kernels were applied at (the resampled window rate); "
        "must match the training window rate (the corpus-min sfreq).",
    )
    parser.add_argument(
        "--bipolar", action="store_true",
        help="Use the bipolar PSD sidecars (-psd-bipolar.npz). Pair with a run / "
        "kernels CSV trained on data.signal_mode=bipolar.",
    )
    parser.add_argument(
        "--notch_freqs", type=float, nargs="+", default=None,
        help="Use the notched PSD sidecars (must match precompute_psd.py "
        "--notch_freqs), e.g. --notch_freqs 60 120.",
    )
    parser.add_argument(
        "--native", action="store_true",
        help="Use the native-rate sidecars (precompute_psd.py --native); only coherent "
        "when the run's recordings share one native rate.",
    )
    args = parser.parse_args()

    suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)
    freqs, rel, counts = _class_relative_psd(args.windows_csv, suffix)
    if 0 not in rel or 1 not in rel:
        raise SystemExit(f"Need both classes in training PSDs; got counts {counts}.")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.windows_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.kernels_csv).stem

    kdf = pd.read_csv(args.kernels_csv)
    a_path = out_dir / f"kernel_sensitivity_{stem}.png"
    _panel_a_figure(kdf, freqs, rel, counts, args.kernel_sfreq, args.fmax, a_path)
    print(f"wrote {a_path}")

    top = kdf.head(args.top_taps)
    n_top = min(args.top_taps, len(kdf))
    for left_mode, tag in (("taps", "taps"), ("response", "response")):
        path = out_dir / f"kernel_{tag}_psd_{stem}.png"
        _kernel_grid_figure(top, freqs, rel, args.kernel_sfreq, args.fmax, path, left_mode)
        print(f"wrote {path}  (top {n_top} kernels)")


if __name__ == "__main__":
    main()
