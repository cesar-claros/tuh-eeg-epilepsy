"""Relate the top HYDRA kernels to the training-set PSD (panels A + B).

For a run, this reads the training subjects' cached PSDs (``-psd.npz`` from
``precompute_psd.py``) and a top-kernel CSV (``top_kernels.csv`` /
``top_discriminative_kernels.csv`` / ``top_classifier_kernels.csv``), recomputes
each kernel's magnitude response ``|H(f)|`` from its saved weights / dilation /
representation (View B: the first-difference high-pass is folded into ``diff``
kernels so every spectrum is the response to the raw signal), and draws, per kernel
CSV, one figure with:

- Panel A: the channel-averaged, class-split **relative** PSD (epilepsy vs
  no-epilepsy, each normalized to unit total power, in dB) with the kernels'
  aggregate spectral sensitivity ``sum |H(f)|^2`` overlaid, so you can see where the
  selected kernels look relative to where the class spectra differ.
- Panel B: each kernel's class power log-ratio ``10 log10(P_epi / P_noepi)`` where
  ``P_class = integral |H(f)|^2 * PSD_rel_class(f) df`` (Parseval: the kernel's
  expected output power under the class spectral shape). Positive = the kernel sees
  relatively more power in epilepsy.

PSDs are normalized to unit power PER CLASS first, so the comparison reflects
spectral SHAPE, not the broadband amplitude/loudness difference between cohorts.
Kernels are compared against the channel-averaged PSD because multichannel HYDRA
mixes random channel subsets (the reading is spectral, not spatial).

This is a sensitivity / response view: HYDRA features are competition win-counts,
not filtered power, so ``|H(f)|`` says which frequencies drive a kernel, and
``P_class`` is a principled proxy for how strongly it responds per class, not the
literal feature value.

Example
-------
::

    python src/kernel_psd.py \
        --windows_csv logs/train/runs/<ts>/windows_train.csv \
        --kernels_csv logs/train/runs/<ts>/top_classifier_kernels.csv \
        --kernels_csv logs/train/runs/<ts>/top_discriminative_kernels.csv \
        --out_dir logs/train/runs/<ts>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}


def _load_recording_psd(edf_path: str):
    """Load a recording's ``-psd.npz`` channel-averaged, or None if missing."""
    npz = Path(str(edf_path).replace(".edf", "-psd.npz"))
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=False)
    # Channel-average immediately: kernels mix random channel subsets, so the
    # relevant input spectrum is the mean over channels (shape, not per-sensor).
    return d["freqs"], d["psd"].mean(axis=0), int(d["n_times"])


def _class_relative_psd(windows_csv: str):
    """Channel-averaged, subject-aggregated, per-class relative PSD.

    Returns ``(freqs, {0: psd_rel, 1: psd_rel}, {0: n_subj, 1: n_subj})`` with each
    class PSD normalized to unit total power over the full frequency grid.
    """
    df = pd.read_csv(windows_csv)
    freqs = None
    per_class_subjects: dict = {0: [], 1: []}
    for subj, g in df.groupby("subject"):
        cls = int(bool(g["epilepsy"].iloc[0]))
        acc = None
        wsum = 0.0
        for edf in sorted(set(g["path"])):
            out = _load_recording_psd(edf)
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

    Evaluates over the full band (comb replicas included), so the power integral
    captures the kernel's true response. For 'diff' kernels the undilated
    first-difference high-pass ``2|sin(pi f / sfreq)|`` is applied (effective
    response on the raw signal x).
    """
    w = np.array([row[f"w{i}"] for i in range(9)], dtype=float)
    d = float(row["dilation"])
    j = np.arange(9)
    phase = (2.0 * np.pi * d / sfreq) * np.outer(freqs, j)  # (n_freqs, 9)
    mag = np.abs((w[None, :] * np.exp(-1j * phase)).sum(axis=1))
    if str(row.get("representation", "raw")) == "diff":
        mag = mag * (2.0 * np.abs(np.sin(np.pi * freqs / sfreq)))
    return mag


def _kernel_psd_figure(kernels_csv, freqs, rel, counts, sfreq, fmax, top_n, out_path):
    """Build the A + B figure for one kernel CSV. Returns a summary dict."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kdf = pd.read_csv(kernels_csv)
    is_disc = "favors" in kdf.columns
    # |H(f)|^2 per kernel on the full grid, and the per-class power integral
    # (df constant -> drops out of the ratio; this is a weighted mean of |H|^2).
    H2 = np.stack([_kernel_response(r, freqs, sfreq) ** 2 for _, r in kdf.iterrows()])
    p_epi = (H2 * rel[1][None, :]).sum(axis=1)
    p_no = (H2 * rel[0][None, :]).sum(axis=1)
    logratio = 10.0 * np.log10((p_epi + 1e-30) / (p_no + 1e-30))

    fmask = freqs <= fmax
    fx = freqs[fmask]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.5, 6.4))

    # --- Panel A: relative PSD (dB) + aggregate kernel sensitivity ---
    for cls in (0, 1):
        axA.plot(
            fx, 10.0 * np.log10(rel[cls][fmask] + 1e-30),
            color=CLASS_COLORS[cls], lw=1.4, label=f"{CLASS_NAMES[cls]} (n={counts[cls]})",
        )
    axA.set_xlabel("frequency (Hz)")
    axA.set_ylabel("relative PSD (dB)")
    axA.set_xlim(0, fmax)
    axA.legend(fontsize=8, loc="upper right")
    axS = axA.twinx()
    sens = H2.sum(axis=0)
    sens = sens / sens.max()
    axS.fill_between(fx, 0, sens[fmask], color="0.5", alpha=0.18)
    axS.plot(fx, sens[fmask], color="0.35", lw=1.2, label="kernel sensitivity")
    for h2 in H2[: min(top_n, len(H2))]:
        axS.plot(fx, (h2 / h2.max())[fmask], color="0.55", lw=0.4, alpha=0.35)
    axS.set_ylim(0, 1.05)
    axS.set_ylabel(r"kernel $\sum|H(f)|^2$ (norm.)")
    axS.legend(fontsize=8, loc="upper left")
    axA.set_title(f"A: class relative PSD vs kernel sensitivity ({Path(kernels_csv).stem})", fontsize=9)

    # --- Panel B: per-kernel class power log-ratio ---
    order = np.argsort(logratio)
    y = np.arange(len(order))
    if is_disc:
        fav = kdf["favors"].to_numpy()[order]
        colors = [CLASS_COLORS[int(f)] for f in fav]
    else:
        colors = [CLASS_COLORS[1] if logratio[i] >= 0 else CLASS_COLORS[0] for i in order]
    axB.barh(y, logratio[order], color=colors, height=0.8)
    axB.axvline(0.0, color="0.3", lw=0.7)
    axB.set_yticks([])
    axB.set_ylabel("kernels (sorted)")
    axB.set_xlabel(r"class power log-ratio $10\log_{10}(P_{\mathrm{epi}}/P_{\mathrm{no}})$  (+ = epilepsy)")
    axB.set_title("B: per-kernel relative-power preference by class", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    summary = {
        "csv": Path(kernels_csv).name,
        "n_kernels": len(kdf),
        "mean_logratio_dB": float(np.mean(logratio)),
        "frac_favor_epilepsy": float(np.mean(logratio >= 0)),
    }
    if is_disc:
        fav = kdf["favors"].to_numpy()
        agree = np.mean((logratio >= 0) == (fav == 1))
        summary["favors_vs_power_agreement"] = float(agree)
    else:
        rank = kdf["count"].to_numpy() if "count" in kdf.columns else np.arange(len(kdf))[::-1]
        if np.std(rank) > 0:
            summary["corr_rank_vs_abs_logratio"] = float(
                np.corrcoef(rank, np.abs(logratio))[0, 1]
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows_csv", required=True, help="windows_train.csv from a run.")
    parser.add_argument(
        "--kernels_csv", action="append", required=True,
        help="A top-kernel CSV (repeatable: pass once per ranking).",
    )
    parser.add_argument("--out_dir", default=None, help="Output dir (default: windows_csv dir).")
    parser.add_argument("--fmax", type=float, default=60.0, help="Max frequency to plot (Hz).")
    parser.add_argument("--top_n", type=int, default=10, help="Individual |H| traces drawn in panel A.")
    parser.add_argument(
        "--kernel_sfreq", type=float, default=250.0,
        help="Sample rate (Hz) the kernels were applied at (the resampled window rate).",
    )
    args = parser.parse_args()

    freqs, rel, counts = _class_relative_psd(args.windows_csv)
    if 0 not in rel or 1 not in rel:
        raise SystemExit(f"Need both classes in training PSDs; got counts {counts}.")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.windows_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for kcsv in args.kernels_csv:
        out_path = out_dir / f"kernel_psd_{Path(kcsv).stem}.png"
        summary = _kernel_psd_figure(
            kcsv, freqs, rel, counts, args.kernel_sfreq, args.fmax, args.top_n, out_path
        )
        print(f"wrote {out_path}")
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in summary.items()))


if __name__ == "__main__":
    main()
