"""Per-channel spectrogram grid for one recording (diagnostic).

Companion to ``psd_segments.py``. Instead of one channel-averaged spectrogram plus time
segments, this shows the FULL-recording spectrogram of EVERY channel in a grid, so you
can see which channel(s) carry an anomalous band and when it appears. Reprocesses the EDF
the same way the PSD sidecars were made (rename -> pick EEG -> resample-or-native ->
notch -> bipolar), so pass the SAME ``--bipolar`` / ``--notch_freqs`` / ``--native`` flags
the rankings were built with. It computes the spectrograms straight from the signal, so
no precomputed PSD sidecar is needed. Run in the container (needs the corpus / EDF).

    python src/spectrogram_channels.py --edf /path/<rec>.edf --bipolar --native --notch_freqs 60 120
    python src/spectrogram_channels.py --edf /path/<rec>.edf --bipolar --native --notch_freqs 60 120 --fmax 80 --band 28 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_EPS = 1e-30


def _default_psd_dir() -> Path:
    """Default output location: <project root>/diagnostics/psd (created if missing)."""
    import rootutils

    out = rootutils.setup_root(__file__, pythonpath=True) / "diagnostics" / "psd"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _process_raw(edf, target_sfreq, notch_freqs, bipolar):
    """Reprocess the EDF to match the PSD; returns (data (n_ch, T), ch_names, fs)."""
    import mne
    import rootutils

    rootutils.setup_root(__file__, pythonpath=True)
    from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: PLC0415

    raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    TUHEEGEpilepsy._rename_channels(raw)
    eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
    raw.pick(eeg)
    if target_sfreq is not None and not np.isclose(raw.info["sfreq"], target_sfreq):
        raw.resample(target_sfreq, verbose="error")
    fs = float(raw.info["sfreq"])
    if notch_freqs:
        valid = [f for f in notch_freqs if f < fs / 2.0]
        if valid:
            raw.notch_filter(valid, verbose="ERROR")
    if bipolar:
        raw = TUHEEGEpilepsy._apply_bipolar(raw)
        if raw is None:
            raise SystemExit("Too few bipolar pairs for this recording.")
    return raw.get_data(), list(raw.ch_names), fs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", required=True, help="Path to the recording's .edf.")
    parser.add_argument("--bipolar", action="store_true", help="Bipolar montage (match the rankings).")
    parser.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Notch (match the rankings).")
    parser.add_argument("--native", action="store_true", help="Native rate, no resample (match the rankings).")
    parser.add_argument("--target_sfreq", type=float, default=256.0,
                        help="Resample rate (Hz) when NOT --native (default 256, the precompute default).")
    parser.add_argument("--fmax", type=float, default=None, help="Max frequency to show (Hz); default Nyquist.")
    parser.add_argument("--win_sec", type=float, default=2.0,
                        help="Spectrogram window length (s); sets the frequency resolution (default 2 -> 0.5 Hz).")
    parser.add_argument("--ncols", type=int, default=5, help="Number of grid columns (default 5).")
    parser.add_argument("--band", type=float, nargs=2, default=None, metavar=("F_LO", "F_HI"),
                        help="Draw dashed lines at this band on every channel (e.g. the anomalous band).")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap (default magma).")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: <root>/diagnostics/psd/<rec>_spectrograms.png).")
    args = parser.parse_args()

    target_sfreq = None if args.native else args.target_sfreq
    data, ch_names, fs = _process_raw(args.edf, target_sfreq, args.notch_freqs, args.bipolar)

    from scipy.signal import spectrogram

    nper = max(64, int(round(args.win_sec * fs)))
    f_s, t_s, Sxx = spectrogram(data, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    Sdb = 10.0 * np.log10(Sxx + _EPS)  # (n_ch, n_freq, n_time)
    fmax = args.fmax if args.fmax is not None else fs / 2.0
    fmask = f_s <= fmax
    # Shared, robust color scale so channels are comparable and a loud channel stands out.
    vmin, vmax = np.percentile(Sdb[:, fmask, :], [5, 99])

    n_ch = data.shape[0]
    ncols = max(1, args.ncols)
    nrows = int(np.ceil(n_ch / ncols))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.2 * nrows),
                             squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    im = None
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n_ch:
            ax.axis("off")
            continue
        im = ax.pcolormesh(t_s, f_s[fmask], Sdb[k][fmask], shading="auto",
                           cmap=args.cmap, vmin=vmin, vmax=vmax)
        ax.set_title(ch_names[k], fontsize=7)
        ax.set_ylim(0, fmax)
        if args.band:
            ax.axhline(args.band[0], color="cyan", lw=0.5, ls="--")
            ax.axhline(args.band[1], color="cyan", lw=0.5, ls="--")
        ax.tick_params(labelsize=5)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, label="power (dB)")
    band_note = f", band {args.band[0]:.0f}-{args.band[1]:.0f} Hz" if args.band else ""
    fig.suptitle(
        f"{Path(args.edf).stem}  -  per-channel spectrogram "
        f"({'native ' if args.native else ''}{fs:.0f} Hz, {n_ch} channels{band_note})",
        fontsize=11,
    )
    if hasattr(fig, "supxlabel"):  # matplotlib >= 3.4
        fig.supxlabel("time (s)", fontsize=9)
        fig.supylabel("frequency (Hz)", fontsize=9)

    out = Path(args.out) if args.out else _default_psd_dir() / f"{Path(args.edf).stem}_spectrograms.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  ({n_ch} channels, 0-{fmax:.0f} Hz, {args.win_sec:g}s window)")


if __name__ == "__main__":
    main()
