"""Explain a recording's PSD features with the time segments that produce them (4x4).

Spectrogram-guided diagnostic for one recording (e.g. a row from
``rank_psd_anomaly.py``). It reprocesses the EDF the same way the PSD was made
(rename -> pick EEG -> resample-or-native -> notch -> bipolar), targets a suspicious
frequency band (auto-detected as the biggest deviation from a smooth 1/f baseline,
or ``--band F_LO F_HI``), computes a spectrogram, and shows the time windows where
that band's power is highest, so you can see what is generating the feature.

Layout (4x4):
  row 0: PSD (band shaded) | spectrogram (band + picked segments) |
         band-power over time | worst-channel PSD
  rows 1-3: 6 fixed-length segments x [bipolar stack | worst channel]; the last is a
         low-band-power baseline for contrast.

Pass the SAME sidecar flags used at precompute (``--bipolar`` / ``--notch_freqs`` /
``--native``). Needs the corpus / EDF (run in the container).

    python src/psd_segments.py --edf /path/to/<rec>.edf --bipolar --notch_freqs 60 120 --native
    python src/psd_segments.py --edf /path/to/<rec>.edf --bipolar --notch_freqs 60 120 --band 80 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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


def _auto_band(freqs, cm, smooth_hz=5.0, bw=4.0, fmin=1.0):
    """Target band = window around the biggest positive deviation from a smooth 1/f fit."""
    df = float(freqs[1] - freqs[0])
    lp = 10.0 * np.log10(cm + _EPS)
    w = max(5, int(round(smooth_hz / df)))
    w += 1 - (w % 2)
    smooth = np.convolve(lp, np.ones(w) / w, mode="same")
    resid = (lp - smooth) * (freqs >= fmin)  # ignore < fmin Hz (drift) and notch dips are negative
    fpk = float(freqs[int(np.argmax(resid))])
    return max(0.0, fpk - bw / 2.0), fpk + bw / 2.0


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


def _stack(ax, seg, ch_names, fs, t0, worst=None):
    """Plot a bipolar/montage stack of `seg` (n_ch, n) offset vertically."""
    n_ch, n = seg.shape
    t = t0 + np.arange(n) / fs
    spacing = 6.0 * (np.median(np.std(seg, axis=1)) + _EPS)
    for k in range(n_ch):
        c = "tab:red" if k == worst else "0.25"
        lw = 0.8 if k == worst else 0.4
        ax.plot(t, seg[k] + (n_ch - 1 - k) * spacing, color=c, lw=lw)
    ax.set_yticks([(n_ch - 1 - k) * spacing for k in range(n_ch)])
    ax.set_yticklabels(ch_names, fontsize=4)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", required=True, help="Path to the recording's .edf.")
    parser.add_argument("--bipolar", action="store_true", help="Bipolar montage (match the sidecar).")
    parser.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Notch (match the sidecar).")
    parser.add_argument("--native", action="store_true", help="Native rate (match the sidecar).")
    parser.add_argument("--band", type=float, nargs=2, default=None, metavar=("F_LO", "F_HI"),
                        help="Target band (Hz); default auto-detect the biggest 1/f deviation.")
    parser.add_argument("--bandwidth", type=float, default=4.0, help="Auto-band width (Hz) around the peak.")
    parser.add_argument("--seg_sec", type=float, default=6.0, help="Segment length (s).")
    parser.add_argument("--out", default=None, help="Output PNG (default: <edf dir>/<rec>_segments.png).")
    args = parser.parse_args()

    suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)
    npz = Path(str(args.edf).replace(".edf", suffix))
    if not npz.exists():
        raise SystemExit(f"PSD sidecar {npz.name} not found; run precompute_psd.py with the matching flags.")
    d = np.load(npz, allow_pickle=False)
    freqs, psd, sfreq_psd = d["freqs"], d["psd"], float(d["sfreq"])
    cm = psd.mean(axis=0)
    band = tuple(args.band) if args.band else _auto_band(freqs, cm, bw=args.bandwidth)

    data, ch_names, fs = _process_raw(
        args.edf, None if args.native else sfreq_psd, args.notch_freqs, args.bipolar
    )

    from scipy.signal import spectrogram

    nper = max(64, int(round(fs)))  # ~1 s spectrogram window
    f_s, t_s, Sxx = spectrogram(data, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    Sm = Sxx.mean(axis=0)  # (n_freq, n_time), channel-averaged power
    bmask = (f_s >= band[0]) & (f_s <= band[1])
    bp = Sm[bmask].mean(axis=0)  # band power over time (selection signal)

    # worst channel: most band power in the cached per-channel PSD
    pmask = (freqs >= band[0]) & (freqs <= band[1])
    worst = int(np.argmax(psd[:, pmask].sum(axis=1)))

    # segments: tile into seg_sec windows, rank by mean band power; top-5 + baseline
    seg_n = int(round(args.seg_sec * fs))
    n_win = data.shape[1] // seg_n
    if n_win < 2:
        raise SystemExit(f"Recording too short for {args.seg_sec}s segments.")
    starts = np.arange(n_win) * seg_n
    win_bp = np.array([bp[(t_s >= s / fs) & (t_s < (s + seg_n) / fs)].mean() for s in starts])
    order = np.argsort(win_bp)[::-1]
    chosen = list(order[:5]) + [order[-1]]  # 5 hottest + 1 baseline
    labels = ["hot"] * 5 + ["baseline"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(16, 11))
    fig.suptitle(
        f"{Path(args.edf).stem}  -  target band {band[0]:.1f}-{band[1]:.1f} Hz, "
        f"worst channel {ch_names[worst]} ({'native ' if args.native else ''}{fs:.0f} Hz)",
        fontsize=11,
    )

    # row 0: context
    axes[0, 0].semilogy(freqs, cm + _EPS, color="k", lw=0.8)
    axes[0, 0].axvspan(band[0], band[1], color="tab:orange", alpha=0.25)
    axes[0, 0].set_title("PSD (channel-avg)", fontsize=8)
    axes[0, 0].set_xlabel("Hz", fontsize=7)
    axes[0, 1].pcolormesh(t_s, f_s, 10 * np.log10(Sm + _EPS), shading="auto", cmap="magma")
    axes[0, 1].axhspan(band[0], band[1], color="tab:cyan", alpha=0.2)
    for s in chosen[:-1]:
        axes[0, 1].axvspan(starts[s] / fs, (starts[s] + seg_n) / fs, color="w", alpha=0.18)
    axes[0, 1].set_title("spectrogram (band + hot segments)", fontsize=8)
    axes[0, 1].set_ylabel("Hz", fontsize=7)
    axes[0, 2].plot(t_s, bp, color="tab:orange", lw=0.7)
    for s, lab in zip(chosen, labels):
        axes[0, 2].axvspan(starts[s] / fs, (starts[s] + seg_n) / fs,
                           color="0.3" if lab == "baseline" else "tab:orange", alpha=0.2)
    axes[0, 2].set_title(f"band power over time ({band[0]:.0f}-{band[1]:.0f} Hz)", fontsize=8)
    axes[0, 3].semilogy(freqs, psd[worst] + _EPS, color="tab:red", lw=0.8)
    axes[0, 3].axvspan(band[0], band[1], color="tab:orange", alpha=0.25)
    axes[0, 3].set_title(f"worst-channel PSD: {ch_names[worst]}", fontsize=8)
    for ax in axes[0]:
        ax.tick_params(labelsize=6)

    # rows 1-3: 6 segments x [stack | worst channel]
    cells = [(r, c) for r in (1, 2, 3) for c in range(4)]
    for i, (s, lab) in enumerate(zip(chosen, labels)):
        a, b = cells[2 * i], cells[2 * i + 1]
        sl = slice(int(starts[s]), int(starts[s]) + seg_n)
        t0 = starts[s] / fs
        tag = "baseline" if lab == "baseline" else f"hot #{i + 1}"
        _stack(axes[a], data[:, sl], ch_names, fs, t0, worst=worst)
        axes[a].set_title(f"{tag} {t0:.0f}-{t0 + args.seg_sec:.0f}s  (stack)", fontsize=7)
        axes[b].plot(t0 + np.arange(seg_n) / fs, data[worst, sl], color="tab:red", lw=0.6)
        axes[b].set_title(f"{tag}  {ch_names[worst]}", fontsize=7)
        axes[b].set_xlim(t0, t0 + args.seg_sec)
        axes[b].tick_params(labelsize=5)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out) if args.out else Path(args.edf).with_name(Path(args.edf).stem + "_segments.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}  (band {band[0]:.1f}-{band[1]:.1f} Hz, worst channel {ch_names[worst]})")


if __name__ == "__main__":
    main()
