"""Characterize a recording's periodic (harmonic-comb) artifact.

For one recording, shows the three views that identify the artifact and pick a remover: the mean
PSD (the comb, with the detected fundamental's harmonics marked), the averaged autocorrelation
(the periodicity, with the period marked), and the period-averaged time-domain TEMPLATE (the
artifact waveform on its strongest channel). The template shape is the tell: a QRS-like spike at
~1 Hz says cardiac (use ECG/template removal), a square/pulse says a device (use Zapline / comb
removal). Uses the same detector as precompute_badchannels.py.

    python src/plot_periodic_artifact.py --edf /path/to/rec.edf
    python src/plot_periodic_artifact.py --edf /path/to/rec.edf --fmax 80 --notch 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import rootutils  # noqa: E402
from scipy.signal import welch  # noqa: E402

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.precompute_badchannels import estimate_periodicity, period_average  # noqa: E402


def _acf_at(ch: np.ndarray, k: int) -> float:
    """Normalized autocorrelation of ``ch`` at a single lag ``k``."""
    ch = ch - ch.mean()
    a0 = float(np.dot(ch, ch))
    return float(np.dot(ch[:-k], ch[k:]) / a0) if (a0 > 0 and 0 < k < ch.size) else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edf", required=True, help="EDF path.")
    p.add_argument("--fmax", type=float, default=80.0, help="Max frequency for the PSD panel (Hz).")
    p.add_argument("--hp", type=float, default=1.0, help="High-pass (Hz) applied before analysis.")
    p.add_argument("--notch", type=float, default=0.0, help="Notch (Hz); 0 = off (show the raw comb).")
    p.add_argument("--fmin_hz", type=float, default=0.5, help="Min fundamental searched (Hz).")
    p.add_argument("--fmax_hz", type=float, default=6.0, help="Max fundamental searched (Hz).")
    p.add_argument("--fundamental_hz", type=float, default=None,
                   help="Override the auto-detected fundamental (Hz) for the template/markers, e.g. if you read a "
                   "different rate off the autocorrelation panel.")
    p.add_argument("--out", default=None, help="Output PNG (default: diagnostics/psd/<stem>_periodic.png).")
    args = p.parse_args()

    edf = Path(args.edf)
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    TUHEEGEpilepsy._rename_channels(raw)
    raw.pick([c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"])
    raw.filter(l_freq=args.hp, h_freq=None, verbose="error")
    if args.notch and args.notch > 0:
        raw.notch_filter(args.notch, verbose="error")
    fs = float(raw.info["sfreq"])
    data = raw.get_data()  # (C, T) volts

    per = estimate_periodicity(data, fs, fmin_hz=args.fmin_hz, fmax_hz=args.fmax_hz)
    if args.fundamental_hz:
        per["fundamental_hz"], per["period_s"] = args.fundamental_hz, 1.0 / args.fundamental_hz
    f0, period_s, strength = per["fundamental_hz"], per["period_s"], per["comb_strength"]

    # PSD (mean across channels), dB.
    nperseg = min(int(4.0 * fs), data.shape[1])
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    keep = freqs <= args.fmax
    psd_db = 10.0 * np.log10(psd[:, keep].mean(0) + 1e-20)
    freqs = freqs[keep]

    # Strongest-periodicity channel -> its period-averaged template.
    best_t, best_tmpl, best_name = None, None, "?"
    if not np.isnan(f0):
        k = int(round(period_s * fs))
        scores = [_acf_at(ch, k) for ch in data]
        ch_i = int(np.argmax(scores))
        best_name = raw.ch_names[ch_i]
        best_t, best_tmpl = period_average(data[ch_i], fs, period_s)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # (1) PSD with harmonic markers.
    axes[0].plot(freqs, psd_db, color="tab:blue", lw=0.8)
    if not np.isnan(f0):
        for h in range(1, int(args.fmax / f0) + 1):
            axes[0].axvline(h * f0, color="tab:red", ls=":", lw=0.6, alpha=0.6)
    axes[0].set(title="mean PSD (comb + harmonics)", xlabel="Frequency (Hz)", ylabel="PSD (dB)")
    # (2) averaged autocorrelation with the period marked.
    lags, acf = per["lags_s"], per["acf"]
    lim = lags <= (5 * period_s if not np.isnan(period_s) else lags[-1])
    axes[1].plot(lags[lim], acf[lim], color="tab:blue", lw=0.9)
    if not np.isnan(period_s):
        for m in range(1, 6):
            axes[1].axvline(m * period_s, color="tab:red", ls=":", lw=0.7, alpha=0.7)
    axes[1].axhline(0, color="0.7", lw=0.5)
    axes[1].set(title="averaged autocorrelation", xlabel="lag (s)", ylabel="normalized ACF")
    # (3) period-averaged waveform (the artifact template).
    if best_tmpl is not None:
        axes[2].plot(best_t * 1e3, best_tmpl * 1e6, color="tab:red", lw=1.2)
        axes[2].set(title=f"period-averaged waveform ({best_name})", xlabel="time (ms)", ylabel="uV")
    else:
        axes[2].text(0.5, 0.5, "no stable period", ha="center", va="center")
        axes[2].set_axis_off()

    f0s = f"{f0:.3f} Hz" if not np.isnan(f0) else "n/a"
    fig.suptitle(f"{edf.stem}  |  periodic artifact: fundamental {f0s}  "
                 f"(period {period_s * 1e3:.0f} ms)  strength {strength:.2f}  "
                 f"cardiac_like={per['cardiac_like']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out) if args.out else root / "diagnostics" / "psd" / f"{edf.stem}_periodic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  (fundamental {f0s}, strength {strength:.3f}, cardiac_like {per['cardiac_like']})")


if __name__ == "__main__":
    main()
