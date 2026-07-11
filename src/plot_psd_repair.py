"""Plot one recording's per-channel PSD before vs after bad-channel/segment repair.

Overlays the per-channel Welch PSD of the raw signal against the repaired signal (bad channels
from the ``-bads.json`` sidecar spherical-spline interpolated, bad segments excluded via
annotations), so you can see whether the repair removes the anomaly the detector flagged. Shown
on the referential channels, where interpolation happens (before any bipolar montage). Run after
``precompute_badchannels.py`` has written the sidecar for this recording.

    python src/plot_psd_repair.py --edf /path/to/rec.edf
    python src/plot_psd_repair.py --edf /path/to/rec.edf --fmax 80 --notch 60 --bandpass 0.1 75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import rootutils  # noqa: E402
from scipy.signal import welch  # noqa: E402

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.aas import apply_aas  # noqa: E402
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402


def _welch_db(data: np.ndarray, fs: float, win_sec: float, fmax: float):
    """Per-channel Welch PSD in dB (10 log10) up to fmax. data: (C, T) volts."""
    nperseg = min(int(win_sec * fs), data.shape[1])
    freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    keep = freqs <= fmax
    return freqs[keep], 10.0 * np.log10(psd[:, keep] + 1e-20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edf", required=True, help="EDF path (its -bads.json sidecar must exist).")
    p.add_argument("--fmax", type=float, default=80.0, help="Max frequency to plot (Hz).")
    p.add_argument("--win_sec", type=float, default=4.0, help="Welch segment length (s).")
    p.add_argument("--bandpass", type=float, nargs=2, default=[0.1, 75.0],
                   help="Band-pass (Hz) applied to both before/after (default 0.1 75).")
    p.add_argument("--notch", type=float, default=60.0, help="Notch (Hz) applied to both; 0 to disable.")
    p.add_argument("--bipolar", action="store_true",
                   help="Form the TCP bipolar montage AFTER repair and plot the bipolar-channel PSDs (what the "
                   "model sees). Pairs involving an interpolated electrode are highlighted.")
    p.add_argument("--aas", action="store_true",
                   help="Also apply Average Artifact Subtraction of the periodic artifact (using the sidecar's "
                   "periodic_artifact.fundamental_hz) in the 'after' signal, to check the comb is removed.")
    p.add_argument("--aas_fmax", type=float, default=2.5, help="Only AAS a fundamental <= this (Hz).")
    p.add_argument("--ncols", type=int, default=5, help="Subplot columns.")
    p.add_argument("--out", default=None, help="Output PNG (default: diagnostics/psd/<stem>_psd_repair.png).")
    args = p.parse_args()

    edf = Path(args.edf)
    sidecar = edf.parent / edf.name.replace(".edf", "-bads.json")
    if not sidecar.exists():
        raise SystemExit(f"No sidecar {sidecar.name}; run precompute_badchannels.py --edf {edf} first.")
    meta = json.loads(sidecar.read_text())
    bad_channels = list(meta.get("bad_channels", []))
    bad_segments = [(float(s), float(e)) for s, e in meta.get("bad_segments", [])]

    raw = mne.io.read_raw_edf(edf, preload=True, verbose="error")
    TUHEEGEpilepsy._rename_channels(raw)
    raw.pick([c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"])
    TUHEEGEpilepsy._set_montage(raw)
    raw.filter(l_freq=args.bandpass[0], h_freq=args.bandpass[1], verbose="error")
    if args.notch and args.notch > 0:
        raw.notch_filter(args.notch, verbose="error")
    fs = float(raw.info["sfreq"])
    present_bad = [c for c in bad_channels if c in raw.ch_names]

    def _annot(rr):
        if bad_segments:
            rr.set_annotations(mne.Annotations(
                onset=[s for s, _ in bad_segments], duration=[e - s for s, e in bad_segments],
                description=["BAD_artifact"] * len(bad_segments)))
        return rr

    # Repair on the referential channels: interpolation must precede any bipolar re-reference.
    after_ref = raw.copy()
    if present_bad:
        after_ref.info["bads"] = present_bad
        after_ref.interpolate_bads(reset_bads=True, verbose="error")
    if args.aas:
        f0 = (meta.get("periodic_artifact") or {}).get("fundamental_hz")
        if f0 and 0 < f0 <= args.aas_fmax:
            after_ref = mne.io.RawArray(apply_aas(after_ref.get_data(), fs, 1.0 / f0),
                                        after_ref.info, verbose="error")
            print(f"applied AAS at fundamental {f0:.3f} Hz")
        else:
            print(f"--aas: no cardiac-band fundamental in sidecar (f0={f0}); skipped")

    if args.bipolar:
        before_r = TUHEEGEpilepsy._apply_bipolar(raw.copy())
        after_r = TUHEEGEpilepsy._apply_bipolar(after_ref)
        if before_r is None or after_r is None:
            raise SystemExit("Could not form the TCP bipolar montage (too few channels).")
        ch_names = list(after_r.ch_names)
        before = before_r.get_data()
        after = _annot(after_r).get_data(reject_by_annotation="omit" if bad_segments else None)
        # A bipolar pair is affected if either of its electrodes was interpolated.
        affected = {name for name in ch_names if any(b in name.split("-") for b in present_bad)}
        montage_label, mark = "bipolar (TCP)", "affected"
    else:
        ch_names = list(raw.ch_names)
        before = raw.get_data()
        after = _annot(after_ref).get_data(reject_by_annotation="omit" if bad_segments else None)
        affected = set(present_bad)
        montage_label, mark = "referential", "interp"

    freqs_b, psd_b = _welch_db(before, fs, args.win_sec, args.fmax)
    freqs_a, psd_a = _welch_db(after, fs, args.win_sec, args.fmax)

    order = sorted(range(len(ch_names)), key=lambda i: ch_names[i])
    n = len(ch_names)
    ncols = args.ncols
    nrows = int(np.ceil((n + 1) / ncols))  # +1 for the mean panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax_i, ch_i in enumerate(order):
        ax = axes[ax_i]
        name = ch_names[ch_i]
        is_bad = name in affected
        ax.plot(freqs_b, psd_b[ch_i], color="tab:red" if is_bad else "0.6", lw=0.9,
                label="raw", zorder=2 if is_bad else 1)
        ax.plot(freqs_a, psd_a[ch_i], color="tab:blue", lw=0.9, label="repaired", zorder=3)
        ax.set_title(f"{name}{f'  ({mark})' if is_bad else ''}", color="tab:red" if is_bad else "black", fontsize=8)
        ax.tick_params(labelsize=6)
    ax = axes[n]  # mean-across-channels panel
    ax.plot(freqs_b, psd_b.mean(0), color="0.4", lw=1.3, label="raw mean")
    ax.plot(freqs_a, psd_a.mean(0), color="tab:blue", lw=1.3, label="repaired mean")
    ax.set_title("MEAN (all channels)", fontsize=8)
    ax.legend(fontsize=6)
    for ax in axes[n + 1:]:
        ax.axis("off")

    fig.suptitle(f"{edf.stem}  |  {montage_label} PSD before (raw) vs after repair\n"
                 f"interpolated: {present_bad or 'none'}   bad segments: {len(bad_segments)}", fontsize=10)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("PSD (dB)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = "_psd_repair_bipolar" if args.bipolar else "_psd_repair"
    out = Path(args.out) if args.out else root / "diagnostics" / "psd" / f"{edf.stem}{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  (interpolated {len(present_bad)} channels, omitted {len(bad_segments)} segments)")


if __name__ == "__main__":
    main()
