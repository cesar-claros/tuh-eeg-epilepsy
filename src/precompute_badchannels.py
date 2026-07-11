"""Offline precomputation of bad channels and bad time segments per recording.

Scans each recording, flags corrupted channels (railing / flat, dead, and noisy) and
high-amplitude artifact segments, and writes a sibling ``<name>-bads.json`` next to the
``.edf``. The training flow does NOT trigger this; run it once, then enable the repair with
``data.interpolate_bad_channels=true`` and/or ``data.drop_bad_segments=true`` (the engine reads
the sidecar to interpolate bad channels via spherical splines before the bipolar montage, and to
drop windows overlapping bad segments).

Detection runs on the referential 10-20 channels (before bipolar) of a high-passed, notched copy:

- Bad channels: PyPREP ``NoisyChannels.find_all_bads`` (deviation + correlation + HF noise +
  RANSAC + dropout), unioned with channels that are flat/railed for most of the recording
  (``mne.preprocessing.annotate_amplitude`` with a ``flat`` threshold; railing pins the signal
  at a constant, i.e. flat). If more than ``--max_bad_frac`` of channels are bad, interpolation
  is unreliable, so the sidecar is flagged ``too_many_bad_channels`` (the engine can then drop
  the recording rather than interpolate it).
- Bad segments: ``annotate_amplitude`` peak/flat detection on the GOOD channels only (so a bad
  channel's railing does not flag segments that interpolation would otherwise fix).

Idempotent: a recording whose ``-bads.json`` exists is skipped unless ``--overwrite``. CPU-only.

Examples
--------
From the ``code/`` directory inside the container::

    python src/precompute_badchannels.py --n_jobs 8
    python src/precompute_badchannels.py --n_jobs 8 --no_ransac          # faster, less thorough
    python src/precompute_badchannels.py --n_jobs 8 --peak_uv 400 --flat_uv 1 --data_dir /scratch/.../data
    python src/precompute_badchannels.py --edf /path/a.edf /path/b.edf   # only these recordings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import rootutils
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402

BADS_SUFFIX = "-bads.json"


def _sidecar_path(edf_path: Path) -> Path:
    return edf_path.parent / edf_path.name.replace(".edf", BADS_SUFFIX)


def _autocorr_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation of 1-D ``x`` for lags 0..max_lag via FFT (O(n log n))."""
    n = x.size
    fx = np.fft.rfft(x - x.mean(), n=2 * n)
    return np.fft.irfft(fx * np.conj(fx))[: max_lag + 1]


def estimate_periodicity(data: np.ndarray, fs: float, fmin_hz: float = 0.5,
                         fmax_hz: float = 6.0) -> dict:
    """Detect a periodic (harmonic-comb) artifact by averaged autocorrelation.

    A periodic artifact shows up as a comb of harmonics in the PSD and a strong autocorrelation
    peak at its period. ``data`` is (C, T) in volts. Per-channel normalized autocorrelations are
    averaged (phase-invariant, so a comb with channel-varying phase still reinforces), and the
    dominant peak in the period band ``[1/fmax_hz, 1/fmin_hz]`` gives the fundamental. Returns
    fundamental_hz, period_s, comb_strength (peak height ~0-1; higher = more periodic),
    cardiac_like (fundamental in the 0.7-2 Hz heart-rate band), and lags_s / acf for plotting.
    """
    x = np.asarray(data, dtype=float)
    n = x.shape[1]
    max_lag = int(min(n - 1, fs / fmin_hz * 1.5))
    acc, used = np.zeros(max_lag + 1), 0
    for ch in x:
        ac = _autocorr_fft(ch, max_lag)
        if ac[0] > 0:
            acc += ac / ac[0]
            used += 1
    acf = acc / max(used, 1)
    lags_s = np.arange(max_lag + 1) / fs
    lo, hi = int(fs / fmax_hz), min(max_lag, int(fs / fmin_hz))
    out = {"fundamental_hz": float("nan"), "period_s": float("nan"), "comb_strength": 0.0,
           "cardiac_like": False, "band_edge": False, "lags_s": lags_s, "acf": acf}
    if hi > lo + 1:
        # comb_strength = strongest periodic peak in the band (stable flag). The dominant peak
        # (argmax) can lock onto a MULTIPLE of the true period (subharmonic/octave error: acf at 2T
        # or 3T slightly exceeds acf at T), so step down to the fundamental if a divisor period T/d
        # has a comparably strong autocorrelation peak. (A plain argmax already ignores the fine
        # high-frequency ripples, so we only probe divisors of the winning lag, not every short lag.)
        comb_strength = float(acf[lo:hi].max())
        k = lo + int(np.argmax(acf[lo:hi]))
        # Probe divisors largest-first (shortest candidate period first) so a genuine subharmonic
        # error steps all the way down to the true fundamental T, not an intermediate multiple.
        for div in (5, 4, 3, 2):
            kk = int(round(k / div))
            if kk < lo:
                continue
            near = acf[max(kk - 1, 0): kk + 2]
            if near.size and near.max() >= 0.6 * acf[k]:
                k = max(kk - 1, 0) + int(np.argmax(near))
                break
        f0 = fs / k
        # band_edge: the dominant period is at the shortest lag the band allows, i.e. the true
        # periodicity is >= fmax_hz -> a fast brain rhythm (alpha/beta), NOT a low-fundamental
        # comb artifact. Callers should not flag these.
        out.update(fundamental_hz=float(f0), period_s=float(k / fs), comb_strength=comb_strength,
                   cardiac_like=bool(0.7 <= f0 <= 2.0), band_edge=bool(f0 >= 0.92 * fmax_hz))
    return out


def period_average(x: np.ndarray, fs: float, period_s: float, max_periods: int = 400):
    """Average successive one-period segments of 1-D ``x`` -> ``(t, template)``.

    Reveals the artifact waveform: the phase-locked periodic component reinforces while the
    (non-locked) neural signal averages out. Returns ``(None, None)`` if too few periods.
    """
    period = int(round(period_s * fs))
    k = min(x.size // period, max_periods) if period >= 2 else 0
    if k < 2:
        return None, None
    return np.arange(period) / fs, x[: k * period].reshape(k, period).mean(0)


def _write_sidecar(path: Path, bad_channels, bad_segments, n_eeg, too_many, note, params,
                   periodic=None) -> None:
    payload = {
        "bad_channels": bad_channels,               # canonical 10-20 names, e.g. ["T3", "O2"]
        "bad_segments": bad_segments,               # [[start_s, end_s], ...] on the recording clock
        "n_eeg": n_eeg,
        "n_bad_channels": len(bad_channels),
        "too_many_bad_channels": bool(too_many),
        "periodic_artifact": periodic,              # {fundamental_hz, comb_strength, cardiac_like, flagged}
        "note": note,
        "params": {k: params[k] for k in
                   ("flat_uv", "peak_uv", "hp", "notch", "ransac", "max_bad_frac", "min_seg_s",
                    "periodic_strength_thr")},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _detect_one(edf_path: Path, params: dict) -> str:
    """Detect bad channels + segments for one recording and write its ``-bads.json``."""
    sidecar = _sidecar_path(edf_path)
    if sidecar.exists() and not params["overwrite"]:
        return "skip"
    try:
        from pyprep.find_noisy_channels import NoisyChannels

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
        TUHEEGEpilepsy._rename_channels(raw)
        eeg = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eeg"]
        if len(eeg) < 4:
            _write_sidecar(sidecar, [], [], len(eeg), True, "too few EEG channels", params)
            return "few_channels"
        raw.pick(eeg)
        TUHEEGEpilepsy._set_montage(raw)

        # Detection copy: high-pass (+ optional notch) to remove drift/line noise, which
        # stabilizes both the amplitude thresholds and PyPREP's RANSAC prediction.
        det = raw.copy().filter(l_freq=params["hp"], h_freq=None, verbose="error")
        if params["notch"]:
            det.notch_filter(params["notch"], verbose="error")

        flat_v = params["flat_uv"] * 1e-6
        peak_v = params["peak_uv"] * 1e-6

        # 1) Bad channels: PyPREP NoisyChannels only (deviation/correlation/HF/RANSAC PLUS its
        #    own robust flat/NaN/dropout checks, which catch railed/dead channels). We do NOT add
        #    a separate absolute-threshold flat check: on low-amplitude recordings a fixed uV
        #    threshold marks normal quiet EEG as "flat" and over-rejects (it was flagging every
        #    channel on a 6 uV-ptp recording).
        nc = NoisyChannels(det, random_state=params["seed"], do_detrend=False)
        nc.find_all_bads(ransac=params["ransac"])
        bad = set(nc.get_bads())
        bad_channels = sorted(bad)
        n_eeg = len(eeg)
        too_many = (len(bad_channels) / n_eeg) > params["max_bad_frac"]

        # 2) High-amplitude (peak) and railed (flat) artifact segments on the GOOD channels only.
        #    flat is a near-zero threshold so only true railing/clipping is caught, not quiet EEG.
        good = [c for c in det.ch_names if c not in bad]
        bad_segments = []
        if good:
            annots, _ = mne.preprocessing.annotate_amplitude(
                det.copy().pick(good), peak=peak_v, flat=flat_v,
                bad_percent=params["seg_bad_percent"], min_duration=params["min_seg_s"],
                picks="eeg", verbose="error")
            bad_segments = [[round(float(o), 4), round(float(o + d), 4)]
                            for o, d in zip(annots.onset, annots.duration)]

        # 3) Periodic (harmonic-comb) artifact: a whole-recording, all-channel periodic signal
        #    (e.g. cardiac or a device) that interpolation/segment-drop cannot fix. Measured on the
        #    same detection copy; recorded for triage (repair vs exclude) corpus-wide.
        per = estimate_periodicity(det.get_data(), float(det.info["sfreq"]),
                                   fmin_hz=params["periodic_fmin"], fmax_hz=params["periodic_fmax"])
        f0 = per["fundamental_hz"]
        periodic = {
            "fundamental_hz": round(f0, 4) if not np.isnan(f0) else None,
            "comb_strength": round(per["comb_strength"], 4),
            "cardiac_like": per["cardiac_like"],
            "band_edge": per["band_edge"],
            # A genuine comb artifact is a strong periodicity in the INTERIOR of the band; a
            # band-edge peak is a normal fast rhythm (alpha/beta), so it is not flagged.
            "flagged": bool(per["comb_strength"] >= params["periodic_strength_thr"] and not per["band_edge"]),
        }

        _write_sidecar(sidecar, bad_channels, bad_segments, n_eeg, too_many, "", params, periodic)
        return "ok"
    except Exception as e:  # noqa: BLE001
        logger.error(f"bad-channel detection failed for {edf_path.name}: {type(e).__name__}: {e}")
        return "error"


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute bad channels + bad segments for the TUH EEG corpus.")
    parser.add_argument("--data_dir", type=str, default=str(root / "data"),
                        help="Parent directory of the version folder (default: <PROJECT_ROOT>/data).")
    parser.add_argument("--version", type=str, default="v3.0.0", help="Corpus version subfolder (default: v3.0.0).")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel workers (joblib).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N recordings (debugging).")
    parser.add_argument("--edf", nargs="+", default=None,
                        help="Explicit EDF path(s) to process directly, bypassing the corpus scan (e.g. specific "
                        "recordings you flagged). Overrides --data_dir / --version / --limit.")
    parser.add_argument("--flat_uv", type=float, default=0.1,
                        help="Peak-to-peak (uV) below which a SEGMENT is flat/railed (default 0.1; near-zero, so "
                        "only true railing/clipping is caught, not low-amplitude EEG). Dead channels are handled "
                        "by PyPREP, not this threshold.")
    parser.add_argument("--peak_uv", type=float, default=500.0,
                        help="Peak-to-peak (uV) above which a segment is a high-amplitude artifact (default 500).")
    parser.add_argument("--hp", type=float, default=1.0, help="High-pass (Hz) applied before detection (default 1).")
    parser.add_argument("--notch", type=float, default=60.0,
                        help="Notch (Hz) applied before detection; 0 to disable (default 60).")
    parser.add_argument("--no_ransac", action="store_true", help="Disable PyPREP RANSAC (faster, less thorough).")
    parser.add_argument("--max_bad_frac", type=float, default=0.3,
                        help="Flag the recording too_many_bad_channels if > this fraction of channels is bad "
                        "(interpolation then unreliable; default 0.3).")
    parser.add_argument("--min_seg_s", type=float, default=0.2, help="Minimum bad-segment duration in seconds.")
    parser.add_argument("--seg_bad_percent", type=float, default=5.0,
                        help="annotate_amplitude bad_percent for segment detection (default 5).")
    parser.add_argument("--seed", type=int, default=42, help="RANSAC random state (reproducibility).")
    parser.add_argument("--periodic_strength_thr", type=float, default=0.5,
                        help="comb_strength (normalized autocorrelation at the fundamental) >= this, AND not a "
                        "band-edge (fast-rhythm) peak, flags a periodic (harmonic-comb) artifact (default 0.5).")
    parser.add_argument("--periodic_fmin", type=float, default=0.5,
                        help="Min fundamental (Hz) searched for the periodic artifact (default 0.5).")
    parser.add_argument("--periodic_fmax", type=float, default=6.0,
                        help="Max fundamental (Hz) searched for the periodic artifact (default 6).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if the -bads.json exists.")
    args = parser.parse_args()

    if args.edf:
        # Direct paths: skip the (slow) corpus scan entirely and detect on exactly these files.
        paths = [Path(p) for p in args.edf]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"--edf path(s) not found: {missing}")
    else:
        recording_ids = list(range(args.limit)) if args.limit is not None else None
        tuh = TUHEEGEpilepsy(data_dir=args.data_dir, version=args.version, recording_ids=recording_ids)
        paths = [Path(p) for p in tuh.descriptions["path"].tolist()]
    params = {
        "flat_uv": args.flat_uv, "peak_uv": args.peak_uv, "hp": args.hp,
        "notch": (args.notch if args.notch and args.notch > 0 else None),
        "ransac": not args.no_ransac, "max_bad_frac": args.max_bad_frac,
        "min_seg_s": args.min_seg_s, "seg_bad_percent": args.seg_bad_percent, "seed": args.seed,
        "periodic_strength_thr": args.periodic_strength_thr, "periodic_fmin": args.periodic_fmin,
        "periodic_fmax": args.periodic_fmax, "overwrite": args.overwrite,
    }
    logger.info(f"Detecting bad channels/segments for {len(paths)} recordings "
                f"(ransac={params['ransac']}, flat<{args.flat_uv}uV, peak>{args.peak_uv}uV)...")
    if args.n_jobs == 1:
        results = [_detect_one(p, params) for p in tqdm(paths, desc="bad-channel detection")]
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(_detect_one)(p, params) for p in tqdm(paths, desc="bad-channel detection"))
    counts = {k: results.count(k) for k in ("ok", "skip", "few_channels", "error")}
    logger.info(f"Done: {counts}. Sidecars written next to each .edf as *{BADS_SUFFIX}.")


if __name__ == "__main__":
    main()
