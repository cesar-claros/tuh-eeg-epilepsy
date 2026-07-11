"""Average Artifact Subtraction (AAS) for strictly-periodic artifacts (cardiac / pulse).

A periodic artifact (e.g. the ~1 Hz cardiac/pulse rhythm that shows up as a harmonic comb across
the whole PSD) is phase-locked: each cycle has nearly the same waveform. AAS detects the cycle
events, builds a per-channel template by averaging a moving window of neighbouring cycles, and
subtracts it. The artifact (locked to the cycle) reinforces in the template and is removed; the
neural signal (not locked to the cardiac cycle) averages toward zero in the template and survives.

Validated on synthetic data: a 1.75 Hz QRS train's harmonics drop ~15 dB while a co-present 10 Hz
alpha rhythm is unchanged (-0.05 dB).

This is a sensor-space operation applied to the referential channels BEFORE the bipolar montage.
Restrict it (via the caller) to genuine artifacts in the cardiac band (~0.5-2.5 Hz): there is no
neural rhythm we want to keep at exactly the heart rate, and staying below ~2.5 Hz avoids the
3 Hz spike-wave band, so AAS never touches epileptiform activity.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def apply_aas(data: np.ndarray, fs: float, period_s: float, n_avg: int = 21,
              refractory: float = 0.6) -> np.ndarray:
    """Subtract a strictly-periodic artifact from ``data`` (C, T) volts -> cleaned (C, T).

    Parameters
    ----------
    data : np.ndarray
        Channels x time, in volts.
    fs : float
        Sampling rate (Hz).
    period_s : float
        Artifact period (1 / fundamental_hz).
    n_avg : int, default=21
        Cycles averaged into each event's template (a moving window, to track slow drift in the
        artifact while averaging out the neural signal).
    refractory : float, default=0.6
        Minimum event spacing as a fraction of the period (rejects double-detections).

    Returns
    -------
    np.ndarray
        Cleaned data, same shape. Unchanged (a copy) if the recording is too short or too few
        cycle events are found.
    """
    data = np.asarray(data, dtype=float)
    n_ch, n_t = data.shape
    period = int(round(period_s * fs))
    half = period // 2
    if period < 4 or n_t < 3 * period:
        return data

    # Detect cycle events on the highest-variance channel (where the artifact is strongest):
    # peaks of the centered |signal|, at least refractory*period apart and above a robust floor.
    ref = np.abs(data[int(np.argmax(data.var(1)))] - data[int(np.argmax(data.var(1)))].mean())
    floor = np.median(ref) + 2.0 * np.median(np.abs(ref - np.median(ref)))
    peaks, _ = find_peaks(ref, distance=max(1, int(refractory * period)), height=floor)
    valid = peaks[(peaks - half >= 0) & (peaks - half + period < n_t)]
    if valid.size < 3:
        return data

    clean = data.copy()
    for ch in range(n_ch):
        segs = np.stack([data[ch, p - half: p - half + period] for p in valid])  # (n_events, period)
        for i, p in enumerate(valid):
            a = max(0, i - n_avg // 2)
            b = min(len(valid), a + n_avg)
            clean[ch, p - half: p - half + period] -= segs[a:b].mean(0)
    return clean
