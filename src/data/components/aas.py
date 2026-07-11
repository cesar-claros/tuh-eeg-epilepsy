"""Average Artifact Subtraction (AAS) for strictly-periodic artifacts (cardiac / pulse).

A periodic artifact (e.g. the ~1 Hz cardiac/pulse rhythm that shows up as a harmonic comb across
the whole PSD) is phase-locked to its own cycle: each period has nearly the same waveform. AAS
tiles the recording into fixed period-length segments, builds a per-channel template by averaging
a moving window of neighbouring segments, and subtracts it. The artifact (locked to the period)
reinforces in the template and is removed; the neural signal (not locked to the cardiac cycle)
averages toward zero in the template and survives.

The alignment is FIXED-PERIOD (reshape the signal into contiguous period-length blocks), not
peak-triggered. Peak detection is unreliable for the smooth, low-amplitude pulse artifacts seen in
this corpus (no sharp QRS to lock onto), so a fixed period grid from the sidecar fundamental is
used instead. Validated on synthetic data: a sharp 1.75 Hz QRS train drops ~17 dB at the
fundamental and a smooth 1.75 Hz pulse drops ~31 dB, while a co-present 10 Hz alpha rhythm is
unchanged (-0.3 dB) in both cases.

This is a sensor-space operation applied to the referential channels BEFORE the bipolar montage.
Restrict it (via the caller) to genuine artifacts in the cardiac band (~0.5-2.5 Hz): there is no
neural rhythm we want to keep at exactly the heart rate, and staying below ~2.5 Hz avoids the
3 Hz spike-wave band, so AAS never touches epileptiform activity.
"""

from __future__ import annotations

import numpy as np


def apply_aas(data: np.ndarray, fs: float, period_s: float, n_avg: int = 21) -> np.ndarray:
    """Subtract a strictly-periodic artifact from ``data`` (C, T) volts -> cleaned (C, T).

    The signal is tiled into contiguous ``period``-sample segments; each segment has a
    moving-average template (over ``n_avg`` neighbouring segments) subtracted from it. No cycle
    detection is used: the period comes directly from ``period_s`` (1 / fundamental_hz), which is
    robust for the smooth pulse artifacts that defeat peak triggering.

    Parameters
    ----------
    data : np.ndarray
        Channels x time, in volts.
    fs : float
        Sampling rate (Hz).
    period_s : float
        Artifact period (1 / fundamental_hz).
    n_avg : int, default=21
        Segments averaged into each segment's template (a moving window, to track slow drift in the
        artifact while averaging out the neural signal).

    Returns
    -------
    np.ndarray
        Cleaned data, same shape. Unchanged (a copy) if the recording is too short (fewer than
        three whole periods).
    """
    data = np.asarray(data, dtype=float)
    n_ch, n_t = data.shape
    period = int(round(period_s * fs))
    if period < 4 or n_t < 3 * period:
        return data
    n_seg = n_t // period
    if n_seg < 3:
        return data

    clean = data.copy()
    for ch in range(n_ch):
        segs = data[ch, : n_seg * period].reshape(n_seg, period)  # (n_seg, period), contiguous tiles
        for i in range(n_seg):
            a = max(0, i - n_avg // 2)
            b = min(n_seg, a + n_avg)
            clean[ch, i * period: (i + 1) * period] -= segs[a:b].mean(0)
    return clean
