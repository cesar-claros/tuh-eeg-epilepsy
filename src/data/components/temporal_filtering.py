"""Real-data context for the sensor-space temporal FIR filters."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from math import isfinite

import mne


def filter_context_seconds(
    sfreq: float,
    filter_freq: Sequence[float | None] | None = None,
    notch_freqs: Sequence[float] | None = None,
    minimum_seconds: float = 4.0,
) -> float:
    """Return real context needed on each side of a filtered window, in seconds.

    Args:
        sfreq: Native sampling frequency in Hz, before resampling.
        filter_freq: ``[low, high]`` pass-band edges; either can be ``None``.
        notch_freqs: Requested notch centers in Hz, or no notch when empty.
        minimum_seconds: Minimum context when any filter is active. The default
            preserves the existing four-second loader margin for shorter filters.

    Returns:
        The larger of the minimum and the sum of the filters' half-supports.
        Returns zero when neither filter is active. Supports are derived from
        the same zero-phase Hamming FIR defaults used by ``Raw.filter`` and
        ``Raw.notch_filter``; designs are cached by rate and frequencies.

    Raises:
        ValueError: Arguments are invalid or MNE cannot design the requested
            filter at the native sampling rate, including a notch near Nyquist.

    Sequential filter radii add. With sufficient real samples this removes crop
    edge effects from the retained window. At recording boundaries, the caller
    can only provide available samples and MNE's reflection padding still applies.
    This context does not describe subsequent resampling or ICA edge behavior.
    """
    sfreq = float(sfreq)
    minimum_seconds = float(minimum_seconds)
    if not isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive.")
    if not isfinite(minimum_seconds) or minimum_seconds < 0:
        raise ValueError("minimum_seconds must be finite and nonnegative.")

    cutoffs = None
    if filter_freq is not None and len(filter_freq):
        if len(filter_freq) != 2:
            raise ValueError("filter_freq must contain exactly [low, high].")
        low, high = (None if value is None else float(value) for value in filter_freq)
        if any(value is not None and (not isfinite(value) or value <= 0) for value in (low, high)):
            raise ValueError("Filter cutoffs must be finite and positive, or None.")
        if low is not None and high is not None and low >= high:
            raise ValueError("The lower filter cutoff must be below the upper cutoff.")
        if low is not None or high is not None:
            cutoffs = (low, high)
    notches = tuple(float(value) for value in notch_freqs) if notch_freqs is not None else ()
    if any(not isfinite(value) or value <= 0 for value in notches):
        raise ValueError("Notch frequencies must be finite and positive.")
    if cutoffs is None and not notches:
        return 0.0

    radius_samples = _filter_radius_samples(sfreq, cutoffs, notches)
    return max(minimum_seconds, radius_samples / sfreq)


@lru_cache(maxsize=128)
def _filter_radius_samples(
    sfreq: float,
    cutoffs: tuple[float | None, float | None] | None,
    notches: tuple[float, ...],
) -> int:
    """Design each default FIR once and add its zero-phase sample radius."""
    radius = 0
    if cutoffs is not None:
        taps = mne.filter.create_filter(None, sfreq, l_freq=cutoffs[0], h_freq=cutoffs[1], verbose="ERROR")
        radius += (len(taps) - 1) // 2
    if notches:
        # Public notch_filter defaults: width=f/200, transition bandwidth=1 Hz.
        # Its band-stop call uses outer edges and a 0.5 Hz transition per side.
        half_transition = 0.5
        lower_edges = [freq - freq / 400.0 - half_transition for freq in notches]
        upper_edges = [freq + freq / 400.0 + half_transition for freq in notches]
        taps = mne.filter.create_filter(
            None,
            sfreq,
            l_freq=upper_edges,
            h_freq=lower_edges,
            l_trans_bandwidth=half_transition,
            h_trans_bandwidth=half_transition,
            verbose="ERROR",
        )
        radius += (len(taps) - 1) // 2
    return radius
