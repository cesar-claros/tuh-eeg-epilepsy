"""Numerical FIR-context checks; run with pytest in the HPC container."""

from __future__ import annotations

from math import ceil

import mne
import numpy as np
import pytest
import rootutils
from numpy.typing import NDArray

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components.temporal_filtering import filter_context_seconds  # noqa: E402


def _filter_signal(data: NDArray[np.float64], sfreq: float) -> NDArray[np.float64]:
    """Apply the baseline's native-rate high-pass and notch defaults."""
    filtered = mne.filter.filter_data(data, sfreq, l_freq=0.1, h_freq=None, verbose="ERROR")
    return mne.filter.notch_filter(filtered, sfreq, freqs=[60.0], verbose="ERROR")


def test_no_filter_needs_no_context() -> None:
    """The minimum margin only applies when a temporal filter is active."""
    assert filter_context_seconds(256.0) == 0.0
    assert filter_context_seconds(256.0, [None, None], []) == 0.0


def test_sequential_filter_radii_add() -> None:
    """A high-pass followed by a notch depends on both filters' supports."""
    highpass = filter_context_seconds(256.0, [0.1, None], minimum_seconds=0.0)
    notch = filter_context_seconds(256.0, notch_freqs=[60.0], minimum_seconds=0.0)
    combined = filter_context_seconds(256.0, [0.1, None], [60.0], minimum_seconds=0.0)

    assert combined == pytest.approx(highpass + notch)
    assert 16.0 < highpass < 17.0
    assert 19.0 < combined < 21.0


def test_short_filters_preserve_the_existing_minimum_margin() -> None:
    """Existing broad-band and notch-only configurations retain at least 4 s."""
    assert filter_context_seconds(256.0, [1.0, 45.0]) == 4.0
    assert filter_context_seconds(256.0, notch_freqs=[60.0]) == 4.0
    assert filter_context_seconds(256.0, [1.0, 45.0], minimum_seconds=7.0) == 7.0


@pytest.mark.parametrize("minimum_seconds", [-1.0, float("nan"), float("inf")])
def test_invalid_minimum_margin_is_rejected(minimum_seconds: float) -> None:
    """Invalid context cannot produce an out-of-bounds or nonfinite crop."""
    with pytest.raises(ValueError, match="minimum_seconds"):
        filter_context_seconds(256.0, [0.1, None], minimum_seconds=minimum_seconds)


@pytest.mark.parametrize("sfreq", [128.0, 256.0])
@pytest.mark.parametrize("center_seconds", [2.0, 60.0])
def test_contextualized_crop_matches_filtering_the_continuous_signal(sfreq: float, center_seconds: float) -> None:
    """Actual FIR filtering agrees on a retained interior, including short windows."""
    rng = np.random.default_rng(42)
    times = np.arange(round(180.0 * sfreq), dtype=np.float64) / sfreq
    data = rng.normal(size=(2, len(times))) + 2.0 * np.sin(2.0 * np.pi * 0.07 * times)
    start = round(60.0 * sfreq)
    stop = start + round(center_seconds * sfreq)
    margin = ceil(filter_context_seconds(sfreq, [0.1, None], [60.0]) * sfreq)

    whole = _filter_signal(data, sfreq)
    contextualized = _filter_signal(data[:, start - margin : stop + margin], sfreq)
    retained = contextualized[:, margin : margin + stop - start]

    np.testing.assert_allclose(retained, whole[:, start:stop], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("frequency", "expected_gain"),
    [(0.01, (0.0, 0.1)), (1.0, (0.98, 1.02)), (60.0, (0.0, 0.05)), (80.0, (0.98, 1.02))],
)
def test_baseline_removes_drift_and_line_noise_without_a_broad_lowpass(
    frequency: float, expected_gain: tuple[float, float]
) -> None:
    """The new baseline suppresses drift/60 Hz while retaining a high-frequency tone."""
    sfreq = 256.0
    times = np.arange(round(180.0 * sfreq), dtype=np.float64) / sfreq
    signal = np.sin(2.0 * np.pi * frequency * times)[None, :]
    margin = round(30.0 * sfreq)

    filtered = _filter_signal(signal, sfreq)
    gain = np.linalg.norm(filtered[:, margin:-margin]) / np.linalg.norm(signal[:, margin:-margin])

    assert expected_gain[0] <= gain <= expected_gain[1]


@pytest.mark.parametrize(
    ("sfreq", "cutoffs", "notches"),
    [
        (0.0, [0.1, None], None),
        (256.0, [0.1], None),
        (256.0, [-0.1, None], None),
        (256.0, [float("nan"), None], None),
        (256.0, [45.0, 1.0], None),
        (100.0, [0.1, 60.0], None),
        (120.0, [0.1, None], [60.0]),
        (256.0, None, [float("nan")]),
        (256.0, None, [-60.0]),
    ],
)
def test_invalid_filter_requests_fail_explicitly(
    sfreq: float, cutoffs: list[float | None] | None, notches: list[float] | None
) -> None:
    """Invalid native-rate designs are never silently converted into no filtering."""
    with pytest.raises(ValueError):
        filter_context_seconds(sfreq, cutoffs, notches)
