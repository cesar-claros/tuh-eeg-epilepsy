"""Polyphase resampling of the context-padded crop in WindowDataset._process; synthetic 250 Hz input, no EDF.

MNE's default FFT resampling shifted 250 Hz segments by about 2 ms (half an input sample) and depended on the
segment length (2026-10-26). These tests check consistency with the whole recording and the sample times.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components.window_dataset import WindowDataset  # noqa: E402

NATIVE, TARGET, DURATION, WINDOW = 250.0, 256.0, 300.0, 120.0
CHANNELS = ["a", "b"]


def _signal() -> np.ndarray:
    """Two channels of nonperiodic content: rhythms, a slow drift, 60 Hz line noise and white noise."""
    t = np.arange(int(DURATION * NATIVE)) / NATIVE
    rng = np.random.default_rng(0)
    base = (
        20e-6 * np.sin(2 * np.pi * 9.7 * t)
        + 15e-6 * np.sin(2 * np.pi * 2.3 * t + 0.4)
        + 30e-6 * np.sin(2 * np.pi * 0.05 * t)
    )
    line = 5e-6 * np.sin(2 * np.pi * 60.0 * t)
    noise = 5e-6 * rng.normal(size=(2, t.size))
    return np.stack([base + line, 0.5 * base - line]) + noise


def _raw(data: np.ndarray) -> mne.io.RawArray:
    return mne.io.RawArray(data.copy(), mne.create_info(CHANNELS, NATIVE, ch_types="eeg"), verbose="ERROR")


def _dataset(start: float) -> WindowDataset:
    plan = pd.DataFrame(
        {
            "subject": ["s"],
            "epilepsy": [1],
            "start": [start],
            "end": [start + WINDOW],
            "path": ["synthetic.edf"],
            "description_row": [{"path": "synthetic.edf", "montage": "test"}],
        }
    )
    return WindowDataset(
        plan,
        "raw",
        CHANNELS,
        TARGET,
        int(WINDOW * TARGET),
        {"test": {"channels": CHANNELS}},
        [0.1, None],
        False,
        False,
        None,
        (),
        0.0,
        False,
        ["epilepsy"],
        notch_freqs=[60.0],
    )


def _whole_recording(data: np.ndarray) -> np.ndarray:
    """Filter the complete recording, then resample it: the reference the windowed loader should match."""
    raw = _raw(data)
    raw.filter(l_freq=0.1, h_freq=None, verbose="ERROR")
    raw.notch_filter([60.0], verbose="ERROR")
    return mne.filter.resample(raw.get_data(), up=TARGET, down=NATIVE, axis=-1, method="polyphase", verbose="ERROR")


def _window(data: np.ndarray, start: float) -> np.ndarray:
    """``_process`` output for the window at ``start``, reading ``data`` in place of an EDF."""
    dataset = _dataset(start)

    def read(*_: Any, **__: Any) -> mne.io.RawArray:
        return _raw(data)

    with patch("src.data.components.window_dataset.TUHEEGEpilepsy._read_raw_edf", side_effect=read):
        window = dataset._process(dataset.plan.iloc[0])
    assert window is not None and window.shape == (2, int(WINDOW * TARGET))
    return window


@pytest.mark.parametrize("start", [40.0, 160.0])  # 160 s: the last window, with a right margin cut at the end
def test_window_matches_the_whole_recording_filtered_and_resampled(start: float) -> None:
    data = _signal()
    window = _window(data, start)
    first = int(start * TARGET)
    reference = _whole_recording(data)[:, first : first + window.shape[1]]
    # Filtering with the full context is exact and polyphase resampling is local, so only the
    # float32 output rounding remains (about 3e-8 of the peak in a local MNE 1.11 replica of this check).
    assert np.abs(window - reference).max() < 1e-6 * np.abs(reference).max()


def test_resampling_keeps_the_sample_times() -> None:
    """Tones sampled at 250 Hz come out as the same tones sampled at 256 Hz (FFT resampling: 12 percent off)."""
    t_in = np.arange(int(DURATION * NATIVE)) / NATIVE
    t_out = np.arange(int(DURATION * TARGET)) / TARGET

    def tones(t: np.ndarray) -> np.ndarray:
        return 20e-6 * np.sin(2 * np.pi * 9.7 * t) + 15e-6 * np.sin(2 * np.pi * 30.0 * t + 0.4)

    window = _window(np.stack([tones(t_in), -tones(t_in)]), 40.0)
    truth = tones(t_out)[int(40.0 * TARGET) : int(40.0 * TARGET) + window.shape[1]]
    assert np.abs(window[0] - truth).max() < 5e-3 * np.abs(truth).max()
    assert np.abs(window[1] + truth).max() < 5e-3 * np.abs(truth).max()
