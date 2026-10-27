"""Statistics and decision rule of the recording-start transient measurement (src/start_transient.py); CPU."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.shapeconv_sae import ROBUST_CLIP  # noqa: E402
from src.start_transient import (  # noqa: E402
    EXCESS_TOLERANCE,
    analyze,
    bootstrap,
    clip_tolerance,
    group_difference,
    per_second_stats,
    plot,
    skip_length,
    start_excess,
    transient_end,
)


def test_per_second_stats_use_usable_channels_only() -> None:
    signal = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # unusable: must not count
        ]
    )
    rows = np.zeros_like(signal)
    rows[0, 1] = ROBUST_CLIP
    rows[2] = ROBUST_CLIP
    stats = per_second_stats(signal, rows, np.array([True, True, False]), sfreq=4)
    assert stats is not None and stats.shape == (3, 2)
    np.testing.assert_allclose(stats[0], [1.0, math.sqrt(2.0)])  # rms over channels and samples
    np.testing.assert_allclose(stats[1], [3.0, 0.0])  # mean over channels of summed |diff|
    np.testing.assert_allclose(stats[2], [1 / 8, 0.0])  # clipped share of usable channel-samples
    assert per_second_stats(signal, rows, np.zeros(3, dtype=bool), sfreq=4) is None


def test_start_excess_removes_a_uniform_level_difference() -> None:
    second = np.exp(np.random.default_rng(0).normal(size=(5, 120)))
    first = 2.0 * second
    np.testing.assert_allclose(start_excess(first, second), 0.0, atol=1e-12)
    first[:, :10] *= 3.0
    excess = start_excess(first, second)
    np.testing.assert_allclose(excess[:, :10], math.log(3.0))
    np.testing.assert_allclose(excess[:, 10:], 0.0, atol=1e-12)


def test_transient_end_and_skip_rounding() -> None:
    curve = np.zeros(120)
    assert transient_end(curve, EXCESS_TOLERANCE) == 0
    curve[:23] = 0.5
    assert transient_end(curve, EXCESS_TOLERANCE) == 23
    curve[5] = -0.5  # a drop counts for the two-sided statistics only
    assert transient_end(curve, EXCESS_TOLERANCE) == 23
    low = np.zeros(120)
    low[:12] = -0.5
    assert transient_end(low, EXCESS_TOLERANCE) == 12 and transient_end(low, 1e-4, two_sided=False) == 0
    curve[89] = 1.0
    assert transient_end(curve, EXCESS_TOLERANCE) == 90  # unresolved before the offset seconds
    curve[100] = 1.0
    assert transient_end(curve, EXCESS_TOLERANCE) == 90  # seconds from 90 on define the offset, not the end
    assert skip_length({"rms": 0, "line_length": 0, "clip_frac": 0}, 19.8) == 20
    assert skip_length({"rms": 23, "line_length": 5, "clip_frac": 0}, 19.8) == 30
    assert skip_length({"rms": 40, "line_length": 5, "clip_frac": 0}, 19.8) == 40
    assert clip_tolerance(np.zeros((3, 120))) == pytest.approx(1e-4)
    assert clip_tolerance(np.full((3, 120), 0.002)) == pytest.approx(1e-3)


def test_bootstrap_resamples_patients() -> None:
    values = np.repeat(np.arange(6, dtype=float)[:, None], 4, axis=1)
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    estimate, low, high = bootstrap(values, clusters, np.mean, 200, np.random.default_rng(1))
    np.testing.assert_allclose(estimate, 2.5)
    assert (low <= estimate).all() and (estimate <= high).all() and (low >= 0.5).all() and (high <= 4.5).all()


def test_group_difference_flags_only_clear_effects() -> None:
    rng = np.random.default_rng(2)
    clusters = np.array([f"p{i // 2}" for i in range(40)])
    in_a = np.arange(40) < 20
    clear = np.where(in_a, 0.3, 0.0) + rng.normal(scale=0.01, size=40)
    assert group_difference(clear, in_a, ~in_a, clusters, 300, rng)["flag"]
    none = rng.normal(scale=0.01, size=40)
    assert not group_difference(none, in_a, ~in_a, clusters, 300, rng)["flag"]
    small = np.where(in_a, 0.02, 0.0)  # significant but below the 5 percent effect size
    assert not group_difference(small, in_a, ~in_a, clusters, 300, rng)["flag"]
    assert group_difference(clear, in_a, np.zeros(40, dtype=bool), clusters, 10, rng)["difference"] is None


def _synthetic(amplitude_class_1: float) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """40 recordings of 20 patients: a decaying start excess 1 + a exp(-t/8) on rms and line length."""
    t = np.arange(120)
    labels = np.array([(r // 2) % 2 for r in range(40)])  # one label per patient, both labels at each rate
    second = np.ones((40, 3, 120))
    second[:, 2] = 0.0
    first = second.copy()
    for r in range(40):
        amplitude = amplitude_class_1 if labels[r] == 1 else 2.0
        first[r, :2] = 1.0 + amplitude * np.exp(-t / 8.0)
    meta = pl.DataFrame(
        {
            "subject": [f"p{r // 2}" for r in range(40)],
            "epilepsy": labels,
            "sfreq": [250.0 if r < 20 else 256.0 for r in range(40)],
        }
    )
    return first, second, meta


def test_analyze_finds_the_transient_end_and_proposes_a_skip(tmp_path: Path) -> None:
    first, second, meta = _synthetic(amplitude_class_1=2.0)
    table, recordings, decision = analyze(first, second, meta, filter_edge_s=19.8, n_boot=50, seed=0)
    # 1 + 2 exp(-t/8) <= 1.1 from t = 8 ln 20 = 23.97, so seconds 0..23 exceed the tolerance.
    assert decision["transient_end_s"] == {"rms": 24, "line_length": 24, "clip_frac": 0}
    assert decision["proposed_skip_s"] == 30 and not decision["unresolved"]
    assert decision["confound_flags"] == []
    assert set(table["group"].unique()) == {"all", "class 0", "class 1", "rate 250", "rate 256"}
    assert recordings.height == 40 and "rms_burden_first_30s" in recordings.columns
    plot(table, decision, tmp_path / "start_transient.png")
    assert (tmp_path / "start_transient.png").stat().st_size > 0


def test_analyze_flags_a_class_dependent_transient() -> None:
    first, second, meta = _synthetic(amplitude_class_1=6.0)
    _, _, decision = analyze(first, second, meta, filter_edge_s=19.8, n_boot=50, seed=0)
    comparison = decision["comparisons_first_30s"]["rms"]["class_1_minus_class_0"]
    assert comparison["difference"] > 0 and comparison["flag"]
    assert "rms: class_1_minus_class_0" in decision["confound_flags"]
    assert decision["transient_end_s"]["rms"] > 24  # the larger class-1 excess decays later
