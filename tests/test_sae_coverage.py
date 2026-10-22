"""Metadata coverage invariants; run with pytest in the HPC container."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components.sae_coverage import CoverageChunk, plan_coverage  # noqa: E402


@pytest.mark.parametrize(
    ("n_times", "center_length", "context"),
    [(101, 13, 7), (1000, 127, 31), (27, 4, 5), (29, 5, 3), (19, 100, 2), (1, 1, 0)],
)
def test_eligible_samples_are_scored_exactly_once(n_times: int, center_length: int, context: int) -> None:
    """Every eligible sample is included, regardless of divisibility or phase."""
    chunks = plan_coverage(n_times, center_length, context)

    scored = [t for chunk in chunks for t in range(chunk.center_start, chunk.center_stop)]

    assert scored == list(range(context, n_times - context))
    assert all(0 < chunk.center_length <= center_length for chunk in chunks)
    assert all(previous.center_stop == following.center_start for previous, following in zip(chunks, chunks[1:]))


def test_inputs_supply_real_context_without_padding() -> None:
    """Input intervals stay inside the row, including the first and last chunks."""
    n_times, context = 53, 6

    chunks = plan_coverage(n_times, center_length=11, context=context)

    assert chunks[0].input_start == 0
    assert chunks[-1].input_stop == n_times
    for chunk in chunks:
        assert 0 <= chunk.input_start < chunk.input_stop <= n_times
        assert chunk.center_start - chunk.input_start == context
        assert chunk.input_stop - chunk.center_stop == context
        assert chunk.input_length == chunk.center_length + 2 * context


def test_short_tail_is_scored_instead_of_dropped_or_duplicated() -> None:
    """A one-sample tail keeps full context without repeating earlier centers."""
    chunks = plan_coverage(n_times=24, center_length=5, context=4)

    assert chunks == [
        CoverageChunk(0, 13, 4, 9),
        CoverageChunk(5, 18, 9, 14),
        CoverageChunk(10, 23, 14, 19),
        CoverageChunk(15, 24, 19, 20),
    ]
    assert chunks[-1].center_length == 1
    assert chunks[-1].input_length == 9


def test_zero_context_covers_the_whole_row() -> None:
    """No boundary samples are excluded when no context is required."""
    chunks = plan_coverage(n_times=7, center_length=3, context=0)

    assert chunks == [
        CoverageChunk(0, 3, 0, 3),
        CoverageChunk(3, 6, 3, 6),
        CoverageChunk(6, 7, 6, 7),
    ]


@pytest.mark.parametrize(("n_times", "context"), [(8, 4), (7, 4), (1, 10)])
def test_no_eligible_center_returns_empty_plan(n_times: int, context: int) -> None:
    """Rows without enough real context have no scored samples."""
    assert plan_coverage(n_times, center_length=3, context=context) == []


def test_a_single_eligible_sample_is_retained() -> None:
    """The smallest nonempty eligible interval produces one valid chunk."""
    assert plan_coverage(9, center_length=100, context=4) == [CoverageChunk(0, 9, 4, 5)]


def test_exact_multiple_has_no_empty_tail() -> None:
    """An exact division of the eligible interval creates only full centers."""
    chunks = plan_coverage(22, center_length=5, context=6)

    assert len(chunks) == 2
    assert all(chunk.center_length == 5 for chunk in chunks)


@pytest.mark.parametrize(
    ("n_times", "center_length", "context", "invalid_name"),
    [
        (0, 3, 0, "n_times"),
        (-1, 3, 0, "n_times"),
        (10, 0, 1, "center_length"),
        (10, -3, 1, "center_length"),
        (10, 3, -1, "context"),
    ],
)
def test_invalid_lengths_are_rejected(n_times: int, center_length: int, context: int, invalid_name: str) -> None:
    """Invalid geometry fails explicitly before any chunks are returned."""
    with pytest.raises(ValueError, match=invalid_name):
        plan_coverage(n_times, center_length, context)


@pytest.mark.parametrize("argument", ["n_times", "center_length", "context"])
@pytest.mark.parametrize("value", [1.5, "3", True, None])
def test_noninteger_arguments_are_rejected(argument: str, value: object) -> None:
    """Sample indices and lengths cannot be fractional, boolean, or missing."""
    arguments = {"n_times": 20, "center_length": 5, "context": 2}
    arguments[argument] = value  # type: ignore[assignment]

    with pytest.raises(TypeError, match=argument):
        plan_coverage(**arguments)


def test_chunks_are_immutable() -> None:
    """Shuffling metadata cannot accidentally mutate the interval definitions."""
    chunk = plan_coverage(17, center_length=5, context=3)[0]

    with pytest.raises(FrozenInstanceError):
        setattr(chunk, "center_start", 0)


def test_reordering_chunks_preserves_coverage() -> None:
    """Sampling order is independent of the deterministic coverage plan."""
    chunks = plan_coverage(31, center_length=7, context=4)

    reordered = list(reversed(chunks))
    scored = sorted(t for chunk in reordered for t in range(chunk.center_start, chunk.center_stop))

    assert scored == list(range(4, 27))
    assert plan_coverage(31, center_length=7, context=4) == chunks
