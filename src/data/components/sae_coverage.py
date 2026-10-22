"""Plan disjoint scored centers with real context for local SAE training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CoverageChunk:
    """Half-open sample intervals within one parent signal row.

    Attributes:
        input_start: First loaded sample, including left context.
        input_stop: Exclusive stop of the loaded samples, including right context.
        center_start: First sample scored by the reconstruction loss.
        center_stop: Exclusive stop of the scored samples.
    """

    input_start: int
    input_stop: int
    center_start: int
    center_stop: int

    @property
    def center_length(self) -> int:
        """Return the number of scored samples."""
        return self.center_stop - self.center_start

    @property
    def input_length(self) -> int:
        """Return the number of loaded samples, including context."""
        return self.input_stop - self.input_start


def plan_coverage(n_times: int, center_length: int, context: int) -> list[CoverageChunk]:
    """Cover all samples with sufficient real context exactly once for scoring.

    The eligible interval is ``[context, n_times - context)``. Its disjoint
    centers are at most ``center_length`` samples long, including a final short
    center when necessary. Each input interval adds ``context`` real samples
    on both sides. No padding, pooling-grid alignment, filtering, or shuffling
    is performed. Callers can shuffle the returned chunks without changing
    their scored coverage and should record excluded boundary samples.

    Args:
        n_times: Positive length of the parent signal row.
        center_length: Positive maximum number of scored samples per chunk.
        context: Nonnegative number of real context samples required per side.

    Returns:
        Chunks in temporal order. Returns an empty list if
        ``n_times <= 2 * context``, because no sample has the required context.

    Raises:
        TypeError: If any argument is not an integer, or is a boolean.
        ValueError: If either length is nonpositive or context is negative.
    """
    for name, value in (
        ("n_times", n_times),
        ("center_length", center_length),
        ("context", context),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if n_times <= 0:
        raise ValueError("n_times must be positive")
    if center_length <= 0:
        raise ValueError("center_length must be positive")
    if context < 0:
        raise ValueError("context must be nonnegative")

    eligible_stop = n_times - context
    chunks = []
    for start in range(context, eligible_stop, center_length):
        stop = min(start + center_length, eligible_stop)
        chunks.append(
            CoverageChunk(
                input_start=start - context,
                input_stop=stop + context,
                center_start=start,
                center_stop=stop,
            )
        )
    return chunks
