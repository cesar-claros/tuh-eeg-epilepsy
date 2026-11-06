"""Pure logic of the exploratory map of K, R, D and L (``documentation/separable_band_rkd_map_protocol.md``).

The 36 cells and the convergence arm, the effects within each budget and their interactions with the budget (as weight
vectors over cells, so that one product gives both the point value and every bootstrap draw), the best healthy cell
per budget, the pre-stated `shifts` read, the rate and distortion front, and the seed reads of amendment A1 (section
14). It reuses ``src/band_sweep_core.py``, which stays unchanged. No corpus; numpy only.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from src.band_sweep_core import Setting, parse_setting
from src.reconstruction_baselines_core import Interval

RS = (4, 8, 16)
KS = (32, 64, 128)
DS = (4, 8)
LS = (128, 256)
P = 512
REFERENCE = {"K": 64, "D": 4, "L": 256}  # the levels that every within-R effect is measured against
CONTRASTS = {"K128": ("K", 128), "K32": ("K", 32), "D8": ("D", 8), "L128": ("L", 128)}
PAIRS = (("L", "D"), ("K", "L"), ("K", "D"))  # two-factor interactions within one R; K at 64 and 128 only
CONVERGENCE = Setting(K=64, L=128, D=8, P=P, R=16, epochs=60)
PAIR_LEVELS = {"K": (64, 128), "D": (4, 8), "L": (256, 128)}  # (reference, other) for the two-factor contrasts


def grid(seed: int = 1) -> list[Setting]:
    """The 36 cells at ``seed`` in a fixed order: R, then K, then D, then L."""
    return [Setting(K=k, L=length, D=d, P=P, R=r, seed=seed) for r, k, d, length in itertools.product(RS, KS, DS, LS)]


def map_settings(seed: int = 1) -> list[Setting]:
    """Every setting of the map stage at ``seed``: the 36 cells, and at seed 1 the convergence arm (amendment A1 does
    not repeat it). Existing cells are skipped by name."""
    return [*grid(seed), CONVERGENCE] if seed == 1 else grid(seed)


def _weights(
    cells: Sequence[Setting], select: Mapping[str, int], factor: str, level: int, reference: int
) -> np.ndarray:
    """Weights over ``cells`` of the mean at ``level`` minus the mean at ``reference``, within ``select``."""
    weights = np.zeros(len(cells))
    for value, sign in ((level, 1.0), (reference, -1.0)):
        members = [
            i
            for i, c in enumerate(cells)
            if getattr(c, factor) == value and all(getattr(c, f) == v for f, v in select.items())
        ]
        if not members:
            raise ValueError(f"no cell has {factor}={value} within {dict(select)}")
        weights[members] += sign / len(members)
    return weights


def effect_weights(cells: Sequence[Setting]) -> dict[str, np.ndarray]:
    """Weights of every within-R effect (``R4:L128``), of its change across R (``R16-R4:L128``, ``R8-R4:L128``) and of
    the two-factor interactions within each R (``R4:L*D``). A pair with K uses K 64 and 128 only, by its levels.
    """
    out: dict[str, np.ndarray] = {}
    for r in RS:
        for name, (factor, level) in CONTRASTS.items():
            out[f"R{r}:{name}"] = _weights(cells, {"R": r}, factor, level, REFERENCE[factor])
        for a, b in PAIRS:
            (a0, a1), (b0, b1) = PAIR_LEVELS[a], PAIR_LEVELS[b]
            keep = {"R": r}
            out[f"R{r}:{a}*{b}"] = _weights(cells, keep | {b: b1}, a, a1, a0) - _weights(
                cells, keep | {b: b0}, a, a1, a0
            )
    for later in (8, 16):
        for name in CONTRASTS:
            out[f"R{later}-R4:{name}"] = out[f"R{later}:{name}"] - out[f"R4:{name}"]
    return out


def intervals(
    values: np.ndarray, draws: np.ndarray, weights: Mapping[str, np.ndarray], confidence: float = 0.95
) -> dict[str, Interval]:
    """Point values and percentile intervals of linear contrasts of per-cell EVs.

    ``values`` is ``(n_cells,)`` and ``draws`` ``(n_boot, n_cells)``, both pooled EVs in cell order.
    """
    tail = (1.0 - confidence) / 2
    out = {}
    for name, w in weights.items():
        low, high = np.quantile(draws @ w, [tail, 1.0 - tail])
        out[name] = Interval(float(values @ w), float(low), float(high))
    return out


def best_healthy(ev: Mapping[str, float], health: Mapping[str, bool], cells: Sequence[Setting], r: int) -> str | None:
    """The healthy cell of layer ``r`` with the highest EV; ``None`` when no cell of the layer is healthy."""
    layer = [c.name for c in cells if c.R == r and health[c.name]]
    return max(layer, key=lambda name: ev[name]) if layer else None


def same_at(name: str, r: int) -> str:
    """The cell at budget ``r`` with the K, L, D and seed of cell ``name``."""
    return replace(parse_setting(name), R=r).name


def shift_read(best_low: str | None, best_high: str | None, versus: Interval | None, r_high: int) -> str:
    """``shifts`` when the best healthy cell at ``r_high`` differs from the ``r_high`` version of the best at R=4 and
    beats it (interval above 0); ``stable`` otherwise; ``undefined`` when a layer has no healthy cell."""
    if best_low is None or best_high is None:
        return "undefined"
    if best_high == same_at(best_low, r_high):
        return "stable"
    if versus is None:
        raise ValueError("a shift needs the interval of the high-budget best against its counterpart")
    return "shifts" if versus.low > 0 else "stable"


def front(points: Mapping[str, tuple[float, float]]) -> list[str]:
    """The cells on the rate and distortion front: no other cell has at least the EV at no more bits, with one strict.

    ``points`` maps a cell to ``(bits per second, EV)``. The front is returned in increasing bits.
    """
    names = list(points)
    keep = []
    for a in names:
        bits_a, ev_a = points[a]
        dominated = any(
            points[b][0] <= bits_a and points[b][1] >= ev_a and (points[b][0] < bits_a or points[b][1] > ev_a)
            for b in names
            if b != a
        )
        if not dominated:
            keep.append(a)
    return sorted(keep, key=lambda n: points[n][0])


def seed_noise(differences: Sequence[float]) -> dict[str, float]:
    """The median and the largest absolute EV difference between two seeds over cells (amendment A1, read 3)."""
    values = np.abs(np.asarray(differences, dtype=float))
    return {"median_abs": float(np.median(values)), "largest_abs": float(values.max())}


def replicates(first: float, second: float, both: Interval) -> bool:
    """An effect replicates when its point values at the two seeds share a sign and its two-seed interval excludes 0
    (amendment A1, read 4)."""
    same_sign = (first > 0 and second > 0) or (first < 0 and second < 0)
    return same_sign and (both.low > 0 or both.high < 0)
