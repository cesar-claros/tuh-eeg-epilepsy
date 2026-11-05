"""Checks of the pure logic of the exploratory map, ``src/band_map_core.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the 36 cells and the 10 that the sweep screen already holds, effect weights that recover known effects and
their changes across R, the intervals of contrasts, the best healthy cell, the `shifts` read and the front.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402

import src.band_map_core as core  # noqa: E402
import src.band_sweep_core as sweep  # noqa: E402
from src.reconstruction_baselines_core import Interval  # noqa: E402


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def test_grid_and_reuse() -> None:
    cells = core.grid()
    assert len(cells) == 36 and len({c.name for c in cells}) == 36 and {c.P for c in cells} == {512}
    assert {c.amplitudes_per_second for c in cells} == {2.0, 4.0, 8.0}
    assert len(core.map_settings()) == 37 and core.map_settings()[-1] == core.CONVERGENCE
    screen = {sweep.BASE, *sweep.screen_settings()}
    assert len(screen & set(cells)) == 10, "the screen must already hold 10 cells of the grid"


def _synthetic_ev(cells: list[sweep.Setting]) -> np.ndarray:
    """EV with known effects: L=128 +0.01 (+0.02 more at R=16), D=8 +0.003, K=128 -0.004 at R=4 and +0.006 at R=16,
    K=32 -0.002, an L*D interaction of +0.004 at R=8 only, and R itself."""
    ev = []
    for c in cells:
        value = 0.05 + 0.03 * np.log2(c.R / 4)
        value += (0.01 + (0.02 if c.R == 16 else 0.0)) * (c.L == 128) + 0.003 * (c.D == 8)
        value += {4: -0.004, 8: 0.0, 16: 0.006}[c.R] * (c.K == 128) - 0.002 * (c.K == 32)
        value += 0.004 * (c.R == 8) * (c.L == 128) * (c.D == 8)
        ev.append(value)
    return np.array(ev)


def test_effect_weights_recover_known_effects() -> None:
    cells = core.grid()
    ev = _synthetic_ev(cells)
    weights = core.effect_weights(cells)
    assert all(abs(w.sum()) < 1e-12 for w in weights.values()), "a contrast must not depend on the mean level"
    effect = {name: float(ev @ w) for name, w in weights.items()}
    assert abs(effect["R4:L128"] - 0.01) < 1e-12 and abs(effect["R16:L128"] - 0.03) < 1e-12
    assert abs(effect["R16-R4:L128"] - 0.02) < 1e-12 and abs(effect["R8-R4:L128"] - 0.002) < 1e-12
    assert abs(effect["R4:D8"] - 0.003) < 1e-12 and abs(effect["R16-R4:D8"]) < 1e-12
    assert abs(effect["R4:K128"] + 0.004) < 1e-12 and abs(effect["R16-R4:K128"] - 0.010) < 1e-12
    assert abs(effect["R4:K32"] + 0.002) < 1e-12
    assert abs(effect["R8:L*D"] - 0.004) < 1e-12 and abs(effect["R4:L*D"]) < 1e-12 and abs(effect["R16:K*D"]) < 1e-12


def test_intervals_of_contrasts() -> None:
    cells = core.grid()
    ev = _synthetic_ev(cells)
    rng = np.random.default_rng(0)
    draws = ev[None] + rng.normal(0, 1e-4, (500, len(cells)))
    out = core.intervals(ev, draws, core.effect_weights(cells))
    assert abs(out["R16-R4:L128"].value - 0.02) < 1e-12
    assert all(i.low <= i.value <= i.high for i in out.values())
    assert out["R16-R4:L128"].low > 0.019, "a clear change must have a narrow interval above 0"


def test_best_healthy_and_counterparts() -> None:
    cells = core.grid()
    ev = dict(zip([c.name for c in cells], _synthetic_ev(cells)))
    health = dict.fromkeys(ev, True)
    assert core.best_healthy(ev, health, cells, 4) == "K64_L128_D8_P512_R4_E30_s1"
    assert core.best_healthy(ev, health, cells, 16) == "K128_L128_D8_P512_R16_E30_s1"
    health_no128 = {n: "K128" not in n for n in ev}
    assert core.best_healthy(ev, health_no128, cells, 16) == "K64_L128_D8_P512_R16_E30_s1"
    assert core.best_healthy(ev, dict.fromkeys(ev, False), cells, 8) is None
    assert core.same_at("K64_L128_D8_P512_R4_E30_s1", 16) == "K64_L128_D8_P512_R16_E30_s1"


def test_shift_read() -> None:
    low, high = "K64_L128_D8_P512_R4_E30_s1", "K128_L128_D8_P512_R16_E30_s1"
    up, flat = Interval(0.01, 0.005, 0.015), Interval(0.01, -0.002, 0.02)
    assert core.shift_read(low, high, up, 16) == "shifts"
    assert core.shift_read(low, high, flat, 16) == "stable", "a difference inside the noise is no shift"
    assert core.shift_read(low, core.same_at(low, 16), None, 16) == "stable"
    assert core.shift_read(None, high, up, 16) == "undefined" and core.shift_read(low, None, up, 16) == "undefined"
    _expect_value_error(lambda: core.shift_read(low, high, None, 16), "a shift without its interval was read")


def test_front() -> None:
    points = {"a": (10.0, 0.05), "b": (20.0, 0.10), "c": (20.0, 0.08), "d": (30.0, 0.09), "e": (40.0, 0.20)}
    assert core.front(points) == ["a", "b", "e"]


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
