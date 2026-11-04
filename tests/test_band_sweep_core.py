"""Checks of the pure logic of the band-limited sweep, ``src/band_sweep_core.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: setting names and budgets, the 16 factorial settings and the screen list, contrasts that recover known effects
and their bootstrap intervals, the separable spec of a setting and its check, the scoring of one setting with its own
ceiling, near-copy groups, the health condition, and the selection and confirmation rules.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import src.band_sweep_core as core  # noqa: E402
import src.reconstruction_baselines_core as base  # noqa: E402
from src.models.components.separable_sae import SeparableSAE  # noqa: E402
from src.models.components.waveform_sae import WaveformSpec  # noqa: E402


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def test_setting_names_and_budgets() -> None:
    assert core.BASE.name == "K64_L256_D4_P512_R4_E30_s1"
    assert core.parse_setting(core.BASE.name) == core.BASE
    assert core.parse_setting(core.BASE.with_seed(2).name).seed == 2
    assert core.BASE.amplitudes_per_second == 2.0
    assert abs(core.BASE.bits_per_second - 4 * (6 + 9 + 2) * 0.5) < 1e-12
    _expect_value_error(lambda: core.parse_setting("K64_L256"), "a partial name was parsed")
    _expect_value_error(lambda: core.parse_setting(core.BASE.name.replace("_s1", "_s3")), "seed 3 was parsed")


def test_factorial_holds_sixteen_settings_at_one_budget() -> None:
    cells = core.factorial()
    assert len(cells) == 16 and cells[0][1] == core.BASE and cells[0][0] == (-1, -1, -1, -1)
    assert len({s for _, s in cells}) == 16
    assert {s.amplitudes_per_second for _, s in cells} == {2.0}, "the screen must hold the budget fixed"
    for codes, setting in cells:
        assert (setting.K == 128) == (codes[0] > 0) and (setting.L == 128) == (codes[1] > 0)
        assert (setting.D == 8) == (codes[2] > 0) and (setting.P, setting.R) == ((256, 2) if codes[3] > 0 else (512, 4))
    screen = core.screen_settings()
    assert len(screen) == 18 and core.BASE not in screen and len({s.name for s in screen}) == 18
    assert {s.R for s in core.DESCRIPTIVE.values()} == {4, 8, 16} and core.DESCRIPTIVE["E60"].epochs == 60


def test_contrasts_recover_known_effects() -> None:
    codes = np.array([c for c, _ in core.factorial()], dtype=float)
    k, l_, d, pr = codes.T
    values = 0.1 + 0.02 * k + 0.01 * l_ - 0.004 * pr + 0.005 * k * l_ + 0.002 * k * d * pr
    effects = core.contrasts(values)
    assert len(effects) == 15 and set(effects) == set(core.effect_names())
    expected = {"K": 0.04, "L": 0.02, "PR": -0.008, "K:L": 0.01, "K:D:PR": 0.004}
    for name, value in effects.items():
        assert abs(value - expected.get(name, 0.0)) < 1e-12, (name, value)


def test_effect_intervals_wrap_the_point_effects() -> None:
    rng = np.random.default_rng(0)
    signal = rng.uniform(1.0, 2.0, 40)
    codes = np.array([c for c, _ in core.factorial()], dtype=float)
    ev = 0.1 + 0.03 * codes[:, 0]
    residuals = (1 - ev)[:, None] * signal[None] * rng.uniform(0.98, 1.02, (16, 40))
    counts = base.bootstrap_counts(np.arange(40) % 2, n_boot=300)
    intervals = core.effect_intervals(residuals, signal, counts)
    point = core.contrasts(np.array([base.pooled_ev(r, signal) for r in residuals]))
    for name, interval in intervals.items():
        assert interval.value == point[name] and interval.low <= interval.value <= interval.high
    assert intervals["K"].low > 0.05, "a clear main effect must have an interval far above 0"


def test_separable_spec_and_check() -> None:
    setting = core.Setting(K=128, L=128, D=8, P=256, R=2)
    spec = core.separable_spec(setting, WaveformSpec())
    assert (spec.n_atoms, spec.atom_len, spec.n_projections, spec.pool_len, spec.top_r) == (128, 128, 8, 256, 2)
    assert spec.loss_support == "extended" and spec.step_matched
    model = SeparableSAE(spec=spec, device="cpu")
    core.check_setting(model, setting)
    _expect_value_error(lambda: core.check_setting(model, core.BASE), "a model of another setting was accepted")


def test_score_setting_uses_its_own_ceiling() -> None:
    setting = core.Setting(K=6, L=8, D=2, P=16, R=2)
    model = SeparableSAE(spec=core.separable_spec(setting, WaveformSpec()), device="cpu")
    model.atoms = nn.Parameter(
        model._constrain(torch.randn(6, 8 + 4 * 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
    )
    rows = torch.randn(3, 4, 8 * 16, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    valid = torch.ones(3, 4, dtype=torch.bool)
    out = core.score_setting(rows, valid, model, base.band_weights(rows.shape[-1], sfreq=32.0))
    assert int(out["ceiling_violations"]) == 0 and float(out["ceiling_excess"]) <= base.CEILING_TOLERANCE
    assert torch.allclose(out["residual_bands"].sum(-1), out["residual_total"], rtol=1e-10)
    assert bool((out["ceiling_total"] <= out["residual_total"] + 1e-9).all()), "SVD-2 must bound an R=2 code"
    assert int(out["atom_wins"].sum()) <= 2 * 3 * 8 and out["atom_wins"].shape == (6,)


def test_near_copy_groups_and_health() -> None:
    rng = np.random.default_rng(3)
    wave = rng.standard_normal(32)
    copies = np.stack([np.roll(np.pad(wave, 8), shift)[:48] for shift in range(5)])
    others = rng.standard_normal((10, 48))
    assert core.largest_near_copy_group(np.vstack([copies, others])) == 5, "shifted copies must form one group"
    assert core.largest_near_copy_group(others) == 1
    assert core.healthy([np.r_[np.zeros(6), np.ones(58)]]) and not core.healthy([np.r_[np.zeros(7), np.ones(57)]])


def test_selection_rule() -> None:
    up, flat = base.Interval(0.02, 0.01, 0.03), base.Interval(0.01, -0.005, 0.02)
    ranked = [("a", 0.09), ("b", 0.12), ("c", 0.11), ("d", 0.08)]
    versus = {"a": up, "b": flat, "c": up, "d": up}
    health = {"a": True, "b": True, "c": False, "d": True}
    assert core.select(ranked, versus, health) == "a", "the best setting with an interval above 0 and good health wins"
    assert core.select(ranked, {**versus, "a": flat, "d": flat}, health) is None


def test_confirmation_rule() -> None:
    big, small, flat = (
        base.Interval(0.02, 0.01, 0.03),
        base.Interval(0.006, 0.002, 0.01),
        base.Interval(0.02, -0.01, 0.05),
    )
    assert core.confirmation_outcome({1: big, 2: big}, True) == "adopt"
    assert core.confirmation_outcome({1: big, 2: flat}, True) == "base_holds", "both seeds must beat the base"
    assert core.confirmation_outcome({1: small, 2: small}, True) == "base_holds", "the margin must hold"
    assert core.confirmation_outcome({1: big, 2: big}, False) == "base_holds", "the health condition must hold"
    assert core.confirmation_outcome({1: big, 2: small}, True) == "adopt", "the margin is on the mean gain"
    _expect_value_error(lambda: core.confirmation_outcome({1: big}, True), "one seed was accepted")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
