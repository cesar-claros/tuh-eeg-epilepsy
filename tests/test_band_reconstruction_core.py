"""Checks of the pure logic of the band-limited reconstruction, ``src/band_reconstruction_core.py`` (no corpus; CPU).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the band columns, the band-pass gain and its edge error on synthetic 1/f noise, the 1-30 Hz bank, the
rank-r ceiling against the SVD and against random separable codes at R=1 and R=4, the scoring of one batch with its
mask, ceilings, budget and edge energy, and the outcomes of the two-arm rule.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from torch import nn  # noqa: E402

import src.band_reconstruction_core as core  # noqa: E402
import src.reconstruction_baselines_core as base  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402

SMALL = {"n_atoms": 6, "atom_len": 16, "pool_len": 32, "n_projections": 2}
CHANNELS = 4


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _small_model(seed: int, top_r: int) -> SeparableSAE:
    model = SeparableSAE(spec=SeparableSpec(**SMALL, top_r=top_r), device="cpu")
    width = SMALL["atom_len"] + CHANNELS * SMALL["n_projections"]
    model.atoms = nn.Parameter(model._constrain(_randn(SMALL["n_atoms"], width, seed=seed)))
    return model


def test_band_columns_and_arms() -> None:
    assert [base.BANDS[i] for i in core.IN_BAND] == [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
    assert [base.BANDS[i] for i in core.FAST] == [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
    assert set(core.BAR.values()) <= set(core.ARMS) and set(core.FIXED.values()) <= set(core.ARMS)
    assert core.ARMS[: len(core.MODEL_ARMS)] == core.MODEL_ARMS and {"SVD-1", "SVD-4"} <= set(core.ARMS)


def test_band_pass_gain() -> None:
    t = np.arange(30_720) / base.SFREQ
    middle = slice(5 * 256, -5 * 256)

    def gain(f: float) -> float:
        x = np.sin(2 * np.pi * f * t)
        y = core.band_pass(x)
        return float((y[middle] ** 2).mean() / (x[middle] ** 2).mean())

    assert gain(10.0) > 0.99 and gain(2.0) > 0.99, "the pass band must pass"
    assert abs(gain(1.0) - 0.25) < 0.01 and abs(gain(30.0) - 0.25) < 0.01, "each edge must be -6 dB after two passes"
    assert gain(0.5) < 1e-3 and gain(40.0) < 1e-2, "the stop bands must stop"
    x32 = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    assert core.band_pass(x32).dtype == np.float32


def test_band_pass_edge_error_on_one_over_f_noise() -> None:
    rng = np.random.default_rng(0)
    seg, fs = 30_720, 256
    errors, second = [], []
    for _ in range(4):
        n = 6 * seg
        f = np.fft.rfftfreq(n, 1 / fs)
        f[0] = f[1]
        x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / f**0.5, n)
        x = sosfiltfilt(butter(4, 0.1, btype="highpass", fs=fs, output="sos"), x)
        truth = core.band_pass(x)[2 * seg : 3 * seg]
        alone = core.band_pass(x[2 * seg : 3 * seg])
        e = (alone - truth) ** 2 / truth.var()
        errors.append(e.mean())
        second.append(max(e[fs : 2 * fs].mean(), e[-2 * fs : -fs].mean()))
    assert max(errors) < 0.01, f"segment error {max(errors):.4f}: filter the cache with context instead"
    assert max(second) < 0.01, f"the error reaches past the first second: {max(second):.4f}"


def test_band_bank_lies_in_the_band() -> None:
    bank = core.band_bank()
    frequencies = base.bank_frequencies(low=core.BAND_LOW_HZ, high=core.BAND_HIGH_HZ)
    assert bank.shape == (64, 256) and np.allclose(np.linalg.norm(bank, axis=1), 1.0)
    assert np.allclose(bank[0::2], -bank[1::2]) and frequencies[0] == 1.0 and abs(frequencies[-1] - 30.0) < 1e-9
    spectrum = np.abs(np.fft.rfft(bank[0::2], n=8192, axis=1))
    peaks = np.fft.rfftfreq(8192, 1 / 256.0)[spectrum.argmax(axis=1)]
    assert peaks.max() < 31.0, "an atom peaks above the band"


def test_rank_r_is_the_svd_optimum_and_bounds_separable_codes() -> None:
    rows = _randn(3, CHANNELS, 5 * 32, seed=1)
    sv = torch.linalg.svdvals(base.windows(rows, 32)).pow(2)
    for r in (1, 2, 4):
        residual = base.window_energy(rows - core.rank_reconstruction(rows, r, 32), 32)
        assert torch.allclose(residual, sv.sum(-1) - sv[..., :r].sum(-1), atol=1e-9), f"rank {r} is not optimal"
    assert torch.allclose(core.rank_reconstruction(rows, 1, 32), base.rank1_reconstruction(rows, 32))
    valid = torch.ones(3, CHANNELS, dtype=torch.bool)
    for r in (1, 4):
        ceiling = base.window_energy(rows - core.rank_reconstruction(rows, r, 32), 32)
        for seed in range(3):
            model = _small_model(seed=10 + seed, top_r=r)
            code = model._encode_reconstruct(rows, valid, model._dictionary())[2]
            assert bool((ceiling <= base.window_energy(rows - code, 32) + 1e-9).all()), f"an R={r} code beat SVD-{r}"
    _expect_value_error(lambda: core.rank_reconstruction(rows, 5, 32), "a rank above the channel count was accepted")


def test_edge_energy() -> None:
    rows = torch.ones(2, 3, 640, dtype=torch.float64)
    rows[:, :, :64] = 2.0
    edge, interior = core.edge_energy(rows, sfreq=64.0, seconds=1.0)
    assert torch.allclose(edge, torch.full((2,), 3 * (4 * 64 + 64) / 2.0, dtype=torch.float64))
    assert torch.allclose(interior, torch.full((2,), 3 * 64.0, dtype=torch.float64))
    _expect_value_error(lambda: core.edge_energy(rows[..., :100], sfreq=64.0), "edges covering the segment passed")


def _models() -> dict[str, SeparableSAE]:
    models = {}
    for i, name in enumerate(core.MODEL_ARMS):
        r = 4 if name.startswith("B4") else 1
        models[name] = _small_model(seed=40 + i, top_r=r)
    return models


def test_score_arms_respects_ceilings_mask_and_budget() -> None:
    models = _models()
    rows = _randn(3, CHANNELS, 4 * 32, seed=2)
    valid = torch.ones(3, CHANNELS, dtype=torch.bool)
    moment = torch.zeros(128, 128, dtype=torch.float64)
    base.accumulate_moments(rows, moment, torch.zeros(4, 4, dtype=torch.float64), 32)
    basis, _ = base.top_components(moment, 64)
    weights = base.band_weights(rows.shape[-1], sfreq=32.0)
    slow = {name: torch.zeros(SMALL["n_atoms"], dtype=torch.bool) for name in core.BUDGET_ARMS}
    slow["B4-trained"][:3] = True
    out = core.score_arms(rows, valid, models, basis, weights, slow, pool=32, sfreq=32.0)
    assert out["residual_bands"].shape == (len(core.ARMS), 3, len(base.BANDS))
    assert int(out["ceiling_violations"].sum()) == 0 and float(out["ceiling_excess"].max()) <= base.CEILING_TOLERANCE
    assert torch.allclose(out["residual_bands"].sum(-1), out["residual_total"], rtol=1e-10)
    i1, i4 = core.BUDGET_ARMS.index("B1-trained"), core.BUDGET_ARMS.index("B4-trained")
    assert torch.equal(out["slots_won"][i1], out["windows_won"][i1]), "at R=1 every winning window has one atom"
    assert bool((out["slots_won"][i4] <= 4 * out["windows_won"][i4]).all())
    assert int(out["slots_won"][i4].sum()) > int(out["windows_won"][i4].sum()), "R=4 must keep several atoms"
    assert int(out["slots_slow"][i1].sum()) == 0 and int(out["windows_slow"][i1].sum()) == 0
    assert bool((out["slots_slow"][i4] <= out["slots_won"][i4]).all())
    assert torch.equal(out["atom_wins"].sum(dim=1), out["slots_won"].sum(dim=1))
    assert bool((out["edge_energy"] > 0).all()) and bool((out["interior_energy"] > 0).all())
    none = core.score_arms(rows, torch.zeros_like(valid), models, basis, weights, slow, pool=32, sfreq=32.0)
    for i, name in enumerate(core.ARMS):
        assert torch.allclose(none["residual_total"][i], none["signal_total"]), f"{name} reconstructed a rejected row"
    models["B1-fixed"] = _small_model(seed=99, top_r=2)
    _expect_value_error(
        lambda: core.score_arms(rows, valid, models, basis, weights, slow, pool=32, sfreq=32.0),
        "a model without a ceiling was accepted",
    )


def test_two_arm_rule() -> None:
    up, flat = base.Interval(0.1, 0.05, 0.15), base.Interval(0.0, -0.01, 0.01)
    gates = {"ceilings": True, "band_pass_ran": True}
    passing = core.arm_conditions(up, 0.10, up, up)
    assert passing == {"T": True, "F1": True, "F2": True, "F3": True}
    slow_only = core.arm_conditions(up, 0.02, up, up)
    assert not slow_only["F1"] and core.arm_conditions(up, core.FAST_FLOOR, up, up)["F1"]
    assert core.band_outcome(gates, {1: slow_only, 4: passing}) == "sweep", "R=4 alone must give sweep"
    assert core.band_outcome(gates, {1: passing, 4: slow_only}) == "sweep"
    assert core.band_outcome(gates, {1: slow_only, 4: slow_only}) == "delta_only"
    failed_total = core.arm_conditions(flat, 0.10, up, up)
    assert core.band_outcome(gates, {1: failed_total, 4: slow_only}) == "stop"
    assert (
        core.band_outcome(gates, {1: core.arm_conditions(up, 0.1, flat, up), 4: core.arm_conditions(up, 0.1, up, flat)})
        == "delta_only"
    )
    assert core.band_outcome({**gates, "ceilings": False}, {1: passing, 4: passing}) == "blocked"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
