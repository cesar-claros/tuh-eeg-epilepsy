"""Checks of the pure logic of the reconstruction baselines, ``src/reconstruction_baselines_core.py`` (no corpus; CPU).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the fixed waveform bank, the leading components, the fixed atoms inside a ``SeparableSAE``, the windows,
the second moments, the PCA at full rank and its rise with ``k``, the rank-1 ceiling against the SVD and against a
random separable dictionary, the band weights with Parseval, the scoring of one batch with its mask, the patient
bootstrap and its intervals, and the outcomes of the rule.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import src.reconstruction_baselines_core as core  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402

# A small geometry for the scoring checks: PCA needs C * P >= 64.
SMALL = {"n_atoms": 6, "atom_len": 16, "pool_len": 32, "n_projections": 2}
SMALL_CHANNELS = 4


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _small_model(seed: int) -> SeparableSAE:
    model = SeparableSAE(spec=SeparableSpec(**SMALL), device="cpu")
    width = SMALL["atom_len"] + SMALL_CHANNELS * SMALL["n_projections"]
    model.atoms = nn.Parameter(model._constrain(_randn(SMALL["n_atoms"], width, seed=seed)))
    return model


def test_bank_has_unit_waveforms_in_signed_pairs_at_their_frequencies() -> None:
    bank = core.gabor_bank()
    frequencies = core.bank_frequencies()
    assert bank.shape == (2 * core.BANK_FREQUENCIES, core.ATOM_LEN) == (core.N_ATOMS, core.ATOM_LEN)
    assert np.allclose(np.linalg.norm(bank, axis=1), 1.0)
    assert np.allclose(bank[0::2], -bank[1::2]), "each waveform must appear with both signs"
    assert int((frequencies < 4.0).sum()) == 15 and frequencies[0] == 0.5 and abs(frequencies[-1] - 45.0) < 1e-9
    spectrum = np.abs(np.fft.rfft(bank[0::2], n=8192, axis=1))
    peaks = np.fft.rfftfreq(8192, 1.0 / core.SFREQ)[spectrum.argmax(axis=1)]
    fast = frequencies >= 4.0
    assert np.all(np.abs(peaks[fast] - frequencies[fast]) <= 0.05 * frequencies[fast]), (
        "a waveform misses its frequency"
    )
    slowest = bank[0]
    assert slowest.min() > -1e-12 and slowest.argmax() in (127, 128), "the slowest atom must be a centered bump"


def test_top_components_are_orthonormal_descending_and_sign_fixed() -> None:
    a = _randn(12, 12, seed=1)
    matrix = a @ a.T
    vectors, values = core.top_components(matrix, 5)
    assert vectors.shape == (12, 5) and values.shape == (12,)
    assert torch.allclose(vectors.T @ vectors, torch.eye(5, dtype=vectors.dtype), atol=1e-10)
    assert bool((values[:-1] >= values[1:]).all()), "eigenvalues must be descending"
    assert torch.allclose(matrix @ vectors, vectors * values[:5], atol=1e-8)
    lead = vectors.gather(0, vectors.abs().argmax(dim=0, keepdim=True))
    assert bool((lead > 0).all()), "the largest entry of every vector must be positive"
    flipped, _ = core.top_components(matrix, 5)
    assert torch.equal(vectors, flipped)
    _expect_value_error(lambda: core.top_components(matrix[:, :5], 2), "a non-square matrix was accepted")


def test_fixed_atoms_come_back_from_the_model() -> None:
    spec = SeparableSpec(n_atoms=core.N_ATOMS, atom_len=core.ATOM_LEN, pool_len=core.POOL_LEN, n_projections=4)
    model = SeparableSAE(spec=spec, device="cpu")
    model.atoms = nn.Parameter(torch.zeros(core.N_ATOMS, core.ATOM_LEN + 20 * 4))
    patterns = np.linalg.qr(np.random.default_rng(0).standard_normal((20, 4)))[0]
    bank = core.gabor_bank()
    core.set_fixed_atoms(model, bank, patterns)
    expected = np.einsum("cd,kl->kdcl", patterns, bank)
    assert np.allclose(model._templates(integrated=False), expected, atol=1e-6), "the model changed the fixed atoms"
    assert not model.atoms.requires_grad
    _expect_value_error(lambda: core.set_fixed_atoms(model, bank[:10], patterns), "a short bank was accepted")
    _expect_value_error(lambda: core.set_fixed_atoms(model, 2 * bank, patterns), "non-unit waveforms were accepted")
    _expect_value_error(lambda: core.set_fixed_atoms(model, bank, patterns[:, :2]), "two patterns were accepted")


def test_windows_round_trip_and_refuse_partial_windows() -> None:
    rows = _randn(2, 3, 4 * 8, seed=2)
    parts = core.windows(rows, 8)
    assert parts.shape == (2, 4, 3, 8) and torch.equal(parts[1, 2], rows[1, :, 16:24])
    assert torch.equal(core.unwindow(parts), rows)
    _expect_value_error(lambda: core.windows(rows[..., :-1], 8), "a partial window was accepted")


def test_moments_match_a_direct_sum() -> None:
    rows = _randn(3, 2, 4 * 8, seed=3)
    window_moment = torch.zeros(16, 16, dtype=torch.float64)
    channel_moment = torch.zeros(2, 2, dtype=torch.float64)
    assert core.accumulate_moments(rows, window_moment, channel_moment, 8) == (12, 96)
    flat = torch.stack([rows[b, :, 8 * m : 8 * (m + 1)].flatten() for b in range(3) for m in range(4)])
    assert torch.allclose(window_moment, flat.T @ flat)
    assert torch.allclose(channel_moment, sum(rows[b] @ rows[b].T for b in range(3)))


def test_pca_is_exact_at_full_rank_and_rises_with_k() -> None:
    rows = _randn(6, 2, 4 * 8, seed=4) * torch.linspace(3.0, 0.2, 8).repeat(4)
    window_moment = torch.zeros(16, 16, dtype=torch.float64)
    core.accumulate_moments(rows, window_moment, torch.zeros(2, 2, dtype=torch.float64), 8)
    basis, values = core.top_components(window_moment, 16)
    reconstructions = core.pca_reconstructions(rows, basis, (1, 2, 4, 8, 16), 8)
    energy = rows.pow(2).sum()
    ev = {k: float(1 - (rows - r).pow(2).sum() / energy) for k, r in reconstructions.items()}
    assert abs(ev[16] - 1.0) < 1e-10, "PCA at full rank must reconstruct exactly"
    assert all(ev[a] <= ev[b] + 1e-12 for a, b in zip((1, 2, 4, 8), (2, 4, 8, 16))), f"EV fell with k: {ev}"
    assert abs(ev[1] - float(values[0] / values.sum())) < 1e-10, "in-sample PCA-1 EV is the top eigenvalue share"
    _expect_value_error(lambda: core.pca_reconstructions(rows, basis[:, :2], (1, 4), 8), "a short basis was accepted")


def test_rank1_is_the_svd_optimum_and_bounds_a_separable_code() -> None:
    rows = _randn(3, 4, 5 * 32, seed=5)
    reconstruction = core.rank1_reconstruction(rows, 32)
    residual = core.window_energy(rows - reconstruction, 32)
    sv = torch.linalg.svdvals(core.windows(rows, 32))
    assert torch.allclose(residual, sv.pow(2).sum(-1) - sv[..., 0].pow(2), atol=1e-9)
    for seed in range(5):
        u, v = _randn(3, 5, 4, 1, seed=10 + seed), _randn(3, 5, 1, 32, seed=20 + seed)
        other = core.unwindow(u @ v)
        assert bool((residual <= core.window_energy(rows - other, 32) + 1e-9).all()), "a random rank-1 code won"
    model = _small_model(seed=6)
    valid = torch.ones(3, 4, dtype=torch.bool)
    s4 = model._encode_reconstruct(rows, valid, model._dictionary())[2]
    assert bool((residual <= core.window_energy(rows - s4, 32) + 1e-9).all()), "the separable code beat the ceiling"


def test_band_weights_follow_parseval_and_place_a_tone() -> None:
    n_times = 30_720
    weights = core.band_weights(n_times)
    assert weights.shape == (len(core.BANDS), n_times // 2 + 1) and core.fast_bands() == [2, 3, 4, 5]
    rows = _randn(2, 3, n_times, seed=7)
    bands = core.band_energy(rows, weights)
    assert torch.allclose(bands.sum(dim=1), rows.pow(2).sum(dim=(1, 2)), rtol=1e-10)
    t = torch.arange(n_times, dtype=torch.float64) / core.SFREQ
    tone = torch.sin(2 * torch.pi * 10.0 * t).expand(1, 2, -1)
    share = core.band_energy(tone, weights)[0] / tone.pow(2).sum()
    assert share[core.BAND_NAMES.index("8_13")] > 1 - 1e-9, "a 10 Hz tone must fall in the 8 to 13 Hz band"
    odd = core.band_weights(1001)
    rows = _randn(1, 1, 1001, seed=8)
    assert torch.allclose(core.band_energy(rows, odd).sum(), rows.pow(2).sum(), rtol=1e-10)
    _expect_value_error(lambda: core.band_weights(64, bands=((0.0, 10.0), (20.0, np.inf))), "a gap was accepted")


def test_score_batch_respects_the_ceiling_and_masks_rejected_rows() -> None:
    models = {name: _small_model(seed=30 + i) for i, name in enumerate(core.S4_ARMS)}
    rows = _randn(3, SMALL_CHANNELS, 4 * 32, seed=9)
    valid = torch.ones(3, SMALL_CHANNELS, dtype=torch.bool)
    window_moment = torch.zeros(128, 128, dtype=torch.float64)
    core.accumulate_moments(rows, window_moment, torch.zeros(4, 4, dtype=torch.float64), 32)
    basis, _ = core.top_components(window_moment, 64)
    weights = core.band_weights(rows.shape[-1])
    slow = torch.ones(SMALL["n_atoms"], dtype=torch.bool)
    result = core.score_batch(rows, valid, models, basis, weights, slow, pool=32)
    assert result.residual_bands.shape == (len(core.ARMS), 3, len(core.BANDS))
    assert int(result.ceiling_violations.sum()) == 0 and float(result.ceiling_excess.max()) <= core.CEILING_TOLERANCE
    assert torch.allclose(result.residual_bands.sum(-1), result.residual_total, rtol=1e-10)
    assert torch.allclose(result.signal_bands.sum(-1), result.signal_total, rtol=1e-10)
    ceiling = result.residual_total[core.ARMS.index("SVD-1")]
    for name in core.S4_ARMS:
        assert bool((ceiling <= result.residual_total[core.ARMS.index(name)] + 1e-9).all())
    assert bool((result.windows_won <= 4).all()) and torch.equal(result.windows_slow, result.windows_won)
    assert torch.allclose(result.explained_slow, result.explained_all)
    assert int(result.atom_wins.sum()) == int(result.windows_won.sum())
    none = core.score_batch(rows, torch.zeros_like(valid), models, basis, weights, slow, pool=32)
    for i, name in enumerate(core.ARMS):
        assert torch.allclose(none.residual_total[i], none.signal_total), f"{name} reconstructed a rejected row"
    assert int(none.usable_rows.sum()) == 0


def test_bootstrap_keeps_fold_sizes_and_is_seeded() -> None:
    folds = np.repeat(np.arange(5), [3, 4, 3, 4, 2])
    counts = core.bootstrap_counts(folds, n_boot=200, seed=1)
    assert counts.shape == (200, 16) and (counts.sum(axis=1) == 16).all()
    for fold in range(5):
        assert (counts[:, folds == fold].sum(axis=1) == (folds == fold).sum()).all(), "a fold changed size"
    assert np.array_equal(counts, core.bootstrap_counts(folds, n_boot=200, seed=1))
    assert not np.array_equal(counts, core.bootstrap_counts(folds, n_boot=200, seed=2))


def test_pooled_ev_and_its_intervals() -> None:
    rng = np.random.default_rng(3)
    signal = rng.uniform(1.0, 2.0, 30)
    residual = signal * rng.uniform(0.6, 0.8, 30)
    counts = core.bootstrap_counts(np.arange(30) % 5, n_boot=500)
    assert abs(core.pooled_ev(residual, signal) - (1 - residual.sum() / signal.sum())) < 1e-12
    assert np.allclose(core.pooled_ev(residual, signal, np.ones((2, 30))), core.pooled_ev(residual, signal))
    interval = core.ev_interval(residual, signal, counts)
    assert interval.low <= interval.value <= interval.high and interval.high - interval.low < 0.1
    same = core.paired_interval(residual, residual, signal, counts)
    assert same.value == same.low == same.high == 0.0
    better = core.paired_interval(0.5 * residual, residual, signal, counts)
    assert better.low > 0, "a uniformly better arm must have an interval above 0"


def test_parseval_error_and_smallest_k() -> None:
    bands = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert core.parseval_error(bands, np.array([3.0, 7.0])) == 0.0
    assert abs(core.parseval_error(bands, np.array([3.0, 8.0])) - 1 / 8) < 1e-12
    curve = {1: 0.1, 2: 0.2, 4: 0.3}
    assert (
        core.smallest_k(curve, 0.15) == 2 and core.smallest_k(curve, 0.3) == 4 and core.smallest_k(curve, 0.5) is None
    )


def test_rule_outcomes() -> None:
    up, down = core.Interval(0.1, 0.05, 0.15), core.Interval(0.0, -0.01, 0.01)
    gates = {"ceiling": True, "parseval": True}
    met = core.conditions(up, 0.10, up, up)
    assert met == {"T": True, "H1": True, "H2": True, "H3": True}
    assert core.reconstruction_outcome(gates, met) == "sweep"
    assert core.reconstruction_outcome({**gates, "ceiling": False}, met) == "blocked"
    assert core.reconstruction_outcome(gates, core.conditions(down, 0.10, up, up)) == "stop"
    assert core.reconstruction_outcome(gates, core.conditions(up, 0.04, up, up)) == "slow_only", "the floor was ignored"
    assert core.reconstruction_outcome(gates, core.conditions(up, 0.10, down, up)) == "slow_only"
    assert core.reconstruction_outcome(gates, core.conditions(up, 0.10, up, down)) == "slow_only"
    assert core.conditions(up, core.FAST_FLOOR, up, up)["H1"], "the floor itself must pass"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
