"""Checks of the pure logic of the band-code readout, ``src/band_code_readout_core.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure. Covered:
the counts (with and without log1p), template counts and channel energies of a known code, the asymmetry index (0 for a
symmetric pattern, 1 for a pattern on one side), the channel pairs, the count check, the code details against the
model's own encoder on a small dictionary, that a feature uses only its own segment, the response tails, the energies
with the channel mask, the level and the relation of band power on scaled and raw rows, the recurrence of atoms, and the
rule with its four outcomes and ``blocked``.
"""

from __future__ import annotations

import math

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import src.band_code_readout_core as core  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.spectral_features import SpectralFeatures  # noqa: E402

SMALL = {"n_atoms": 6, "atom_len": 16, "pool_len": 32, "n_projections": 2}
CHANNELS = 4


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _small_model(seed: int, top_r: int) -> SeparableSAE:
    model = SeparableSAE(spec=SeparableSpec(**SMALL, top_r=top_r), device="cpu")
    width = SMALL["atom_len"] + CHANNELS * SMALL["n_projections"]
    model.atoms = nn.Parameter(model._constrain(_randn(SMALL["n_atoms"], width, seed=seed)))
    return model


def _known_details() -> tuple[core.CodeDetails, torch.Tensor]:
    """One segment, 3 atoms, 3 windows of 4 samples, 4 channels and 2 patterns per atom.

    Atom 0 wins window 0 (amplitude 2, pattern 1) and window 2 (amplitude 1, pattern 0); atom 1 wins window 1
    (amplitude 3, pattern 1); atom 2 never wins. Atom 0's patterns are symmetric between channels (0, 1) on the left
    and (2, 3) on the right; atom 1's pattern 1 lies on the left only.
    """
    code = torch.zeros(1, 3, 12, dtype=torch.float64)
    projection = torch.zeros(1, 3, 12, dtype=torch.long)
    code[0, 0, 1], projection[0, 0, 1] = 2.0, 1
    code[0, 0, 11], projection[0, 0, 11] = 1.0, 0
    code[0, 1, 5], projection[0, 1, 5] = 3.0, 1
    projection[0, 2, :] = 1  # a winning projection without a win must not count
    patterns = torch.zeros(3, 4, 2, dtype=torch.float64)
    patterns[0, :, 0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    patterns[0, :, 1] = torch.tensor([0.6, 0.0, 0.0, 0.6]) / math.sqrt(0.72)
    patterns[1, :, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    patterns[1, :, 1] = torch.tensor([0.8, 0.6, 0.0, 0.0])
    patterns[2, :, :] = 0.5
    candidates = torch.tensor([[[2.0, 0.5, 1.0], [0.1, 3.0, 0.2], [0.4, 0.3, 0.9]]], dtype=torch.float64)
    return core.CodeDetails(code, projection, candidates, torch.zeros(1, 4, 12)), patterns


def _features(minutes: float = 1.0) -> dict[str, np.ndarray]:
    details, patterns = _known_details()
    return core.code_features(details, patterns, pool=4, top_r=2, minutes=minutes, left=[0, 1], right=[2, 3])


def test_counts_of_a_known_code() -> None:
    out = _features()
    assert np.allclose(np.expm1(out["hard"]), [[2, 1, 0]]) and np.allclose(np.expm1(out["soft"]), [[3, 3, 0]])
    assert np.allclose(out["code"], np.concatenate([out["hard"], out["soft"]], axis=1))
    assert np.allclose(out["counts"], [[2, 1, 0, 3, 3, 0]]), "the counts of Code-hydra are per minute, without log1p"
    assert np.allclose(np.expm1(out["code"]), out["counts"], rtol=1e-12, atol=0)
    template_hard = np.expm1(out["templates"][0, :6]).reshape(3, 2)
    template_soft = np.expm1(out["templates"][0, 6:]).reshape(3, 2)
    assert np.allclose(template_hard, [[1, 1], [0, 1], [0, 0]]), template_hard
    assert np.allclose(template_soft, [[1, 2], [0, 3], [0, 0]]), template_soft
    assert out["won_total"].tolist() == [3] and out["positive_total"].tolist() == [3] and out["slots"].tolist() == [6]
    half = _features(minutes=2.0)
    assert np.allclose(np.expm1(half["hard"]), [[1, 0.5, 0]]), "counts are per minute"


def test_channel_energy_of_a_known_code() -> None:
    _, patterns = _known_details()
    out = _features()
    energy = np.expm1(out["channels"]).reshape(3, 4)
    expected0 = 4.0 * patterns[0, :, 1].numpy() ** 2 + 1.0 * patterns[0, :, 0].numpy() ** 2
    expected1 = 9.0 * patterns[1, :, 1].numpy() ** 2
    assert np.allclose(energy[0], expected0) and np.allclose(energy[1], expected1) and np.allclose(energy[2], 0)
    assert np.isclose(energy[0].sum(), 4.0 + 1.0), "unit patterns put the energy a^2 on the channels"


def test_asymmetry_index() -> None:
    out = _features()
    assert np.allclose(out["asymmetry"], [[0.0, 1.0, 0.0]]), out["asymmetry"]
    names = [n for pair in core.LEFT_RIGHT for n in pair][::-1] + ["Cz-Pz"]
    left, right = core.channel_index(names)
    assert [names[i] for i in left] == [a for a, _ in core.LEFT_RIGHT]
    assert [names[i] for i in right] == [b for _, b in core.LEFT_RIGHT]
    try:
        core.channel_index(names[1:])
    except ValueError:
        pass
    else:
        raise AssertionError("a missing channel of a pair was accepted")


def test_count_check() -> None:
    good = core.count_check(np.array([3, 6]), np.array([3, 6]), 6)
    assert good["matches_code"] and good["within_slots"] and abs(good["zero_slot_share"] - 0.25) < 1e-12
    assert not core.count_check(np.array([3, 5]), np.array([3, 6]), 6)["matches_code"]
    assert not core.count_check(np.array([3, 7]), np.array([3, 7]), 6)["within_slots"]


def test_details_match_the_model() -> None:
    model = _small_model(seed=1, top_r=2)
    rows = _randn(3, CHANNELS, 4 * SMALL["pool_len"], seed=2)
    valid = torch.ones(3, CHANNELS, dtype=torch.bool)
    valid[1, 2] = False
    details = core.encode_details(model, rows, valid)
    code, _, reconstruction, _, _ = model._encode_reconstruct(rows, valid, model._dictionary())
    assert torch.equal(details.code, code) and torch.equal(details.reconstruction, reconstruction)
    windows = details.code.unflatten(-1, (-1, SMALL["pool_len"])).amax(dim=-1)
    won = windows > 0
    assert (won.sum(dim=1) <= 2).all(), "at most R winners per window"
    assert torch.equal(windows[won], details.candidates[won]), "a winner keeps its best response"
    assert (details.candidates >= windows).all()
    _, patterns = model._split(model._dictionary())
    out = core.code_features(details, patterns.detach(), SMALL["pool_len"], 2, 0.5, [0, 1], [2, 3])
    check = core.count_check(out["won_total"], out["positive_total"], int(out["slots"][0]))
    assert check["matches_code"] and check["within_slots"]
    for i in range(len(rows)):
        alone = core.encode_details(model, rows[i : i + 1], valid[i : i + 1])
        single = core.code_features(alone, patterns.detach(), SMALL["pool_len"], 2, 0.5, [0, 1], [2, 3])
        for name in core.CODE_SETS:
            assert np.allclose(single[name][0], out[name][i], atol=1e-12), f"{name} of segment {i} uses other segments"


def test_tails() -> None:
    candidates = torch.arange(10, dtype=torch.float64).view(1, 1, 10)
    details = core.CodeDetails(torch.zeros(1, 1, 40), torch.zeros(1, 1, 40, dtype=torch.long), candidates, None)
    out = core.code_features(details, torch.ones(1, 4, 1) / 2, pool=4, top_r=1, minutes=1.0, left=[0], right=[1])
    assert np.allclose(np.expm1(out["tails"]), [[8.1, 9.0]]), out["tails"]
    assert np.allclose(np.expm1(_features()["tails"]).reshape(3, 2)[:, 1], [2.0, 3.0, 0.9])


def test_segment_energy_with_the_mask() -> None:
    rows = _randn(2, 3, 8, seed=3)
    reconstruction = _randn(2, 3, 8, seed=4)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    residual, signal = core.segment_energy(rows, reconstruction, valid)
    expected = (rows - reconstruction * valid.unsqueeze(-1)).pow(2).sum(dim=(1, 2)).numpy()
    assert np.allclose(residual, expected) and np.allclose(signal, rows.pow(2).sum(dim=(1, 2)).numpy())


def test_level_and_band_power_on_scaled_rows() -> None:
    noise = _randn(2, CHANNELS, 4096, seed=5).float()
    noise = noise - noise.median(dim=-1, keepdim=True).values
    gains = torch.tensor([1.0, 3.0, 10.0, 30.0]).view(1, CHANNELS, 1)
    x = noise * gains
    level = core.level_features(x)
    assert np.allclose(level - core.level_features(noise), np.log10(gains.view(1, -1).numpy()), atol=1e-5)
    reader = SeparableSAE(spec=SeparableSpec(**SMALL, top_r=1), device="cpu")
    rows, valid = reader._full_rows(x)
    assert valid.all() and rows.abs().max() < 20, "the test needs rows without the clip"
    spectral = SpectralFeatures(sfreq=256.0, bands=core.BANDS, nperseg=512, device="cpu")
    scaled, raw = (spectral(t).double().view(2, CHANNELS, 8).numpy() for t in (rows, x))
    assert np.allclose(scaled[..., 4:], raw[..., 4:], atol=1e-5), "the relative powers must agree"
    shift = scaled[..., :4] - raw[..., :4]
    assert np.allclose(shift, -2 * level[..., None], atol=1e-4), "log power on scaled rows = raw - 2 log10(1.4826 MAD)"


def test_recur() -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal(32)
    u = np.linalg.qr(rng.standard_normal((20, 3)))[0]
    assert core.recur((w, u), (w, u))
    assert core.recur((w, u), (-np.roll(np.pad(w, 4), 3), u[:, ::-1])), "sign, lag and pattern order do not matter"
    assert not core.recur((w, u), (rng.standard_normal(32), u)), "another waveform"
    other = np.linalg.qr(rng.standard_normal((20, 3)))[0]
    assert not core.recur((w, u), (w, other)), "other topographies"


def test_rule() -> None:
    assert core.readout_outcome(False, True, True, True) == "blocked"
    assert core.readout_outcome(True, True, True, True) == "code_adds"
    assert core.readout_outcome(True, True, True, False) == "beats_band_power"
    assert core.readout_outcome(True, True, False, True) == "review"
    assert core.readout_outcome(True, False, True, True) == "review"
    assert core.readout_outcome(True, False, True, False) == "no_advantage"
    assert core.readout_outcome(True, False, False, False) == "no_advantage"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
