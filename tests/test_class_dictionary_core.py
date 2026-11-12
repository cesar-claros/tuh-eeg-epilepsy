"""Checks of the pure logic of the class-conditional dictionaries, ``src/class_dictionary_core.py`` (no corpus; CPU).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure. Covered:
the dictionary names, the split of every arm (pass 1's dictionary patients, pure class arms, disjoint stratified
halves) and the checks of gate 2 on good and broken splits, the capped segment draw, the energies of one batch against
the sweep's scoring, the sign of the score on synthetic residuals, the recurrence share on known atoms, the training
check, and the rule with its five outcomes and ``blocked``.
"""

from __future__ import annotations

import copy

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import src.class_dictionary_core as core  # noqa: E402
from src.band_sweep_core import score_setting  # noqa: E402
from src.frozen_readout_core import dictionary_split, segment_draw  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.reconstruction_baselines_core import band_weights  # noqa: E402

SMALL = {"n_atoms": 6, "atom_len": 16, "pool_len": 32, "n_projections": 2}
CHANNELS = 4


def _plan(n: int = 100) -> tuple[dict[str, int], dict[str, int]]:
    subjects = [f"p{i:03d}" for i in range(n)]
    return {s: int(i % 2 == 0) for i, s in enumerate(subjects)}, {s: (i // 2) % 5 for i, s in enumerate(subjects)}


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_names() -> None:
    names = core.dictionary_names()
    assert len(names) == 12 and len(set(names)) == 12 and names[:2] == ["epi_s1", "epi_s2"]
    assert core.parse_name("non-start_s2") == ("non-start", 2) and core.parse_name("A_s1") == ("A", 1)
    for bad in ("C_s1", "epi_s3", "epi"):
        try:
            core.parse_name(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_split_of_every_arm() -> None:
    labels, folds = _plan()
    for fold in range(5):
        split = core.class_split(labels, folds, fold, seed=1)
        assert all(core.check_split(split, labels, folds).values()), core.check_split(split, labels, folds)
        train, val = dictionary_split(labels, folds, fold, 1)
        assert split["dictionary_patients"] == train and split["monitor_patients"] == val
        arms = split["arms"]
        assert arms["epi"]["train"] == [p for p in train if labels[p] == 1]
        assert arms["non"]["val"] == [p for p in val if labels[p] == 0]
        assert arms["A"]["val"] == val and arms["B"]["val"] == val
        assert abs(len(arms["A"]["train"]) - len(arms["epi"]["train"])) <= 1, "a half holds about as many patients"
        assert core.class_split(labels, folds, fold, seed=1) == split, "the split is deterministic"
    assert core.halves(train, labels, 0) != core.halves(train, labels, 1), "the halves depend on the fold"


def test_split_checks_catch_errors() -> None:
    labels, folds = _plan()
    split = core.class_split(labels, folds, 0, seed=1)
    held = next(p for p in labels if folds[p] == 0)
    broken = copy.deepcopy(split)
    broken["arms"]["A"]["train"].append(held)
    checks = core.check_split(broken, labels, folds)
    assert not checks["no_held_out_patient"] and not checks["halves_complete"]
    mixed = copy.deepcopy(split)
    mixed["arms"]["non"]["train"].append(split["arms"]["epi"]["train"][0])
    checks = core.check_split(mixed, labels, folds)
    assert not checks["class_arms_pure"] and not checks["starts_match_their_arms"]
    shifted = copy.deepcopy(split)
    moved = [p for p in shifted["arms"]["B"]["train"] if labels[p] == 1][:3]
    shifted["arms"]["A"]["train"] += moved
    shifted["arms"]["B"]["train"] = [p for p in shifted["arms"]["B"]["train"] if p not in moved]
    assert not core.check_split(shifted, labels, folds)["halves_stratified"]


def test_capped_positions() -> None:
    few = [("rec", float(t)) for t in range(5)]
    assert core.capped_positions("p001", few) == [0, 1, 2, 3, 4]
    many = [("rec", float(t)) for t in range(30)]
    chosen = core.capped_positions("p001", many)
    assert len(chosen) == core.SEGMENT_CAP == 12 and chosen == sorted(set(chosen))
    assert chosen == segment_draw("p001", many, 12) and chosen == core.capped_positions("p001", many)
    assert chosen != core.capped_positions("p002", many), "the draw depends on the patient"


def test_energies_match_the_sweep_scoring() -> None:
    model = SeparableSAE(spec=SeparableSpec(**SMALL, top_r=2), device="cpu")
    width = SMALL["atom_len"] + CHANNELS * SMALL["n_projections"]
    model.atoms = nn.Parameter(model._constrain(_randn(SMALL["n_atoms"], width, seed=1)))
    rows = _randn(3, CHANNELS, 4 * SMALL["pool_len"], seed=2)
    valid = torch.ones(3, CHANNELS, dtype=torch.bool)
    valid[1, 2] = False
    weights = band_weights(rows.shape[-1], 256.0)
    out = core.segment_energies(rows, valid, model, weights)
    with torch.no_grad():
        reference = score_setting(rows, valid, model, weights)
    for key in ("residual_total", "residual_bands", "signal_total", "signal_bands", "atom_wins"):
        assert np.allclose(out[key], reference[key].numpy(), rtol=1e-12, atol=0), key


def test_score_sign_and_seeds() -> None:
    subject = np.repeat(["p1", "p2", "p3"], 2)
    epi = {
        "epi_s1": np.array([0.70, 0.72, 0.80, 0.80, 0.75, 0.75]),
        "epi_s2": np.array([0.71, 0.71, 0.81, 0.79, 0.75, 0.76]),
    }
    non = {
        "non_s1": np.array([0.80, 0.80, 0.70, 0.72, 0.75, 0.75]),
        "non_s2": np.array([0.79, 0.81, 0.71, 0.71, 0.74, 0.75]),
    }
    shares = epi | non
    difference = core.pair_difference(shares, *core.PAIRS["Score"], core.SEEDS)
    patients, score = core.patient_scores(difference, subject)
    assert patients.tolist() == ["p1", "p2", "p3"]
    assert score[0] > 0 > score[1], "a patient whom the epilepsy dictionary fits better scores above 0"
    assert np.isclose(score[0], np.mean([0.10, 0.08, 0.08, 0.10])) and abs(score[2]) < 0.01
    one = core.pair_difference(shares, "non", "epi", (1,))
    assert np.allclose(one, non["non_s1"] - epi["epi_s1"])
    try:
        core.patient_scores(np.array([0.1, np.nan]), np.array(["a", "a"]))
    except ValueError:
        pass
    else:
        raise AssertionError("a non-finite residual share was accepted")
    shares_of = core.residual_shares(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
    assert shares_of[0] == 0.5 and np.isnan(shares_of[1])


def test_recurrence_share() -> None:
    rng = np.random.default_rng(0)
    waveforms = rng.standard_normal((4, 32))
    patterns = np.linalg.qr(rng.standard_normal((4, 20, 3)))[0]
    atoms = (waveforms, patterns)
    assert core.recurrence_share(atoms, atoms) == 1.0
    other = (rng.standard_normal((4, 32)), np.linalg.qr(rng.standard_normal((4, 20, 3)))[0])
    assert core.recurrence_share(atoms, other) == 0.0
    half = (np.concatenate([waveforms[:2], other[0][2:]]), np.concatenate([patterns[:2], other[1][2:]]))
    assert core.recurrence_share(atoms, half) == 0.5


def test_training_moved() -> None:
    assert core.training_moved([0.9, 0.85, 0.8]) and not core.training_moved([0.8, 0.85])
    assert not core.training_moved([0.8]) and not core.training_moved([])


def test_rule() -> None:
    assert core.score_outcome(False, True, True, True) == "blocked"
    assert core.score_outcome(True, True, True, True) == "score_adds"
    assert core.score_outcome(True, True, True, False) == "beats_band_power"
    assert core.score_outcome(True, True, False, True) == "review"
    assert core.score_outcome(True, False, False, True) == "review"
    assert core.score_outcome(True, True, False, False) == "separates_only"
    assert core.score_outcome(True, False, False, False) == "no_advantage"
    assert set(core.OUTCOMES) == {
        "score_adds",
        "beats_band_power",
        "review",
        "separates_only",
        "no_advantage",
        "blocked",
    }


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
