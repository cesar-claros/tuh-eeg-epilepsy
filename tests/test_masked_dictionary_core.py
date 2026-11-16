"""Checks of the pure logic of the class-masked dictionary, ``src/masked_dictionary_core.py`` (no corpus; CPU).

Runs under pytest. Covered: the dictionary names, the permuted labels (class counts kept, never the true labels,
fixed by the fold), the split of a fold and the checks of gate 2 on good and broken splits, the group counts of a
known code, the shares and their seed mean, the patient scores, the training check, and the rule with its outcomes.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import rootutils
import torch

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.masked_dictionary_core as core  # noqa: E402
from src.frozen_readout_core import dictionary_split  # noqa: E402


def _plan(n: int = 100) -> tuple[dict[str, int], dict[str, int]]:
    subjects = [f"p{i:03d}" for i in range(n)]
    return {s: int(i % 2 == 0) for i, s in enumerate(subjects)}, {s: (i // 2) % 5 for i, s in enumerate(subjects)}


def test_names() -> None:
    names = core.dictionary_names()
    assert names == ["sup_s1", "sup_s2", "perm_s1", "perm_s2", "mask_s1", "mask_s2", "strong_s1"]
    assert core.parse_name("perm_s2") == ("perm", 2) and core.parse_name("strong_s1") == ("strong", 1)
    for bad in ("strong_s2", "sup_s3", "sup", "epi_s1"):
        with pytest.raises(ValueError):
            core.parse_name(bad)
    assert core.HINGE == {"sup": 1.0, "perm": 1.0, "mask": 0.0, "strong": 10.0}
    assert core.N_SHARED == 64 and core.N_EPILEPSY_ONLY == 16
    assert {arm for arm, _ in core.SCORES.values()} <= set(core.HINGE)


def test_permuted_labels() -> None:
    labels, _ = _plan(40)
    patients = sorted(labels)
    first = core.permuted_labels(patients, labels, fold=0, part=0)
    assert set(first) == set(patients) and sum(first.values()) == sum(labels.values())
    assert first != labels and first == core.permuted_labels(patients[::-1], labels, fold=0, part=0)
    assert first != core.permuted_labels(patients, labels, fold=1, part=0)
    assert first != core.permuted_labels(patients, labels, fold=0, part=1)
    pair = {"a": 0, "b": 1}  # two patients: the only other arrangement is the swap
    assert core.permuted_labels(["a", "b"], pair, fold=0, part=0) == {"a": 1, "b": 0}
    with pytest.raises(ValueError, match="both classes"):
        core.permuted_labels(["a", "c"], {"a": 0, "c": 0}, fold=0, part=0)


def test_split_and_its_checks() -> None:
    labels, folds = _plan()
    for fold in range(5):
        split = core.masked_split(labels, folds, fold, seed=1)
        train, val = dictionary_split(dict(labels), dict(folds), fold, 1)
        assert split["dictionary_patients"] == list(train) and split["monitor_patients"] == list(val)
        checks = core.check_split(split, labels, folds)
        assert all(checks.values()), checks
        assert split == core.masked_split(labels, folds, fold, seed=1)  # fixed by the fold and the seed
    split = core.masked_split(labels, folds, 0, seed=1)
    held = next(s for s, f in folds.items() if f == 0)
    broken = copy.deepcopy(split)
    broken["dictionary_patients"].append(held)
    broken["permuted_labels"][held] = labels[held]
    assert not core.check_split(broken, labels, folds)["no_held_out_patient"]
    broken = copy.deepcopy(split)
    broken["permuted_labels"] = {p: labels[p] for p in broken["permuted_labels"]}
    assert not core.check_split(broken, labels, folds)["permuted_differ"]
    broken = copy.deepcopy(split)
    flipped = next(p for p in broken["dictionary_patients"] if broken["permuted_labels"][p] == 0)
    broken["permuted_labels"][flipped] = 1
    assert not core.check_split(broken, labels, folds)["permuted_counts_kept"]
    broken = copy.deepcopy(split)
    broken["monitor_patients"].append(broken["dictionary_patients"][0])
    assert not core.check_split(broken, labels, folds)["monitor_apart"]
    broken = copy.deepcopy(split)
    del broken["permuted_labels"][broken["dictionary_patients"][0]]
    assert not core.check_split(broken, labels, folds)["permuted_cover"]


def test_group_counts() -> None:
    code = torch.zeros(2, 5, 12, dtype=torch.float64)  # K = 5 with 3 shared atoms, 3 windows of 4 samples
    code[0, 0, 1], code[0, 2, 5], code[0, 3, 6], code[0, 4, 6] = 2.0, 1.0, 3.0, 4.0
    code[1, 1, 9] = 5.0
    out = core.group_counts(code, n_shared=3, pool=4)
    assert out["wins_shared"].tolist() == [2, 1] and out["wins_epilepsy"].tolist() == [2, 0]
    assert out["energy_shared"].tolist() == [5.0, 25.0] and out["energy_epilepsy"].tolist() == [25.0, 0.0]
    assert out["leak_windows"].tolist() == [1, 0] and out["windows"].tolist() == [3, 3]  # both in window 1
    assert out["atom_wins"].tolist() == [[1, 0, 1, 1, 1], [0, 1, 0, 0, 0]]
    wins = core.share(out["wins_epilepsy"], out["wins_shared"])
    assert wins.tolist() == [0.5, 0.0]
    assert core.share(np.array([0, 3]), np.array([0, 0])).tolist() == [0.0, 1.0]  # an empty segment gives 0


def test_scores() -> None:
    shares = {"sup_s1": np.array([0.2, 0.4, 0.0, 1.0]), "sup_s2": np.array([0.4, 0.0, 0.2, 0.0])}
    mean = core.seed_mean(shares, "sup", (1, 2))
    assert np.allclose(mean, [0.3, 0.2, 0.1, 0.5]) and np.array_equal(
        core.seed_mean(shares, "sup", (1,)), shares["sup_s1"]
    )
    patients, values = core.patient_scores(mean, np.array(["b", "a", "b", "a"]))
    assert patients.tolist() == ["a", "b"] and np.allclose(values, [0.35, 0.2])
    assert (
        core.training_moved([0.9, 0.95, 0.8]) and not core.training_moved([0.9, 0.9]) and not core.training_moved([0.9])
    )


def test_rule() -> None:
    assert core.masked_outcome(False, True, True, True) == "blocked"
    assert core.masked_outcome(True, True, True, True) == "closes_gap"
    assert core.masked_outcome(True, True, False, True) == "closes_gap"
    assert core.masked_outcome(True, True, True, False) == "adds_to_drug"
    assert core.masked_outcome(True, True, False, False) == "separates_only"
    assert core.masked_outcome(True, False, True, False) == "review"
    assert core.masked_outcome(True, False, True, True) == "review"
    assert core.masked_outcome(True, False, False, True) == "no_signal"
    assert core.masked_outcome(True, False, False, False) == "no_signal"
    reached = {
        core.masked_outcome(g, s, d, c)
        for g in (True, False)
        for s in (True, False)
        for d in (True, False)
        for c in (True, False)
    }
    assert reached == set(core.OUTCOMES)
