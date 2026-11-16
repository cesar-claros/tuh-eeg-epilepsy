"""Checks of the pure logic of the class-masked dictionary, ``src/masked_dictionary_core.py`` (no corpus; CPU).

Runs under pytest. Covered: the dictionary names, the permuted labels (class counts kept, never the true labels,
fixed by the fold), the split of a fold and the checks of gate 2 on good and broken splits, the group counts of a
known code, the shares and their seed mean, the patient scores, the training check, and the rule with its outcomes.
For the hinge redesign: the names of the pilot and of stage 2, the reads of the pilot on known counts, the first-fit
gate, the pilot rule with its bounds, and the outcome rule with ``collapsed``.
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


# ---------------------------------------------------------------------- hinge redesign


def test_redesign_names() -> None:
    names = core.pilot_names()
    assert names[0] == "sup_e1_s1"  # the dictionary of the first-fit gate
    assert names == [f"{arm}_e{eta}_s1" for eta in (1, 10, 100, 1000) for arm in ("sup", "perm")]
    assert core.pilot_name("perm", 1000.0) == "perm_e1000_s1" and core.eta_label(10.0) == "10"
    for arm, eta in (("mask", 1.0), ("sup", 5.0)):
        with pytest.raises(ValueError):
            core.pilot_name(arm, eta)
    assert core.stage2_names() == ["sup2_s1", "sup2_s2", "perm2_s1", "perm2_s2"]
    assert core.parse_stage2("perm2_s2") == ("perm2", 2)
    for bad in ("sup_s1", "sup2_s3", "mask2_s1"):
        with pytest.raises(ValueError):
            core.parse_stage2(bad)
    assert not set(core.stage2_names()) & set(core.dictionary_names())  # no name of the first run is reused
    assert (core.WARM_EPOCHS, core.RAMP_EPOCHS, core.MIN_USED, core.LEAK_FACTOR) == (10, 5, 8, 0.5)
    assert (core.PILOT_FOLD, core.PILOT_SEED, core.REFERENCE) == (0, 1, "mask_s1")
    assert {arm for arm, _ in core.SCORES2.values()} == set(core.SEEDS2)


def test_pilot_reads() -> None:
    wins = np.zeros((4, 80), dtype=np.int64)
    wins[:, :64] = 5
    wins[0, 64], wins[1, 65], wins[1, 66] = 3, 1, 2  # group 1 (segments 0 and 1): three atoms in use
    wins[2, 70] = 9  # group 0: an atom that wins only there does not count as in use
    assert core.used_atoms(wins) == 4 and core.used_atoms(wins[:2]) == 3
    reads = core.pilot_reads(wins, np.array([9, 9, 4, 0]), np.array([10, 10, 10, 10]), np.array([1, 1, 0, 0]))
    assert reads == {"use": 3, "leak": 0.2}  # 4 of the 20 windows of group 0
    with pytest.raises(ValueError, match="both groups"):
        core.pilot_reads(wins, np.zeros(4), np.full(4, 10), np.ones(4, dtype=int))


def test_first_fit_check() -> None:
    assert core.first_fit_check(0.1, [1.0] * 8 + [0.0] * 8) == {"leaks": True, "atoms_in_use": True}
    assert core.first_fit_check(0.0, [1.0] * 16) == {"leaks": False, "atoms_in_use": True}
    assert core.first_fit_check(0.1, [1.0] * 7 + [0.0] * 9) == {"leaks": True, "atoms_in_use": False}


def test_pilot_rule() -> None:
    reads = {
        1.0: {"use": 16, "leak": 0.15},  # in use, but the leak is above half of the reference's
        10.0: {"use": 12, "leak": 0.10},  # works, at the bound of the leak
        100.0: {"use": 8, "leak": 0.02},  # works, at the bound of the use
        1000.0: {"use": 7, "leak": 0.0},  # the atoms are lost
    }
    rule = core.pilot_rule(reads, reference_leak=0.2)
    assert rule["works"] == {1.0: False, 10.0: True, 100.0: True, 1000.0: False}
    assert rule["outcome"] == "go" and rule["eta_star"] == 100.0  # the largest strength that works
    none = core.pilot_rule({1.0: reads[1.0], 1000.0: reads[1000.0]}, reference_leak=0.2)
    assert none == {"works": {1.0: False, 1000.0: False}, "outcome": "no_working_hinge", "eta_star": None}
    assert core.pilot_rule({10.0: {"use": 12, "leak": 0.1001}}, reference_leak=0.2)["outcome"] == "no_working_hinge"


def test_redesign_outcome() -> None:
    assert core.redesign_outcome(False, False, True, True, True) == "blocked"  # a gate of the first protocol
    assert core.redesign_outcome(True, False, True, True, True) == "collapsed"  # gate 8: no score counts
    for separates, drug, gap in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
        expected = core.masked_outcome(True, separates, drug, gap)
        assert core.redesign_outcome(True, True, separates, drug, gap) == expected
    assert set(core.OUTCOMES2) == {*core.OUTCOMES, "collapsed"}
