"""Checks of the pure logic of the frozen readout, ``src/frozen_readout_core.py`` (numpy, scipy, scikit-learn; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first
failure. Covered: folds, the dictionary split, the segment draw, the low-pass, the patient-relative
threshold, the label-free ``z`` rule, the patient weights, the head with its regularization, scaler
mask and sparse form, AUROC and the paired interval, the conditions and outcomes of the rule, the
rate guard, the bag-size link and the positive-control and training checks.
"""

from __future__ import annotations

import inspect

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import src.frozen_readout_core as core  # noqa: E402

LABELS = {f"p{i:03d}": int(i < 79) for i in range(156)}


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def test_folds_are_stratified_disjoint_and_stable() -> None:
    folds = core.assign_folds(LABELS, seed=1)
    assert set(folds) == set(LABELS) and set(folds.values()) == set(range(core.N_FOLDS))
    for label in (0, 1):
        sizes = [sum(folds[s] == f and LABELS[s] == label for s in LABELS) for f in range(core.N_FOLDS)]
        assert max(sizes) - min(sizes) <= 1, f"class {label} is not stratified: {sizes}"
    assert folds == core.assign_folds(dict(reversed(list(LABELS.items()))), seed=1), "the input order changed the folds"
    assert folds != core.assign_folds(LABELS, seed=2), "a new pass must give new folds"


def test_dictionary_split_excludes_the_fold_and_is_stratified() -> None:
    folds = core.assign_folds(LABELS, seed=1)
    train, val = core.dictionary_split(LABELS, folds, 2, seed=1)
    outside = [s for s in LABELS if folds[s] != 2]
    assert sorted(train + val) == sorted(outside) and not set(train) & set(val), "the split must partition the rest"
    assert all(folds[s] != 2 for s in train + val), "a held-out patient entered the dictionary"
    assert abs(len(val) / len(outside) - core.DICTIONARY_VAL_SHARE) < 0.02
    assert abs(sum(LABELS[s] for s in val) - len(val) / 2) <= 2, "the validation group is not stratified"


def test_segment_draw_is_stable_and_free_of_input_order() -> None:
    keys = [("a.edf", float(t)) for t in range(40)]
    first = core.segment_draw("p1", keys, 9)
    assert len(set(first)) == 9 and first == sorted(first) and first == core.segment_draw("p1", keys, 9)
    assert first != core.segment_draw("p2", keys, 9), "the draw must depend on the patient"
    reordered = list(reversed(keys))
    assert {keys[i] for i in first} == {reordered[i] for i in core.segment_draw("p1", reordered, 9)}
    _expect_value_error(lambda: core.segment_draw("p1", keys[:8], 9), "a small bag was accepted")


def test_low_pass_removes_high_frequencies_only() -> None:
    sfreq = 256.0
    t = np.arange(30720) / sfreq
    low, high = np.sin(2 * np.pi * 20.0 * t), np.sin(2 * np.pi * 115.0 * t)
    x = np.stack([low + high, np.zeros_like(t)]).astype(np.float32)
    out = core.low_pass(x, sfreq)
    assert out.shape == x.shape and out.dtype == np.float32 and not out[1].any(), "a flat row must stay flat"
    middle = slice(1024, -1024)
    assert np.sqrt(np.mean((out[0][middle] - low[middle]) ** 2)) < 0.05, "the 115 Hz tone was not removed"


def _table(height: np.ndarray, template: np.ndarray, segment: np.ndarray, n_segments: int = 2, n_templates: int = 2):
    return core.PeakTable(segment, template, height, n_segments, n_templates)


def test_threshold_is_relative_to_the_patient() -> None:
    rng = np.random.default_rng(0)
    height = np.r_[rng.normal(10.0, 1.0, 200), 40.0, 45.0]
    template = np.zeros(len(height), dtype=np.int64)
    segment = np.r_[np.zeros(100, dtype=np.int64), np.ones(100, dtype=np.int64), 0, 1]
    table = _table(height, template, segment)
    counts = core.peak_counts(table, 6.0)
    assert counts.shape == (2, 2) and counts[:, 0].tolist() == [1.0, 1.0], "only the two outliers must count"
    assert not counts[:, 1].any(), "a template without peaks must count nothing"
    louder = _table(height * 7.0, template, segment)
    assert np.array_equal(core.peak_counts(louder, 6.0), counts), "the count depends on the patient's level"
    flat = _table(np.ones(5), np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64))
    assert not core.peak_counts(flat, 6.0).any(), "a template with a = 0 must count nothing"
    features = core.rate_features(table, 6.0, minutes_per_segment=2.0)
    assert np.allclose(features[:, 0], np.log1p(0.5))


def test_z_rule_uses_no_labels_and_finds_the_target() -> None:
    assert "label" not in " ".join(inspect.signature(core.choose_z).parameters), "the z rule must not take labels"
    rng = np.random.default_rng(1)
    tables = []
    for _ in range(4):
        height = rng.standard_exponential(4000) + 1.0
        tables.append(_table(height, rng.integers(0, 2, 4000), rng.integers(0, 2, 4000)))
    z, rates = core.choose_z(tables, minutes_per_segment=2.0)
    values = [rates[g] for g in core.Z_GRID]
    assert values == sorted(values, reverse=True), "the rate must fall as z rises"
    assert z == min(core.Z_GRID, key=lambda g: abs(rates[g] - core.TARGET_RATE))


def test_patient_weights_give_every_patient_the_same_total() -> None:
    subjects = np.array(["a"] * 100 + ["b"] * 9 + ["c"])
    weights = core.patient_weights(subjects)
    totals = [weights[subjects == s].sum() for s in "abc"]
    assert np.allclose(totals, totals[0]) and abs(weights.mean() - 1.0) < 1e-12


def _rows(n_patients: int = 40, windows: int = 9, signal: float = 2.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    subjects = np.repeat([f"s{i:02d}" for i in range(n_patients)], windows)
    labels = np.repeat(np.arange(n_patients) % 2, windows)
    x = rng.normal(size=(len(subjects), 12)).astype(np.float32)
    x[:, 0] += signal * labels
    return x, labels, subjects


def test_head_finds_signal_and_keeps_patients_apart() -> None:
    x, labels, subjects = _rows()
    held = np.isin(subjects, np.unique(subjects)[:8])
    result = core.fit_head(x, labels, subjects, held, StandardScaler)
    assert len(result.subject) == 8 and result.n_windows.tolist() == [9] * 8, "the score must be per patient"
    assert core.auroc(result.label, result.score) >= 0.9 and result.c in core.C_GRID
    leak = held.copy()
    leak[np.flatnonzero(held)[0]] = False
    _expect_value_error(lambda: core.fit_head(x, labels, subjects, leak, StandardScaler), "a split patient passed")


def test_regularization_ties_go_to_the_smaller_value() -> None:
    x, labels, subjects = _rows(signal=20.0)
    best, means = core.select_c(x, labels, subjects, StandardScaler)
    assert len(set(means.values())) == 1 and best == min(core.C_GRID), f"a tie must pick the smallest C: {means}"


def test_weighted_head_resists_a_dominant_patient() -> None:
    x, labels, subjects = _rows(n_patients=20, windows=3, signal=2.0, seed=2)
    extra = np.zeros((600, x.shape[1]), dtype=np.float32)
    extra[:, 0] = -3.0  # a positive patient with many windows, against the direction of all others
    all_x = np.concatenate([x, extra])
    all_y = np.r_[labels, np.ones(600, dtype=labels.dtype)]
    all_s = np.r_[subjects, np.array(["dominant"] * 600)]
    probes = np.isin(all_s, np.unique(subjects)[:6])
    plain = core.fit_head(all_x, all_y, all_s, probes, StandardScaler, weighted=False)
    balanced = core.fit_head(all_x, all_y, all_s, probes, StandardScaler, weighted=True)
    assert plain.coef[0] < balanced.coef[0], "the weights must reduce the pull of the dominant patient"
    assert balanced.coef[0] > 0, "the weighted head must keep the direction of most patients"


def test_scaler_mask_selects_the_rows_that_fit_the_scaler() -> None:
    seen = []

    class Recording(StandardScaler):
        def fit(self, x, y=None, sample_weight=None):
            seen.append(len(x))
            return super().fit(x, y, sample_weight)

    x, labels, subjects = _rows()
    held = np.isin(subjects, np.unique(subjects)[:8])
    mask = np.zeros(len(x), dtype=bool)
    mask[::3] = True
    core.fit_head(x, labels, subjects, held, Recording, scaler_mask=mask)
    assert seen[-1] == int(mask[~held].sum()), "the final scaler did not use the masked rows"


def test_sparse_head_keeps_to_its_limit() -> None:
    x, labels, subjects = _rows()
    held = np.isin(subjects, np.unique(subjects)[:8])
    result, kept = core.fit_sparse_head(x, labels, subjects, held, StandardScaler, max_templates=3)
    assert 0 < len(kept) <= 3 and 0 in kept.tolist(), "the informative feature must be kept"
    assert core.auroc(result.label, result.score) >= 0.8


def _arm(score: np.ndarray, fold: np.ndarray, label: np.ndarray) -> core.ArmScores:
    return core.ArmScores.from_rows(np.array([f"s{i:03d}" for i in range(len(score))]), fold, label, score)


def test_auroc_and_paired_interval() -> None:
    assert core.auroc(np.array([1, 1, 0, 0]), np.array([2.0, 1.0, 1.0, 0.0])) == 0.875, "ties count one half"
    assert np.isnan(core.auroc(np.zeros(3), np.zeros(3))), "an empty class has no AUROC"
    rng = np.random.default_rng(3)
    label, fold = np.tile([0, 1], 75), np.repeat(np.arange(5), 30)
    good = _arm(label + rng.normal(0.0, 0.5, 150), fold, label)
    noise = _arm(rng.normal(size=150), fold, label)
    assert core.mean_fold_auroc(good) > 0.85 and 0.3 < core.mean_fold_auroc(noise) < 0.7
    assert core.paired_interval(good, good, n_boot=200) == (0.0, 0.0, 0.0), "an arm must not differ from itself"
    difference, lower, upper = core.paired_interval(good, noise, n_boot=500)
    assert lower > 0 and lower < difference < upper
    assert abs(difference - (core.mean_fold_auroc(good) - core.mean_fold_auroc(noise))) < 1e-12
    other = _arm(noise.score, np.roll(fold, 1), label)
    _expect_value_error(lambda: core.paired_interval(good, other, n_boot=10), "misaligned folds were accepted")


def test_conditions_and_outcomes_of_the_rule() -> None:
    good = {"Spectral": (0.1, 0.01, 0.2), "S4-untrained": (0.1, 0.02, 0.2), "HYDRA": (-0.1, -0.14, 0.0)}
    passed = core.conditions(good)
    assert all(passed.values()), "this interval set passes every condition"
    assert core.conditions({**good, "HYDRA": (-0.1, -0.16, 0.0)})["HYDRA"] is False, "the HYDRA margin was ignored"
    assert core.conditions({**good, "Spectral": (0.1, -0.01, 0.2)})["Spectral"] is False, "a tie with Spectral passed"
    missing = core.conditions({k: v for k, v in good.items() if k != "HYDRA"})
    assert missing["HYDRA"] is None
    cases = (
        (passed, 1, True, None, "confirm"),
        (passed, 2, True, True, "review"),
        (passed, 2, True, False, "stop"),
        (passed, 2, True, None, "incomplete"),
        (passed, 1, False, None, "blocked"),
        (passed, 1, None, None, "incomplete"),
        (missing, 1, True, None, "incomplete"),
        ({**missing, "Spectral": False}, 1, True, None, "stop"),
    )
    for values, pass_index, gates, guard, expected in cases:
        assert core.readout_outcome(values, pass_index, gates, guard) == expected, f"{pass_index} {gates} {guard}"


def test_rate_guard_counts_only_large_groups() -> None:
    rng = np.random.default_rng(4)
    label = np.tile([0, 1], 40)
    worse = rng.normal(size=80)
    better = label + rng.normal(0.0, 0.3, 80)
    fold = np.zeros(80, dtype=int)
    first, second = _arm(better, fold, label), _arm(worse, fold, label)
    rates = dict.fromkeys(first.subject.tolist(), 256.0)
    ok, report = core.rate_guard(first, second, rates)
    assert ok and report["256.0"]["counts"] and report["256.0"]["difference"] > 0
    ok, _ = core.rate_guard(second, first, rates)
    assert not ok, "a wrong sign in a large group must block"
    small = dict(rates)
    for subject in first.subject[:4].tolist():
        small[subject] = 250.0
    _, report = core.rate_guard(first, second, small)
    assert not report["250.0"]["counts"], "a small group must not count"


def test_bag_size_link_and_small_checks() -> None:
    label = np.tile([0, 1], 20)
    score = np.arange(40, dtype=float)
    arm = _arm(score, np.zeros(40, dtype=int), label)
    links = core.bag_size_link(arm, {s: 2 ** (i + 1) for i, s in enumerate(arm.subject.tolist())})
    assert abs(links["epilepsy"] - 1.0) < 1e-9 and abs(links["non_epilepsy"] - 1.0) < 1e-9
    assert np.isnan(core.bag_size_link(arm, dict.fromkeys(arm.subject.tolist(), 9))["epilepsy"]), "no bag variation"
    assert core.planted_fraction(np.array([2.0, 2.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5
    assert core.training_passed([0.7, 0.8], [0.9, 0.9]) and not core.training_passed([0.7, 0.95], [0.9, 0.9])


def test_permuted_scores_cover_the_held_out_patients() -> None:
    x, labels, subjects = _rows()
    held = np.isin(subjects, np.unique(subjects)[:8])
    patients, scores = core.permuted_scores(x, labels, subjects, held, 1.0, StandardScaler, np.random.default_rng(0))
    assert patients.tolist() == np.unique(subjects)[:8].tolist() and scores.shape == (8,)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
