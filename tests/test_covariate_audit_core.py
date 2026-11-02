"""Checks of the pure logic of the covariate audit, ``src/covariate_audit_core.py`` (numpy, scikit-learn; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first
failure. Covered: session paths, patient covariates, the age validity gate, the pair mask and its ties,
the bootstrap weights, a score of age plus noise near chance, recovery of a known AUROC, the controls,
residuals and second-level models without leakage, the outcomes of the rule, the medication status and
limit, and the inventory, which must drop other patients and hold no label and no value of age or sex.
"""

from __future__ import annotations

import inspect
import json
import math

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402

import src.covariate_audit_core as core  # noqa: E402
from src.frozen_readout_core import auroc  # noqa: E402


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _cohort(seed: int = 0, age_shift: float = 0.0, n_pos: int = 79, n_neg: int = 77) -> tuple[np.ndarray, ...]:
    """Labels, stratified folds, ages (mean 45, sd 20, epilepsy shifted by ``age_shift`` years) and sexes."""
    rng = np.random.default_rng(seed)
    label = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    fold = np.r_[np.arange(n_pos) % 5, np.arange(n_neg) % 5]
    age = 45.0 + 20.0 * rng.standard_normal(len(label)) + age_shift * label
    sex = rng.choice(np.array(["F", "M"]), len(label))
    return label, fold, age, sex


def test_session_of_reads_the_directory_and_refuses_others() -> None:
    assert core.session_of("/d/00_epilepsy/aaaaaanr/s012_2003/02_tcp_le/aaaaaanr_s012_t001.edf") == ("s012", 2003)
    _expect_value_error(lambda: core.session_of("/d/aaaaaanr/aaaaaanr_s012_t001.edf"), "a path without a session")


def test_patient_covariates_handle_missing_values_and_conflicts() -> None:
    assert math.isnan(core.clean_age(-1)) and math.isnan(core.clean_age(200)) and core.clean_age(0) == 0.0
    assert core.clean_sex("X") == core.MISSING_SEX and core.clean_sex("F") == "F"
    assert core.patient_age([40.0, math.nan, 42.0]) == 41.0, "the mean must skip a missing age"
    assert math.isnan(core.patient_age([math.nan, math.nan]))
    assert core.patient_sex(["M", "M"]) == "M"
    assert core.patient_sex(["M", "F"]) == core.MISSING_SEX, "a conflict counts as missing"
    assert core.patient_sex(["M", core.MISSING_SEX]) == core.MISSING_SEX


def test_age_validity_compares_age_and_year_differences() -> None:
    share, pairs = core.age_validity({"a": [(2010.0, 40.0), (2012.0, 42.0), (2013.0, 50.0)], "b": [(2010.0, 30.0)]})
    assert pairs == 3 and abs(share - 1 / 3) < 1e-12, "only the first pair matches; a one-session patient has no pair"
    share, pairs = core.age_validity({"a": [(2010.0, 40.0), (2012.0, 43.0)]})
    assert pairs == 1 and share == 1.0, "a difference of one year is within tolerance"
    share, pairs = core.age_validity({"a": [(2010.0, 40.0), (2012.0, math.nan)]})
    assert pairs == 0 and math.isnan(share), "a missing age makes no pair"


def test_header_gates_apply_their_thresholds() -> None:
    age = np.array([40.0] * 9 + [math.nan])
    sex = np.array(["F"] * 10)
    assert core.header_gates(age, sex, 0.95)["header_coverage_ok"], "90% is enough"
    assert not core.header_gates(np.array([40.0] * 8 + [math.nan] * 2), sex, 0.95)["header_coverage_ok"]
    assert not core.header_gates(age, sex, 0.5)["age_validity_ok"]
    assert not core.header_gates(age, sex, math.nan)["age_validity_ok"], "no session pair cannot pass"


def test_pair_mask_needs_same_sex_and_close_age() -> None:
    label = np.array([1, 1, 1, 0, 0])
    age = np.array([40.0, 40.0, math.nan, 44.0, 46.0])
    sex = np.array(["F", "M", "F", "F", "F"])
    mask = core.pair_mask(label, age, sex)
    assert mask.tolist() == [[True, False], [False, False], [False, False]], mask.tolist()
    assert core.pair_mask(label, age, sex, caliper=6.0).tolist()[0] == [True, True], "the caliper is inclusive"
    unknown = np.array(["", "", "", "", ""])
    assert not core.pair_mask(label, age, unknown).any(), "a missing sex has no pair"


def test_matched_read_counts_ties_at_one_half_and_only_masked_pairs() -> None:
    label = np.array([1, 1, 0, 0])
    fold = np.array([0, 0, 0, 0])
    score = np.array([1.0, 2.0, 1.0, 3.0])
    mask = np.array([[True, False], [True, True]])
    read = core.matched_read(score, label, fold, mask, standardize=False, n_boot=20)
    assert abs(read.value - (0.5 + 1.0 + 0.0) / 3) < 1e-12, read
    assert (read.pairs, read.patients_epilepsy, read.patients_non_epilepsy) == (3, 2, 2)
    empty = core.matched_read(score, label, fold, np.zeros((2, 2), bool), n_boot=20)
    assert math.isnan(empty.value) and math.isnan(empty.low) and empty.pairs == 0


def test_bootstrap_weights_match_explicit_resampling() -> None:
    rng = np.random.default_rng(3)
    fold_pos, fold_neg = np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 1, 1, 1])
    values = rng.random((6, 5))
    mask = rng.random((6, 5)) < 0.5
    n_boot, seed = 400, 11
    low, high = core.pair_bootstrap(values, mask, fold_pos, fold_neg, n_boot=n_boot, seed=seed)
    draw = np.random.default_rng(seed)
    pos = core._draw_counts(fold_pos, n_boot, draw)
    neg = core._draw_counts(fold_neg, n_boot, draw)
    assert (pos[:, fold_pos == 0].sum(axis=1) == 3).all() and (neg[:, fold_neg == 1].sum(axis=1) == 3).all()
    explicit = []
    for b in range(n_boot):
        total, summed = 0.0, 0.0
        for i in range(6):
            for j in range(5):
                if mask[i, j]:
                    total += pos[b, i] * neg[b, j]
                    summed += pos[b, i] * neg[b, j] * values[i, j]
        if total > 0:
            explicit.append(summed / total)
    assert abs(np.percentile(explicit, 2.5) - low) < 1e-9 and abs(np.percentile(explicit, 97.5) - high) < 1e-9
    same = core.pair_bootstrap(np.full((6, 5), 0.7), mask, fold_pos, fold_neg, n_boot=50)
    assert abs(same[0] - 0.7) < 1e-9 and abs(same[1] - 0.7) < 1e-9, "a constant must have no width"


def test_standardize_within_fold_ignores_missing_scores() -> None:
    fold = np.array([0, 0, 0, 1, 1, 1])
    out = core.standardize_within_fold(np.array([1.0, 2.0, math.nan, 10.0, 20.0, 30.0]), fold)
    assert math.isnan(out[2]) and abs(np.nanmean(out[:3])) < 1e-12 and abs(out[3:].mean()) < 1e-12
    assert (core.standardize_within_fold(np.ones(6), fold) == 0).all(), "a flat fold must stay finite"


def test_matching_closes_the_age_route() -> None:
    label, fold, age, sex = _cohort(age_shift=-20.0)
    rng = np.random.default_rng(1)
    score = age / 20.0 + 0.3 * rng.standard_normal(len(label))
    unmatched = auroc(label, score)
    mask = core.pair_mask(label, age, sex)
    matched = core.matched_read(score, label, fold, mask, n_boot=500)
    assert unmatched < 0.3, f"this score must separate the classes by age, not {unmatched}"
    assert abs(matched.value - 0.5) < 0.1, f"matching must close the age route, not {matched.value}"
    assert matched.low < 0.5 < matched.high


def test_a_known_auroc_is_recovered_on_matched_pairs() -> None:
    values = []
    for seed in range(20):
        label, fold, age, sex = _cohort(seed=seed, age_shift=-10.0)
        rng = np.random.default_rng(seed + 100)
        score = 2**0.5 * 0.5244 * label + rng.standard_normal(len(label))  # AUROC 0.70
        values.append(core.matched_read(score, label, fold, core.pair_mask(label, age, sex), n_boot=2).value)
    assert abs(np.mean(values) - 0.70) < 0.02, np.mean(values)


def test_controls_pass_on_a_good_cohort_and_fail_on_a_broken_scale() -> None:
    label, fold, age, sex = _cohort(age_shift=-10.0)
    report = core.control_reads(label, fold, age, sex, n_draws=40)
    assert report["passed"], report
    assert report["age_plus_noise_unmatched"] < 0.5, "epilepsy patients are younger here, so this must lie below 0.5"
    assert abs(report["known_matched"] - core.CONTROL_AUROC) < core.CONTROL_RECOVERY
    no_pairs = core.control_reads(label, fold, age, np.full(len(label), core.MISSING_SEX), n_draws=2)
    assert not no_pairs["passed"], "no pair at all cannot pass"


def test_residuals_remove_the_covariates_and_use_training_rows_only() -> None:
    label, fold, age, sex = _cohort()
    rng = np.random.default_rng(2)
    x = np.column_stack([0.1 * age + rng.standard_normal(len(age)), (sex == "M") + rng.standard_normal(len(age))])
    train = fold != 0
    out = core.residualize(x, age, sex, train)
    assert abs(np.corrcoef(out[train, 0], age[train])[0, 1]) < 1e-6, "the age route must be gone in training"
    assert abs(np.corrcoef(out[train, 1], sex[train] == "M")[0, 1]) < 1e-6, "the sex route must be gone in training"
    changed = x.copy()
    changed[~train] += 1000.0
    moved = core.residualize(changed, age, sex, train)
    assert np.allclose(moved[train], out[train]), "held-out rows must not move the coefficients"
    assert np.allclose(moved[~train] - out[~train], 1000.0), "held-out rows take the training coefficients"


def test_second_level_models_show_added_value_and_no_gain_from_noise() -> None:
    label, fold, age, sex = _cohort(age_shift=-14.0)
    rng = np.random.default_rng(4)
    m0 = core.second_level_scores(label, fold, age, sex)
    noise = core.standardize_within_fold(rng.standard_normal(len(label)), fold)
    with_noise = core.second_level_scores(label, fold, age, sex, noise)
    useful = core.standardize_within_fold(1.5 * label + rng.standard_normal(len(label)), fold)
    with_signal = core.second_level_scores(label, fold, age, sex, useful)
    base = auroc(label, m0)
    assert base > 0.6, f"age must carry class signal here, not {base}"
    assert auroc(label, with_noise) < base + 0.05, "noise must not add"
    assert auroc(label, with_signal) > base + 0.1, "a useful score must add"


def _read(low: float, high: float = 0.9) -> core.MatchedRead:
    return core.MatchedRead((low + high) / 2, low, high, 100, 60, 60)


def test_rule_gives_each_outcome() -> None:
    good, flat = _read(0.55), _read(0.45)
    assert core.audit_outcome(False, 60, 60, good, good, good) == "blocked"
    assert core.audit_outcome(True, 29, 60, good, good, good) == "too_few"
    assert core.audit_outcome(True, 60, 29) == "too_few"
    assert core.audit_outcome(True, 60, 60) == "incomplete"
    assert core.audit_outcome(True, 60, 60, good) == "holds", "only the primary read decides holds"
    assert core.audit_outcome(True, 60, 60, good, flat, flat) == "holds"
    assert core.audit_outcome(True, 60, 60, flat, good, flat) == "review", "the refit read can only give review"
    assert core.audit_outcome(True, 60, 60, flat, flat, good) == "review", "HYDRA can only give review"
    assert core.audit_outcome(True, 60, 60, flat, flat, flat) == "not_resolved"
    assert core.audit_outcome(True, 60, 60, flat) == "incomplete", "not_resolved needs the other two reads"
    assert core.audit_outcome(True, 30, 30, flat, flat, flat) == "not_resolved", "30 per class is enough"


def test_the_exclusion_needs_an_upper_bound_below_its_limit() -> None:
    assert core.excludes_above_bound(_read(0.4, 0.64)) and not core.excludes_above_bound(_read(0.4, 0.65))


def test_medication_status_follows_the_addendum() -> None:
    terms = ["keppra", "valproate"]
    assert core.medication_status(["aspirin", "Keppra 500 mg"], terms, "unknown") == "on", "any session counts"
    assert core.medication_status(["aspirin", "tylenol"], terms, "unknown") == "off"
    assert core.medication_status(["aspirin", ""], terms, "unknown") == "unknown"
    assert core.medication_status(["aspirin", ""], terms, "off") == "off"
    assert core.medication_status([], terms, "off") == "unknown", "no session cannot be off"
    assert core.medication_status(["keppras"], terms, "unknown") == "off", "a term must match a whole word"
    _expect_value_error(lambda: core.medication_status(["x"], [], "off"), "no terms")
    _expect_value_error(lambda: core.medication_status(["x"], terms, "maybe"), "an unknown meaning")


def test_medication_read_takes_the_first_eligible_group() -> None:
    label = np.r_[np.ones(40, int), np.zeros(40, int)]
    fold = np.arange(80) % 5
    rng = np.random.default_rng(5)
    score = 3.0 * label + rng.standard_normal(80)
    mask = core.all_pairs(label)
    both = np.array(["off"] * 15 + ["on"] * 25 + ["off"] * 25 + ["on"] * 15)
    limit = core.medication_read(score, label, fold, mask, both, n_boot=200)
    assert limit["result"] == "survives" and limit["group"] == "off", limit
    only_on = np.array(["unknown"] * 15 + ["on"] * 25 + ["off"] * 25 + ["on"] * 15)
    assert core.medication_read(score, label, fold, mask, only_on, n_boot=200)["group"] == "on"
    none = np.array(["on"] * 40 + ["off"] * 40)
    assert core.medication_read(score, label, fold, mask, none, n_boot=20)["result"] == "inseparable"
    flat = core.medication_read(rng.standard_normal(80), label, fold, mask, both, n_boot=200)
    assert flat["result"] == "unresolved", flat


def test_is_flagged_reads_only_yes() -> None:
    assert core.is_flagged("Y") and core.is_flagged(" y ")
    assert not core.is_flagged("N") and not core.is_flagged(float("nan")) and not core.is_flagged(None)


def test_tokens_read_the_forms_a_sheet_may_use() -> None:
    subjects = {"aaaaaanr", "aaaaaebo"}
    for cell in ("aaaaaanr", "AAAAAANR", "aaaaaanr_s001_t000", "/x/aaaaaanr/y"):
        assert core.subject_token(cell, subjects) == "aaaaaanr", cell
    assert core.subject_token("epilepsy seizures", subjects) is None, "other eight-letter words are not patients"
    assert core.subject_token(float("nan"), subjects) is None
    for cell in ("s001", "S001", "s001_2003", "aaaaaanr_s001_t000", 1, 1.0, "1"):
        assert core.session_token(cell) == "s001", cell
    for cell in ("sedation", 1.5, 2003.5, None, ""):
        assert core.session_token(cell) is None, cell


SHEET = {
    "Subject": ["aaaaaanr", "aaaaaanr", "aaaaaebo", "lockedpt"],
    "Session": ["s001", "s002", "s001", "s001"],
    "Age at EEG": [31, 32, 77, 55],
    "Sex": ["F", "F", "M", "M"],
    "Epilepsy": [1, 1, 0, 0],
    "Medications": ["Keppra 500 mg", "Keppra", None, "secretdrug"],
    "History": ["45 yo with spells", "spells", "male with syncope", "secretword"],
}
SUBJECTS = {"aaaaaanr", "aaaaaebo"}
READ = {("aaaaaanr", "s001"), ("aaaaaebo", "s001")}


def test_inventory_finds_the_join_and_drops_other_patients() -> None:
    report = core.inventory_sheet("Sheet1", SHEET, SUBJECTS, READ)
    assert report["subject_column"] == "Subject" and report["session_column"] == "Session"
    assert report["rows_kept"] == 3 and report["patients_kept"] == 2 and report["read_sessions_joined"] == 1.0
    dumped = json.dumps(report)
    assert "lockedpt" not in dumped and "secretdrug" not in dumped and "secretword" not in dumped


def test_inventory_holds_no_label_and_no_value_of_age_or_sex() -> None:
    report = core.inventory_sheet("Sheet1", SHEET, SUBJECTS, READ)
    columns = {c["name"]: c for c in report["columns"]}
    assert "values" not in columns["Age at EEG"] and "tokens" not in columns["Age at EEG"], "a number lists nothing"
    assert "values" not in columns["Sex"] and "tokens" not in columns["Sex"], "a column named for sex lists nothing"
    assert "values" not in columns["Epilepsy"], "a number lists nothing"
    assert "values" not in columns["Subject"] and "values" not in columns["Session"], "the join lists nothing"
    assert "keppra" in columns["Medications"]["tokens"], "the mapping needs the drug names"
    assert all(not any(ch.isdigit() for ch in text) for text in columns["History"]["values"]), "text with a digit"
    assert all(not any(ch.isdigit() for ch in text) for text in columns["History"]["tokens"])
    assert "male" not in columns["History"]["tokens"], "a word for sex must not be listed"
    assert not any("male" in text for text in columns["History"]["values"]), "a word for sex must not be listed"
    assert "spells" in columns["History"]["values"] and "syncope" in columns["History"]["tokens"]
    assert "dosage" not in core.BLOCKED_NAME_TOKENS and not core._name_is_blocked("Dosage")
    assert core._name_is_blocked("patient_age") and core._name_is_blocked("Gender")


def test_inventory_without_a_subject_column_lists_only_the_names() -> None:
    sheet = {"Summary": ["all", "patients"], "Count": [200, 698]}
    report = core.inventory_sheet("Notes", sheet, SUBJECTS, READ)
    assert report == {"sheet": "Notes", "subject_column": None, "session_column": None, "column_names": list(sheet)}
    _expect_value_error(lambda: core.inventory_sheet("x", sheet, SUBJECTS, READ, subject_column="nope"), "no column")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(test):
            test()
    print("all covariate audit core checks passed")
