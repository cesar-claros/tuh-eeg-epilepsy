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


def test_a_patient_with_many_sessions_cannot_decide_the_validity_gate() -> None:
    wrong = [(2000.0 + i, 40.0 + i) for i in range(9)] + [(2009.0, 70.0)]
    sessions = {"heavy": wrong} | {f"p{i}": [(2010.0, 40.0), (2012.0, 42.0)] for i in range(9)}
    by_pairs, pairs = core.age_validity(sessions)
    by_patient, patients = core.patient_age_validity(sessions)
    assert pairs == 54 and abs(by_pairs - 45 / 54) < 1e-12 and by_pairs < core.AGE_VALIDITY_MIN, by_pairs
    assert patients == 10 and abs(by_patient - (36 / 45 + 9) / 10) < 1e-12, by_patient
    assert by_patient >= core.AGE_VALIDITY_MIN, "this cohort is what the changed gate must pass"
    single, count = core.patient_age_validity({"a": [(2010.0, 40.0)]})
    assert math.isnan(single) and count == 0, "a patient without a pair does not count"
    failing, _ = core.patient_age_validity({f"p{i}": [(2010.0, 40.0), (2012.0, 50.0)] for i in range(5)})
    assert failing == 0.0, "errors in every patient must still fail"


def test_the_repaired_age_uses_the_median_birth_year() -> None:
    sessions = {"a": [(2010.0, 40.0), (2011.0, 41.0), (2012.0, 70.0)], "b": [(2010.0, 30.0)], "c": []}
    assert core.median_birth_year(sessions["a"]) == 1970.0, "one wrong session must not move the median"
    assert math.isnan(core.median_birth_year(sessions["c"]))
    subjects = np.array(["a", "b", "c"])
    repaired = core.repaired_ages(subjects, np.array([2012.0, 2010.0, 2010.0]), sessions)
    assert repaired[0] == 42.0 and repaired[1] == 30.0 and math.isnan(repaired[2]), repaired


def test_the_read_age_fit_counts_only_checkable_patients() -> None:
    sessions = {"a": [(2010.0, 40.0), (2012.0, 42.0)], "b": [(2010.0, 30.0)], "c": [(2010.0, 50.0), (2012.0, 52.0)]}
    subjects = np.array(["a", "b", "c"])
    repaired = np.array([41.0, 30.0, 51.0])
    fit, patients = core.read_age_fit(subjects, np.array([46.0, 99.0, 57.0]), repaired, sessions)
    assert patients == 2 and fit == 0.5, "b has one session, a is exactly at the caliper, and c is beyond it"
    empty, none = core.read_age_fit(np.array(["b"]), np.array([30.0]), np.array([30.0]), sessions)
    assert math.isnan(empty) and none == 0


def test_header_gates_apply_their_thresholds() -> None:
    age = np.array([40.0] * 9 + [math.nan])
    sex = np.array(["F"] * 10)
    good = core.header_gates(age, sex, 0.95, 0.95)
    assert good["header_coverage_ok"], "90% is enough"
    assert core.gates_pass(good)
    assert not core.header_gates(np.array([40.0] * 8 + [math.nan] * 2), sex, 0.95, 0.95)["header_coverage_ok"]
    low_validity = core.header_gates(age, sex, 0.5, 0.95)
    assert not low_validity["age_validity_ok"] and not core.gates_pass(low_validity)
    assert not core.header_gates(age, sex, math.nan, 0.95)["age_validity_ok"], "no session pair cannot pass"
    low_fit = core.header_gates(age, sex, 0.95, 0.5)
    assert not low_fit["age_read_fit_ok"] and not core.gates_pass(low_fit)
    assert not core.header_gates(age, sex, 0.95, math.nan)["age_read_fit_ok"], "no checkable patient cannot pass"


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
    assert core.audit_outcome(False, 60, 60, good, good, good, good) == "blocked"
    assert core.audit_outcome(True, 29, 60, good, good, good, good) == "too_few"
    assert core.audit_outcome(True, 60, 29) == "too_few"
    assert core.audit_outcome(True, 60, 60) == "incomplete"
    assert core.audit_outcome(True, 60, 60, good) == "incomplete", "holds needs the repaired read"
    assert core.audit_outcome(True, 60, 60, good, repaired=good) == "holds", "only the primary read decides holds"
    assert core.audit_outcome(True, 60, 60, good, flat, flat, good) == "holds"
    assert core.audit_outcome(True, 60, 60, good, good, good, flat) == "review", "a repaired read that disagrees"
    assert core.audit_outcome(True, 60, 60, flat, flat, flat, good) == "review", "a repaired read that disagrees"
    assert core.audit_outcome(True, 60, 60, flat, good, flat, flat) == "review", "the refit read can only give review"
    assert core.audit_outcome(True, 60, 60, flat, flat, good, flat) == "review", "HYDRA can only give review"
    assert core.audit_outcome(True, 60, 60, flat, flat, flat, flat) == "not_resolved"
    assert core.audit_outcome(True, 60, 60, flat, repaired=flat) == "incomplete", "not_resolved needs the other reads"
    assert core.audit_outcome(True, 30, 30, flat, flat, flat, flat) == "not_resolved", "30 per class is enough"


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


OLDER = {"phenytoin": ["dilantin", "phenytoin"], "valproate": ["depakote"]}
NEWER = {"levetiracetam": ["keppra", "keppr"], "topiramate": ["topamax", "tpm"]}
UNNAMED = ["seizure medication"]


def test_drugs_named_counts_a_drug_once_and_whole_words_only() -> None:
    assert core.drugs_named("Dilantin, phenytoin and depakote", OLDER) == {"phenytoin", "valproate"}
    assert core.drugs_named("dilantine", OLDER) == set(), "a term must match a whole word"
    assert core.names_any("aspirin 81", ["asa", "aspirin"]) and not core.names_any("casa", ["asa"])
    assert not core.names_any("anything", []), "no terms name nothing"


def test_drug_exposure_is_a_mean_per_listing_session() -> None:
    both = core.drug_exposure(["keppra, dilantin", "keppra", ""], OLDER, NEWER, UNNAMED)
    assert both.drugs_per_session == 1.5, "an empty session must not count"
    assert both.older and not both.newer_only and both.named == {"levetiracetam", "phenytoin"}
    many = core.drug_exposure(["keppra"] * 6, OLDER, NEWER, UNNAMED)
    assert many.drugs_per_session == 1.0, "more sessions must not raise the count"
    assert many.newer_only and not many.older
    off = core.drug_exposure(["aspirin"], OLDER, NEWER, UNNAMED)
    assert off.drugs_per_session == 0.0 and not off.older and not off.newer_only
    unnamed = core.drug_exposure(["seizure medication"], OLDER, NEWER, UNNAMED)
    assert unnamed.drugs_per_session == 1.0 and not unnamed.older and not unnamed.newer_only, "it enters neither class"
    assert math.isnan(core.drug_exposure(["", ""], OLDER, NEWER, UNNAMED).drugs_per_session)


def test_group_read_reads_two_groups_and_handles_an_empty_one() -> None:
    fold = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([3.0, 2.0, 2.0, 5.0, 2.0, 1.0])
    first, second = np.array([1, 1, 1, 1, 0, 0], bool), np.array([0, 0, 0, 0, 1, 1], bool)
    read = core.group_read(score, first, second, fold, n_boot=50)
    assert abs(read.value - (2 + 1.5 + 1.5 + 2) / 8) < 1e-12 and (read.first, read.second) == (4, 2), read
    wider = core.group_read(score, first, second, fold, n_boot=400, confidence=0.99)
    narrower = core.group_read(score, first, second, fold, n_boot=400, confidence=0.5)
    assert wider.high - wider.low > narrower.high - narrower.low, "a higher confidence must be wider"
    empty = core.group_read(score, first, np.zeros(6, bool), fold, n_boot=50)
    assert math.isnan(empty.value) and empty.second == 0


def test_rank_read_recovers_a_correlation_and_its_absence() -> None:
    rng = np.random.default_rng(6)
    fold = np.arange(76) % 5
    x = rng.standard_normal(76)
    strong = core.rank_read(x, x + 0.2 * rng.standard_normal(76), fold, n_boot=500)
    assert strong[0] > 0.9 and strong[1] > 0.8, strong
    noise = core.rank_read(x, rng.standard_normal(76), fold, n_boot=500)
    assert noise[1] < noise[0] < noise[2] and noise[1] < 0 < noise[2], noise
    ties = core.rank_read(
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0] * 10), rng.standard_normal(60), np.arange(60) % 5, 200
    )
    assert np.isfinite(ties[1]) and np.isfinite(ties[2]), "ties must not break the interval"


def test_partial_rank_removes_a_shared_cause() -> None:
    rng = np.random.default_rng(8)
    cause = rng.standard_normal(400)
    x, y = cause + 0.3 * rng.standard_normal(400), cause + 0.3 * rng.standard_normal(400)
    assert np.corrcoef(x, y)[0, 1] > 0.8
    assert abs(core.partial_rank(x, y, cause)) < 0.15, "the shared cause must be gone"


def test_drug_rule_gives_each_outcome() -> None:
    up = core.GroupRead(0.8, 0.6, 0.9, 66, 10)
    flat = core.GroupRead(0.55, 0.4, 0.7, 66, 10)
    down = core.GroupRead(0.3, 0.1, 0.45, 35, 30)
    no_count, up_count = (0.1, -0.1, 0.3), (0.4, 0.1, 0.6)
    assert core.drug_outcome(flat, no_count, flat, flat) == "no_measured_tracking"
    assert core.drug_outcome(None, no_count, None, None) == "no_measured_tracking", "small groups cannot pass"
    assert core.drug_outcome(up, no_count, flat, flat) == "tracks"
    assert core.drug_outcome(flat, up_count, flat, flat) == "tracks"
    assert core.drug_outcome(flat, no_count, up, flat) == "tracks"
    assert core.drug_outcome(flat, no_count, down, flat) == "tracks", "two classes can differ in either direction"
    assert core.drug_outcome(down, no_count, flat, flat) == "no_measured_tracking", "less score on drug is not tracking"
    assert core.drug_outcome(flat, (-0.4, -0.6, -0.1), flat, flat) == "no_measured_tracking", "the same for the count"
    assert core.drug_outcome(up, no_count, flat, up) == "nonspecific", "a control that also rises"
    assert core.drug_outcome(up, no_count, flat, None) == "tracks", "a control too small to read"
    assert core.drug_outcome(flat, no_count, flat, up) == "no_measured_tracking", "a control alone decides nothing"
    assert core.drug_outcome(up, None, up, up) == "incomplete"
    assert abs(core.DRUG_FAMILY_CONFIDENCE - (1 - 0.05 / 3)) < 1e-12


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
