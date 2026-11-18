"""Checks of the pure logic of the within-patient drug protocol, ``src/drug_signature_core.py`` (no corpus; CPU).

Runs under pytest. Covered: the contrasts and the deciding tests; the side of a session for every kind of drug list;
"added" and "removed"; a consistent report; a patient with both sides and the floor of 4 segments; the control
halves and the control patients; the band-power layout and the four measures; the alpha frequency on spectra with a
known peak (gate 5 at the level of the spectrum); the patient difference with equal weight for sessions; the test
against no change and the test against time, with and without a planted effect; the strata and the stratified test
against time of amendment A1, on a planted time confound; and every outcome of the rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.drug_signature_core as core  # noqa: E402


def test_constants() -> None:
    assert core.CONTRASTS == {"W-sodium": "sodium", "W-gaba": "gaba", "W-any": "any", "W-lev": "sv2a"}
    assert core.DECIDING == {"W-sodium": ("M-alpha-frequency", -1), "W-gaba": ("M-beta", 1), "W-any": ("M-level", 1)}
    assert "W-lev" not in core.DECIDING and set(core.DECIDING) < set(core.CONTRASTS)  # levetiracetam decides nothing
    assert (core.MIN_SEGMENTS, core.MIN_PATIENTS, core.MIN_TIME_GROUP, core.N_FLIPS, core.ALPHA) == (
        4,
        15,
        10,
        10_000,
        0.05,
    )
    assert {m for m, _ in core.DECIDING.values()} <= set(core.MEASURES)
    assert core.drug_groups({"a": ["x", "y"], "b": ["z"]}) == {"x": "a", "y": "a", "z": "b"}
    with pytest.raises(ValueError, match="x is in the groups"):
        core.drug_groups({"a": ["x"], "b": ["x"]})


def test_session_side() -> None:
    side = core.session_side
    assert side("W-sodium", False, False, False, set()) == "unknown"  # nothing listed
    assert side("W-sodium", True, True, False, {"sodium", "sv2a"}) == "with"
    assert side("W-sodium", True, True, False, {"sv2a"}) == "without"  # a maintenance drug of another group
    assert side("W-sodium", True, False, False, set()) == "without"  # other drugs only
    assert side("W-sodium", True, True, True, set()) == "unknown"  # "seizure medication": the group is not known
    assert side("W-sodium", True, True, True, {"sodium"}) == "with"  # a named drug of the group decides
    assert side("W-any", True, True, True, set()) == "with" and side("W-any", True, False, False, set()) == "without"
    assert side("W-any", False, False, False, set()) == "unknown"
    assert (
        side("W-lev", True, True, False, {"sv2a"}) == "with"
        and side("W-gaba", True, True, False, {"sv2a"}) == "without"
    )
    with pytest.raises(KeyError):
        side("W-other", True, True, False, set())


def test_direction_and_consistency() -> None:
    assert core.direction([2, 3], [0, 1]) == "added" and core.direction([0], [1, 2]) == "removed"
    assert core.direction([0, 3], [1, 2]) == "removed"  # the same mean order: not later
    assert core.consistent(["without", "with", "with"]) and core.consistent(["with", "unknown", "without"])
    assert not core.consistent(["with", "without", "with"]) and not core.consistent(["with", "with"])


def test_patient_entry() -> None:
    sides = ["without", "unknown", "with", "with"]
    entry = core.patient_entry(sides, [5, 9, 2, 2], [0, 1, 2, 3], [2010, 2011, 2012, 2014], ["a", "b", "a", "a"])
    assert entry is not None
    assert (entry["sessions_with"], entry["sessions_without"]) == (2, 1)
    assert (entry["segments_with"], entry["segments_without"], entry["eligible"]) == (4, 5, True)
    assert entry["direction"] == "added" and entry["consistent"] and entry["same_acquisition"]
    assert entry["gap_years"] == pytest.approx(3.0)
    few = core.patient_entry(sides, [5, 9, 2, 1], [0, 1, 2, 3], [2010] * 4, ["a", "b", "a", "c"])
    assert few is not None and not few["eligible"] and not few["same_acquisition"]  # 3 segments with the drug
    preview = core.patient_entry(sides, [None] * 4, [0, 1, 2, 3], [2010] * 4, ["a"] * 4)
    assert preview is not None and preview["eligible"] is None and preview["segments_with"] is None
    assert core.patient_entry(["with", "with", "unknown"], [4, 4, 4], [0, 1, 2], [2010] * 3, ["a"] * 3) is None


def test_controls() -> None:
    assert core.control_halves(2) == ([0], [1]) and core.control_halves(5) == ([0, 1, 2], [3, 4])
    with pytest.raises(ValueError):
        core.control_halves(1)
    control = core.control_entry(["with", "unknown", "with", "with"], [4, 9, 2, 2])
    assert control is not None and control["side"] == "with"
    assert (control["earlier"], control["later"]) == ([0, 2], [3])  # the unknown session is not counted
    assert (control["segments_earlier"], control["segments_later"], control["eligible"]) == (6, 2, False)
    assert core.control_entry(["with", "without"], [4, 4]) is None  # a change: not a control
    assert core.control_entry(["with", "unknown"], [4, 4]) is None  # one known session
    assert core.control_entry(["without", "without"], [None, None])["eligible"] is None
    assert core.contrast_eligible(15) and not core.contrast_eligible(14)


def test_measures() -> None:
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(6, core.N_CHANNELS, 10))
    values = core.measure_values(cube.reshape(6, -1), np.arange(6.0))
    assert set(values) == set(core.MEASURES) and np.array_equal(values["M-alpha-frequency"], np.arange(6.0))
    np.testing.assert_allclose(values["M-beta"], cube[:, :, 3].mean(1))
    np.testing.assert_allclose(values["M-level"], cube[:, :, :4].mean((1, 2)))
    np.testing.assert_allclose(values["M-slow"], (cube[:, :, 1] - cube[:, :, 2]).mean(1))
    relative = cube.copy()
    relative[:, :, 5:] += 100.0  # the relative powers do not enter a measure
    np.testing.assert_allclose(core.measure_values(relative.reshape(6, -1), np.zeros(6))["M-level"], values["M-level"])
    with pytest.raises(ValueError, match="200 columns"):
        core.band_cube(np.zeros((3, 100)))


def test_alpha_frequency_follows_the_peak() -> None:
    freqs = np.arange(0, 64.5, 0.5)  # the grid of a Welch spectrum with pieces of 2 s

    def spectrum(peak: float) -> np.ndarray:
        return 1.0 / (1.0 + freqs) + 4.0 * np.exp(-0.5 * ((freqs - peak) / 0.8) ** 2)

    peaks = np.array([8.0, 9.0, 10.0, 11.0])
    found = core.alpha_frequency(np.stack([spectrum(p) for p in peaks]), freqs)
    assert np.all(np.diff(found) > 0.5)  # the measure follows the peak
    shifted = core.alpha_frequency(np.stack([spectrum(p + 0.5) for p in peaks]), freqs)
    assert np.all(np.abs((shifted - found) - 0.5) < 0.15)  # a shift of 0.5 Hz, although a bin is 0.5 Hz wide
    assert np.isnan(core.alpha_frequency(np.zeros((1, len(freqs))), freqs)[0])
    outside = spectrum(9.0)
    outside[freqs > 13] += 50.0  # power outside 6 to 13 Hz does not move the measure
    assert core.alpha_frequency(outside[None], freqs)[0] == pytest.approx(found[1])


def test_patient_difference() -> None:
    values = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 4.0, 2.0])
    session = np.array(["s1", "s1", "s1", "s1", "s2", "s3", "s3"])
    side = np.array(["with"] * 5 + ["without"] * 2)
    # With: the session means 1 and 9 count equally, so 5 and not the mean of the five segments (2.6).
    assert core.patient_difference(values, session, side) == pytest.approx(5.0 - 3.0)
    with pytest.raises(ValueError, match="both sides"):
        core.patient_difference(values[:5], session[:5], side[:5])


def test_sign_flip_test() -> None:
    rng = np.random.default_rng(1)
    effect = rng.normal(1.0, 1.0, 20)
    up = core.sign_flip_test(effect, 1, n_flips=2000)
    assert up["p"] < 0.01 and up["patients"] == 20 and up["share_expected"] > 0.7 and up["t"] > 3
    assert core.sign_flip_test(effect, -1, n_flips=2000)["p"] > 0.99  # the wrong direction never passes
    assert core.sign_flip_test(-effect, -1, n_flips=2000)["p"] < 0.01
    null = [core.sign_flip_test(rng.normal(0.0, 1.0, 15), 1, n_flips=500, seed=s)["p"] for s in range(200)]
    assert 0.02 < np.mean(np.array(null) <= 0.05) < 0.10  # the false alarm rate is near 5%
    assert core.sign_flip_test(np.zeros(5), 1, n_flips=100)["p"] == 1.0
    for bad in (([1.0], 1), ([1.0, 2.0], 0)):
        with pytest.raises(ValueError):
            core.sign_flip_test(*bad)


def test_time_test() -> None:
    rng = np.random.default_rng(2)
    control = rng.normal(0.0, 1.0, 30)
    assert core.time_test(rng.normal(1.5, 1.0, 9), control, 1) == {"ran": False, "added": 9, "control": 30}
    planted = core.time_test(rng.normal(1.5, 1.0, 20), control, 1, n_perm=2000)
    assert planted["ran"] and planted["p"] < 0.01 and core.passes_time(planted)
    same = core.time_test(rng.normal(0.0, 1.0, 20), control, 1, n_perm=2000)
    assert same["p"] > 0.05 and not core.passes_time(same)
    # A change that is as large in the control patients is time, and it does not pass.
    drift = core.time_test(rng.normal(1.5, 1.0, 20), rng.normal(1.5, 1.0, 30), 1, n_perm=2000)
    assert drift["p"] > 0.05
    assert not core.passes_time({"ran": False})
    assert core.passes_change(0.016, 3) and not core.passes_change(0.017, 3) and core.passes_change(0.05, 1)


def test_strata() -> None:
    assert core.stratum(0.0, True) == "same_year|same_acquisition"
    assert core.stratum(0.0, False) == "same_year|other_acquisition"
    assert core.stratum(1.5, True) == "later_year|same_acquisition"
    assert core.stratum(0.5, False) == "later_year|other_acquisition"
    assert {core.stratum(g, a) for g in (0.0, 2.0) for a in (True, False)} == set(core.STRATA)
    entry = core.control_entry(["with", "unknown", "with", "with"], [4, 9, 4, 4])
    assert (entry["earlier"], entry["later"]) == ([0, 2], [3])  # the unknown session is not counted
    assert core.control_stratum(entry, [2010, 2015, 2010, 2010], ["a", "b", "a", "a"]) == "same_year|same_acquisition"
    assert core.control_stratum(entry, [2010, 2010, 2010, 2012], ["a", "a", "a", "a"]) == "later_year|same_acquisition"
    assert core.control_stratum(entry, [2010, 2010, 2011, 2012], ["a", "a", "b", "a"]) == "later_year|other_acquisition"


def test_stratified_time_test() -> None:
    rng = np.random.default_rng(0)
    # A time confound: every patient with a later year changes by 3, with or without a new drug. The "added"
    # patients are mostly in later years, and the control patients are mostly in the same year.
    added = np.concatenate([rng.normal(0.0, 1.0, 4), rng.normal(3.0, 1.0, 16)])
    added_strata = ["same"] * 4 + ["later"] * 16
    control = np.concatenate([rng.normal(0.0, 1.0, 40), rng.normal(3.0, 1.0, 12)])
    control_strata = ["same"] * 40 + ["later"] * 12
    assert core.time_test(added, control, 1, n_perm=2000)["p"] <= 0.05  # the test as written passes: a false effect
    confound = core.stratified_time_test(added, added_strata, control, control_strata, 1, n_perm=2000)
    assert confound["ran"] and confound["stratified"] and not core.passes_time(confound)
    assert confound["strata"] == {"later": [16, 12], "same": [4, 40]} and confound["strata_left_out"] == []
    assert abs(confound["difference"]) < 1.0  # about 3 standard errors

    # A drug effect on top of the time change: the stratified test finds it.
    planted = core.stratified_time_test(added + 1.5, added_strata, control, control_strata, 1, n_perm=2000)
    assert core.passes_time(planted) and abs(planted["difference"] - 1.5) < 1.0
    assert not core.passes_time(core.stratified_time_test(added + 1.5, added_strata, control, control_strata, -1))

    # The statistic: the mean over the "added" patients of their difference minus the control mean of their stratum.
    exact = core.stratified_time_test(
        [1.0, 3.0, 5.0] * 4, ["a", "a", "b"] * 4, [0.0] * 6 + [1.0] * 6, ["a"] * 6 + ["b"] * 6, 1
    )
    assert exact["difference"] == pytest.approx((8 * 2.0 + 4 * 4.0) / 12)
    assert (exact["mean_added"], exact["mean_control"]) == (pytest.approx(3.0), pytest.approx(0.5))

    # One stratum only: the same test as the one of section 8.
    single = core.stratified_time_test(added, ["s"] * 20, control[:30], ["s"] * 30, 1, n_perm=2000)
    assert single["difference"] == pytest.approx(core.time_test(added, control[:30], 1, n_perm=2000)["difference"])

    # A stratum with no control patient leaves the test, and the floors apply to the strata that enter.
    left = core.stratified_time_test(added, ["x"] * 4 + ["later"] * 16, control, control_strata, 1, n_perm=2000)
    assert left["ran"] and (left["added"], left["control"]) == (16, 12) and left["strata_left_out"] == ["same", "x"]
    few = core.stratified_time_test(added, ["x"] * 12 + ["later"] * 8, control, control_strata, 1)
    assert not few["ran"] and few["added"] == 8 and "p" not in few
    with pytest.raises(ValueError, match="one stratum"):
        core.stratified_time_test(added, added_strata[:-1], control, control_strata, 1)


def test_outcome() -> None:
    eligible = ["W-sodium", "W-any"]
    change = {"W-sodium": True, "W-any": False}
    assert core.signature_outcome(False, eligible, change, {"W-sodium": True}) == "blocked"
    assert core.signature_outcome(True, [], {}, {}) == "too_few_patients"
    assert core.signature_outcome(True, eligible, {"W-sodium": False, "W-any": False}, {}) == "no_effect_measured"
    assert core.signature_outcome(True, eligible, change, {"W-sodium": False, "W-any": True}) == "time_or_drug"
    assert core.signature_outcome(True, eligible, change, {"W-sodium": True, "W-any": False}) == "within_patient_effect"
    assert set(core.OUTCOMES) == {
        "within_patient_effect",
        "time_or_drug",
        "no_effect_measured",
        "too_few_patients",
        "blocked",
    }
