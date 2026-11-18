"""Checks of stage 1 of ``src/drug_signature.py`` on written files (no corpus; CPU; seconds).

Runs under pytest. Covered: the montage of a path; the mechanism file against the drug classes; the inventory on a
small spreadsheet, first as a preview by sessions and then with segment counts and the floor of 4 segments: the side
of every session, the patients and the controls of every contrast, the counts at the read sessions and the counts of
the between-patient read; and the refusal of a recording plan with other patients. For stages 2 and 3: the posterior
channels, the spectrum and the alpha frequency check of gate 5; the sides of the descriptive variants; the gates on
written files (a pass, changed features, a missing patient, a preview inventory); and the five outcomes of the rule on
planted cohorts (an effect, the same change in the control patients, no effect, too few patients, a failed gate),
and the stratified test against time of amendment A1 on a planted time confound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.drug_signature as driver  # noqa: E402
from src.drug_signature_core import CONTRASTS, drug_groups, session_side  # noqa: E402
from src.drug_signature_core import passes_time as core_passes  # noqa: E402

# Patient, label, and the medication text of each session in order (None: the session has no row in the sheet).
PATIENTS = {
    "aaaaaaab": (1, ["Keppra", "keppra, Dilantin", "dilantin, keppra"]),
    "aaaaaaac": (1, ["phenobarbital", "aspirin"]),
    "aaaaaaad": (1, ["seizure medication", "Tegretol"]),
    "aaaaaaae": (0, ["", "aspirin"]),
    "aaaaaaaf": (0, ["lisinopril"]),
    "aaaaaaag": (1, ["Lamictal", None]),
}
SEGMENTS = {
    "aaaaaaab": [5, 2, 3],
    "aaaaaaac": [3, 6],
    "aaaaaaad": [4, 4],
    "aaaaaaae": [4, 4],
    "aaaaaaaf": [8],
    "aaaaaaag": [4, 4],
}


def _path(subject: str, index: int) -> str:
    folder = "00_epilepsy" if PATIENTS[subject][0] else "01_no_epilepsy"
    return (
        f"/data/v3.0.0/{folder}/{subject}/s{index + 1:03d}_{2010 + index}/01_tcp_ar/{subject}_s{index + 1:03d}_t000.edf"
    )


def _roots(tmp: Path) -> dict[str, Path]:
    roots = {name: tmp / name for name in ("readout", "audit", "root")}
    for name in ("readout", "audit"):
        roots[name].mkdir()
    labels = {subject: label for subject, (label, _) in PATIENTS.items()}
    (roots["readout"] / "folds.json").write_text(json.dumps({"labels": labels}))
    plan = [
        {"subject": s, "path": _path(s, i), "session": f"s{i + 1:03d}", "year": 2010 + i}
        for s, (_, texts) in PATIENTS.items()
        for i in range(len(texts))
    ]
    pl.DataFrame(plan).write_csv(roots["audit"] / "plan_recordings.csv")
    read = [{**row, "start": 40.0} for row in plan if row["session"] == "s001"]  # the first session is the read one
    pl.DataFrame(read).write_csv(roots["audit"] / "read_segments.csv")
    sheet = [
        {"Subject ID": f"{s}_s{i + 1:03d}", "Major Medications (Comma Separated List)": text}
        for s, (_, texts) in PATIENTS.items()
        for i, text in enumerate(texts)
        if text is not None
    ]
    roots["metadata"] = tmp / "metadata.xlsx"
    pd.DataFrame(sheet).to_excel(roots["metadata"], sheet_name="TUEP", index=False)
    return roots


def _keys(roots: dict[str, Path]) -> None:
    rows = [
        {
            "subject": s,
            "path": _path(s, i),
            "start": 40.0 + 120.0 * k,
            "session": f"s{i + 1:03d}",
            "year": 2010 + i,
            "sfreq": 250.0,
        }
        for s, counts in SEGMENTS.items()
        for i, n in enumerate(counts)
        for k in range(n)
    ]
    roots["root"].mkdir(exist_ok=True)
    pl.DataFrame(rows).write_csv(roots["root"] / "segment_keys.csv")


def _run(roots: dict[str, Path]) -> dict:
    return driver.inventory(roots["root"], roots["readout"], roots["audit"], roots["metadata"])


def test_montage_and_mechanisms(tmp_path: Path) -> None:
    assert driver.montage_of("/x/00_epilepsy/aaaaaaab/s001_2010/02_tcp_le/a.edf") == "tcp_le"
    assert driver.montage_of("/x/a.edf") == "unknown"
    mapping = driver.load_mapping(driver.CONFIGS / "medication_mapping.json")
    classes = driver.load_drug_classes(driver.CONFIGS / "drug_classes.json", mapping)
    mechanisms = driver.load_mechanisms(driver.CONFIGS / "drug_mechanisms.json", classes)
    assert sorted(mechanisms["groups"]) == ["gaba", "several", "sodium", "sv2a"]
    assert sum(len(v) for v in mechanisms["groups"].values()) == 13  # the 13 maintenance drugs, each in one group
    assert mechanisms["groups"]["several"] == ["valproate", "topiramate", "zonisamide", "felbamate"]
    broken = json.loads((driver.CONFIGS / "drug_mechanisms.json").read_text())
    broken["groups"]["sodium"].remove("phenytoin")
    (tmp_path / "m.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="do not cover"):
        driver.load_mechanisms(tmp_path / "m.json", classes)


def test_inventory_preview(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    record = _run(roots)
    assert record["preview"] and record["patients"] == 6 and record["sessions"] == 12
    assert record["sessions_in_sheet"] == 11 and record["sessions_with_a_list"] == 10
    rows = {(r["subject"], r["session"]): r for r in pl.read_csv(roots["root"] / "session_drugs.csv").to_dicts()}
    first = rows[("aaaaaaab", "s002")]
    assert first["drugs"] == "levetiracetam+phenytoin" and first["groups"] == "sodium+sv2a"
    assert (first["W-sodium"], first["W-lev"], first["W-gaba"], first["W-any"]) == ("with", "with", "without", "with")
    assert rows[("aaaaaaab", "s001")]["W-sodium"] == "without"
    unnamed = rows[("aaaaaaad", "s001")]  # "seizure medication": treated, and the group is not known
    assert (unnamed["W-any"], unnamed["W-sodium"], unnamed["W-gaba"]) == ("with", "unknown", "unknown")
    assert not unnamed["drugs"]  # no drug is named
    assert rows[("aaaaaaae", "s001")]["W-any"] == "unknown" and rows[("aaaaaaae", "s002")]["W-any"] == "without"
    missing = rows[("aaaaaaag", "s002")]
    assert not missing["in_sheet"] and missing["W-sodium"] == "unknown" and missing["segments"] is None
    assert rows[("aaaaaaab", "s001")]["montages"] == "tcp_ar"

    sodium = record["contrasts"]["W-sodium"]
    assert sodium["patients_with_both_sides"] == 1 and sodium["eligible_patients"] == 1  # a preview: no floor
    assert (sodium["added"], sodium["removed"], sodium["consistent_report"]) == (1, 0, 1)
    assert not sodium["contrast_eligible"] and sodium["deciding"] and not sodium["time_test_can_run"]
    assert sodium["control_patients"] == 1  # aaaaaaac, never on one; aaaaaaae has one known session only
    gaba = record["contrasts"]["W-gaba"]
    assert (gaba["patients_with_both_sides"], gaba["removed"]) == (1, 1)
    assert (
        record["contrasts"]["W-any"]["patients_with_both_sides"] == 1 and not record["contrasts"]["W-lev"]["deciding"]
    )
    patients = pl.read_csv(roots["root"] / "contrast_patients.csv").to_dicts()
    assert {(p["contrast"], p["subject"], p["direction"]) for p in patients} == {
        ("W-sodium", "aaaaaaab", "added"),
        ("W-gaba", "aaaaaaac", "removed"),
        ("W-any", "aaaaaaac", "removed"),
    }
    assert record["status_at_read_sessions"] == {
        "epilepsy": {"on": 4, "off": 0, "unknown": 0},
        "non_epilepsy": {"on": 0, "off": 1, "unknown": 1},
    }
    assert not record["counts_reproduce"]  # a small cohort: not the counts of the audit
    between = record["between_patients"]
    assert between["treated_epilepsy_patients"] == 4
    assert between["patients_per_group"] == {"sodium": 1, "sv2a": 1, "gaba": 1, "several": 0}
    assert between["T-lev-only"] == [1, 3] and between["T-gaba"] == [1, 3]


def test_inventory_with_segments(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    _keys(roots)
    record = _run(roots)
    assert not record["preview"]
    patients = {
        (p["contrast"], p["subject"]): p for p in pl.read_csv(roots["root"] / "contrast_patients.csv").to_dicts()
    }
    sodium = patients[("W-sodium", "aaaaaaab")]
    assert (sodium["segments_with"], sodium["segments_without"], sodium["eligible"]) == (5, 5, True)
    gaba = patients[("W-gaba", "aaaaaaac")]
    assert (gaba["segments_with"], gaba["segments_without"], gaba["eligible"]) == (3, 6, False)  # below the floor
    assert record["contrasts"]["W-sodium"]["eligible_patients"] == 1
    assert (
        record["contrasts"]["W-gaba"]["eligible_patients"] == 0
        and record["contrasts"]["W-gaba"]["patients_with_both_sides"] == 1
    )
    rows = {(r["subject"], r["session"]): r for r in pl.read_csv(roots["root"] / "session_drugs.csv").to_dicts()}
    assert rows[("aaaaaaab", "s002")]["segments"] == 2 and rows[("aaaaaaab", "s002")]["rates"] == 250
    controls = {
        (c["contrast"], c["subject"]): c for c in pl.read_csv(roots["root"] / "control_patients.csv").to_dicts()
    }
    assert controls[("W-lev", "aaaaaaab")]["segments_earlier"] == 7 and not controls[("W-lev", "aaaaaaab")]["eligible"]
    assert controls[("W-sodium", "aaaaaaac")]["eligible"] is False  # 3 segments in the earlier half


def test_inventory_refuses_other_patients(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    plan = json.loads((roots["readout"] / "folds.json").read_text())
    plan["labels"]["aaaaaaah"] = 0
    (roots["readout"] / "folds.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="different patients"):
        _run(roots)


# ---------------------------------------------------------------------- stages 2 and 3


def test_posterior_channels_and_alpha_check() -> None:
    torch = pytest.importorskip("torch")
    channels = ["FP1-F7", "T5-O1", "P3-O1", "t6-o2", "P4-O2"]
    assert driver.posterior_index(channels) == [1, 3, 2, 4]  # in the order of POSTERIOR, and the case does not matter
    with pytest.raises(ValueError, match="no channels"):
        driver.posterior_index(channels[:3])
    t = torch.arange(30 * 256) / 256
    x = torch.zeros(2, 5, len(t))
    x[0, 1:] = torch.sin(2 * torch.pi * 10 * t)
    x[1, 1:] = torch.sin(2 * torch.pi * 8 * t)
    x[:, 0] = torch.sin(2 * torch.pi * 12 * t)  # not a posterior channel: it must not move the measure
    psd, freqs = driver.posterior_spectrum(x, [1, 2, 3, 4], 256.0)
    assert psd.shape == (2, 257) and freqs[1] == 0.5 and freqs[-1] == 128.0
    found = driver.alpha_frequency(psd, freqs)
    assert abs(found[0] - 10.0) < 0.01 and abs(found[1] - 8.0) < 0.01
    check = driver.alpha_check()
    assert check["pass"] and check["largest_shift_error"] <= driver.SHIFT_TOLERANCE


def test_variant_side() -> None:
    mechanisms = json.loads((driver.CONFIGS / "drug_mechanisms.json").read_text())

    def side(name: str, drugs: str, **base: str) -> str:
        return driver.variant_side(name, {"drugs": drugs, **base}, mechanisms)

    assert side("W-sodium-no-lamotrigine", "lamotrigine+phenytoin", **{"W-sodium": "with"}) == "unknown"
    assert side("W-sodium-no-lamotrigine", "phenytoin", **{"W-sodium": "with"}) == "with"
    assert side("W-sodium-wide", "valproate", **{"W-sodium": "without"}) == "with"
    assert side("W-sodium-wide", "levetiracetam", **{"W-sodium": "without"}) == "without"
    assert side("W-sodium-wide", "valproate", **{"W-sodium": "unknown"}) == "unknown"
    assert side("W-gaba-wide", "topiramate", **{"W-gaba": "without"}) == "with"
    assert side("W-gaba-wide", None, **{"W-gaba": "without"}) == "without"  # an empty cell is read as None


def _signature_root(
    tmp: Path,
    n_added: int = 20,
    n_controls: int = 12,
    effect: float = -0.5,
    drift: float = 0.0,
    same_year_controls: int = 0,
) -> dict[str, Path]:
    """A planted cohort for ``W-sodium``: ``n_added`` patients on levetiracetam who get phenytoin in the second
    session, where the alpha frequency moves by ``effect``; ``n_controls`` patients on levetiracetam in both
    sessions, where it moves by ``drift``. Their second session is one year later. ``same_year_controls`` more
    control patients have both sessions in the same year and no change. No other contrast has a patient with both
    sides. The band power is noise.
    """
    owner = drug_groups(json.loads((driver.CONFIGS / "drug_mechanisms.json").read_text())["groups"])
    rng = np.random.default_rng(0)
    rows, keys, alpha, features = [], [], [], []
    plans = [(f"added{i:03d}", "levetiracetam+phenytoin", effect, 1) for i in range(n_added)]
    plans += [(f"ctrl{i:04d}", "levetiracetam", drift, 1) for i in range(n_controls)]
    plans += [(f"same{i:04d}", "levetiracetam", 0.0, 0) for i in range(same_year_controls)]
    for subject, second, shift, years in sorted(plans):
        base, level = rng.normal(9.5, 0.5), rng.normal(0.0, 1.0, 200)
        for order, (drugs, delta) in enumerate((("levetiracetam", 0.0), (second, shift))):
            groups = {owner[d] for d in drugs.split("+")}
            session = f"s{order + 1:03d}"
            year = 2010 + order * years
            rows.append(
                {
                    "subject": subject,
                    "label": 1,
                    "session": session,
                    "order": order,
                    "year": year,
                    "recordings": 1,
                    "segments": 4,
                    "montages": "tcp_ar",
                    "rates": "250",
                    "drugs": drugs,
                    **{name: session_side(name, True, True, False, groups) for name in CONTRASTS},
                }
            )
            for k in range(4):
                keys.append(
                    {
                        "subject": subject,
                        "path": f"/d/00_epilepsy/{subject}/{session}_{year}/01_tcp_ar/{subject}_{session}.edf",
                        "start": 40.0 + 120.0 * k,
                        "session": session,
                        "year": year,
                        "sfreq": 250.0,
                    }
                )
                alpha.append(base + delta + rng.normal(0.0, 0.05))
                features.append(level + rng.normal(0.0, 0.05, 200))
    roots = {name: tmp / name for name in ("root", "readout")}
    (roots["readout"] / "features").mkdir(parents=True)
    roots["root"].mkdir()
    subject = np.array([key["subject"] for key in keys])
    stacked = np.stack(features).astype(np.float32)
    patients, controls = driver.contrast_rows(rows)
    record = {
        "preview": False,
        "counts_reproduce": True,
        "contrasts": {name: driver.contrast_summary(name, patients, controls, True) for name in CONTRASTS},
    }
    (roots["root"] / "drug_inventory.json").write_text(json.dumps(record))
    pl.DataFrame(rows).write_csv(roots["root"] / "session_drugs.csv")
    pl.DataFrame(keys).write_csv(roots["root"] / "segment_keys.csv")
    np.savez(roots["root"] / driver.FEATURES, subject=subject, features=stacked, alpha_frequency=np.array(alpha))
    # The files of the first readout: the same features for all segments, and the read segments in another order.
    chosen = np.arange(len(subject)) % 4 < 2
    np.savez(roots["readout"] / "features" / "Spectral-all.npz", subject=subject, features=stacked, in_nine=chosen)
    order = np.flatnonzero(chosen)[::-1]
    np.savez(roots["readout"] / "features" / "Spectral.npz", subject=subject[order], features=stacked[order])
    labels = {name: 1 for name in sorted(set(subject.tolist()))}
    (roots["readout"] / "folds.json").write_text(json.dumps({"labels": labels}))
    return roots


def _pass_gates(roots: dict[str, Path]) -> None:
    (roots["root"] / "gates.json").write_text(json.dumps({"status": "pass", "gates": {}}))


def test_gates(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    roots = _signature_root(tmp_path)
    record = driver.gates(roots["root"], roots["readout"])
    assert record["status"] == "pass" and all(record["gates"].values())
    assert record["feature_gap_all_segments"] == 0.0 and record["feature_gap_read_segments"] == 0.0
    assert record["segments"] == 32 * 8 and record["segments_without_alpha_frequency"] == 0
    assert json.loads((roots["root"] / "gates.json").read_text())["status"] == "pass"

    stored = dict(np.load(roots["root"] / driver.FEATURES))
    changed = stored | {"features": stored["features"] + 0.01}
    np.savez(roots["root"] / driver.FEATURES, **changed)
    record = driver.gates(roots["root"], roots["readout"])
    assert record["status"] == "blocked" and not record["gates"]["features_reproduce"]
    assert record["gates"]["keys_complete"] and record["gates"]["no_test_patient"]
    np.savez(roots["root"] / driver.FEATURES, **stored)

    plan = json.loads((roots["readout"] / "folds.json").read_text())
    plan["labels"]["other001"] = 0  # a patient of the readout plan with no segment
    (roots["readout"] / "folds.json").write_text(json.dumps(plan))
    assert not driver.gates(roots["root"], roots["readout"])["gates"]["no_test_patient"]
    del plan["labels"]["other001"]
    (roots["readout"] / "folds.json").write_text(json.dumps(plan))

    inventory = json.loads((roots["root"] / "drug_inventory.json").read_text())
    (roots["root"] / "drug_inventory.json").write_text(json.dumps(inventory | {"preview": True}))
    record = driver.gates(roots["root"], roots["readout"])
    assert record["status"] == "blocked" and not record["gates"]["inventory_has_segments"]
    (roots["root"] / "drug_inventory.json").write_text(json.dumps(inventory))

    sessions = pl.read_csv(roots["root"] / "session_drugs.csv")
    sessions.with_columns(pl.lit(5).alias("segments")).write_csv(roots["root"] / "session_drugs.csv")
    assert not driver.gates(roots["root"], roots["readout"])["gates"]["keys_complete"]


def test_outcome_within_patient_effect(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, effect=-0.5, drift=0.0)
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "within_patient_effect"
    assert record["eligible_deciding_contrasts"] == ["W-sodium"] and record["level_per_contrast"] == 0.05
    sodium = record["deciding"]["W-sodium"]
    assert sodium["patients"] == 20 and sodium["passes_no_change"] and sodium["passes_time"]
    assert abs(sodium["against_no_change"]["mean"] + 0.5) < 0.05 and sodium["against_no_change"]["share_expected"] == 1
    assert sodium["against_time"]["ran"] and (sodium["against_time"]["added"], sodium["against_time"]["control"]) == (
        20,
        12,
    )
    assert sodium["against_time"]["stratified"]
    assert sodium["against_time"]["strata"] == {"later_year|same_acquisition": [20, 12]}
    written = sodium["against_time_as_written"]  # one stratum: the stratified test equals the test of section 8
    assert written["ran"] and written["difference"] == pytest.approx(sodium["against_time"]["difference"])
    assert abs(sodium["against_time"]["mean_control"]) < 0.05
    assert record["deciding"]["W-gaba"]["patients"] == 0 and not record["deciding"]["W-gaba"]["passes_no_change"]
    assert record["deciding"]["W-gaba"]["against_no_change"] is None
    table = pl.read_csv(roots["root"] / "drug_signature_summary.csv")
    assert len(table) == record["summary_rows"]
    reads = set(zip(table["contrast"].to_list(), table["measure"].to_list(), table["read"].to_list(), strict=True))
    assert ("W-sodium", "M-alpha-frequency", "deciding") in reads
    assert ("W-sodium", "M-alpha-frequency", "added") in reads and ("W-sodium", "M-beta", "descriptive") in reads
    assert ("W-sodium-no-lamotrigine", "M-alpha-frequency", "variant") in reads
    assert ("W-sodium", "B-relative-8_13", "descriptive") in reads
    assert not any(contrast == "W-gaba" for contrast, _, _ in reads)  # no patient has both sides
    noise = table.filter((pl.col("contrast") == "W-sodium") & (pl.col("measure") == "M-level"))
    assert noise["sided"].to_list() == ["two"] and noise["p"][0] > 0.05  # the band power holds no effect


def test_outcome_time_or_drug(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, effect=-0.5, drift=-0.5)  # the control patients change by the same amount
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "time_or_drug"
    sodium = record["deciding"]["W-sodium"]
    assert sodium["passes_no_change"] and sodium["against_time"]["ran"] and not sodium["passes_time"]


def test_outcome_time_confound_is_not_an_effect(tmp_path: Path) -> None:
    # Amendment A1. Every patient with a later year changes by -0.5, with or without a new drug. 30 more control
    # patients have both sessions in the same year and no change. The test of section 8 pools all control patients
    # and passes; the stratified test compares the "added" patients with the control patients of later years only.
    roots = _signature_root(tmp_path, effect=-0.5, drift=-0.5, same_year_controls=30)
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    sodium = record["deciding"]["W-sodium"]
    assert sodium["passes_no_change"]
    assert core_passes(sodium["against_time_as_written"]) and sodium["against_time_as_written"]["control"] == 42
    assert sodium["against_time"]["strata"] == {
        "later_year|same_acquisition": [20, 12],
        "same_year|same_acquisition": [0, 30],
    }
    assert sodium["against_time"]["strata_left_out"] == ["same_year|same_acquisition"]
    assert (sodium["against_time"]["added"], sodium["against_time"]["control"]) == (20, 12)
    assert sodium["against_time"]["ran"] and not sodium["passes_time"]
    assert record["outcome"] == "time_or_drug"


def test_outcome_effect_beyond_the_time_change(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, effect=-1.0, drift=-0.5, same_year_controls=30)
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "within_patient_effect"
    assert abs(record["deciding"]["W-sodium"]["against_time"]["difference"] + 0.5) < 0.05


def test_outcome_time_test_cannot_run(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, n_controls=5)  # fewer than 10 control patients: the test against time is absent
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "time_or_drug" and not record["deciding"]["W-sodium"]["against_time"]["ran"]


def test_outcome_no_effect_measured(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, effect=0.0)
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "no_effect_measured" and not record["deciding"]["W-sodium"]["passes_no_change"]


def test_outcome_wrong_direction_is_no_effect(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, effect=0.5)  # the alpha frequency rises: against the expected direction
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "no_effect_measured"
    assert record["deciding"]["W-sodium"]["against_no_change"]["p"] > 0.99


def test_outcome_too_few_patients(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path, n_added=8)
    _pass_gates(roots)
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "too_few_patients" and record["eligible_deciding_contrasts"] == []
    assert record["level_per_contrast"] is None
    assert not record["deciding"]["W-sodium"]["passes_no_change"]  # a strong effect, and the contrast is not eligible


def test_outcome_blocked(tmp_path: Path) -> None:
    roots = _signature_root(tmp_path)
    (roots["root"] / "gates.json").write_text(json.dumps({"status": "blocked", "gates": {"counts_reproduce": False}}))
    record = driver.run_test(roots["root"])
    assert record["outcome"] == "blocked" and "deciding" not in record
    assert not (roots["root"] / "drug_signature_summary.csv").exists()
