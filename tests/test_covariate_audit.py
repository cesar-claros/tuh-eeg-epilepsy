"""Checks of the covariate audit driver ``src/covariate_audit.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first
failure. Covered: the read plan and its draw, the inventory on a written spreadsheet, the extract on
written EDF headers, the refusal without a mapping, a block before any EEG score is opened, and the
read end to end on written scores: a score with its own signal holds, and a score of age alone does not.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

import src.covariate_audit as driver  # noqa: E402
from src.frozen_readout_core import assign_folds, segment_draw  # noqa: E402

N_PATIENTS, N_SEGMENTS, N_BOOT = 156, 8, 200
SEED = 7
MAPPING = {
    "subject_column": "Subject",
    "session_column": "Session",
    "medication_columns": ["Medications"],
    "antiseizure_terms": ["keppra", "valproate"],
    "empty_means": "unknown",
    "empty_terms": ["not listed"],
}


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _subject(i: int) -> str:
    """Eight letters from an index, the form of a corpus patient ID."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "aaaaa" + letters[i // 676 % 26] + letters[i // 26 % 26] + letters[i % 26]


def _write_edf(path: Path, age: int, sex: str) -> None:
    """A file holding only the first 88 bytes of an EDF header, as the parser reads them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    field = f"{path.stem[:8]} {sex} 01-JAN-1970 {path.stem[:8]} Age:{age}".ljust(80)
    path.write_bytes(b"0       " + field.encode("ascii"))


def _parse(path: str) -> tuple[int, str]:
    """The parser's reading of the fields of ``_write_edf``, without its heavy module."""
    text = Path(path).read_bytes()[8:88].decode("ascii")
    return int(text.split("Age:")[1].split()[0]), text.split()[1]


def _corpus(base: Path, year_drift: int = 0) -> tuple[dict[str, list[tuple[str, float]]], dict[str, int], np.ndarray]:
    """Written EDF headers of 156 patients with two sessions each, the plan keys, the labels and the ages.

    ``year_drift`` adds years to the second session's age, which breaks the age validity gate.
    """
    rng = np.random.default_rng(SEED)
    keys, labels, ages = {}, {}, []
    for i in range(N_PATIENTS):
        subject = _subject(i)
        labels[subject] = int(i < 79)
        age = int(np.clip(rng.normal(45, 20), 18, 90))
        ages.append(age)
        sex = "F" if rng.random() < 0.5 else "M"
        items = []
        for number, (session, year) in enumerate((("s001", 2010), ("s002", 2012))):
            path = base / "00_epilepsy" / subject / f"{session}_{year}" / "01_tcp_ar" / f"{subject}_{session}_t000.edf"
            _write_edf(path, age + 2 * number + year_drift * number, sex)
            items += [(str(path), float(start)) for start in range(0, 600, 120)]
        keys[subject] = sorted(items)
    return keys, labels, np.array(ages, dtype=float)


def _readout(base: Path, keys: dict, labels: dict[str, int], age: np.ndarray, spectral: np.ndarray) -> Path:
    """A written readout root, with the score of ``Spectral`` given and the others as noise."""
    root = base / "readout"
    (root / "scores").mkdir(parents=True)
    (root / "features").mkdir()
    subjects = sorted(labels)
    folds = assign_folds(labels, SEED)
    (root / "folds.json").write_text(
        json.dumps({"seed": SEED, "n_segments": N_SEGMENTS, "folds": folds, "labels": labels})
    )
    pl.DataFrame(
        {
            "subject": subjects,
            "label": [labels[s] for s in subjects],
            "fold": [folds[s] for s in subjects],
            "segments": [len(keys[s]) for s in subjects],
            "recordings": [2] * len(subjects),
            "native_rate": [250.0] * len(subjects),
        }
    ).write_csv(root / "patients.csv")
    rng = np.random.default_rng(SEED + 1)
    arms = {"Spectral": spectral, "HYDRA": rng.standard_normal(len(subjects))}
    arms |= {"S4-R1": rng.standard_normal(len(subjects)), "S4-proj": rng.standard_normal(len(subjects))}
    for name, score in arms.items():
        pl.DataFrame(
            {
                "arm": [name] * len(subjects),
                "subject": subjects,
                "fold": [folds[s] for s in subjects],
                "label": [labels[s] for s in subjects],
                "score": score.tolist(),
            }
        ).write_csv(root / "scores" / f"{name}.csv")
    rows = np.repeat(np.arange(len(subjects)), N_SEGMENTS)
    features = rng.standard_normal((len(rows), 200)).astype(np.float32)
    features[:, 0] += 0.5 * spectral[rows].astype(np.float32)
    np.savez(
        root / "features" / "Spectral.npz",
        subject=np.array(subjects)[rows],
        label=np.array([labels[subjects[i]] for i in rows]),
        features=features,
    )
    return root


def _spreadsheet(path: Path, labels: dict[str, int]) -> None:
    """One row per session, plus one locked patient."""
    rows = []
    for i, subject in enumerate(sorted(labels)):
        for session in ("s001", "s002"):
            rows.append(
                {
                    "Subject": subject,
                    "Session": session,
                    "Age": 40 + i % 30,
                    "Epilepsy": labels[subject],
                    "Medications": "Keppra 500 mg" if labels[subject] and i % 3 else "aspirin",
                }
            )
    rows.append({"Subject": "lockedpt", "Session": "s001", "Age": 33, "Epilepsy": 1, "Medications": "secretdrug"})
    pd.DataFrame(rows).to_excel(path, index=False)


def _prepared(base: Path, spectral_from_age: bool = False, year_drift: int = 0) -> tuple[Path, Path, Path, np.ndarray]:
    keys, labels, age = _corpus(base / "corpus", year_drift)
    rng = np.random.default_rng(SEED + 2)
    subjects = sorted(labels)
    y = np.array([labels[s] for s in subjects])
    ordered_age = age[np.argsort([_subject(i) for i in range(N_PATIENTS)])]
    if spectral_from_age:
        spectral = -ordered_age / 20.0 + 0.5 * rng.standard_normal(N_PATIENTS)
    else:
        spectral = 1.2 * y + rng.standard_normal(N_PATIENTS)
    readout = _readout(base, keys, labels, ordered_age, spectral)
    root = base / "audit"
    root.mkdir()
    driver.write_read_plan(root, keys, N_SEGMENTS)
    driver.write_covariates(root, _parse)
    metadata = base / "metadata.xlsx"
    _spreadsheet(metadata, labels)
    return root, readout, metadata, y


def test_read_plan_uses_the_draw_of_the_readout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        keys, _, _ = _corpus(base / "corpus")
        driver.write_read_plan(base, keys, N_SEGMENTS)
        read = pl.read_csv(base / "read_segments.csv", schema_overrides={"subject": pl.Utf8})
        subject = _subject(5)
        expected = [keys[subject][i] for i in segment_draw(subject, keys[subject], N_SEGMENTS)]
        got = read.filter(pl.col("subject") == subject)
        assert list(zip(got["path"].to_list(), got["start"].to_list())) == expected, "the draw must not change"
        recordings = pl.read_csv(base / "plan_recordings.csv", schema_overrides={"subject": pl.Utf8})
        assert recordings.filter(pl.col("subject") == subject).height == 2, "two recordings per patient"


def test_inventory_on_a_spreadsheet_drops_the_locked_patient() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        keys, labels, _ = _corpus(base / "corpus")
        driver.write_read_plan(base, keys, N_SEGMENTS)
        metadata = base / "metadata.xlsx"
        _spreadsheet(metadata, labels)
        report = driver.write_inventory(base, metadata, set(labels))
        sheet = report["sheets"][0]
        assert sheet["subject_column"] == "Subject" and sheet["session_column"] == "Session"
        assert sheet["patients_kept"] == N_PATIENTS and sheet["read_sessions_joined"] == 1.0
        text = (base / "inventory.json").read_text()
        assert "lockedpt" not in text and "secretdrug" not in text, "a locked patient reached the inventory"
        columns = {c["name"]: c for c in sheet["columns"]}
        assert "values" not in columns["Age"] and "values" not in columns["Epilepsy"]
        assert "keppra" in columns["Medications"]["tokens"]


def test_extract_reads_the_headers_of_the_read_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _, _, _ = _prepared(Path(tmp))
        covariates = pl.read_csv(root / "covariates.csv", schema_overrides={"subject": pl.Utf8, "sex": pl.Utf8})
        assert covariates.height == N_PATIENTS
        assert covariates["age"].null_count() == 0 and set(covariates["sex"].to_list()) == {"F", "M"}
        assert (covariates["files_per_read_session"].to_numpy() == 1).all(), "each session holds one recording here"
        sessions = pl.read_csv(root / "session_ages.csv", schema_overrides={"subject": pl.Utf8})
        assert sessions.height == 2 * N_PATIENTS and set(sessions["session"].to_list()) == {"s001", "s002"}


def test_the_read_refuses_to_run_without_the_mapping() -> None:
    _expect_value_error(lambda: driver.load_mapping(None), "no mapping")
    _expect_value_error(lambda: driver.load_mapping(Path("/no/such/file.json")), "a missing file")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mapping.json"
        path.write_text(json.dumps({**MAPPING, "antiseizure_terms": []}))
        _expect_value_error(lambda: driver.load_mapping(path), "an empty key")
        path.write_text(json.dumps(MAPPING))
        assert driver.load_mapping(path)["empty_means"] == "unknown"


def test_medication_status_uses_the_read_sessions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metadata.xlsx"
        pd.DataFrame(
            [
                {"Subject": "aaaaaaaa", "Session": "s001", "Medications": "Keppra"},
                {"Subject": "aaaaaaaa", "Session": "s002", "Medications": "aspirin"},
                {"Subject": "bbbbbbbb", "Session": "s001", "Medications": "not listed"},
                {"Subject": "cccccccc", "Session": "s001", "Medications": "aspirin"},
                {"Subject": "lockedpt", "Session": "s001", "Medications": "Keppra"},
            ]
        ).to_excel(path, index=False)
        sessions = {"aaaaaaaa": ["s002"], "bbbbbbbb": ["s001"], "cccccccc": ["s001"], "dddddddd": ["s001"]}
        status = driver.medication_by_patient(path, MAPPING, sessions)
        assert status == {"aaaaaaaa": "off", "bbbbbbbb": "unknown", "cccccccc": "off", "dddddddd": "unknown"}, status
        with_read = driver.medication_by_patient(path, MAPPING, {"aaaaaaaa": ["s001", "s002"]})
        assert with_read == {"aaaaaaaa": "on"}, "any read session counts, and only read sessions count"
        _expect_value_error(
            lambda: driver.medication_by_patient(path, {**MAPPING, "medication_columns": ["Nope"]}, sessions),
            "a column that does not exist",
        )


def test_a_failed_gate_blocks_before_any_eeg_score_is_opened() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout, metadata, _ = _prepared(Path(tmp), year_drift=9)
        for path in (readout / "scores").iterdir():
            path.unlink()
        (readout / "features" / "Spectral.npz").unlink()
        decision = driver.audit(root, readout, metadata, MAPPING, n_boot=N_BOOT)
        assert decision["outcome"] == "blocked", decision
        assert not decision["gates"]["age_validity_ok"]
        assert "demographics" not in decision and "primary" not in decision


def test_a_score_with_its_own_signal_holds_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout, metadata, _ = _prepared(Path(tmp))
        decision = driver.audit(root, readout, metadata, MAPPING, n_boot=N_BOOT)
        assert decision["outcome"] == "holds", decision
        assert decision["controls"]["passed"] and decision["gates"]["header_coverage"] == 1.0
        assert decision["primary"]["low"] > 0.5 and decision["primary"]["patients_epilepsy"] >= 30
        assert decision["medication_table"]["epilepsy"]["on"] > 0
        summary = pl.read_csv(root / "audit_summary.csv")
        reads = set(summary["read"].to_list())
        expected = {"matched", "matched_difference", "alone", "added_value", "sensitivity", "band_power"}
        assert expected <= reads, reads
        assert (root / "refit_scores.csv").is_file() and (root / "medication.csv").is_file()
        groups = set(summary.filter(pl.col("read") == "sensitivity")["group"].to_list())
        assert {"2.5 years", "10.0 years", "same rate", "bag ratio", "age 40 to 64"} <= groups, groups


def test_a_score_of_age_alone_does_not_hold_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout, metadata, _ = _prepared(Path(tmp), spectral_from_age=True)
        decision = driver.audit(root, readout, metadata, MAPPING, n_boot=N_BOOT)
        assert decision["outcome"] in ("not_resolved", "review"), decision
        assert decision["primary"]["low"] < 0.5, decision["primary"]


def test_the_corpus_parser_reads_the_written_header() -> None:
    from src.data.components.utils import parse_age_and_gender_from_edf_header

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aaaaaanr_s001_t000.edf"
        _write_edf(path, 42, "F")
        assert parse_age_and_gender_from_edf_header(str(path)) == (42, "F"), (
            "the audit's test files must match the parser"
        )
        assert _parse(str(path)) == (42, "F")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(test):
            test()
    print("all covariate audit driver checks passed")
