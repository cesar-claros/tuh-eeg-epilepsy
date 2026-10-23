"""Uncapped metadata audits create or preserve patient splits without reading EEG samples."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src import count_pool  # noqa: E402


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    """Old manifests contain just one selected window per patient."""
    directory = tmp_path / "old_manifests"
    directory.mkdir()
    for split, subject, label in (("train", "a", True), ("val", "b", False), ("test", "c", True)):
        with (directory / f"windows_{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["subject", "epilepsy", "path", "start", "end"])
            writer.writeheader()
            writer.writerow(
                {"subject": subject, "epilepsy": label, "path": "/old/selected_only.edf", "start": 120, "end": 240}
            )
    return directory


@pytest.fixture
def eligible(tmp_path: Path) -> pd.DataFrame:
    """Two recordings for one patient and unequal complete-window counts."""
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "path": tmp_path / f"{subject}_{index}.edf",
                "epilepsy": label,
                "duration": duration,
                "sfreq": 250.0,
                "n_seizure": 0,
                "session": "s001",
                "year": 2020,
            }
            for index, (subject, label, duration) in enumerate(
                [("a", True, 250.0), ("a", True, 360.0), ("b", False, 239.999999), ("c", True, 120.0)]
            )
        ]
    )


def test_all_recordings_and_segments_preserve_patient_assignments(
    split_dir: Path, eligible: pd.DataFrame, tmp_path: Path
) -> None:
    """Old sampled paths do not restrict the inventory, and longer patients keep more windows."""
    output = tmp_path / "audit"
    source_bytes = {path: path.read_bytes() for path in split_dir.glob("*.csv")}

    count_pool._audit_segments(eligible, split_dir, output, 120.0, 0.0)

    manifests = {split: pd.read_csv(output / f"windows_{split}.csv") for split in ("train", "val", "test")}
    assert {split: len(frame) for split, frame in manifests.items()} == {"train": 5, "val": 1, "test": 1}
    assert {split: set(frame["subject"]) for split, frame in manifests.items()} == {
        "train": {"a"},
        "val": {"b"},
        "test": {"c"},
    }
    windows = pd.concat(manifests.values())
    assert set(windows["path"]) == set(eligible["path"].astype(str))
    assert (windows["end"] - windows["start"] == 120.0).all()
    assert manifests["train"]["window_idx_within_subject"].tolist() == list(range(5))
    assert not windows.duplicated(["path", "start", "end"]).any()
    for _, record in eligible.iterrows():
        selected = windows[windows["path"] == str(record["path"])]
        assert selected["start"].min() == 0
        assert selected["end"].max() <= record["duration"]
    by_subject = pd.read_csv(output / "segment_counts_by_subject.csv").set_index("subject")
    assert by_subject["segments"].to_dict() == {"a": 5, "b": 1, "c": 1}
    assert by_subject["recordings"].to_dict() == {"a": 2, "b": 1, "c": 1}
    by_class = pd.read_csv(output / "segment_counts_by_class.csv").set_index("epilepsy")
    assert by_class["segments"].to_dict() == {0: 1, 1: 6}
    assert by_class["segment_hours"].sum() == pytest.approx(7 / 30)
    provenance = json.loads((output / "audit_provenance.json").read_text())
    assert (provenance["recordings"], provenance["subjects"], provenance["segments"]) == (4, 3, 7)
    for path, content in source_bytes.items():
        assert path.read_bytes() == content
        assert (
            provenance["source_manifests"][path.stem.removeprefix("windows_")]["sha256"]
            == hashlib.sha256(content).hexdigest()
        )


@pytest.mark.parametrize("case", [(120, 0, 1), (239.999999, 0, 1), (240, 0, 2), (260, 20, 2)])
def test_complete_segments_never_round_up_partial_tails(
    split_dir: Path, eligible: pd.DataFrame, tmp_path: Path, case: tuple[float, float, int]
) -> None:
    """Exact boundaries and a nonzero offset use only complete intervals."""
    duration, offset, expected = case
    record = eligible.iloc[:1].assign(duration=duration)
    output = tmp_path / "audit"
    count_pool._audit_segments(record, split_dir, output, 120.0, offset)
    windows = pd.read_csv(output / "windows_train.csv")
    assert len(windows) == expected
    assert windows["start"].tolist() == [offset + 120 * index for index in range(expected)]
    assert windows["end"].max() <= duration


@pytest.mark.parametrize("problem", ["missing_subject", "label_mismatch", "split_overlap"])
def test_assignment_errors_fail_before_writing_audit(
    split_dir: Path, eligible: pd.DataFrame, tmp_path: Path, problem: str
) -> None:
    """An eligible patient is never dropped or silently assigned to a different split."""
    if problem == "missing_subject":
        eligible.loc[0, "subject"] = "unassigned"
    elif problem == "label_mismatch":
        eligible.loc[0, "epilepsy"] = False
    else:
        with (split_dir / "windows_val.csv").open("a") as handle:
            handle.write("a,True,/old/another.edf,0,120\n")
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="assignments|disagree"):
        count_pool._audit_segments(eligible, split_dir, output, 120.0, 0.0)
    assert not output.exists()


def test_cli_uses_the_same_eligible_pool_without_sampling(
    split_dir: Path, eligible: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate, seizure, duration and repair exclusions precede uncapped enumeration."""
    excluded = [
        eligible.iloc[0].to_dict() | {"subject": f"excluded{index}", "path": tmp_path / f"excluded{index}.edf"} | change
        for index, change in enumerate(
            [{"sfreq": 400}, {"sfreq": 512}, {"sfreq": 1000}, {"n_seizure": 1}, {"duration": 119}, {}]
        )
    ]
    engine = SimpleNamespace(descriptions=pd.concat([eligible, pd.DataFrame(excluded)], ignore_index=True))
    monkeypatch.setattr(count_pool, "TUHEEGEpilepsy", Mock(return_value=engine))
    monkeypatch.setattr(count_pool, "_read_bads", lambda path: ((), (), Path(path).name == "excluded5.edf"))
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_pool.py",
            "--require_keep_labels",
            "--exclude_native_sfreqs",
            "400",
            "512",
            "1000",
            "--interpolate_bad_channels",
            "--audit_segments_dir",
            str(output),
            "--split_manifest_dir",
            str(split_dir),
            "--expected_recordings",
            "4",
            "--expected_subjects",
            "3",
        ],
    )

    count_pool.main()

    provenance = json.loads((output / "audit_provenance.json").read_text())
    assert (provenance["recordings"], provenance["subjects"], provenance["segments"]) == (4, 3, 7)
    assert "No Balancing or Caps" in (output / "audit_report.md").read_text()


@pytest.fixture
def fresh_cohort(tmp_path: Path) -> pd.DataFrame:
    """The reported 97/98 patient class counts, with unequal recording durations."""
    return pd.DataFrame(
        [
            {
                "subject": f"p{label}_{index:03d}",
                "path": tmp_path / f"p{label}_{index:03d}.edf",
                "epilepsy": label,
                "duration": 120 * (1 + index % 3),
                "sfreq": 250.0,
                "n_seizure": 0,
                "session": "s001",
                "year": 2020,
            }
            for label, size in ((0, 97), (1, 98))
            for index in range(size)
        ]
    )


def test_fresh_splits_depend_only_on_unique_patients_and_seed(fresh_cohort: pd.DataFrame) -> None:
    """Recording order, multiplicity and duration cannot change a patient's split."""
    assignments = count_pool._fresh_subject_splits(fresh_cohort, 42)
    reordered = fresh_cohort.sample(frac=1, random_state=7).assign(duration=36000)
    more_recordings = pd.concat([reordered, reordered.iloc[:10].assign(path="extra.edf")], ignore_index=True)
    assert assignments == count_pool._fresh_subject_splits(more_recordings, 42)
    assert assignments != count_pool._fresh_subject_splits(fresh_cohort, 43)
    assert set(assignments) == set(fresh_cohort["subject"])
    assert {split: sum(value[0] == split for value in assignments.values()) for split in ("train", "val", "test")} == {
        "train": 117,
        "val": 39,
        "test": 39,
    }
    for split, size in (("train", 117), ("val", 39), ("test", 39)):
        positives = sum(value == (split, 1) for value in assignments.values())
        assert abs(positives - size * 98 / 195) < 1


def test_fresh_splits_reject_conflicting_patient_labels(fresh_cohort: pd.DataFrame) -> None:
    """Repeated recordings cannot give a patient two diagnoses during stratification."""
    conflicting = pd.concat([fresh_cohort, fresh_cohort.iloc[:1].assign(epilepsy=1)], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting epilepsy labels"):
        count_pool._fresh_subject_splits(conflicting, 42)


def test_fresh_cli_saves_patient_splits_and_every_complete_segment(
    fresh_cohort: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh mode needs no reference manifests and retains all 388 synthetic segments."""
    engine = SimpleNamespace(descriptions=fresh_cohort)
    monkeypatch.setattr(count_pool, "TUHEEGEpilepsy", Mock(return_value=engine))
    monkeypatch.setattr(count_pool, "_read_bads", lambda path: ((), (), False))
    output = tmp_path / "fresh_audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_pool.py",
            "--require_keep_labels",
            "--exclude_native_sfreqs",
            "400",
            "512",
            "1000",
            "--interpolate_bad_channels",
            "--audit_segments_dir",
            str(output),
            "--fresh_splits",
            "--split_seed",
            "42",
            "--expected_recordings",
            "195",
            "--expected_subjects",
            "195",
        ],
    )

    count_pool.main()

    patients = pd.read_csv(output / "patient_splits.csv").set_index("subject")
    assert len(patients) == patients.index.nunique() == 195
    assert patients["split"].value_counts().to_dict() == {"train": 117, "val": 39, "test": 39}
    total_segments = 0
    for split in ("train", "val", "test"):
        windows = pd.read_csv(output / f"windows_{split}.csv")
        total_segments += len(windows)
        expected_patients = set(patients.index[patients["split"] == split])
        assert set(windows["subject"]) == expected_patients
        assert (windows["end"] - windows["start"] == 120).all()
        assert windows["epilepsy"].equals(windows["subject"].map(patients["epilepsy"]))
    assert total_segments == 388
    provenance = json.loads((output / "audit_provenance.json").read_text())
    assert provenance["patient_split"]["mode"] == "fresh_stratified"
    assert provenance["patient_split"]["seed"] == 42
    assert provenance["source_manifests"] == {}
    assert (
        provenance["output_sha256"]["patient_splits.csv"]
        == hashlib.sha256((output / "patient_splits.csv").read_bytes()).hexdigest()
    )
    assert "Fresh stratified patient splits: 60/20/20, seed 42" in (output / "audit_report.md").read_text()
