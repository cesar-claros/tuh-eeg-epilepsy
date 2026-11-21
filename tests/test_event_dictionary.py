"""Checks of stage 0 of the Events corpus protocol (no corpus; CPU; seconds).

Runs under pytest. Covered: the annotation parser and its refusals; the records of a file; the training crops; the
status of a unit (kept,
ear channel, edge of a piece, outside the pieces); the units of an interval; the interval statistics, the covered
time and the seconds with two classes; the folds (every patient once, each class spread over the folds, the same
result for the same seed) and the floor for a class; the channel names of the corpus; and the inventory on a written
tree with a header reader that reads no file: the counts per set and class, a file with no annotation, a header that
cannot be read, the patients and the folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.event_dictionary as driver  # noqa: E402
import src.event_dictionary_core as core  # noqa: E402


def test_constants() -> None:
    assert core.CLASSES == {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
    assert len(core.MONTAGE) == 22 and core.MONTAGE[13] == "T4-A2" and core.MONTAGE[8] == "A1-T3"
    assert core.MONTAGE[0] == "FP1-F7" and core.MONTAGE[21] == "P4-O2"
    assert len(core.ELECTRODES) == 17 and "A1" not in core.ELECTRODES and "CZ" in core.ELECTRODES
    assert set(core.PRIORITY) == set(core.CLASSES.values()) and core.PRIORITY[0] == "spsw"


def test_parse_rec() -> None:
    rows = core.parse_rec("13,90.4,91.4,6\n\n0, 5.0, 8.0, 1\n")
    assert rows == [(13, 90.4, 91.4, "bckg"), (0, 5.0, 8.0, "spsw")]
    assert core.parse_rec("") == []
    for bad in ("1,2,3", "22,0,1,6", "0,0,1,7", "0,2,1,6", "-1,0,1,6"):
        with pytest.raises(ValueError):
            core.parse_rec(bad)


def test_pieces() -> None:
    assert core.pieces(650.0) == [(0.0, 300.0), (300.0, 600.0)]  # the rest of 50 s is below 1 minute
    assert core.pieces(661.7) == [(0.0, 300.0), (300.0, 600.0), (600.0, 660.0)]  # cut to whole windows of 2 s
    assert core.pieces(59.0) == [] and core.pieces(300.0) == [(0.0, 300.0)] and core.pieces(250.0) == [(0.0, 250.0)]


def test_crop_starts() -> None:
    assert core.crop_starts([10, 11, 12, 30, 31, 100], 250.0) == [4.0, 24.0, 94.0]  # one crop holds 10, 11 and 12
    assert core.crop_starts([12, 10, 11, 10], 250.0) == [4.0]
    assert core.crop_starts([1], 250.0) == [0.0] and core.crop_starts([248], 250.0) == [240.0]  # kept in the file
    assert core.crop_starts([10, 14], 250.0) == [4.0, 8.0]  # 14 is not held by the crop from 4 to 14
    assert core.crop_starts([3], 8.0) == [] and core.crop_starts([], 250.0) == []
    for start, second in zip(core.crop_starts([10, 30, 100, 248], 250.0), (10, 30, 100, 248), strict=True):
        assert start % 2 == 0 and start <= second and second + 1 <= start + core.CROP_S


def test_crop_plan_and_shifts() -> None:
    assert core.crop_plan([10, 11, 12, 30], 250.0) == [(4.0, 10), (24.0, 30)]  # the crop and the second it is for
    assert core.crop_shifts(4.0, 10, 250.0) == [2.0, 4.0, 6.0, 8.0]  # 0.0 would not hold the second after second 10
    assert core.crop_shifts(0.0, 1, 250.0) == [0.0]  # at the start of the record: no copy to the left
    assert core.crop_shifts(240.0, 248, 250.0) == [240.0]  # at the end of the record
    for start in core.crop_shifts(24.0, 30, 250.0):
        assert start % 2 == 0 and start <= 30 and 32 <= start + core.CROP_S
    assert core.crop_shifts(24.0, 30, 250.0) == [22.0, 24.0, 26.0, 28.0]


def test_unit_status() -> None:
    held = core.pieces(661.7)
    assert core.unit_status(0, 50.2, held) == "kept"
    assert core.unit_status(8, 50.2, held) == "ear_channel" and core.unit_status(13, 50.2, held) == "ear_channel"
    assert core.unit_status(0, 0.2, held) == "piece_edge"  # within 0.5 s of the start of a record
    assert core.unit_status(0, 0.5, held) == "kept" and core.unit_status(0, 298.5, held) == "kept"
    assert core.unit_status(0, 298.7, held) == "piece_edge"
    assert core.unit_status(0, 299.5, held) == "piece_edge"  # across the border of two records
    assert core.unit_status(0, 659.5, held) == "piece_edge"  # across the end of the last record
    assert core.unit_status(0, 670.0, held) == "outside_pieces"
    assert core.unit_status(0, 620.0, core.pieces(650.0)) == "outside_pieces"  # in the rest that no record holds


def test_units_and_statistics() -> None:
    rows = [(0, 10.4, 13.4, "spsw"), (0, 12.0, 13.0, "bckg"), (1, 10.4, 11.4, "bckg"), (8, 10.0, 11.0, "eyem")]
    found = core.units(rows, 250.0)
    assert [u["label"] for u in found] == ["spsw", "spsw", "spsw", "bckg", "bckg", "eyem"]
    assert [round(u["start"], 1) for u in found[:3]] == [10.4, 11.4, 12.4]  # one unit per whole second
    # The background unit at 12.0 on channel 0 overlaps the spike unit at 11.4: two classes, so both leave.
    assert [u["status"] for u in found] == ["kept", "class_conflict", "kept", "class_conflict", "kept", "ear_channel"]
    assert [u["piece"] for u in found] == [0] * 6
    same = core.units([(0, 10.0, 11.0, "gped"), (0, 10.5, 11.5, "gped"), (0, 11.0, 12.0, "gped")], 250.0)
    assert [u["status"] for u in same] == ["kept", "overlap", "kept"]  # the third does not overlap the first
    assert core.piece_of(299.5, core.pieces(661.7)) is None and core.piece_of(330.0, core.pieces(661.7)) == 1
    assert core.units([(0, 5.0, 5.6, "spsw")], 250.0) == []  # shorter than one second: no unit
    statistics = core.interval_statistics(rows)
    assert (statistics["rows"], statistics["one_second"], statistics["whole_seconds"]) == (4, 3, 4)
    assert (statistics["shortest"], statistics["longest"]) == (pytest.approx(1.0), pytest.approx(3.0))
    assert (statistics["overlaps_same_class"], statistics["overlaps_other_class"]) == (0, 1)
    assert core.interval_statistics([(0, 1.0, 3.0, "bckg"), (0, 2.0, 4.0, "bckg")])["overlaps_same_class"] == 1
    assert core.covered_seconds(rows) == pytest.approx(3.4)  # the union of 10.0-11.0, 10.4-13.4 and the others
    assert core.covered_seconds([(0, 1.0, 2.0, "bckg"), (3, 5.0, 6.5, "bckg")]) == pytest.approx(2.5)
    mixed = core.mixed_seconds(found)
    assert mixed == {"marked": 3, "two_classes": 2, "two_event_classes": 1}  # second 10: spsw, bckg and eyem


def test_folds_and_floor() -> None:
    classes_of = {f"s{i:02d}": {"spsw", "bckg"} for i in range(10)}
    classes_of |= {f"g{i:02d}": {"gped"} for i in range(11)}
    classes_of |= {f"b{i:02d}": {"bckg"} for i in range(23)}
    folds = core.assign_folds(classes_of)
    assert set(folds) == set(classes_of) and set(folds.values()) == set(range(5))
    assert folds == core.assign_folds(classes_of) and folds != core.assign_folds(classes_of, seed=1)
    for prefix, low, high in (("s", 2, 2), ("g", 2, 3), ("b", 4, 5)):
        counts = [sum(f == fold for p, f in folds.items() if p.startswith(prefix)) for fold in range(5)]
        assert min(counts) >= low and max(counts) <= high  # each class is spread evenly
    sizes = [sum(f == fold for f in folds.values()) for fold in range(5)]
    assert max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError, match="no class"):
        core.assign_folds({"a": set()})
    assert core.class_eligible(15, [3, 3, 3, 3, 3]) and not core.class_eligible(14, [3, 3, 3, 3, 2])
    assert not core.class_eligible(20, [8, 5, 4, 2, 1])


def test_channel_names() -> None:
    assert driver.electrode_of("EEG FP1-REF") == "FP1" and driver.electrode_of("EEG T3-LE") == "T3"
    assert driver.electrode_of("eeg cz-ref") == "CZ" and driver.electrode_of("PHOTIC-REF") == "PHOTIC"
    assert driver.reference_of(["EEG FP1-REF", "EEG EKG1-REF"]) == "AR"
    assert driver.reference_of(["EEG FP1-LE", "PHOTIC PH"]) == "LE" and driver.reference_of(["X"]) == "unknown"


# A small corpus: the patient or session folder, the file, its duration, and its annotation text (None: no .rec file).
TREE = {
    "train": [
        ("00000001", "00000001_00000001.edf", 250.0, "0,10.4,13.4,1\n1,10.4,11.4,6\n8,10.0,11.0,4\n"),
        ("00000001", "00000001_00000002.edf", 130.0, "2,0.2,1.2,6\n3,50.0,51.0,3\n"),
        ("00000002", "00000002_00000001.edf", 60.0, "4,5.0,7.0,2\n4,30.0,31.0,6\n4,30.5,31.5,6\n"),
        ("00000003", "00000003_00000001.edf", 60.0, None),
        ("00000004", "00000004_00000001.edf", -1.0, "0,1.0,2.0,6\n"),  # its header cannot be read
        ("00000005", "00000005_00000001.edf", 61.0, "0,20.0,21.0,5\n0,70.0,71.0,6\n"),
    ],
    "eval": [("032", "bckg_032_a_.edf", 40.0, "5,3.0,4.0,6\n"), ("032", "bckg_032_a_1.edf", 40.0, "5,3.0,4.0,4\n")],
}
CHANNELS = [f"EEG {electrode}-REF" for electrode in core.ELECTRODES] + ["EEG A1-REF", "EEG EKG1-REF"]


def _tree(tmp: Path) -> tuple[Path, dict[str, float]]:
    durations = {}
    for name, files in TREE.items():
        for group, file, duration, text in files:
            folder = tmp / "edf" / name / group
            folder.mkdir(parents=True, exist_ok=True)
            (folder / file).write_bytes(b"")
            durations[file] = duration
            if text is not None:
                (folder / file).with_suffix(".rec").write_text(text)
    return tmp, durations


def _reader(durations: dict[str, float]):
    def read(path: Path) -> dict:
        if durations[path.name] < 0:
            raise OSError("bad header")
        channels = CHANNELS[1:] if path.name == "00000002_00000001.edf" else CHANNELS  # one file lacks an electrode
        return {"duration": durations[path.name], "sfreq": 250.0, "channels": channels}

    return read


def test_inventory(tmp_path: Path) -> None:
    data, durations = _tree(tmp_path / "corpus")
    root = tmp_path / "root"
    record = driver.inventory(data, root, _reader(durations))
    train, held = record["sets"]["train"], record["sets"]["eval"]
    assert (train["files"], train["headers_not_read"], train["files_without_rec"], train["groups"]) == (6, 1, 1, 5)
    assert train["hours"] == pytest.approx((250 + 130 + 60 + 60 + 61) / 3600)
    assert train["hours_in_pieces"] == pytest.approx((250 + 130 + 60 + 60 + 60) / 3600)
    assert train["rates"] == {250.0: 5} and train["references"] == {"AR": 5}
    assert train["files_without_an_electrode"] == 1 and not train["reproduces_readme"]
    assert (train["annotation_rows"], train["rows_of_one_second"], train["rows_of_whole_seconds"]) == (10, 8, 10)
    assert (train["shortest_row_s"], train["longest_row_s"]) == (pytest.approx(1.0), pytest.approx(3.0))
    assert train["rows_beyond_the_file"] == 1  # the row at 70 s of a file of 61 s
    assert train["units_that_leave"] == {
        "ear_channel": 1,
        "piece_edge": 1,
        "outside_pieces": 1,
        "overlap": 1,
        "class_conflict": 0,
    }
    assert (train["pieces"], train["pieces_with_a_kept_unit"], train["pieces_with_a_kept_event_unit"]) == (5, 4, 4)
    spikes = train["classes"]["spsw"]
    assert (spikes["files"], spikes["groups"], spikes["intervals"], spikes["units"], spikes["kept_units"]) == (
        1,
        1,
        1,
        3,
        3,
    )
    assert train["classes"]["bckg"]["units"] == 5 and train["classes"]["bckg"]["kept_units"] == 2
    assert train["classes"]["eyem"]["units"] == 1 and train["classes"]["eyem"]["kept_units"] == 0  # an ear channel
    assert train["classes"]["gped"]["units"] == 2 and spikes["readme_files"] == 27
    assert (spikes["kept_seconds"], spikes["pieces"]) == (3, 1) and train["classes"]["gped"]["kept_seconds"] == 2
    assert (spikes["crops"], train["classes"]["gped"]["crops"], train["event_crops"]) == (1, 1, 4)
    assert (train["marked_seconds"], train["seconds_with_two_classes"]) == (10, 1)
    assert (held["files"], held["groups"], held["classes"]["eyem"]["files"]) == (2, 1, 1)

    files = pl.read_csv(root / "files.csv", schema_overrides={"group": pl.Utf8})
    assert files.height == 8 and files.filter(~pl.col("header_ok")).height == 1
    assert files.filter(pl.col("file") == "00000003_00000001.edf")["has_rec"].to_list() == [False]
    first = files.filter(pl.col("file") == "00000001_00000001.edf").to_dicts()[0]
    assert (first["pieces"], first["kept_spsw"], first["ear_channel"]) == (1, 3, 1)
    assert (first["crops_spsw"], first["event_crops"]) == (1, 1)
    assert first["covered_s"] == pytest.approx(3.4)

    patients = pl.read_csv(root / "patients.csv", schema_overrides={"patient": pl.Utf8}).to_dicts()
    assert [p["patient"] for p in patients] == ["00000001", "00000002", "00000005"]  # patients with a kept unit
    assert patients[0]["files"] == 2 and patients[0]["kept_spsw"] == 3 and patients[0]["kept_pled"] == 1
    folds = json.loads((root / "folds.json").read_text())
    assert set(folds["folds"]) == {"00000001", "00000002", "00000005"} and folds["seed"] == core.FOLD_SEED
    classes = record["training_classes"]
    assert classes["spsw"]["patients"] == 1 and sum(classes["spsw"]["patients_per_fold"]) == 1
    assert not classes["spsw"]["eligible"] and classes["spsw"]["decides"] and not classes["artf"]["decides"]
    assert sum(classes["bckg"]["kept_units_per_fold"]) == 2 and sum(classes["bckg"]["kept_seconds_per_fold"]) == 2
    assert classes["spsw"]["patients_with_background"] == 1 and classes["spsw"]["median_seconds_per_patient"] == 3
    assert patients[0]["seconds_spsw"] == 3 and patients[0]["event_pieces"] == 2
    assert patients[0]["crops_spsw"] == 1 and patients[0]["event_crops"] == 2
    assert sum(classes["spsw"]["crops_per_fold"]) == 1
    assert json.loads((root / "event_inventory.json").read_text())["folds"]["patients"] == 3


def test_inventory_needs_the_corpus_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="holds edf"):
        driver.inventory(tmp_path, tmp_path / "root", lambda path: {})
