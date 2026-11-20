"""Events corpus protocol: do dictionary waveforms mark the annotated events of the TUH EEG Events Corpus?

Protocol: ``documentation/event_corpus_dictionary_protocol.md``. ``src/event_dictionary_core.py`` holds the units,
the folds and the rules. This file holds stage 0, the inventory, which the user asked for before the later stages
are fixed. Subcommand:

- ``inventory`` (CPU, minutes) reads the header of every file (duration, rate, channel names) and every ``.rec``
  annotation file of ``edf/train`` and ``edf/eval``. It reads no signal. It writes ``files.csv`` (one row per file),
  ``patients.csv`` (the training patients with their classes and fold), ``folds.json`` and ``event_inventory.json``
  (the counts per set and class, the lengths of the annotated intervals, the marked seconds, the records of
  5 minutes and the training crops of 10 s per class, the folds, and the floor for each class).

The evaluation set is for one final read (protocol section 11). The inventory reports its headers and annotation
counts only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.event_dictionary_core import (  # noqa: E402
    CLASSES,
    ELECTRODES,
    EPILEPTIFORM,
    FOLD_SEED,
    MIN_FOLD_PATIENTS,
    MIN_PATIENTS,
    N_FOLDS,
    REASONS,
    assign_folds,
    class_eligible,
    covered_seconds,
    crop_starts,
    interval_statistics,
    mixed_seconds,
    parse_rec,
    pieces,
    units,
)

PROTOCOL = "documentation/event_corpus_dictionary_protocol.md"
SETS = ("train", "eval")
LABELS = tuple(CLASSES.values())
# AAREADME, v2.0.1: the files of each set, and the files that hold each class.
# The columns of files.csv after the set, the group and the file; a file with no header or no annotation has nulls.
COLUMNS = (
    "header_ok",
    "duration_s",
    "sfreq",
    "reference",
    "electrodes_ok",
    "pieces",
    "piece_seconds",
    "has_rec",
    "covered_s",
    "rows",
    "one_second",
    "whole_seconds",
    "shortest",
    "longest",
    "overlaps_same_class",
    "overlaps_other_class",
    "marked",
    "two_classes",
    "two_event_classes",
    "beyond_file",
    "marked_pieces",
    "event_pieces",
    "event_crops",
    *(f"{kind}_{label}" for label in LABELS for kind in ("intervals", "units", "kept", "seconds", "pieces", "crops")),
    *REASONS[1:],
)
README = {
    "train": {"files": 359, "spsw": 27, "gped": 51, "pled": 48, "eyem": 46, "artf": 164, "bckg": 211},
    "eval": {"files": 159, "spsw": 9, "gped": 28, "pled": 33, "eyem": 35, "artf": 46, "bckg": 89},
}


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def read_header(path: Path) -> dict[str, Any]:
    """The duration, the rate and the channel names of an EDF file. No signal is read."""
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    rate = float(raw.info["sfreq"])
    return {"duration": raw.n_times / rate, "sfreq": rate, "channels": list(raw.ch_names)}


def electrode_of(channel: str) -> str:
    """The electrode of a channel name of the corpus, such as ``FP1`` from ``EEG FP1-REF`` or ``EEG FP1-LE``."""
    name = channel.upper().strip()
    name = name[4:] if name.startswith("EEG ") else name
    return name.split("-")[0].strip()


def reference_of(channels: list[str]) -> str:
    """``LE`` (linked ears), ``AR`` (average reference) or ``unknown``, from the channel names."""
    names = [c.upper().strip() for c in channels if c.upper().strip().startswith("EEG ")]
    if any(name.endswith("-LE") for name in names):
        return "LE"
    return "AR" if any(name.endswith("-REF") for name in names) else "unknown"


def list_files(data_dir: Path, name: str) -> list[Path]:
    """The EDF files of a set, sorted.

    Raises
    ------
    FileNotFoundError
        If the set has no folder.
    """
    folder = data_dir / "edf" / name
    if not folder.is_dir():
        raise FileNotFoundError(f"no folder {folder}: set the corpus root (the folder that holds edf/)")
    return sorted(folder.rglob("*.edf"))


def file_row(path: Path, name: str, header_reader: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    """The inventory row of one file: its header, its annotation rows and its units."""
    row: dict[str, Any] = {"set": name, "group": path.parent.name, "file": path.name} | dict.fromkeys(COLUMNS)
    row |= {"header_ok": False, "has_rec": False, "electrodes_ok": False}
    try:
        header = header_reader(path)
    except Exception as error:  # noqa: BLE001 - a broken header is counted, and the inventory goes on
        logger.error(f"header of {path} not read: {error}")
        return row
    electrodes = {electrode_of(c) for c in header["channels"]}
    duration = float(header["duration"])
    row |= {
        "header_ok": True,
        "duration_s": duration,
        "sfreq": float(header["sfreq"]),
        "reference": reference_of(header["channels"]),
        "electrodes_ok": all(e in electrodes for e in ELECTRODES),
        "pieces": len(pieces(duration)),
        "piece_seconds": sum(stop - start for start, stop in pieces(duration)),
    }
    annotation = path.with_suffix(".rec")
    if not annotation.is_file():
        return row
    rows = parse_rec(annotation.read_text())
    file_units = units(rows, duration)
    statistics = interval_statistics(rows)
    row |= {"has_rec": True, "covered_s": covered_seconds(rows), **statistics, **mixed_seconds(file_units)}
    row |= {"beyond_file": sum(stop > duration + 1e-3 for _, _, stop, _ in rows)}
    for label in LABELS:
        mine = [u for u in file_units if u["label"] == label]
        row[f"intervals_{label}"] = sum(r[3] == label for r in rows)
        row[f"units_{label}"] = len(mine)
        kept = [u for u in mine if u["status"] == "kept"]
        row[f"kept_{label}"] = len(kept)
        seconds = {int(u["start"] + 1e-6) for u in kept}  # distinct seconds with a kept unit
        row[f"seconds_{label}"] = len(seconds)
        row[f"crops_{label}"] = len(crop_starts(sorted(seconds), duration))
        row[f"pieces_{label}"] = len({u["piece"] for u in kept})
    kept = [u for u in file_units if u["status"] == "kept"]
    row["marked_pieces"] = len({u["piece"] for u in kept})
    row["event_pieces"] = len({u["piece"] for u in kept if u["label"] != "bckg"})
    event_seconds = {int(u["start"] + 1e-6) for u in kept if u["label"] != "bckg"}
    row["event_crops"] = len(crop_starts(sorted(event_seconds), duration))
    for reason in REASONS[1:]:
        row[reason] = sum(u["status"] == reason for u in file_units)
    return row


def set_summary(frame: pl.DataFrame, name: str) -> dict[str, Any]:
    """The counts of one set: files, groups, time, rates, references, intervals and the units of each class."""
    read = frame.filter(pl.col("header_ok"))
    marked = read.filter(pl.col("has_rec"))
    per_class = {}
    for label in LABELS:
        with_class = marked.filter(pl.col(f"units_{label}") > 0)
        per_class[label] = {
            "files": with_class.height,
            "groups": with_class["group"].n_unique(),
            "intervals": int(marked[f"intervals_{label}"].sum()),
            "units": int(marked[f"units_{label}"].sum()),
            "kept_units": int(marked[f"kept_{label}"].sum()),
            "kept_seconds": int(marked[f"seconds_{label}"].sum()),
            "pieces": int(marked[f"pieces_{label}"].sum()),
            "crops": int(marked[f"crops_{label}"].sum()),
            "readme_files": README[name][label],
        }
    rows = int(marked["rows"].sum())
    return {
        "files": frame.height,
        "files_in_readme": README[name]["files"],
        "headers_not_read": frame.height - read.height,
        "files_without_rec": read.height - marked.height,
        "groups": frame["group"].n_unique(),
        "hours": float(read["duration_s"].sum() / 3600),
        "hours_in_pieces": float(read["piece_seconds"].sum() / 3600),
        "hours_marked": float(marked["covered_s"].sum() / 3600),
        "median_file_minutes": float(read["duration_s"].median() / 60) if read.height else None,
        "shortest_file_s": float(read["duration_s"].min()) if read.height else None,
        "rates": dict(sorted(Counter(read["sfreq"].to_list()).items())),
        "references": dict(Counter(read["reference"].to_list())),
        "files_without_an_electrode": read.height - int(read["electrodes_ok"].sum()),
        "annotation_rows": rows,
        "rows_of_one_second": int(marked["one_second"].sum()),
        "rows_of_whole_seconds": int(marked["whole_seconds"].sum()),
        "shortest_row_s": float(marked["shortest"].min()) if marked.height else None,
        "longest_row_s": float(marked["longest"].max()) if marked.height else None,
        "overlaps_same_class": int(marked["overlaps_same_class"].sum()),
        "overlaps_other_class": int(marked["overlaps_other_class"].sum()),
        "rows_beyond_the_file": int(marked["beyond_file"].sum()),
        "marked_seconds": int(marked["marked"].sum()),
        "seconds_with_two_classes": int(marked["two_classes"].sum()),
        "seconds_with_two_event_classes": int(marked["two_event_classes"].sum()),
        "units_that_leave": {reason: int(marked[reason].sum()) for reason in REASONS[1:]},
        "pieces": int(read["pieces"].sum()),
        "pieces_with_a_kept_unit": int(marked["marked_pieces"].sum()),
        "pieces_with_a_kept_event_unit": int(marked["event_pieces"].sum()),
        "event_crops": int(marked["event_crops"].sum()),
        "classes": per_class,
        "reproduces_readme": frame.height == README[name]["files"]
        and all(per_class[label]["files"] == README[name][label] for label in LABELS),
    }


def patient_table(frame: pl.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The training patients with their kept units per class, and their folds. A patient's classes for the folds
    are those with at least one kept unit."""
    marked = frame.filter(pl.col("header_ok") & pl.col("has_rec"))
    kept = marked.group_by("group").agg(
        pl.len().alias("files"),
        (pl.col("duration_s").sum() / 60).alias("minutes"),
        *(pl.col(f"kept_{label}").sum() for label in LABELS),
        *(pl.col(f"seconds_{label}").sum() for label in LABELS),
        *(pl.col(f"crops_{label}").sum() for label in LABELS),
        pl.col("event_pieces").sum(),
        pl.col("event_crops").sum(),
    )
    classes_of = {
        row["group"]: {label for label in LABELS if row[f"kept_{label}"] > 0} for row in kept.iter_rows(named=True)
    }
    classes_of = {patient: labels for patient, labels in classes_of.items() if labels}
    folds = assign_folds(classes_of)
    rows = [
        {"patient": row["group"], "fold": folds[row["group"]], **{k: v for k, v in row.items() if k != "group"}}
        for row in kept.sort("group").iter_rows(named=True)
        if row["group"] in folds
    ]
    return rows, folds


def fold_summary(patients: list[dict[str, Any]]) -> dict[str, Any]:
    """Per class: the patients and kept units of each fold, and whether the class meets the floor."""
    out = {}
    for label in LABELS:
        mine = [p for p in patients if p[f"kept_{label}"] > 0]
        per_fold = [sum(p["fold"] == fold for p in mine) for fold in range(N_FOLDS)]
        out[label] = {
            "patients": len(mine),
            "patients_per_fold": per_fold,
            "kept_units_per_fold": [
                int(sum(p[f"kept_{label}"] for p in mine if p["fold"] == fold)) for fold in range(N_FOLDS)
            ],
            "kept_seconds_per_fold": [
                int(sum(p[f"seconds_{label}"] for p in mine if p["fold"] == fold)) for fold in range(N_FOLDS)
            ],
            "crops_per_fold": [
                int(sum(p[f"crops_{label}"] for p in mine if p["fold"] == fold)) for fold in range(N_FOLDS)
            ],
            "median_units_per_patient": float(np.median([p[f"kept_{label}"] for p in mine])) if mine else None,
            "median_seconds_per_patient": float(np.median([p[f"seconds_{label}"] for p in mine])) if mine else None,
            "patients_with_background": sum(p["kept_bckg"] > 0 for p in mine),
            "eligible": class_eligible(len(mine), per_fold),
            "decides": label in EPILEPTIFORM,
        }
    return out


def inventory(
    data_dir: Path, root: Path, header_reader: Callable[[Path], dict[str, Any]] = read_header
) -> dict[str, Any]:
    """Stage 0: the files, the annotations, the units and the folds of the corpus. No signal is read."""
    rows = []
    for name in SETS:
        paths = list_files(data_dir, name)
        logger.info(f"{name}: {len(paths)} files")
        rows += [file_row(path, name, header_reader) for path in paths]
    frame = pl.DataFrame(rows, infer_schema_length=None)
    patients, folds = patient_table(frame.filter(pl.col("set") == "train"))
    record = {
        "protocol": f"{PROTOCOL}, section 8",
        "corpus": str(data_dir),
        "note": "headers and annotation files only: no signal was read; units on a repaired channel are not counted",
        "floor": {"patients": MIN_PATIENTS, "patients_per_fold": MIN_FOLD_PATIENTS},
        "folds": {"count": N_FOLDS, "seed": FOLD_SEED, "patients": len(folds)},
        "sets": {name: set_summary(frame.filter(pl.col("set") == name), name) for name in SETS},
        "training_classes": fold_summary(patients),
    }
    root.mkdir(parents=True, exist_ok=True)
    frame.write_csv(root / "files.csv")
    pl.DataFrame(patients, infer_schema_length=None).write_csv(root / "patients.csv")
    _save_json(root / "folds.json", {"seed": FOLD_SEED, "folds": folds})
    _save_json(root / "event_inventory.json", record)
    eligible = [label for label, entry in record["training_classes"].items() if entry["eligible"]]
    logger.info(f"inventory: {frame.height} files, {len(folds)} training patients, eligible classes {eligible}")
    return record


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("inventory", help="files, annotations, units and folds; no signal")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--events-dir", type=Path, required=True, help="the corpus root: the folder that holds edf/")
    args = parser.parse_args()
    inventory(args.events_dir.resolve(), args.root.resolve())


if __name__ == "__main__":
    main()
