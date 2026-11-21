"""Pure logic of the Events corpus protocol: the annotations, the pieces, the units and their exclusions, the folds
and the floor for a class.

Protocol: ``documentation/event_corpus_dictionary_protocol.md``. ``src/event_dictionary.py`` holds the code that reads
the files. This module holds stage 0 (the inventory); the later stages are added after their confirmation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# The codes of the ``.rec`` files (AAREADME of the TUH EEG Events Corpus, v2.0.1).
CLASSES = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
EPILEPTIFORM = ("spsw", "gped", "pled")
# The order in which a patient with several classes is assigned for the folds: the rarest class of interest first.
PRIORITY = ("spsw", "pled", "gped", "eyem", "artf", "bckg")
# The channel indexes of the annotations: the TCP bipolar montage in the order of the README.
MONTAGE = (
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)
EAR_CHANNELS = (8, 13)  # A1-T3 and T4-A2: not in the 20-channel montage of this project
ELECTRODES = tuple(sorted({e for i, name in enumerate(MONTAGE) if i not in EAR_CHANNELS for e in name.split("-")}))
# A piece is one record of the scenario: 5 minutes (protocol section 3). A last piece of at least 1 minute is kept.
PIECE_S, MIN_PIECE_S, WINDOW_S, EDGE_S = 300.0, 60.0, 2.0, 0.5
CROP_S = 10.0  # a training crop around a mark: 5 windows of 2 s
CROP_SHIFTS_S = (-4.0, -2.0, 0.0, 2.0, 4.0)
N_FOLDS, FOLD_SEED = 5, 0
MIN_PATIENTS, MIN_FOLD_PATIENTS = 15, 2
REASONS = ("kept", "ear_channel", "piece_edge", "outside_pieces", "overlap", "class_conflict")
TOLERANCE = 1e-6


def parse_rec(text: str) -> list[tuple[int, float, float, str]]:
    """The rows of a ``.rec`` file: channel index, start and stop in seconds, and the class.

    Raises
    ------
    ValueError
        If a row does not have four fields, or holds an unknown channel or code, or stops before it starts.
    """
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"row {number} does not have four fields: {line!r}")
        channel, start, stop, code = int(fields[0]), float(fields[1]), float(fields[2]), int(float(fields[3]))
        if not 0 <= channel < len(MONTAGE) or code not in CLASSES or stop <= start:
            raise ValueError(f"row {number} holds a bad channel, code or interval: {line!r}")
        rows.append((channel, start, stop, CLASSES[code]))
    return rows


def pieces(duration: float) -> list[tuple[float, float]]:
    """The records of a file: 5 minutes each, and a last one of at least 1 minute cut to whole windows of 2 s."""
    out = [(i * PIECE_S, (i + 1) * PIECE_S) for i in range(int((duration + TOLERANCE) // PIECE_S))]
    start = len(out) * PIECE_S
    rest = math.floor((duration - start + TOLERANCE) / WINDOW_S) * WINDOW_S
    if rest >= MIN_PIECE_S:
        out.append((start, start + rest))
    return out


def crop_plan(seconds: Sequence[int], duration: float) -> list[tuple[float, int]]:
    """The training crops of 10 s for marked seconds, on the grid of 2 s: the start of each crop and the marked
    second that it was made for. The seconds are taken in order; a second that an earlier crop holds gets no crop of
    its own, so the crops of a record do not repeat a mark. A crop is centered on its second where the record allows
    it."""
    last_start = math.floor((duration - CROP_S + TOLERANCE) / WINDOW_S) * WINDOW_S
    if last_start < 0:
        return []
    plan: list[tuple[float, int]] = []
    for second in sorted(set(seconds)):
        if plan and second + 1.0 <= plan[-1][0] + CROP_S + TOLERANCE:
            continue
        start = math.floor((second - (CROP_S - 1.0) / 2) / WINDOW_S) * WINDOW_S
        plan.append((float(min(max(start, 0.0), last_start)), int(second)))
    return plan


def crop_starts(seconds: Sequence[int], duration: float) -> list[float]:
    """The starts of the training crops of ``crop_plan``."""
    return [start for start, _ in crop_plan(seconds, duration)]


def crop_shifts(start: float, second: int, duration: float) -> list[float]:
    """The starts of a crop and of its shifted copies (2 and 4 s to each side, the user's decision of 2026-11-20). A
    copy is kept if it lies in the record and still holds the two seconds from the marked second on, so that a unit
    that starts inside that second stays in the crop."""
    out = []
    for shift in CROP_SHIFTS_S:
        moved = start + shift
        if (
            moved >= -TOLERANCE
            and moved + CROP_S <= duration + TOLERANCE
            and moved <= second
            and second + 2 <= moved + CROP_S
        ):
            out.append(float(moved))
    return out


def unit_status(channel: int, start: float, file_pieces: Sequence[tuple[float, float]]) -> str:
    """Whether the unit ``[start, start + 1)`` on a channel enters the read, or why it leaves (protocol section 3)."""
    if channel in EAR_CHANNELS:
        return "ear_channel"
    stop = start + 1.0
    for first, last in file_pieces:
        if start >= first - TOLERANCE and stop <= last + TOLERANCE:
            inside = start >= first + EDGE_S - TOLERANCE and stop <= last - EDGE_S + TOLERANCE
            return "kept" if inside else "piece_edge"
    # In no piece: across the border of two pieces, or in the part of the file that no piece holds.
    held = any(start < last and stop > first for first, last in file_pieces)
    return "piece_edge" if held else "outside_pieces"


def piece_of(start: float, file_pieces: Sequence[tuple[float, float]]) -> int | None:
    """The index of the piece that holds the unit ``[start, start + 1)`` in full; ``None`` if no piece does."""
    for index, (first, last) in enumerate(file_pieces):
        if start >= first - TOLERANCE and start + 1.0 <= last + TOLERANCE:
            return index
    return None


def units(rows: Sequence[tuple[int, float, float, str]], duration: float) -> list[dict[str, Any]]:
    """One unit per whole second of every annotated interval: channel, start, class, piece and status.

    The annotations hold rows that overlap on one channel (inventory of 2026-11-20). On a channel, the units are
    taken in the order of their start. A unit that overlaps a kept unit of the same class leaves as ``overlap``, so
    the kept units of a channel do not share signal. If the two classes differ, both leave as ``class_conflict``.
    """
    file_pieces = pieces(duration)
    out = []
    for channel, start, stop, label in rows:
        for i in range(int(math.floor(stop - start + TOLERANCE))):
            first = start + i
            out.append(
                {
                    "channel": channel,
                    "start": first,
                    "label": label,
                    "piece": piece_of(first, file_pieces),
                    "status": unit_status(channel, first, file_pieces),
                }
            )
    last: dict[int, dict[str, Any]] = {}
    for unit in sorted(out, key=lambda u: (u["channel"], u["start"])):
        if unit["status"] != "kept":
            continue
        previous = last.get(unit["channel"])
        if previous is not None and unit["start"] < previous["start"] + 1.0 - TOLERANCE:
            if unit["label"] == previous["label"]:
                unit["status"] = "overlap"
            else:
                unit["status"] = previous["status"] = "class_conflict"
            continue
        last[unit["channel"]] = unit
    return out


def interval_statistics(rows: Sequence[tuple[int, float, float, str]]) -> dict[str, Any]:
    """What the unit depends on: the lengths of the intervals, and rows that overlap on one channel."""
    lengths = np.array([stop - start for _, start, stop, _ in rows], dtype=np.float64)
    same, other = 0, 0
    by_channel: dict[int, list[tuple[float, float, str]]] = {}
    for channel, start, stop, label in rows:
        by_channel.setdefault(channel, []).append((start, stop, label))
    for items in by_channel.values():
        items.sort()
        for (_, stop_a, label_a), (start_b, _, label_b) in zip(items, items[1:], strict=False):
            if start_b < stop_a - TOLERANCE:
                same += label_a == label_b
                other += label_a != label_b
    return {
        "rows": len(rows),
        "one_second": int((np.abs(lengths - 1.0) < 1e-3).sum()),
        "whole_seconds": int((np.abs(lengths - np.round(lengths)) < 1e-3).sum()),
        "shortest": float(lengths.min()) if len(lengths) else None,
        "longest": float(lengths.max()) if len(lengths) else None,
        "overlaps_same_class": int(same),
        "overlaps_other_class": int(other),
    }


def covered_seconds(rows: Sequence[tuple[int, float, float, str]]) -> float:
    """The time of a file with an annotation on any channel: the length of the union of the intervals."""
    total, end = 0.0, -math.inf
    for start, stop in sorted((start, stop) for _, start, stop, _ in rows):
        if start > end:
            total, end = total + stop - start, stop
        elif stop > end:
            total, end = total + stop - end, stop
    return total


def mixed_seconds(file_units: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Seconds (by the floor of the unit start) with a mark, with two classes, and with two classes that are not
    background, on different channels."""
    by_second: dict[int, set[str]] = {}
    for unit in file_units:
        by_second.setdefault(int(math.floor(unit["start"] + TOLERANCE)), set()).add(unit["label"])
    return {
        "marked": len(by_second),
        "two_classes": sum(len(labels) > 1 for labels in by_second.values()),
        "two_event_classes": sum(len(labels - {"bckg"}) > 1 for labels in by_second.values()),
    }


def assign_folds(classes_of: Mapping[str, set[str]], n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    """Patients to folds. Each patient goes to the group of its first class in ``PRIORITY``; the groups are shuffled
    and dealt in turn, so each fold gets patients of each class as evenly as the counts allow.

    Raises
    ------
    ValueError
        If a patient has no class.
    """
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    turn = 0
    for label in PRIORITY:
        group = sorted(p for p, labels in classes_of.items() if p not in folds and label in labels)
        for index in rng.permutation(len(group)):
            folds[group[index]] = turn % n_folds
            turn += 1
    missing = sorted(set(classes_of) - set(folds))
    if missing:
        raise ValueError(f"patients with no class: {missing[:5]}")
    return folds


def class_eligible(patients: int, per_fold: Sequence[int]) -> bool:
    """The floor for a class (protocol section 8): 15 training patients, and at least 2 in every fold."""
    return patients >= MIN_PATIENTS and min(per_fold) >= MIN_FOLD_PATIENTS
