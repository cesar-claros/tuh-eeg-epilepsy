"""Within-patient drug protocol: when a patient's drug list changes between sessions, does the EEG of that patient
change in the direction that the literature reports?

Protocol: ``documentation/drug_signature_detection_protocol.md``. ``src/drug_signature_core.py`` holds the sides, the
eligibility rules, the measures, the tests and the rule. A root holds one run. Subcommands, in order:

- ``keys`` (HPC; reads no signal) writes the patient, recording, start, session and native rate of every segment of
  the 156 patients, in the order of the features of the first readout (``segment_keys.csv``).
- ``inventory`` (no EEG, no score) writes the drugs and the side of every session (``session_drugs.csv``), the
  patients of every contrast (``contrast_patients.csv``, ``control_patients.csv``) and the counts
  (``drug_inventory.json``). Without ``segment_keys.csv`` in the root it still runs, and the floor of 4 segments on
  each side is then not checked: the counts are a preview by sessions, and the record says so.
- ``features`` (HPC; one pass over the stored signals, no training) writes the band power and the alpha frequency
  of every segment (``segment_features.npz``).
- ``gates`` checks gates 2 to 6 of section 9 (the launcher runs the tests, gate 1) and writes ``gates.json``.
- ``test`` applies the paired tests and the rule of sections 8 and 10 (``drug_signature_decision.json`` and
  ``drug_signature_summary.csv``). The test against time is the stratified one of amendment A1 (section 20.3); the
  test as section 8 wrote it is kept as a descriptive read. If a gate failed, it writes the outcome ``blocked`` and
  opens no result.

The 39 test patients stay locked: every step keeps only the patients of the readout plan.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.covariate_audit import listed_text, load_drug_classes, load_mapping, read_sessions  # noqa: E402
from src.covariate_audit_core import drugs_named, medication_status, names_any, session_of  # noqa: E402
from src.drug_signature_core import (  # noqa: E402
    ALPHA,
    BAND_NAMES,
    CONTRASTS,
    DECIDING,
    MEASURES,
    MIN_PATIENTS,
    MIN_SEGMENTS,
    MIN_TIME_GROUP,
    N_BANDS,
    POSTERIOR,
    alpha_frequency,
    band_cube,
    contrast_eligible,
    control_entry,
    control_stratum,
    drug_groups,
    measure_values,
    passes_change,
    passes_time,
    patient_difference,
    patient_entry,
    session_side,
    sign_flip_test,
    signature_outcome,
    stratified_time_test,
    stratum,
    time_test,
)

CONFIGS = ROOT / "configs" / "covariate_audit"
PROTOCOL = "documentation/drug_signature_detection_protocol.md"
AUDIT_COUNTS = {1: {"on": 66, "off": 10, "unknown": 3}, 0: {"on": 4, "off": 58, "unknown": 15}}  # gate 2
PREDICTION = (
    "no_effect_measured about 35%, too_few_patients 25%, within_patient_effect 20%, time_or_drug 15%, blocked 5% "
    "(protocol section 10)"
)
FEATURES = "segment_features.npz"
NPERSEG = 512  # pieces of 2 s at 256 Hz, as the band-power features
FEATURE_TOLERANCE = 1e-4  # gate 3
SHIFT_TOLERANCE = 0.1  # gate 5, in Hz
# Descriptive variants of two contrasts (section 8): the base contrast and how the side of a session changes.
VARIANTS = {"W-sodium-no-lamotrigine": "W-sodium", "W-sodium-wide": "W-sodium", "W-gaba-wide": "W-gaba"}


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def load_mechanisms(path: Path, classes: dict[str, Any]) -> dict[str, Any]:
    """The mechanism groups of protocol section 4, which must cover the named drugs of the drug classes exactly.

    Raises
    ------
    ValueError
        If the groups and the named drugs differ.
    """
    mechanisms = json.loads(path.read_text())
    named = set(classes["older"]) | set(classes["newer"])
    if set(drug_groups(mechanisms["groups"])) != named:
        raise ValueError("the mechanism groups do not cover the named maintenance drugs exactly")
    return mechanisms


def montage_of(path: str) -> str:
    """The recording montage of an EDF path, such as ``tcp_ar`` from ``.../01_tcp_ar/...``; ``unknown`` if absent."""
    found = re.search(r"/\d{2}_(tcp_[a-z_]+)/", path)
    return found.group(1) if found else "unknown"


def plan_sessions(audit_root: Path, subjects: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """The sessions of each patient in the campaign's recording plan: year, recordings and montages."""
    frame = pl.read_csv(audit_root / "plan_recordings.csv", schema_overrides={"subject": pl.Utf8})
    sessions: dict[str, dict[str, dict[str, Any]]] = {}
    for subject, path, session, year in frame.select("subject", "path", "session", "year").iter_rows():
        if subject not in subjects:
            continue
        entry = sessions.setdefault(subject, {}).setdefault(
            session, {"year": int(year), "recordings": 0, "montages": set()}
        )
        entry["recordings"] += 1
        entry["montages"].add(montage_of(path))
    return sessions


def segment_counts(root: Path, subjects: set[str]) -> dict[tuple[str, str], dict[str, Any]] | None:
    """The segments and the native rates of each session, from ``segment_keys.csv``; ``None`` without that file."""
    path = root / "segment_keys.csv"
    if not path.is_file():
        return None
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8})
    counts: dict[tuple[str, str], dict[str, Any]] = {}
    for subject, session, rate in frame.select("subject", "session", "sfreq").iter_rows():
        if subject in subjects:
            entry = counts.setdefault((subject, session), {"segments": 0, "rates": set()})
            entry["segments"] += 1
            entry["rates"].add(float(rate))
    return counts


def session_rows(
    sessions: dict[str, dict[str, dict[str, Any]]],
    texts: dict[tuple[str, str | None], str],
    labels: dict[str, int],
    mapping: dict[str, Any],
    classes: dict[str, Any],
    mechanisms: dict[str, Any],
    counts: dict[tuple[str, str], dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """One row per session of the recording plan: the drugs, the groups and the side for every contrast."""
    spellings = classes["older"] | classes["newer"]
    owner = drug_groups(mechanisms["groups"])
    rows = []
    for subject in sorted(sessions):
        for order, session in enumerate(sorted(sessions[subject])):
            entry = sessions[subject][session]
            text = texts.get((subject, session))
            listed = bool(text and text.strip())
            drugs = drugs_named(text, spellings) if listed else set()
            groups = {owner[drug] for drug in drugs}
            maintenance = listed and names_any(text, mapping["narrow_terms"])
            unnamed = listed and names_any(text, classes["unnamed"])
            counted = counts.get((subject, session), {"segments": 0, "rates": set()}) if counts is not None else None
            rows.append(
                {
                    "subject": subject,
                    "label": labels[subject],
                    "session": session,
                    "order": order,
                    "year": entry["year"],
                    "recordings": entry["recordings"],
                    "segments": counted["segments"] if counted is not None else None,
                    "montages": "+".join(sorted(entry["montages"])),
                    "rates": "+".join(f"{r:g}" for r in sorted(counted["rates"])) if counted is not None else None,
                    "in_sheet": text is not None,
                    "listed": listed,
                    "maintenance": maintenance,
                    "unnamed": unnamed,
                    "drugs": "+".join(sorted(drugs)),
                    "groups": "+".join(sorted(groups)),
                    **{name: session_side(name, listed, maintenance, unnamed, groups) for name in CONTRASTS},
                }
            )
    return rows


def contrast_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The patients with both sides of each contrast, and the control patients of each contrast."""
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    patients, controls = [], []
    for name in CONTRASTS:
        for subject, items in by_subject.items():
            sides = [item[name] for item in items]
            segments = [item["segments"] for item in items]
            entry = patient_entry(
                sides,
                segments,
                [item["order"] for item in items],
                [item["year"] for item in items],
                [(item["montages"], item["rates"]) for item in items],
            )
            if entry is not None:
                patients.append({"contrast": name, "subject": subject, "label": items[0]["label"], **entry})
            control = control_entry(sides, segments)
            if control is not None:
                controls.append(
                    {
                        "contrast": name,
                        "subject": subject,
                        "label": items[0]["label"],
                        "side": control["side"],
                        "sessions": len(control["earlier"]) + len(control["later"]),
                        "segments_earlier": control["segments_earlier"],
                        "segments_later": control["segments_later"],
                        "eligible": control["eligible"],
                    }
                )
    return patients, controls


def _count(items: list[dict[str, Any]], **where: Any) -> int:
    return sum(all(item[key] == value for key, value in where.items()) for item in items)


def contrast_summary(
    name: str, patients: list[dict[str, Any]], controls: list[dict[str, Any]], counted: bool
) -> dict[str, Any]:
    """The counts of one contrast (section 6). Without segment counts, ``eligible`` holds the patients with both
    sides, and the contrast is marked as a preview."""
    both = [p for p in patients if p["contrast"] == name]
    eligible = [p for p in both if p["eligible"]] if counted else both
    usable = [c for c in controls if c["contrast"] == name and (c["eligible"] if counted else True)]
    gaps = [p["gap_years"] for p in eligible]
    return {
        "patients_with_both_sides": len(both),
        "eligible_patients": len(eligible),
        "eligible_epilepsy": _count(eligible, label=1),
        "eligible_non_epilepsy": _count(eligible, label=0),
        "added": _count(eligible, direction="added"),
        "removed": _count(eligible, direction="removed"),
        "consistent_report": _count(eligible, consistent=True),
        "same_acquisition": _count(eligible, same_acquisition=True),
        "median_gap_years": float(np.median(gaps)) if gaps else None,
        "control_patients": len(usable),
        "contrast_eligible": contrast_eligible(len(eligible)),
        "time_test_can_run": _count(eligible, direction="added") >= MIN_TIME_GROUP and len(usable) >= MIN_TIME_GROUP,
        "deciding": name in DECIDING,
    }


def between_patients(
    texts: dict[tuple[str, str | None], str],
    read: dict[str, list[str]],
    status: dict[str, str],
    labels: dict[str, int],
    classes: dict[str, Any],
    mechanisms: dict[str, Any],
) -> dict[str, Any]:
    """The counts of the between-patient read of section 15, at the 8 segments that every method gets."""
    spellings = classes["older"] | classes["newer"]
    owner = drug_groups(mechanisms["groups"])
    treated = sorted(s for s in read if labels[s] == 1 and status[s] == "on")
    drugs = {
        s: set().union(*(drugs_named(texts.get((s, session), ""), spellings) for session in read[s])) for s in treated
    }
    groups = {s: {owner[d] for d in drugs[s]} for s in treated}
    per_drug: dict[str, int] = {}
    for named in drugs.values():
        for drug in named:
            per_drug[drug] = per_drug.get(drug, 0) + 1
    return {
        "treated_epilepsy_patients": len(treated),
        "patients_per_drug": dict(sorted(per_drug.items(), key=lambda item: -item[1])),
        "patients_per_group": {g: sum(g in groups[s] for s in treated) for g in mechanisms["groups"]},
        "T-sodium": [sum("sodium" in groups[s] for s in treated), sum("sodium" not in groups[s] for s in treated)],
        "T-gaba": [sum("gaba" in groups[s] for s in treated), sum("gaba" not in groups[s] for s in treated)],
        "T-lev-only": [
            sum(drugs[s] == {"levetiracetam"} for s in treated),
            sum(drugs[s] != {"levetiracetam"} for s in treated),
        ],
    }


def inventory(
    root: Path,
    readout: Path,
    audit_root: Path,
    metadata: Path,
    mapping_path: Path = CONFIGS / "medication_mapping.json",
    classes_path: Path = CONFIGS / "drug_classes.json",
    mechanisms_path: Path = CONFIGS / "drug_mechanisms.json",
) -> dict[str, Any]:
    """Stage 1: the drugs and the side of every session, the patients of every contrast and the counts.

    Raises
    ------
    ValueError
        If a patient of the readout plan has no session in the recording plan.
    """
    labels = {str(s): int(v) for s, v in json.loads((readout / "folds.json").read_text())["labels"].items()}
    mapping = load_mapping(mapping_path)
    classes = load_drug_classes(classes_path, mapping)
    mechanisms = load_mechanisms(mechanisms_path, classes)
    sessions = plan_sessions(audit_root, set(labels))
    if set(sessions) != set(labels):
        raise ValueError("the recording plan and the readout plan hold different patients")
    texts = listed_text(metadata, mapping, {subject: sorted(values) for subject, values in sessions.items()})
    counts = segment_counts(root, set(labels))
    rows = session_rows(sessions, texts, labels, mapping, classes, mechanisms, counts)
    patients, controls = contrast_rows(rows)
    read = {s: v for s, v in read_sessions(audit_root).items() if s in labels}
    status = {
        s: medication_status([texts.get((s, session), "") for session in v], mapping["narrow_terms"], "unknown")
        for s, v in read.items()
    }
    table = {
        c: {k: sum(status[s] == k and labels[s] == c for s in status) for k in ("on", "off", "unknown")} for c in (1, 0)
    }
    record = {
        "protocol": f"{PROTOCOL}, section 6",
        "preview": counts is None,
        "note": (
            "segment counts are known: the floor of 4 segments on each side is applied"
            if counts is not None
            else "no segment_keys.csv: the counts are by sessions, and the floor of 4 segments is not checked"
        ),
        "floors": {"segments_per_side": MIN_SEGMENTS, "patients": MIN_PATIENTS, "time_test_group": MIN_TIME_GROUP},
        "patients": len(labels),
        "sessions": len(rows),
        "sessions_in_sheet": _count(rows, in_sheet=True),
        "sessions_with_a_list": _count(rows, listed=True),
        "sessions_by_side": {
            name: {side: sum(r[name] == side for r in rows) for side in ("with", "without", "unknown")}
            for name in CONTRASTS
        },
        "status_at_read_sessions": {"epilepsy": table[1], "non_epilepsy": table[0]},
        "counts_reproduce": table == AUDIT_COUNTS,
        "contrasts": {name: contrast_summary(name, patients, controls, counts is not None) for name in CONTRASTS},
        "between_patients": between_patients(texts, read, status, labels, classes, mechanisms),
    }
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(root / "session_drugs.csv")
    pl.DataFrame(patients, infer_schema_length=None).write_csv(root / "contrast_patients.csv")
    pl.DataFrame(controls, infer_schema_length=None).write_csv(root / "control_patients.csv")
    _save_json(root / "drug_inventory.json", record)
    sizes = {name: record["contrasts"][name]["eligible_patients"] for name in CONTRASTS}
    logger.info(f"inventory: {'preview by sessions' if counts is None else 'with segment counts'}; eligible {sizes}")
    return record


def write_keys(args: argparse.Namespace) -> None:
    """The key of every segment of the 156 patients, in the order of the features of the first readout: patient by
    patient, and by recording and start inside a patient. The data module is built and no signal is read.

    Raises
    ------
    ValueError
        If the patients or labels differ from those of the readout root.
    """
    from src.frozen_readout import build_cohort, load_sources

    root, readout = args.root.resolve(), args.readout_root.resolve()
    labels = json.loads((readout / "folds.json").read_text())["labels"]
    _, _, sources, locked, _ = load_sources(args)
    cohort = build_cohort(sources, locked)
    if cohort.labels != labels:
        raise ValueError("the patients or labels differ from those of the readout root")
    rates = {name: [float(d["sfreq"]) for d in source.plan["description_row"]] for name, source in sources.items()}
    rows = []
    for subject in sorted(cohort.labels):
        for (path, start), (name, row) in zip(cohort.keys[subject], cohort.rows[subject], strict=True):
            session, year = session_of(path)
            rows.append(
                {
                    "subject": subject,
                    "path": path,
                    "start": start,
                    "session": session,
                    "year": year,
                    "sfreq": rates[name][row],
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(root / "segment_keys.csv")
    logger.info(f"segment keys: {len(rows)} segments of {len(cohort.labels)} patients")


# ---------------------------------------------------------------------- stage 2: the features


def posterior_index(channels: Sequence[str]) -> list[int]:
    """The positions of the posterior channels of the alpha frequency measure.

    Raises
    ------
    ValueError
        If a posterior channel is absent.
    """
    names = [str(c).lower() for c in channels]
    absent = [c for c in POSTERIOR if c.lower() not in names]
    if absent:
        raise ValueError(f"the montage has no channels {absent}")
    return [names.index(c.lower()) for c in POSTERIOR]


def posterior_spectrum(
    x: Any, index: Sequence[int], sfreq: float, nperseg: int = NPERSEG
) -> tuple[np.ndarray, np.ndarray]:
    """The Welch spectrum of each segment ``(segments, channels, samples)``, averaged over the posterior channels,
    with its frequencies. The same pieces, window and overlap as the band-power features."""
    import torch

    rows = x[:, list(index)].float().flatten(0, 1)
    spectrum = torch.stft(
        rows,
        n_fft=nperseg,
        hop_length=nperseg // 2,
        win_length=nperseg,
        window=torch.hann_window(nperseg, device=rows.device),
        center=False,
        return_complex=True,
    )
    psd = spectrum.abs().pow(2).mean(-1).view(x.shape[0], len(index), -1).mean(1)
    return psd.cpu().numpy().astype(np.float64), np.arange(psd.shape[1]) * sfreq / nperseg


def alpha_check(sfreq: float = 256.0, seconds: float = 30.0) -> dict[str, Any]:
    """Gate 5: on synthetic signals with a known peak at 8, 9, 10 and 11 Hz in noise, the alpha frequency follows
    the peak, and a shift of 0.5 Hz is recovered within ``SHIFT_TOLERANCE``."""
    import torch

    generator = torch.Generator().manual_seed(0)
    t = torch.arange(int(seconds * sfreq)) / sfreq

    def measure(peak: float) -> float:
        x = 5.0 * torch.sin(2 * torch.pi * peak * t) + torch.randn(1, 4, len(t), generator=generator)
        psd, freqs = posterior_spectrum(x, range(4), sfreq)
        return float(alpha_frequency(psd, freqs)[0])

    peaks = (8.0, 9.0, 10.0, 11.0)
    found = [measure(peak) for peak in peaks]
    shifts = [measure(peak + 0.5) - value for peak, value in zip(peaks, found, strict=True)]
    error = max(abs(shift - 0.5) for shift in shifts)
    follows = all(b - a > 0.5 for a, b in zip(found, found[1:], strict=False))
    return {"found": found, "largest_shift_error": error, "pass": bool(follows and error <= SHIFT_TOLERANCE)}


def write_features(args: argparse.Namespace) -> None:
    """Stage 2: the band power and the alpha frequency of every segment, in the order of ``segment_keys.csv``. One
    pass over the stored signals with the low-pass of the first readout; nothing is trained.

    Raises
    ------
    ValueError
        If the segments differ from ``segment_keys.csv``.
    """
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.frozen_readout import EXTRACT_BATCH, BandLimited, PooledSegments, build_cohort, every_row, load_sources
    from src.utils import instantiate_feature

    root = args.root.resolve()
    path = root / FEATURES
    if path.is_file():
        logger.info(f"{path.name}: finished")
        return
    keys = pl.read_csv(root / "segment_keys.csv", schema_overrides={"subject": pl.Utf8})
    cfg, _, sources, locked, _ = load_sources(args)
    sfreq = float(cfg.data.target_sfreq)
    cohort = build_cohort(sources, locked)
    view = BandLimited(PooledSegments(sources, every_row(cohort, sorted(cohort.labels))), sfreq)
    subjects = view.plan["subject"].astype(str).tolist()
    if subjects != keys["subject"].to_list():
        raise ValueError("the segments differ from segment_keys.csv: write the keys again")
    extractor = instantiate_feature(OmegaConf.load(ROOT / "configs" / "feature" / "spectral.yaml"))
    index = posterior_index(view.target_channels)
    band, alpha = [], []
    with torch.no_grad():
        for x, y in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
            band.append(extractor(x, y).detach().cpu().float().numpy())
            alpha.append(alpha_frequency(*posterior_spectrum(x, index, sfreq)))
    np.savez(path, subject=np.array(subjects), features=np.concatenate(band), alpha_frequency=np.concatenate(alpha))
    logger.info(f"{path.name}: {len(subjects)} segments")


# ---------------------------------------------------------------------- gates


def _sorted_rows(x: np.ndarray) -> np.ndarray:
    return x[np.lexsort(x.T[::-1])]


def gates(root: Path, readout: Path) -> dict[str, Any]:
    """Gates 2 to 6 of section 9, on the inventory, the keys and the features; writes ``gates.json``."""
    record_in = json.loads((root / "drug_inventory.json").read_text())
    labels = json.loads((readout / "folds.json").read_text())["labels"]
    keys = pl.read_csv(root / "segment_keys.csv", schema_overrides={"subject": pl.Utf8})
    ours = np.load(root / FEATURES)
    saved = np.load(readout / "features" / "Spectral-all.npz")
    fixed = np.load(readout / "features" / "Spectral.npz")
    subject = keys["subject"].to_numpy().astype(str)
    aligned = len(ours["features"]) == len(subject) == len(saved["features"]) and bool(
        (ours["subject"].astype(str) == subject).all() and (saved["subject"].astype(str) == subject).all()
    )
    gap_all = float(np.abs(ours["features"] - saved["features"]).max()) if aligned else float("inf")
    gap_read = 0.0
    if aligned:
        chosen = saved["in_nine"].astype(bool)
        for name in np.unique(fixed["subject"].astype(str)):
            a = _sorted_rows(ours["features"][chosen & (subject == name)].astype(np.float64))
            b = _sorted_rows(fixed["features"][fixed["subject"].astype(str) == name].astype(np.float64))
            gap_read = max(gap_read, float(np.abs(a - b).max()) if a.shape == b.shape else float("inf"))
    sessions = pl.read_csv(root / "session_drugs.csv", schema_overrides={"subject": pl.Utf8})
    counted = {(s, k): n for s, k, n in sessions.select("subject", "session", "segments").iter_rows()}
    found: dict[tuple[str, str], int] = {}
    for s, k in keys.select("subject", "session").iter_rows():
        found[(s, k)] = found.get((s, k), 0) + 1
    check = alpha_check()
    passed = {
        "inventory_has_segments": not record_in["preview"],
        "counts_reproduce": bool(record_in["counts_reproduce"]),
        "features_reproduce": aligned and gap_all <= FEATURE_TOLERANCE and gap_read <= FEATURE_TOLERANCE,
        "keys_complete": keys.null_count().sum_horizontal().item() == 0
        and all(counted.get(key) == n for key, n in found.items()),
        "alpha_frequency_check": check["pass"],
        "no_test_patient": set(subject.tolist()) == set(labels),
    }
    record = {
        "status": "pass" if all(passed.values()) else "blocked",
        "gates": passed,
        "feature_gap_all_segments": gap_all,
        "feature_gap_read_segments": gap_read,
        "alpha_check": check,
        "segments": len(subject),
        "segments_without_alpha_frequency": int(np.isnan(ours["alpha_frequency"]).sum()),
    }
    _save_json(root / "gates.json", record)
    logger.info(f"gates: {record['status']} {passed}")
    return record


# ---------------------------------------------------------------------- stage 3: the paired test


def variant_side(name: str, row: dict[str, Any], mechanisms: dict[str, Any]) -> str:
    """The side of a session for a descriptive variant of section 8."""
    base = row[VARIANTS[name]]
    drugs = set(filter(None, (row["drugs"] or "").split("+")))
    if name == "W-sodium-no-lamotrigine":
        return "unknown" if "lamotrigine" in drugs else base
    group = "sodium" if name == "W-sodium-wide" else "gaba"
    return "with" if base != "unknown" and drugs & set(mechanisms["wide"][group]) else base


def contrast_sessions(
    rows: list[dict[str, Any]], side_of: Callable[[dict[str, Any]], str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """For one contrast: the eligible patients with the side of each of their sessions, and the eligible control
    patients with ``with`` for the sessions of the later half and ``without`` for the earlier half. Each entry holds
    its stratum for the test against time (amendment A1)."""
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    patients: dict[str, dict[str, Any]] = {}
    controls: dict[str, dict[str, Any]] = {}
    for subject, items in by_subject.items():
        items = sorted(items, key=lambda item: item["order"])
        sides = [side_of(item) for item in items]
        segments = [item["segments"] for item in items]
        years = [item["year"] for item in items]
        acquisitions = [(item["montages"], item["rates"]) for item in items]
        entry = patient_entry(sides, segments, [item["order"] for item in items], years, acquisitions)
        if entry is not None and entry["eligible"]:
            patients[subject] = entry | {
                "sides": {item["session"]: side for item, side in zip(items, sides, strict=True)},
                "stratum": stratum(entry["gap_years"], entry["same_acquisition"]),
            }
        control = control_entry(sides, segments)
        if control is not None and control["eligible"]:
            halves = {items[i]["session"]: "without" for i in control["earlier"]}
            controls[subject] = {
                "sides": halves | {items[i]["session"]: "with" for i in control["later"]},
                "stratum": control_stratum(control, years, acquisitions),
            }
    return patients, controls


def differences(
    values: np.ndarray, subject: np.ndarray, session: np.ndarray, sides: dict[str, dict[str, str]]
) -> dict[str, float]:
    """The difference of each patient for one measure; a patient with no finite value on a side is left out."""
    out: dict[str, float] = {}
    for name, by_session in sides.items():
        mine = subject == name
        side = np.array([by_session.get(k, "unknown") for k in session[mine]])
        usable = np.isfinite(values[mine])
        try:
            value = patient_difference(values[mine][usable], session[mine][usable], side[usable])
        except ValueError:
            continue
        if np.isfinite(value):
            out[name] = value
    return out


def _two_sided(d: np.ndarray) -> dict[str, Any]:
    up, down = sign_flip_test(d, 1), sign_flip_test(d, -1)
    return up | {"p": min(1.0, 2 * min(up["p"], down["p"])), "sided": "two"}


def all_measures(features: np.ndarray, alpha: np.ndarray) -> dict[str, np.ndarray]:
    """The four measures and the 10 channel-mean band-power numbers, one value per segment."""
    cube = band_cube(features).mean(axis=1)
    numbers = {f"B-absolute-{b}": cube[:, i] for i, b in enumerate(BAND_NAMES)}
    numbers |= {f"B-relative-{b}": cube[:, N_BANDS + i] for i, b in enumerate(BAND_NAMES)}
    return measure_values(features, alpha) | numbers


def run_test(root: Path, mechanisms_path: Path = CONFIGS / "drug_mechanisms.json") -> dict[str, Any]:
    """Stage 3: the paired tests and the rule (sections 8 and 10); writes the decision and its table. If a gate
    failed, the record holds the outcome ``blocked`` and no result."""
    gate_record = json.loads((root / "gates.json").read_text())
    header = {"protocol": f"{PROTOCOL}, section 10", "prediction": PREDICTION, "gates": gate_record}
    if gate_record["status"] != "pass":
        record = {"outcome": signature_outcome(False, [], {}, {}), **header}
        _save_json(root / "drug_signature_decision.json", record)
        logger.info("drug signature: blocked; no result is opened")
        return record
    from scipy.stats import spearmanr

    mechanisms = json.loads(mechanisms_path.read_text())
    keys = pl.read_csv(root / "segment_keys.csv", schema_overrides={"subject": pl.Utf8})
    subject, session = keys["subject"].to_numpy().astype(str), keys["session"].to_numpy().astype(str)
    stored = np.load(root / FEATURES)
    measures = all_measures(stored["features"], stored["alpha_frequency"])
    rows = pl.read_csv(root / "session_drugs.csv", schema_overrides={"subject": pl.Utf8}).to_dicts()
    inventory_record = json.loads((root / "drug_inventory.json").read_text())

    sides: dict[str, Callable[[dict[str, Any]], str]] = {name: (lambda row, n=name: row[n]) for name in CONTRASTS}
    sides |= {name: (lambda row, n=name: variant_side(n, row, mechanisms)) for name in VARIANTS}
    groups = {name: contrast_sessions(rows, side_of) for name, side_of in sides.items()}

    def paired(name: str, measure: str, keep: Callable[[dict[str, Any]], bool] | None = None) -> np.ndarray:
        patients = groups[name][0]
        chosen = {s: e["sides"] for s, e in patients.items() if keep is None or keep(e)}
        return np.array(list(differences(measures[measure], subject, session, chosen).values()))

    table: list[dict[str, Any]] = []

    def add(name: str, measure: str, kind: str, d: np.ndarray, expected: int | None) -> dict[str, Any] | None:
        if len(d) < 2:
            return None
        result = sign_flip_test(d, expected) | {"sided": "one"} if expected else _two_sided(d)
        table.append({"contrast": name, "measure": measure, "read": kind, "expected": expected, **result})
        return result

    eligible = [n for n in DECIDING if inventory_record["contrasts"][n]["contrast_eligible"]]
    deciding: dict[str, Any] = {}
    change, time = {}, {}
    for name, (measure, expected) in DECIDING.items():
        patients, controls = groups[name]
        d = paired(name, measure)
        result = add(name, measure, "deciding", d, expected)
        chosen = {s: e["sides"] for s, e in patients.items() if e["direction"] == "added"}
        added = differences(measures[measure], subject, session, chosen)
        tau = differences(measures[measure], subject, session, {s: e["sides"] for s, e in controls.items()})
        against_time = stratified_time_test(
            list(added.values()),
            [patients[s]["stratum"] for s in added],
            list(tau.values()),
            [controls[s]["stratum"] for s in tau],
            expected,
        )
        change[name] = bool(name in eligible and result is not None and passes_change(result["p"], len(eligible)))
        time[name] = passes_time(against_time)
        gaps = [e["gap_years"] for s, e in patients.items()]
        deciding[name] = {
            "measure": measure,
            "expected": expected,
            "eligible": name in eligible,
            "patients": len(patients),
            "against_no_change": result,
            "passes_no_change": change[name],
            "against_time": against_time,
            "passes_time": time[name],
            "against_time_as_written": time_test(list(added.values()), list(tau.values()), expected),
            "gap_rank_correlation": (
                float(spearmanr(gaps, expected * paired(name, measure)).statistic)
                if len(d) == len(gaps) and len(d) >= 5 and len(set(gaps)) > 1
                else None
            ),
        }
        for kind, keep in (
            ("same acquisition", lambda e: e["same_acquisition"]),
            ("consistent report", lambda e: e["consistent"]),
            ("added", lambda e: e["direction"] == "added"),
            ("removed", lambda e: e["direction"] == "removed"),
        ):
            add(name, measure, kind, paired(name, measure, keep), expected)
    for name in sides:
        for measure in measures:
            if name in DECIDING and DECIDING[name][0] == measure:
                continue
            add(name, measure, "descriptive", paired(name, measure), None)
        if name in VARIANTS:
            measure, expected = DECIDING[VARIANTS[name]]
            add(name, measure, "variant", paired(name, measure), expected)
    outcome = signature_outcome(True, eligible, change, time)
    record = {
        "outcome": outcome,
        **header,
        "eligible_deciding_contrasts": eligible,
        "level_per_contrast": ALPHA / len(eligible) if eligible else None,
        "deciding": deciding,
        "measures": list(MEASURES),
        "summary_rows": len(table),
    }
    pl.DataFrame(table, infer_schema_length=None).write_csv(root / "drug_signature_summary.csv")
    _save_json(root / "drug_signature_decision.json", record)
    logger.info(f"drug signature: {outcome} (no change {change}, time {time})")
    return record


def _corpus(command: argparse.ArgumentParser) -> None:
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
    command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
    command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
    command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
    command.add_argument("--num-workers", type=int, default=4)


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    _corpus(commands.add_parser("keys", help="the key of every segment (HPC, no signal)"))
    command = commands.add_parser("inventory", help="the drugs of every session and the size of every contrast")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
    command.add_argument("--audit-root", type=Path, required=True, help="the covariate audit root")
    command.add_argument("--metadata", type=Path, required=True, help="the metadata spreadsheet of the corpus")
    _corpus(commands.add_parser("features", help="band power and alpha frequency of every segment (HPC)"))
    command = commands.add_parser("gates", help="gates 2 to 6")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True)
    command = commands.add_parser("test", help="the paired tests and the rule")
    command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "keys":
        write_keys(args)
    elif args.command == "inventory":
        inventory(root, args.readout_root.resolve(), args.audit_root.resolve(), args.metadata.resolve())
    elif args.command == "features":
        write_features(args)
    elif args.command == "gates":
        gates(root, args.readout_root.resolve())
    else:
        run_test(root)


if __name__ == "__main__":
    main()
