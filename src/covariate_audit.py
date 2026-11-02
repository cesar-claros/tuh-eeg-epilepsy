"""Covariate audit: does the EEG score of the frozen readout hold on pairs matched for age and sex?

Protocol: ``documentation/covariate_audit_protocol.md``. Run on the HPC from ``code/``: CPU only, and it reads no
signal. A root holds one audit of one readout root. ``src/covariate_audit_core.py`` holds the pure logic and the rule.
Subcommands, in order:

- ``inventory`` (step 0) fixes the read segments and writes the columns of the metadata spreadsheet, with no label, no
  score and no value of age or sex.
- ``extract`` reads age and sex from the EDF headers and writes ``covariates.csv``. It reports only counts.
- ``read`` (steps 1 and 2) applies the gates and the controls, then the rule, the medication read and the descriptive
  reads. It needs the mapping of the addendum, and it opens no EEG score before the gates and the controls pass.

The 39 test patients stay locked: every patient outside the readout's folds is dropped before any statistic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import rootutils
from loguru import logger
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.covariate_audit_core import (  # noqa: E402
    AGE_BANDS,
    CHANCE,
    EXCLUSION_BOUND,
    MISSING_SEX,
    SENSITIVITY_CALIPERS,
    MatchedRead,
    age_validity,
    all_pairs,
    audit_outcome,
    clean_age,
    clean_sex,
    control_reads,
    excludes_above_bound,
    gates_pass,
    group_pairs,
    header_gates,
    inventory_sheet,
    is_flagged,
    is_missing,
    matched_difference,
    matched_read,
    medication_read,
    medication_status,
    pair_mask,
    patient_age,
    patient_age_validity,
    patient_sex,
    ratio_pairs,
    read_age_fit,
    repaired_ages,
    residualize,
    second_level_scores,
    session_of,
    session_token,
    standardize_within_fold,
    subject_token,
)
from src.frozen_readout_core import (  # noqa: E402
    N_BOOTSTRAP,
    N_FOLDS,
    ArmScores,
    fit_head,
    mean_fold_auroc,
    paired_interval,
    segment_draw,
)

METADATA_FILE = "v3.0.0/DOCS/metadata_v00r.xlsx"
PRIMARY, REFIT, COMPARISON = "Spectral", "Spectral-refit", "HYDRA"
DESCRIPTIVE_ARMS = ("S4-R1", "S4-proj")
SPECTRAL_CHANNELS = 20
SPECTRAL_BANDS = ("1 to 4", "4 to 8", "8 to 13", "13 to 30", "30 to 45")  # as configs/feature/spectral.yaml
SPECTRAL_KINDS = ("absolute", "relative")
MAPPING_KEYS = ("subject_column", "session_column", "medication_columns", "antiseizure_terms", "empty_means")


def load_plan(readout: Path) -> dict[str, Any]:
    """The fold plan of the readout root (``folds.json``)."""
    return json.loads((readout / "folds.json").read_text())


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def metadata_path(args: argparse.Namespace) -> Path:
    """The spreadsheet of the corpus, or the path that the command names."""
    return args.metadata.resolve() if args.metadata else args.data_dir.resolve() / METADATA_FILE


def read_sheets(path: Path, header_row: int = 0) -> dict[str, dict[str, list[Any]]]:
    """Every sheet of the spreadsheet as columns of plain cells."""
    sheets = pd.read_excel(path, sheet_name=None, header=header_row, dtype=object)
    return {str(name): {str(c): frame[c].tolist() for c in frame.columns} for name, frame in sheets.items()}


# ---------------------------------------------------------------------- step 0


def write_read_plan(root: Path, keys: dict[str, list[tuple[str, float]]], n_segments: int) -> None:
    """Fix ``read_segments.csv`` (the segments of the fixed read) and ``plan_recordings.csv``; no header is read.

    ``keys`` holds the sorted ``(path, start)`` of each patient, and the draw is the one of the frozen readout.
    """
    read, recordings = [], []
    for subject in sorted(keys):
        items = keys[subject]
        for position in segment_draw(subject, items, n_segments):
            path, start = items[position]
            session, year = session_of(path)
            read.append({"subject": subject, "path": path, "start": start, "session": session, "year": year})
        for path in sorted({item[0] for item in items}):
            session, year = session_of(path)
            recordings.append({"subject": subject, "path": path, "session": session, "year": year})
    pl.DataFrame(read).write_csv(root / "read_segments.csv")
    pl.DataFrame(recordings).write_csv(root / "plan_recordings.csv")


def read_sessions(root: Path) -> dict[str, list[str]]:
    """The sessions of the read segments of each patient."""
    frame = pl.read_csv(root / "read_segments.csv", schema_overrides={"subject": pl.Utf8})
    sessions: dict[str, set[str]] = {}
    for subject, session in zip(frame["subject"].to_list(), frame["session"].to_list()):
        sessions.setdefault(subject, set()).add(session)
    return {subject: sorted(values) for subject, values in sessions.items()}


def write_inventory(
    root: Path,
    metadata: Path,
    subjects: set[str],
    header_row: int = 0,
    subject_column: str | None = None,
    session_column: str | None = None,
) -> dict[str, Any]:
    """Write ``inventory.json``: one report per sheet, on the rows of ``subjects`` only (section 5).

    A named column applies to the sheets that hold it; the other sheets look for their own join.
    """
    wanted = {(s, session) for s, sessions in read_sessions(root).items() for session in sessions}
    sheets = read_sheets(metadata, header_row)
    report = {
        "metadata": str(metadata),
        "patients": len(subjects),
        "read_sessions": len(wanted),
        "sheets": [
            inventory_sheet(
                name,
                columns,
                subjects,
                wanted,
                subject_column if subject_column in columns else None,
                session_column if session_column in columns else None,
            )
            for name, columns in sheets.items()
        ],
    }
    _save_json(root / "inventory.json", report)
    return report


def run_inventory(args: argparse.Namespace) -> None:
    """Step 0: the read plan and the inventory. The data module is built to reuse the cohort, and reads no signal.

    Raises
    ------
    ValueError
        If the patients or labels differ from those of the readout root.
    """
    from src.frozen_readout import build_cohort, load_sources

    root, readout = args.root.resolve(), args.readout_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = load_plan(readout)
    _, _, sources, locked, _ = load_sources(args)
    cohort = build_cohort(sources, locked)
    if cohort.labels != plan["labels"]:
        raise ValueError("the patients or labels differ from those of the readout root")
    write_read_plan(root, cohort.keys, plan["n_segments"])
    report = write_inventory(
        root, metadata_path(args), set(plan["labels"]), args.header_row, args.subject_column, args.session_column
    )
    joined = {sheet["sheet"]: sheet.get("read_sessions_joined") for sheet in report["sheets"]}
    logger.info(f"Inventory written; read sessions joined per sheet: {joined}")


# ---------------------------------------------------------------------- extract


def _header(path: str, parse: Callable[[str], tuple[int, str]]) -> tuple[float, str]:
    try:
        age, sex = parse(path)
    except UnicodeDecodeError:
        return float("nan"), MISSING_SEX
    return clean_age(age), clean_sex(sex)


def write_covariates(root: Path, parse: Callable[[str], tuple[int, str]]) -> None:
    """Write ``covariates.csv`` (per patient, over the read segments) and ``session_ages.csv`` (for the validity gate).

    Age and sex describe what the EEG read saw, so they come from the patient's read segments (section 4).
    """
    overrides = {"subject": pl.Utf8}
    read = pl.read_csv(root / "read_segments.csv", schema_overrides=overrides)
    recordings = pl.read_csv(root / "plan_recordings.csv", schema_overrides=overrides)
    paths = set(read["path"].to_list()) | set(recordings["path"].to_list())
    headers = {path: _header(path, parse) for path in sorted(paths)}
    files = Counter(zip(recordings["subject"].to_list(), recordings["session"].to_list()))
    rows = []
    for subject, group in read.group_by("subject", maintain_order=True):
        name = str(subject[0])
        values = [headers[p] for p in group["path"].to_list()]
        sessions = sorted(set(group["session"].to_list()))
        rows.append(
            {
                "subject": name,
                "age": patient_age([v[0] for v in values]),
                "sex": patient_sex([v[1] for v in values]),
                "read_sessions": len(sessions),
                "read_year": float(group["year"].mean()),
                "files_per_read_session": float(np.mean([files[(name, session)] for session in sessions])),
            }
        )
    pl.DataFrame(rows).write_csv(root / "covariates.csv")
    session_rows = []
    for key, group in recordings.group_by(["subject", "session", "year"], maintain_order=True):
        age = patient_age([headers[p][0] for p in group["path"].to_list()])
        session_rows.append({"subject": str(key[0]), "session": str(key[1]), "year": int(key[2]), "age": age})
    pl.DataFrame(session_rows).write_csv(root / "session_ages.csv")
    missing_age = sum(np.isnan(row["age"]) for row in rows)
    missing_sex = sum(row["sex"] == MISSING_SEX for row in rows)
    logger.info(f"Covariates of {len(rows)} patients written: age missing {missing_age}, sex missing {missing_sex}")


def run_extract(args: argparse.Namespace) -> None:
    """Read the headers named by the read plan of this root."""
    from src.data.components.utils import parse_age_and_gender_from_edf_header

    write_covariates(args.root.resolve(), parse_age_and_gender_from_edf_header)


# ---------------------------------------------------------------------- read


@dataclass
class Patients:
    """The readout's patients in sorted order, with their folds, labels and covariates."""

    subject: np.ndarray
    fold: np.ndarray
    label: np.ndarray
    age: np.ndarray
    sex: np.ndarray
    segments: np.ndarray
    rate: np.ndarray
    files_per_session: np.ndarray
    read_year: np.ndarray

    @property
    def complete(self) -> np.ndarray:
        """Patients with both covariates."""
        return np.isfinite(self.age) & (self.sex != MISSING_SEX)


def load_patients(root: Path, readout: Path, plan: dict[str, Any]) -> Patients:
    """Align the covariates with the fold plan.

    Raises
    ------
    ValueError
        If the covariates do not hold exactly the patients of the plan.
    """
    overrides = {"subject": pl.Utf8, "sex": pl.Utf8}
    covariates = pl.read_csv(root / "covariates.csv", schema_overrides=overrides).sort("subject")
    covariates = covariates.with_columns(pl.col("sex").fill_null(MISSING_SEX))
    subject = np.array(covariates["subject"].to_list())
    if set(subject) != set(plan["labels"]):
        raise ValueError("the covariates do not hold the patients of the plan")
    patients = pl.read_csv(readout / "patients.csv", schema_overrides={"subject": pl.Utf8}).sort("subject")
    if patients["subject"].to_list() != subject.tolist():
        raise ValueError("patients.csv does not hold the patients of the plan")
    return Patients(
        subject,
        np.array([plan["folds"][s] for s in subject]),
        np.array([plan["labels"][s] for s in subject]),
        covariates["age"].cast(pl.Float64).to_numpy(),
        np.array(covariates["sex"].to_list()),
        patients["segments"].cast(pl.Float64).to_numpy(),
        patients["native_rate"].cast(pl.Float64).to_numpy(),
        covariates["files_per_read_session"].cast(pl.Float64).to_numpy(),
        covariates["read_year"].cast(pl.Float64).to_numpy(),
    )


def load_mapping(path: Path | None) -> dict[str, Any]:
    """The medication mapping of the addendum; ``read`` refuses to run without it.

    Raises
    ------
    ValueError
        If there is no mapping, or a key is missing or empty.
    """
    if path is None or not path.is_file():
        raise ValueError("the read needs the mapping of the addendum (protocol section 5)")
    mapping = json.loads(path.read_text())
    empty = [key for key in MAPPING_KEYS if not mapping.get(key)]
    if empty:
        raise ValueError(f"the mapping lacks {empty}")
    return mapping


def _metadata_rows(
    metadata: Path, mapping: dict[str, Any], subjects: set[str], wanted: list[str]
) -> tuple[dict[str, list[Any]], list[tuple[str, str | None, int]]]:
    """The columns of the mapping's sheet, and ``(patient, session, row)`` of the rows of ``subjects``.

    Other patients are dropped here, before any cell is read.

    Raises
    ------
    ValueError
        If the mapping names a sheet or column that does not exist.
    """
    sheets = read_sheets(metadata, int(mapping.get("header_row", 0)))
    sheet = mapping.get("sheet", next(iter(sheets)))
    if sheet not in sheets:
        raise ValueError(f"the spreadsheet has no sheet {sheet}")
    columns = sheets[sheet]
    absent = [name for name in [mapping["subject_column"], mapping["session_column"], *wanted] if name not in columns]
    if absent:
        raise ValueError(f"the spreadsheet has no columns {absent}")
    rows = []
    for row, cell in enumerate(columns[mapping["subject_column"]]):
        owner = subject_token(cell, subjects)
        if owner is not None:
            rows.append((owner, session_token(columns[mapping["session_column"]][row]), row))
    return columns, rows


def medication_by_patient(
    metadata: Path, mapping: dict[str, Any], sessions: dict[str, list[str]], terms: list[str] | None = None
) -> dict[str, str]:
    """The status of each patient from the read sessions and the mapping (section 8).

    ``terms`` replaces the deciding list of the mapping, for the narrow read. ``empty_terms`` of the mapping are cells
    that count as empty.
    """
    columns, rows = _metadata_rows(metadata, mapping, set(sessions), mapping["medication_columns"])
    empty_terms = {str(t).strip().lower() for t in mapping.get("empty_terms", [])}
    texts: dict[tuple[str, str | None], list[str]] = {}
    for owner, session, row in rows:
        cells = [columns[name][row] for name in mapping["medication_columns"]]
        listed = [str(c) for c in cells if not is_missing(c) and str(c).strip().lower() not in empty_terms]
        texts.setdefault((owner, session), []).extend(listed)
    return {
        subject: medication_status(
            [" ; ".join(texts.get((subject, session), [])) for session in values],
            mapping["antiseizure_terms"] if terms is None else terms,
            mapping["empty_means"],
        )
        for subject, values in sessions.items()
    }


def flags_by_patient(
    metadata: Path, mapping: dict[str, Any], sessions: dict[str, list[str]]
) -> dict[str, dict[str, bool]]:
    """For each column of ``flag_columns``, the patients with a ``Y`` at any read session (sections 4 and 10).

    A column derived from the label is counted and never adjusted for. A session without a row counts as no flag.
    """
    wanted = list(mapping.get("flag_columns", []))
    columns, rows = _metadata_rows(metadata, mapping, set(sessions), wanted)
    flags = {name: dict.fromkeys(sessions, False) for name in wanted}
    for owner, session, row in rows:
        if session in sessions[owner]:
            for name in wanted:
                flags[name][owner] = flags[name][owner] or is_flagged(columns[name][row])
    return flags


def aligned_scores(path: Path, subject: np.ndarray) -> np.ndarray:
    """The saved scores of an arm in the patients' order.

    Raises
    ------
    ValueError
        If a patient has no score.
    """
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8})
    scores = dict(zip(frame["subject"].to_list(), frame["score"].to_list()))
    missing = [s for s in subject if s not in scores]
    if missing:
        raise ValueError(f"{path.name} has no score for {len(missing)} patients")
    return np.array([scores[s] for s in subject], dtype=float)


def refit_spectral(root: Path, readout: Path, patients: Patients, seed: int) -> np.ndarray:
    """The refit read of section 6: the ``Spectral`` head with the covariates regressed out of each feature.

    Patients without both covariates stay out, and have a missing score. Writes ``refit_scores.csv``.
    """
    data = np.load(readout / "features" / f"{PRIMARY}.npz")
    subject, label, x = data["subject"].astype(str), data["label"], data["features"]
    index = {s: i for i, s in enumerate(patients.subject)}
    rows = np.array([index[s] for s in subject])
    keep = patients.complete[rows]
    subject, label, x, rows = subject[keep], label[keep], x[keep], rows[keep]
    age, sex, fold = patients.age[rows], patients.sex[rows], patients.fold[rows]
    out = np.full(len(patients.subject), float("nan"))
    records = []
    for f in range(N_FOLDS):
        held = fold == f
        result = fit_head(residualize(x, age, sex, ~held), label, subject, held, StandardScaler, seed=seed + f)
        for s, value in zip(result.subject, result.score):
            out[index[str(s)]] = float(value)
            records.append({"subject": str(s), "fold": f, "score": float(value), "c": result.c})
    pl.DataFrame(records).write_csv(root / "refit_scores.csv")
    return out


def _row(read: str, arm: str, group: str, result: MatchedRead) -> dict[str, Any]:
    return {"read": read, "arm": arm, "group": group, **result.as_dict()}


def _value_row(
    read: str, arm: str, group: str, value: float, low: float | None = None, high: float | None = None
) -> dict[str, Any]:
    return {
        "read": read,
        "arm": arm,
        "group": group,
        "value": float(value),
        "low": low,
        "high": high,
        "pairs": None,
        "patients_epilepsy": None,
        "patients_non_epilepsy": None,
    }


def demographics(patients: Patients) -> dict[str, Any]:
    """Age quartiles and sex counts by class (section 10)."""
    report = {}
    for label, name in ((1, "epilepsy"), (0, "non_epilepsy")):
        where = patients.label == label
        ages = patients.age[where]
        counts = Counter(patients.sex[where].tolist())
        report[name] = {
            "patients": int(where.sum()),
            "age_missing": int(np.isnan(ages).sum()),
            "age_quartiles": np.nanquantile(ages, [0.25, 0.5, 0.75]).tolist(),
            "female": counts["F"],
            "male": counts["M"],
            "sex_missing": counts[MISSING_SEX],
        }
    return report


def medication_table(patients: Patients, status: np.ndarray) -> dict[str, dict[str, int]]:
    """The class by status table of section 8."""
    table = {}
    for label, name in ((1, "epilepsy"), (0, "non_epilepsy")):
        counts = Counter(status[patients.label == label].tolist())
        table[name] = {group: counts[group] for group in ("on", "off", "unknown")}
    return table


def flag_table(patients: Patients, flags: dict[str, dict[str, bool]]) -> dict[str, dict[str, int]]:
    """Patients with each flag, by class (sections 4 and 10)."""
    return {
        name: {
            group: int(sum(values[s] for s in patients.subject[patients.label == label]))
            for label, group in ((1, "epilepsy"), (0, "non_epilepsy"))
        }
        for name, values in flags.items()
    }


def _added_value(patients: Patients, score: np.ndarray, n_boot: int) -> tuple[float, float, float]:
    cc = patients.complete
    z = standardize_within_fold(score, patients.fold)[cc]
    s, f, y, age, sex = patients.subject[cc], patients.fold[cc], patients.label[cc], patients.age[cc], patients.sex[cc]
    m0 = ArmScores.from_rows(s, f, y, second_level_scores(y, f, age, sex))
    m1 = ArmScores.from_rows(s, f, y, second_level_scores(y, f, age, sex, z))
    return paired_interval(m1, m0, n_boot=n_boot)


def _m0_read(patients: Patients, n_boot: int) -> tuple[float, float, float]:
    cc = patients.complete
    s, f, y, age, sex = patients.subject[cc], patients.fold[cc], patients.label[cc], patients.age[cc], patients.sex[cc]
    m0 = ArmScores.from_rows(s, f, y, second_level_scores(y, f, age, sex))
    chance = ArmScores.from_rows(s, f, y, np.zeros(len(s)))
    _, low, high = paired_interval(m0, chance, n_boot=n_boot)
    return mean_fold_auroc(m0), low + CHANCE, high + CHANCE


def _band_rows(readout: Path, patients: Patients, mask: np.ndarray, n_boot: int) -> list[dict[str, Any]]:
    """The band-power table on matched pairs, and each band's rank correlation with age inside each class."""
    data = np.load(readout / "features" / f"{PRIMARY}.npz")
    subject, x = data["subject"].astype(str), data["features"]
    per_channel = 2 * len(SPECTRAL_BANDS)
    means = np.stack([x[subject == s].mean(axis=0) for s in patients.subject])
    means = means.reshape(len(patients.subject), SPECTRAL_CHANNELS, per_channel).mean(axis=1)
    rows = []
    for k, kind in enumerate(SPECTRAL_KINDS):
        for b, band in enumerate(SPECTRAL_BANDS):
            value = means[:, k * len(SPECTRAL_BANDS) + b]
            name = f"{kind} {band} Hz"
            read = matched_read(value, patients.label, patients.fold, mask, standardize=False, n_boot=n_boot)
            rows.append(_row("band_power", name, "matched", read))
            for label, group in ((1, "epilepsy"), (0, "non_epilepsy")):
                where = (patients.label == label) & np.isfinite(patients.age)
                rho = spearmanr(value[where], patients.age[where]).statistic
                rows.append(_value_row("band_age_rank", name, group, rho))
    return rows


def _sensitivity_rows(patients: Patients, scores: dict[str, np.ndarray], n_boot: int) -> list[dict[str, Any]]:
    """The match at other calipers, also by rate and by bag ratio, and within the age bands."""
    label, fold, age, sex = patients.label, patients.fold, patients.age, patients.sex
    masks = {f"{caliper} years": pair_mask(label, age, sex, caliper) for caliper in SENSITIVITY_CALIPERS}
    masks["same rate"] = pair_mask(label, age, sex, extra=group_pairs(label, patients.rate))
    masks["bag ratio"] = pair_mask(label, age, sex, extra=ratio_pairs(label, patients.segments))
    for name, low, high in AGE_BANDS:
        inside = (age >= low) & (age < high)
        masks[f"age {name}"] = all_pairs(label, inside)
    return [
        _row("sensitivity", arm, group, matched_read(score, label, fold, mask, n_boot=n_boot))
        for arm, score in scores.items()
        for group, mask in masks.items()
    ]


def audit(root: Path, readout: Path, metadata: Path, mapping: dict[str, Any], n_boot: int = N_BOOTSTRAP) -> dict:
    """Gates, controls, rule, medication read and descriptive reads. Writes the summary and the decision."""
    plan = load_plan(readout)
    patients = load_patients(root, readout, plan)
    label, fold, age, sex = patients.label, patients.fold, patients.age, patients.sex
    frame = pl.read_csv(root / "session_ages.csv", schema_overrides={"subject": pl.Utf8})
    ages: dict[str, list[tuple[float, float]]] = {}
    for subject, year, value in zip(frame["subject"].to_list(), frame["year"].to_list(), frame["age"].to_list()):
        ages.setdefault(subject, []).append((float(year), float("nan") if value is None else float(value)))
    by_pairs, session_pairs = age_validity(ages)
    validity, valid_patients = patient_age_validity(ages)
    repaired_age = repaired_ages(patients.subject, patients.read_year, ages)
    fit, fit_patients = read_age_fit(patients.subject, age, repaired_age, ages)
    gates = header_gates(age, sex, validity, fit) | {
        "age_validity_patients": valid_patients,
        "age_validity_by_pairs": by_pairs,
        "age_session_pairs": session_pairs,
        "age_read_fit_patients": fit_patients,
    }
    mask = pair_mask(label, age, sex)
    counts = {"epilepsy": int(mask.any(axis=1).sum()), "non_epilepsy": int(mask.any(axis=0).sum())}
    decision: dict[str, Any] = {"rule": "protocol section 7", "gates": gates, "matched_patients": counts}
    rows: list[dict[str, Any]] = []

    def finish(outcome: str) -> dict[str, Any]:
        decision["outcome"] = outcome
        _save_json(root / "audit_decision.json", decision)
        if rows:
            pl.DataFrame(rows, infer_schema_length=None).write_csv(root / "audit_summary.csv")
        logger.info(f"Covariate audit: {outcome}")
        return decision

    if not gates_pass(gates):
        logger.error("A header gate failed: no result is read")
        return finish(audit_outcome(False, *counts.values()))

    decision["demographics"] = demographics(patients)
    status_by_patient = medication_by_patient(metadata, mapping, read_sessions(root))
    status = np.array([status_by_patient[s] for s in patients.subject])
    pl.DataFrame({"subject": patients.subject.tolist(), "status": status.tolist()}).write_csv(root / "medication.csv")
    decision["medication_table"] = medication_table(patients, status)
    narrow = None
    if mapping.get("narrow_terms"):
        by_narrow = medication_by_patient(metadata, mapping, read_sessions(root), mapping["narrow_terms"])
        narrow = np.array([by_narrow[s] for s in patients.subject])
        decision["medication_table_narrow"] = medication_table(patients, narrow)
    flags = flags_by_patient(metadata, mapping, read_sessions(root))
    decision["flag_table"] = flag_table(patients, flags)
    outcome = audit_outcome(True, *counts.values())
    if outcome == "too_few":
        return finish(outcome)

    controls = control_reads(label, fold, age, sex, seed=plan["seed"])
    decision["controls"] = controls
    if not controls["passed"]:
        logger.error(f"A control failed: no EEG result is read; {controls}")
        return finish(audit_outcome(False, *counts.values()))

    # Only from here on does the audit open an EEG score.
    scores = {
        name: aligned_scores(readout / "scores" / f"{name}.csv", patients.subject) for name in (PRIMARY, COMPARISON)
    }
    refit = refit_spectral(root, readout, patients, plan["seed"])
    primary = matched_read(scores[PRIMARY], label, fold, mask, n_boot=n_boot)
    refit_read = matched_read(refit, label, fold, mask, n_boot=n_boot)
    comparison = matched_read(scores[COMPARISON], label, fold, mask, n_boot=n_boot)
    repaired_mask = pair_mask(label, repaired_age, sex)
    repaired_read = matched_read(scores[PRIMARY], label, fold, repaired_mask, n_boot=n_boot)
    decision |= {
        "primary": primary.as_dict(),
        "refit": refit_read.as_dict(),
        "hydra": comparison.as_dict(),
        "repaired": repaired_read.as_dict(),
        f"excludes_above_{EXCLUSION_BOUND}": excludes_above_bound(primary),
    }
    decision["medication"] = medication_read(scores[PRIMARY], label, fold, mask, status, n_boot=n_boot)
    if narrow is not None:
        decision["medication_narrow"] = medication_read(scores[PRIMARY], label, fold, mask, narrow, n_boot=n_boot)
    known = status != "unknown"
    on = (status == "on").astype(float)
    rows.append(
        _row(
            "medication_alone",
            "on or off",
            "known",
            matched_read(on, label, fold, all_pairs(label, known), False, n_boot),
        )
    )

    rows += [
        _row("matched", name, "5 years, same sex", read)
        for name, read in ((PRIMARY, primary), (REFIT, refit_read), (COMPARISON, comparison))
    ]
    rows.append(_row("matched", PRIMARY, "repaired age", repaired_read))
    read = matched_read(scores[COMPARISON], label, fold, repaired_mask, n_boot=n_boot)
    rows.append(_row("matched", COMPARISON, "repaired age", read))
    for name in DESCRIPTIVE_ARMS:
        scores[name] = aligned_scores(readout / "scores" / f"{name}.csv", patients.subject)
        rows.append(
            _row("matched", name, "5 years, same sex", matched_read(scores[name], label, fold, mask, n_boot=n_boot))
        )
    rows.append(
        _row(
            "matched_difference",
            f"{PRIMARY} minus {COMPARISON}",
            "5 years, same sex",
            matched_difference(scores[PRIMARY], scores[COMPARISON], label, fold, mask, n_boot),
        )
    )
    usable = np.isfinite(age)
    rows.append(
        _row(
            "alone",
            "age",
            "older epilepsy above 0.5",
            matched_read(age, label, fold, all_pairs(label, usable), False, n_boot),
        )
    )
    known_sex = sex != MISSING_SEX
    male = (sex == "M").astype(float)
    rows.append(
        _row(
            "alone",
            "sex",
            "male epilepsy above 0.5",
            matched_read(male, label, fold, all_pairs(label, known_sex), False, n_boot),
        )
    )
    rows.append(_value_row("alone", "M0", "age, age squared, sex", *_m0_read(patients, n_boot)))
    for name in (PRIMARY, COMPARISON):
        rows.append(_value_row("added_value", name, "M1 minus M0", *_added_value(patients, scores[name], n_boot)))
    rows += _sensitivity_rows(patients, {name: scores[name] for name in (PRIMARY, COMPARISON)}, n_boot)
    rows += _band_rows(readout, patients, mask, n_boot)
    listed = mapping.get("comorbidity_columns", [])
    if listed:
        comorbid = np.array([any(flags[name][s] for name in listed) for s in patients.subject])
        alone = matched_read(comorbid.astype(float), label, fold, all_pairs(label), False, n_boot)
        rows.append(_row("alone", "listed comorbidity", "more in epilepsy above 0.5", alone))
        without = mask & all_pairs(label, ~comorbid)
        for name in (PRIMARY, COMPARISON):
            read = matched_read(scores[name], label, fold, without, n_boot=n_boot)
            rows.append(_row("comorbidity", name, "matched, none of the listed", read))
    files = patients.files_per_session
    rows.append(
        _row(
            "alone",
            "files per read session",
            "more in epilepsy above 0.5",
            matched_read(files, label, fold, all_pairs(label), False, n_boot),
        )
    )
    for value, group in ((1, "epilepsy"), (0, "non_epilepsy")):
        rows.append(_value_row("files_per_read_session", "median", group, np.median(files[label == value])))

    final = audit_outcome(True, *counts.values(), primary, refit_read, comparison, repaired_read)
    return finish(final)


def run_read(args: argparse.Namespace) -> None:
    """Steps 1 and 2. The mapping is checked first, so that a read cannot start without the addendum."""
    mapping = load_mapping(args.mapping)
    root = args.root.resolve()
    (root / "medication_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    audit(root, args.readout_root.resolve(), metadata_path(args), mapping)


# ---------------------------------------------------------------------- command line


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, description: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--root", type=Path, required=True)
        return sub

    def corpus(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        sub.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        sub.add_argument("--metadata", type=Path, default=None, help=f"default: <data dir>/{METADATA_FILE}")

    inventory = command("inventory", "fix the read plan and write the spreadsheet's columns; no label, score or age")
    corpus(inventory)
    inventory.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
    inventory.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
    inventory.add_argument("--num-workers", type=int, default=4)
    inventory.add_argument("--header-row", type=int, default=0)
    inventory.add_argument("--subject-column", default=None, help="only if the inventory did not find it")
    inventory.add_argument("--session-column", default=None, help="only if the inventory did not find it")
    command("extract", "read age and sex from the EDF headers of the read plan")
    read = command("read", "apply the gates, the controls, the rule and the descriptive reads")
    corpus(read)
    read.add_argument("--mapping", type=Path, default=None, help="the medication mapping of the addendum")
    args = parser.parse_args()
    if args.command == "inventory":
        run_inventory(args)
    elif args.command == "extract":
        run_extract(args)
    else:
        run_read(args)


if __name__ == "__main__":
    main()
