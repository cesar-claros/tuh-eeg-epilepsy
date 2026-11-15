"""Spectral decomposition and score fusion: do the three best fixed-read scores hold different information, and which
part of ``Spectral`` carries its score?

Protocol: ``documentation/spectral_decomposition_fusion_protocol.md``. CPU only: it reads the saved files of frozen
readout pass 1, of the encoder check and of the covariate audit, and no signal. ``src/spectral_decomposition_core.py``
holds the column layout, the arms, the rank-mean and the rule. Subcommands, in order:

- ``fit`` checks the saved ``Spectral`` features (patients, segments, column layout, channel order; ``inputs.json``),
  fits the head of pass 1 on every arm of section 5 (``scores/``), and the permuted-label read of the two smallest
  arms (``permutation_null.json``). A finished arm is skipped.
- ``gates`` checks gates 2 to 6 of section 6 (the launcher runs the tests, gate 1) and writes ``gates.json``.
- ``read`` applies the rule of section 7 and the reads of section 8 (``fusion_decision.json``,
  ``decomposition_summary.csv``); it stops if a gate failed.

The 39 test patients stay locked: every stage refuses a patient outside the plan of pass 1. The cohort helpers
repeat those of ``src/band_code_readout.py``, so that this module needs no torch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
from loguru import logger
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.covariate_audit_core import (  # noqa: E402
    MISSING_SEX,
    group_read,
    matched_read,
    pair_mask,
    standardize_within_fold,
)
from src.encoder_check_core import alone_interval, drug_gain  # noqa: E402
from src.frozen_readout_core import (  # noqa: E402
    N_FOLDS,
    N_PERMUTATIONS,
    ArmScores,
    auroc,
    bag_size_link,
    fit_head,
    mean_fold_auroc,
    paired_interval,
    permuted_scores,
    pooled_auroc,
)
from src.spectral_decomposition_core import (  # noqa: E402
    AUROC_TOLERANCE,
    FULL,
    FUSED,
    LAYOUT_TOLERANCE,
    NULL_ARMS,
    PER_CHANNEL,
    SCORE_TOLERANCE,
    arm_features,
    arm_names,
    channel_problems,
    fusion_outcome,
    layout_gap,
    needs_channels,
    rank_mean,
)

PREDICTION = "no_gain about 70%, with Fusion between 0.70 and 0.72 (protocol section 7)"
SAVED = {"Spectral": "Spectral.csv", "HYDRA": "HYDRA.csv"}  # score files of the readout root
ENCODER_SCORES = "D.csv"  # the REVE fixed read of the encoder check
MATCHED = ("Fusion", "Spectral")


# ---------------------------------------------------------------------- files


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot write {type(value).__name__} as JSON")


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_plain) + "\n")


def _scores(path: Path) -> ArmScores:
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8})
    columns = (frame[name].to_numpy() for name in ("fold", "label", "score"))
    return ArmScores.from_rows(frame["subject"].to_numpy().astype(str), *columns)


def load_cohort(readout_root: Path, audit_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """The patients in sorted order, with fold, label, drug status, age, sex and segments per patient."""
    overrides = {"subject": pl.Utf8}
    patients = pl.read_csv(readout_root / "patients.csv", schema_overrides=overrides).sort("subject")
    subject = np.array(patients["subject"].to_list())
    exposure = pl.read_csv(audit_root / "drug_exposure.csv", schema_overrides=overrides)
    status = dict(zip(exposure["subject"].to_list(), exposure["status"].to_list(), strict=True))
    covariates = pl.read_csv(audit_root / "covariates.csv", schema_overrides={"subject": pl.Utf8, "sex": pl.Utf8})
    age = dict(zip(covariates["subject"].to_list(), covariates["age"].cast(pl.Float64).to_list(), strict=True))
    sex = dict(zip(covariates["subject"].to_list(), covariates["sex"].to_list(), strict=True))
    return {
        "subject": subject,
        "fold": np.array([plan["folds"][s] for s in subject]),
        "label": np.array([plan["labels"][s] for s in subject]),
        "status": np.array([status.get(s, "unknown") for s in subject]),
        "age": np.array([np.nan if age.get(s) is None else float(age[s]) for s in subject]),
        "sex": np.array([sex.get(s) or MISSING_SEX for s in subject]),
        "segments": dict(zip(subject.tolist(), patients["segments"].to_list(), strict=True)),
    }


def _matches_plan(arm: ArmScores, plan: dict[str, Any]) -> bool:
    """Whether an arm holds exactly the patients of the plan, with their folds and labels."""
    patients = sorted(plan["folds"])
    return (
        arm.subject.tolist() == patients
        and arm.fold.tolist() == [plan["folds"][s] for s in patients]
        and arm.label.tolist() == [plan["labels"][s] for s in patients]
    )


def load_features(readout_root: Path, plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The saved ``Spectral`` rows: patient, label and features per segment.

    Raises
    ------
    ValueError
        If a row belongs to a patient outside the plan (the test patients are locked).
    """
    data = np.load(readout_root / "features" / "Spectral.npz")
    subject, label, x = data["subject"].astype(str), data["label"], data["features"]
    outside = sorted(set(subject.tolist()) - set(plan["folds"]))
    if outside:
        raise ValueError(f"{len(outside)} patients of the features are outside the plan: locked ({outside[:3]})")
    return subject, label, x


def channel_order(readout_root: Path) -> tuple[list[str] | None, list[str]]:
    """Gate 5: the channel order of the columns, from the dictionaries' records of the readout root.

    Returns the names, or ``None`` with the reasons when the order cannot be established.
    """
    files = sorted((readout_root / "dictionaries").glob("*/sae_channels.txt"))
    if not files:
        return None, ["the readout root records no channel order (dictionaries/*/sae_channels.txt)"]
    orders = {tuple(f.read_text().split()) for f in files}
    if len(orders) != 1:
        return None, [f"the {len(files)} channel records of the readout root disagree"]
    names = list(orders.pop())
    problems = channel_problems(names)
    return (None, problems) if problems else (names, [])


# ---------------------------------------------------------------------- fit


def check_inputs(readout_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """The label-free checks of the saved features: patients and segments (gate 2), column layout (gate 4) and
    channel order (gate 5)."""
    subject, label, x = load_features(readout_root, plan)
    patients, counts = np.unique(subject, return_counts=True)
    names, problems = channel_order(readout_root)
    columns_ok = x.ndim == 2 and x.shape[1] % PER_CHANNEL == 0
    return {
        "patients": len(patients),
        "patients_ok": patients.tolist() == sorted(plan["folds"]),
        "segments_ok": bool((counts == plan["n_segments"]).all()),
        "labels_ok": bool(all(int(y) == plan["labels"][s] for s, y in zip(subject, label, strict=True))),
        "columns": int(x.shape[1]),
        "layout_gap": layout_gap(x) if columns_ok else math.inf,
        "channels": names,
        "channels_ok": names is not None and columns_ok and len(names) * PER_CHANNEL == x.shape[1],
        "channel_problems": problems,
    }


def fit(root: Path, readout_root: Path) -> None:
    """The head of pass 1 on every arm of section 5, and the permuted-label read of the two smallest arms.

    Raises
    ------
    ValueError
        If the saved features do not hold the patients and segments of the plan.
    """
    plan = _load_json(readout_root / "folds.json")
    inputs = check_inputs(readout_root, plan)
    _save_json(root / "inputs.json", inputs)
    if not (inputs["patients_ok"] and inputs["segments_ok"] and inputs["labels_ok"]):
        raise ValueError(f"the saved Spectral features do not match the plan of pass 1: see {root / 'inputs.json'}")
    subject, label, x = load_features(readout_root, plan)
    channels = inputs["channels"] if inputs["channels_ok"] else None
    (root / "scores").mkdir(parents=True, exist_ok=True)
    for arm in arm_names():
        if needs_channels(arm) and channels is None:
            logger.warning(f"{arm}: the channel order is not established; skipped")
            continue
        out = root / "scores" / f"{arm}.csv"
        if out.is_file():
            continue
        features = arm_features(arm, x, channels)
        rows = []
        for fold in range(N_FOLDS):
            held = np.array([plan["folds"][s] == fold for s in subject])
            result = fit_head(features, label, subject, held, StandardScaler, seed=plan["seed"] + fold)
            rows += [
                {"arm": arm, "subject": s, "fold": fold, "label": int(y), "score": float(v), "c": result.c}
                for s, y, v in zip(result.subject, result.label, result.score, strict=True)
            ]
            logger.info(f"head {arm} fold {fold}: C {result.c}, AUROC {auroc(result.label, result.score):.3f}")
        pl.DataFrame(rows).write_csv(out)
    _permutations(root, plan, subject, label, x)


def _permutations(root: Path, plan: dict[str, Any], subject: np.ndarray, label: np.ndarray, x: np.ndarray) -> None:
    """Section 8: the heads of ``NULL_ARMS`` with the training labels shuffled between patients."""
    path = root / "permutation_null.json"
    if path.is_file():
        return
    record = {}
    for arm in NULL_ARMS:
        frame = pl.read_csv(root / "scores" / f"{arm}.csv")
        chosen = {row["fold"]: row["c"] for row in frame.select("fold", "c").unique().to_dicts()}
        features = arm_features(arm, x)
        rng = np.random.default_rng(plan["seed"])
        draws = []
        for _ in range(N_PERMUTATIONS):
            values = []
            for fold in range(N_FOLDS):
                held = np.array([plan["folds"][s] == fold for s in subject])
                patients, pooled = permuted_scores(features, label, subject, held, chosen[fold], StandardScaler, rng)
                values.append(auroc(np.array([plan["labels"][s] for s in patients]), pooled))
            draws.append(float(np.mean(values)))
        record[arm] = {"draws": draws, "mean": float(np.mean(draws)), "sd": float(np.std(draws))}
    _save_json(path, record)


# ---------------------------------------------------------------------- gates


def _expected_arms(inputs: dict[str, Any]) -> list[str]:
    return [a for a in arm_names() if inputs["channels_ok"] or not needs_channels(a)]


def gates(root: Path, readout_root: Path, encoder_root: Path) -> dict[str, Any]:
    """Gates 2 to 6 of section 6; writes ``gates.json``. A missing channel order skips arms and does not block."""
    plan = _load_json(readout_root / "folds.json")
    inputs = _load_json(root / "inputs.json")
    saved = {name: _scores(readout_root / "scores" / file) for name, file in SAVED.items()}
    saved["REVE"] = _scores(encoder_root / "scores" / ENCODER_SCORES)
    scores_ok = {name: _matches_plan(arm, plan) for name, arm in saved.items()}
    full_path = root / "scores" / f"{FULL}.csv"
    reproduction: dict[str, Any] = {"ok": False}
    if full_path.is_file() and scores_ok["Spectral"]:
        full = _scores(full_path)
        if _matches_plan(full, plan):
            old_c = pl.read_csv(readout_root / "scores" / SAVED["Spectral"]).select("fold", "c").unique().sort("fold")
            new_c = pl.read_csv(full_path).select("fold", "c").unique().sort("fold")
            reproduction = {
                "same_c": old_c["c"].to_list() == new_c["c"].to_list(),
                "score_gap": float(np.abs(full.score - saved["Spectral"].score).max()),
                "auroc": mean_fold_auroc(full),
                "saved_auroc": mean_fold_auroc(saved["Spectral"]),
            }
            reproduction["ok"] = bool(
                reproduction["same_c"]
                and reproduction["score_gap"] <= SCORE_TOLERANCE
                and abs(reproduction["auroc"] - reproduction["saved_auroc"]) <= AUROC_TOLERANCE
            )
    expected = _expected_arms(inputs)
    complete = {
        arm: (root / "scores" / f"{arm}.csv").is_file() and _matches_plan(_scores(root / "scores" / f"{arm}.csv"), plan)
        for arm in expected
    }
    record = {
        "inputs_ok": bool(
            inputs["patients_ok"] and inputs["segments_ok"] and inputs["labels_ok"] and all(scores_ok.values())
        ),
        "score_files": scores_ok,
        "reproduction": reproduction,
        "layout_gap": inputs["layout_gap"],
        "layout_ok": inputs["layout_gap"] <= LAYOUT_TOLERANCE,
        "channels_ok": inputs["channels_ok"],
        "channel_problems": inputs["channel_problems"],
        "skipped_arms": [a for a in arm_names() if a not in expected],
        "complete_ok": all(complete.values()),
        "incomplete_arms": [a for a, ok in complete.items() if not ok],
    }
    blocking = ("inputs_ok", "layout_ok", "complete_ok")
    record["status"] = "pass" if all(record[k] for k in blocking) and reproduction["ok"] else "fail"
    _save_json(root / "gates.json", record)
    logger.info(f"gates: {record['status']}")
    return record


# ---------------------------------------------------------------------- read


def _interval(values: tuple[float, float, float]) -> dict[str, float]:
    return dict(zip(("value", "low", "high"), values, strict=True))


def _arm_summary(name: str, arm: ArmScores, cohort: dict[str, Any], n_boot: int) -> dict[str, Any]:
    value, low, high = alone_interval(arm, n_boot)
    gain = drug_gain(arm.score, cohort["label"], cohort["fold"], cohort["status"], n_boot)
    z = standardize_within_fold(arm.score, cohort["fold"])
    off = cohort["status"] == "off"
    cell = group_read(z, (cohort["label"] == 1) & off, (cohort["label"] == 0) & off, cohort["fold"], n_boot)
    links = bag_size_link(arm, cohort["segments"])
    return {
        "arm": name,
        "auroc": value,
        "low": low,
        "high": high,
        "pooled_auroc": pooled_auroc(arm),
        "gain": gain[0],
        "gain_low": gain[1],
        "gain_high": gain[2],
        "off_off": cell.value,
        "off_off_low": cell.low,
        "off_off_high": cell.high,
        "off_off_epilepsy": cell.first,
        "off_off_non_epilepsy": cell.second,
        "bag_link_epilepsy": links["epilepsy"],
        "bag_link_non_epilepsy": links["non_epilepsy"],
    }


def _within_fold_spearman(first: ArmScores, second: ArmScores) -> float:
    values = [
        float(spearmanr(first.score[first.fold == f], second.score[second.fold == f]).statistic)
        for f in np.unique(first.fold)
    ]
    return float(np.mean(values))


def _fused(arms: dict[str, ArmScores], names: tuple[str, ...]) -> ArmScores:
    first = arms[names[0]]
    return ArmScores(first.subject, first.fold, first.label, rank_mean([arms[n].score for n in names], first.fold))


def read(root: Path, readout_root: Path, audit_root: Path, encoder_root: Path, n_boot: int = 10_000) -> dict[str, Any]:
    """The rule of section 7 and the reads of section 8; writes ``fusion_decision.json`` and its table.

    Raises
    ------
    ValueError
        If the gates have not passed, or if a saved arm does not hold the patients of the plan.
    """
    plan = _load_json(readout_root / "folds.json")
    gate_record = _load_json(root / "gates.json")
    if gate_record["status"] != "pass":
        record = {
            "protocol": "documentation/spectral_decomposition_fusion_protocol.md, section 7",
            "outcome": fusion_outcome(False, False),
            "prediction": PREDICTION,
            "gates": gate_record,
        }
        _save_json(root / "fusion_decision.json", record)
        logger.error("a gate failed: no score is read")
        return record
    cohort = load_cohort(readout_root, audit_root, plan)
    arms = {name: _scores(readout_root / "scores" / file) for name, file in SAVED.items()}
    arms["REVE"] = _scores(encoder_root / "scores" / ENCODER_SCORES)
    same_cohort = cohort["subject"].tolist() == sorted(plan["folds"])
    if not (same_cohort and all(_matches_plan(arm, plan) for arm in arms.values())):
        raise ValueError("a saved arm does not hold the patients of the plan in their folds")
    arms["Fusion"] = _fused(arms, FUSED)
    pairs = [(a, b) for i, a in enumerate(FUSED) for b in FUSED[i + 1 :]]
    for a, b in pairs:
        arms[f"Fusion-{a}-{b}"] = _fused(arms, (a, b))
    deciding = paired_interval(arms["Fusion"], arms["Spectral"], n_boot=n_boot)
    met = {"F": deciding[1] > 0}
    outcome = fusion_outcome(True, met["F"])
    spec = {a: _scores(root / "scores" / f"{a}.csv") for a in arm_names() if a not in gate_record["skipped_arms"]}
    summary = [_arm_summary(n, a, cohort, n_boot) for n, a in {**arms, **spec}.items()]
    for row in summary:
        name = row["arm"]
        reference = "Spectral" if name in arms else FULL
        base = arms[reference] if name in arms else spec[reference]
        if name == reference:
            row |= {"reference": reference, "difference": 0.0, "difference_low": 0.0, "difference_high": 0.0}
            continue
        value, low, high = paired_interval(arms[name] if name in arms else spec[name], base, n_boot=n_boot)
        row |= {"reference": reference, "difference": value, "difference_low": low, "difference_high": high}
    pl.DataFrame(summary).write_csv(root / "decomposition_summary.csv")
    mask = pair_mask(cohort["label"], cohort["age"], cohort["sex"])
    matched = {}
    for name in MATCHED:
        m = matched_read(arms[name].score, cohort["label"], cohort["fold"], mask, n_boot=n_boot)
        matched[name] = {"value": m.value, "low": m.low, "high": m.high, "pairs": m.pairs}
    agreement = {
        f"{a} with {b}": _within_fold_spearman(arms[a], arms[b]) for a, b in pairs + [(a, "Fusion") for a in FUSED]
    }
    null_path = root / "permutation_null.json"
    record = {
        "protocol": "documentation/spectral_decomposition_fusion_protocol.md, section 7",
        "outcome": outcome,
        "prediction": PREDICTION,
        "gates": gate_record,
        "conditions": met,
        "deciding": {"F_Fusion_minus_Spectral": _interval(deciding)},
        "patients": len(cohort["subject"]),
        "known_drug_status": int((cohort["status"] != "unknown").sum()),
        "roots": {"readout": str(readout_root), "audit": str(audit_root), "encoder": str(encoder_root)},
        "descriptive": {
            "arms": summary,
            "rank_correlation_within_fold": agreement,
            "matched": matched,
            "permutation_null": _load_json(null_path) if null_path.is_file() else {},
        },
    }
    _save_json(root / "fusion_decision.json", record)
    logger.info(f"spectral decomposition and fusion: {outcome} (conditions {met})")
    return record


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("fit", "the input checks, the heads of the arms and the permuted-label read"),
        ("gates", "gates 2 to 6"),
        ("read", "the rule and the descriptive reads"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        if name != "fit":
            command.add_argument("--encoder-root", type=Path, required=True, help="the root of the encoder check")
        if name == "read":
            command.add_argument("--audit-root", type=Path, required=True, help="the covariate audit root")
            command.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()
    root, readout_root = args.root.resolve(), args.readout_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "fit":
        fit(root, readout_root)
    elif args.command == "gates":
        gates(root, readout_root, args.encoder_root.resolve())
    else:
        read(root, readout_root, args.audit_root.resolve(), args.encoder_root.resolve(), args.n_boot)


if __name__ == "__main__":
    main()
