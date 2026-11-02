"""Frozen encoder check: does a strong public encoder read above band power, and does it add to the drug?

Protocol: ``documentation/frozen_encoder_check_protocol.md``. Run on the HPC from ``code/``. A root holds one check of
one readout root and one audit root. ``src/encoder_check_core.py`` holds the pure logic and the rule. Subcommands, in
order:

- ``step0`` reads the protocol on features that exist, with no encoder. It decides nothing.
- ``smoke`` loads the weights and embeds a few real segments, to find setup errors before a long run.
- ``extract`` runs the encoder, with no label: ``all`` (every campaign segment and the windows of read ``H``),
  ``level`` (the fixed read on a corpus-wide scale) and ``extra`` (the recordings the campaign left out).
- ``gates`` applies the gates of section 7. No head fits on real labels of the deciding read here.
- ``read`` applies the rule and writes the descriptive reads. It refuses to run before the gates pass.

The 39 test patients stay locked: only their IDs are read, to refuse them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import rootutils
from loguru import logger
from sklearn.preprocessing import StandardScaler

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.covariate_audit_core import group_read, standardize_within_fold  # noqa: E402
from src.encoder_check_core import (  # noqa: E402
    BAND,
    ELECTRODES,
    EMBED_DIM,
    HIGH_RATES,
    N_PERMUTATIONS,
    N_SPLITS,
    PROBE_WINDOW_SAMPLES,
    SEGMENT_SAMPLES,
    SEGMENT_SECONDS,
    SFREQ,
    age_correlation,
    alone_interval,
    common_average,
    drug_gain,
    embed_segment,
    encoder_outcome,
    extra_recordings,
    fold_scores,
    gain_ceiling,
    gate_report,
    helwan_scores,
    level_scale,
    linear_scores,
    mlp_scores,
    patient_rows,
    random_windows,
    rotate_rows,
    scale_level,
    segment_starts,
    split_spread,
    tiled_length,
    zscore_channels,
)
from src.frozen_readout_core import (  # noqa: E402
    N_BOOTSTRAP,
    N_FOLDS,
    ArmScores,
    auroc,
    bag_size_link,
    fit_head,
    mean_fold_auroc,
    paired_interval,
    permuted_scores,
    segment_draw,
)

CHECKPOINT = "brain-bzh/reve-base"
ENCODE_BATCH = 48
LOAD_BATCH = 2
READS = ("H", "D-level", "A", "A-mlp", "A-seiz", "A-rates", "C")


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def load_plan(readout: Path) -> dict[str, Any]:
    """The fold plan of the readout root (``folds.json``)."""
    return json.loads((readout / "folds.json").read_text())


# ---------------------------------------------------------------------- encoder


class ReveEncoder:
    """The frozen REVE encoder: windows ``(batch, channels, samples)`` to one embedding per window (section 3)."""

    def __init__(self, device: str, names: Iterable[str] = ELECTRODES, checkpoint: str = CHECKPOINT) -> None:
        import torch
        from braindecode.models import REVE

        self.torch, self.device = torch, device
        self.model = REVE.from_pretrained(checkpoint).to(device).eval()
        self.positions = self.model.get_positions(list(names)).to(device)

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        """The mean over each window's tokens, in batches."""
        parts = []
        with self.torch.no_grad():
            for start in range(0, len(windows), ENCODE_BATCH):
                x = self.torch.from_numpy(windows[start : start + ENCODE_BATCH]).to(self.device)
                pos = self.positions.unsqueeze(0).expand(len(x), -1, -1)
                features = self.model(x, pos=pos, return_features=True)["features"]
                pooled = features.mean(dim=tuple(range(1, features.ndim - 1)))
                parts.append(pooled.float().cpu().numpy())
        return np.concatenate(parts)


def segment_view(frame: pd.DataFrame, montages: dict) -> Any:
    """The project's window reader on referential channels, with REVE's band and rate (section 3, steps 1 and 3)."""
    from src.data.components.window_dataset import WindowDataset

    return WindowDataset(
        frame,
        "raw",
        list(ELECTRODES),
        SFREQ,
        SEGMENT_SAMPLES,
        montages,
        list(BAND),
        True,
        False,
        None,
        (),
        0.0,
        False,
        ["epilepsy"],
        notch_freqs=None,
        bipolar=False,
        interpolate_bad_channels=True,
    )


def _campaign(args: argparse.Namespace, plan: dict[str, Any]) -> tuple[Any, Any, dict, set[str]]:
    """The cohort of the readout, its sources and its montages; no signal is read.

    Raises
    ------
    ValueError
        If the patients or labels differ from those of the readout root.
    """
    from src.frozen_readout import build_cohort, load_sources

    _, _, sources, locked, _ = load_sources(args)
    cohort = build_cohort(sources, locked)
    if cohort.labels != plan["labels"]:
        raise ValueError("the patients or labels differ from those of the readout root")
    return sources, cohort, next(iter(sources.values())).montages, locked


def _frame(sources: dict[str, Any], rows: list[tuple[str, int]]) -> pd.DataFrame:
    """The plan rows of ``rows``, in that order."""
    return pd.concat([sources[name].plan.iloc[[row]] for name, row in rows], ignore_index=True)


def _segments(view: Any, workers: int) -> Iterable[np.ndarray]:
    """Each segment of a view as ``(channels, samples)``, in plan order."""
    from torch.utils.data import DataLoader

    for x, _ in DataLoader(view, batch_size=LOAD_BATCH, shuffle=False, num_workers=workers):
        yield from x.numpy()


def _feature_path(root: Path, name: str) -> Path:
    return root / "features" / f"{name}.npz"


# ---------------------------------------------------------------------- extract


def extract_all(
    root: Path, cohort: Any, sources: dict, montages: dict, plan: dict, workers: int, encode: Callable
) -> None:
    """Every campaign segment, the windows of read ``H``, the fixed signal and its statistics. Reads no label."""
    path = _feature_path(root, "REVE-all")
    if path.is_file():
        logger.info("REVE-all exists; skipped")
        return
    patients = sorted(cohort.labels)
    rows, subject, fixed = [], [], []
    for s in patients:
        chosen = set(segment_draw(s, cohort.keys[s], plan["n_segments"]))
        for position, row in enumerate(cohort.rows[s]):
            rows.append(row)
            subject.append(s)
            fixed.append(position in chosen)
    frame = _frame(sources, rows)
    probes = {s: random_windows(s, len(cohort.rows[s])) for s in patients}
    embeddings = np.zeros((len(rows), EMBED_DIM), dtype=np.float32)
    valid = np.ones(len(rows), bool)
    probe_rows, probe_subject = [], []
    n_channels = len(ELECTRODES)
    signal = np.lib.format.open_memmap(
        root / "fixed_signal.npy", mode="w+", dtype=np.float32, shape=(sum(fixed), n_channels, SEGMENT_SAMPLES)
    )
    sums, squares = np.zeros((len(patients), n_channels)), np.zeros((len(patients), n_channels))
    counts = np.zeros(len(patients))
    index_of = {s: i for i, s in enumerate(patients)}
    position_in_patient: dict[str, int] = {}
    probe_length = tiled_length(PROBE_WINDOW_SAMPLES)
    next_fixed = 0
    for index, raw in enumerate(_segments(segment_view(frame, montages), workers)):
        s = subject[index]
        valid[index] = bool(np.abs(raw).max() > 0)
        referenced = common_average(raw)
        scaled = zscore_channels(referenced)
        embeddings[index] = embed_segment(scaled, encode)
        position = position_in_patient.get(s, 0)
        position_in_patient[s] = position + 1
        starts = [start for segment, start in probes[s] if segment == position]
        if starts:
            windows = np.stack([scaled[:, start : start + probe_length] for start in starts]).astype(np.float32)
            probe_rows.append(encode(windows))
            probe_subject += [s] * len(starts)
        if fixed[index]:
            signal[next_fixed] = referenced
            next_fixed += 1
            i = index_of[s]
            sums[i] += referenced.sum(axis=1)
            squares[i] += (referenced.astype(np.float64) ** 2).sum(axis=1)
            counts[i] += referenced.shape[1]
        if index % 500 == 0:
            logger.info(f"Segment {index} of {len(rows)}")
    signal.flush()
    labels = plan["labels"]
    (root / "features").mkdir(exist_ok=True)
    np.savez(
        _feature_path(root, "REVE-H"),
        subject=np.array(probe_subject),
        label=np.array([labels[s] for s in probe_subject]),
        features=np.concatenate(probe_rows),
    )
    np.savez(root / "fixed_statistics.npz", patients=np.array(patients), sums=sums, squares=squares, counts=counts)
    # Written last, because its existence marks this step as done.
    np.savez(
        path,
        subject=np.array(subject),
        label=np.array([labels[s] for s in subject]),
        features=embeddings,
        fixed=np.array(fixed),
        valid=valid,
    )
    _save_repeat(root, encode, np.load(root / "fixed_signal.npy", mmap_mode="r"))
    logger.info(f"REVE-all written: {len(rows)} segments, {int((~valid).sum())} failed to load")


def _save_repeat(root: Path, encode: Callable, signal: np.ndarray) -> None:
    """Gate 4: the features of 20 segments, twice."""
    first = np.stack([embed_segment(zscore_channels(np.asarray(s)), encode) for s in signal[:20]])
    second = np.stack([embed_segment(zscore_channels(np.asarray(s)), encode) for s in signal[:20]])
    _save_json(root / "repeat.json", {"max_difference": float(np.abs(first - second).max())})


def extract_level(root: Path, plan: dict, encode: Callable) -> None:
    """Read ``D-level``: the fixed read on one scale per channel, from the training patients of each fold."""
    data = np.load(_feature_path(root, "REVE-all"))
    fixed = data["fixed"]
    subject = data["subject"].astype(str)[fixed]
    stats = np.load(root / "fixed_statistics.npz")
    patients = stats["patients"].astype(str)
    signal = np.load(root / "fixed_signal.npy", mmap_mode="r")
    for fold in range(N_FOLDS):
        path = _feature_path(root, f"REVE-level_f{fold}")
        if path.is_file():
            continue
        train = np.array([plan["folds"][s] != fold for s in patients])
        mean, sd = level_scale(stats["sums"], stats["squares"], stats["counts"], train)
        rows = np.stack([embed_segment(scale_level(np.asarray(s), mean, sd), encode) for s in signal])
        labels = np.array([plan["labels"][s] for s in subject])
        np.savez(path, subject=subject, label=labels, features=rows)
        logger.info(f"D-level fold {fold} written")


def extract_extra(
    root: Path, args: argparse.Namespace, cohort: Any, locked: set[str], plan: dict, encode: Callable
) -> None:
    """The recordings of these patients that the campaign left out, on the campaign's 2-minute grid."""
    from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy
    from src.data.components.window_dataset import _read_bads

    path = _feature_path(root, "REVE-extra")
    if path.is_file():
        logger.info("REVE-extra exists; skipped")
        return
    engine = TUHEEGEpilepsy(data_dir=str(args.data_dir.resolve()), add_annotations=True, interpolate_bad_channels=True)
    descriptions = engine.descriptions.to_dict("records")
    campaign = {key[0] for keys in cohort.keys.values() for key in keys}
    chosen = extra_recordings(descriptions, set(cohort.labels), campaign, locked)
    records, seizure, high_rate = [], [], []
    for item in chosen:
        row = item["row"]
        if _read_bads(str(row["path"]))[2]:
            continue  # the campaign also leaves out a recording with too many bad channels
        for start in segment_starts(float(row["duration"])):
            records.append(
                {
                    "subject": row["subject"],
                    "path": row["path"],
                    "start": start,
                    "end": start + SEGMENT_SECONDS,
                    "epilepsy": row["epilepsy"],
                    "description_row": pd.Series(row),
                }
            )
            seizure.append(item["seizure"])
            high_rate.append(item["high_rate"])
    frame = pd.DataFrame(records)
    logger.info(f"Extra read: {len(chosen)} recordings, {len(frame)} segments")
    embeddings = np.zeros((len(frame), EMBED_DIM), dtype=np.float32)
    valid = np.ones(len(frame), bool)
    for index, raw in enumerate(_segments(segment_view(frame, engine.montages), args.num_workers)):
        valid[index] = bool(np.abs(raw).max() > 0)
        embeddings[index] = embed_segment(zscore_channels(common_average(raw)), encode)
        if index % 500 == 0:
            logger.info(f"Extra segment {index} of {len(frame)}")
    subject = frame["subject"].astype(str).to_numpy()
    np.savez(
        path,
        subject=subject,
        label=np.array([plan["labels"][s] for s in subject]),
        features=embeddings,
        seizure=np.array(seizure),
        high_rate=np.array(high_rate),
        valid=valid,
    )


def run_extract(args: argparse.Namespace) -> None:
    """Run the extraction steps in order; a finished step is skipped."""
    root, readout = args.root.resolve(), args.readout_root.resolve()
    (root / "features").mkdir(parents=True, exist_ok=True)
    plan = load_plan(readout)
    sources, cohort, montages, locked = _campaign(args, plan)
    encode = ReveEncoder(args.device)
    extract_all(root, cohort, sources, montages, plan, args.num_workers, encode)
    extract_level(root, plan, encode)
    extract_extra(root, args, cohort, locked, plan, encode)


def run_smoke(args: argparse.Namespace) -> None:
    """Load the weights and embed 4 real segments of the first patient; it writes no feature."""
    plan = load_plan(args.readout_root.resolve())
    sources, cohort, montages, _ = _campaign(args, plan)
    encode = ReveEncoder(args.device)
    first = sorted(cohort.labels)[0]
    view = segment_view(_frame(sources, cohort.rows[first][:4]), montages)
    for raw in _segments(view, 0):
        embedding = embed_segment(zscore_channels(common_average(raw)), encode)
        logger.info(
            f"Segment {raw.shape}, peak {np.abs(raw).max():.2e}, embedding {embedding.shape}, "
            f"mean {embedding.mean():.3f}, sd {embedding.std():.3f}"
        )
    logger.info("Smoke read done")


# ---------------------------------------------------------------------- step 0


@dataclass
class Cohort:
    """The patients of the readout in sorted order, with their folds, labels and covariates."""

    subject: np.ndarray
    fold: np.ndarray
    label: np.ndarray
    status: np.ndarray
    rate: np.ndarray
    segments: dict[str, int]
    age: dict[str, float]


def load_cohort(readout: Path, audit: Path, plan: dict[str, Any]) -> Cohort:
    """The patients, drug status and age from the files of the readout and the audit."""
    overrides = {"subject": pl.Utf8}
    patients = pl.read_csv(readout / "patients.csv", schema_overrides=overrides).sort("subject")
    subject = np.array(patients["subject"].to_list())
    exposure = pl.read_csv(audit / "drug_exposure.csv", schema_overrides=overrides)
    status = dict(zip(exposure["subject"].to_list(), exposure["status"].to_list()))
    covariates = pl.read_csv(audit / "covariates.csv", schema_overrides=overrides)
    age = dict(zip(covariates["subject"].to_list(), covariates["age"].cast(pl.Float64).to_list()))
    return Cohort(
        subject,
        np.array([plan["folds"][s] for s in subject]),
        np.array([plan["labels"][s] for s in subject]),
        np.array([status[s] for s in subject]),
        patients["native_rate"].cast(pl.Float64).to_numpy(),
        dict(zip(subject.tolist(), patients["segments"].to_list())),
        {s: float(age[s]) for s in subject},
    )


def saved_scores(path: Path, cohort: Cohort) -> ArmScores:
    """The saved held-out scores of an arm of the readout, aligned by patient.

    Raises
    ------
    ValueError
        If the file does not hold exactly these patients in these folds.
    """
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8}).sort("subject")
    if frame["subject"].to_list() != cohort.subject.tolist() or frame["fold"].to_list() != cohort.fold.tolist():
        raise ValueError(f"{path.name} does not hold the patients of the plan")
    return ArmScores(cohort.subject, cohort.fold, cohort.label, frame["score"].to_numpy())


def run_step0(root: Path, readout: Path, audit: Path, n_splits: int = N_SPLITS) -> dict[str, Any]:
    """Section 5.1: the head and the split on the band-power and HYDRA features that exist. It decides nothing.

    The random splits scale once over all patients and rotate wide rows to their own span, which keeps the L2 scores
    and makes 1,000 splits take minutes. These steps use no label.
    """
    root.mkdir(parents=True, exist_ok=True)
    plan = load_plan(readout)
    cohort = load_cohort(readout, audit, plan)
    report: dict[str, Any] = {"rule": "protocol section 5.1, decides nothing"}
    for name in ("Spectral-all", "HYDRA-all"):
        data = np.load(readout / "features" / f"{name}.npz")
        x = patient_rows(data["features"], data["subject"].astype(str), cohort.subject).astype(np.float64)
        x = StandardScaler().fit_transform(x)
        rotated = rotate_rows(x)
        saved = saved_scores(readout / "scores" / f"{name}.csv", cohort)
        mlp = fold_scores(rotated, cohort.label, cohort.fold, mlp_scores, plan["seed"])
        arm = ArmScores(cohort.subject, cohort.fold, cohort.label, mlp)
        report[name] = {
            "linear_folds": alone_interval(saved),
            "mlp_folds": alone_interval(arm),
            "mlp_minus_linear": paired_interval(arm, saved),
            "linear_splits": split_spread(rotated, cohort.label, linear_scores, n_splits, plan["seed"]),
        }
        logger.info(f"Step 0, {name}: {report[name]}")
    _save_json(root / "step0.json", report)
    return report


# ---------------------------------------------------------------------- gates


def _load_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["subject"].astype(str), data["label"], data["features"]


def fixed_rows(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The segment rows of the deciding read."""
    data = np.load(_feature_path(root, "REVE-all"))
    fixed = data["fixed"]
    return data["subject"].astype(str)[fixed], data["label"][fixed], data["features"][fixed]


def helwan_read(root: Path, cohort: Cohort, seed: int) -> ArmScores:
    """Read ``H``: 60 random 5 s windows per patient and the probe of Helwan et al., on our folds."""
    subject, label, x = _load_rows(_feature_path(root, "REVE-H"))
    fold_of = dict(zip(cohort.subject.tolist(), cohort.fold.tolist()))
    folds = np.array([fold_of[s] for s in subject])
    scores = {}
    for f in range(N_FOLDS):
        patients, values = helwan_scores(x, label, subject, folds == f, seed + f)
        scores.update(dict(zip(patients.tolist(), values.tolist())))
    return ArmScores(cohort.subject, cohort.fold, cohort.label, np.array([scores[s] for s in cohort.subject]))


def run_gates(root: Path, readout: Path, audit: Path) -> dict[str, Any]:
    """The gates of section 7. The permuted head uses a fixed ``C``, so no head fits the real labels of read ``D``."""
    plan = load_plan(readout)
    cohort = load_cohort(readout, audit, plan)
    fold_of = dict(zip(cohort.subject.tolist(), cohort.fold.tolist()))
    subject, label, x = fixed_rows(root)
    rng = np.random.default_rng(plan["seed"])
    folds = np.array([fold_of[s] for s in subject])
    draws = []
    for _ in range(N_PERMUTATIONS):
        values = []
        for f in range(N_FOLDS):
            patients, pooled = permuted_scores(x, label, subject, folds == f, 1.0, StandardScaler, rng)
            values.append(auroc(np.array([plan["labels"][s] for s in patients]), pooled))
        draws.append(float(np.mean(values)))
    spectral = np.load(readout / "features" / "Spectral.npz")
    age_spectral = age_correlation(spectral["features"], spectral["subject"].astype(str), cohort.age, fold_of)
    age_encoder = age_correlation(x, subject, cohort.age, fold_of)
    repeat = json.loads((root / "repeat.json").read_text())["max_difference"]
    valid = [np.load(_feature_path(root, name))["valid"] for name in ("REVE-all", "REVE-extra")]
    failures = float(1.0 - np.concatenate(valid).mean())
    helwan = helwan_read(root, cohort, plan["seed"])
    report = gate_report(float(np.mean(draws)), age_encoder, age_spectral, repeat, failures, mean_fold_auroc(helwan))
    _save_json(root / "gates.json", report)
    logger.info(f"Gates: {report}")
    return report


# ---------------------------------------------------------------------- read


def _segment_head(x: np.ndarray, label: np.ndarray, subject: np.ndarray, cohort: Cohort, seed: int) -> ArmScores:
    """The head of the frozen readout on segment rows, over the outer folds."""
    fold_of = dict(zip(cohort.subject.tolist(), cohort.fold.tolist()))
    folds = np.array([fold_of[s] for s in subject])
    scores: dict[str, float] = {}
    for f in range(N_FOLDS):
        result = fit_head(x, label, subject, folds == f, StandardScaler, seed=seed + f)
        scores.update(dict(zip(result.subject.tolist(), result.score.tolist())))
    return ArmScores(cohort.subject, cohort.fold, cohort.label, np.array([scores[s] for s in cohort.subject]))


def _level_head(root: Path, cohort: Cohort, seed: int) -> ArmScores:
    """Read ``D-level``: each fold has its own features, because the scale comes from its training patients."""
    scores: dict[str, float] = {}
    fold_of = dict(zip(cohort.subject.tolist(), cohort.fold.tolist()))
    for f in range(N_FOLDS):
        subject, label, x = _load_rows(_feature_path(root, f"REVE-level_f{f}"))
        folds = np.array([fold_of[s] for s in subject])
        result = fit_head(x, label, subject, folds == f, StandardScaler, seed=seed + f)
        scores.update(dict(zip(result.subject.tolist(), result.score.tolist())))
    return ArmScores(cohort.subject, cohort.fold, cohort.label, np.array([scores[s] for s in cohort.subject]))


def _patient_arm(x: np.ndarray, cohort: Cohort, fit: Callable[..., np.ndarray], seed: int) -> ArmScores:
    return ArmScores(cohort.subject, cohort.fold, cohort.label, fold_scores(x, cohort.label, cohort.fold, fit, seed))


def read_summary(name: str, arm: ArmScores, spectral: ArmScores, cohort: Cohort, n_boot: int) -> dict[str, Any]:
    """Every number that section 5.3 asks of a read."""
    value, low, high = alone_interval(arm, n_boot)
    difference = paired_interval(arm, spectral, n_boot=n_boot)
    gain = drug_gain(arm.score, cohort.label, cohort.fold, cohort.status, n_boot)
    z = standardize_within_fold(arm.score, cohort.fold)
    epilepsy_off = (cohort.label == 1) & (cohort.status == "off")
    non_epilepsy_off = (cohort.label == 0) & (cohort.status == "off")
    cell = group_read(z, epilepsy_off, non_epilepsy_off, cohort.fold, n_boot)
    links = bag_size_link(arm, cohort.segments)
    at_250 = cohort.rate == 250.0
    return {
        "read": name,
        "auroc": value,
        "low": low,
        "high": high,
        "minus_spectral": difference[0],
        "minus_spectral_low": difference[1],
        "minus_spectral_high": difference[2],
        "gain": gain[0],
        "gain_low": gain[1],
        "gain_high": gain[2],
        "off_off": cell.value,
        "off_off_low": cell.low,
        "off_off_high": cell.high,
        "bag_link_epilepsy": links["epilepsy"],
        "bag_link_non_epilepsy": links["non_epilepsy"],
        "auroc_250_hz": auroc(cohort.label[at_250], arm.score[at_250]),
    }


def run_read(
    root: Path,
    readout: Path,
    audit: Path,
    accept_review: bool = False,
    n_boot: int = N_BOOTSTRAP,
    n_splits: int = N_SPLITS,
) -> dict[str, Any]:
    """The rule on read ``D``, then the descriptive reads. Writes the summary, the scores and the decision.

    Raises
    ------
    ValueError
        If the gates did not pass. A ``review`` of read ``H`` may go on only when the user asks for it.
    """
    gates_path = root / "gates.json"
    gates = json.loads(gates_path.read_text()) if gates_path.is_file() else {"status": "missing"}
    if gates["status"] not in ("pass", "review") or (gates["status"] == "review" and not accept_review):
        raise ValueError(f"the gates did not pass: {gates['status']}")
    plan = load_plan(readout)
    seed = plan["seed"]
    cohort = load_cohort(readout, audit, plan)
    spectral = saved_scores(readout / "scores" / "Spectral.csv", cohort)
    subject, label, x = fixed_rows(root)
    decisive = _segment_head(x, label, subject, cohort, seed)
    difference = paired_interval(decisive, spectral, n_boot=n_boot)
    gain = drug_gain(decisive.score, cohort.label, cohort.fold, cohort.status, n_boot)
    outcome = encoder_outcome(difference, gain)

    data = np.load(_feature_path(root, "REVE-all"))
    every_subject, every_x = data["subject"].astype(str), data["features"]
    extra = np.load(_feature_path(root, "REVE-extra"))
    extra_subject, extra_x = extra["subject"].astype(str), extra["features"]
    seizure, high_rate = extra["seizure"], extra["high_rate"]

    def with_extra(keep: np.ndarray) -> np.ndarray:
        rows = np.concatenate([every_x, extra_x[keep]])
        names = np.concatenate([every_subject, extra_subject[keep]])
        return patient_rows(rows, names, cohort.subject).astype(np.float64)

    campaign = with_extra(np.zeros(len(extra_subject), bool))
    closest = with_extra(np.ones(len(extra_subject), bool))
    arms = {
        "D": decisive,
        "H": helwan_read(root, cohort, seed),
        "D-level": _level_head(root, cohort, seed),
        "A": _patient_arm(campaign, cohort, linear_scores, seed),
        "A-mlp": _patient_arm(campaign, cohort, mlp_scores, seed),
        "A-seiz": _patient_arm(with_extra(seizure & ~high_rate), cohort, mlp_scores, seed),
        "A-rates": _patient_arm(with_extra(high_rate & ~seizure), cohort, mlp_scores, seed),
        "C": _patient_arm(closest, cohort, mlp_scores, seed),
    }
    (root / "scores").mkdir(exist_ok=True)
    rows = []
    for name, arm in arms.items():
        rows.append(read_summary(name, arm, spectral, cohort, n_boot))
        pl.DataFrame(
            {
                "subject": arm.subject.tolist(),
                "fold": arm.fold.tolist(),
                "label": arm.label.tolist(),
                "score": arm.score.tolist(),
            }
        ).write_csv(root / "scores" / f"{name}.csv")
        logger.info(f"Read {name}: {rows[-1]['auroc']:.3f}")
    pl.DataFrame(rows).write_csv(root / "encoder_summary.csv")

    known = cohort.status != "unknown"
    spread = split_spread(closest, cohort.label, mlp_scores, n_splits, seed)
    decision = {
        "rule": "protocol section 7",
        "outcome": outcome,
        "gates": gates["status"],
        "accepted_review": bool(accept_review and gates["status"] == "review"),
        "difference_from_spectral": difference,
        "gain_over_drug": gain,
        "gain_ceiling": gain_ceiling(cohort.label, cohort.status == "on", known),
        "D": rows[0],
        "spread_of_C_over_random_splits": spread,
        "extra_patients": int(len(set(extra_subject.tolist()))),
        "extra_segments": {
            "seizure_only": int((seizure & ~high_rate).sum()),
            "high_rate_only": int((high_rate & ~seizure).sum()),
            "both": int((seizure & high_rate).sum()),
        },
        "high_rates": list(HIGH_RATES),
    }
    _save_json(root / "encoder_decision.json", decision)
    logger.info(f"Frozen encoder check: {outcome}")
    return decision


# ---------------------------------------------------------------------- command line


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, description: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        sub.add_argument("--audit-root", type=Path, required=True, help="the root of the covariate audit")
        return sub

    def corpus(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        sub.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        sub.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        sub.add_argument("--num-workers", type=int, default=4)
        sub.add_argument("--device", default="cuda")

    step0 = command("step0", "read the protocol on existing features; decides nothing")
    step0.add_argument("--splits", type=int, default=N_SPLITS)
    corpus(command("smoke", "load the weights and embed 4 segments"))
    corpus(command("extract", "run the encoder on every read, with no label"))
    command("gates", "apply the gates")
    read = command("read", "apply the rule and write the descriptive reads")
    read.add_argument("--accept-review", action="store_true", help="go on after a miss of read H")
    read.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()
    root, readout, audit = args.root.resolve(), args.readout_root.resolve(), args.audit_root.resolve()
    if args.command == "step0":
        run_step0(root, readout, audit, args.splits)
    elif args.command == "smoke":
        run_smoke(args)
    elif args.command == "extract":
        run_extract(args)
    elif args.command == "gates":
        run_gates(root, readout, audit)
    else:
        run_read(root, readout, audit, args.accept_review, n_splits=args.splits)


if __name__ == "__main__":
    main()
