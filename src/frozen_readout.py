"""Frozen dictionary feature readout: do separable peak features carry signal beyond controls?

Protocol: ``documentation/frozen_feature_readout_protocol.md``. Run on the HPC from ``code/``; a root holds one
pass. ``src/frozen_readout_core.py`` holds the pure logic and the rule. Subcommands, in order:

- ``folds`` fixes the outer folds and the segment count from the plans, without reading signals, and records the
  baselines of bag size and native rate.
- ``control`` is the positive control of the feature code on planted events (no classifier).
- ``fit`` trains (or, with ``--arm untrained``, only initializes) the separable dictionary of one outer fold.
- ``extract`` writes the features of one arm: per fold for dictionary arms, once for fixed arms.
- ``heads`` fits the heads, the sparse head and the permutation null.
- ``summarize`` applies the rule and writes ``readout_summary.csv`` and ``readout_decision.json``.
- ``review`` writes the top templates of each fold for the explanation review.

The 39 test patients stay locked: only their IDs are read, to refuse them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import rootutils
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.frozen_readout_core import (  # noqa: E402
    DECIDING,
    MAX_SEGMENTS,
    N_FOLDS,
    N_PERMUTATIONS,
    PASS_SEEDS,
    PLANTED_FRACTION_MIN,
    PRIMARY,
    SPARSE_TEMPLATES,
    ArmScores,
    PeakTable,
    assign_folds,
    auroc,
    bag_size_link,
    choose_z,
    conditions,
    dictionary_split,
    fit_head,
    fit_sparse_head,
    low_pass,
    mean_fold_auroc,
    paired_interval,
    peak_counts,
    permuted_scores,
    planted_fraction,
    pooled_auroc,
    rate_features,
    rate_guard,
    readout_outcome,
    segment_draw,
    training_passed,
)
from src.models.components.hydra_transformer import _SparseScaler  # noqa: E402
from src.models.components.waveform_sae import CHECKPOINT_NAME  # noqa: E402
from src.separable_factor_analysis import best_lag, dominant_frequency  # noqa: E402
from src.separable_synthetic import (  # noqa: E402
    N_PROJECTIONS,
    PEAK_RADIUS,
    PlantedEvents,
    _git,
    build_model,
    json_safe,
    local_maxima,
    stage_1b_config,
)

EXTRACT_BATCH = 8
PLANTED_AMPLITUDE = 6
# The four event templates of the positive control (protocol section 7; projections count from 0).
CONTROL_ROOT = "logs/separable_synthetic/gate1b_random_none_20261030_124956"
CONTROL_TEMPLATES = (
    ("1b_s4ext_a6_s0", 0, 29, {"T3": 2, "C4": 3}),
    ("1b_s4ext_a6_s1", 1, 51, {"T3": 0, "C4": 2}),
)
TOP_TEMPLATES, TOP_SEGMENTS = 10, 5
RECUR_WAVEFORM, RECUR_PROJECTION = 0.9, 0.8


@dataclass(frozen=True)
class Arm:
    """How an arm is read: ``nine`` (fixed segments), ``all`` (weighted windows) or ``patient`` (patient-mean rows)."""

    read: str
    per_fold: bool
    scaler: str


ARMS = {
    "S4-proj": Arm("nine", True, "sparse"),
    "S4-atom": Arm("nine", True, "sparse"),
    "S4-R1": Arm("nine", True, "sparse"),
    "S4-untrained": Arm("nine", True, "sparse"),
    "Spectral": Arm("nine", False, "standard"),
    "HYDRA": Arm("nine", False, "sparse"),
    "S4-proj-all": Arm("all", True, "sparse"),
    "Spectral-all": Arm("all", False, "standard"),
    # 10,000 windows of about 12,000 features do not fit in memory, so this read is one mean row per patient.
    "HYDRA-all": Arm("patient", False, "sparse"),
}
FIXED_CONFIGS = {
    "Spectral": "spectral",
    "HYDRA": "hydra_transformer",
    "Spectral-all": "spectral",
    "HYDRA-all": "hydra_transformer",
}


class SparseRows:
    """``_SparseScaler`` on numpy rows, with the interface of the head."""

    def fit(self, x: np.ndarray) -> SparseRows:
        """Fit the scaler."""
        self.inner = _SparseScaler()
        self.inner.fit(torch.from_numpy(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale the rows."""
        return self.inner.transform(torch.from_numpy(x)).numpy()


SCALERS: dict[str, Callable[[], Any]] = {"sparse": SparseRows, "standard": StandardScaler}


# ---------------------------------------------------------------------- patients and views


@dataclass
class Cohort:
    """The training and validation patients: labels, plan rows in sorted key order, recordings and native rate."""

    labels: dict[str, int]
    rows: dict[str, list[tuple[str, int]]]
    keys: dict[str, list[tuple[str, float]]]
    recordings: dict[str, int]
    rates: dict[str, float]


def build_cohort(sources: dict[str, Any], locked: set[str]) -> Cohort:
    """Read the patients from the plans of ``sources``, without reading signals.

    Raises
    ------
    ValueError
        If a locked patient appears, or a patient is in two sources or has mixed labels.
    """
    labels: dict[str, int] = {}
    entries: dict[str, list[tuple[tuple[str, float], tuple[str, int], float]]] = {}
    origin: dict[str, str] = {}
    for name, source in sources.items():
        plan = source.plan
        for row, (subject, label, start, description) in enumerate(
            zip(
                plan["subject"].astype(str),
                plan["epilepsy"].astype(int),
                plan["start"].astype(float),
                plan["description_row"],
            )
        ):
            if subject in locked:
                raise ValueError(f"locked test patient {subject} in the {name} plan")
            if origin.setdefault(subject, name) != name:
                raise ValueError(f"patient {subject} is in two sources")
            if labels.setdefault(subject, int(label)) != int(label):
                raise ValueError(f"patient {subject} has mixed labels")
            key = (str(description["path"]), float(start))
            entries.setdefault(subject, []).append((key, (name, row), float(description["sfreq"])))
    rows, keys, recordings, rates = {}, {}, {}, {}
    for subject, items in entries.items():
        items.sort(key=lambda item: item[0])
        keys[subject] = [item[0] for item in items]
        rows[subject] = [item[1] for item in items]
        recordings[subject] = len({item[0][0] for item in items})
        counts = Counter(item[2] for item in items)
        rates[subject] = min(counts, key=lambda rate: (-counts[rate], rate))
    return Cohort(labels, rows, keys, recordings, rates)


def fixed_rows(cohort: Cohort, patients: list[str], n: int) -> list[tuple[str, int]]:
    """The ``n`` drawn segments of each patient, patient by patient."""
    chosen: list[tuple[str, int]] = []
    for subject in sorted(patients):
        chosen += [cohort.rows[subject][i] for i in segment_draw(subject, cohort.keys[subject], n)]
    return chosen


def every_row(cohort: Cohort, patients: list[str]) -> list[tuple[str, int]]:
    """All segments of each patient, patient by patient."""
    return [row for subject in sorted(patients) for row in cohort.rows[subject]]


class PooledSegments(Dataset):
    """Selected rows of several sources as one dataset, with an aligned ``plan`` of subject and label."""

    def __init__(self, sources: dict[str, Any], rows: list[tuple[str, int]]) -> None:
        self.sources, self.rows = sources, rows
        self.target_channels = list(next(iter(sources.values())).target_channels)
        subjects = {name: source.plan["subject"].astype(str).to_numpy() for name, source in sources.items()}
        labels = {name: source.plan["epilepsy"].astype(int).to_numpy() for name, source in sources.items()}
        self.plan = pd.DataFrame(
            {
                "subject": [subjects[name][row] for name, row in rows],
                "epilepsy": [labels[name][row] for name, row in rows],
            }
        )

    def __len__(self) -> int:
        """Number of selected rows."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The segment and label of a selected row."""
        name, row = self.rows[index]
        return self.sources[name][row]


class BandLimited(Dataset):
    """A dataset with the low-pass of protocol section 3 on every segment, before the row scaling."""

    def __init__(self, base: Any, sfreq: float) -> None:
        self.base, self.sfreq = base, sfreq
        self.plan, self.target_channels = base.plan, base.target_channels

    def __len__(self) -> int:
        """Number of segments."""
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The band-limited segment and its label."""
        x, y = self.base[index]
        return torch.from_numpy(low_pass(x.numpy(), self.sfreq).copy()), y


def load_sources(args: argparse.Namespace, seed: int = 0) -> tuple[Any, Any, dict[str, Any], set[str], dict[str, Any]]:
    """The composed configuration, the data module, its training and validation sources with their signal cache,
    the locked test IDs, and the audit. No signal is read here.

    Raises
    ------
    ValueError
        If the patient counts changed or a signal cache is cold.
    """
    import hydra

    from src.run_patient_waveform_sweep import check_audit
    from src.utils import check_split_disjoint

    manifests, cache_root = args.manifests.resolve(), args.signal_cache_dir.expanduser().resolve()
    audit = check_audit(manifests)
    existing = {p.name for p in cache_root.iterdir() if p.is_dir()}
    cfg = stage_1b_config(
        manifests, args.data_dir.resolve(), args.num_workers, audit["skip_start_s"], seed, "none", "random"
    )
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    for split, patients in (("train", 117), ("val", 39), ("test", 39)):
        if getattr(datamodule, f"{split}_df").subject.nunique() != patients:
            raise ValueError(f"{split} patient count changed during loading")
    sources = {"train": datamodule.data_train, "val": datamodule.data_val}
    for name, source in sources.items():
        source.enable_signal_cache(cache_root)
        if source.signal_cache.identity not in existing:
            raise ValueError(f"{name} signals have no entries under {cache_root}; pass the campaign cache")
    locked = set(datamodule.test_df["subject"].astype(str))
    return cfg, datamodule, sources, locked, audit


def load_plan(root: Path) -> dict[str, Any]:
    """The fold plan of a root (``folds.json``)."""
    return json.loads((root / "folds.json").read_text())


def _save_json(path: Path, value: object) -> None:
    from src.run_patient_waveform_sweep import save_json

    save_json(path, json_safe(value))


def write_folds(args: argparse.Namespace) -> dict[str, Any]:
    """Fix the outer folds and the segment count of a root from the plans, and record the baselines.

    Raises
    ------
    ValueError
        If the root already holds a different plan.
    """
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _, _, sources, locked, audit = load_sources(args)
    cohort = build_cohort(sources, locked)
    seed = PASS_SEEDS[args.pass_index]
    folds = assign_folds(cohort.labels, seed)
    bags = {s: len(rows) for s, rows in cohort.rows.items()}
    plan = {
        "pass": args.pass_index,
        "seed": seed,
        "n_segments": min(MAX_SEGMENTS, min(bags.values())),
        "smallest_bag": min(bags.values()),
        "skip_start_s": audit["skip_start_s"],
        "folds": folds,
        "labels": cohort.labels,
        "commit": _git("rev-parse", "HEAD").strip(),
    }
    path = root / "folds.json"
    if path.is_file():
        old = load_plan(root)
        if any(old[key] != plan[key] for key in ("pass", "seed", "n_segments", "folds", "labels")):
            raise ValueError(f"{path} holds a different plan; use a new root")
        return old
    _save_json(path, plan)
    subjects = sorted(cohort.labels)
    pl.DataFrame(
        {
            "subject": subjects,
            "label": [cohort.labels[s] for s in subjects],
            "fold": [folds[s] for s in subjects],
            "segments": [bags[s] for s in subjects],
            "recordings": [cohort.recordings[s] for s in subjects],
            "native_rate": [cohort.rates[s] for s in subjects],
        }
    ).write_csv(root / "patients.csv")
    labels = np.array([cohort.labels[s] for s in subjects])
    baselines = {
        "segments_per_patient": auroc(labels, np.array([bags[s] for s in subjects], dtype=float)),
        "recordings_per_patient": auroc(labels, np.array([cohort.recordings[s] for s in subjects], dtype=float)),
        "native_rate": auroc(labels, np.array([cohort.rates[s] for s in subjects])),
        "patients_by_class_and_rate": {
            f"{label}|{rate}": count
            for (label, rate), count in Counter((cohort.labels[s], cohort.rates[s]) for s in subjects).items()
        },
    }
    _save_json(root / "baselines.json", baselines)
    logger.info(f"Pass {args.pass_index}: {len(subjects)} patients, {plan['n_segments']} segments each; {baselines}")
    return plan


# ---------------------------------------------------------------------- peaks


@dataclass
class PatientPeaks:
    """The peaks of one patient under one dictionary, with its per-segment code features."""

    subject: str
    projection: PeakTable
    atom: PeakTable | None
    r1: np.ndarray | None
    minutes: float


def batch_peaks(responses: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment, template and height of every peak of responses ``(B, M, T)``: the peak rule of section 10.6."""
    segment, template, position = torch.nonzero(local_maxima(responses, PEAK_RADIUS), as_tuple=True)
    height = responses[segment, template, position]
    return segment.cpu().numpy(), template.cpu().numpy(), height.cpu().numpy()


def stream_patients(
    model: Any, view: Any, sfreq: float, num_workers: int, details: bool = False
) -> Iterator[PatientPeaks]:
    """One ``PatientPeaks`` per patient, in plan order; a patient's rows must be contiguous.

    Template ``k * D + d`` is atom ``k``, projection ``d``. With ``details``, also the peaks of the response after
    the maximum over projections and the won amplitudes of the ``R=1`` code.
    """
    subjects = view.plan["subject"].astype(str).to_numpy()
    n_atoms = model.spec.n_atoms
    n_templates = n_atoms * model.spec.n_projections
    index, position, current, minutes = 0, 0, None, 0.0

    def fresh() -> dict[str, list[np.ndarray]]:
        names = ("seg_p", "tmp_p", "hgt_p", "seg_a", "tmp_a", "hgt_a", "r1")
        return {name: [] for name in names}

    def finish(subject: str, n_segments: int) -> PatientPeaks:
        def table(prefix: str, templates: int) -> PeakTable:
            parts = [np.concatenate(stores[f"{name}_{prefix}"]) for name in ("seg", "tmp", "hgt")]
            return PeakTable(parts[0].astype(np.int64), parts[1].astype(np.int64), parts[2], n_segments, templates)

        return PatientPeaks(
            subject,
            table("p", n_templates),
            table("a", n_atoms) if details else None,
            np.stack(stores["r1"]) if details else None,
            minutes,
        )

    stores: dict[str, list[np.ndarray]] = fresh()
    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=num_workers):
        with torch.no_grad():
            rows, _ = model._rows(x)
            raw = model._correlate(F.pad(rows, model.padding), model._dictionary())
            peaks_p = batch_peaks(F.relu(raw).flatten(1, 2))
            peaks_a = batch_peaks(F.relu(raw.amax(dim=2))) if details else None
            r1 = None
            if details:
                code, _ = model.encode(x)
                r1 = (code.sum(-1) / (code.shape[-1] // model.spec.pool_len)).cpu().numpy()
        minutes = x.shape[-1] / sfreq / 60.0
        for j in range(len(x)):
            subject = subjects[index]
            index += 1
            if subject != current:
                if current is not None:
                    yield finish(current, position)
                current, position, stores = subject, 0, fresh()
            where = peaks_p[0] == j
            stores["seg_p"].append(np.full(int(where.sum()), position))
            stores["tmp_p"].append(peaks_p[1][where])
            stores["hgt_p"].append(peaks_p[2][where])
            if details:
                where = peaks_a[0] == j
                stores["seg_a"].append(np.full(int(where.sum()), position))
                stores["tmp_a"].append(peaks_a[1][where])
                stores["hgt_a"].append(peaks_a[2][where])
                stores["r1"].append(r1[j])
            position += 1
    if current is not None:
        yield finish(current, position)


def _load_dictionary(folder: Path, cfg: Any, seed: int, device: str) -> Any:
    import hydra

    spec = hydra.utils.instantiate(cfg.feature.spec)
    train_spec = hydra.utils.instantiate(cfg.feature.train_spec)
    model = build_model("s4ext", spec, train_spec, seed, device)
    model.load_checkpoint(folder / CHECKPOINT_NAME)
    return model


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def positive_control(args: argparse.Namespace) -> dict[str, Any]:
    """Do the features see planted events? Paired rates with and without events, for four event templates."""
    root = args.root.resolve()
    plan = load_plan(root)
    cfg, _, sources, locked, _ = load_sources(args)
    sfreq = float(cfg.data.target_sfreq)
    cohort = build_cohort(sources, locked)
    val = sources["val"]
    patients = [s for s, rows in cohort.rows.items() if rows[0][0] == "val" and cohort.labels[s] == 0]
    indices = np.array([row for _, row in fixed_rows(cohort, patients, plan["n_segments"])])
    templates = []
    for arm, seed, atom, projections in CONTROL_TEMPLATES:
        model = _load_dictionary(args.control_root.resolve() / arm, cfg, seed, _device())

        def read(amplitude: int, model: Any = model, seed: int = seed) -> list[PatientPeaks]:
            view = BandLimited(PlantedEvents(val, indices, amplitude, seed, "val", sfreq), sfreq)
            return list(stream_patients(model, view, sfreq, args.num_workers))

        plain, planted = read(0), read(PLANTED_AMPLITUDE)
        minutes = plain[0].minutes
        z, rates = choose_z([p.projection for p in plain], minutes)
        for focus, projection in projections.items():
            template = atom * N_PROJECTIONS + projection

            def rate(table: PeakTable, template: int = template) -> float:
                return peak_counts(table, z)[:, template].sum() / (table.n_segments * minutes)

            without = np.array([rate(p.projection) for p in plain])
            with_events = np.array([rate(p.projection) for p in planted])
            fraction = planted_fraction(with_events, without)
            templates.append(
                {
                    "arm": arm,
                    "atom": atom,
                    "projection": projection,
                    "focus": focus,
                    "z": z,
                    "z_rates": {str(k): v for k, v in rates.items()},
                    "mean_rate_without": float(without.mean()),
                    "mean_rate_with": float(with_events.mean()),
                    "fraction_of_patients_higher": fraction,
                }
            )
            logger.info(f"Positive control {arm} {focus}: {fraction:.3f} of patients higher, z {z}")
    record = {
        "rule": "protocol section 7",
        "patients": len(patients),
        "passed": all(t["fraction_of_patients_higher"] >= PLANTED_FRACTION_MIN for t in templates),
        "templates": templates,
        "commit": _git("rev-parse", "HEAD").strip(),
    }
    _save_json(root / "positive_control.json", record)
    return record


# ---------------------------------------------------------------------- dictionaries


def dictionary_dir(root: Path, arm: str, fold: int) -> Path:
    """The directory of one dictionary of a root."""
    return root / "dictionaries" / f"{arm}_f{fold}"


def fit_dictionary(args: argparse.Namespace) -> None:
    """Train the separable dictionary of an outer fold, or with ``--arm untrained`` keep its random start.

    Both arms share a seed, so they share a start. A finished arm is skipped. The arm directory is created only
    after the configuration and the data are ready, so a setup error leaves nothing behind.

    Raises
    ------
    RuntimeError
        Without a CUDA GPU.
    """
    import hydra
    import lightning

    from src.utils import split_provenance

    root = args.root.resolve()
    output = dictionary_dir(root, args.arm, args.fold)
    if (output / CHECKPOINT_NAME).is_file():
        logger.info(f"{output.name}: finished")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("fit a dictionary inside the HPC container on an allocated CUDA GPU")
    plan = load_plan(root)
    seed = plan["seed"] + args.fold
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    cfg, datamodule, sources, locked, _ = load_sources(args, seed)
    sfreq = float(cfg.data.target_sfreq)
    cohort = build_cohort(sources, locked)
    train_patients, val_patients = dictionary_split(plan["labels"], plan["folds"], args.fold, plan["seed"])
    views = {
        "train": BandLimited(PooledSegments(sources, every_row(cohort, train_patients)), sfreq),
        "val": BandLimited(PooledSegments(sources, fixed_rows(cohort, val_patients, plan["n_segments"])), sfreq),
    }
    spec = hydra.utils.instantiate(cfg.feature.spec)
    train_spec = hydra.utils.instantiate(cfg.feature.train_spec)
    if args.arm == "untrained":
        train_spec = replace(train_spec, epochs=1, lr=0.0)
    provenance = split_provenance(datamodule, cfg.data)
    provenance["train_subjects"] = train_patients
    provenance["frozen_readout"] = {"pass": plan["pass"], "fold": args.fold, "arm": args.arm}
    base = datamodule.train_dataloader()
    loaders = [
        DataLoader(views[split], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for split in ("train", "val")
    ]
    lightning.seed_everything(seed, workers=True)
    model = build_model("s4ext", spec, train_spec, seed, "cuda")
    output.mkdir(parents=True, exist_ok=True)
    _save_json(
        output / "fit_provenance.json",
        {
            "pass": plan["pass"],
            "fold": args.fold,
            "arm": args.arm,
            "seed": seed,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "commit": _git("rev-parse", "HEAD").strip(),
            "uncommitted": _git("status", "--porcelain").splitlines(),
        },
    )
    model.fit_unsupervised(*loaders, provenance=provenance)
    model.save_artifacts(output)


# ---------------------------------------------------------------------- features


def feature_path(root: Path, arm: str, fold: int | None) -> Path:
    """The feature file of an arm: one per fold for dictionary arms, one for fixed arms."""
    return root / "features" / (f"{arm}_f{fold}.npz" if ARMS[arm].per_fold else f"{arm}.npz")


def _save_features(
    path: Path, subject: list[str], labels: dict[str, int], features: np.ndarray, **extra: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        subject=np.array(subject),
        label=np.array([labels[s] for s in subject]),
        features=features.astype(np.float32),
        **extra,
    )


def extract_dictionary_arms(args: argparse.Namespace) -> None:
    """Write the features of one dictionary of an outer fold, on the fixed segments or on all segments.

    ``z`` comes from this fold's training patients on the fixed segments, without labels. The all-segment read reuses
    it, so run it after the fixed read.
    """
    root = args.root.resolve()
    plan = load_plan(root)
    seed = plan["seed"] + args.fold
    cfg, _, sources, locked, _ = load_sources(args, seed)
    sfreq = float(cfg.data.target_sfreq)
    cohort = build_cohort(sources, locked)
    patients = sorted(cohort.labels)
    model = _load_dictionary(dictionary_dir(root, args.dictionary, args.fold), cfg, seed, _device())
    labels, folds = plan["labels"], plan["folds"]
    z_path = root / "features" / f"z_{args.dictionary}_f{args.fold}.json"
    if args.read == "nine":
        rows = fixed_rows(cohort, patients, plan["n_segments"])
        view = BandLimited(PooledSegments(sources, rows), sfreq)
        details = args.dictionary == "trained"
        peaks = list(stream_patients(model, view, sfreq, args.num_workers, details))
        minutes = peaks[0].minutes
        training = [p for p in peaks if folds[p.subject] != args.fold]
        z, rates = choose_z([p.projection for p in training], minutes)
        record = {"z": z, "rates": {str(k): v for k, v in rates.items()}}
        subject = [p.subject for p in peaks for _ in range(p.projection.n_segments)]
        arm = "S4-proj" if details else "S4-untrained"
        projection = np.concatenate([rate_features(p.projection, z, minutes) for p in peaks])
        _save_features(
            feature_path(root, arm, args.fold), subject, labels, projection, key=np.array([f"{n}|{i}" for n, i in rows])
        )
        if details:
            z_atom, rates_atom = choose_z([p.atom for p in training], minutes)
            record.update(z_atom=z_atom, rates_atom={str(k): v for k, v in rates_atom.items()})
            atom = np.concatenate([rate_features(p.atom, z_atom, minutes) for p in peaks])
            _save_features(feature_path(root, "S4-atom", args.fold), subject, labels, atom)
            _save_features(
                feature_path(root, "S4-R1", args.fold), subject, labels, np.concatenate([p.r1 for p in peaks])
            )
        _save_json(z_path, record)
    else:
        z = json.loads(z_path.read_text())["z"]
        rows = every_row(cohort, patients)
        nine = set(fixed_rows(cohort, patients, plan["n_segments"]))
        view = BandLimited(PooledSegments(sources, rows), sfreq)
        subject, parts = [], []
        for p in stream_patients(model, view, sfreq, args.num_workers):
            subject += [p.subject] * p.projection.n_segments
            parts.append(rate_features(p.projection, z, p.minutes))
        _save_features(
            feature_path(root, "S4-proj-all", args.fold),
            subject,
            labels,
            np.concatenate(parts),
            in_nine=np.array([row in nine for row in rows]),
        )
    logger.info(f"Features of {args.dictionary} fold {args.fold}, read {args.read}: done")


def extract_fixed_arm(args: argparse.Namespace) -> None:
    """Write the features of a fixed arm (no dictionary), once for all folds."""
    from src.utils import instantiate_feature

    root = args.root.resolve()
    plan = load_plan(root)
    arm = ARMS[args.arm]
    cfg, _, sources, locked, _ = load_sources(args)
    sfreq = float(cfg.data.target_sfreq)
    cohort = build_cohort(sources, locked)
    patients = sorted(cohort.labels)
    nine = set(fixed_rows(cohort, patients, plan["n_segments"]))
    rows = fixed_rows(cohort, patients, plan["n_segments"]) if arm.read == "nine" else every_row(cohort, patients)
    view = BandLimited(PooledSegments(sources, rows), sfreq)
    subjects = view.plan["subject"].astype(str).to_numpy()
    extractor = instantiate_feature(OmegaConf.load(ROOT / "configs" / "feature" / f"{FIXED_CONFIGS[args.arm]}.yaml"))
    index = 0
    parts: list[np.ndarray] = []
    totals: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    with torch.no_grad():
        for x, y in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
            features = extractor(x, y).detach().cpu().float().numpy()
            if arm.read == "patient":
                for row in features:
                    subject = subjects[index]
                    totals[subject] = totals.get(subject, 0.0) + row
                    counts[subject] += 1
                    index += 1
            else:
                parts.append(features)
    if arm.read == "patient":
        ordered = sorted(totals)
        _save_features(
            feature_path(root, args.arm, None),
            ordered,
            plan["labels"],
            np.stack([totals[s] / counts[s] for s in ordered]),
        )
    else:
        extra = {"in_nine": np.array([row in nine for row in rows])} if arm.read == "all" else {}
        _save_features(
            feature_path(root, args.arm, None), list(subjects), plan["labels"], np.concatenate(parts), **extra
        )
    logger.info(f"Features of {args.arm}: done")


# ---------------------------------------------------------------------- heads


def _score_rows(arm: str, fold: int, result: Any) -> list[dict[str, Any]]:
    return [
        {"arm": arm, "subject": s, "fold": fold, "label": int(y), "score": float(v), "windows": int(n), "c": result.c}
        for s, y, v, n in zip(result.subject, result.label, result.score, result.n_windows)
    ]


def run_heads(args: argparse.Namespace) -> None:
    """Fit the head of every arm whose features exist, the sparse head and the permutation null.

    An arm with a score file is skipped. Fixed arms still fit their scaler and head per fold.
    """
    root = args.root.resolve()
    plan = load_plan(root)
    folds = plan["folds"]
    (root / "scores").mkdir(parents=True, exist_ok=True)
    (root / "coef").mkdir(parents=True, exist_ok=True)
    for name, arm in ARMS.items():
        path = root / "scores" / f"{name}.csv"
        if path.is_file():
            continue
        wanted = [feature_path(root, name, fold if arm.per_fold else None) for fold in range(N_FOLDS)]
        if not all(p.is_file() for p in wanted):
            logger.warning(f"{name}: features are missing; skipped")
            continue
        rows: list[dict[str, Any]] = []
        for fold, feature_file in enumerate(wanted):
            data = np.load(feature_file)
            subject, label, x = data["subject"].astype(str), data["label"], data["features"]
            held = np.array([folds[s] == fold for s in subject])
            result = fit_head(
                x,
                label,
                subject,
                held,
                SCALERS[arm.scaler],
                weighted=arm.read == "all",
                scaler_mask=data["in_nine"] if arm.read == "all" else None,
                seed=plan["seed"] + fold,
            )
            rows += _score_rows(name, fold, result)
            if name == PRIMARY:
                np.save(root / "coef" / f"{PRIMARY}_f{fold}.npy", result.coef)
            logger.info(f"Head {name} fold {fold}: C {result.c}, AUROC {auroc(result.label, result.score):.3f}")
        pl.DataFrame(rows).write_csv(path)
    _run_sparse(root, plan)
    _run_permutations(root, plan)


def _run_sparse(root: Path, plan: dict[str, Any]) -> None:
    path = root / "scores" / "S4-sparse.csv"
    files = [feature_path(root, PRIMARY, fold) for fold in range(N_FOLDS)]
    if path.is_file() or not all(p.is_file() for p in files):
        return
    rows, kept = [], {}
    for fold, feature_file in enumerate(files):
        data = np.load(feature_file)
        subject, label, x = data["subject"].astype(str), data["label"], data["features"]
        held = np.array([plan["folds"][s] == fold for s in subject])
        result, templates = fit_sparse_head(
            x, label, subject, held, SCALERS["sparse"], SPARSE_TEMPLATES, seed=plan["seed"] + fold
        )
        rows += _score_rows("S4-sparse", fold, result)
        kept[str(fold)] = templates.tolist()
    pl.DataFrame(rows).write_csv(path)
    _save_json(root / "coef" / "S4-sparse_templates.json", kept)


def _run_permutations(root: Path, plan: dict[str, Any]) -> None:
    path = root / "permutation_null.json"
    scores = root / "scores" / f"{PRIMARY}.csv"
    if path.is_file() or not scores.is_file():
        return
    selected = {row["fold"]: row["c"] for row in pl.read_csv(scores).select("fold", "c").unique().to_dicts()}
    rng = np.random.default_rng(plan["seed"])
    draws = []
    for _ in range(N_PERMUTATIONS):
        values = []
        for fold in range(N_FOLDS):
            data = np.load(feature_path(root, PRIMARY, fold))
            subject, label, x = data["subject"].astype(str), data["label"], data["features"]
            held = np.array([plan["folds"][s] == fold for s in subject])
            patients, pooled = permuted_scores(x, label, subject, held, selected[fold], SCALERS["sparse"], rng)
            values.append(auroc(np.array([plan["labels"][s] for s in patients]), pooled))
        draws.append(float(np.mean(values)))
    _save_json(path, {"arm": PRIMARY, "draws": draws, "mean": float(np.mean(draws)), "sd": float(np.std(draws))})


# ---------------------------------------------------------------------- decision


def read_scores(path: Path) -> ArmScores:
    """The held-out patient scores of an arm, aligned by patient."""
    frame = pl.read_csv(path)
    return ArmScores.from_rows(
        frame["subject"].cast(pl.Utf8).to_numpy().astype(str),
        frame["fold"].to_numpy(),
        frame["label"].to_numpy(),
        frame["score"].to_numpy(),
    )


def training_gate(root: Path) -> bool | None:
    """Every trained dictionary explains more validation energy than its untrained start; ``None`` if one is missing."""
    trained, untrained = [], []
    for fold in range(N_FOLDS):
        paths = [dictionary_dir(root, arm, fold) / "sae_training.csv" for arm in ("trained", "untrained")]
        if not all(p.is_file() for p in paths):
            return None
        trained.append(float(pl.read_csv(paths[0])["val_residual_frac"].min()))
        untrained.append(float(pl.read_csv(paths[1])["val_residual_frac"][0]))
    return training_passed(trained, untrained)


def summarize(root: Path) -> dict[str, Any]:
    """Apply the rule of section 7 to a root; write ``readout_summary.csv`` and ``readout_decision.json``."""
    plan = load_plan(root)
    patients = pl.read_csv(root / "patients.csv").with_columns(pl.col("subject").cast(pl.Utf8))
    rates = dict(zip(patients["subject"].to_list(), patients["native_rate"].to_list()))
    segments = dict(zip(patients["subject"].to_list(), patients["segments"].to_list()))
    arms = {p.stem: read_scores(p) for p in sorted((root / "scores").glob("*.csv"))}
    intervals = {
        name: paired_interval(arms[PRIMARY], arms[name], seed=plan["seed"])
        for name in DECIDING
        if PRIMARY in arms and name in arms
    }
    described = {
        name: paired_interval(arms[PRIMARY], arms[name], seed=plan["seed"])
        for name in ("S4-atom", "S4-R1", "S4-sparse")
        if PRIMARY in arms and name in arms
    }
    guard_ok, guard = (None, {})
    if PRIMARY in arms and "Spectral" in arms:
        guard_ok, guard = rate_guard(arms[PRIMARY], arms["Spectral"], rates)
    control_path = root / "positive_control.json"
    control = json.loads(control_path.read_text())["passed"] if control_path.is_file() else None
    training = training_gate(root)
    gates = False if control is False or training is False else (None if control is None or training is None else True)
    passed = conditions(intervals)
    outcome = readout_outcome(passed, plan["pass"], gates, guard_ok)
    rows = []
    for name, arm in arms.items():
        group = np.array([rates[s] for s in arm.subject])
        rows.append(
            {
                "arm": name,
                "read": ARMS[name].read if name in ARMS else "nine",
                "mean_fold_auroc": mean_fold_auroc(arm),
                "pooled_auroc": pooled_auroc(arm),
                "patients": len(arm.subject),
                **{f"auroc_rate_{r:g}": auroc(arm.label[group == r], arm.score[group == r]) for r in np.unique(group)},
                **{f"bag_link_{k}": v for k, v in bag_size_link(arm, segments).items()},
            }
        )
    if rows:
        pl.DataFrame(rows, infer_schema_length=None).write_csv(root / "readout_summary.csv")
    baselines = root / "baselines.json"
    null = root / "permutation_null.json"
    decision = {
        "rule": "documentation/frozen_feature_readout_protocol.md, section 7",
        "pass": plan["pass"],
        "outcome": outcome,
        "conditions": passed,
        "intervals": {k: dict(zip(("difference", "lower", "upper"), v)) for k, v in intervals.items()},
        "descriptive_intervals": {k: dict(zip(("difference", "lower", "upper"), v)) for k, v in described.items()},
        "gates": {"positive_control": control, "training": training},
        "rate_guard": {"ok": guard_ok, "groups": guard},
        "baselines": json.loads(baselines.read_text()) if baselines.is_file() else None,
        "permutation_null": json.loads(null.read_text()) if null.is_file() else None,
        "commit": _git("rev-parse", "HEAD").strip(),
    }
    _save_json(root / "readout_decision.json", decision)
    logger.info(f"Frozen readout, pass {plan['pass']}: {outcome}; {passed}")
    return decision


def review(root: Path) -> None:
    """Write the top templates of each fold for the explanation review of section 7.

    Writes ``review/templates.csv``, ``review/f{fold}.npz`` (waveforms and projections of the top templates) and
    ``review/recurrence.json`` (how many top templates recur in the other folds).
    """
    plan = load_plan(root)
    (root / "review").mkdir(parents=True, exist_ok=True)
    rows, tops = [], {}
    for fold in range(N_FOLDS):
        folder = dictionary_dir(root, "trained", fold)
        time = np.load(folder / "sae_time_atoms.npy")
        channel = np.load(folder / "sae_channel_atoms.npy")
        channels = (folder / "sae_channels.txt").read_text().split()
        coef = np.load(root / "coef" / f"{PRIMARY}_f{fold}.npy")
        data = np.load(feature_path(root, PRIMARY, fold))
        subject, x, keys = data["subject"].astype(str), data["features"], data["key"]
        training = np.array([plan["folds"][s] != fold for s in subject])
        top = np.argsort(-np.abs(coef))[:TOP_TEMPLATES]
        atoms, projections = top // N_PROJECTIONS, top % N_PROJECTIONS
        frequency = dominant_frequency(time[atoms])
        for rank, (template, atom, projection) in enumerate(zip(top, atoms, projections)):
            pattern = channel[atom, :, projection]
            strongest = np.argsort(-x[:, template] * training)[:TOP_SEGMENTS]
            rows.append(
                {
                    "fold": fold,
                    "rank": rank,
                    "template": int(template),
                    "atom": int(atom),
                    "projection": int(projection),
                    "weight": float(coef[template]),
                    "dominant_hz": float(frequency[rank]),
                    "top_channels": " ".join(channels[i] for i in np.argsort(-np.abs(pattern))[:4]),
                    "strongest_rows": " ".join(f"{subject[i]}:{keys[i]}" for i in strongest),
                }
            )
        tops[fold] = (time[atoms], channel[atoms, :, projections])
        np.savez(root / "review" / f"f{fold}.npz", waveforms=tops[fold][0], projections=tops[fold][1], templates=top)
    pl.DataFrame(rows).write_csv(root / "review" / "templates.csv")
    _save_json(root / "review" / "recurrence.json", recurrence(tops))


def recurrence(tops: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """For each fold, how many top templates have a match in every other fold, by waveform and projection."""
    counts = {}
    for fold, (waveforms, projections) in tops.items():
        recurring = 0
        for waveform, pattern in zip(waveforms, projections):
            recurring += all(
                any(
                    abs(best_lag(waveform, other_wave)[0]) >= RECUR_WAVEFORM
                    and abs(float(pattern @ other_pattern)) >= RECUR_PROJECTION
                    for other_wave, other_pattern in zip(*tops[other])
                )
                for other in tops
                if other != fold
            )
        counts[str(fold)] = recurring
    return {"rule": {"waveform": RECUR_WAVEFORM, "projection": RECUR_PROJECTION}, "recurring_templates": counts}


# ---------------------------------------------------------------------- command line


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def data_command(name: str, description: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        command.add_argument("--num-workers", type=int, default=4)
        return command

    folds = data_command("folds", "fix the folds and segment count, and record the baselines; reads no signal")
    folds.add_argument("--pass-index", type=int, choices=sorted(PASS_SEEDS), default=1)
    control = data_command("control", "positive control of the feature code on planted events")
    control.add_argument("--control-root", type=Path, default=Path(CONTROL_ROOT))
    fit = data_command("fit", "fit the dictionary of one outer fold")
    fit.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    fit.add_argument("--arm", choices=("trained", "untrained"), required=True)
    extract = data_command("extract", "write the features of a dictionary of one fold")
    extract.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    extract.add_argument("--dictionary", choices=("trained", "untrained"), required=True)
    extract.add_argument("--read", choices=("nine", "all"), default="nine")
    fixed = data_command("extract-fixed", "write the features of a fixed arm")
    fixed.add_argument("--arm", choices=sorted(FIXED_CONFIGS), required=True)
    for name, description in (
        ("heads", "fit the heads, the sparse head and the permutation null"),
        ("summarize", "apply the rule of section 7"),
        ("review", "write the top templates for the explanation review"),
    ):
        commands.add_parser(name, help=description).add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "folds":
        write_folds(args)
    elif args.command == "control":
        positive_control(args)
    elif args.command == "fit":
        fit_dictionary(args)
    elif args.command == "extract":
        extract_dictionary_arms(args)
    elif args.command == "extract-fixed":
        extract_fixed_arm(args)
    elif args.command == "heads":
        run_heads(args)
    elif args.command == "summarize":
        summarize(args.root.resolve())
    else:
        review(args.root.resolve())


if __name__ == "__main__":
    main()
