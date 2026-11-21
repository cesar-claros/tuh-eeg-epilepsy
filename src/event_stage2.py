"""Events corpus protocol, stage 2a: dictionaries trained on crops around the marks, the supervised reference, the
rule and the scan.

Protocol: ``documentation/event_corpus_dictionary_protocol.md``, sections 4 to 8 and 22. On the root of the pilot
(``src/event_pilot.py``): its signal cache, its unit table ``pilot/units.csv`` and its ranks ``pilot/rank.npy``, which
hold the random waveforms and the simple measures. Subcommands, in order:

- ``fit --fold F --band B`` (GPU): the crop dictionaries of one fold and band on the training patients of the fold:
  ``Crops`` (the five event classes), ``Crops-balanced`` (the three deciding classes, the same weight per class and
  patient) and one set per deciding class at 64 and at 16 waveforms. The model, its settings and its training loop
  are those of the adopted dictionary of the epilepsy corpus; a crop is cut from the scaled record, so the model
  does not scale it again.
- ``reference --fold F`` (GPU): a small convolutional classifier on the 2 s around a channel-second of the 1 to 70 Hz
  rows, trained with the event labels of the training folds; its output for each class, as a rank inside the record.
- ``features --fold F`` (GPU): the rank of every unit for every waveform of the fold's dictionaries.
- ``read``: the choice of the waveform on the training folds, the scores, the conditions S, L and B and the outcome
  (``stage2.json``, ``stage2_summary.csv``).
- ``scan`` (GPU): the scan of records of 5 minutes for the chosen features (``scan.json``); reported, no rule.

Stage 2b, the dictionaries on all records, is not built: the user decided on 2026-11-20 to run stage 2a first.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import rootutils
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.event_dictionary_core import CLASSES, CROP_S, N_FOLDS, crop_plan, crop_shifts, pieces  # noqa: E402
from src.event_pilot import ADOPTED, EVENT_CLASSES, cache_path, held_out_means  # noqa: E402
from src.event_response_core import (  # noqa: E402
    BANDS,
    DECIDING,
    LEVEL,
    SFREQ,
    alarm_starts,
    alarms_per_record,
    auroc_columns,
    auroc_of,
    band_rows,
    class_outcome,
    found_marks,
    interval,
    measures,
    pair_counts,
    patient_draws,
    rank_in_record,
    responses,
    second_scores,
    simple_names,
    summarize,
    threshold_for_rate,
)
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.shapeconv_sae import ShapeConvSAE  # noqa: E402
from src.models.components.waveform_sae import WaveformTrainSpec  # noqa: E402

PROTOCOL = "documentation/event_corpus_dictionary_protocol.md"
SEED = 20261120
CROP_N = int(CROP_S * SFREQ)
# A training set: the classes of its crops, whether each class of a patient is a unit of its own, and its sizes.
SETS: dict[str, dict[str, Any]] = {
    "Crops": {"classes": EVENT_CLASSES, "by_class": False, "atoms": (64,)},
    "Crops-balanced": {"classes": DECIDING, "by_class": True, "atoms": (64,)},
    **{f"Crops-{name}": {"classes": (name,), "by_class": False, "atoms": (64, 16)} for name in DECIDING},
}
MIN_STEPS, MIN_EPOCHS = 390, 30  # the steps of a fit of the adopted dictionary; the adopted number of passes
MIN_ATOMS_IN_USE = 8  # gate 6: a crop dictionary with fewer waveforms in use is no candidate
REFERENCE_STEPS, REFERENCE_BATCH, REFERENCE_WINDOW = 2000, 256, 2 * SFREQ
SCAN_RATES = (0.2, 0.5, 1.0, 2.0, 5.0)  # alarms per 5 minutes on the background-only records of the training patients
FAMILIES = ("trained", "untrained", "simple", "reference")


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def load_units(root: Path) -> pl.DataFrame:
    """The unit table of the pilot, in the order of its rank files."""
    return pl.read_csv(root / "pilot" / "units.csv", schema_overrides={"patient": pl.Utf8})


def record_signal(root: Path, patient: str, file: str, start: float, stop: float) -> np.ndarray:
    """One record of a cached recording."""
    signal = np.load(cache_path(root, patient, file), mmap_mode="r")
    return np.asarray(signal[:, int(start * SFREQ) : int(stop * SFREQ)])


# ---------------------------------------------------------------------- crops and fits


class PreScaledSAE(SeparableSAE):
    """The separable model on rows that are scaled already (per record and channel, protocol section 3): its row
    step keeps the validity mask and the trim to whole windows, and it does not scale the rows of a crop again."""

    def _rows(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: N803 - the name of the base class
        raw = X.to(self.device)
        if raw.ndim != 3:
            raise ValueError(f"expected windows of shape (B, C, T), got {tuple(raw.shape)}")
        n_windows, n_channels, _ = raw.shape
        flat = raw.flatten(0, 1)
        valid = ShapeConvSAE._valid_rows(flat)
        rows = torch.where(valid.unsqueeze(-1), flat, torch.zeros_like(flat))
        n_times = rows.shape[-1] // self.spec.pool_len * self.spec.pool_len
        if n_times == 0:
            raise ValueError(f"windows of {rows.shape[-1]} samples are shorter than pool_len={self.spec.pool_len}")
        return rows[:, :n_times].view(n_windows, n_channels, n_times), valid.view(n_windows, n_channels)


class CropSet(Dataset):
    """Crops in memory with the ``plan`` (subject and a label of 0) that the patient loop of the model needs."""

    def __init__(self, x: torch.Tensor, subjects: list[str], channels: list[str]) -> None:
        self.x, self.target_channels = x, channels
        self.plan = pd.DataFrame({"subject": subjects, "epilepsy": 0})

    def __len__(self) -> int:
        """Number of crops, with their shifted copies."""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One crop ``(channels, samples)`` and the label 0."""
        return self.x[index].float(), torch.tensor(0.0)


def build_crops(root: Path, table: pl.DataFrame, band: str) -> tuple[torch.Tensor, pl.DataFrame]:
    """Every crop of every event class in one band, with its shifted copies, cut from the scaled record: the rows
    ``(crops, channels, samples)`` and one line per crop (patient, fold, class)."""
    events = table.filter(pl.col("label") != "bckg")
    parts, lines = [], []
    for (patient, file, _), items in events.group_by("patient", "file", "piece", maintain_order=True):
        start, stop = float(items["piece_start"][0]), float(items["piece_stop"][0])
        rows = band_rows(record_signal(root, patient, file, start, stop), band)
        for name in items["label"].unique(maintain_order=True).to_list():
            seconds = items.filter(pl.col("label") == name)["start"].floor().cast(pl.Int64).to_list()
            for base, second in crop_plan(seconds, stop - start):
                for moved in crop_shifts(base, second, stop - start):
                    first = int(round(moved * SFREQ))
                    parts.append(rows[:, first : first + CROP_N].to(torch.float16))
                    lines.append({"patient": patient, "fold": int(items["fold"][0]), "label": name, "start": moved})
    return torch.stack(parts), pl.DataFrame(lines)


def adopted_settings(sweep_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The model and training settings of the adopted dictionary, from its checkpoint."""
    state = torch.load(
        sweep_root / "dictionaries" / f"{ADOPTED}_f0" / "sae_state.pt", map_location="cpu", weights_only=False
    )
    return dict(state["spec"]), dict(state["fit"]["train_spec"])


def epochs_for(n_subjects: int, patients_per_batch: int) -> int:
    """The passes of a crop fit: at least the adopted 30, and enough for the 390 steps of an adopted fit. A crop set
    has few patients, so 30 passes would be few steps."""
    return max(MIN_EPOCHS, math.ceil(MIN_STEPS / math.ceil(n_subjects / patients_per_batch)))


def fit(
    root: Path,
    fold: int,
    band: str,
    settings: tuple[dict[str, Any], dict[str, Any]],
    device: str = "cuda",
    steps: int | None = None,
) -> None:
    """The crop dictionaries of one fold and band. A finished dictionary is kept."""
    table = load_units(root)
    channels = json.loads((root / "cache" / "channels.json").read_text())
    spec_fields, train_fields = settings
    x, lines = build_crops(root, table, band)
    train = (lines["fold"] != fold).to_numpy()
    label, patient = lines["label"].to_numpy(), lines["patient"].to_numpy()
    for name, entry in SETS.items():
        for n_atoms in entry["atoms"]:
            folder = root / "stage2" / "fits" / f"f{fold}" / band / f"{name}-K{n_atoms}"
            if (folder / "fit.json").is_file():
                logger.info(f"{folder.name} fold {fold} band {band}: finished")
                continue
            chosen = np.flatnonzero(train & np.isin(label, entry["classes"]))
            if not len(chosen):  # no mark of these classes in the training folds: no dictionary, and no candidate
                _save_json(folder / "fit.json", {"set": name, "atoms": n_atoms, "band": band, "fold": fold, "crops": 0})
                continue
            subjects = [f"{patient[i]}|{label[i]}" if entry["by_class"] else str(patient[i]) for i in chosen]
            dataset = CropSet(x[chosen], subjects, channels)
            n_subjects = len(set(subjects))
            train_spec = WaveformTrainSpec(**train_fields)
            epochs = epochs_for(n_subjects, train_spec.patients_per_batch)
            if steps is not None:  # a short fit for the tests
                epochs = max(1, math.ceil(steps / math.ceil(n_subjects / train_spec.patients_per_batch)))
            model = PreScaledSAE(
                spec=SeparableSpec(**(spec_fields | {"n_atoms": n_atoms})),
                train_spec=replace(train_spec, epochs=epochs),
                random_state=SEED + fold,
                device=device,
            )
            loader = DataLoader(dataset, batch_size=train_spec.segment_batch, num_workers=0)
            model.fit_unsupervised(loader, None, provenance={"train_subjects": sorted(set(subjects))})
            folder.mkdir(parents=True, exist_ok=True)
            model.save_artifacts(folder)
            last = model.history[-1]
            _save_json(
                folder / "fit.json",
                {
                    "set": name,
                    "atoms": n_atoms,
                    "band": band,
                    "fold": fold,
                    "crops": int(len(chosen)),
                    "subjects": n_subjects,
                    "patients": int(len(set(patient[chosen].tolist()))),
                    "epochs": epochs,
                    "steps": int(sum(row["n_steps"] for row in model.history)),
                    "objective_first": float(model.history[0]["objective"]),
                    "objective_last": float(last["objective"]),
                    "atoms_in_use": int(n_atoms - last["n_atoms_never_won"]),
                },
            )
            logger.info(f"{folder.name} fold {fold} band {band}: {len(chosen)} crops, {epochs} passes")


def dictionary_columns(root: Path, fold: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """The waveforms of the fold's dictionaries per band, stacked, and one description per column."""
    atoms: dict[str, np.ndarray] = {}
    columns: list[dict[str, Any]] = []
    for band in BANDS:
        parts = []
        for name, entry in SETS.items():
            for n_atoms in entry["atoms"]:
                folder = root / "stage2" / "fits" / f"f{fold}" / band / f"{name}-K{n_atoms}"
                record = json.loads((folder / "fit.json").read_text())
                if not record["crops"]:
                    continue
                waves = np.load(folder / "sae_time_atoms.npy").astype(np.float32)
                parts.append(waves / np.linalg.norm(waves, axis=1, keepdims=True).clip(1e-12))
                columns += [
                    {
                        "band": band,
                        "family": "trained",
                        "set": name,
                        "atoms": n_atoms,
                        "index": k,
                        "candidate": record["atoms_in_use"] >= MIN_ATOMS_IN_USE,
                    }
                    for k in range(len(waves))
                ]
        atoms[band] = np.concatenate(parts)
    return atoms, columns


def features(root: Path, fold: int, device: str = "cuda") -> None:
    """The rank of every unit for every waveform of the fold's dictionaries (``stage2/rank_f{fold}.npy``)."""
    table = load_units(root)
    waves, columns = dictionary_columns(root, fold)
    atoms = {band: torch.from_numpy(values).to(device) for band, values in waves.items()}
    parts = []
    with torch.no_grad():
        for (patient, file, _), items in table.group_by("patient", "file", "piece", maintain_order=True):
            record = record_signal(root, patient, file, float(items["piece_start"][0]), float(items["piece_stop"][0]))
            start = np.round(items["start"].to_numpy() * SFREQ).astype(int)
            channel = torch.as_tensor(np.array(items["channel"].to_numpy()), device=device)
            distinct = sorted(set(start.tolist()))
            index = torch.tensor([distinct.index(s) for s in start.tolist()], device=device)
            bands = []
            for band in BANDS:
                rows = band_rows(record, band).to(device)
                n_seconds = rows.shape[-1] // SFREQ
                grids, at = [], []
                for first in range(0, len(atoms[band]), 128):
                    response = responses(rows, atoms[band][first : first + 128])
                    used = response[..., : n_seconds * SFREQ]
                    grids.append(used.reshape(*used.shape[:2], n_seconds, SFREQ).amax(-1))
                    at.append(torch.stack([response[..., s : s + SFREQ].amax(-1) for s in distinct]))
                grid, values = torch.cat(grids, dim=1), torch.cat(at, dim=2)
                ranks = torch.stack([rank_in_record(values[i], grid, s) for i, s in enumerate(distinct)])
                bands.append(ranks[index, channel].cpu().numpy())
            parts.append(np.concatenate(bands, axis=1).astype(np.float32))
    folder = root / "stage2"
    np.save(folder / f"rank_f{fold}.npy", np.concatenate(parts))
    _save_json(folder / f"columns_f{fold}.json", columns)
    logger.info(f"features fold {fold}: {table.height} units, {len(columns)} columns")


# ---------------------------------------------------------------------- the supervised reference


class ReferenceNet(torch.nn.Module):
    """Three convolution layers on a window of one channel, then one value per class (protocol section 5)."""

    def __init__(self, n_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 9, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, 9, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, 9, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1),
        )
        self.head = torch.nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The log probability of every class for windows ``(n, samples)``."""
        return torch.log_softmax(self.head(self.body(x.unsqueeze(1)).squeeze(-1)), dim=-1)


def reference_windows(rows: torch.Tensor, starts: list[int]) -> torch.Tensor:
    """The windows of 2 s around seconds that start at the given samples, for every channel: ``(starts, channels,
    512)``, with zeros beyond the record. A window starts half a second before its second."""
    padded = torch.nn.functional.pad(rows, (SFREQ // 2, REFERENCE_WINDOW))
    return torch.stack([padded[:, s : s + REFERENCE_WINDOW] for s in starts])


def reference(root: Path, fold: int, device: str = "cuda", steps: int = REFERENCE_STEPS) -> None:
    """Train the reference on the units of the training folds and write its output for every unit as a rank inside
    the record, one column per event class (``stage2/reference_f{fold}.npy``)."""
    table = load_units(root)
    names = list(CLASSES.values())
    target = torch.tensor([names.index(name) for name in table["label"].to_list()])
    groups = list(table.group_by("patient", "file", "piece", maintain_order=True))
    windows = []
    for (patient, file, _), items in groups:
        rows = band_rows(
            record_signal(root, patient, file, float(items["piece_start"][0]), float(items["piece_stop"][0])), "70"
        )
        start = np.round(items["start"].to_numpy() * SFREQ).astype(int).tolist()
        at = reference_windows(rows, start)
        windows.append(at[torch.arange(len(start)), torch.as_tensor(np.array(items["channel"].to_numpy()))])
    windows_all = torch.cat(windows).to(torch.float16)
    train = torch.as_tensor((table["fold"] != fold).to_numpy())
    generator = torch.Generator().manual_seed(SEED + fold)
    torch.manual_seed(SEED + fold)
    model = ReferenceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = torch.bincount(target[train], minlength=len(names)).float()
    weight = (1.0 / counts.clamp_min(1))[target] * train  # each class with the same weight; training folds only
    for _ in range(steps):
        index = torch.multinomial(weight, REFERENCE_BATCH, replacement=True, generator=generator)
        sign = torch.where(torch.rand(len(index), 1, generator=generator) < 0.5, -1.0, 1.0)
        batch = (windows_all[index].float() * sign).to(device)  # the sign of a discharge depends on the bipolar pair
        loss = torch.nn.functional.nll_loss(model(batch), target[index].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    parts = []
    with torch.no_grad():
        for (patient, file, _), items in groups:
            rows = band_rows(
                record_signal(root, patient, file, float(items["piece_start"][0]), float(items["piece_stop"][0])), "70"
            ).to(device)
            n_seconds = rows.shape[-1] // SFREQ
            grid = model(reference_windows(rows, [g * SFREQ for g in range(n_seconds)]).flatten(0, 1))
            grid = grid.view(n_seconds, rows.shape[0], -1).permute(1, 2, 0)  # (channels, classes, seconds)
            start = np.round(items["start"].to_numpy() * SFREQ).astype(int).tolist()
            channel = np.array(items["channel"].to_numpy()).tolist()
            distinct = sorted(set(start))
            values = model(reference_windows(rows, distinct).flatten(0, 1)).view(len(distinct), rows.shape[0], -1)
            ranks = {s: rank_in_record(values[i], grid, s) for i, s in enumerate(distinct)}  # (channels, classes)
            parts.append(np.stack([ranks[s][c].cpu().numpy() for s, c in zip(start, channel, strict=True)]))
    folder = root / "stage2"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"reference_f{fold}.npy", np.concatenate(parts)[:, : len(EVENT_CLASSES)].astype(np.float32))
    torch.save(model.state_dict(), folder / f"reference_f{fold}.pt")
    logger.info(f"reference fold {fold}: {int(train.sum())} training units, last loss {loss.item():.3f}")


# ---------------------------------------------------------------------- read


def fold_choice(
    matrix: np.ndarray,
    candidates: list[int],
    label: np.ndarray,
    patient: np.ndarray,
    fold: np.ndarray,
    target: str,
    f: int,
) -> dict[str, Any] | None:
    """In one fold: the candidate column with the highest AUROC for the target on the training folds, and its pair
    counts on the held-out fold. ``None`` without a unit of the target on a side or without a candidate."""
    positive, train, held = label == target, fold != f, fold == f
    if not candidates or not (positive & train).any() or not (positive & held).any():
        return None
    scores = auroc_columns(matrix[train][:, candidates], positive[train])
    if not np.isfinite(scores).any():
        return None
    best = int(candidates[int(np.nanargmax(scores))])
    return {
        "column": best,
        "train_auroc": float(np.nanmax(scores)),
        "counts": pair_counts(matrix[held, best], positive[held], patient[held]),
    }


def trained_candidates(columns: list[dict[str, Any]], target: str) -> list[int]:
    """The columns that a class can choose: ``Crops``, ``Crops-balanced`` and its own class crops, in every band,
    from dictionaries with enough waveforms in use."""
    allowed = {"Crops", "Crops-balanced", f"Crops-{target}"}
    return [i for i, c in enumerate(columns) if c["set"] in allowed and c["candidate"]]


def read(root: Path, n_boot: int = 10_000) -> dict[str, Any]:
    """Stage 2a: scores, the conditions S, L and B, and the outcome (protocol section 7), without the dictionaries on
    all records."""
    table = load_units(root)
    label, patient, fold = table["label"].to_numpy(), table["patient"].to_numpy(), table["fold"].to_numpy()
    pilot = np.load(root / "pilot" / "rank.npy")
    pilot_columns = json.loads((root / "pilot" / "columns.json").read_text())
    pilot_status = json.loads((root / "pilot.json").read_text())["status"]
    draws = [
        {"patients": np.unique(patient[fold == f]), "counts": patient_draws(patient[fold == f], n_boot, seed=f)}
        for f in range(N_FOLDS)
    ]
    chosen: dict[str, dict[str, list[Any]]] = {t: {family: [] for family in FAMILIES} for t in EVENT_CLASSES}
    described: dict[str, dict[str, list[Any]]] = {t: {family: [] for family in FAMILIES} for t in EVENT_CLASSES}
    means: dict[str, dict[str, list[Any]]] = {t: {family: [] for family in FAMILIES} for t in EVENT_CLASSES}
    fits = []
    for f in range(N_FOLDS):
        matrix = np.load(root / "stage2" / f"rank_f{f}.npy")
        columns = json.loads((root / "stage2" / f"columns_f{f}.json").read_text())
        reference_ranks = np.load(root / "stage2" / f"reference_f{f}.npy")
        found = [json.loads(p.read_text()) for p in sorted((root / "stage2" / "fits" / f"f{f}").glob("*/*/fit.json"))]
        fits += [fit for fit in found if fit["crops"]]
        for k, target in enumerate(EVENT_CLASSES):
            arms = {
                "trained": (matrix, trained_candidates(columns, target), columns),
                "untrained": (
                    pilot,
                    [i for i, c in enumerate(pilot_columns) if c["family"] == "untrained"],
                    pilot_columns,
                ),
                "simple": (pilot, [i for i, c in enumerate(pilot_columns) if c["family"] == "simple"], pilot_columns),
                "reference": (reference_ranks, [k], [{"family": "reference", "class": t} for t in EVENT_CLASSES]),
            }
            for family, (values, candidates, names) in arms.items():
                entry = fold_choice(values, candidates, label, patient, fold, target, f)
                chosen[target][family].append(entry)
                described[target][family].append(names[entry["column"]] if entry is not None else None)
                held = fold == f
                means[target][family].append(
                    None
                    if entry is None
                    else {
                        "class": float(np.nanmean(values[held & (label == target), entry["column"]])),
                        "background": float(np.nanmean(values[held & (label == "bckg"), entry["column"]])),
                    }
                )
    classes: dict[str, Any] = {}
    summary = []
    for target in EVENT_CLASSES:
        arms = {family: summarize(chosen[target][family], draws) for family in FAMILIES}
        entry: dict[str, Any] = {}
        for family, arm in arms.items():
            known = arm["score"] is not None
            kept = [m for m in means[target][family] if m is not None]
            entry[family] = {
                "score": arm["score"],
                "interval": interval(arm["boot"]) if known else None,
                "chosen": described[target][family],
                "train_auroc": [e["train_auroc"] if e is not None else None for e in chosen[target][family]],
                "mean_rank_class": float(np.mean([m["class"] for m in kept])) if kept else None,
                "mean_rank_background": float(np.mean([m["background"] for m in kept])) if kept else None,
            }
            summary.append(
                {
                    "class": target,
                    "family": family,
                    "score": arm["score"],
                    "low": entry[family]["interval"][0] if known else None,
                    "high": entry[family]["interval"][1] if known else None,
                }
            )
        if arms["trained"]["score"] is not None:
            gains = {}
            for other in ("untrained", "simple", "reference"):
                if arms[other]["score"] is None:
                    continue
                difference = arms["trained"]["boot"] - arms[other]["boot"]
                gains[other] = {
                    "gain": arms["trained"]["score"] - arms[other]["score"],
                    "low": interval(difference, LEVEL)[0],
                    "high": interval(difference, LEVEL)[1],
                }
            entry["gains"] = gains
            if "untrained" in gains and "simple" in gains:
                entry["outcome"] = class_outcome(arms["trained"]["score"], gains["untrained"], gains["simple"])
        classes[target] = entry
    identified = [t for t in DECIDING if classes[t].get("outcome", "").startswith("identified")]
    record = {
        "protocol": f"{PROTOCOL}, sections 7 and 22 (stage 2a: no dictionary on all records)",
        "prediction": "none about 85%, epileptiform_identified about 10% (section 21.6)",
        "pilot_status": pilot_status,
        "outcome": "blocked" if pilot_status != "pass" else "epileptiform_identified" if identified else "none",
        "identified": identified,
        "units": table.height,
        "dictionaries": len(fits),
        "dictionaries_below_the_use_floor": sum(fit["atoms_in_use"] < MIN_ATOMS_IN_USE for fit in fits),
        "classes": classes,
        "fits": fits,
    }
    pl.DataFrame(summary, infer_schema_length=None).write_csv(root / "stage2_summary.csv")
    _save_json(root / "stage2.json", record)
    _save_json(
        root / "stage2" / "chosen.json",
        {t: {family: [e["column"] if e else None for e in chosen[t][family]] for family in FAMILIES} for t in DECIDING},
    )
    logger.info(f"stage 2a: {record['outcome']} {[(t, classes[t].get('outcome')) for t in DECIDING]}")
    return record


# ---------------------------------------------------------------------- scan


def feature_seconds(
    rows: dict[str, torch.Tensor], description: dict[str, Any], waves: np.ndarray | None, net: Any
) -> torch.Tensor:
    """The value ``(channels, seconds)`` of one feature on every whole second of a record."""
    if description["family"] == "reference":
        data = rows["70"]
        n_seconds = data.shape[-1] // SFREQ
        out = net(reference_windows(data, [g * SFREQ for g in range(n_seconds)]).flatten(0, 1))
        return out.view(n_seconds, data.shape[0], -1)[:, :, description["column"]].T
    data = rows[description["band"]]
    n_seconds = data.shape[-1] // SFREQ
    if description["family"] == "simple":
        windows = data[:, : n_seconds * SFREQ].reshape(data.shape[0], n_seconds, SFREQ)
        index = simple_names(description["band"]).index(description["dictionary"])
        return measures(windows, description["band"])[..., index]
    response = responses(data, torch.from_numpy(waves).to(data.device).unsqueeze(0))[:, 0, : n_seconds * SFREQ]
    return response.reshape(data.shape[0], n_seconds, SFREQ).amax(-1)


def scan(root: Path, device: str = "cuda") -> dict[str, Any]:
    """The scan of records of 5 minutes (protocol section 8) for the chosen feature of every deciding class, family
    and fold: the threshold from the background-only records of the training patients, then on the held-out fold the
    marked seconds found and the alarms per 5 minutes. Reported; no rule."""
    from src.event_response_core import random_atoms

    table = load_units(root)
    files = pl.read_csv(root / "files.csv", schema_overrides={"group": pl.Utf8})
    files = files.filter((pl.col("set") == "train") & pl.col("header_ok") & pl.col("has_rec"))
    folds = {p: int(f) for p, f in table.select("patient", "fold").unique().iter_rows()}
    event_patients = set(table.filter(pl.col("label") != "bckg")["patient"].to_list())
    background_only = set(folds) - event_patients
    chosen = json.loads((root / "stage2" / "chosen.json").read_text())
    pilot_columns = json.loads((root / "pilot" / "columns.json").read_text())
    # The features to scan: per fold, class and family, what to compute.
    plans: dict[tuple[int, str, str], dict[str, Any]] = {}
    nets = {}
    for f in range(N_FOLDS):
        waves, columns = dictionary_columns(root, f)
        offsets = np.cumsum([0] + [len(waves[band]) for band in BANDS])
        net = ReferenceNet().to(device)
        net.load_state_dict(torch.load(root / "stage2" / f"reference_f{f}.pt", map_location=device))
        nets[f] = net.eval()
        for k, target in enumerate(DECIDING):
            for family in FAMILIES:
                column = chosen[target][family][f]
                if column is None:
                    continue
                if family == "trained":
                    band = columns[column]["band"]
                    wave = waves[band][column - offsets[list(BANDS).index(band)]]
                    plans[(f, target, family)] = {"family": "trained", "band": band, "waves": wave}
                elif family == "untrained":
                    c = pilot_columns[column]
                    wave = random_atoms(c["band"], int(c["dictionary"][1:]))[c["index"]]
                    plans[(f, target, family)] = {"family": "untrained", "band": c["band"], "waves": wave}
                elif family == "simple":
                    plans[(f, target, family)] = pilot_columns[column] | {"waves": None}
                else:
                    plans[(f, target, family)] = {
                        "family": "reference",
                        "column": EVENT_CLASSES.index(target),
                        "waves": None,
                    }
    needed = sorted({plan.get("band", "70") for plan in plans.values()})  # the reference reads 1 to 70 Hz
    marks: dict[tuple[str, str, int], dict[str, list[int]]] = {}
    for (patient, file, piece), items in table.filter(pl.col("label").is_in(DECIDING)).group_by(
        "patient", "file", "piece"
    ):
        for name in items["label"].unique().to_list():
            seconds = items.filter(pl.col("label") == name)["start"].floor().cast(pl.Int64).unique().to_list()
            marks.setdefault((patient, file, piece), {})[name] = sorted(seconds)
    scores: dict[tuple[int, str, str], dict[str, list[Any]]] = {
        key: {"train_background": [], "held_background": [], "held_class": []} for key in plans
    }
    with torch.no_grad():
        for patient, file, duration in files.select("group", "file", "duration_s").iter_rows():
            if patient not in folds or not cache_path(root, patient, file).is_file():
                continue
            for piece, (start, stop) in enumerate(pieces(duration)):
                marked = marks.get((patient, file, piece), {})
                if patient not in background_only and not marked:
                    continue
                signal = record_signal(root, patient, file, start, stop)
                rows = {band: band_rows(signal, band).to(device) for band in needed}
                for (f, target, family), plan in plans.items():
                    held = folds[patient] == f
                    if patient in background_only:
                        part = "held_background" if held else "train_background"
                    elif held and target in marked:
                        part = "held_class"
                    else:
                        continue
                    values = second_scores(feature_seconds(rows, plan, plan["waves"], nets[f])).cpu().numpy()
                    scores[(f, target, family)][part].append((values, marked.get(target, [])))
    out: dict[str, Any] = {}
    for target in DECIDING:
        out[target] = {}
        for family in FAMILIES:
            rates = {}
            for rate in SCAN_RATES:
                found = total = records = records_hit = back_records = back_hit = 0
                held_rates, unmarked = [], []
                for f in range(N_FOLDS):
                    parts = scores.get((f, target, family))
                    if parts is None or not parts["train_background"]:
                        continue
                    threshold = threshold_for_rate([s for s, _ in parts["train_background"]], rate)
                    held_rates.append(alarms_per_record([s for s, _ in parts["held_background"]], threshold))
                    back_records += len(parts["held_background"])
                    back_hit += sum(len(alarm_starts(s, threshold)) > 0 for s, _ in parts["held_background"])
                    for values, seconds in parts["held_class"]:
                        found += found_marks(values, seconds, threshold)
                        total += len(seconds)
                        records += 1
                        records_hit += len(alarm_starts(values, threshold)) > 0
                        away = np.ones(len(values), dtype=bool)
                        for second in seconds:
                            away[max(second - 1, 0) : second + 2] = False
                        starts = alarm_starts(values, threshold)
                        unmarked.append((int(away[starts].sum()), int(away.sum())))
                seconds_away = sum(n for _, n in unmarked)
                rates[str(rate)] = {
                    "marked_seconds": total,
                    "found": found,
                    "share_found": found / total if total else None,
                    "alarms_per_5_min_on_held_out_background": float(np.nanmean(held_rates)) if held_rates else None,
                    "alarms_per_5_min_in_unmarked_time_of_class_records": (
                        sum(a for a, _ in unmarked) * 300.0 / seconds_away if seconds_away else None
                    ),
                    "class_records_with_an_alarm": records_hit / records if records else None,
                    "background_records_with_an_alarm": back_hit / back_records if back_records else None,
                }
            out[target][family] = rates
    record = {
        "protocol": f"{PROTOCOL}, section 8 (reported, no rule)",
        "rates": list(SCAN_RATES),
        "background_only_patients": len(background_only),
        "classes": out,
    }
    _save_json(root / "scan.json", record)
    logger.info(f"scan: {len(plans)} features on {len(background_only)} background-only patients")
    return record


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("fit", help="the crop dictionaries of one fold and band (GPU)")
    command.add_argument("--fold", type=int, required=True)
    command.add_argument("--band", choices=list(BANDS), required=True)
    command.add_argument("--sweep-root", type=Path, required=True, help="the root with the adopted dictionaries")
    for name in ("reference", "features"):
        commands.add_parser(name).add_argument("--fold", type=int, required=True)
    commands.add_parser("read").add_argument("--n-boot", type=int, default=10_000)
    commands.add_parser("scan")
    for command in commands.choices.values():
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "fit":
        torch.use_deterministic_algorithms(True)
        fit(root, args.fold, args.band, adopted_settings(args.sweep_root.resolve()), args.device)
    elif args.command == "reference":
        reference(root, args.fold, args.device)
    elif args.command == "features":
        features(root, args.fold, args.device)
    elif args.command == "read":
        read(root, args.n_boot)
    else:
        scan(root, args.device)


if __name__ == "__main__":
    main()
