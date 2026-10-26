"""Patient-first waveform training with complete manifest pools and bounded gradient memory."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger as log
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.models.components.waveform_sae import (
    CHECKPOINT_FORMAT,
    MODEL_TAG,
    WaveformSAE,
    WaveformSpec,
    WaveformTrainSpec,
)


class PatientDataset(Dataset):
    """Attach patient and manifest-row indices without copying or shortening the source dataset.

    The source must expose an aligned ``plan`` with subject and binary epilepsy
    columns. Missing IDs, inconsistent patient labels or length mismatches raise
    ValueError. Signal loading remains lazy and uses the source's preprocessing.
    """

    def __init__(self, source: Dataset) -> None:
        self.source = source
        plan = getattr(source, "plan", None)
        if plan is None or len(plan) != len(source) or not {"subject", "epilepsy"}.issubset(plan.columns):
            raise ValueError("patient training requires a lazy dataset with an aligned subject/epilepsy plan")
        if plan["subject"].isna().any() or not plan["epilepsy"].isin([0, 1]).all() or len(plan) == 0:
            raise ValueError("patient plan must be nonempty with valid subject IDs and binary epilepsy labels")
        subjects = plan["subject"].astype(str).to_numpy()
        self.subjects, inverse = np.unique(subjects, return_inverse=True)
        if any(not s.strip() for s in self.subjects):
            raise ValueError("empty patient ID")
        self.patient = torch.from_numpy(inverse).long()
        self.labels = torch.full((len(self.subjects),), -1.0)
        groups: list[list[int]] = [[] for _ in self.subjects]
        for row, (patient, label) in enumerate(zip(inverse, plan["epilepsy"].to_numpy())):
            if self.labels[patient] not in (-1, label):
                raise ValueError(f"patient {self.subjects[patient]} has inconsistent labels")
            self.labels[patient] = float(label)
            groups[patient].append(row)
        self.groups = [torch.tensor(group, dtype=torch.long) for group in groups]

    def __len__(self) -> int:
        """Number of retained source segments, including those not drawn in a given epoch."""
        return len(self.patient)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load a segment and verify its label against the manifest; return X, y, patient, row."""
        x, y = self.source[index]
        patient = self.patient[index]
        if torch.as_tensor(y).numel() != 1 or float(y) != float(self.labels[patient]):
            raise ValueError(f"loaded label disagrees with patient metadata at row {index}")
        return x, self.labels[patient], patient, index


class PatientBatches(Sampler[list[int]]):
    """One shuffled visit per patient, each drawing min(S, J_i) segments without replacement.

    ``groups`` holds every manifest index. Draws are redone each epoch; manifests
    are never capped. Patients are distinct within each batch, including the final
    partial batch. ``seed + epoch`` makes draws reproducible without global RNG state.
    """

    def __init__(self, groups: list[torch.Tensor], patients: int, segments: int, seed: int) -> None:
        if not groups or patients < 1 or segments < 1 or any(len(group) == 0 for group in groups):
            raise ValueError("need nonempty patient groups and positive batch sizes")
        self.groups, self.patients, self.segments, self.seed = groups, patients, segments, seed
        self.epoch = 0

    def __len__(self) -> int:
        """Optimizer steps per patient pass."""
        return math.ceil(len(self.groups) / self.patients)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield complete manifest-row indices for each patient batch."""
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=generator)
        for batch in order.split(self.patients):
            indices = []
            for patient in batch.tolist():
                group = self.groups[patient]
                selected = torch.randperm(len(group), generator=generator)[: self.segments]
                indices.extend(group[selected].tolist())
            yield indices


def patient_loader(base: DataLoader, model: WaveformSAE, training: bool) -> DataLoader:
    """Wrap a complete lazy dataset with patient IDs and optional patient-first batching."""
    dataset = PatientDataset(base.dataset)
    common = {"num_workers": base.num_workers, "pin_memory": base.pin_memory}
    if training:
        spec = model.train_spec
        sampler = PatientBatches(dataset.groups, spec.patients_per_batch, spec.segments_per_patient, model.random_state)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=model.train_spec.segment_batch, shuffle=False, **common)


def segment_values(
    model: WaveformSAE, x: torch.Tensor, statistics: tuple[torch.Tensor, torch.Tensor] | None = None
) -> tuple[torch.Tensor, ...]:
    """Per-segment reconstruction, features, zero baseline and validity; autograd stays connected.

    Each reconstruction divides by its own usable channel-samples. Completely
    invalid segments are marked false and excluded by patient reductions.
    """
    rows, valid = model._full_rows(x)
    atoms = model._dictionary()
    code, selected = model._selection(model._responses(rows, atoms))
    reconstruction = model._decode(code, atoms) * valid.unsqueeze(-1)
    denominator = (2 * valid.sum(-1) * rows.shape[-1]).clamp_min(1)
    loss = (rows - reconstruction).square().sum((1, 2)) / denominator
    zero = rows.square().sum((1, 2)) / denominator
    features = code.sum(-1) / (rows.shape[-1] // model.spec.pool_len)
    if statistics is not None:
        with torch.no_grad():
            usable = valid.any(-1)
            totals, per_atom = model._statistics(rows[usable], reconstruction[usable], code[usable], selected[usable])
            statistics[0].add_(totals)
            statistics[1].add_(per_atom)
    return loss, features, zero, valid.any(-1)


def patient_means(values: torch.Tensor, patient: torch.Tensor, count: int) -> torch.Tensor:
    """Mean vectors/scalars within each patient; refuse an empty patient instead of changing weights."""
    numbers = torch.bincount(patient, minlength=count)
    if bool((numbers == 0).any()):
        raise ValueError("a patient has no usable segments; audit signal validity before training")
    sums = values.new_zeros((count, *values.shape[1:])).index_add(0, patient, values)
    return sums / numbers.reshape((-1,) + (1,) * (values.ndim - 1))


@dataclass
class PatientSummary:
    """Full-pool per-patient means and usable segment counts, in sorted patient order."""

    features: torch.Tensor
    reconstruction: torch.Tensor
    zero: torch.Tensor
    counts: torch.Tensor
    skipped: int
    totals: torch.Tensor
    per_atom: torch.Tensor


@torch.no_grad()
def summarize_patients(
    model: WaveformSAE, loader: DataLoader, description: str = "Patient full-pool evaluation"
) -> PatientSummary:
    """Stream the complete pool with progress and wall timings; retain the same reductions.

    Loader wait measures time awaiting prefetched batches, not total worker CPU
    time. Compute includes transfer, normalization, forward passes and reduction;
    CUDA is synchronized before the final elapsed-time report.
    """
    started = time.perf_counter()
    count = len(loader.dataset.subjects)
    log.info("{}: {} segments, {} patients, {} batches", description, len(loader.dataset), count, len(loader))
    sums = torch.zeros(count, model.spec.n_atoms + 2, dtype=torch.float64, device=model.device)
    numbers = torch.zeros(count, dtype=torch.long, device=model.device)
    statistics = (
        torch.zeros(7, dtype=torch.float64, device=model.device),
        torch.zeros(4, model.spec.n_atoms, dtype=torch.float64, device=model.device),
    )
    skipped = 0
    loader_wait, compute = 0.0, 0.0
    with tqdm(total=len(loader.dataset), desc=description, unit="segment") as progress:
        ready = time.perf_counter()
        for x, _, patient, _ in loader:
            loaded = time.perf_counter()
            loader_wait += loaded - ready
            loss, features, zero, usable = segment_values(model, x, statistics)
            index = patient.to(model.device)[usable]
            values = torch.cat([features, loss[:, None], zero[:, None]], dim=1)[usable]
            sums.index_add_(0, index, values.double())
            numbers.index_add_(0, index, torch.ones_like(index))
            skipped += int((~usable).sum())
            compute += time.perf_counter() - loaded
            progress.update(len(x))
            ready = time.perf_counter()
    if bool((numbers == 0).any()):
        empty = loader.dataset.subjects[(numbers == 0).cpu().numpy()].tolist()
        raise ValueError(f"patients have no usable segments: {empty}; audit signal validity")
    means = (sums / numbers[:, None]).float()
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    log.info(
        "{} finished: {:.1f}s total, {:.1f}s loader wait, {:.1f}s compute, {:.2f} segments/s, {} skipped",
        description,
        elapsed,
        loader_wait,
        compute,
        len(loader.dataset) / max(elapsed, 1e-9),
        skipped,
    )
    return PatientSummary(means[:, :-2], means[:, -2], means[:, -1], numbers, skipped, *statistics)


def _store_atom_statistics(model: WaveformSAE, summary: PatientSummary) -> None:
    model.atom_stats = {
        "atom": list(range(model.spec.n_atoms)),
        **{key: value.tolist() for key, value in model._atom_table(summary.totals, summary.per_atom).items()},
    }


def _check_splits(train: DataLoader, val: DataLoader | None, provenance: dict[str, Any]) -> None:
    subjects = set(train.dataset.subjects)
    if subjects != set(map(str, provenance.get("train_subjects", []))):
        raise ValueError("training patient metadata disagrees with dictionary provenance")
    if val is not None and subjects.intersection(val.dataset.subjects):
        raise ValueError("training and validation patients overlap")


def _batch_weights(model: WaveformSAE, x: torch.Tensor, patient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Validity depends only on the signal, so this inexpensive preprocessing pass
    # determines exact denominators before any chunk's backward pass.
    _, inverse = torch.unique(patient, sorted=True, return_inverse=True)
    inverse = inverse.to(model.device)
    with torch.no_grad():
        usable = torch.cat([model._full_rows(chunk)[1].any(-1) for chunk in x.split(model.train_spec.segment_batch)])
    counts = torch.bincount(inverse[usable], minlength=int(inverse.max()) + 1)
    if bool((counts == 0).any()):
        raise ValueError("sampled patient has no usable segments; audit signal validity or increase the bag draw")
    weights = usable / counts[inverse] / len(counts)
    return inverse, weights


def reconstruction_backward(
    model: WaveformSAE,
    x: torch.Tensor,
    patient: torch.Tensor,
    statistics: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[float, int]:
    """Accumulate the mean-of-patient-means gradient in segment chunks; perform no optimizer step.

    With ``statistics``, also add the summed statistics of the usable segments
    (``WaveformSAE._statistics``, no gradient), including the won windows per atom.
    """
    _, weights = _batch_weights(model, x, patient)
    total = 0.0
    for start in range(0, len(x), model.train_spec.segment_batch):
        stop = start + model.train_spec.segment_batch
        losses, _, _, _ = segment_values(model, x[start:stop], statistics)
        loss = (losses * weights[start:stop]).sum()
        loss.backward()
        total += float(loss.detach())
    return total, int((weights == 0).sum())


def fit_patient_reconstruction(
    model: WaveformSAE, base: DataLoader, validation: DataLoader | None, provenance: dict[str, Any]
) -> None:
    """Pretrain on patient draws, selecting minimum full-pool patient validation reconstruction.

    With no validation loader retain the final state. Epochs are patient passes,
    not exhaustive segment passes. Initialization uses a separate patient draw.
    """
    train = patient_loader(base, model, True)
    val = patient_loader(validation, model, False) if validation is not None else None
    _check_splits(train, val, provenance)
    spec = model.train_spec
    init_sampler = PatientBatches(train.dataset.groups, spec.patients_per_batch, 1, model.random_state + 100000)
    init_loader = DataLoader(base.dataset, batch_sampler=init_sampler, num_workers=base.num_workers)
    model._init_atoms(init_loader, torch.Generator().manual_seed(model.random_state))
    optimizer = torch.optim.Adam([model.atoms], lr=spec.lr)
    best, best_atoms, best_epoch = math.inf, None, 0
    best_summary = None
    model.history = []
    model.won_windows = {}
    saved_draws: list[list[list[int]]] = []
    for epoch in range(spec.epochs):
        epoch_started = time.perf_counter()
        train.batch_sampler.epoch = epoch
        total, patients, seen, skipped, steps = 0.0, 0, 0, 0, 0
        epoch_draws = []
        statistics = (
            torch.zeros(7, dtype=torch.float64, device=model.device),
            torch.zeros(4, model.spec.n_atoms, dtype=torch.float64, device=model.device),
        )
        for x, _, patient, indices in tqdm(train, desc=f"Patient reconstruction {epoch + 1}/{spec.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss, invalid = reconstruction_backward(model, x, patient, statistics)
            optimizer.step()
            with torch.no_grad():
                model.atoms.copy_(model._constrain(model.atoms))
            n = int(patient.unique().numel())
            total += loss * n
            patients += n
            seen += len(x)
            skipped += invalid
            steps += 1
            epoch_draws.append(indices.tolist())
        moved = model._recenter(optimizer) if spec.recenter_atoms and epoch + 1 < spec.epochs else 0
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        summary = (
            summarize_patients(model, val, f"Reconstruction validation {epoch + 1}/{spec.epochs}")
            if val is not None
            else None
        )
        val_loss = float(summary.reconstruction.mean()) if summary is not None else math.nan
        if val_loss < best:
            best, best_atoms, best_epoch = val_loss, model.atoms.detach().clone(), epoch + 1
            best_summary = summary
        record = {
            "epoch": float(epoch + 1),
            "objective": total / patients,
            "val_objective": val_loss,
            **model._reconstruction_summary(summary.totals if summary is not None else None, prefix="val_"),
            **model._reconstruction_summary(summary.totals if summary is not None else None, prefix="val_full_"),
            "val_patient_residual_frac": (
                float((summary.reconstruction / summary.zero).mean()) if summary is not None else math.nan
            ),
            "patients_seen": float(patients),
            "segments_seen": float(seen),
            "segments_skipped": float(skipped),
            "val_segments_skipped": float(summary.skipped) if summary is not None else math.nan,
            "n_steps": float(steps),
            "n_recentered": float(moved),
            **model._won_window_summary(epoch + 1, *statistics),
        }
        model.history.append(record)
        saved_draws.append(epoch_draws)
        log.info(f"Waveform SAE epoch {epoch + 1}: patient reconstruction {total / patients:.6g}, val {val_loss:.6g}")
        log.info(
            "Reconstruction epoch {} timings: {:.1f}s training, {:.1f}s validation/selection, {:.1f}s total",
            epoch + 1,
            training_seconds,
            time.perf_counter() - validation_started,
            time.perf_counter() - epoch_started,
        )
    if best_atoms is not None:
        with torch.no_grad():
            model.atoms.copy_(best_atoms)
    if best_summary is None:
        best_summary = summarize_patients(model, val if val is not None else patient_loader(base, model, False))
    _store_atom_statistics(model, best_summary)
    model.provenance = dict(provenance)
    model.fit_meta = {
        "train_spec": asdict(spec),
        "random_state": model.random_state,
        "sampling_unit": "patient",
        "objective_reduction": "mean_patients_mean_segments_mean_usable_samples",
        "scoring_scope": "full_encoded_segment",
        "selection_scope": "pooling_window",
        "encoder_scope": "full_encoded_segment",
        "decoder_scope": "pooling_window",
        "n_channels": int(model.atoms.shape[1]),
        "channels": getattr(base.dataset, "target_channels", None),
        "margin": model.margin,
        "selected_epoch": best_epoch if best_atoms is not None else spec.epochs,
        "selection_rule": "minimum full-pool patient validation reconstruction" if val else "final epoch",
        "manifest_segments": len(base.dataset),
        "train_patients": len(train.dataset.subjects),
        "atom_stats_split": "val_full" if val is not None else "train_full",
        "pretraining_draws": saved_draws,
    }
    model.fitted = True


@dataclass(frozen=True)
class SupervisedSpec:
    """Pooled-patient head warm-up and joint schedule; all settings are saved with the checkpoint.

    Empty hidden_dims gives logistic regression; positive widths add ReLU hidden
    layers. lambda_center overrides per-run gradient calibration for paired sweeps.
    """

    warmup_epochs: int = 50
    joint_epochs: int = 30
    warmup_lr: float = 1e-2
    dictionary_lr: float = 1e-4
    head_lr: float = 1e-3
    gamma: float = 1e-3
    epsilon: float = 1e-6
    hidden_dims: tuple[int, ...] = ()
    lambda_center: float | None = None
    lambda_multiplier: float = 0.1
    calibration_batches: int = 32
    ramp_epochs: int = 5


class PatientHead(nn.Module):
    """Classifier on standardized patient means, with frozen training-only normalization buffers."""

    def __init__(self, features: torch.Tensor, scale_floor: float, hidden_dims: tuple[int, ...] = ()) -> None:
        super().__init__()
        if (
            not math.isfinite(scale_floor)
            or scale_floor <= 0
            or any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in hidden_dims)
        ):
            raise ValueError("positive scale floor and hidden layer widths are required")
        self.register_buffer("mean", features.mean(0))
        self.register_buffer("scale", features.std(0, correction=0).clamp_min(scale_floor))
        layers: list[nn.Module] = []
        widths = (features.shape[1], *hidden_dims, 1)
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers).to(features.device)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """One logit per patient; nonlinearities, when configured, occur after pooling."""
        return self.network((q - self.mean) / self.scale).squeeze(-1)

    def penalty(self) -> torch.Tensor:
        """Half the squared norm of weight matrices; intercepts are unpenalized."""
        return (
            sum(
                (p.square().sum() for name, p in self.named_parameters() if name.endswith("weight")),
                self.mean.new_zeros(()),
            )
            / 2
        )


def joint_backward(
    model: WaveformSAE,
    head: PatientHead,
    batch: tuple[torch.Tensor, ...],
    scales: tuple[float, float, float],
) -> tuple[float, float]:
    """Exact sampled-bag gradient in two passes; memory holds only one segment chunk's graph.

    scales=(reconstruction coefficient, classification coefficient, gamma).
    First pool detached features, differentiate one BCE per patient into a proxy,
    then replay encoder chunks using that proxy gradient plus tied reconstruction.
    No optimizer step occurs between passes. Encoder has no dropout/BatchNorm.
    """
    x, labels, patient, _ = batch
    inverse, weights = _batch_weights(model, x, patient)
    n = int(inverse.max()) + 1
    with torch.no_grad():
        parts = [segment_values(model, chunk) for chunk in x.split(model.train_spec.segment_batch)]
        features = torch.cat([part[1] for part in parts])
        losses = torch.cat([part[0] for part in parts])
        q = features.new_zeros(n, features.shape[1]).index_add(0, inverse, features * weights[:, None] * n)
        targets = patient_means(labels.to(model.device), inverse, n)
        reconstruction = float((losses * weights).sum())
    proxy = q.detach().requires_grad_(True)
    classification = F.binary_cross_entropy_with_logits(head(proxy), targets) + scales[2] * head.penalty()
    (scales[1] * classification).backward()
    assert proxy.grad is not None
    feature_gradient = proxy.grad[inverse] * weights[:, None] * n
    for start in range(0, len(x), model.train_spec.segment_batch):
        stop = start + model.train_spec.segment_batch
        loss, feature, _, _ = segment_values(model, x[start:stop])
        surrogate = scales[0] * (loss * weights[start:stop]).sum()
        surrogate = surrogate + (feature * feature_gradient[start:stop]).sum()
        surrogate.backward()
    return reconstruction, float(classification.detach())


def validate_supervised(model: WaveformSAE, spec: SupervisedSpec) -> None:
    """Refuse invalid schedules or geometry before expensive pretraining starts."""
    if model.spec.top_r != 1 or model.spec.atom_norm != "unit" or model.train_spec.crop_sampling != "full":
        raise ValueError("joint patient training requires full segments, unit atoms and R=1")
    if model.train_spec.sampling_unit != "patient":
        raise ValueError("joint patient training requires patient reconstruction pretraining")
    for name in ("warmup_epochs", "joint_epochs", "calibration_batches", "ramp_epochs"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("warmup_lr", "dictionary_lr", "head_lr", "epsilon", "lambda_multiplier"):
        value = getattr(spec, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(spec.gamma) or spec.gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if spec.lambda_center is not None and (not math.isfinite(spec.lambda_center) or spec.lambda_center <= 0):
        raise ValueError("lambda_center must be finite and positive")
    if spec.joint_epochs < spec.ramp_epochs:
        raise ValueError("joint_epochs must reach ramp_epochs for checkpoint eligibility")
    if any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in spec.hidden_dims):
        raise ValueError("hidden_dims must contain positive integer widths")


def _head_state(head: PatientHead) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def _warmup(
    head: PatientHead,
    summaries: tuple[PatientSummary, PatientSummary],
    loaders: tuple[DataLoader, DataLoader],
    spec: SupervisedSpec,
    seed: int,
) -> list[dict[str, float]]:
    train, val = summaries
    labels = loaders[0].dataset.labels.to(head.mean.device)
    val_labels = loaders[1].dataset.labels.to(head.mean.device)
    final = head.network[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(float(torch.logit(labels.mean())))
    optimizer = torch.optim.Adam(head.parameters(), lr=spec.warmup_lr)
    generator = torch.Generator().manual_seed(seed)
    best, state = math.inf, _head_state(head)
    history = []
    for epoch in range(spec.warmup_epochs):
        for indices in torch.randperm(len(labels), generator=generator).split(loaders[0].batch_sampler.patients):
            indices = indices.to(labels.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(head(train.features[indices]), labels[indices])
            (loss + spec.gamma * head.penalty()).backward()
            optimizer.step()
        with torch.no_grad():
            value = float(F.binary_cross_entropy_with_logits(head(val.features), val_labels))
        history.append({"epoch": float(epoch + 1), "val_bce": value})
        if value < best:
            best, state = value, _head_state(head)
    head.load_state_dict(state)
    return history


def _projected_gradient(model: WaveformSAE) -> torch.Tensor:
    gradient = model.atoms.grad
    if gradient is None:
        raise ValueError("dictionary gradient is disconnected")
    atoms = model._dictionary().detach()
    return gradient - (gradient * atoms).sum((1, 2), keepdim=True) * atoms


def _calibrate(
    model: WaveformSAE,
    head: PatientHead,
    loader: DataLoader,
    spec: SupervisedSpec,
    r0: float,
) -> tuple[float, list[dict[str, float]], list[list[int]]]:
    ratios: list[float] = []
    records: list[dict[str, float]] = []
    draws: list[list[int]] = []
    loader.batch_sampler.seed = model.random_state + 200000
    epoch = 0
    while len(records) < spec.calibration_batches:
        loader.batch_sampler.epoch = epoch
        for batch in loader:
            model.zero_grad(set_to_none=True)
            reconstruction_backward(model, batch[0], batch[2])
            gr = _projected_gradient(model).clone() / r0
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            joint_backward(model, head, batch, (0.0, 1.0, spec.gamma))
            gc = _projected_gradient(model).clone()
            nr, nc = float(gr.norm()), float(gc.norm())
            if not math.isfinite(nr + nc) or min(nr, nc) <= torch.finfo(gr.dtype).tiny:
                raise ValueError("degenerate calibration gradients; inspect head, sparse codes and feature scales")
            ratios.append(nr / nc)
            records.append(
                {
                    "probe": float(len(records)),
                    "reconstruction_norm": nr,
                    "classification_norm": nc,
                    "ratio": nr / nc,
                    "cosine": float((gr * gc).sum()) / (nr * nc),
                }
            )
            draws.append(batch[3].tolist())
            if len(records) == spec.calibration_batches:
                break
        epoch += 1
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    center = float(np.median(ratios)) if spec.lambda_center is None else spec.lambda_center
    return center, records, draws


@dataclass(frozen=True)
class PatientPreparation:
    """CPU snapshot shared by lambda arms; restore copies, never train these tensors.

    Includes dictionary identity, frozen normalization/head state, reconstruction
    references, warm-up history, and calibration results. ``save``/``load`` use a
    versioned weights-only-compatible payload, independent of Python class pickles.
    """

    identity: str
    atoms: torch.Tensor
    head: dict[str, torch.Tensor]
    segment_length: int
    r0: float
    val_reconstruction: float
    val_zero: float
    frozen_val_patient_auroc: float
    lambda_center: float
    warmup: list[dict[str, float]]
    calibration: list[dict[str, float]]
    probe_draws: list[list[int]]
    val_subjects: list[str]
    val_labels: torch.Tensor
    val_logits: torch.Tensor

    def save(self, path: str | Path) -> None:
        """Save a reusable preparation to ``path`` without retaining any optimizer state."""
        torch.save({"format": 1, "preparation": asdict(self)}, path)

    @classmethod
    def load(cls, path: str | Path) -> PatientPreparation:
        """Load a CPU preparation; raise ValueError for unsupported checkpoint formats."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != 1 or "preparation" not in saved:
            raise ValueError("unsupported patient preparation checkpoint; prepare this seed again")
        return cls(**saved["preparation"])

    def write_reports(self, output_dir: Path) -> None:
        """Copy the saved setup diagnostics to an arm directory without running setup."""
        pl.DataFrame(self.warmup).write_csv(output_dir / "patient_warmup.csv")
        pl.DataFrame(self.calibration).write_csv(output_dir / "patient_gradient_calibration.csv")
        pl.DataFrame(
            {"subject": self.val_subjects, "label": self.val_labels.numpy(), "logit": self.val_logits.numpy()}
        ).write_csv(output_dir / "frozen_patient_scores_val.csv")


def _supervised_loaders(
    model: WaveformSAE, loaders: tuple[DataLoader, DataLoader], spec: SupervisedSpec
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Validate metadata and build patient loaders without reading a signal segment."""
    validate_supervised(model, spec)
    if not model.fitted:
        raise ValueError("pretrain the dictionary before fitting the patient classifier")
    train = patient_loader(loaders[0], model, True)
    train_full, val = patient_loader(loaders[0], model, False), patient_loader(loaders[1], model, False)
    _check_splits(train, val, model.provenance)
    if any(len(loader.dataset.labels.unique()) != 2 for loader in (train, val)):
        raise ValueError("both classes must occur in training and validation patients")
    length = getattr(loaders[0].dataset, "target_len", None)
    if (
        not isinstance(length, int)
        or length <= 0
        or length % model.spec.pool_len
        or getattr(loaders[1].dataset, "target_len", None) != length
    ):
        raise ValueError("patient classification needs fixed equal target_len for training and validation segments")
    return train, train_full, val, length


def _preparation_identity(model: WaveformSAE, loaders: tuple[DataLoader, DataLoader], spec: SupervisedSpec) -> str:
    """Bind setup to dictionary settings, seed, preprocessing and ordered labeled plans.

    Lambda is the only allowed setting change. Metadata hashes cannot detect an
    EDF modified in place; the corpus and preprocessing sidecars must stay fixed.
    """
    settings = asdict(spec)
    for key in ("lambda_center", "lambda_multiplier"):
        settings.pop(key)
    datasets = []
    for loader in loaders:
        source = loader.dataset
        plan = source.plan.to_json(orient="split", index=False, double_precision=15, default_handler=str)
        attributes = (
            "target_channels",
            "target_len",
            "target_sfreq",
            "mode",
            "bipolar",
            "filter_freq",
            "notch_freqs",
            "interpolate_bad_channels",
            "apply_aas",
            "aas_fmax",
            "rename_channels",
            "set_montage",
            "pick_channels",
            "ica_keep_labels",
            "brain_ic_min_gof",
            "brain_ic_use_dipoles",
            "ic_bag_max_k",
            "ic_bag_sign_normalize",
            "ic_bag_rank_by",
            "target_cols",
            "montages",
        )
        datasets.append(
            {
                "plan": hashlib.sha256(plan.encode()).hexdigest(),
                "settings": {key: getattr(source, key, None) for key in attributes},
                "signal_cache_identity": getattr(getattr(source, "signal_cache", None), "identity", None),
            }
        )
    identity = {
        "model": asdict(model.spec),
        "dtype": str(model.atoms.dtype),
        "sampling": asdict(model.train_spec),
        "seed": model.random_state,
        "provenance": model.provenance,
        "supervised": settings,
        "datasets": datasets,
    }
    # Match the existing manifest/campaign JSON convention without adding a runtime dependency.
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def prepare_patient_supervised(
    model: WaveformSAE,
    loaders: tuple[DataLoader, DataLoader],
    spec: SupervisedSpec,
    output_dir: Path,
) -> PatientPreparation:
    """Fit frozen normalization/head and calibrate once for a pretrained seed.

    Args:
        model: Fitted tied dictionary; its atoms remain unchanged.
        loaders: Complete training and validation segment pools.
        spec: Warm-up/calibration settings and the intended joint schedule.
        output_dir: Destination for patient_preparation.pt and setup reports.

    Returns:
        CPU snapshot reusable across lambda centers/multipliers.

    Raises:
        ValueError: Incompatible metadata or degenerate reconstruction/gradient references.
    """
    train, train_full, val, length = _supervised_loaders(model, loaders, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(loaders[0].dataset[0][0].shape[-1]) != length:
        raise ValueError("loaded segment length disagrees with target_len")
    n_pool = length // model.spec.pool_len
    train_summary = summarize_patients(model, train_full, "Preparation train pool")
    val_summary = summarize_patients(model, val, "Preparation validation pool")
    r0 = float(train_summary.reconstruction.mean())
    v0, zero = float(val_summary.reconstruction.mean()), float(val_summary.zero.mean())
    if not math.isfinite(r0 + v0 + zero) or r0 <= 0 or zero <= v0:
        raise ValueError("need positive r0 and positive pretrained validation reconstruction gain")
    torch.manual_seed(model.random_state)
    head = PatientHead(train_summary.features, spec.epsilon / n_pool, tuple(spec.hidden_dims))
    warmup = _warmup(head, (train_summary, val_summary), (train, val), spec, model.random_state + 300000)
    with torch.no_grad():
        frozen_scores = head(val_summary.features).cpu().clone()
    frozen_score = float(roc_auc_score(val.dataset.labels.numpy(), frozen_scores.numpy()))
    center, calibration, probe_draws = _calibrate(model, head, train, spec, r0)
    prepared = PatientPreparation(
        identity=_preparation_identity(model, loaders, spec),
        atoms=model.atoms.detach().cpu().clone(),
        head=_head_state(head),
        segment_length=length,
        r0=r0,
        val_reconstruction=v0,
        val_zero=zero,
        frozen_val_patient_auroc=frozen_score,
        lambda_center=center,
        warmup=warmup,
        calibration=calibration,
        probe_draws=probe_draws,
        val_subjects=val.dataset.subjects.tolist(),
        val_labels=val.dataset.labels.cpu().clone(),
        val_logits=frozen_scores,
    )
    prepared.save(output_dir / "patient_preparation.pt")
    prepared.write_reports(output_dir)
    return prepared


def _restore_patient_head(
    prepared: PatientPreparation, model: WaveformSAE, loaders: tuple[DataLoader, DataLoader], spec: SupervisedSpec
) -> PatientHead:
    """Refuse mismatched setup before reading signals, and restore independent head tensors."""
    if prepared.identity != _preparation_identity(model, loaders, spec):
        raise ValueError(
            "patient preparation identity differs: settings, seed, provenance or labeled manifests changed"
        )
    if not torch.equal(model.atoms.detach().cpu(), prepared.atoms):
        raise ValueError("patient preparation dictionary differs; reload this seed's D0 before each arm")
    scalars = (
        prepared.r0,
        prepared.val_reconstruction,
        prepared.val_zero,
        prepared.frozen_val_patient_auroc,
        prepared.lambda_center,
    )
    if (
        not all(math.isfinite(value) for value in scalars)
        or prepared.r0 <= 0
        or prepared.lambda_center <= 0
        or prepared.val_zero <= prepared.val_reconstruction
        or not 0 <= prepared.frozen_val_patient_auroc <= 1
    ):
        raise ValueError("invalid patient preparation baselines")
    if prepared.segment_length != loaders[0].dataset.target_len:
        raise ValueError("patient preparation segment length differs")
    if any(not bool(torch.isfinite(value).all()) for value in prepared.head.values()) or not bool(
        (prepared.head["scale"] > 0).all()
    ):
        raise ValueError("invalid patient preparation head or feature scales")
    # Construction does not fit statistics. Loading copies buffers/parameters;
    # the preparation is unchanged even when an arm updates its own classifier.
    with torch.random.fork_rng(devices=[]):
        head = PatientHead(torch.zeros(2, model.spec.n_atoms, device=model.device), 1.0, tuple(spec.hidden_dims))
    head.load_state_dict(prepared.head)
    return head


def fit_patient_supervised(
    model: WaveformSAE,
    loaders: tuple[DataLoader, DataLoader],
    spec: SupervisedSpec,
    output_dir: Path,
    prepared: PatientPreparation | None = None,
) -> tuple[PatientHead, dict[str, float]]:
    """Jointly fit one pooled BCE per patient, optionally reusing a seed's frozen preparation.

    ``prepared=None`` performs the usual full preparation first. Supplying a
    preparation skips its full-pool scans, warm-up and probes. It must match the
    current D0, metadata and all settings except lambda center/multiplier, or a
    ValueError is raised before joint updates. Each call restores an independent
    head and creates a fresh Adam optimizer. Sampling, loss, validation and
    checkpoint selection are identical in both paths. Test signals are not read.
    """
    train, train_full, val, length = _supervised_loaders(model, loaders, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    reused_preparation = prepared is not None
    if prepared is None:
        prepared = prepare_patient_supervised(model, loaders, spec, output_dir)
    head = _restore_patient_head(prepared, model, loaders, spec)
    if reused_preparation:
        prepared.write_reports(output_dir)
    r0, v0, zero = prepared.r0, prepared.val_reconstruction, prepared.val_zero
    frozen_score = best_score = prepared.frozen_val_patient_auroc
    labels = val.dataset.labels.numpy()
    best_atoms, best_head, selected_epoch = model.atoms.detach().clone(), _head_state(head), 0
    center = prepared.lambda_center if spec.lambda_center is None else spec.lambda_center
    probe_draws = prepared.probe_draws
    target_weight = center * spec.lambda_multiplier
    train.batch_sampler.seed = model.random_state
    optimizer = torch.optim.Adam(
        [
            {"params": [model.atoms], "lr": spec.dictionary_lr},
            {"params": list(head.parameters()), "lr": spec.head_lr},
        ]
    )
    history, saved_draws = [], []
    threshold = v0 + 0.1 * (zero - v0)
    for epoch in range(spec.joint_epochs):
        epoch_started = time.perf_counter()
        train.batch_sampler.epoch = epoch
        weight = target_weight * min((epoch + 1) / spec.ramp_epochs, 1.0)
        total_r, total_c, patients = 0.0, 0.0, 0
        epoch_draws = []
        for batch in tqdm(train, desc=f"Patient joint {epoch + 1}/{spec.joint_epochs}"):
            optimizer.zero_grad(set_to_none=True)
            r, c = joint_backward(model, head, batch, (1 / r0, weight, spec.gamma))
            optimizer.step()
            with torch.no_grad():
                model.atoms.copy_(model._constrain(model.atoms))
            n = int(batch[2].unique().numel())
            total_r += r * n
            total_c += c * n
            patients += n
            epoch_draws.append(batch[3].tolist())
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        summary = summarize_patients(model, val, f"Joint validation {epoch + 1}/{spec.joint_epochs}")
        with torch.no_grad():
            scores = head(summary.features)
            bce = float(F.binary_cross_entropy_with_logits(scores, val.dataset.labels.to(model.device)))
        auroc = float(roc_auc_score(labels, scores.cpu().numpy()))
        reconstruction = float(summary.reconstruction.mean())
        eligible = epoch + 1 >= spec.ramp_epochs and reconstruction <= threshold
        history.append(
            {
                "epoch": epoch + 1,
                "lambda": weight,
                "train_reconstruction": total_r / patients,
                "train_classification_penalized": total_c / patients,
                "val_reconstruction": reconstruction,
                "val_bce": bce,
                "val_auroc": auroc,
                "eligible": eligible,
                "val_segments_skipped": summary.skipped,
            }
        )
        saved_draws.append(epoch_draws)
        if eligible and auroc > best_score:
            best_score, selected_epoch = auroc, epoch + 1
            best_atoms, best_head = model.atoms.detach().clone(), _head_state(head)
        log.info(f"Patient joint epoch {epoch + 1}: val AUROC {auroc:.4f}, reconstruction {reconstruction:.6g}")
        log.info(
            "Joint epoch {} timings: {:.1f}s training, {:.1f}s validation/selection, {:.1f}s total",
            epoch + 1,
            training_seconds,
            time.perf_counter() - validation_started,
            time.perf_counter() - epoch_started,
        )
    with torch.no_grad():
        model.atoms.copy_(best_atoms)
    head.load_state_dict(best_head)
    selected_summary = summarize_patients(model, val, "Selected-state validation pool")
    _store_atom_statistics(model, selected_summary)
    model.fit_meta["supervised"] = {
        "spec": asdict(spec),
        "r0": r0,
        "lambda_center": center,
        "lambda": target_weight,
        "selected_epoch": selected_epoch,
        "selection_rule": "max eligible val patient AUROC including frozen control",
        "segment_length": length,
        "feature": "retained_amplitude_sum_over_M",
        "preparation_identity": prepared.identity,
    }
    model.save_artifacts(output_dir)
    torch.save(
        {
            "format": 1,
            "spec": asdict(spec),
            "dictionary": torch.load(output_dir / "sae_state.pt", map_location="cpu", weights_only=True),
            "head": _head_state(head),
            "r0": r0,
            "lambda_center": center,
            "lambda": target_weight,
            "selected_epoch": selected_epoch,
            "segment_length": length,
            "preparation_identity": prepared.identity,
            "probe_draws": probe_draws,
            "joint_draws": saved_draws,
        },
        output_dir / "patient_model.pt",
    )
    pl.DataFrame(history).write_csv(output_dir / "patient_joint_training.csv")
    for name, loader in (("train", train_full), ("val", val)):
        summary = selected_summary if name == "val" else summarize_patients(model, loader, "Selected-state train pool")
        with torch.no_grad():
            scores = head(summary.features).cpu().numpy()
        pl.DataFrame(
            {
                "subject": loader.dataset.subjects,
                "label": loader.dataset.labels.numpy(),
                "segments": summary.counts.cpu().numpy(),
                "logit": scores,
                "reconstruction": summary.reconstruction.cpu().numpy(),
            }
        ).write_csv(output_dir / f"patient_scores_{name}.csv")
    return head, {
        "val_patient_auroc": best_score,
        "frozen_val_patient_auroc": frozen_score,
        "selected_joint_epoch": float(selected_epoch),
        "lambda": target_weight,
        "r0": r0,
    }


def load_patient_model(path: str | Path, device: str = "cpu") -> tuple[WaveformSAE, PatientHead]:
    """Restore the selected dictionary and head from patient_model.pt, including frozen normalization.

    The returned model carries training provenance for the usual split/data checks.
    Pool full-bag segment features before applying the head. The saved
    ``model.fit_meta['supervised']['segment_length']`` must match inference input.
    """
    saved = torch.load(path, map_location=device, weights_only=True)
    dictionary = saved["dictionary"]
    if (
        saved.get("format") != 1
        or dictionary.get("model") != MODEL_TAG
        or dictionary.get("format") != CHECKPOINT_FORMAT
    ):
        raise ValueError(
            f"unsupported patient waveform checkpoint (dictionary format {dictionary.get('format')!r}; "
            f"this code reads format {CHECKPOINT_FORMAT}, top_r atoms per pooling window)"
        )
    model = WaveformSAE(
        WaveformSpec(**dictionary["spec"]),
        device=device,
        train_spec=WaveformTrainSpec(**dictionary["fit"]["train_spec"]),
    )
    model.atoms = nn.Parameter(dictionary["atoms"].to(model.device))
    model.fit_meta, model.provenance = dictionary["fit"], dictionary["provenance"]
    model.history, model.atom_stats = dictionary["history"], dictionary["atom_stats"]
    model.won_windows = dictionary.get("won_windows") or {}
    model.fitted = True
    model.validate_artifact()
    head = PatientHead(
        torch.zeros(2, model.spec.n_atoms, device=model.device), 1.0, tuple(saved["spec"]["hidden_dims"])
    )
    head.load_state_dict(saved["head"])
    model.eval()
    head.eval()
    return model, head
