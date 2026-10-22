"""Multichannel waveform SAE with finite-domain, samplewise local competition.

The mathematical definition is in ``Local Competition Waveform Autoencoder
Mathematics.md``. PyTorch tensors use channel-first layout: inputs ``(B, C, T)``,
waveform shapes ``(K, C, L)``, and activations ``(B, K, T)``. There is no temporal
divisibility requirement. The coverage planner is independent of this model.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.components.sae_coverage import CoverageChunk, plan_coverage
from src.models.components.hydra_transform import HydraTransform
from src.models.components.shapeconv_sae import CHECKPOINT_NAME, ShapeConvSAE

__all__ = [
    "CHECKPOINT_NAME",
    "MODEL_TAG",
    "LocalCompetitionSAE",
    "LocalCompetitionSpec",
    "LocalCompetitionTrainSpec",
    "local_competition",
]

MODEL_TAG = "local_competition_sae"
CHECKPOINT_FORMAT = 1
_EPS = 1e-8


@dataclass(frozen=True)
class LocalCompetitionSpec:
    """Model geometry, preprocessing, and tying, persisted in every checkpoint.

    ``competition_scale=P`` sets radius ``P-1``; ``top_r=R`` allows fewer than R
    neighbors with a response greater than or equal to the candidate. All K atoms
    are retained. Shape components have unit temporal norm. Nonnegative channel
    weights have unit row norm in the decoder and, when tied, the encoder.
    Untied encoder channel weights have free row norm. Biases are unrestricted.
    ``pre_emphasis`` defaults to ``none``; ``diff`` enables length-preserving
    first differencing as an optional comparison.
    ``center_shapes`` additionally imposes zero temporal mean on each shape.
    """

    n_atoms: int = 64
    atom_len: int = 80
    competition_scale: int = 32
    top_r: int = 1
    weight_tying: str = "tied"
    pre_emphasis: str = "none"
    center_shapes: bool = False


@dataclass(frozen=True)
class LocalCompetitionTrainSpec:
    """Systematic-coverage training schedule, independent of competition geometry.

    ``center_len`` is the maximum scored samples per chunk; the final center can
    be shorter. Every chunk carries real context on both sides. ``crop_batch``
    bounds chunks per optimizer update. ``shuffle_chunks`` permutes their order
    within each loaded parent batch. The supplied loader controls parent order.
    Zero epochs retains the seeded initialization. No energy gate is applied.
    """

    epochs: int = 30
    lr: float = 1e-3
    center_len: int = 1024
    crop_batch: int = 64
    shuffle_chunks: bool = True


def local_competition(z: torch.Tensor, competition_scale: int, top_r: int) -> torch.Tensor:
    """Select rectified ``z`` of shape ``(B, K, T)`` against available neighbors.

    Each position competes simultaneously with all other positions within radius
    ``competition_scale-1``. Equal competitors count against the candidate, so
    ties can suppress every tied peak. Inputs are expected to be finite and
    nonnegative. The returned tensor preserves surviving values and gradients.

    R=1 uses two stride-one max pools, excluding the candidate. General R uses
    bounded-memory offset comparisons: O(BKT min(P,T)) work, O(BKT) auxiliary
    storage. Generic max-pooling kernel work can depend on P; this function does
    not claim a specialized linear-time GPU rolling-maximum implementation.

    Raises:
        ValueError: If shape, floating dtype, P, or R is invalid.
    """
    if z.ndim != 3 or z.shape[-1] < 1 or not z.is_floating_point():
        raise ValueError("z must be a floating tensor of shape (B, K, T), with T >= 1")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (competition_scale, top_r)):
        raise ValueError("competition_scale and top_r must be integers")
    if competition_scale < 1 or not 1 <= top_r <= competition_scale:
        raise ValueError("need competition_scale >= 1 and 1 <= top_r <= competition_scale")
    radius = min(competition_scale - 1, z.shape[-1] - 1)
    if radius == 0:
        return z
    with torch.no_grad():
        if top_r == 1:
            left = F.max_pool1d(F.pad(z[..., :-1], (radius, 0), value=-math.inf), radius, stride=1)
            right = F.max_pool1d(F.pad(z[..., 1:], (0, radius), value=-math.inf), radius, stride=1)
            keep = z > torch.maximum(left, right)
        else:
            count = torch.zeros_like(z, dtype=torch.int32)
            for offset in range(1, radius + 1):
                count[..., offset:] += (z[..., :-offset] >= z[..., offset:]).to(count.dtype)
                count[..., :-offset] += (z[..., offset:] >= z[..., :-offset]).to(count.dtype)
            keep = count < top_r
    return z * keep.to(z.dtype)


class LocalCompetitionSAE(nn.Module):
    """Dictionary-only SAE exposing the shared ``train_sae.py`` feature hooks.

    Args:
        spec: Encoder/decoder definition; defaults to ``LocalCompetitionSpec``.
        train_spec: Coverage and optimization settings.
        random_state: Seed for Gaussian initialization and chunk permutations.
        device: CPU, CUDA device, or ``auto``.
        pretrained: Optional checkpoint of this model and exactly this spec.

    The channel count is discovered at fitting or checkpoint loading. Channel
    identity/order is checked when loader metadata is available. ``encode`` and
    ``decode`` operate on arbitrary lengths; classifier features are not defined.
    """

    enc_shapes: nn.Parameter | None
    enc_alpha: nn.Parameter | None
    dec_shapes: nn.Parameter | None
    dec_alpha: nn.Parameter | None
    beta: nn.Parameter | None
    mu: nn.Parameter | None

    def __init__(
        self,
        spec: LocalCompetitionSpec | None = None,
        train_spec: LocalCompetitionTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec or LocalCompetitionSpec()
        self.train_spec = train_spec or LocalCompetitionTrainSpec()
        self._check_specs()
        self.random_state = int(random_state)
        self.device = HydraTransform._resolve_device(device)
        for name in ("enc_shapes", "enc_alpha", "dec_shapes", "dec_alpha", "beta", "mu"):
            self.register_parameter(name, None)
        self.history: list[dict[str, float]] = []
        self.provenance: dict[str, Any] = {}
        self.fit_meta: dict[str, Any] = {}
        self.atom_stats: dict[str, list[float]] = {}
        self.fitted = False
        if pretrained is not None:
            self.load_checkpoint(pretrained)

    def _check_specs(self) -> None:
        spec, schedule = self.spec, self.train_spec
        integers = (
            spec.n_atoms,
            spec.atom_len,
            spec.competition_scale,
            spec.top_r,
            schedule.epochs,
            schedule.center_len,
            schedule.crop_batch,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
            raise ValueError("model dimensions, epochs, center_len, and crop_batch must be integers")
        if min(spec.n_atoms, spec.atom_len, spec.competition_scale) < 1:
            raise ValueError("n_atoms, atom_len, and competition_scale must be positive")
        if not 1 <= spec.top_r <= spec.competition_scale:
            raise ValueError("top_r must lie in [1, competition_scale]")
        if spec.weight_tying not in ("tied", "untied") or spec.pre_emphasis not in ("none", "diff"):
            raise ValueError("weight_tying must be tied/untied and pre_emphasis must be none/diff")
        if spec.center_shapes and spec.atom_len < 2:
            raise ValueError("center_shapes requires atom_len >= 2")
        if schedule.epochs < 0 or min(schedule.center_len, schedule.crop_batch) < 1:
            raise ValueError("epochs must be nonnegative; center_len and crop_batch must be positive")
        if not math.isfinite(schedule.lr) or schedule.lr <= 0:
            raise ValueError("lr must be finite and positive")

    @property
    def padding(self) -> tuple[int, int]:
        """Asymmetric same-padding widths for odd or even waveform lengths."""
        left = (self.spec.atom_len - 1) // 2
        return left, self.spec.atom_len - 1 - left

    @property
    def margin(self) -> int:
        """Real context samples on each side of a scored center: ``L + P - 2``."""
        return self.spec.atom_len + self.spec.competition_scale - 2

    @property
    def train_subjects(self) -> list[str]:
        """Training-subject provenance, empty before fitting."""
        return list(self.provenance.get("train_subjects", []))

    def initialize_dictionary(self, n_channels: int) -> None:
        """Initialize Gaussian channel shapes, uniform channel weights, and zero biases.

        The seed is local and does not modify the process-global random generator.
        Raises ``ValueError`` for nonpositive C and ``RuntimeError`` if initialized.
        """
        if not isinstance(n_channels, int) or isinstance(n_channels, bool) or n_channels < 1:
            raise ValueError("n_channels must be positive")
        if self.enc_shapes is not None:
            raise RuntimeError("dictionary is already initialized")
        generator = torch.Generator().manual_seed(self.random_state)
        shape = (self.spec.n_atoms, n_channels, self.spec.atom_len)
        initial_shapes = torch.randn(shape, generator=generator).to(self.device)
        initial_alpha = torch.full(shape[:2], n_channels**-0.5, device=self.device)
        self.enc_shapes = nn.Parameter(initial_shapes)
        self.enc_alpha = nn.Parameter(initial_alpha)
        if self.spec.weight_tying == "untied":
            self.dec_shapes = nn.Parameter(initial_shapes.clone())
            self.dec_alpha = nn.Parameter(initial_alpha.clone())
        self.beta = nn.Parameter(torch.zeros(self.spec.n_atoms, device=self.device))
        self.mu = nn.Parameter(torch.zeros(n_channels, device=self.device))
        self.project_parameters()

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Project shapes and channel weights after Adam; biases remain unrestricted.

        Zero shape vectors receive a deterministic unit vector (zero mean when
        requested). A zero decoder-weight row receives uniform unit weights, so
        complete decoder atoms always retain unit norm. Untied encoder-weight
        rows can be zero. This is projected optimization, not a differentiable
        normalization parameterization.
        """
        self._require_dictionary()
        branches: list[Literal["enc", "dec"]] = ["enc"]
        if self.spec.weight_tying == "untied":
            branches.append("dec")
        for branch in branches:
            shapes, alpha = self._factors(branch)
            unit_alpha = branch == "dec" or self.spec.weight_tying == "tied"
            if self.spec.center_shapes:
                shapes.sub_(shapes.mean(-1, keepdim=True))
            norm = torch.linalg.vector_norm(shapes, dim=-1, keepdim=True)
            fallback = torch.zeros_like(shapes)
            fallback[..., 0] = 1.0
            if self.spec.center_shapes:
                fallback[..., 1] = -1.0
                fallback.div_(math.sqrt(2))
            shapes.copy_(torch.where(norm > _EPS, shapes / norm.clamp_min(_EPS), fallback))
            alpha.clamp_(min=0)
            if unit_alpha:
                norm = torch.linalg.vector_norm(alpha, dim=-1, keepdim=True)
                alpha.copy_(
                    torch.where(norm > _EPS, alpha / norm.clamp_min(_EPS), alpha.new_full((), alpha.shape[1] ** -0.5))
                )

    def _require_dictionary(self) -> None:
        if self.enc_shapes is None:
            raise RuntimeError("dictionary is uninitialized; fit, initialize_dictionary, or load a checkpoint first")

    def _factors(self, branch: Literal["enc", "dec"]) -> tuple[nn.Parameter, nn.Parameter]:
        """Resolve tied factors and require both parameters to be initialized."""
        if branch not in ("enc", "dec"):
            raise ValueError("branch must be enc or dec")
        if branch == "dec" and self.spec.weight_tying == "untied":
            shapes, alpha = self.dec_shapes, self.dec_alpha
        else:
            shapes, alpha = self.enc_shapes, self.enc_alpha
        if shapes is None or alpha is None:
            raise RuntimeError("dictionary is uninitialized; fit, initialize_dictionary, or load a checkpoint first")
        return shapes, alpha

    def _dictionary(self, branch: Literal["enc", "dec"] = "enc") -> torch.Tensor:
        shapes, alpha = self._factors(branch)
        return alpha.unsqueeze(-1) * shapes

    def _rows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] < 1:
            raise ValueError("expected parent windows (B, C, T), with T >= 1")
        if self.enc_shapes is not None and x.shape[1] != self.enc_shapes.shape[1]:
            raise ValueError("input channel count differs from the dictionary")
        dtype = torch.float32 if self.enc_shapes is None else self.enc_shapes.dtype
        raw = x.to(device=self.device, dtype=dtype)
        flat = raw.flatten(0, 1)
        valid = ShapeConvSAE._valid_rows(flat)
        flat = torch.where(valid.unsqueeze(-1), flat, torch.zeros_like(flat))
        if self.spec.pre_emphasis == "diff":
            flat = flat.diff(dim=-1, prepend=flat[:, :1])
        rows = ShapeConvSAE._robust_scale(flat)
        rows = torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))
        return rows.reshape(raw.shape), valid.reshape(raw.shape[:2])

    def _responses(self, rows: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(rows, self.padding), atoms)

    def _select(self, responses: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        if beta is None:
            raise RuntimeError("encoder bias is uninitialized")
        z = F.relu(responses + beta[None, :, None])
        return local_competition(z, self.spec.competition_scale, self.spec.top_r)

    def _decode(self, code: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        mu = self.mu
        if mu is None:
            raise RuntimeError("decoder bias is uninitialized")
        left = self.padding[0]
        synthesis = F.conv_transpose1d(code, atoms)[..., left : left + code.shape[-1]]
        return synthesis + mu[None, :, None]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sparse ``(B,K,T)`` codes and preprocessed ``(B,C,T)`` full parents.

        Preprocess each parent once, before any chunking. Robust scaling depends
        on the full parent; independently normalizing crops would change results.
        Finite boundaries use zero convolution padding and available competitors.
        """
        atoms = self._dictionary("enc")
        rows, _ = self._rows(x)
        return self._select(self._responses(rows, atoms)), rows

    @torch.no_grad()
    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """Return ``(B,C,T)`` reconstruction in the preprocessed signal domain."""
        atoms = self._dictionary("dec")
        if code.ndim != 3 or code.shape[1] != self.spec.n_atoms or code.shape[-1] < 1:
            raise ValueError("expected code of shape (B, n_atoms, T), with T >= 1")
        return self._decode(code.to(device=atoms.device, dtype=atoms.dtype), atoms)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Reject classifier extraction; use ``train_sae.py`` and encode/decode."""
        raise NotImplementedError("LocalCompetitionSAE trains dictionaries only; classifier features are not defined")

    def _chunk_batch(
        self,
        rows: torch.Tensor,
        valid: torch.Tensor,
        items: list[tuple[int, CoverageChunk]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not items:
            raise ValueError("cannot pack an empty chunk batch")
        length = max(chunk.input_length for _, chunk in items)
        crops = rows.new_zeros((len(items), rows.shape[1], length))
        channel_valid = torch.zeros(len(items), rows.shape[1], dtype=torch.bool, device=rows.device)
        center_mask = torch.zeros(len(items), length, dtype=torch.bool, device=rows.device)
        for i, (parent, chunk) in enumerate(items):
            crops[i, :, : chunk.input_length] = rows[parent, :, chunk.input_start : chunk.input_stop]
            channel_valid[i] = valid[parent]
            left = chunk.center_start - chunk.input_start
            center_mask[i, left : left + chunk.center_length] = True
        return crops, channel_valid, center_mask

    def _step(
        self, crops: torch.Tensor, valid: torch.Tensor, center_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self._select(self._responses(crops, self._dictionary("enc")))
        reconstruction = self._decode(code, self._dictionary("dec"))
        mask = valid.unsqueeze(-1) & center_mask.unsqueeze(1)
        error = torch.where(mask, reconstruction - crops, torch.zeros_like(crops))
        count = mask.sum()
        loss = error.square().sum() / (2 * count.clamp_min(1))
        with torch.no_grad():
            scored_code = code * center_mask.unsqueeze(1)
            positive = scored_code > 0
            stats = torch.stack(
                [
                    error.square().sum(),
                    torch.where(mask, crops, torch.zeros_like(crops)).square().sum(),
                    count.to(crops.dtype),
                    positive.sum().to(crops.dtype),
                    center_mask.sum().to(crops.dtype),
                    crops.new_tensor(crops.shape[0]),
                ]
            ).double()
            per_atom = torch.stack(
                [
                    scored_code.square().sum(dim=(0, 2)),
                    scored_code.sum(dim=(0, 2)),
                    positive.sum(dim=(0, 2)).to(crops.dtype),
                ]
            ).double()
        return loss, stats, per_atom

    def _run_epoch(
        self, loader: DataLoader, generator: torch.Generator, optimizer: torch.optim.Optimizer | None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        per_atom = torch.zeros(3, self.spec.n_atoms, dtype=torch.float64, device=self.device)
        counts = dict.fromkeys(
            (
                "parent_windows",
                "invalid_channel_rows",
                "unusable_parent_windows",
                "boundary_excluded_samples",
                "n_steps",
            ),
            0.0,
        )
        training = optimizer is not None
        self.train(training)
        iterator = tqdm(loader, desc="Local SAE training coverage" if training else "Local SAE validation coverage")
        for x, _ in iterator:
            rows, valid = self._rows(x)
            plan = plan_coverage(rows.shape[-1], self.train_spec.center_len, self.margin)
            parents = valid.any(dim=1).nonzero(as_tuple=True)[0].tolist()
            counts["parent_windows"] += rows.shape[0]
            counts["invalid_channel_rows"] += valid.numel() - int(valid.sum())
            counts["unusable_parent_windows"] += rows.shape[0] - len(parents)
            counts["boundary_excluded_samples"] += len(parents) * min(rows.shape[-1], 2 * self.margin)
            items = [(parent, chunk) for parent in parents for chunk in plan]
            if training and self.train_spec.shuffle_chunks:
                order = torch.randperm(len(items), generator=generator).tolist()
                items = [items[i] for i in order]
            for start in range(0, len(items), self.train_spec.crop_batch):
                crops, channel_valid, center_mask = self._chunk_batch(
                    rows, valid, items[start : start + self.train_spec.crop_batch]
                )
                with torch.set_grad_enabled(training):
                    loss, stats, atom_sums = self._step(crops, channel_valid, center_mask)
                    if optimizer is not None:
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("non-finite local-competition reconstruction loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        self.project_parameters()
                        counts["n_steps"] += 1
                totals += stats
                per_atom += atom_sums
        return totals, per_atom, counts

    @staticmethod
    def _summary(totals: torch.Tensor, prefix: str = "") -> dict[str, float]:
        residual, signal, channel_samples, active, samples, chunks = totals.tolist()
        return {
            f"{prefix}objective": residual / (2 * channel_samples) if channel_samples else float("nan"),
            f"{prefix}residual_frac": residual / max(signal, _EPS),
            f"{prefix}active_coefficients": active,
            f"{prefix}scored_samples": samples,
            f"{prefix}scored_channel_samples": channel_samples,
            f"{prefix}n_chunks": chunks,
        }

    @staticmethod
    def atom_usage(per_atom: torch.Tensor, prefix: str = "") -> dict[str, float]:
        """Count atoms without a positive coefficient and the effective number of atoms.

        ``per_atom`` holds the coefficient energy, sum and positive count per atom
        (rows of ``_step``). The effective number is the exponential entropy of the
        coefficient-energy shares: K when all atoms share the energy equally, 1 when
        one atom holds it, and 0 when no coefficient is positive.
        """
        energy, positive = per_atom[0], per_atom[2]
        total = float(energy.sum())
        share = energy / max(total, _EPS)
        entropy = -(share * share.clamp_min(_EPS).log()).sum()
        return {
            f"{prefix}silent_atoms": float((positive == 0).sum()),
            f"{prefix}effective_atoms": float(entropy.exp()) if total > 0 else 0.0,
        }

    def _check_channels(self, loader: DataLoader) -> None:
        shapes, _ = self._factors("enc")
        channels = getattr(loader.dataset, "target_channels", None)
        saved = self.fit_meta.get("channels")
        if channels is not None:
            channels = [str(channel) for channel in channels]
            if len(channels) != shapes.shape[1] or (saved is not None and channels != saved):
                raise ValueError("loader channel names/order differ from the fitted dictionary")
        if loader.drop_last:
            raise ValueError("systematic coverage requires DataLoader(drop_last=False)")

    def fit_unsupervised(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Fit every eligible center once per pass over supplied parent windows.

        Labels are ignored. Parents are normalized before cutting; real context
        of ``margin`` samples is required on both sides. Their outer margins and
        unusable channels are excluded and counted. Short final centers are kept,
        with loss normalized by their actual valid channel-sample count. Coverage
        is over parents emitted by the loader, not automatically entire EDFs.
        Each epoch record also holds the wall time of each pass, the median and
        maximum drift ``1 - cosine`` of the unit-norm decoder atoms over the epoch,
        and the silent and effective atom counts of each split (``atom_usage``).

        Raises:
            ValueError: Missing subject provenance, channel mismatch, dropped
                parent batches, or no eligible training/validation samples.
        """
        if self.fitted:
            self.validate_artifact()
            self._check_channels(dataloader)
            log.info("Local competition SAE already fitted; skipping dictionary training")
            return
        metadata = dict(provenance or {})
        if not metadata.get("train_subjects"):
            raise ValueError("fit_unsupervised requires provenance['train_subjects']")
        channels = getattr(dataloader.dataset, "target_channels", None)
        if self.enc_shapes is None:
            if len(dataloader.dataset) == 0:
                raise ValueError("training dataset is empty")
            n_channels = len(channels) if channels is not None else int(dataloader.dataset[0][0].shape[0])
            self.initialize_dictionary(n_channels)
        shapes, _ = self._factors("enc")
        self.fit_meta = {
            "train_spec": asdict(self.train_spec),
            "random_state": self.random_state,
            "margin": self.margin,
            "n_channels": int(shapes.shape[1]),
            "channels": None if channels is None else [str(channel) for channel in channels],
            "coverage": "all eligible supplied parent-array samples; source/context validity determined upstream",
            "boundary_policy": "exclude L+P-2 samples at each parent-array edge; no valid-time mask from loader",
            "chunk_order": "shuffled within loaded batch" if self.train_spec.shuffle_chunks else "parent then time",
            "initialization": "Gaussian unit shapes, uniform channel weights, zero biases",
            "preprocessing": "full parent: pre-emphasis, per-channel median/MAD scale, clip",
        }
        self._check_channels(dataloader)
        if val_dataloader is not None:
            self._check_channels(val_dataloader)
        generator = torch.Generator().manual_seed(self.random_state + 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_spec.lr)
        self.history = []
        cumulative_steps = cumulative_samples = 0.0
        sfreq = (metadata.get("data") or {}).get("target_sfreq")
        if sfreq is None:
            sfreq = getattr(dataloader.dataset, "target_sfreq", None)
        previous = self._dictionary("dec").detach()
        for epoch in range(self.train_spec.epochs):
            start = time.perf_counter()
            totals, per_atom, counts = self._run_epoch(dataloader, generator, optimizer)
            if float(totals[2]) == 0:
                raise ValueError("no eligible training samples: parents need usable channels and T > 2 * margin")
            train_seconds = time.perf_counter() - start
            current = self._dictionary("dec").detach()
            drift = 1.0 - (current * previous).sum(dim=(1, 2))
            previous = current
            cumulative_steps += counts["n_steps"]
            cumulative_samples += float(totals[4])
            record = {
                "epoch": float(epoch + 1),
                **self._summary(totals),
                **counts,
                "cumulative_steps": cumulative_steps,
                "cumulative_scored_samples": cumulative_samples,
                "scored_seconds": float(totals[4]) / float(sfreq) if sfreq else float("nan"),
                "activation_density": float(totals[3] / (self.spec.n_atoms * totals[4]).clamp_min(1)),
                **self.atom_usage(per_atom),
                "atom_drift_median": float(drift.median()),
                "atom_drift_max": float(drift.max()),
                "epoch_seconds": train_seconds,
            }
            last_totals, last_atoms, split = totals, per_atom, "train"
            if val_dataloader is not None:
                start = time.perf_counter()
                val_totals, val_atoms, val_counts = self._run_epoch(val_dataloader, generator, None)
                if float(val_totals[2]) == 0:
                    raise ValueError("no eligible validation samples for the real-context coverage plan")
                record.update(self._summary(val_totals, "val_"))
                record.update({f"val_{key}": value for key, value in val_counts.items()})
                record["val_scored_seconds"] = float(val_totals[4]) / float(sfreq) if sfreq else float("nan")
                record["val_activation_density"] = float(
                    val_totals[3] / (self.spec.n_atoms * val_totals[4]).clamp_min(1)
                )
                record.update(self.atom_usage(val_atoms, "val_"))
                record["val_seconds"] = time.perf_counter() - start
                last_totals, last_atoms, split = val_totals, val_atoms, "val"
            self.history.append(record)
            self.atom_stats = {
                "atom": list(range(self.spec.n_atoms)),
                "positive_count": last_atoms[2].tolist(),
                "positive_fraction": (last_atoms[2] / last_totals[4].clamp_min(1)).tolist(),
                "mean_amplitude": (last_atoms[1] / last_atoms[2].clamp_min(1)).tolist(),
                "coefficient_energy": last_atoms[0].tolist(),
            }
            self.fit_meta["atom_stats_split"] = split
            log.info(
                f"Local SAE epoch {epoch + 1}/{self.train_spec.epochs}: "
                f"objective {record['objective']:.6g}, residual {record['residual_frac']:.3%}, "
                f"scored samples {int(record['scored_samples'])}, updates {int(record['n_steps'])}"
            )
        self.provenance = metadata
        self.fitted = True
        self.eval()

    def validate_artifact(self) -> None:
        """Refuse a fitted model without its dictionary or training-subject provenance."""
        if self.fitted:
            self._require_dictionary()
            if not self.provenance.get("train_subjects"):
                raise ValueError("fitted LocalCompetitionSAE has no training-subject provenance")

    def save_artifacts(self, output_dir: str | Path) -> None:
        """Save model-specific checkpoint, effective filters, factors, history, and channel order.

        ``sae_atoms.npy`` is the effective decoder in ``(K,C,L)`` layout;
        ``sae_encoder_atoms.npy`` is its encoder counterpart. The checkpoint stores
        raw factor/bias parameters, exact model spec, fit metadata and provenance.
        This is a fitted-dictionary artifact, not an optimizer-resume checkpoint.
        """
        if not self.fitted:
            raise RuntimeError("fit the dictionary before saving an artifact")
        self.validate_artifact()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        decoder = self._dictionary("dec").detach()
        np.save(output_dir / "sae_atoms.npy", decoder.cpu().numpy())
        np.save(output_dir / "sae_encoder_atoms.npy", self._dictionary("enc").detach().cpu().numpy())
        signal = F.pad(decoder.cumsum(-1), (1, 0)) if self.spec.pre_emphasis == "diff" else decoder
        np.save(output_dir / "sae_atoms_signal.npy", signal.cpu().numpy())
        for name, parameter in self.named_parameters():
            np.save(output_dir / f"sae_{name}.npy", parameter.detach().cpu().numpy())
        torch.save(
            {
                "model": MODEL_TAG,
                "format": CHECKPOINT_FORMAT,
                "spec": asdict(self.spec),
                "fit": self.fit_meta,
                "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
                "history": self.history,
                "atom_stats": self.atom_stats,
                "provenance": self.provenance,
            },
            output_dir / CHECKPOINT_NAME,
        )
        if self.history:
            pl.DataFrame(self.history).write_csv(output_dir / "sae_training.csv")
        if self.atom_stats:
            pl.DataFrame(self.atom_stats).write_csv(output_dir / "sae_atom_stats.csv")
        channels = self.fit_meta.get("channels")
        if channels is not None:
            (output_dir / "sae_channels.txt").write_text("\n".join(channels) + "\n", encoding="utf-8")
        log.info(f"Saved local competition SAE to {output_dir}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load only a compatible local-competition checkpoint with subject provenance.

        Raises ``ValueError`` for a model/format/spec/channel mismatch. State keys
        and tensor shapes are loaded strictly; unrelated SAE checkpoints cannot
        silently initialize this model. Training settings need not match because
        loaded models skip fitting rather than resume an optimizer.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("model") != MODEL_TAG or checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: expected {MODEL_TAG} checkpoint format {CHECKPOINT_FORMAT}")
        if checkpoint.get("spec") != asdict(self.spec):
            raise ValueError(f"{path}: configured spec differs from the saved local competition spec")
        provenance = checkpoint.get("provenance") or {}
        if not provenance.get("train_subjects"):
            raise ValueError(f"{path}: checkpoint has no training-subject provenance")
        fit = checkpoint["fit"]
        n_channels = int(fit["n_channels"])
        if fit.get("channels") is not None and len(fit["channels"]) != n_channels:
            raise ValueError(f"{path}: channel metadata length differs from n_channels")
        if self.enc_shapes is None:
            self.initialize_dictionary(n_channels)
        elif self.enc_shapes.shape[1] != n_channels:
            raise ValueError(f"{path}: channel count differs from the initialized dictionary")
        state = checkpoint["state_dict"]
        self.to(dtype=state["enc_shapes"].dtype)
        self.load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.parameters()):
            raise ValueError(f"{path}: non-finite model parameters")
        self.fit_meta = fit
        self.history = checkpoint.get("history", [])
        self.atom_stats = checkpoint.get("atom_stats", {})
        self.provenance = provenance
        self.fitted = True
        self.eval()
