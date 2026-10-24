"""Waveform SAE: one maximum per pooling window, then top-R windows per atom.

Implements ``documentation/Minimal Waveform Autoencoder Mathematics.md`` inside the
project pipeline (the equation numbers below refer to that note); the design choices,
the workflow and the tweak points are in ``documentation/waveform_sae_implementation.md``.

Each atom is one multichannel waveform ``(C, L)``. The encoder correlates the atoms
across the full segment, with zero padding only at its ends, sums channel responses,
applies ReLU, and takes one maximum in each window of ``pool_len`` samples.
Each atom keeps only its ``top_r``
strongest windows in the complete segment, at the original positions and amplitudes.
The decoder is the tied transposed convolution within each pooling window. Waveform
tails are cropped at every window boundary; adjacent windows contribute nothing. The
training minimizes the reconstruction error alone: the sparsity comes from the
selection, not from a threshold or an L1 penalty. One activation has one time and one
coefficient, shared by all channel components of its atom.

This is a second arm next to ``shapeconv_sae.py`` (single-channel atoms, signed code,
learnable thresholds with an L1 penalty). The default preserves the input waveform
without first differencing; per-window/channel robust scaling and clipping use the
shared row helpers. ``forward`` returns retained amplitude sums divided by the
number of pooling windows; these are not event counts. Training scores
complete input segments by default, without an energy gate. Legacy crop modes
recompute full-segment selection with the current atoms before every update.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.components.hydra_transform import HydraTransform
from src.models.components.shapeconv_sae import CHECKPOINT_NAME, ROBUST_CLIP, ShapeConvSAE

__all__ = ["CHECKPOINT_NAME", "MODEL_TAG", "WaveformSAE", "WaveformSpec", "WaveformTrainSpec"]

_EPS = 1e-8
_ENERGY_FLOOR = 1e-6
_PRE_EMPHASIS = ("none", "diff")
_ATOM_NORMS = ("unit", "free")
_INITS = ("patches", "random")
_SAMPLINGS = ("full", "random", "tiled")
_INIT_POOL_FACTOR = 8  # candidate patches drawn per atom for the data-patch initialization
MODEL_TAG = "waveform_sae"
CHECKPOINT_FORMAT = 4


@dataclass(frozen=True)
class WaveformSpec:
    """Dictionary geometry and encoder, shared by training and inference.

    Every field defines the code, so a checkpoint is refused unless the configured
    spec equals the saved one.

    Parameters
    ----------
    n_atoms : int
        Number of multichannel atoms ``K``.
    atom_len : int
        Atom length ``L`` in samples (80 = 312 ms at 256 Hz). Even lengths use the
        asymmetric same padding of equation (4).
    pool_len : int
        Pooling-window length ``P`` in samples. The windows are disjoint and start at
        sample 0 of the encoded row. Each window supplies one candidate and its
        original position. Encoder receptive fields cross pooling boundaries.
        Only decoder placements are truncated to their own pooling window,
        even when ``atom_len > pool_len``.
    top_r : int
        Windows kept per atom and complete encoded segment, ``1 <= R <= T / P``.
        The upper bound is checked against the input length at encoding time.
        ``R = T / P`` retains every window maximum. Format-1 checkpoints used a
        different meaning; formats 2 and 3 used different convolution boundaries.
        These formats must not be loaded with this model.
    pre_emphasis : str, default="none"
        ``diff``: encode the first difference of every row (the same pre-emphasis as
        the ShapeConv SAE; the first encoded sample is zero so the length is kept).
        ``none``: encode the scaled row as is. Atoms live in the encoded domain.
    atom_norm : str
        ``unit``: every atom has unit Frobenius norm over ``(C, L)``, re-imposed after
        each optimizer step (equation 17). ``free``: unit norm at initialization only,
        then the reconstruction loss sets the scale (the prototype's behaviour).
    center_atoms : bool
        Subtract the temporal mean of every channel component before the norm
        (equation 18).
    """

    n_atoms: int = 64
    atom_len: int = 80
    pool_len: int = 32
    top_r: int = 1
    pre_emphasis: str = "none"
    atom_norm: str = "unit"
    center_atoms: bool = False


@dataclass(frozen=True)
class WaveformTrainSpec:
    """Unsupervised training schedule of the dictionary.

    Parameters
    ----------
    epochs : int
        Passes over patients (patient sampling) or segments (segment sampling).
        Patient mode redraws segment subsets on every visit. Zero keeps the
        initialization (a frozen-initialization control).
    lr : float
        Adam learning rate. Atom entries have a typical size of ``1 / sqrt(C * L)``
        (0.025 for 20 channels and 80 samples), so the rate is lower than the
        ShapeConv SAE's.
    crop_len : int
        Legacy crop modes only: scored samples per crop, a multiple of ``pool_len``.
    crop_sampling : str
        ``full`` (default): score complete input segments without cropping or an
        energy gate. Input lengths must be divisible by ``pool_len``.
        ``random``: ``crops_per_window`` crops per window at random grid positions,
        with replacement, so crops can overlap; samples near segment edges have
        fewer possible covering crops. ``tiled``: every
        segment is cut into the consecutive crops that fit, so
        each epoch scores every sample of the tiled span exactly once; the tiling
        starts at a random multiple of ``pool_len`` within the slack that the tiling
        leaves, drawn once per loaded batch, so the few samples outside one epoch's
        tiling are inside another's.
    crops_per_window : int
        ``random`` sampling only: crops drawn per window on every pass, in
        expectation (windows are drawn uniformly within each batch).
    crop_batch : int
        Legacy crop modes only: crops per optimizer step.
    segment_batch : int
        Full mode: segments per GPU gradient chunk. Under segment sampling each
        chunk is an optimizer step; patient sampling accumulates all chunks first.
    sampling_unit : str
        ``patient``: use complete lazy manifests, redraw segments per patient and
        average within patients, then patients. The waveform configuration selects
        this mode. ``segment`` preserves the legacy Python API/control behavior.
    patients_per_batch : int
        Distinct patients per optimizer step under patient sampling.
    segments_per_patient : int
        Maximum segments drawn without replacement per patient per epoch. This is
        a draw size, not a manifest cap; all segments remain available next epoch.
    max_crop_energy_ratio : float | None
        Legacy crop modes only; must be ``None`` in full mode. A crop (and an initialization patch) whose
        scored energy exceeds this multiple of the median energy of its batch is
        excluded. ``None`` keeps every non-flat crop. Inference is unchanged.
    init : str
        ``patches``: atoms start as random multichannel patches of the training
        segments with zero padding only at segment ends (non-flat patches only;
        no energy-ratio gate in full mode). ``random``: Gaussian atoms (the
        prototype's initialization).
    recenter_atoms : bool
        After every epoch but the last, shift each atom along time so that its
        energy centroid sits at the center of its support (whole samples, zero
        fill; the Adam moments move with it). The pooling grid and truncation mean
        this need not preserve the encoded reconstruction. Motivation (synthetic recovery,
        2026-10-06): a waveform that lands over an edge of the atom at the start
        stays cut off, because the missing part lies outside the support at every
        selected placement and receives no gradient; the fit then stops at two to
        three times the residual of the planted solution, for atoms up to twice
        the waveform length. Re-centering brings the missing part inside.
    full_window_val : bool
        Legacy crop modes only: also measure validation on whole windows (every sample,
        not the crop centers) in the same pass over the validation loader: the
        ``val_full_*`` history columns, and the per-atom table from whole windows.
        No extra reading; the windows are encoded in chunks of ``chunk_windows``.
        Full mode always measures whole segments, with ``val_*`` and ``val_full_*``
        as aliases of the same statistics.
    """

    epochs: int = 30
    lr: float = 1e-3
    crop_len: int = 1024
    crop_sampling: str = "full"
    crops_per_window: int = 16
    crop_batch: int = 64
    max_crop_energy_ratio: float | None = None
    init: str = "patches"
    recenter_atoms: bool = True
    full_window_val: bool = True
    segment_batch: int = 2
    sampling_unit: str = "segment"
    patients_per_batch: int = 8
    segments_per_patient: int = 4


@dataclass(frozen=True)
class _CropBatch:
    """Encoded source segments and a gated crop plan, with source identity preserved.

    ``window`` and ``start`` identify each crop in ``rows``; starts are sample
    offsets of the scored region on the pooling grid. Encoder context is gathered
    around that region. ``valid`` indexes source segments, not crops.
    ``energy`` contains all non-flat sampled crop energies before gating.
    """

    rows: torch.Tensor
    valid: torch.Tensor
    window: torch.Tensor
    start: torch.Tensor
    energy: torch.Tensor
    dropped: int
    clipped: torch.Tensor


def _shuffled(dataloader: DataLoader, generator: torch.Generator) -> DataLoader:
    """A loader over the same dataset that reshuffles the windows every pass (the given loader is untouched)."""
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )


class WaveformSAE(nn.Module):
    """Multichannel waveform dictionary with temporal top-R selection and a tied decoder.

    Exposes the hooks the pipeline calls on a learned extractor: ``fit_unsupervised``
    (labels never enter reconstruction loss), ``save_artifacts``, ``validate_artifact`` and the
    ``fitted`` / ``history`` / ``provenance`` / ``fit_meta`` attributes, so
    ``src/train_sae.py feature=waveform_sae`` trains a dictionary. The number of
    channels is read from the data at fit time (or from the checkpoint), and the
    channel order must stay fixed: an atom assigns one waveform to each channel index.

    Parameters
    ----------
    spec : WaveformSpec | None
        Dictionary geometry and encoder; ``None`` uses the defaults.
    train_spec : WaveformTrainSpec | None
        Unsupervised training schedule; ``None`` uses the defaults.
    random_state : int
        Seed for atom initialization and batch order; also controls sampling in
        legacy crop modes (validation crops use ``random_state + 1``).
    device : str | None
        ``cpu`` | ``cuda`` | ``cuda:N`` | ``auto`` (cuda when available).
    pretrained : str | Path | None
        Path to a ``sae_state.pt`` checkpoint written by ``save_artifacts``.
    chunk_windows : int
        Complete segments encoded per pass for crop selection and whole-window
        validation. Response memory scales with ``chunk_windows * n_atoms * T``;
        selection sorts only the ``T / pool_len`` window maxima.

    Raises
    ------
    ValueError
        If a spec field has an unknown or out-of-range value, ``segment_batch`` is
        not positive, full mode enables the energy gate, or a legacy ``crop_len``
        is not a positive multiple of ``pool_len``.
    """

    def __init__(
        self,
        spec: WaveformSpec | None = None,
        train_spec: WaveformTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
        chunk_windows: int = 8,
    ) -> None:
        super().__init__()
        self.spec = spec or WaveformSpec()
        self.train_spec = train_spec or WaveformTrainSpec()
        self._check_specs()
        self.random_state = random_state
        self.chunk_windows = max(int(chunk_windows), 1)
        self.device = HydraTransform._resolve_device(device)
        self.register_parameter("atoms", None)
        self.history: list[dict[str, float]] = []
        self.provenance: dict[str, Any] = {}
        self.fit_meta: dict[str, Any] = {}
        self.atom_stats: dict[str, list[float]] = {}
        self.fitted = False
        if pretrained:
            self.load_checkpoint(pretrained)

    def _check_specs(self) -> None:
        """Refuse unknown option values and geometries the encoder cannot realize."""
        spec, train_spec = self.spec, self.train_spec
        if train_spec.sampling_unit not in ("segment", "patient"):
            raise ValueError("sampling_unit must be segment or patient")
        for name in ("patients_per_batch", "segments_per_patient"):
            value = getattr(train_spec, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if train_spec.sampling_unit == "patient" and (train_spec.crop_sampling != "full" or spec.top_r != 1):
            raise ValueError("patient sampling requires complete segments and top_r=1")
        for name, value, allowed in (
            ("spec.pre_emphasis", spec.pre_emphasis, _PRE_EMPHASIS),
            ("spec.atom_norm", spec.atom_norm, _ATOM_NORMS),
            ("train_spec.init", train_spec.init, _INITS),
            ("train_spec.crop_sampling", train_spec.crop_sampling, _SAMPLINGS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        geometry = (spec.n_atoms, spec.atom_len, spec.pool_len, spec.top_r)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in geometry):
            raise ValueError(f"need positive integer n_atoms, atom_len, pool_len and top_r, got {spec}")
        if (
            isinstance(train_spec.segment_batch, bool)
            or not isinstance(train_spec.segment_batch, int)
            or train_spec.segment_batch < 1
        ):
            raise ValueError("train_spec.segment_batch must be a positive integer")
        if train_spec.crop_sampling == "full":
            if train_spec.max_crop_energy_ratio is not None:
                raise ValueError("full-segment training requires train_spec.max_crop_energy_ratio=None")
        elif train_spec.crop_len < spec.pool_len or train_spec.crop_len % spec.pool_len:
            raise ValueError(
                f"train_spec.crop_len={train_spec.crop_len} must be a positive multiple of "
                f"spec.pool_len={spec.pool_len}"
            )

    # ------------------------------------------------------------------ checkpoint

    def validate_artifact(self) -> None:
        """Refuse a fitted dictionary without training-subject provenance; an unfitted module passes.

        Raises
        ------
        ValueError
            If the module is fitted but records no training subjects.
        """
        if self.fitted and not self.provenance.get("train_subjects"):
            raise ValueError("fitted WaveformSAE records no training subjects; refusing to reuse it")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load atoms, history, fit metadata, atom statistics and provenance from ``sae_state.pt``; mark fitted.

        Raises
        ------
        ValueError
            If the file is not a waveform SAE checkpoint of the current format (a
            ShapeConv SAE checkpoint carries no model tag), if any spec field differs
            from the configured spec, or if it records no training subjects.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if checkpoint.get("model") == MODEL_TAG and checkpoint.get("format") in (1, 2, 3):
            raise ValueError(
                f"{path}: legacy waveform format {checkpoint['format']} uses different selection or boundaries. "
                "Format 4 uses a full-segment encoder, a pooling-window decoder and top_r windows per "
                "segment; retrain or use the matching old code "
                "to evaluate the old checkpoint. Changing the format number is not a valid migration."
            )
        if checkpoint.get("model") != MODEL_TAG or checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path}: not a {MODEL_TAG} checkpoint of format {CHECKPOINT_FORMAT} "
                f"(model {checkpoint.get('model')!r}, format {checkpoint.get('format')!r})"
            )
        saved = checkpoint["spec"]
        differing = {k: (saved.get(k), v) for k, v in asdict(self.spec).items() if saved.get(k) != v}
        if differing:
            raise ValueError(
                f"{path}: spec differs from the checkpoint (saved, config): {differing}; "
                "set feature.spec to the checkpoint's values"
            )
        provenance = checkpoint.get("provenance") or {}
        if not provenance.get("train_subjects"):
            raise ValueError(f"{path} has no training-subject provenance; refusing to reuse it")
        self.atoms = nn.Parameter(checkpoint["atoms"].to(self.device))
        self.history = checkpoint["history"]
        self.provenance = provenance
        self.fit_meta = checkpoint["fit"]
        self.atom_stats = checkpoint.get("atom_stats") or {}
        self.fitted = True
        log.info(
            f"Loaded {self.spec.n_atoms} pretrained waveform atoms ({self.atoms.shape[1]} channels) from {path} "
            f"({len(self.history)} epochs, {len(provenance['train_subjects'])} training subjects)"
        )

    def save_artifacts(self, output_dir: Path) -> None:
        """Write ``sae_state.pt`` and the inspectable files.

        ``sae_atoms.npy`` (``(K, C, L)``, encoded domain), ``sae_atoms_signal.npy``
        (see ``atoms_signal_domain``), ``sae_training.csv`` (one row per epoch),
        ``sae_atom_stats.csv`` (one row per atom, from the last epoch) and
        ``sae_channels.txt`` (channel names in atom order, when the dataset exposes
        them). The checkpoint holds the model tag, the format number, the spec, the
        fit metadata, the atoms, the history, the atom statistics and the provenance.
        """
        output_dir = Path(output_dir)
        np.save(output_dir / "sae_atoms.npy", self.atoms_numpy)
        np.save(output_dir / "sae_atoms_signal.npy", self.atoms_signal_domain)
        torch.save(
            {
                "model": MODEL_TAG,
                "format": CHECKPOINT_FORMAT,
                "spec": asdict(self.spec),
                "fit": self.fit_meta,
                "atoms": self.atoms.detach().cpu(),
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
        if channels:
            (output_dir / "sae_channels.txt").write_text("\n".join(channels) + "\n")
        log.info(f"Saved {self.spec.n_atoms} waveform atoms and {len(self.history)} training epochs to {output_dir}")

    @property
    def train_subjects(self) -> list[str]:
        """Subjects whose windows trained the dictionary (empty before fitting)."""
        return list(self.provenance.get("train_subjects", []))

    @property
    def atoms_numpy(self) -> np.ndarray:
        """The constrained atoms as a ``(n_atoms, n_channels, atom_len)`` float32 array, in the encoded domain."""
        return self._dictionary().detach().cpu().numpy()

    @property
    def atoms_signal_domain(self) -> np.ndarray:
        """Synthesis shapes in the scaled signal domain: ``(n_atoms, n_channels, atom_len + 1)`` under ``diff``.

        The inverse of ``atom_len`` differences from a zero baseline is
        ``[0, cumsum(w)]`` per channel. The relative scale of the channels is kept
        (no re-normalization). Under ``none`` it is the atom itself.
        """
        atoms = self._dictionary().detach()
        if self.spec.pre_emphasis == "diff":
            atoms = F.pad(atoms.cumsum(-1), (1, 0))
        return atoms.cpu().numpy()

    # ------------------------------------------------------------------ primitives

    @property
    def padding(self) -> tuple[int, int]:
        """Left and right same-padding widths ``(p_L, p_R)`` of equation (4)."""
        left = (self.spec.atom_len - 1) // 2
        return left, self.spec.atom_len - 1 - left

    @property
    def margin(self) -> int:
        """Symmetric encoder context around a scored crop: ``max(p_L, p_R)`` samples.

        Scored starts remain on the pooling grid. Context can cross pooling
        boundaries and is zero only outside the full encoded source segment.
        Crop selection additionally needs the current full-segment window mask.
        """
        return max(self.padding)

    @staticmethod
    def _unit(atoms: torch.Tensor) -> torch.Tensor:
        """Unit Frobenius norm of every atom over ``(C, L)``; a zero atom stays zero."""
        return atoms / torch.linalg.vector_norm(atoms, dim=(1, 2), keepdim=True).clamp_min(_EPS)

    def _constrain(self, atoms: torch.Tensor) -> torch.Tensor:
        """The atom constraint: optional per-channel temporal centering, then unit norm when ``atom_norm="unit"``."""
        if self.spec.center_atoms:
            atoms = atoms - atoms.mean(-1, keepdim=True)
        return self._unit(atoms) if self.spec.atom_norm == "unit" else atoms

    def _dictionary(self) -> torch.Tensor:
        """The constrained atoms ``(K, C, L)`` that the encoder and the decoder both use.

        Raises
        ------
        RuntimeError
            Before the dictionary exists (no fit and no checkpoint).
        """
        if self.atoms is None:
            raise RuntimeError("WaveformSAE has no dictionary yet: call fit_unsupervised or pass pretrained")
        return self._constrain(self.atoms)

    def _rows(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoded rows ``(B, C, T')`` of a window batch ``(B, C, T)`` and the usable-channel mask ``(B, C)``.

        Every (window, channel) row goes through the ShapeConv SAE row pipeline:
        optional first difference, median centering, MAD scaling and the clip. Rows
        that ``ShapeConvSAE._valid_rows`` rejects (constant or non-finite) come back
        as zeros with a false mask. ``T'`` is ``T`` trimmed to a multiple of
        ``pool_len``.

        Raises
        ------
        ValueError
            If the batch is not 3-D, its channel count differs from the dictionary's,
            or the windows are shorter than one pooling window.
        """
        raw = X.to(self.device)
        if raw.ndim != 3:
            raise ValueError(f"expected windows of shape (B, C, T), got {tuple(raw.shape)}")
        n_windows, n_channels, _ = raw.shape
        if self.atoms is not None and n_channels != self.atoms.shape[1]:
            raise ValueError(f"windows have {n_channels} channels but the dictionary has {self.atoms.shape[1]}")
        flat = raw.flatten(0, 1)
        valid = ShapeConvSAE._valid_rows(flat)
        if self.spec.pre_emphasis == "diff":
            flat = flat.diff(dim=-1, prepend=flat[:, :1])
        rows = ShapeConvSAE._robust_scale(flat)
        rows = torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))
        n_times = rows.shape[-1] // self.spec.pool_len * self.spec.pool_len
        if n_times == 0:
            raise ValueError(f"windows of {rows.shape[-1]} samples are shorter than pool_len={self.spec.pool_len}")
        if self.spec.top_r > n_times // self.spec.pool_len:
            raise ValueError(
                f"top_r={self.spec.top_r} exceeds the {n_times // self.spec.pool_len} complete pooling "
                f"windows in a segment of {n_times} encoded samples"
            )
        return rows[:, :n_times].view(n_windows, n_channels, n_times), valid.view(n_windows, n_channels)

    def _pool_batch(self, rows: torch.Tensor) -> torch.Tensor:
        """Arrange ``(B, D, M*P)`` as independent windows ``(B*M, D, P)`` for convolution."""
        if rows.ndim != 3 or rows.shape[-1] == 0 or rows.shape[-1] % self.spec.pool_len:
            raise ValueError("expected a 3-D tensor with a positive length that is a multiple of pool_len")
        return rows.unflatten(-1, (-1, self.spec.pool_len)).transpose(1, 2).flatten(0, 1)

    def _responses(self, rows: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Full-segment correlations ``(B, K, T)``, padded only at segment ends (equations 5-6).

        Contextual crop inputs can have lengths off the pooling grid; slice their
        scored responses before selection so the source grid is preserved.
        """
        if rows.ndim != 3 or rows.shape[-1] == 0:
            raise ValueError("expected a 3-D tensor with a positive time length")
        return F.conv1d(F.pad(rows, self.padding), atoms)

    def _select_windows(self, candidates: torch.Tensor) -> torch.Tensor:
        """Boolean top-R mask along the window axis; equal scores prefer the earlier window."""
        n_pool = candidates.shape[-1]
        if not 1 <= self.spec.top_r <= n_pool:
            raise ValueError(f"top_r={self.spec.top_r} must be between 1 and the segment's {n_pool} windows")
        if self.spec.top_r == n_pool:
            return torch.ones_like(candidates, dtype=torch.bool)
        index = candidates.argsort(dim=-1, descending=True, stable=True)[..., : self.spec.top_r]
        return torch.zeros_like(candidates, dtype=torch.bool).scatter(-1, index, True)

    def _select(self, responses: torch.Tensor, selected_windows: torch.Tensor | None = None) -> torch.Tensor:
        """ReLU, one maximum per window, then top-R windows; scatter at the original positions.

        Ties prefer the earlier position within a window and the earlier window
        within a segment. A supplied boolean ``(..., M)`` mask fixes the window
        decisions from the complete source segment, permitting local crop
        recomputation without ranking its windows again. There is no competition
        across atoms or batch members.
        """
        if responses.shape[-1] == 0 or responses.shape[-1] % self.spec.pool_len:
            raise ValueError("response length must be a positive multiple of pool_len")
        rectified = F.relu(responses).unflatten(-1, (-1, self.spec.pool_len))
        values, positions = rectified.max(dim=-1)
        if selected_windows is None:
            selected_windows = self._select_windows(values)
        elif selected_windows.shape != values.shape or selected_windows.dtype != torch.bool:
            raise ValueError("selected_windows must be a boolean mask matching the window candidates")
        kept = values * selected_windows
        return torch.zeros_like(rectified).scatter(-1, positions.unsqueeze(-1), kept.unsqueeze(-1)).flatten(-2)

    def _decode(self, code: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Synthesize each pooling window and discard its tails before concatenating.

        The ``(B*M, K, P)`` code produces ``P+L-1`` samples per window. Retaining
        exactly ``[p_L:p_L+P]`` gives ``(B, C, T)`` without overlap-add, including
        for even atom lengths and atoms longer than the pooling window. This is
        equation (14). Weights are tied, but these window boundaries mean this
        is not the adjoint of the full-segment ``_responses``.
        """
        left = self.padding[0]
        windows = self._pool_batch(code)
        reconstruction = F.conv_transpose1d(windows, atoms)[..., left : left + self.spec.pool_len]
        return reconstruction.unflatten(0, (code.shape[0], -1)).transpose(1, 2).flatten(-2)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse code of a window batch and the encoded rows it explains.

        Parameters
        ----------
        X : torch.Tensor
            Window batch ``(B, C, T)``; moved to the module device internally.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The non-negative code ``(B, K, T')`` (at most ``top_r`` nonzero entries
            per atom and segment, at most one per pooling window, at their original positions) and the encoded
            rows ``(B, C, T')``. ``rows - decode(code)`` is the residual. Memory
            scales with ``B * K * T'``.
        """
        atoms = self._dictionary()
        rows, _ = self._rows(X)
        return self._select(self._responses(rows, atoms)), rows

    @torch.no_grad()
    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(B, C, T')`` from ``(B, K, T')``, discarding tails at each pooling boundary.

        The result is in the encoded-row domain. ``T'`` must contain a positive
        integer number of complete pooling windows, otherwise ``ValueError`` is
        raised. The dictionary must already exist, otherwise ``RuntimeError`` is
        raised.
        """
        return self._decode(code.to(self.device), self._dictionary())

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable segment features ``(B, K)``: retained amplitude sum divided by ``M``.

        ``y`` is ignored. Complete segments are required; zero usable channels give
        zero features. Patient training excludes those segments from both means.
        This amplitude feature is not an occurrence count under top-R selection.
        """
        rows, _ = self._full_rows(X)
        code = self._select(self._responses(rows, self._dictionary()))
        return code.sum(-1) / (rows.shape[-1] // self.spec.pool_len)

    # ------------------------------------------------------------------ training

    def _full_rows(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess complete segments, refusing partial pooling windows instead of trimming them."""
        rows, valid = self._rows(x)
        if rows.shape[-1] != x.shape[-1]:
            raise ValueError(
                f"full-segment scoring requires input length {x.shape[-1]} to be a multiple "
                f"of pool_len={self.spec.pool_len}; no samples may be trimmed"
            )
        return rows, valid

    def _segment_step(
        self, rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score every sample, with global top-R per atom/segment and a usable-channel denominator."""
        responses = self._responses(rows, atoms)
        candidates = F.relu(responses.unflatten(-1, (-1, self.spec.pool_len)).amax(-1))
        selected = self._select_windows(candidates)
        code = self._select(responses, selected)
        mask = valid.unsqueeze(-1).to(rows.dtype)
        reconstruction = self._decode(code, atoms) * mask
        loss = 0.5 * (rows - reconstruction).pow(2).sum() / (mask.sum() * rows.shape[-1]).clamp_min(1.0)
        with torch.no_grad():
            stats, per_atom = self._statistics(rows, reconstruction, code, selected)
        return loss, stats, per_atom

    def _center(self, t: torch.Tensor) -> torch.Tensor:
        """Remove encoder context from a crop input or response tensor (last axis)."""
        return t[..., self.margin : self.margin + self.train_spec.crop_len]

    def _gate(self, energy: torch.Tensor) -> torch.Tensor:
        """Keep mask by energy: non-flat and, with a gate, at most that multiple of the batch median."""
        keep = energy > _ENERGY_FLOOR
        ratio = self.train_spec.max_crop_energy_ratio
        if ratio is not None and bool(keep.any()):
            keep = keep & (energy <= float(ratio) * energy[keep].median())
        return keep

    def _crops(self, x: torch.Tensor, generator: torch.Generator) -> _CropBatch:
        """Plan legacy grid-aligned crops and retain their complete encoded source segments.

        A crop starts at a multiple of ``pool_len`` in the encoded row, so its pooling
        grid is the window's grid. With ``crop_sampling="tiled"`` the crops of a
        window are consecutive and their scored centers are disjoint; the tiling
        starts at a random grid offset within the slack it leaves, drawn once per
        batch. With ``"random"`` windows and starts are drawn uniformly with
        replacement, ``crops_per_window`` per window in expectation. Crops whose
        scored part is flat (failed loads) are dropped, and so are crops above the
        energy gate.

        Returns
        -------
        _CropBatch
            Encoded rows and validity masks, kept source indices and sample
            offsets, non-flat crop energies, number of gated non-flat crops,
            and ``[clipped samples, usable samples]``. Selection is not cached:
            every optimizer step ranks the source segments with current atoms.

        Raises
        ------
        ValueError
            If full-segment mode is active or the segments are shorter than the scored crop length.
        """
        if self.train_spec.crop_sampling == "full":
            raise ValueError("full-segment training does not construct crops")
        rows, valid = self._rows(x)
        total = self.train_spec.crop_len
        if rows.shape[-1] < total:
            raise ValueError(f"segments of {rows.shape[-1]} samples are shorter than crop_len={total}")
        n_windows, _, n_times = rows.shape
        if self.train_spec.crop_sampling == "tiled":
            crop_len = self.train_spec.crop_len
            n_tiles = (n_times - total) // crop_len + 1
            slack = n_times - total - (n_tiles - 1) * crop_len
            n_phases = slack // self.spec.pool_len + 1
            offset = int(torch.randint(0, n_phases, (1,), generator=generator)) * self.spec.pool_len
            window = torch.arange(n_windows, device=rows.device).repeat_interleave(n_tiles)
            start = (offset + torch.arange(n_tiles, device=rows.device) * crop_len).repeat(n_windows)
        else:
            views = rows.unfold(-1, total, self.spec.pool_len)
            n_crops = n_windows * self.train_spec.crops_per_window
            window = torch.randint(0, n_windows, (n_crops,), generator=generator).to(rows.device)
            start = torch.randint(0, views.shape[2], (n_crops,), generator=generator).to(rows.device)
            start = start * self.spec.pool_len
        crops = self._crop_inputs(rows, window, start)
        energy = self._center(crops).pow(2).sum(dim=(1, 2))
        keep = self._gate(energy)
        nonflat = energy > _ENERGY_FLOOR
        clipped = torch.stack([(rows.abs() >= ROBUST_CLIP).sum(), valid.sum() * rows.shape[-1]]).double()
        return _CropBatch(
            rows,
            valid,
            window[keep],
            start[keep],
            energy[nonflat],
            int(nonflat.sum()) - int(keep.sum()),
            clipped,
        )

    def _crop_inputs(self, rows: torch.Tensor, window: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Gather scored crops plus encoder context, zero-filling only beyond source ends.

        ``start`` locates the scored region, not the contextual input. Gathering
        only requested samples avoids padding or copying entire source batches.
        """
        offsets = torch.arange(-self.margin, self.train_spec.crop_len + self.margin, device=rows.device)
        positions = start[:, None] + offsets
        inside = (positions >= 0) & (positions < rows.shape[-1])
        crops = rows[window[:, None], :, positions.clamp(0, rows.shape[-1] - 1)].transpose(1, 2)
        return crops * inside.unsqueeze(1).to(rows.dtype)

    @torch.no_grad()
    def _crop_window_mask(
        self, rows: torch.Tensor, window: torch.Tensor, start: torch.Tensor, atoms: torch.Tensor
    ) -> torch.Tensor:
        """Recompute global window decisions, then slice them for each crop.

        Each distinct source segment is ranked once in this call, in bounded
        batches. The masks have no derivative on a fixed selection branch;
        ``_step`` recomputes retained local responses with gradients. Call again
        after every parameter update, including when reusing a sampled crop plan.
        """
        sources, inverse = window.unique(sorted=True, return_inverse=True)
        n_pool = rows.shape[-1] // self.spec.pool_len
        selected = torch.empty(sources.numel(), self.spec.n_atoms, n_pool, dtype=torch.bool, device=rows.device)
        for first in range(0, sources.numel(), self.chunk_windows):
            batch = slice(first, first + self.chunk_windows)
            responses = self._responses(rows[sources[batch]], atoms)
            candidates = F.relu(responses.unflatten(-1, (-1, self.spec.pool_len)).amax(-1))
            selected[batch] = self._select_windows(candidates)
        crop_pools = self.train_spec.crop_len // self.spec.pool_len
        return selected.unfold(-1, crop_pools, 1)[inverse, :, start // self.spec.pool_len]

    def _step(
        self, batch: _CropBatch, index: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruction loss of one crop batch on its scored part (equation 28), with summed statistics.

        Unusable channels have zero input, so they add nothing to the channel-summed
        response; their reconstruction is masked out of the loss.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The loss, then the two statistics tensors of ``_statistics`` on the
            scored part.
        """
        window, start = batch.window[index], batch.start[index]
        crops = self._crop_inputs(batch.rows, window, start)
        selected = self._crop_window_mask(batch.rows, window, start, atoms)
        responses = self._center(self._responses(crops, atoms))
        code = self._select(responses, selected)
        mask = batch.valid[window].unsqueeze(-1).to(crops.dtype)
        target = self._center(crops)
        reconstruction = self._decode(code, atoms) * mask
        loss = 0.5 * (target - reconstruction).pow(2).sum() / (mask.sum() * target.shape[-1]).clamp_min(1.0)
        with torch.no_grad():
            stats, per_atom = self._statistics(target, reconstruction, code, selected)
        return loss, stats, per_atom

    def _statistics(
        self,
        target: torch.Tensor,
        reconstruction: torch.Tensor,
        code: torch.Tensor,
        selected_windows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed statistics of scored rows ``(n, C, S)``, their masked reconstruction and their code ``(n, K, S)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``[residual energy, signal energy, <signal, reconstruction>, reconstruction
            energy, positive code entries, items, pooling windows per atom]`` and per
            atom ``(4, K)`` the sum of squared coefficients, the sum of coefficients,
            the number of positive coefficients, and the number of selected window
            slots in the scored region. Both float64. Rejected windows are not
            counted as selected zero slots.
        """
        positive = (code > 0).to(target.dtype)
        n_items, scored_len = target.shape[0], target.shape[-1]
        stats = torch.stack(
            [
                (target - reconstruction).pow(2).sum(),
                target.pow(2).sum(),
                (target * reconstruction).sum(),
                reconstruction.pow(2).sum(),
                positive.sum(),
                target.new_tensor(float(n_items)),
                target.new_tensor(float(n_items * (scored_len // self.spec.pool_len))),
            ]
        ).double()
        per_atom = torch.stack(
            [
                code.pow(2).sum(dim=(0, 2)),
                code.sum(dim=(0, 2)),
                positive.sum(dim=(0, 2)),
                selected_windows.sum(dim=(0, 2)),
            ]
        ).double()
        return stats, per_atom

    @torch.no_grad()
    def _window_stats(
        self, rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``_statistics`` on whole encoded rows ``(B, C, T')``, every sample scored, in chunks of ``chunk_windows``."""
        totals = torch.zeros(7, dtype=torch.float64, device=self.device)
        per_atom = torch.zeros(4, self.spec.n_atoms, dtype=torch.float64, device=self.device)
        for start in range(0, rows.shape[0], self.chunk_windows):
            chunk, mask = rows[start : start + self.chunk_windows], valid[start : start + self.chunk_windows]
            _, stats, atom_sums = self._segment_step(chunk, mask, atoms)
            totals += stats
            per_atom += atom_sums
        return totals, per_atom

    def _sample_patches(self, dataloader: DataLoader, generator: torch.Generator) -> torch.Tensor:
        """Sample ``(K, C, L)`` encoder receptive fields across segments, then gate."""
        spec = self.spec
        per_batch = max(math.ceil(_INIT_POOL_FACTOR * spec.n_atoms / max(len(dataloader), 1)), 1)
        pool = []
        for x, _ in tqdm(dataloader, desc="Waveform SAE patch initialization"):
            rows, _ = self._rows(x)
            views = F.pad(rows, self.padding).unfold(-1, spec.atom_len, 1)
            window = torch.randint(0, views.shape[0], (per_batch,), generator=generator).to(rows.device)
            start = torch.randint(0, views.shape[2], (per_batch,), generator=generator).to(rows.device)
            patches = views[window, :, start]
            pool.append(patches[self._gate(patches.pow(2).sum(dim=(1, 2)))])
        candidates = torch.cat(pool)
        if candidates.shape[0] < spec.n_atoms:
            raise ValueError(
                f"patch initialization needs at least n_atoms={spec.n_atoms} non-flat patches, "
                f"got {candidates.shape[0]}; supply more windows"
            )
        chosen = torch.randperm(candidates.shape[0], generator=generator)[: spec.n_atoms].to(candidates.device)
        return candidates[chosen]

    def _init_atoms(self, dataloader: DataLoader, generator: torch.Generator) -> None:
        """Create the dictionary: data patches or Gaussian atoms, centered when configured, unit norm."""
        if self.train_spec.init == "random":
            n_channels = int(dataloader.dataset[0][0].shape[0])
            atoms = torch.randn(self.spec.n_atoms, n_channels, self.spec.atom_len, generator=generator)
        else:
            atoms = self._sample_patches(dataloader, generator)
        self.atoms = nn.Parameter(self._unit(self._constrain(atoms.to(self.device))))
        log.info(
            f"Initialized {self.spec.n_atoms} waveform atoms ({self.atoms.shape[1]} channels, {self.train_spec.init})"
        )

    @torch.no_grad()
    def _evaluate(
        self, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Full-segment statistics by default; legacy modes score fixed crops and, with
        ``full_window_val``, on its whole windows, in one pass over the loader.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]
            Crop totals and per-atom sums, the whole-window totals and per-atom sums
            (``None`` when ``full_window_val`` is off), and the clip counts
            ``[clipped samples, usable samples]`` of the loader's rows.
        """
        generator = torch.Generator().manual_seed(self.random_state + 1)
        atoms = self._dictionary()
        full_segments = self.train_spec.crop_sampling == "full"
        full = full_segments or self.train_spec.full_window_val
        totals = torch.zeros(7, dtype=torch.float64, device=self.device)
        per_atom = torch.zeros(4, self.spec.n_atoms, dtype=torch.float64, device=self.device)
        full_totals = torch.zeros_like(totals) if full else None
        full_per_atom = torch.zeros_like(per_atom) if full else None
        clip = torch.zeros(2, dtype=torch.float64, device=self.device)
        for x, _ in dataloader:
            if full_segments:
                rows, valid = self._full_rows(x)
                clip += torch.stack([(rows.abs() >= ROBUST_CLIP).sum(), valid.sum() * rows.shape[-1]]).double()
                usable = valid.any(dim=1)
                stats, atom_sums = self._window_stats(rows[usable], valid[usable], atoms)
                totals += stats
                per_atom += atom_sums
                continue
            crops = self._crops(x, generator)
            clip += crops.clipped
            for start in range(0, crops.window.numel(), self.train_spec.crop_batch):
                index = torch.arange(
                    start, min(start + self.train_spec.crop_batch, crops.window.numel()), device=self.device
                )
                _, stats, atom_sums = self._step(crops, index, atoms)
                totals += stats
                per_atom += atom_sums
            if full_totals is not None and full_per_atom is not None:
                stats, atom_sums = self._window_stats(crops.rows, crops.valid, atoms)
                full_totals += stats
                full_per_atom += atom_sums
        if full_segments:
            if not bool(totals[5] > 0):
                raise ValueError("full-segment validation has no usable segments")
            return totals, per_atom, totals, per_atom, clip
        return totals, per_atom, full_totals, full_per_atom, clip

    @staticmethod
    def _reconstruction_summary(totals: torch.Tensor | None, prefix: str = "") -> dict[str, float]:
        """Residual fraction, best scalar gain and residual fraction at that gain from summed statistics.

        The gain ``g = <x, x_hat> / <x_hat, x_hat>`` minimizes ``||x - g x_hat||^2``;
        a value far from 1 means the tied unit-norm model mis-scales its
        reconstruction (overlapping placements). ``None`` gives NaNs.
        """
        keys = (f"{prefix}residual_frac", f"{prefix}gain", f"{prefix}residual_frac_at_gain")
        if totals is None:
            return dict.fromkeys(keys, float("nan"))
        residual, signal, cross, recon = (float(v) for v in totals[:4])
        at_gain = 1.0 - cross * cross / max(recon * signal, _EPS)
        return dict(zip(keys, (residual / max(signal, _EPS), cross / max(recon, _EPS), at_gain)))

    def _atom_table(self, totals: torch.Tensor, per_atom: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-atom usage on the scored samples behind ``totals`` and ``per_atom``, each ``(K,)``.

        ``norm``; ``energy_share`` (share of ``sum a^2 ||w||^2``, the energy of an
        atom's full placements with truncation and overlaps ignored); ``mean_amplitude`` (mean positive
        coefficient); ``positive_frac`` (share of selected slots in the scored
        region that hold a positive coefficient; zero when no slots were scored);
        ``active_window_frac`` (positive coefficients per scored pooling window).
        """
        norms = torch.linalg.vector_norm(self._dictionary().detach(), dim=(1, 2)).double()
        energy = per_atom[0] * norms.pow(2)
        slots = per_atom[3].clamp_min(1.0)
        return {
            "norm": norms,
            "energy_share": energy / energy.sum().clamp_min(_EPS),
            "mean_amplitude": per_atom[1] / per_atom[2].clamp_min(1.0),
            "positive_frac": per_atom[2] / slots,
            "active_window_frac": per_atom[2] / totals[6].clamp_min(1.0),
        }

    @staticmethod
    def _shifted(t: torch.Tensor, shift: int) -> torch.Tensor:
        """``t`` moved by ``shift`` samples along the last axis (positive = later), zero filled."""
        out = torch.zeros_like(t)
        if shift > 0:
            out[..., shift:] = t[..., :-shift]
        elif shift < 0:
            out[..., :shift] = t[..., -shift:]
        else:
            out.copy_(t)
        return out

    @torch.no_grad()
    def _recenter(self, optimizer: torch.optim.Optimizer) -> int:
        """Shift every atom whose energy centroid is at least one sample off center; returns the number moved.

        The shift is the whole-sample part of the offset, so an atom is never
        moved by less than one sample. The Adam moments of the atom move with it,
        and the constraint is re-imposed afterwards (the zero fill lowers the
        norm of a shifted atom).
        """
        atoms = self.atoms
        energy = atoms.pow(2).sum(1)
        positions = torch.arange(atoms.shape[-1], device=atoms.device, dtype=atoms.dtype)
        centroid = (energy * positions).sum(-1) / energy.sum(-1).clamp_min(_EPS)
        shifts = ((atoms.shape[-1] - 1) / 2 - centroid).trunc().long()
        moved = torch.nonzero(shifts).flatten().tolist()
        state = optimizer.state.get(atoms) or {}
        for k in moved:
            shift = int(shifts[k])
            atoms[k] = self._shifted(atoms[k], shift)
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][k] = self._shifted(state[key][k], shift)
        if moved:
            atoms.copy_(self._constrain(atoms))
        return len(moved)

    def _median_coherence(self) -> float:
        """Median over atom pairs of the maximum absolute normalized cross-correlation over lags."""
        atoms = self._unit(self._dictionary().detach())
        if atoms.shape[0] < 2:
            return 0.0
        coherence = F.conv1d(atoms, atoms, padding=atoms.shape[-1] - 1).abs().amax(-1)
        off_diagonal = ~torch.eye(atoms.shape[0], dtype=torch.bool, device=atoms.device)
        return float(coherence[off_diagonal].median())

    def fit_unsupervised(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Learn atoms by reconstruction of complete input segments by default.

        With ``sampling_unit='patient'``, delegate to the patient-first loop using
        dataset.plan. That loop averages per-segment usable-sample losses within
        patients, then patients, and restores minimum patient validation loss.
        The remaining description applies to the segment-sampling control.

        A seeded copy of the loader reshuffles segments each pass. Full mode
        scores each segment with usable channels once, in groups of at most
        ``segment_batch``, without an energy gate. Both encoding and top-R
        selection use each complete segment; synthesis remains window-confined.
        Legacy random/tiled crop modes remain available for reproduction.
        Labels are ignored. The loss is differentiated through the atom constraint, and the
        constraint is re-imposed on the parameter after every step. After every
        epoch but the last, with ``recenter_atoms``, atoms whose energy centroid is
        off center are shifted (validation is measured before). ``history``
        records, per epoch: ``objective``; ``residual_frac``, ``gain`` and
        ``residual_frac_at_gain`` on the training segments (or legacy crops) and, with the ``val_``
        prefix and ``val_full_`` prefix on the same complete validation segments
        in full mode (NaN without a validation loader). In legacy modes, ``val_``
        describes fixed crops and ``val_full_`` the optional whole-segment pass;
        ``selected_zero_frac`` (share of top-R slots that hold a zero);
        ``n_effective_atoms`` (participation ratio of the energy shares) and
        ``max_energy_share``; ``median_coherence``; ``atom_norm_min`` and
        ``atom_norm_max``; full mode records ``segments_seen``, ``segments_skipped``,
        and ``scored_samples`` (usable channel-samples). Legacy modes record
        ``crops_dropped_frac`` and ``crop_energy_mean_over_median``;
        ``clipped_frac`` and ``val_clipped_frac`` (share of usable encoded samples at
        the clip); ``n_recentered``; ``n_steps``. ``atom_stats`` keeps the per-atom table of the last epoch
        (whole validation segments, else legacy validation crops, else training data). A module loaded from a checkpoint
        (``fitted``) returns at once and keeps its provenance.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(X, y)`` with ``X`` of shape ``(B, C, T)``.
        val_dataloader : DataLoader | None
            Held-out windows scored after every epoch (monitoring only).
        provenance : dict | None
            Recorded with the checkpoint: must contain ``train_subjects`` (see
            ``src.utils.split_provenance``).

        Raises
        ------
        ValueError
            If ``provenance`` names no training subjects, full mode would trim
            input samples or an epoch has no usable training segments, or legacy
            windows are shorter than the scored crop length.
        """
        if self.fitted:
            log.info("Waveform SAE already fitted (pretrained checkpoint); skipping fit_unsupervised")
            return
        provenance = dict(provenance or {})
        if not provenance.get("train_subjects"):
            raise ValueError("fit_unsupervised needs provenance['train_subjects'] (the subjects behind the loader)")
        if self.train_spec.sampling_unit == "patient":
            from src.models.components.patient_waveform import fit_patient_reconstruction

            fit_patient_reconstruction(self, dataloader, val_dataloader, provenance)
            return
        spec, train_spec = self.spec, self.train_spec
        full_segments = train_spec.crop_sampling == "full"
        generator = torch.Generator().manual_seed(self.random_state)
        loader = _shuffled(dataloader, generator)
        self._init_atoms(loader, generator)
        optimizer = torch.optim.Adam([self.atoms], lr=train_spec.lr)
        self.history = []
        last: tuple[torch.Tensor, torch.Tensor] | None = None
        stats_split = "train"
        for epoch in range(train_spec.epochs):
            totals = torch.zeros(7, dtype=torch.float64, device=self.device)
            per_atom = torch.zeros(4, spec.n_atoms, dtype=torch.float64, device=self.device)
            loss_sum, n_steps, n_dropped = 0.0, 0, 0
            segments_seen, segments_skipped, scored_samples = 0, 0, 0
            clip = torch.zeros(2, dtype=torch.float64, device=self.device)
            energies = [torch.zeros(0, device=self.device)]
            for x, _ in tqdm(loader, desc=f"Waveform SAE epoch {epoch + 1}/{train_spec.epochs}"):
                if full_segments:
                    rows, valid = self._full_rows(x)
                    clip += torch.stack([(rows.abs() >= ROBUST_CLIP).sum(), valid.sum() * rows.shape[-1]]).double()
                    order = torch.nonzero(valid.any(dim=1)).flatten()
                    segments_seen += rows.shape[0]
                    segments_skipped += rows.shape[0] - order.numel()
                    step_batch = train_spec.segment_batch
                else:
                    crops = self._crops(x, generator)
                    energies.append(crops.energy)
                    n_dropped += crops.dropped
                    clip += crops.clipped
                    order = torch.randperm(crops.window.numel(), generator=generator).to(self.device)
                    step_batch = train_spec.crop_batch
                for start in range(0, order.numel(), step_batch):
                    index = order[start : start + step_batch]
                    if full_segments:
                        loss, stats, atom_sums = self._segment_step(rows[index], valid[index], self._dictionary())
                        scored_samples += int(valid[index].sum()) * rows.shape[-1]
                    else:
                        loss, stats, atom_sums = self._step(crops, index, self._dictionary())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        self.atoms.copy_(self._constrain(self.atoms))
                    totals += stats
                    per_atom += atom_sums
                    loss_sum += float(loss.detach())
                    n_steps += 1
            if full_segments and n_steps == 0:
                raise ValueError("full-segment training has no usable segments in this epoch")
            crop_val = full_val = None
            val_clipped_frac = float("nan")
            if val_dataloader is not None:
                val = self._evaluate(val_dataloader)
                crop_val = (val[0], val[1])
                full_val = None if val[2] is None or val[3] is None else (val[2], val[3])
                val_clipped_frac = float(val[4][0] / val[4][1].clamp_min(1.0))
            last = full_val or crop_val or (totals, per_atom)
            stats_split = "val_full" if full_val else ("val" if crop_val else "train")
            table = self._atom_table(totals, per_atom)
            share = table["energy_share"]
            coherence = self._median_coherence()
            # Measured on the trained state of the epoch; the re-centering (never after the
            # last epoch) only prepares the next one.
            final = epoch + 1 == train_spec.epochs
            n_recentered = self._recenter(optimizer) if train_spec.recenter_atoms and not final else 0
            all_energy = torch.cat(energies)
            n_nonflat = int(all_energy.numel())
            record = {
                "epoch": epoch + 1,
                "objective": 0.5 * float(totals[0]) / scored_samples if full_segments else loss_sum / max(n_steps, 1),
                **self._reconstruction_summary(totals),
                **self._reconstruction_summary(None if crop_val is None else crop_val[0], prefix="val_"),
                **self._reconstruction_summary(None if full_val is None else full_val[0], prefix="val_full_"),
                "selected_zero_frac": float((per_atom[3] - per_atom[2]).sum() / per_atom[3].sum().clamp_min(1.0)),
                "active_window_frac": float(table["active_window_frac"].mean()),
                "n_effective_atoms": float(1.0 / share.pow(2).sum()) if float(share.sum()) > 0 else 0.0,
                "max_energy_share": float(share.max()),
                "median_coherence": coherence,
                "atom_norm_min": float(table["norm"].min()),
                "atom_norm_max": float(table["norm"].max()),
                **(
                    {
                        "segments_seen": segments_seen,
                        "segments_skipped": segments_skipped,
                        "scored_samples": scored_samples,
                    }
                    if full_segments
                    else {
                        "crops_dropped_frac": n_dropped / max(n_nonflat, 1),
                        "crop_energy_mean_over_median": (
                            float(all_energy.mean() / all_energy.median().clamp_min(_EPS))
                            if n_nonflat
                            else float("nan")
                        ),
                    }
                ),
                "clipped_frac": float(clip[0] / clip[1].clamp_min(1.0)),
                "val_clipped_frac": val_clipped_frac,
                "n_recentered": n_recentered,
                "n_steps": n_steps,
            }
            self.history.append(record)
            log.info(
                f"Waveform SAE epoch {epoch + 1}/{train_spec.epochs}: residual {record['residual_frac']:.1%} of "
                f"signal power (val {'segments' if full_segments else 'crops'} {record['val_residual_frac']:.1%}, "
                "val full "
                f"{record['val_full_residual_frac']:.1%}), gain {record['gain']:.2f}, "
                f"effective atoms {record['n_effective_atoms']:.1f}/{spec.n_atoms}, zero slots "
                f"{record['selected_zero_frac']:.1%}, median coherence {record['median_coherence']:.2f}, "
                f"clipped {record['clipped_frac']:.3%}, "
                f"re-centered {n_recentered}, steps {n_steps}"
            )
        if last is not None:
            self.atom_stats = {
                "atom": list(range(spec.n_atoms)),
                **{k: v.tolist() for k, v in self._atom_table(*last).items()},
            }
        channels = getattr(dataloader.dataset, "target_channels", None)
        self.fit_meta = {
            "train_spec": asdict(train_spec),
            "random_state": self.random_state,
            "margin": self.margin,
            "scoring_scope": "full_encoded_segment" if full_segments else "crop_centers",
            "selection_scope": "full_encoded_segment",
            "encoder_scope": "full_encoded_segment",
            "decoder_scope": "pooling_window",
            "n_channels": int(self.atoms.shape[1]),
            "channels": None if channels is None else [str(c) for c in channels],
            "atom_stats_split": None if last is None else stats_split,
        }
        self.provenance = provenance
        self.fitted = True
