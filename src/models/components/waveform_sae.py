"""Waveform SAE: one maximum per pooling window, then top-R windows per atom.

Implements ``documentation/Minimal Waveform Autoencoder Mathematics.md`` inside the
project pipeline (the equation numbers below refer to that note); the design choices,
the workflow and the tweak points are in ``documentation/waveform_sae_implementation.md``.

Each atom is one multichannel waveform ``(C, L)``. The encoder correlates the atoms
with a segment, sums the channel responses, applies ReLU, and takes one maximum in
each pooling window of ``pool_len`` samples. Each atom keeps only its ``top_r``
strongest windows in the complete segment, at the original positions and amplitudes.
The decoder is the tied transposed convolution, and
training minimizes the reconstruction error alone: the sparsity comes from the
selection, not from a threshold or an L1 penalty. One activation has one time and one
coefficient, shared by all channel components of its atom.

This is a second arm next to ``shapeconv_sae.py`` (single-channel atoms, signed code,
learnable thresholds with an L1 penalty). The two arms share the row pipeline (first
difference, robust scale per window and channel, clip), so they differ in the model
only. ``forward`` (classifier features) is not implemented: counts saturate at
``top_r`` per segment when enough windows have positive candidates. Crop training
recomputes full-segment selection with the current atoms before every update.
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
_SAMPLINGS = ("random", "tiled")
_INIT_POOL_FACTOR = 8  # candidate patches drawn per atom for the data-patch initialization
MODEL_TAG = "waveform_sae"
CHECKPOINT_FORMAT = 2


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
        original position; adjacent retained placements may overlap.
    top_r : int
        Windows kept per atom and complete encoded segment, ``1 <= R <= T / P``.
        The upper bound is checked against the input length at encoding time.
        ``R = T / P`` retains every window maximum. Format-1 checkpoints used a
        different meaning and must not be loaded with this encoder.
    pre_emphasis : str
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
    pre_emphasis: str = "diff"
    atom_norm: str = "unit"
    center_atoms: bool = False


@dataclass(frozen=True)
class WaveformTrainSpec:
    """Unsupervised training schedule of the dictionary.

    Parameters
    ----------
    epochs : int
        Passes over the training windows. Every pass draws fresh random crops, so the
        update count is ``epochs * windows * crops_per_window / crop_batch``. Zero
        keeps the initialization (a frozen-initialization control).
    lr : float
        Adam learning rate. Atom entries have a typical size of ``1 / sqrt(C * L)``
        (0.025 for 20 channels and 80 samples), so the rate is lower than the
        ShapeConv SAE's.
    crop_len : int
        Scored length of a training crop in samples; a multiple of ``pool_len``.
    crop_sampling : str
        ``random``: ``crops_per_window`` crops per window at random grid positions,
        with replacement, so crops can overlap (the ShapeConv SAE's scheme; every
        sample has the same chance of being scored in an epoch). ``tiled``: every
        window is cut into the consecutive crops that fit with their margins, so
        each epoch scores every sample of the tiled span exactly once; the tiling
        starts at a random multiple of ``pool_len`` within the slack that the tiling
        leaves, drawn once per loaded batch, so the few samples outside one epoch's
        tiling are inside another's.
    crops_per_window : int
        ``random`` sampling only: crops drawn per window on every pass, in
        expectation (windows are drawn uniformly within each batch).
    crop_batch : int
        Crops per optimizer step.
    max_crop_energy_ratio : float | None
        Training-crop artifact gate: a crop (and an initialization patch) whose
        scored energy exceeds this multiple of the median energy of its batch is
        excluded. ``None`` keeps every non-flat crop. Inference is unchanged.
    init : str
        ``patches``: atoms start as random multichannel patches of the training
        windows (through the energy gate). ``random``: Gaussian atoms (the
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
        Also measure the validation statistics on whole windows (every sample,
        not the crop centers) in the same pass over the validation loader: the
        ``val_full_*`` history columns, and the per-atom table from whole windows.
        No extra reading; the windows are encoded in chunks of ``chunk_windows``.
    """

    epochs: int = 30
    lr: float = 1e-3
    crop_len: int = 1024
    crop_sampling: str = "random"
    crops_per_window: int = 16
    crop_batch: int = 64
    max_crop_energy_ratio: float | None = 10.0
    init: str = "patches"
    recenter_atoms: bool = True
    full_window_val: bool = True


@dataclass(frozen=True)
class _CropBatch:
    """Encoded source segments and a gated crop plan, with source identity preserved.

    ``window`` and ``start`` identify each crop in ``rows``; starts are sample
    offsets on the pooling grid. ``valid`` indexes source segments, not crops.
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
    (labels are never read), ``save_artifacts``, ``validate_artifact`` and the
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
        Seed for the atom initialization, the batch order and the crop sampling (the
        validation crops use ``random_state + 1``).
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
        If a spec field has an unknown or out-of-range value, or ``crop_len`` is not
        a positive multiple of ``pool_len``.
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
        if train_spec.crop_len < spec.pool_len or train_spec.crop_len % spec.pool_len:
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
        if checkpoint.get("model") == MODEL_TAG and checkpoint.get("format") == 1:
            raise ValueError(
                f"{path}: format-1 waveform checkpoints select top_r positions inside every pooling window. "
                "Format 2 selects top_r windows per segment; retrain for this encoder or use the old code "
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
        """Unscored samples on each side of a training crop.

        ``L + P - 2`` (equation 27) rounded up to a multiple of ``pool_len``, so a
        crop that starts on the pooling grid keeps that grid. This margin gives
        matching center activations and reconstruction only when the crop also
        receives the current full-segment selection mask.
        """
        spec = self.spec
        return spec.pool_len * math.ceil((spec.atom_len + spec.pool_len - 2) / spec.pool_len)

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

    def _responses(self, rows: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Channel-summed correlations ``(n, K, T)`` of rows ``(n, C, T)`` with atoms ``(K, C, L)`` (equations 5-6)."""
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
        """Tied synthesis ``(n, C, T)`` of a code ``(n, K, T)``: equation (14), the adjoint of ``_responses``."""
        left = self.padding[0]
        return F.conv_transpose1d(code, atoms)[..., left : left + code.shape[-1]]

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
        """Reconstruction ``(B, C, T')`` of a code ``(B, K, T')`` in the encoded-row domain."""
        return self._decode(code.to(self.device), self._dictionary())

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Classifier features are not defined for this model yet.

        Raises
        ------
        NotImplementedError
            Always. Counts can saturate at ``top_r`` per atom and segment,
            so the feature blocks are an open decision; see
            ``documentation/waveform_sae_implementation.md``. Use ``encode``.
        """
        raise NotImplementedError(
            "WaveformSAE has no classifier features yet (activation counts can saturate under top-R "
            "selection); train dictionaries with src/train_sae.py and inspect them through encode()"
        )

    # ------------------------------------------------------------------ training

    def _center(self, t: torch.Tensor) -> torch.Tensor:
        """The scored part of a crop-shaped tensor (last axis)."""
        return t[..., self.margin : self.margin + self.train_spec.crop_len]

    def _gate(self, energy: torch.Tensor) -> torch.Tensor:
        """Keep mask by energy: non-flat and, with a gate, at most that multiple of the batch median."""
        keep = energy > _ENERGY_FLOOR
        ratio = self.train_spec.max_crop_energy_ratio
        if ratio is not None and bool(keep.any()):
            keep = keep & (energy <= float(ratio) * energy[keep].median())
        return keep

    def _crops(self, x: torch.Tensor, generator: torch.Generator) -> _CropBatch:
        """Plan grid-aligned crops and retain their complete encoded source segments.

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
            If the windows are shorter than a crop with its margins.
        """
        rows, valid = self._rows(x)
        total = self.train_spec.crop_len + 2 * self.margin
        if rows.shape[-1] < total:
            raise ValueError(f"windows of {rows.shape[-1]} samples are shorter than crop_len + 2 * margin = {total}")
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
            rows, valid, window[keep], start[keep], energy[nonflat],
            int(nonflat.sum()) - int(keep.sum()), clipped,
        )

    def _crop_inputs(self, rows: torch.Tensor, window: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        """Gather crop inputs from their source segments and grid-aligned sample offsets."""
        total = self.train_spec.crop_len + 2 * self.margin
        return rows.unfold(-1, total, self.spec.pool_len)[window, :, start // self.spec.pool_len]

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
        selected = torch.empty(
            sources.numel(), self.spec.n_atoms, n_pool, dtype=torch.bool, device=rows.device
        )
        for first in range(0, sources.numel(), self.chunk_windows):
            batch = slice(first, first + self.chunk_windows)
            responses = self._responses(rows[sources[batch]], atoms)
            candidates = F.relu(responses.unflatten(-1, (-1, self.spec.pool_len)).amax(-1))
            selected[batch] = self._select_windows(candidates)
        crop_pools = (self.train_spec.crop_len + 2 * self.margin) // self.spec.pool_len
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
        code = self._select(self._responses(crops, atoms), selected)
        mask = batch.valid[window].unsqueeze(-1).to(crops.dtype)
        target = self._center(crops)
        reconstruction = self._center(self._decode(code, atoms)) * mask
        loss = 0.5 * (target - reconstruction).pow(2).sum() / (mask.sum() * target.shape[-1]).clamp_min(1.0)
        with torch.no_grad():
            first = self.margin // self.spec.pool_len
            selected_center = selected[..., first : first + self.train_spec.crop_len // self.spec.pool_len]
            stats, per_atom = self._statistics(target, reconstruction, self._center(code), selected_center)
        return loss, stats, per_atom

    def _statistics(
        self, target: torch.Tensor, reconstruction: torch.Tensor, code: torch.Tensor,
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
            [code.pow(2).sum(dim=(0, 2)), code.sum(dim=(0, 2)),
             positive.sum(dim=(0, 2)), selected_windows.sum(dim=(0, 2))]
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
            responses = self._responses(chunk, atoms)
            selected = self._select_windows(F.relu(responses.unflatten(-1, (-1, self.spec.pool_len)).amax(-1)))
            code = self._select(responses, selected)
            reconstruction = self._decode(code, atoms) * mask.unsqueeze(-1).to(chunk.dtype)
            stats, atom_sums = self._statistics(chunk, reconstruction, code, selected)
            totals += stats
            per_atom += atom_sums
        return totals, per_atom

    def _sample_patches(self, dataloader: DataLoader, generator: torch.Generator) -> torch.Tensor:
        """``n_atoms`` random multichannel patches ``(K, C, L)`` drawn from every batch, through the energy gate."""
        spec = self.spec
        per_batch = max(math.ceil(_INIT_POOL_FACTOR * spec.n_atoms / max(len(dataloader), 1)), 1)
        pool = []
        for x, _ in tqdm(dataloader, desc="Waveform SAE patch initialization"):
            rows, _ = self._rows(x)
            views = rows.unfold(-1, spec.atom_len, 1)
            window = torch.randint(0, rows.shape[0], (per_batch,), generator=generator).to(rows.device)
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
        """Summed statistics on fixed crops of a loader (the same crops at every call) and, with
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
        full = self.train_spec.full_window_val
        totals = torch.zeros(7, dtype=torch.float64, device=self.device)
        per_atom = torch.zeros(4, self.spec.n_atoms, dtype=torch.float64, device=self.device)
        full_totals = torch.zeros_like(totals) if full else None
        full_per_atom = torch.zeros_like(per_atom) if full else None
        clip = torch.zeros(2, dtype=torch.float64, device=self.device)
        for x, _ in dataloader:
            crops = self._crops(x, generator)
            clip += crops.clipped
            for start in range(0, crops.window.numel(), self.train_spec.crop_batch):
                index = torch.arange(start, min(start + self.train_spec.crop_batch, crops.window.numel()),
                                     device=self.device)
                _, stats, atom_sums = self._step(crops, index, atoms)
                totals += stats
                per_atom += atom_sums
            if full_totals is not None and full_per_atom is not None:
                stats, atom_sums = self._window_stats(crops.rows, crops.valid, atoms)
                full_totals += stats
                full_per_atom += atom_sums
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
        atom's placements with overlaps ignored); ``mean_amplitude`` (mean positive
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
        """Learn the atoms by reconstruction of random grid-aligned crops of the windows.

        The windows are re-shuffled every pass by a seeded copy of the loader. The
        atoms are initialized (``train_spec.init``), then each batch is turned into
        encoded rows and cut into crops; only the central ``crop_len`` samples of a
        crop are scored, and crops above the energy gate are dropped. Full-segment
        window selection is recomputed before each step with the current atoms;
        the crop responses are then recomputed with gradients. Labels are
        ignored. The loss is differentiated through the atom constraint, and the
        constraint is re-imposed on the parameter after every step. After every
        epoch but the last, with ``recenter_atoms``, atoms whose energy centroid is
        off center are shifted (validation is measured before). ``history``
        records, per epoch: ``objective``; ``residual_frac``, ``gain`` and
        ``residual_frac_at_gain`` on the training crops and, with the ``val_``
        prefix, on fixed validation crops, and with the ``val_full_`` prefix on the
        whole validation windows (``full_window_val``; NaN without a validation
        loader);
        ``selected_zero_frac`` (share of top-R slots that hold a zero);
        ``n_effective_atoms`` (participation ratio of the energy shares) and
        ``max_energy_share``; ``median_coherence``; ``atom_norm_min`` and
        ``atom_norm_max``; ``crops_dropped_frac`` and ``crop_energy_mean_over_median``;
        ``clipped_frac`` and ``val_clipped_frac`` (share of usable encoded samples at
        the clip); ``n_recentered``; ``n_steps``. ``atom_stats`` keeps the per-atom table of the last epoch
        (whole validation windows, else validation crops, else training crops). A module loaded from a checkpoint
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
            If ``provenance`` names no training subjects, or the windows are shorter
            than a crop with its margins.
        """
        if self.fitted:
            log.info("Waveform SAE already fitted (pretrained checkpoint); skipping fit_unsupervised")
            return
        provenance = dict(provenance or {})
        if not provenance.get("train_subjects"):
            raise ValueError("fit_unsupervised needs provenance['train_subjects'] (the subjects behind the loader)")
        spec, train_spec = self.spec, self.train_spec
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
            clip = torch.zeros(2, dtype=torch.float64, device=self.device)
            energies = [torch.zeros(0, device=self.device)]
            for x, _ in tqdm(loader, desc=f"Waveform SAE epoch {epoch + 1}/{train_spec.epochs}"):
                crops = self._crops(x, generator)
                energies.append(crops.energy)
                n_dropped += crops.dropped
                clip += crops.clipped
                order = torch.randperm(crops.window.numel(), generator=generator).to(self.device)
                for start in range(0, crops.window.numel(), train_spec.crop_batch):
                    index = order[start : start + train_spec.crop_batch]
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
                "objective": loss_sum / max(n_steps, 1),
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
                "crops_dropped_frac": n_dropped / max(n_nonflat, 1),
                "crop_energy_mean_over_median": (
                    float(all_energy.mean() / all_energy.median().clamp_min(_EPS)) if n_nonflat else float("nan")
                ),
                "clipped_frac": float(clip[0] / clip[1].clamp_min(1.0)),
                "val_clipped_frac": val_clipped_frac,
                "n_recentered": n_recentered,
                "n_steps": n_steps,
            }
            self.history.append(record)
            log.info(
                f"Waveform SAE epoch {epoch + 1}/{train_spec.epochs}: residual {record['residual_frac']:.1%} of "
                f"signal power (val crops {record['val_residual_frac']:.1%}, val windows "
                f"{record['val_full_residual_frac']:.1%}), gain {record['gain']:.2f}, "
                f"effective atoms {record['n_effective_atoms']:.1f}/{spec.n_atoms}, zero slots "
                f"{record['selected_zero_frac']:.1%}, median coherence {record['median_coherence']:.2f}, "
                f"crops dropped {record['crops_dropped_frac']:.1%}, clipped {record['clipped_frac']:.3%}, "
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
            "selection_scope": "full_encoded_segment",
            "n_channels": int(self.atoms.shape[1]),
            "channels": None if channels is None else [str(c) for c in channels],
            "atom_stats_split": None if last is None else stats_split,
        }
        self.provenance = provenance
        self.fitted = True
