"""ShapeConv sparse autoencoder: a learned shapelet dictionary as window features.

Implements ``documentation/shapeconv_sae_design.md`` (and its demo
``documentation/shapeconv_sae.py``) inside the project pipeline; the mathematics and
the design choices are recorded in ``documentation/shapeconv_sae_implementation.md``.

The encoder is the ShapeConv convolution of Qu et al. (ICLR 2024, "CNN kernels can be
the best shapelets") without its global min-pool: every channel is cross-correlated
with a bank of zero-mean, unit-norm atoms shared across channels, and the map is
turned into a signed sparse code by a learnable per-atom soft threshold with an L1
penalty (``mode="shrink"``) or by a TopK selection per crop (``mode="topk"``), then
thinned by non-maximum suppression. This is a one-pass approximation of sparse
inference, not a converged sparse solver. The decoder is the tied transposed
convolution. Codes are signed: on a bipolar montage one discharge appears with
opposite polarity on the two derivations around the focus.

Atoms are learned without labels from random crops of the training windows; each
crop carries context on both sides and only its central part is scored, so every
scored sample has the full set of atom placements available. At extraction time each
window yields, per atom, the channel-activation count (NMS peaks with
``abs(a) > amp_min``, summed over channels), the peak absolute sparse coefficient,
and the maximum absolute patch cosine over channels and time. ``events`` returns the
activations themselves (window, channel, atom, sample, signed coefficient).

Every (window, channel) row is first-differenced (``pre_emphasis="diff"``, the
default; ``none`` skips it), then centered on its median, scaled by its MAD and
clipped, so atoms, thresholds and coefficients are in robust-std units of the
encoded row. The difference flattens the ``1/f^2`` EEG spectrum so that the
background response of a matched filter is near unit variance for any atom; MAD
scaling alone is not a whitening.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger as log
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.components.hydra_transform import HydraTransform

__all__ = ["CHECKPOINT_NAME", "AtomSpec", "ShapeConvSAE", "TrainSpec"]

_EPS = 1e-8
_ENERGY_FLOOR = 1e-6
_FLAT_PATCH_NORM = 1e-3
_MAD_TO_STD = 1.4826
_CLIP = 20.0
_MODES = ("shrink", "topk")
_PRE_EMPHASIS = ("none", "diff", "diff_ar")
_STRUCTURAL_FIELDS = ("n_atoms", "atom_len", "mode", "pre_emphasis", "ar_order")
_RESEED_PER_STEP = 4
_RESEED_POOL_FACTOR = 4
CHECKPOINT_NAME = "sae_state.pt"
CHECKPOINT_FORMAT = 4
# Spec fields absent from older checkpoints and the values that reproduce their behaviour.
_SPEC_MIGRATIONS = {
    2: {"pre_emphasis": "none", "ar_order": 8, "nms_half_width": None, "amp_min_relative": False},
    3: {"ar_order": 8, "nms_half_width": None, "amp_min_relative": False},
}
_AR_RIDGE = 1e-6


@dataclass(frozen=True)
class AtomSpec:
    """Dictionary geometry and encoder, shared by training and extraction.

    Parameters
    ----------
    n_atoms : int
        Number of atoms (shapelets) in the dictionary.
    atom_len : int
        Atom length in samples (80 = 312 ms at 256 Hz).
    mode : str
        ``shrink``: signed soft threshold ``sign(z) * relu(abs(z) - theta_k)`` with a
        learnable per-atom ``theta_k`` and an L1 penalty (``TrainSpec.lam``).
        ``topk``: keep the ``topk`` largest ``abs(z)`` per crop during training. At
        extraction there is no budget, so set ``amp_min`` (or ``rho_min``).
    thresh : float
        Initial ``theta_k`` in robust-std units (``shrink``).
    topk : int
        Nonzero code entries per crop (``topk``).
    rho_min : float
        Optional gate on the cosine between an atom and the centered patch it
        covers; matches with ``abs(rho) < rho_min`` are discarded. 0 disables it.
    amp_min : float
        Extraction only: an NMS peak counts as an activation when ``abs(a) > amp_min``.
    pre_emphasis : str
        ``diff``: encode the first difference of every row (after which the row is
        scaled). EEG power falls as about ``1/f^2``, so a smooth atom correlated with
        the raw signal has a background response many times the row's robust std
        and fires everywhere; the first difference multiplies the spectrum by
        ``4 sin^2(w/2)`` and removes most of that low-frequency dominance. It is a
        pre-emphasis, not a whitening: the background response scale still depends
        on the atom and on the band, and is measured per atom (``atom_response_std``)
        rather than assumed. Atoms then live in the differenced domain;
        ``atoms_signal_domain`` integrates them back. ``diff_ar``: the difference
        followed by an AR(``ar_order``) whitening filter fitted on the training rows
        (pooled Yule-Walker), for band-limited signals where the difference alone
        leaves the background coloured. ``none``: encode the scaled row as is.
    ar_order : int
        Order of the whitening filter (``diff_ar`` only). Structural: a checkpoint
        must match.
    nms_half_width : int | None
        Half-width of the non-maximum-suppression neighbourhood in samples; ``None``
        uses ``atom_len // 2``. Two activations of one atom closer than this on one
        channel keep only the larger, so the value sets the closest pair of events
        that can both be detected. The training context follows it.
    amp_min_relative : bool
        Interpret ``amp_min`` in units of each atom's measured background response
        standard deviation (``atom_response_std``, from the validation crops) instead
        of robust-std units of the row. Requires a fit with a validation loader.
    """

    n_atoms: int = 64
    atom_len: int = 80
    mode: str = "shrink"
    thresh: float = 3.0
    topk: int = 4
    rho_min: float = 0.0
    amp_min: float = 0.0
    pre_emphasis: str = "diff"
    ar_order: int = 8
    nms_half_width: int | None = None
    amp_min_relative: bool = False


@dataclass(frozen=True)
class TrainSpec:
    """Unsupervised training schedule of the dictionary.

    Parameters
    ----------
    epochs : int
        Passes over the training windows. Every pass draws fresh random crops, so
        the update count is ``epochs * rows * crops_per_row / crop_batch``.
    lr : float
        Adam learning rate.
    lam : float
        L1 weight in ``shrink`` mode (per-sample L1 of the code on the scored part
        of the crop, as in the demo).
    crop_len : int
        Scored length of a training crop in samples (256 = 1 s at 256 Hz).
    context_len : int | None
        Unscored context added on each side of a crop, so that every scored sample
        has all ``atom_len`` atom placements and full NMS neighbourhoods available.
        ``None`` uses ``atom_len - 1 + atom_len // 2``; 0 disables the context.
    crops_per_row : int
        Random crops drawn per (window, channel) row on every pass, in expectation
        (rows are drawn uniformly within each batch).
    crop_batch : int
        Crops per optimizer step.
    lambda_div : float
        Weight of the diversity penalty: the mean over atom pairs of the maximum
        absolute cross-correlation over lags. Zero disables it.
    n_init_samples : int
        Sub-sequences sampled across the whole training loader for the k-means
        initialization of the atoms.
    """

    epochs: int = 30
    lr: float = 3e-3
    lam: float = 0.5
    crop_len: int = 256
    context_len: int | None = None
    crops_per_row: int = 16
    crop_batch: int = 1024
    lambda_div: float = 0.0
    n_init_samples: int = 20000


class ShapeConvSAE(nn.Module):
    """Learned shapelet dictionary (ShapeConv encoder, tied decoder) as a feature extractor.

    Plugs into the sklearn-style ``Trainer`` like ``HydraTransformer``: ``forward``
    maps a window batch ``(B, C, T)`` to features ``(B, 3 * n_atoms)``. Unlike HYDRA
    the atoms are learned: either the trainer calls ``fit_unsupervised`` on the
    training dataloader (labels are never read), or a dictionary trained earlier by
    ``src/train_sae.py`` is loaded through ``pretrained`` and the fit is skipped.

    Parameters
    ----------
    spec : AtomSpec | None
        Dictionary geometry and encoder; ``None`` uses the defaults. When a
        checkpoint is loaded, its saved spec replaces this one unless
        ``override_spec`` is set (structural fields must match in both cases).
    train_spec : TrainSpec | None
        Unsupervised training schedule; ``None`` uses the defaults.
    random_state : int
        Seed for the atom initialization, the batch order and the crop sampling.
    device : str | None
        ``cpu`` | ``cuda`` | ``cuda:N`` | ``auto`` (cuda when available).
    chunk_rows : int
        (window, channel) rows scored per pass at extraction time. Memory scales
        with ``chunk_rows * n_atoms * T``; 16 rows of a 2-minute window at 256 Hz
        with 64 atoms need about 0.5 GiB.
    pretrained : str | Path | None
        Path to a ``sae_state.pt`` checkpoint written by ``save_artifacts``.
    override_spec : bool
        With ``pretrained``: keep the non-structural fields of ``spec`` (``thresh``,
        ``topk``, ``rho_min``, ``amp_min``) instead of the saved ones. Logged.

    Raises
    ------
    ValueError
        If ``spec.mode`` or ``spec.pre_emphasis`` has an unknown value.
    """

    def __init__(
        self,
        spec: AtomSpec | None = None,
        train_spec: TrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        chunk_rows: int = 16,
        pretrained: str | Path | None = None,
        override_spec: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec or AtomSpec()
        self.train_spec = train_spec or TrainSpec()
        if self.spec.mode not in _MODES:
            raise ValueError(f"spec.mode must be one of {_MODES}, got {self.spec.mode!r}")
        if self.spec.pre_emphasis not in _PRE_EMPHASIS:
            raise ValueError(f"spec.pre_emphasis must be one of {_PRE_EMPHASIS}, got {self.spec.pre_emphasis!r}")
        self.random_state = random_state
        self.chunk_rows = chunk_rows
        self.device = HydraTransform._resolve_device(device)
        generator = torch.Generator().manual_seed(random_state)
        init = torch.randn(self.spec.n_atoms, 1, self.spec.atom_len, generator=generator)
        self.atoms = nn.Parameter(self._project(init).to(self.device))
        self.log_thresh = nn.Parameter(
            torch.full((self.spec.n_atoms,), math.log(self.spec.thresh), device=self.device)
        )
        self.register_buffer("ar_coef", torch.zeros(max(self.spec.ar_order, 0), device=self.device))
        self.register_buffer("amp_min_atom", torch.zeros(self.spec.n_atoms, device=self.device))
        self.history: list[dict[str, float]] = []
        self.provenance: dict[str, Any] = {}
        self.fit_meta: dict[str, Any] = {}
        self.atom_response_std: np.ndarray | None = None
        self.artifact_format = CHECKPOINT_FORMAT
        self.fitted = False
        self._pool_energy = torch.zeros(0, device=self.device)
        self._pool_patches = torch.zeros(0, self.spec.atom_len, device=self.device)
        if pretrained:
            self.load_checkpoint(pretrained, override_spec=override_spec)
        if self.spec.mode == "topk" and self.spec.amp_min <= 0 and self.spec.rho_min <= 0:
            log.warning("topk mode with amp_min=0 and rho_min=0: extraction counts every NMS peak (background)")

    # ------------------------------------------------------------------ checkpoint

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickle (joblib) and migrate an instance written by an older module version.

        An instance pickled before ``pre_emphasis`` existed encoded raw rows, so its
        spec is rebuilt with ``pre_emphasis="none"`` instead of inheriting the
        class default; missing fit metadata is filled from the constructor values
        it was fitted with and marked as migrated.
        """
        super().__setstate__(state)
        if getattr(self, "artifact_format", None) == CHECKPOINT_FORMAT:
            return
        present = vars(self.spec)
        fill = {k: v for k, v in _SPEC_MIGRATIONS[2].items() if k not in present}
        if fill:
            self.spec = replace(self.spec, **fill)
            log.warning(f"Migrated a pickled ShapeConvSAE: spec fields filled {fill}")
        if "ar_coef" not in self._buffers:
            self.register_buffer("ar_coef", torch.zeros(max(self.spec.ar_order, 0), device=self.device))
        if "amp_min_atom" not in self._buffers:
            self.register_buffer("amp_min_atom", torch.zeros(self.spec.n_atoms, device=self.device))
        if not getattr(self, "fit_meta", None):
            self.fit_meta = {
                "train_spec": asdict(self.train_spec), "random_state": self.random_state, "migrated": True,
            }
        if not hasattr(self, "atom_response_std"):
            self.atom_response_std = None
        self.artifact_format = CHECKPOINT_FORMAT

    def validate_artifact(self) -> None:
        """Refuse a fitted dictionary whose provenance or schema is unusable; an unfitted module passes.

        Raises
        ------
        ValueError
            If the module is fitted but records no training subjects, or its
            artifact schema is not the current one.
        """
        if not self.fitted:
            return
        if self.artifact_format != CHECKPOINT_FORMAT:
            raise ValueError(
                f"fitted ShapeConvSAE has artifact format {self.artifact_format}, expected {CHECKPOINT_FORMAT}"
            )
        if not self.provenance.get("train_subjects"):
            raise ValueError("fitted ShapeConvSAE records no training subjects; refusing to reuse it")

    @staticmethod
    def _saved_spec(checkpoint: dict[str, Any], path: str | Path) -> AtomSpec:
        """The ``AtomSpec`` a checkpoint was trained with; formats 2 and 3 are migrated (fields filled)."""
        version = checkpoint.get("format")
        spec_dict = dict(checkpoint.get("spec") or {})
        required = {f.name for f in fields(AtomSpec)}
        if version in _SPEC_MIGRATIONS:
            fill = {k: v for k, v in _SPEC_MIGRATIONS[version].items() if k not in spec_dict}
            spec_dict.update(fill)
            log.warning(f"{path}: format-{version} checkpoint migrated; spec fields filled {fill}")
        elif version != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: unsupported checkpoint format {version!r} (expected {CHECKPOINT_FORMAT})")
        missing, unknown = required - set(spec_dict), set(spec_dict) - required
        if missing or unknown:
            raise ValueError(f"{path}: checkpoint spec fields missing {sorted(missing)} / unknown {sorted(unknown)}")
        return AtomSpec(**spec_dict)

    def load_checkpoint(self, path: str | Path, override_spec: bool = False) -> None:
        """Load atoms, thresholds, history, fit metadata and provenance from ``sae_state.pt``; mark fitted.

        The saved ``AtomSpec`` is restored by default, so a reloaded dictionary is
        encoded exactly as it was trained. With ``override_spec`` the constructor's
        non-structural fields win and the difference is logged. The training
        schedule and seed the dictionary was fitted with are kept in ``fit_meta``
        and are not replaced by the constructor's values.

        Raises
        ------
        ValueError
            If the checkpoint format is unsupported, its spec is incomplete, a
            structural field (``n_atoms``, ``atom_len``, ``mode``, ``pre_emphasis``)
            differs from ``spec``, or it carries no training-subject provenance.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        saved = self._saved_spec(checkpoint, path)
        for key in _STRUCTURAL_FIELDS:
            if getattr(saved, key) != getattr(self.spec, key):
                raise ValueError(
                    f"{path} has {key}={getattr(saved, key)!r} but spec.{key}={getattr(self.spec, key)!r}; "
                    f"set feature.spec.{key} to the checkpoint's value"
                )
        differing = {
            f.name: (getattr(saved, f.name), getattr(self.spec, f.name))
            for f in fields(AtomSpec)
            if getattr(saved, f.name) != getattr(self.spec, f.name)
        }
        if differing and override_spec:
            log.warning(f"Overriding saved inference spec (saved, config): {differing}")
        elif differing:
            log.warning(f"Restoring saved inference spec; config differed on (saved, config): {differing}")
            self.spec = saved
        provenance = checkpoint.get("provenance") or {}
        if not provenance.get("train_subjects"):
            raise ValueError(f"{path} has no training-subject provenance; refusing to reuse it")
        missing, unexpected = self.load_state_dict(checkpoint["state_dict"], strict=False)
        tolerated = {"amp_min_atom"} | (set() if self.spec.pre_emphasis == "diff_ar" else {"ar_coef"})
        if unexpected or not set(missing) <= tolerated:
            raise ValueError(f"{path}: state_dict mismatch (missing {missing}, unexpected {unexpected})")
        self.history = checkpoint["history"]
        self.provenance = provenance
        self.fit_meta = checkpoint.get("fit") or {
            "train_spec": checkpoint.get("train_spec"), "random_state": checkpoint.get("random_state"),
            "migrated": True,
        }
        response = checkpoint.get("atom_response_std")
        self.atom_response_std = None if response is None else np.asarray(response, dtype=np.float32)
        self.artifact_format = CHECKPOINT_FORMAT
        self.fitted = True
        log.info(
            f"Loaded {self.spec.n_atoms} pretrained atoms from {path} "
            f"({len(self.history)} epochs, {len(provenance['train_subjects'])} training subjects)"
        )

    def save_artifacts(self, output_dir: Path) -> None:
        """Write ``sae_state.pt`` and the inspectable arrays.

        ``sae_atoms.npy`` (encoded domain), ``sae_atoms_signal.npy`` (integrated
        synthesis shapes, ``atom_len + 1`` samples), ``sae_atom_response_std.npy``
        (when measured) and ``sae_training.csv``. The checkpoint stores the fit
        metadata of the dictionary as fitted, not the constructor's current values.
        """
        output_dir = Path(output_dir)
        fit_meta = self.fit_meta or {"train_spec": asdict(self.train_spec), "random_state": self.random_state}
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "spec": asdict(self.spec),
                "fit": fit_meta,
                "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
                "history": self.history,
                "provenance": self.provenance,
                "atom_response_std": None if self.atom_response_std is None else self.atom_response_std.tolist(),
            },
            output_dir / CHECKPOINT_NAME,
        )
        np.save(output_dir / "sae_atoms.npy", self.atoms_numpy)
        np.save(output_dir / "sae_atoms_signal.npy", self.atoms_signal_domain)
        if self.atom_response_std is not None:
            np.save(output_dir / "sae_atom_response_std.npy", self.atom_response_std)
        if self.calibrated_thresholds is not None:
            np.save(output_dir / "sae_amp_min_atom.npy", self.calibrated_thresholds)
        if self.history:
            pl.DataFrame(self.history).write_csv(output_dir / "sae_training.csv")
        log.info(f"Saved {self.spec.n_atoms} atoms and {len(self.history)} training epochs to {output_dir}")

    @property
    def train_subjects(self) -> list[str]:
        """Subjects whose windows trained the dictionary (empty before fitting)."""
        return list(self.provenance.get("train_subjects", []))

    @property
    def atoms_numpy(self) -> np.ndarray:
        """The normalized atoms as a ``(n_atoms, atom_len)`` float32 array, in the encoded domain."""
        return self._project(self.atoms.detach()).squeeze(1).cpu().numpy()

    @property
    def atoms_signal_domain(self) -> np.ndarray:
        """Integrated synthesis shapes ``(n_atoms, atom_len + 1)`` under ``diff``; the atoms themselves under ``none``.

        Under ``diff`` the inverse of ``atom_len`` differences from a zero baseline is
        ``[0, cumsum(s)]``, with ``atom_len + 1`` samples; a zero-sum atom returns to
        that baseline. The shape is re-centered and re-normalized for display, so
        its reconstruction scale is not kept, and the clip makes the preprocessing
        irreversible in any case. It is the component the atom synthesizes, not the
        filter it applies; see ``atoms_analysis_filter``. Under ``none`` it is the
        atom itself (``atom_len`` samples).
        """
        atoms = self._project(self.atoms.detach())
        if self.spec.pre_emphasis == "diff_ar":
            atoms = self._unwhiten(atoms)
        if self.spec.pre_emphasis in ("diff", "diff_ar"):
            atoms = self._project(F.pad(atoms.cumsum(-1), (1, 0)))
        return atoms.squeeze(1).cpu().numpy()

    def _unwhiten(self, atoms: torch.Tensor) -> torch.Tensor:
        """Inverse of the AR whitening (all-pole recursion), truncated at ``4 * ar_order`` extra samples."""
        p = self.spec.ar_order
        if p == 0:
            return atoms
        coef = self.ar_coef.detach()
        n_out = atoms.shape[-1] + 4 * p
        out = torch.zeros(*atoms.shape[:-1], n_out, device=atoms.device, dtype=atoms.dtype)
        for t in range(n_out):
            drive = atoms[..., t] if t < atoms.shape[-1] else 0.0
            back = sum(coef[i - 1] * out[..., t - i] for i in range(1, min(p, t) + 1))
            out[..., t] = drive + back
        return out

    @property
    def atoms_analysis_filter(self) -> np.ndarray:
        """The original-domain analysis filter of each atom, ``(n_atoms, atom_len + 1)`` under ``diff``.

        ``<s, diff(x)> = <f, x>`` with ``f[u] = s[u - 1] - s[u]`` (``s[-1] = s[L] = 0``),
        so ``f`` is what the encoder correlates with the scaled original row. Unit
        normalized for display. Under ``none`` it is the atom itself.
        """
        atoms = self._project(self.atoms.detach())
        if self.spec.pre_emphasis == "diff_ar" and self.spec.ar_order > 0:
            # <s, W x> = <W^T s, x>: correlate with the causal whitening kernel.
            kernel = self._whitening_kernel().flip(-1)
            atoms = F.conv1d(F.pad(atoms, (self.spec.ar_order, self.spec.ar_order)), kernel)
        if self.spec.pre_emphasis in ("diff", "diff_ar"):
            atoms = self._project(F.pad(atoms, (1, 0)) - F.pad(atoms, (0, 1)))
        return atoms.squeeze(1).cpu().numpy()

    def _whitening_kernel(self) -> torch.Tensor:
        """The causal whitening FIR ``[-a_p, ..., -a_1, 1]`` as a ``(1, 1, ar_order + 1)`` conv1d kernel."""
        one = torch.ones(1, device=self.ar_coef.device, dtype=self.ar_coef.dtype)
        return torch.cat([-self.ar_coef.flip(0), one]).view(1, 1, -1)

    def _whiten(self, rows: torch.Tensor) -> torch.Tensor:
        """Apply ``y[t] = x[t] - sum_i a_i x[t - i]`` to rows ``(m, T)``; identity while the filter is unfitted."""
        p = self.spec.ar_order
        if p == 0 or not bool(self.ar_coef.any()):
            return rows
        return F.conv1d(F.pad(rows.unsqueeze(1), (p, 0)), self._whitening_kernel()).squeeze(1)

    def _rows(self, X: torch.Tensor) -> torch.Tensor:
        """(window, channel) rows of a batch ``(B, C, T)``: pre-emphasis, then robust scaling; ``(B*C, T')``."""
        rows = X.to(self.device).flatten(0, 1)
        if self.spec.pre_emphasis in ("diff", "diff_ar"):
            rows = rows.diff(dim=-1)
        if self.spec.pre_emphasis == "diff_ar":
            rows = self._whiten(self._robust_scale(rows))
        return self._robust_scale(rows)

    @torch.no_grad()
    def _fit_ar(self, dataloader: DataLoader) -> None:
        """Fit the AR whitening filter by pooled Yule-Walker on the differenced, scaled training rows."""
        p = self.spec.ar_order
        if p == 0:
            return
        self.ar_coef.zero_()
        autocorr = torch.zeros(p + 1, device=self.device, dtype=torch.float64)
        for x, _ in tqdm(dataloader, desc="ShapeConv SAE AR whitening fit"):
            rows = self._rows(x).double()
            for k in range(p + 1):
                autocorr[k] += (rows[:, k:] * rows[:, : rows.shape[1] - k]).sum()
        autocorr = autocorr / autocorr[0].clamp_min(_EPS)
        index = torch.arange(p, device=self.device)
        toeplitz = autocorr[(index[:, None] - index[None, :]).abs()]
        toeplitz = toeplitz + _AR_RIDGE * torch.eye(p, device=self.device, dtype=torch.float64)
        self.ar_coef.copy_(torch.linalg.solve(toeplitz, autocorr[1 : p + 1]).to(self.ar_coef.dtype))
        log.info(f"Fitted AR({p}) whitening: a = {[round(float(v), 3) for v in self.ar_coef]}")

    @property
    def thresholds(self) -> np.ndarray:
        """The per-atom soft thresholds ``theta_k`` (robust-std units), ``(n_atoms,)``."""
        return self.log_thresh.detach().exp().cpu().numpy()

    @property
    def nms_half(self) -> int:
        """Half-width of the NMS neighbourhood in samples (``nms_half_width`` or ``atom_len // 2``)."""
        return self.spec.atom_len // 2 if self.spec.nms_half_width is None else int(self.spec.nms_half_width)

    @property
    def context(self) -> int:
        """Unscored samples on each side of a training crop: all placements plus the NMS neighbourhood."""
        if self.train_spec.context_len is None:
            return self.spec.atom_len - 1 + self.nms_half
        return self.train_spec.context_len

    # ------------------------------------------------------------------ primitives

    @staticmethod
    def _project(atoms: torch.Tensor) -> torch.Tensor:
        """Zero mean and unit L2 norm along the last axis; a zero vector stays zero."""
        atoms = atoms - atoms.mean(-1, keepdim=True)
        return atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(_EPS)

    @staticmethod
    def _robust_scale(x: torch.Tensor) -> torch.Tensor:
        """Center every row on its median, scale by 1.4826 * MAD, clip at +-20.

        A floor of 1 percent of the plain std keeps a mostly-flat row from
        exploding; an all-constant row maps to zeros.
        """
        median = x.median(dim=-1, keepdim=True).values
        centered = x - median
        mad = centered.abs().median(dim=-1, keepdim=True).values * _MAD_TO_STD
        scale = torch.maximum(mad, 0.01 * centered.std(dim=-1, keepdim=True))
        return (centered / scale.clamp_min(_EPS)).clamp(-_CLIP, _CLIP)

    @staticmethod
    def _sample_subsequences(
        rows: torch.Tensor, length: int, n: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Draw ``n`` random sub-sequences of ``length`` samples from ``rows`` ``(m, T)``."""
        m, n_times = rows.shape
        row_idx = torch.randint(0, m, (n,), generator=generator).to(rows.device)
        start = torch.randint(0, n_times - length + 1, (n,), generator=generator).to(rows.device)
        return rows.unfold(1, length, 1)[row_idx, start]

    def _scores(self, x: torch.Tensor, atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Best-fit coefficient and centered-patch cosine of every atom at every valid position.

        Parameters
        ----------
        x : torch.Tensor
            Rows ``(n, 1, T)``.
        atoms : torch.Tensor
            Normalized atoms ``(n_atoms, 1, atom_len)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``y = <atom, patch>`` and ``rho = y / ||patch - mean(patch)||``, both
            ``(n, n_atoms, T - atom_len + 1)``. With a zero-mean unit-norm atom and a
            free intercept ``b``, ``min_{b,c} ||patch - b - c atom||^2 = E (1 - rho^2)``
            at ``c = y``, where ``E`` is the centered patch energy. Near the energy
            floor ``rho`` is a stabilized score, not the exact cosine.
        """
        length = self.spec.atom_len
        amplitude = F.conv1d(x, atoms)
        box = torch.ones(1, 1, length, device=x.device, dtype=x.dtype)
        window_sum = F.conv1d(x, box)
        window_energy = F.conv1d(x * x, box) - window_sum * window_sum / length
        rho = amplitude / window_energy.clamp_min(_ENERGY_FLOOR).sqrt()
        return amplitude, rho

    def _local_maxima(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Strict local maxima of a non-negative map within +-``nms_half`` samples, earliest on ties.

        A position survives when it is positive, strictly larger than every value in
        the ``nms_half`` samples before it, and at least as large as every value in
        the ``nms_half`` samples after it. A plateau therefore keeps
        only its first sample. The comparison is per (row, channel); there is no
        competition across atoms or channels.
        """
        half = self.nms_half
        positive = magnitude > 0
        if half == 0:
            return positive
        n_pos = magnitude.shape[-1]
        left = F.max_pool1d(F.pad(magnitude, (half, 0)), half, stride=1)[..., :n_pos]
        right = F.max_pool1d(F.pad(magnitude, (0, half)), half, stride=1)[..., 1 : n_pos + 1]
        return positive & (magnitude > left) & (magnitude >= right)

    def _code(self, amplitude: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Signed sparse code: optional cosine gate, soft threshold (``shrink``), then NMS on ``abs``."""
        z = amplitude
        if self.spec.rho_min > 0:
            z = z * (rho.abs() >= self.spec.rho_min)
        if self.spec.mode == "shrink":
            theta = self.log_thresh.exp().view(1, -1, 1)
            z = torch.sign(z) * F.relu(z.abs() - theta)
        return z * self._local_maxima(z.abs())

    def _sparse_code(self, crops: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Training code ``(n, n_atoms, P)``: ``_code`` plus, in ``topk`` mode, the K largest ``abs`` per crop."""
        amplitude, rho = self._scores(crops, atoms)
        z = self._code(amplitude, rho)
        if self.spec.mode != "topk":
            return z
        flat = z.flatten(1)
        index = flat.abs().topk(min(self.spec.topk, flat.shape[1]), dim=1).indices
        return torch.zeros_like(flat).scatter(1, index, flat.gather(1, index)).view_as(z)

    def _diversity(self, atoms: torch.Tensor) -> torch.Tensor:
        """Mean over atom pairs of the maximum absolute cross-correlation over lags (sign and shift aware)."""
        n_atoms = self.spec.n_atoms
        if n_atoms < 2:
            return atoms.new_zeros(())
        coherence = F.conv1d(atoms, atoms, padding=self.spec.atom_len - 1).abs().amax(-1)
        off_diagonal = ~torch.eye(n_atoms, dtype=torch.bool, device=atoms.device)
        return coherence[off_diagonal].mean()

    # ------------------------------------------------------------------ training

    def _center(self, t: torch.Tensor) -> torch.Tensor:
        """The scored part of a context crop (last axis)."""
        return t[..., self.context : self.context + self.train_spec.crop_len]

    def _scored_positions(self) -> tuple[int, int]:
        """Code positions whose atom overlaps the scored part of a crop, as a half-open slice."""
        length = self.spec.atom_len
        return max(self.context - length + 1, 0), self.context + self.train_spec.crop_len

    def _crops(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Random crops ``(n, 1, crop_len + 2 context)`` of a window batch, centered on their scored part.

        Rows are drawn uniformly within the batch, ``crops_per_row`` per row in
        expectation. Crops whose scored part is flat (failed loads) are dropped.

        Raises
        ------
        ValueError
            If the windows are shorter than a context crop.
        """
        train_spec = self.train_spec
        rows = self._rows(x)
        total = train_spec.crop_len + 2 * self.context
        if rows.shape[1] < total:
            raise ValueError(f"windows of {rows.shape[1]} samples are shorter than crop_len + 2 * context = {total}")
        n_crops = rows.shape[0] * train_spec.crops_per_row
        crops = self._sample_subsequences(rows, total, n_crops, generator).unsqueeze(1)
        crops = crops - self._center(crops).mean(-1, keepdim=True)
        return crops[self._center(crops).pow(2).sum(dim=(1, 2)) > _ENERGY_FLOOR]

    def _step(
        self, crops: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loss of one crop batch on its scored part, with statistics, per-atom usage and the residual.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The loss; ``[residual_energy, signal_energy, n_starts, n_crops]`` on the
            scored part, where ``n_starts`` counts code entries whose placement
            starts inside the scored part (a stationary per-crop rate, unlike the
            ``(crop_len + atom_len - 1)`` overlapping placements the L1 term sums);
            nonzero entries per atom over the overlapping placements (usage); and
            the detached scored residual ``(n, 1, crop_len)``.
        """
        lo, hi = self._scored_positions()
        z = self._sparse_code(crops, atoms)
        residual = self._center(crops - F.conv_transpose1d(z, atoms))
        loss = 0.5 * residual.pow(2).mean()
        z_scored = z[..., lo:hi]
        if self.spec.mode == "shrink":
            loss = loss + self.train_spec.lam * z_scored.abs().sum(dim=(1, 2)).mean() / residual.shape[-1]
        if self.train_spec.lambda_div > 0:
            loss = loss + self.train_spec.lambda_div * self._diversity(atoms)
        active = z_scored.detach() != 0
        starts = self._center(z.detach()) != 0
        stats = torch.stack(
            [
                residual.detach().pow(2).sum(),
                self._center(crops).pow(2).sum(),
                starts.sum().to(crops.dtype),
                torch.tensor(float(crops.shape[0]), device=crops.device),
            ]
        )
        return loss, stats, active.sum(dim=(0, 2)), residual.detach()

    def _shuffled_loader(self, dataloader: DataLoader, generator: torch.Generator) -> DataLoader:
        """A loader over the same dataset that reshuffles the windows every pass (the given loader is untouched)."""
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )

    def _init_atoms(self, dataloader: DataLoader, generator: torch.Generator) -> None:
        """K-means on random unit-norm sub-sequences drawn from every batch of the loader."""
        spec, train_spec = self.spec, self.train_spec
        per_batch = max(math.ceil(train_spec.n_init_samples / max(len(dataloader), 1)), 1)
        samples = []
        for x, _ in tqdm(dataloader, desc="ShapeConv SAE k-means init"):
            rows = self._rows(x)
            samples.append(self._sample_subsequences(rows, spec.atom_len, per_batch, generator))
        candidates = self._project(torch.cat(samples))
        candidates = candidates[candidates.abs().amax(-1) > 0]
        if len(candidates) < spec.n_atoms:
            raise ValueError(
                f"k-means init needs at least n_atoms={spec.n_atoms} non-flat sub-sequences, "
                f"got {len(candidates)}; raise n_init_samples or supply more windows"
            )
        kmeans = KMeans(n_clusters=spec.n_atoms, n_init=1, random_state=self.random_state)
        kmeans.fit(candidates.cpu().numpy())
        centers = torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.atoms.copy_(self._project(centers).unsqueeze(1))
        log.info(f"Initialized {spec.n_atoms} atoms by k-means on {len(candidates)} sub-sequences")

    @torch.no_grad()
    def _collect_reseed_patches(self, residual: torch.Tensor) -> None:
        """Keep the highest-energy residual patches of a step in a bounded pool.

        Candidates are local maxima of the residual energy within one minibatch,
        so they are locally separated; overlapping crops and later minibatches can
        still contribute the same transient. Non-finite and near-flat patches are
        excluded.
        """
        length = self.spec.atom_len
        energy = F.avg_pool1d(residual.pow(2), length, stride=1)
        energy = energy * self._local_maxima(energy)
        flat = energy.flatten()
        top = flat.topk(min(_RESEED_PER_STEP, flat.numel()))
        rows, starts = top.indices // energy.shape[-1], top.indices % energy.shape[-1]
        patches = residual.squeeze(1).unfold(1, length, 1)[rows, starts]
        norms = (patches - patches.mean(-1, keepdim=True)).norm(dim=-1)
        valid = (top.values > 0) & torch.isfinite(patches).all(-1) & (norms > _FLAT_PATCH_NORM)
        if not bool(valid.any()):
            return
        self._pool_energy = torch.cat([self._pool_energy, top.values[valid]])
        self._pool_patches = torch.cat([self._pool_patches, patches[valid]])
        cap = _RESEED_POOL_FACTOR * self.spec.n_atoms
        if self._pool_energy.numel() > cap:
            keep = self._pool_energy.topk(cap).indices
            self._pool_energy, self._pool_patches = self._pool_energy[keep], self._pool_patches[keep]

    @torch.no_grad()
    def _reset_dead_atoms(self, usage: torch.Tensor, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
        """Re-seed atoms that never fired in the epoch from the residual pool; returns (dead, re-seeded).

        Re-seeded atoms get the median threshold and cleared Adam moments. Atoms
        without a valid replacement patch are left as they are.
        """
        dead = torch.nonzero(usage == 0).flatten()
        n_dead = int(dead.numel())
        n_new = min(n_dead, int(self._pool_energy.numel()))
        if n_new == 0:
            self._clear_pool()
            return n_dead, 0
        chosen = self._pool_energy.topk(n_new).indices
        targets = dead[:n_new]
        self.atoms[targets] = self._project(self._pool_patches[chosen]).unsqueeze(1)
        self.log_thresh[targets] = self.log_thresh.median()
        for param in (self.atoms, self.log_thresh):
            state = optimizer.state.get(param)
            if state:
                state["exp_avg"][targets] = 0.0
                state["exp_avg_sq"][targets] = 0.0
        self._clear_pool()
        return n_dead, n_new

    def _clear_pool(self) -> None:
        """Empty the residual pool; called at every epoch start and on every re-seed exit."""
        self._pool_energy = self._pool_energy[:0]
        self._pool_patches = self._pool_patches[:0]

    @torch.no_grad()
    def _val_stats(self, val_dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Validation statistics on fixed crops: summed step stats and the per-atom response scale.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``[residual_energy, signal_energy, n_starts, n_crops]`` summed over the
            crops, and the per-atom standard deviation of the pre-threshold
            correlation ``y`` over all positions (the empirical background response
            scale of each atom on the development cohort, events included).
        """
        generator = torch.Generator().manual_seed(self.random_state + 1)
        atoms = self._project(self.atoms)
        totals = torch.zeros(4, device=self.device)
        sum_sq = torch.zeros(self.spec.n_atoms, device=self.device)
        count = 0.0
        for x, _ in val_dataloader:
            crops = self._crops(x, generator)
            for start in range(0, crops.shape[0], self.train_spec.crop_batch):
                batch = crops[start : start + self.train_spec.crop_batch]
                _, stats, _, _ = self._step(batch, atoms)
                totals += stats
                amplitude, _ = self._scores(batch, atoms)
                sum_sq += amplitude.pow(2).sum(dim=(0, 2))
                count += float(amplitude.shape[0] * amplitude.shape[2])
        return totals, (sum_sq / max(count, 1.0)).sqrt()

    def fit_unsupervised(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Learn the atoms by sparse reconstruction of random context crops of the windows.

        The windows are re-shuffled every pass by a seeded copy of the loader (the
        given loader keeps its order for feature extraction). Each batch is scaled
        per (window, channel) row and cut into random crops; only the central
        ``crop_len`` samples of a crop are scored. Labels are ignored. Atoms are
        re-projected after every step. Atoms that never enter a code in an epoch
        are re-seeded from a pool of the worst-reconstructed residual patches of
        that epoch, except after the last epoch. ``history`` records, per epoch, the
        residual as a fraction of signal power (train and, if given, validation),
        the nonzero code entries per crop, the training objective, the dead and
        re-seeded atom counts and the step count. A module loaded from a checkpoint
        (``fitted``) returns at once and keeps its provenance.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(X, y)`` with ``X`` of shape ``(B, C, T)``.
        val_dataloader : DataLoader | None
            Held-out windows scored after every epoch (monitoring only).
        provenance : dict | None
            Recorded with the checkpoint: must contain ``train_subjects`` (the
            subjects behind ``dataloader``); may contain the data settings and a
            window-plan fingerprint (see ``src.utils.split_provenance``).

        Raises
        ------
        ValueError
            If ``crop_len`` is shorter than ``atom_len``, if the windows are shorter
            than a context crop, or if ``provenance`` names no training subjects.
        """
        if self.fitted:
            log.info("ShapeConv SAE already fitted (pretrained checkpoint); skipping fit_unsupervised")
            return
        spec, train_spec = self.spec, self.train_spec
        if train_spec.crop_len < spec.atom_len:
            raise ValueError(f"crop_len={train_spec.crop_len} must be >= atom_len={spec.atom_len}")
        provenance = dict(provenance or {})
        if not provenance.get("train_subjects"):
            raise ValueError("fit_unsupervised needs provenance['train_subjects'] (the subjects behind the loader)")
        generator = torch.Generator().manual_seed(self.random_state)
        loader = self._shuffled_loader(dataloader, generator)
        if spec.pre_emphasis == "diff_ar":
            self._fit_ar(loader)
        self._init_atoms(loader, generator)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_spec.lr)
        self.history = []
        response_std: torch.Tensor | None = None
        for epoch in range(train_spec.epochs):
            self._clear_pool()
            usage = torch.zeros(spec.n_atoms, dtype=torch.long, device=self.device)
            totals = torch.zeros(4, device=self.device)
            loss_sum, n_steps = 0.0, 0
            for x, _ in tqdm(loader, desc=f"ShapeConv SAE epoch {epoch + 1}/{train_spec.epochs}"):
                crops = self._crops(x, generator)
                order = torch.randperm(crops.shape[0], generator=generator).to(self.device)
                for start in range(0, crops.shape[0], train_spec.crop_batch):
                    batch = crops[order[start : start + train_spec.crop_batch]]
                    loss, stats, used, residual = self._step(batch, self._project(self.atoms))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        self.atoms.copy_(self._project(self.atoms))
                    self._collect_reseed_patches(residual)
                    usage += used
                    totals += stats
                    loss_sum += float(loss.detach())
                    n_steps += 1
            last = epoch + 1 == train_spec.epochs
            n_dead, n_new = (int((usage == 0).sum()), 0) if last else self._reset_dead_atoms(usage, optimizer)
            val = None
            if val_dataloader is not None:
                val, response_std = self._val_stats(val_dataloader)
            thresholds = self.log_thresh.detach().exp()
            record = {
                "epoch": epoch + 1,
                "residual_frac": float(totals[0] / totals[1].clamp_min(_EPS)),
                "active_per_crop": float(totals[2] / totals[3].clamp_min(1.0)),
                "objective": loss_sum / max(n_steps, 1),
                "val_residual_frac": float(val[0] / val[1].clamp_min(_EPS)) if val is not None else float("nan"),
                "val_active_per_crop": float(val[2] / val[3].clamp_min(1.0)) if val is not None else float("nan"),
                "median_thresh_over_response": (
                    float((thresholds / response_std.clamp_min(_EPS)).median())
                    if response_std is not None
                    else float("nan")
                ),
                "n_dead": n_dead,
                "n_reseeded": n_new,
                "n_steps": n_steps,
            }
            self.history.append(record)
            log.info(
                f"ShapeConv SAE epoch {epoch + 1}/{train_spec.epochs}: residual {record['residual_frac']:.1%} "
                f"of signal power (val {record['val_residual_frac']:.1%}), starts/crop "
                f"{record['active_per_crop']:.2f}, objective {record['objective']:.4f}, "
                f"dead {n_dead}, re-seeded {n_new}, steps {n_steps}"
            )
        self._clear_pool()
        self.atom_response_std = None if response_std is None else response_std.cpu().numpy().astype(np.float32)
        self.fit_meta = {"train_spec": asdict(train_spec), "random_state": self.random_state, "context": self.context}
        self.provenance = provenance
        self.fitted = True

    # ------------------------------------------------------------------ extraction

    def _event_threshold(self) -> torch.Tensor:
        """Per-atom extraction threshold ``(n_atoms, 1)``.

        A calibrated per-atom threshold (``calibrate_amp_min``) takes precedence;
        otherwise ``amp_min``, absolute or times the measured response std.
        """
        if bool(self.amp_min_atom.any()):
            return self.amp_min_atom.view(-1, 1)
        threshold = torch.full((self.spec.n_atoms, 1), float(self.spec.amp_min), device=self.device)
        if not self.spec.amp_min_relative:
            return threshold
        if self.atom_response_std is None:
            log.warning("amp_min_relative requested but no response scale was measured (fit without a val loader)")
            return threshold
        return threshold * torch.as_tensor(self.atom_response_std, device=self.device).view(-1, 1)

    @torch.no_grad()
    def calibrate_amp_min(
        self,
        dataloader: DataLoader,
        false_alarms_per_channel_minute: float,
        sfreq: float,
        keep_label: int | None = None,
    ) -> None:
        """Set per-atom extraction thresholds so that background activations reach a target rate.

        For every atom, the NMS peaks of the code on the loader's rows (taken as
        background, for example event-free synthetic rows or windows of negative
        training subjects) are collected, and the threshold is placed so that at
        most ``false_alarms_per_channel_minute`` peaks per channel-minute exceed it.
        With ``keep_label`` only the windows whose label equals it are used (the
        design note's calibration on negative training subjects); this is the one
        place the module reads labels, and it must see training windows only.
        An atom with fewer peaks than the allowance keeps a threshold at its
        smallest peak. The thresholds are stored in the checkpoint and take
        precedence over ``amp_min``. On EEG the loader's rows are development-cohort
        background, not verified event-free intervals, so the rate is an activation
        rate in that cohort, not a clinical false-alarm rate.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(X, y)`` with ``X`` of shape ``(B, C, T)``; ``y`` is ignored.
        false_alarms_per_channel_minute : float
            Target rate of surviving background activations per atom.
        sfreq : float
            Sampling rate of the windows, to convert samples to minutes.
        keep_label : int | None
            Use only windows with this label (``None`` uses every window).

        Raises
        ------
        ValueError
            If no window passes the label filter.
        """
        peaks: list[list[torch.Tensor]] = [[] for _ in range(self.spec.n_atoms)]
        minutes = 0.0
        for x, y in tqdm(dataloader, desc="ShapeConv SAE threshold calibration"):
            if keep_label is not None:
                x = x[torch.as_tensor(y).reshape(-1) == keep_label]
                if x.shape[0] == 0:
                    continue
            minutes += x.shape[0] * x.shape[1] * x.shape[2] / sfreq / 60.0
            for _, code, _ in self._row_chunks(x):
                magnitude = code.abs()
                for atom in range(self.spec.n_atoms):
                    values = magnitude[:, atom][magnitude[:, atom] > 0]
                    peaks[atom].append(values)
        if minutes == 0.0:
            raise ValueError(f"no window with label {keep_label!r} in the calibration loader")
        allowed = int(false_alarms_per_channel_minute * minutes)
        thresholds = torch.zeros(self.spec.n_atoms, device=self.device)
        for atom in range(self.spec.n_atoms):
            values = torch.cat(peaks[atom]) if peaks[atom] else torch.zeros(0, device=self.device)
            if values.numel() == 0:
                thresholds[atom] = _EPS
            elif values.numel() <= allowed:
                thresholds[atom] = max(float(values.min()) - _EPS, _EPS)
            else:
                thresholds[atom] = torch.topk(values, allowed + 1).values[-1]
        self.amp_min_atom.copy_(thresholds)
        self.fit_meta = {**self.fit_meta, "amp_min_calibration": {
            "false_alarms_per_channel_minute": float(false_alarms_per_channel_minute),
            "channel_minutes": float(minutes), "sfreq": float(sfreq), "keep_label": keep_label,
        }}
        log.info(
            f"Calibrated per-atom thresholds at {false_alarms_per_channel_minute} per channel-minute over "
            f"{minutes:.1f} channel-minutes: median {float(thresholds.median()):.2f}, "
            f"range {float(thresholds.min()):.2f} to {float(thresholds.max()):.2f}"
        )

    @property
    def calibrated_thresholds(self) -> np.ndarray | None:
        """Per-atom calibrated extraction thresholds ``(n_atoms,)``, or ``None`` when not calibrated."""
        return self.amp_min_atom.cpu().numpy() if bool(self.amp_min_atom.any()) else None

    def _row_chunks(self, X: torch.Tensor):
        """Yield ``(start, code, rho)`` over chunks of the scaled (window, channel) rows of ``X``."""
        rows = self._rows(X).unsqueeze(1)
        atoms = self._project(self.atoms)
        for start in range(0, rows.shape[0], self.chunk_rows):
            amplitude, rho = self._scores(rows[start : start + self.chunk_rows], atoms)
            yield start, self._code(amplitude, rho), rho

    @torch.no_grad()
    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Per-window features from the learned dictionary.

        Parameters
        ----------
        X : torch.Tensor
            Window batch ``(B, C, T)``; moved to the module device internally.
        y : torch.Tensor | None
            Accepted for interface parity with ``HydraTransformer``; unused.

        Returns
        -------
        torch.Tensor
            ``(B, 3 * n_atoms)`` float32, three blocks of ``n_atoms`` columns:
            channel-activation count (NMS peaks with ``abs(a) > amp_min``, summed
            over channels and time; one multichannel event contributes once per
            channel), peak absolute sparse coefficient over channels and time (the
            shrunk coefficient in ``shrink`` mode), and maximum absolute
            centered-patch cosine over channels and time (over all positions,
            including those the gate or the threshold rejected).
        """
        n_windows, n_channels, _ = X.shape
        threshold = self._event_threshold()
        counts, peaks, best = [], [], []
        for _, code, rho in self._row_chunks(X):
            magnitude = code.abs()
            counts.append((magnitude > threshold).sum(-1))
            peaks.append(magnitude.amax(-1))
            best.append(rho.abs().amax(-1))
        shape = (n_windows, n_channels, self.spec.n_atoms)
        return torch.cat(
            [
                torch.cat(counts).view(shape).sum(1).float(),
                torch.cat(peaks).view(shape).amax(1),
                torch.cat(best).view(shape).amax(1),
            ],
            dim=1,
        )

    @torch.no_grad()
    def events(self, X: torch.Tensor) -> pl.DataFrame:
        """Every activation of the code as a row: window, channel, atom, sample, signed coefficient.

        The sample is the first sample of the atom placement within the encoded row
        (under ``diff`` pre-emphasis, sample ``j`` of the difference spans original
        samples ``j`` and ``j + 1``). Activations with ``abs(coefficient) <= amp_min``
        are excluded. Grouping of coincident activations across channels into
        events is left to the caller.
        """
        n_channels = X.shape[1]
        threshold = self._event_threshold()
        parts = []
        for start, code, _ in self._row_chunks(X):
            index = torch.nonzero(code.abs() > threshold)
            row = index[:, 0] + start
            parts.append(
                {
                    "window": (row // n_channels).cpu().numpy(),
                    "channel": (row % n_channels).cpu().numpy(),
                    "atom": index[:, 1].cpu().numpy(),
                    "sample": index[:, 2].cpu().numpy(),
                    "coefficient": code[index[:, 0], index[:, 1], index[:, 2]].cpu().numpy(),
                }
            )
        columns = ("window", "channel", "atom", "sample", "coefficient")
        return pl.DataFrame({c: np.concatenate([p[c] for p in parts]) if parts else np.zeros(0) for c in columns})
