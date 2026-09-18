"""ShapeConv sparse autoencoder: a learned shapelet dictionary as window features.

Implements ``documentation/shapeconv_sae_design.md`` (and its demo
``documentation/shapeconv_sae.py``) inside the project pipeline. The encoder is the
ShapeConv convolution of Qu et al. (ICLR 2024, "CNN kernels can be the best
shapelets") without its global min-pool: every channel is cross-correlated with a
bank of zero-mean, unit-norm atoms shared across channels, and the map is turned into
a signed sparse code by a learnable per-atom soft threshold with an L1 penalty
(``mode="shrink"``, one unrolled ISTA step) or by a TopK selection per crop
(``mode="topk"``), followed by non-maximum suppression. The decoder is the tied
transposed convolution, so every atom is a waveform that reconstructs the signal.
Codes are signed on purpose: on a bipolar montage one discharge appears with
opposite polarity on the two derivations around the focus (phase reversal).

The paper's squared-norm term is available as a gate on the cosine between atom and
window (``rho_min``). It is off by default: the decoder corrects the energy bias of a
plain correlation through the residual (design note, section 3.2).

Atoms are learned without labels from random crops of the training windows. At
extraction time each window yields, per atom, the event count (NMS peaks with
``abs(a) > amp_min``, summed over channels), the peak ``abs(a)`` over channels and
time, and the best ``abs(cosine)`` (the paper's global-max feature).

Every (window, channel) row is centered on its median, scaled by its MAD and clipped
before encoding, so atoms, thresholds and amplitudes live in robust-std units.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

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

__all__ = ["AtomSpec", "ShapeConvSAE", "TrainSpec"]

_EPS = 1e-8
_ENERGY_FLOOR = 1e-6
_MAD_TO_STD = 1.4826
_CLIP = 20.0
_N_INIT_BATCHES = 8
_MODES = ("shrink", "topk")
CHECKPOINT_NAME = "sae_state.pt"


@dataclass(frozen=True)
class AtomSpec:
    """Dictionary geometry and encoder, shared by training and extraction.

    Parameters
    ----------
    n_atoms : int
        Number of atoms (shapelets) in the dictionary.
    atom_len : int
        Atom length in samples (80 = 312 ms at 256 Hz: a spike or sharp wave plus
        the start of its slow wave).
    mode : str
        ``shrink``: signed soft threshold ``sign(z) * relu(abs(z) - theta_k)`` with a
        learnable per-atom ``theta_k`` and an L1 penalty (``TrainSpec.lam``).
        ``topk``: keep the ``topk`` largest ``abs(z)`` per crop during training; at
        extraction there is no budget, so set ``amp_min`` (or ``rho_min``).
    thresh : float
        Initial ``theta_k`` in robust-std units (``shrink``).
    topk : int
        Nonzero code entries per crop (``topk``).
    rho_min : float
        Optional cosine gate between an atom and the window it covers; matches with
        ``abs(rho) < rho_min`` are discarded. 0 disables it.
    amp_min : float
        Extraction only: an NMS peak counts as an event when ``abs(a) > amp_min``.
    """

    n_atoms: int = 64
    atom_len: int = 80
    mode: str = "shrink"
    thresh: float = 3.0
    topk: int = 4
    rho_min: float = 0.0
    amp_min: float = 0.0


@dataclass(frozen=True)
class TrainSpec:
    """Unsupervised training schedule of the dictionary.

    Parameters
    ----------
    epochs : int
        Passes over the training dataloader. Every pass draws fresh random crops,
        so the update count is ``epochs * rows * crops_per_row / crop_batch``; read
        ``history`` (``sae_training.csv``) to see where the residual levels off.
    lr : float
        Adam learning rate.
    lam : float
        L1 weight in ``shrink`` mode (per-sample L1 of the code, as in the demo).
        Raise it until the events per second look plausible, not until the loss
        looks nice.
    crop_len : int
        Training crop length in samples (256 = 1 s at 256 Hz).
    crops_per_row : int
        Random crops drawn per (window, channel) row on every pass.
    crop_batch : int
        Crops per optimizer step.
    lambda_div : float
        Weight of the shift-aware diversity penalty: the mean over atom pairs of
        the maximum cross-correlation over lags. Zero disables it.
    n_init_samples : int
        Sub-sequences sampled for the k-means initialization of the atoms.
    """

    epochs: int = 30
    lr: float = 3e-3
    lam: float = 0.5
    crop_len: int = 256
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
        Dictionary geometry and encoder; ``None`` uses the defaults.
    train_spec : TrainSpec | None
        Unsupervised training schedule; ``None`` uses the defaults.
    random_state : int
        Seed for the atom initialization and the crop sampling.
    device : str | None
        ``cpu`` | ``cuda`` | ``cuda:N`` | ``auto`` (cuda when available).
    chunk_rows : int
        (window, channel) rows scored per pass at extraction time. Memory scales
        with ``chunk_rows * n_atoms * T``; 16 rows of a 2-minute window at 256 Hz
        with 64 atoms need about 0.5 GiB.
    pretrained : str | Path | None
        Path to a ``sae_state.pt`` checkpoint written by ``save_artifacts``. Its
        ``n_atoms`` / ``atom_len`` / ``mode`` must match ``spec``.

    Raises
    ------
    ValueError
        If ``spec.mode`` is not ``shrink`` or ``topk``.
    """

    def __init__(
        self,
        spec: AtomSpec | None = None,
        train_spec: TrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        chunk_rows: int = 16,
        pretrained: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec or AtomSpec()
        self.train_spec = train_spec or TrainSpec()
        if self.spec.mode not in _MODES:
            raise ValueError(f"spec.mode must be one of {_MODES}, got {self.spec.mode!r}")
        self.random_state = random_state
        self.chunk_rows = chunk_rows
        self.device = HydraTransform._resolve_device(device)
        generator = torch.Generator().manual_seed(random_state)
        init = torch.randn(self.spec.n_atoms, 1, self.spec.atom_len, generator=generator)
        self.atoms = nn.Parameter(self._project(init).to(self.device))
        self.log_thresh = nn.Parameter(
            torch.full((self.spec.n_atoms,), math.log(self.spec.thresh), device=self.device)
        )
        self.history: list[dict[str, float]] = []
        self.train_subjects: list[str] = []
        self.fitted = False
        if pretrained:
            self.load_checkpoint(pretrained)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load atoms, thresholds, history and training subjects from ``sae_state.pt``; mark fitted.

        Raises
        ------
        ValueError
            If the checkpoint's ``n_atoms``, ``atom_len`` or ``mode`` differ from ``spec``.
        """
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        saved = checkpoint["spec"]
        for key in ("n_atoms", "atom_len", "mode"):
            if saved[key] != getattr(self.spec, key):
                raise ValueError(
                    f"checkpoint {path} has {key}={saved[key]!r} but spec.{key}={getattr(self.spec, key)!r}"
                )
        self.load_state_dict(checkpoint["state_dict"])
        self.history = checkpoint["history"]
        self.train_subjects = checkpoint["train_subjects"]
        self.fitted = True
        log.info(f"Loaded {self.spec.n_atoms} pretrained atoms from {path} ({len(self.history)} epochs)")

    @property
    def atoms_numpy(self) -> np.ndarray:
        """The normalized atoms as a ``(n_atoms, atom_len)`` float32 array."""
        return self._project(self.atoms.detach()).squeeze(1).cpu().numpy()

    @property
    def thresholds(self) -> np.ndarray:
        """The per-atom soft thresholds ``theta_k`` (robust-std units), ``(n_atoms,)``."""
        return self.log_thresh.detach().exp().cpu().numpy()

    @staticmethod
    def _project(atoms: torch.Tensor) -> torch.Tensor:
        """Zero-mean, unit-L2-norm along the last axis (the dictionary constraint)."""
        atoms = atoms - atoms.mean(-1, keepdim=True)
        return atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(_EPS)

    @staticmethod
    def _robust_scale(x: torch.Tensor) -> torch.Tensor:
        """Center every row on its median, scale by 1.4826 * MAD (a robust std), clip at +-20.

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
        """Match amplitude and cosine of every atom at every valid position.

        Parameters
        ----------
        x : torch.Tensor
            Rows ``(n, 1, T)``.
        atoms : torch.Tensor
            Normalized atoms ``(n_atoms, 1, atom_len)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``amplitude = <atom, window>`` and ``rho = amplitude / ||window - mean||``,
            both ``(n, n_atoms, T - atom_len + 1)``. With unit-norm atoms the amplitude
            is the best scale ``c`` of the atom for that window, and
            ``||window - c * atom||^2 = ||window||^2 (1 - rho^2)``.
        """
        length = self.spec.atom_len
        amplitude = F.conv1d(x, atoms)
        box = torch.ones(1, 1, length, device=x.device, dtype=x.dtype)
        window_sum = F.conv1d(x, box)
        window_energy = F.conv1d(x * x, box) - window_sum * window_sum / length
        rho = amplitude / window_energy.clamp_min(_ENERGY_FLOOR).sqrt()
        return amplitude, rho

    def _peaks(self, z: torch.Tensor) -> torch.Tensor:
        """Local maxima of ``abs(z)`` per atom within +-atom_len/2 samples (non-maximum suppression)."""
        length = self.spec.atom_len
        magnitude = z.abs()
        peak = F.max_pool1d(magnitude, kernel_size=length | 1, stride=1, padding=length // 2)
        return (magnitude > 0) & (magnitude == peak)

    def _code(self, amplitude: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Signed sparse code: optional cosine gate, soft threshold (``shrink``), then NMS."""
        z = amplitude
        if self.spec.rho_min > 0:
            z = z * (rho.abs() >= self.spec.rho_min)
        if self.spec.mode == "shrink":
            theta = self.log_thresh.exp().view(1, -1, 1)
            z = torch.sign(z) * F.relu(z.abs() - theta)
        return z * self._peaks(z)

    def _sparse_code(self, crops: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Training code ``(n, n_atoms, P)``: ``_code`` plus, in ``topk`` mode, the K largest ``abs`` per crop."""
        amplitude, rho = self._scores(crops, atoms)
        z = self._code(amplitude, rho)
        if self.spec.mode != "topk":
            return z
        flat = z.flatten(1)
        index = flat.abs().topk(self.spec.topk, dim=1).indices
        return torch.zeros_like(flat).scatter(1, index, flat.gather(1, index)).view_as(z)

    def _diversity(self, atoms: torch.Tensor) -> torch.Tensor:
        """Mean over atom pairs of the maximum cross-correlation over lags (shift-aware)."""
        n_atoms = self.spec.n_atoms
        coherence = F.conv1d(atoms, atoms, padding=self.spec.atom_len - 1).amax(-1)
        off_diagonal = ~torch.eye(n_atoms, dtype=torch.bool, device=atoms.device)
        return coherence[off_diagonal].mean()

    def _step(
        self, crops: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loss of one crop batch, its ``[residual_energy, signal_energy, n_active, n_crops]``, and per-atom usage."""
        z = self._sparse_code(crops, atoms)
        residual = crops - F.conv_transpose1d(z, atoms)
        loss = 0.5 * residual.pow(2).mean()
        if self.spec.mode == "shrink":
            loss = loss + self.train_spec.lam * z.abs().sum(dim=(1, 2)).mean() / crops.shape[-1]
        if self.train_spec.lambda_div > 0:
            loss = loss + self.train_spec.lambda_div * self._diversity(atoms)
        active = z.detach() != 0
        stats = torch.stack(
            [
                residual.detach().pow(2).sum(),
                crops.pow(2).sum(),
                active.sum().to(crops.dtype),
                torch.tensor(float(crops.shape[0]), device=crops.device),
            ]
        )
        return loss, stats, active.sum(dim=(0, 2))

    def _init_atoms(self, dataloader: DataLoader, generator: torch.Generator) -> None:
        """K-means on random unit-norm sub-sequences of the first batches (paper's init, no time cuts)."""
        spec, train_spec = self.spec, self.train_spec
        per_batch = max(train_spec.n_init_samples // _N_INIT_BATCHES, 1)
        samples = []
        for batch_index, (x, _) in enumerate(dataloader):
            rows = self._robust_scale(x.to(self.device)).flatten(0, 1)
            samples.append(self._sample_subsequences(rows, spec.atom_len, per_batch, generator))
            if batch_index + 1 >= _N_INIT_BATCHES:
                break
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
    def _reset_dead_atoms(
        self, usage: torch.Tensor, crops: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> int:
        """Re-seed atoms that never fired in the epoch on the worst-reconstructed patches of ``crops``.

        The re-seeded atoms get the median threshold and cleared Adam moments.
        """
        dead = torch.nonzero(usage == 0).flatten()
        if dead.numel() == 0:
            return 0
        length = self.spec.atom_len
        atoms = self._project(self.atoms)
        residual = (crops - F.conv_transpose1d(self._sparse_code(crops, atoms), atoms)).squeeze(1)
        energy = F.avg_pool1d(residual.pow(2).unsqueeze(1), length, stride=1).squeeze(1)
        worst = energy.flatten().topk(min(dead.numel(), energy.numel())).indices
        patches = residual.unfold(1, length, 1)[worst // energy.shape[1], worst % energy.shape[1]]
        self.atoms[dead[: len(patches)]] = self._project(patches).unsqueeze(1)
        self.log_thresh[dead] = self.log_thresh.median()
        for param in (self.atoms, self.log_thresh):
            state = optimizer.state.get(param)
            if state:
                state["exp_avg"][dead] = 0.0
                state["exp_avg_sq"][dead] = 0.0
        return int(dead.numel())

    def _crops(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Random zero-mean crops ``(rows * crops_per_row, 1, crop_len)`` of a window batch."""
        train_spec = self.train_spec
        rows = self._robust_scale(x.to(self.device)).flatten(0, 1)
        if rows.shape[1] < train_spec.crop_len:
            raise ValueError(f"windows of {rows.shape[1]} samples are shorter than crop_len")
        n_crops = rows.shape[0] * train_spec.crops_per_row
        crops = self._sample_subsequences(rows, train_spec.crop_len, n_crops, generator)
        return (crops - crops.mean(-1, keepdim=True)).unsqueeze(1)

    @torch.no_grad()
    def _val_stats(self, val_dataloader: DataLoader) -> torch.Tensor:
        """Summed ``[residual_energy, signal_energy, n_active, n_crops]`` on fixed crops of the validation windows."""
        generator = torch.Generator().manual_seed(self.random_state + 1)
        atoms = self._project(self.atoms)
        totals = torch.zeros(4, device=self.device)
        for x, _ in val_dataloader:
            crops = self._crops(x, generator)
            for start in range(0, crops.shape[0], self.train_spec.crop_batch):
                _, stats, _ = self._step(crops[start : start + self.train_spec.crop_batch], atoms)
                totals += stats
        return totals

    def fit_unsupervised(self, dataloader: DataLoader, val_dataloader: DataLoader | None = None) -> None:
        """Learn the atoms by sparse reconstruction of random crops of the windows.

        Every dataloader batch ``(X, y)`` is scaled per (window, channel) row, cut
        into ``crops_per_row`` random crops per row, and consumed in optimizer steps
        of ``crop_batch`` crops. Labels ``y`` are ignored. Atoms are re-projected to
        zero mean and unit norm after every step. After each epoch, atoms that never
        entered a code are re-seeded on the worst residuals. ``history`` records, per
        epoch, the residual as a fraction of signal power (train and, if given,
        validation), the nonzero code entries per crop, the dead-atom count and the
        step count. A module loaded from a checkpoint (``fitted``) returns at once.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(X, y)`` with ``X`` of shape ``(B, C, T)``.
        val_dataloader : DataLoader | None
            Held-out windows scored after every epoch (monitoring only; nothing
            is selected on them).

        Raises
        ------
        ValueError
            If ``crop_len`` is shorter than ``atom_len`` or longer than the windows.
        """
        if self.fitted:
            log.info("ShapeConv SAE already fitted (pretrained checkpoint); skipping fit_unsupervised")
            return
        spec, train_spec = self.spec, self.train_spec
        if train_spec.crop_len < spec.atom_len:
            raise ValueError(f"crop_len={train_spec.crop_len} must be >= atom_len={spec.atom_len}")
        generator = torch.Generator().manual_seed(self.random_state)
        self._init_atoms(dataloader, generator)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_spec.lr)
        self.history = []
        batch = self.atoms.detach()
        for epoch in range(train_spec.epochs):
            usage = torch.zeros(spec.n_atoms, dtype=torch.long, device=self.device)
            totals = torch.zeros(4, device=self.device)
            n_steps = 0
            for x, _ in tqdm(dataloader, desc=f"ShapeConv SAE epoch {epoch + 1}/{train_spec.epochs}"):
                crops = self._crops(x, generator)
                order = torch.randperm(crops.shape[0], generator=generator).to(self.device)
                for start in range(0, crops.shape[0], train_spec.crop_batch):
                    batch = crops[order[start : start + train_spec.crop_batch]]
                    loss, stats, used = self._step(batch, self._project(self.atoms))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        self.atoms.copy_(self._project(self.atoms))
                    usage += used
                    totals += stats
                    n_steps += 1
            n_dead = self._reset_dead_atoms(usage, batch, optimizer)
            val = self._val_stats(val_dataloader) if val_dataloader is not None else None
            record = {
                "epoch": epoch + 1,
                "residual_frac": float(totals[0] / totals[1].clamp_min(_EPS)),
                "active_per_crop": float(totals[2] / totals[3].clamp_min(1.0)),
                "val_residual_frac": float(val[0] / val[1].clamp_min(_EPS)) if val is not None else float("nan"),
                "n_dead": n_dead,
                "n_steps": n_steps,
            }
            self.history.append(record)
            log.info(
                f"ShapeConv SAE epoch {epoch + 1}/{train_spec.epochs}: residual {record['residual_frac']:.1%} "
                f"of signal power (val {record['val_residual_frac']:.1%}), "
                f"activations/crop {record['active_per_crop']:.2f}, resampled {n_dead}, steps {n_steps}"
            )
        self.fitted = True

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
            ``(B, 3 * n_atoms)`` float32, three blocks of ``n_atoms`` columns: event
            counts (NMS peaks with ``abs(a) > amp_min``, summed over channels and
            time), peak ``abs(a)`` over channels and time, and the best ``abs(rho)``
            (cosine) over channels and time.
        """
        n_windows, n_channels, n_times = X.shape
        rows = self._robust_scale(X.to(self.device)).reshape(n_windows * n_channels, 1, n_times)
        atoms = self._project(self.atoms)
        counts, peaks, best = [], [], []
        for start in range(0, rows.shape[0], self.chunk_rows):
            amplitude, rho = self._scores(rows[start : start + self.chunk_rows], atoms)
            magnitude = self._code(amplitude, rho).abs()
            counts.append((magnitude > self.spec.amp_min).sum(-1))
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

    def save_artifacts(self, output_dir: Path) -> None:
        """Write ``sae_state.pt`` (reloadable checkpoint), ``sae_atoms.npy``, and ``sae_training.csv``."""
        output_dir = Path(output_dir)
        torch.save(
            {
                "spec": asdict(self.spec),
                "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
                "history": self.history,
                "train_subjects": list(self.train_subjects),
            },
            output_dir / CHECKPOINT_NAME,
        )
        np.save(output_dir / "sae_atoms.npy", self.atoms_numpy)
        if self.history:
            pl.DataFrame(self.history).write_csv(output_dir / "sae_training.csv")
        log.info(f"Saved {self.spec.n_atoms} atoms and {len(self.history)} training epochs to {output_dir}")
