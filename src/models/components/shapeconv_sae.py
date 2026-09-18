"""ShapeConv sparse autoencoder: a learned shapelet dictionary as window features.

The encoder is the ShapeConv layer of Qu et al. (ICLR 2024, "CNN kernels can be the
best shapelets") with a free match amplitude: a 1-D cross-correlation with
zero-mean, unit-norm atoms, gated by the cosine between each atom and the window it
covers (the paper's squared-norm term, used as a gate instead of a subtraction). The
decoder is the tied transposed convolution, so every atom is a waveform that
reconstructs the signal. Atoms are learned without labels from random crops of the
training windows with a TopK sparse code. At extraction time each window yields, per
atom, the number of gated local maxima summed over channels (event count) and the
best cosine match over channels and time (the paper's global-max feature).

One dictionary is shared by all channels (channels are folded into the batch).
Every (window, channel) row is centered on its median and scaled by its median
absolute deviation before encoding, so atoms and thresholds live in robust-std units.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_N_INIT_BATCHES = 8


@dataclass(frozen=True)
class AtomSpec:
    """Dictionary geometry and the event gate, shared by training and extraction.

    Parameters
    ----------
    n_atoms : int
        Number of atoms (shapelets) in the dictionary.
    atom_len : int
        Atom length in samples (128 = 0.5 s at 256 Hz).
    rho_min : float
        Cosine threshold between an atom and the window it covers. Matches below
        it are discarded, so an atom only fires where the signal has its shape.
    amp_min : float
        Minimum match amplitude (robust-std units) for an event to be counted at
        extraction time. Ignored during training, where TopK selects the code.
    """

    n_atoms: int = 256
    atom_len: int = 128
    rho_min: float = 0.5
    amp_min: float = 0.0


@dataclass(frozen=True)
class TrainSpec:
    """Unsupervised training schedule of the dictionary.

    Parameters
    ----------
    epochs : int
        Passes over the training dataloader.
    lr : float
        Adam learning rate.
    topk : int
        Nonzero code entries kept per crop (the sparsity level).
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

    epochs: int = 5
    lr: float = 1e-3
    topk: int = 4
    crop_len: int = 256
    crops_per_row: int = 16
    crop_batch: int = 1024
    lambda_div: float = 0.01
    n_init_samples: int = 20000


class ShapeConvSAE(nn.Module):
    """Learned shapelet dictionary (ShapeConv encoder, tied decoder) as a feature extractor.

    Plugs into the sklearn-style ``Trainer`` like ``HydraTransformer``: ``forward``
    maps a window batch ``(B, C, T)`` to features ``(B, 2 * n_atoms)``. Unlike HYDRA
    the atoms are learned, so the trainer first calls ``fit_unsupervised`` on the
    training dataloader (labels are never read).

    Parameters
    ----------
    spec : AtomSpec | None
        Dictionary geometry and event gate; ``None`` uses the defaults.
    train_spec : TrainSpec | None
        Unsupervised training schedule; ``None`` uses the defaults.
    random_state : int
        Seed for the atom initialization, the crop sampling, and the dead-atom resets.
    device : str | None
        ``cpu`` | ``cuda`` | ``cuda:N`` | ``auto`` (cuda when available).
    chunk_rows : int
        (window, channel) rows scored per pass at extraction time. Memory scales
        with ``chunk_rows * n_atoms * T``; 16 rows of a 2-minute window at 256 Hz
        with 256 atoms need about 2 GiB.
    """

    def __init__(
        self,
        spec: AtomSpec | None = None,
        train_spec: TrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        chunk_rows: int = 16,
    ) -> None:
        super().__init__()
        self.spec = spec or AtomSpec()
        self.train_spec = train_spec or TrainSpec()
        self.random_state = random_state
        self.chunk_rows = chunk_rows
        self.device = HydraTransform._resolve_device(device)
        generator = torch.Generator().manual_seed(random_state)
        init = torch.randn(self.spec.n_atoms, 1, self.spec.atom_len, generator=generator)
        self.atoms = nn.Parameter(self._project(init).to(self.device))
        self.history: list[dict[str, float]] = []

    @property
    def atoms_numpy(self) -> np.ndarray:
        """The normalized atoms as a ``(n_atoms, atom_len)`` float32 array."""
        return self._project(self.atoms.detach()).squeeze(1).cpu().numpy()

    @staticmethod
    def _project(atoms: torch.Tensor) -> torch.Tensor:
        """Zero-mean, unit-L2-norm along the last axis (the dictionary constraint)."""
        atoms = atoms - atoms.mean(-1, keepdim=True)
        return atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(_EPS)

    @staticmethod
    def _robust_scale(x: torch.Tensor) -> torch.Tensor:
        """Center every row on its median and scale by 1.4826 * MAD (a robust std).

        A floor of 1 percent of the plain std keeps a mostly-flat row from
        exploding; an all-constant row maps to zeros.
        """
        median = x.median(dim=-1, keepdim=True).values
        centered = x - median
        mad = centered.abs().median(dim=-1, keepdim=True).values * _MAD_TO_STD
        scale = torch.maximum(mad, 0.01 * centered.std(dim=-1, keepdim=True))
        return centered / scale.clamp_min(_EPS)

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
            both ``(n, n_atoms, T - atom_len + 1)``. Zero-mean atoms make ``rho`` the
            cosine with the de-meaned window, i.e. the ShapeConv distance with a free
            amplitude: ``||window - c * atom||^2 = ||window||^2 (1 - rho^2)`` at the best ``c``.
        """
        length = self.spec.atom_len
        amplitude = F.conv1d(x, atoms)
        box = torch.ones(1, 1, length, device=x.device, dtype=x.dtype)
        window_sum = F.conv1d(x, box)
        window_energy = F.conv1d(x * x, box) - window_sum * window_sum / length
        rho = amplitude / window_energy.clamp_min(_ENERGY_FLOOR).sqrt()
        return amplitude, rho

    def _gate(self, amplitude: torch.Tensor, rho: torch.Tensor, amp_min: float) -> torch.Tensor:
        """Keep positive matches whose shape agrees with the atom (``rho >= rho_min``)."""
        keep = (rho >= self.spec.rho_min) & (amplitude > max(amp_min, 0.0))
        return amplitude * keep

    def _peaks(self, z: torch.Tensor) -> torch.Tensor:
        """Local maxima of ``z`` per atom within +-atom_len/2 samples (non-max suppression)."""
        length = self.spec.atom_len
        peak = F.max_pool1d(z, kernel_size=length | 1, stride=1, padding=length // 2)
        return (z > 0) & (z == peak)

    def _sparse_code(self, crops: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """TopK code ``(n, n_atoms, P)`` of the crops: gated, peak-picked, then the K largest per crop."""
        amplitude, rho = self._scores(crops, atoms)
        z = self._gate(amplitude, rho, 0.0)
        z = z * self._peaks(z)
        flat = z.flatten(1)
        values, index = flat.topk(self.train_spec.topk, dim=1)
        return torch.zeros_like(flat).scatter(1, index, values).view_as(z)

    def _diversity(self, atoms: torch.Tensor) -> torch.Tensor:
        """Mean over atom pairs of the maximum cross-correlation over lags (shift-aware)."""
        n_atoms = self.spec.n_atoms
        coherence = F.conv1d(atoms, atoms, padding=self.spec.atom_len - 1).amax(-1)
        off_diagonal = ~torch.eye(n_atoms, dtype=torch.bool, device=atoms.device)
        return coherence[off_diagonal].mean()

    def _step(
        self, crops: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruction (+ diversity) loss of one crop batch and the per-atom usage."""
        z = self._sparse_code(crops, atoms)
        reconstruction = F.conv_transpose1d(z, atoms)
        loss_rec = F.mse_loss(reconstruction, crops)
        loss = loss_rec
        if self.train_spec.lambda_div > 0:
            loss = loss + self.train_spec.lambda_div * self._diversity(atoms)
        usage = (z.detach() > 0).sum(dim=(0, 2))
        return loss, loss_rec.detach(), usage

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

    def _reset_dead_atoms(
        self,
        usage: torch.Tensor,
        crops: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> int:
        """Re-seed atoms that never fired in the epoch with random data sub-sequences."""
        dead = torch.nonzero(usage == 0).flatten()
        if dead.numel() == 0:
            return 0
        rows = crops.squeeze(1)
        fresh = self._sample_subsequences(rows, self.spec.atom_len, dead.numel(), generator)
        with torch.no_grad():
            self.atoms[dead] = self._project(fresh).unsqueeze(1)
            state = optimizer.state.get(self.atoms)
            if state:
                state["exp_avg"][dead] = 0.0
                state["exp_avg_sq"][dead] = 0.0
        return int(dead.numel())

    def fit_unsupervised(self, dataloader: DataLoader) -> None:
        """Learn the atoms by sparse reconstruction of random crops of the windows.

        Every dataloader batch ``(X, y)`` is scaled per (window, channel) row, cut
        into ``crops_per_row`` random crops per row, and consumed in optimizer steps
        of ``crop_batch`` crops. Labels ``y`` are ignored. After each epoch, atoms
        that never entered a code are re-seeded from data. Per-epoch reconstruction
        loss and dead-atom counts are kept in ``history``.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(X, y)`` with ``X`` of shape ``(B, C, T)``.

        Raises
        ------
        ValueError
            If ``crop_len`` is shorter than ``atom_len`` or longer than the windows.
        """
        spec, train_spec = self.spec, self.train_spec
        if train_spec.crop_len < spec.atom_len:
            raise ValueError(f"crop_len={train_spec.crop_len} must be >= atom_len={spec.atom_len}")
        generator = torch.Generator().manual_seed(self.random_state)
        self._init_atoms(dataloader, generator)
        optimizer = torch.optim.Adam([self.atoms], lr=train_spec.lr)
        self.history = []
        crops = self.atoms.detach()
        for epoch in range(train_spec.epochs):
            usage = torch.zeros(spec.n_atoms, dtype=torch.long, device=self.device)
            loss_sum, n_steps = 0.0, 0
            for x, _ in tqdm(dataloader, desc=f"ShapeConv SAE epoch {epoch + 1}/{train_spec.epochs}"):
                rows = self._robust_scale(x.to(self.device)).flatten(0, 1)
                if rows.shape[1] < train_spec.crop_len:
                    raise ValueError(f"windows of {rows.shape[1]} samples are shorter than crop_len")
                n_crops = rows.shape[0] * train_spec.crops_per_row
                crops = self._sample_subsequences(rows, train_spec.crop_len, n_crops, generator)
                crops = (crops - crops.mean(-1, keepdim=True)).unsqueeze(1)
                order = torch.randperm(n_crops, generator=generator).to(self.device)
                for start in range(0, n_crops, train_spec.crop_batch):
                    batch = crops[order[start:start + train_spec.crop_batch]]
                    loss, loss_rec, used = self._step(batch, self._project(self.atoms))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        self.atoms.copy_(self._project(self.atoms))
                    usage += used
                    loss_sum += float(loss_rec)
                    n_steps += 1
            n_dead = self._reset_dead_atoms(usage, crops, optimizer, generator)
            record = {
                "epoch": epoch + 1,
                "loss_rec": loss_sum / max(n_steps, 1),
                "n_dead": n_dead,
                "n_steps": n_steps,
            }
            self.history.append(record)
            log.info(
                f"ShapeConv SAE epoch {epoch + 1}/{train_spec.epochs}: "
                f"loss_rec={record['loss_rec']:.4f} dead_atoms={n_dead} steps={n_steps}"
            )

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
            ``(B, 2 * n_atoms)`` float32: columns ``[0, n_atoms)`` are event counts
            per atom (gated local maxima above ``amp_min``, summed over channels and
            time); columns ``[n_atoms, 2 n_atoms)`` are the best cosine per atom over
            channels and time, clamped at zero.
        """
        n_windows, n_channels, n_times = X.shape
        rows = self._robust_scale(X.to(self.device)).reshape(n_windows * n_channels, 1, n_times)
        atoms = self._project(self.atoms)
        counts, best = [], []
        for start in range(0, rows.shape[0], self.chunk_rows):
            amplitude, rho = self._scores(rows[start:start + self.chunk_rows], atoms)
            z = self._gate(amplitude, rho, self.spec.amp_min)
            counts.append(self._peaks(z).sum(-1))
            best.append(rho.amax(-1))
        n_atoms = self.spec.n_atoms
        counts_cat = torch.cat(counts).view(n_windows, n_channels, n_atoms).sum(1).float()
        best_cat = torch.cat(best).view(n_windows, n_channels, n_atoms).amax(1).clamp_min(0.0)
        return torch.cat([counts_cat, best_cat], dim=1)

    def save_artifacts(self, output_dir: Path) -> None:
        """Write the atoms (``sae_atoms.npy``) and the training history (``sae_training.csv``)."""
        output_dir = Path(output_dir)
        np.save(output_dir / "sae_atoms.npy", self.atoms_numpy)
        if self.history:
            pl.DataFrame(self.history).write_csv(output_dir / "sae_training.csv")
        log.info(f"Saved {self.spec.n_atoms} atoms and {len(self.history)} training epochs to {output_dir}")
