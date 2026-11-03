"""Pure logic of the reconstruction baselines (``documentation/separable_reconstruction_baselines_protocol.md``).

No corpus: the fixed atom bank, the spatial patterns, the uncentered window PCA, the rank-1 ceiling, the band split
by FFT, the scoring of one batch of rows by every arm, the pooled explained energy with its patient bootstrap, the
gates and the rule. The tensor functions run on CPU or GPU; the statistics are numpy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SFREQ = 256.0
POOL_LEN = 512  # P: one window of the R=1 code, 2 s
ATOM_LEN = 256  # L
N_ATOMS = 64  # K
N_SPATIAL = 4  # spatial patterns of the fixed atoms, equal to D
TOP_R = 1

# The fixed bank (protocol section 4.1): 32 frequencies, evenly spaced on a log scale, each with both signs.
BANK_FREQUENCIES = 32
BANK_LOW_HZ, BANK_HIGH_HZ = 0.5, 45.0
BANK_CYCLES = 3.0  # sigma = cycles / (2 pi f), at most L / (6 sfreq)

PCA_KS = (1, 2, 4, 8, 16, 32, 64)
BANDS = ((0.0, 1.0), (1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, math.inf))
BAND_NAMES = ("below_1", "1_4", "4_8", "8_13", "13_30", "30_nyquist")
FAST_FROM_HZ = 4.0  # "above 4 Hz": the union of the bands that start at or above this frequency
SLOW_ATOM_HZ = 4.0  # an atom is slow when its dominant frequency is below this

S4_ARMS = ("S4-trained", "S4-untrained", "S4-fixed")
ARMS = (*S4_ARMS, *(f"PCA-{k}" for k in PCA_KS), "SVD-1")

FAST_FLOOR = 0.05  # H1: the least EV above 4 Hz of S4-trained
ROW_MATCH_TOLERANCE = 0.05  # gate 3: held-out EV against the dictionary's validation EV
CEILING_TOLERANCE = 1e-5  # gate 2: relative to the window energy
PARSEVAL_TOLERANCE = 1e-6  # gate 5: relative to the segment energy
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20261103
_EPS = 1e-12


# ---------------------------------------------------------------------- fixed atoms


def bank_frequencies(n: int = BANK_FREQUENCIES, low: float = BANK_LOW_HZ, high: float = BANK_HIGH_HZ) -> np.ndarray:
    """Center frequencies ``(n,)`` in Hz, evenly spaced on a log scale from ``low`` to ``high``."""
    return np.geomspace(low, high, n)


def gabor_bank(
    length: int = ATOM_LEN,
    sfreq: float = SFREQ,
    frequencies: Sequence[float] | None = None,
    cycles: float = BANK_CYCLES,
) -> np.ndarray:
    """Unit waveforms ``(2F, L)``: for each frequency, a cosine under a Gaussian envelope, then its negative.

    The envelope is centered in the support, with ``sigma = min(cycles / (2 pi f), L / (6 sfreq))`` seconds, so that
    about ``cycles`` cycles fit under it and ``+-3 sigma`` fits in the support.
    """
    f = bank_frequencies() if frequencies is None else np.asarray(frequencies, dtype=float)
    t = (np.arange(length) - (length - 1) / 2) / sfreq
    sigma = np.minimum(cycles / (2 * np.pi * f), length / (6 * sfreq))
    waves = np.exp(-(t[None] ** 2) / (2 * sigma[:, None] ** 2)) * np.cos(2 * np.pi * f[:, None] * t[None])
    waves /= np.linalg.norm(waves, axis=1, keepdims=True)
    return np.stack([waves, -waves], axis=1).reshape(-1, length)


def top_components(second_moment: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``n`` leading eigenvectors ``(D, n)`` of a symmetric matrix and all its eigenvalues, both descending.

    The largest entry in magnitude of every vector is positive, so the result does not depend on the solver's sign.

    Raises
    ------
    ValueError
        If the matrix is not square or has fewer than ``n`` rows.
    """
    if second_moment.ndim != 2 or second_moment.shape[0] != second_moment.shape[1] or second_moment.shape[0] < n:
        raise ValueError(f"expected a square matrix with at least {n} rows, got {tuple(second_moment.shape)}")
    values, vectors = torch.linalg.eigh(second_moment)
    values, vectors = values.flip(0), vectors.flip(1)[:, :n]
    lead = vectors.gather(0, vectors.abs().argmax(dim=0, keepdim=True))
    return vectors * torch.where(lead < 0, -1.0, 1.0).to(vectors), values


def set_fixed_atoms(model: Any, waveforms: np.ndarray, patterns: np.ndarray) -> None:
    """Replace the atoms of a ``SeparableSAE`` with waveforms ``(K, L)`` that share spatial patterns ``(C, D)``.

    The atoms go into the packed parameter with the model's own storage scaling, so ``_dictionary`` gives back these
    unit factors.

    Raises
    ------
    ValueError
        If a shape differs from the model's geometry or a factor is not unit-norm.
    """
    spec = model.spec
    n_channels = model.n_channels
    expected = ((spec.n_atoms, spec.atom_len), (n_channels, spec.n_projections))
    if (waveforms.shape, patterns.shape) != expected:
        raise ValueError(f"expected waveforms {expected[0]} and patterns {expected[1]}")
    for name, factor, axis in (("waveforms", waveforms, 1), ("patterns", patterns, 0)):
        if not np.allclose(np.linalg.norm(factor, axis=axis), 1.0, atol=1e-6):
            raise ValueError(f"the {name} must have unit norm")
    device, dtype = model.atoms.device, model.atoms.dtype
    time = torch.as_tensor(waveforms, dtype=dtype, device=device)
    channel = torch.as_tensor(patterns, dtype=dtype, device=device).expand(spec.n_atoms, -1, -1)
    model.atoms = torch.nn.Parameter(model._pack(time, channel).contiguous(), requires_grad=False)


# ---------------------------------------------------------------------- windows, PCA and the ceiling


def windows(rows: torch.Tensor, pool: int = POOL_LEN) -> torch.Tensor:
    """Pooling windows ``(B, M, C, P)`` of rows ``(B, C, M*P)``.

    Raises
    ------
    ValueError
        If the rows are not 3-D or their length is not a positive multiple of ``pool``.
    """
    if rows.ndim != 3 or rows.shape[-1] == 0 or rows.shape[-1] % pool:
        raise ValueError(f"expected rows (B, C, M*{pool}), got {tuple(rows.shape)}")
    return rows.unflatten(-1, (-1, pool)).transpose(1, 2)


def unwindow(parts: torch.Tensor) -> torch.Tensor:
    """Rows ``(B, C, M*P)`` of windows ``(B, M, C, P)``: ``windows`` inverted."""
    return parts.transpose(1, 2).flatten(-2)


def accumulate_moments(
    rows: torch.Tensor, window_moment: torch.Tensor, channel_moment: torch.Tensor, pool: int = POOL_LEN
) -> tuple[int, int]:
    """Add the sums of ``x x^T`` over windows ``(C*P, C*P)`` and over samples ``(C, C)`` in place; float64.

    Returns the number of windows and of samples added.
    """
    flat = windows(rows, pool).flatten(0, 1).flatten(1).to(window_moment.dtype)
    window_moment.addmm_(flat.T, flat)
    channel = rows.to(channel_moment.dtype)
    channel_moment += torch.einsum("bct,bdt->cd", channel, channel)
    return int(flat.shape[0]), int(rows.shape[0] * rows.shape[-1])


def pca_reconstructions(
    rows: torch.Tensor, basis: torch.Tensor, ks: Sequence[int] = PCA_KS, pool: int = POOL_LEN
) -> dict[int, torch.Tensor]:
    """``PCA-k`` reconstructions ``(B, C, T)``: each window projected on the first ``k`` columns of ``basis``.

    Raises
    ------
    ValueError
        If ``basis`` is not ``(C*P, n)`` with ``n`` at least the largest ``k``.
    """
    parts = windows(rows, pool)
    flat = parts.flatten(2)
    if basis.ndim != 2 or basis.shape[0] != flat.shape[-1] or basis.shape[1] < max(ks):
        raise ValueError(f"expected a basis ({flat.shape[-1]}, >= {max(ks)}), got {tuple(basis.shape)}")
    basis = basis.to(flat)
    coefficients = flat @ basis
    return {k: unwindow((coefficients[..., :k] @ basis[:, :k].T).view(parts.shape)) for k in ks}


def rank1_reconstruction(rows: torch.Tensor, pool: int = POOL_LEN) -> torch.Tensor:
    """The best rank-1 approximation ``u1 u1^T X`` of every window ``X`` (Eckart and Young), as rows ``(B, C, T)``.

    ``u1`` is the leading eigenvector of ``X X^T``, computed in float64.
    """
    parts = windows(rows, pool).double()
    gram = parts @ parts.transpose(-1, -2)
    lead = torch.linalg.eigh(gram)[1][..., -1:]
    return unwindow(lead @ (lead.transpose(-1, -2) @ parts)).to(rows.dtype)


# ---------------------------------------------------------------------- energies


def band_weights(n_times: int, sfreq: float = SFREQ, bands: Sequence[tuple[float, float]] = BANDS) -> torch.Tensor:
    """Parseval weights ``(n_bands, n_times // 2 + 1)`` of the real FFT bins.

    ``sum_f w[b, f] |X_f|^2`` is the energy of band ``b`` of a row, and the bands add up to the energy of the row.

    Raises
    ------
    ValueError
        If a bin falls in no band or in two bands.
    """
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    scale = np.full(freqs.shape, 2.0 / n_times)
    scale[0] = 1.0 / n_times
    if n_times % 2 == 0:
        scale[-1] = 1.0 / n_times
    member = np.stack([(freqs >= low) & (freqs < high) for low, high in bands])
    if not (member.sum(axis=0) == 1).all():
        raise ValueError("every FFT bin must fall in exactly one band")
    return torch.from_numpy(member * scale)


def band_energy(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Energy per segment and band ``(B, n_bands)`` of rows ``(B, C, T)``, summed over channels, in float64."""
    # MKL, the FFT of the HPC build, refuses views with zero strides (an expanded tensor), so copy first.
    power = torch.fft.rfft(x.double().contiguous(), dim=-1).abs().pow(2).sum(dim=1)
    return power @ weights.to(power).T


def window_energy(x: torch.Tensor, pool: int = POOL_LEN) -> torch.Tensor:
    """Energy per window ``(B, M)`` of rows ``(B, C, M*P)``, in float64."""
    return windows(x, pool).double().pow(2).sum(dim=(-2, -1))


def fast_bands(bands: Sequence[tuple[float, float]] = BANDS, start: float = FAST_FROM_HZ) -> list[int]:
    """Indices of the bands that make up "above 4 Hz"."""
    return [i for i, (low, _) in enumerate(bands) if low >= start]


@dataclass
class BatchScore:
    """Energies of one batch of segments ``(B,)`` under every arm of ``ARMS``, all float64 on the rows' device.

    ``residual_bands`` is ``(n_arms, B, n_bands)`` and ``residual_total`` is ``(n_arms, B)``. The ceiling fields are
    per S4 arm ``(3,)``. The budget fields read the code of ``S4-trained``: windows with a winner, windows won by a
    slow atom, the explained energy of all windows and of the windows that slow atoms won, and wins per atom ``(K,)``.
    """

    signal_bands: torch.Tensor
    signal_total: torch.Tensor
    residual_bands: torch.Tensor
    residual_total: torch.Tensor
    ceiling_violations: torch.Tensor
    ceiling_excess: torch.Tensor
    usable_rows: torch.Tensor
    windows_won: torch.Tensor
    windows_slow: torch.Tensor
    explained_all: torch.Tensor
    explained_slow: torch.Tensor
    atom_wins: torch.Tensor


@torch.no_grad()
def score_batch(
    rows: torch.Tensor,
    valid: torch.Tensor,
    models: Mapping[str, Any],
    basis: torch.Tensor,
    weights: torch.Tensor,
    slow_atoms: torch.Tensor,
    pool: int = POOL_LEN,
) -> BatchScore:
    """Reconstruct encoded rows ``(B, C, T)`` with every arm and return their energies.

    ``models`` maps each of ``S4_ARMS`` to a ``SeparableSAE``; ``valid`` ``(B, C)`` is the usable-row mask of the
    rows, and every reconstruction is zero on a rejected row. ``slow_atoms`` ``(K,)`` marks the slow atoms of
    ``S4-trained``.
    """
    mask = valid.unsqueeze(-1).to(rows.dtype)
    reconstructions: dict[str, torch.Tensor] = {}
    code = None
    for name in S4_ARMS:
        model = models[name]
        selected_code, _, reconstruction, _, _ = model._encode_reconstruct(rows, valid, model._dictionary())
        reconstructions[name] = reconstruction * mask
        if name == "S4-trained":
            code = selected_code
    for k, reconstruction in pca_reconstructions(rows, basis, PCA_KS, pool).items():
        reconstructions[f"PCA-{k}"] = reconstruction * mask
    reconstructions["SVD-1"] = rank1_reconstruction(rows, pool) * mask
    bands, totals, per_window = [], [], {}
    for name in ARMS:
        residual = rows - reconstructions[name]
        bands.append(band_energy(residual, weights))
        totals.append(residual.double().pow(2).sum(dim=(1, 2)))
        per_window[name] = window_energy(residual, pool)
    signal_windows = window_energy(rows, pool)
    gaps = torch.stack([per_window["SVD-1"] - per_window[name] for name in S4_ARMS])
    values = code.unflatten(-1, (-1, pool)).amax(dim=-1)
    won = values > 0
    winner = won.any(dim=1)
    slow = slow_atoms.to(rows.device)[values.argmax(dim=1)] & winner
    explained = signal_windows - per_window["S4-trained"]
    return BatchScore(
        signal_bands=band_energy(rows, weights),
        signal_total=rows.double().pow(2).sum(dim=(1, 2)),
        residual_bands=torch.stack(bands),
        residual_total=torch.stack(totals),
        ceiling_violations=(gaps > CEILING_TOLERANCE * signal_windows).sum(dim=(1, 2)),
        ceiling_excess=(gaps / signal_windows.clamp_min(_EPS)).amax(dim=(1, 2)),
        usable_rows=valid.sum(dim=-1),
        windows_won=winner.sum(dim=-1),
        windows_slow=slow.sum(dim=-1),
        explained_all=explained.sum(dim=-1),
        explained_slow=(explained * slow).sum(dim=-1),
        atom_wins=won.sum(dim=(0, 2)),
    )


# ---------------------------------------------------------------------- statistics


@dataclass(frozen=True)
class Interval:
    """A point value and its percentile interval."""

    value: float
    low: float
    high: float


def bootstrap_counts(folds: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Draw counts ``(n_boot, n_patients)``: each draw resamples the patients of every fold, keeping fold sizes."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, len(folds)), dtype=np.int64)
    draws = np.arange(n_boot)[:, None]
    for fold in np.unique(folds):
        members = np.flatnonzero(folds == fold)
        picks = members[rng.integers(0, len(members), size=(n_boot, len(members)))]
        np.add.at(counts, (np.broadcast_to(draws, picks.shape), picks), 1)
    return counts


def pooled_ev(residual: np.ndarray, signal: np.ndarray, counts: np.ndarray | None = None) -> Any:
    """``1 - sum residual / sum signal`` over patients ``(n,)``; with ``counts`` ``(n_boot, n)``, one value per draw."""
    if counts is None:
        return float(1.0 - residual.sum() / max(float(signal.sum()), _EPS))
    return 1.0 - (counts @ residual) / np.maximum(counts @ signal, _EPS)


def _interval(value: float, draws: np.ndarray, confidence: float) -> Interval:
    tail = (1.0 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return Interval(float(value), float(low), float(high))


def ev_interval(
    residual: np.ndarray, signal: np.ndarray, counts: np.ndarray, confidence: float = CONFIDENCE
) -> Interval:
    """The pooled EV of one arm and its bootstrap interval."""
    return _interval(pooled_ev(residual, signal), pooled_ev(residual, signal, counts), confidence)


def paired_interval(
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    signal: np.ndarray,
    counts: np.ndarray,
    confidence: float = CONFIDENCE,
) -> Interval:
    """EV of arm ``a`` minus EV of arm ``b`` on the same patients and draws, with its interval."""
    value = pooled_ev(residual_a, signal) - pooled_ev(residual_b, signal)
    draws = pooled_ev(residual_a, signal, counts) - pooled_ev(residual_b, signal, counts)
    return _interval(value, draws, confidence)


def parseval_error(bands: np.ndarray, total: np.ndarray) -> float:
    """Largest relative gap between the band sum ``(..., n_bands)`` and the time-domain energy ``(...)``."""
    return float((np.abs(bands.sum(axis=-1) - total) / np.maximum(total, _EPS)).max())


def smallest_k(curve: Mapping[int, float], target: float) -> int | None:
    """The smallest ``k`` whose PCA EV reaches ``target``; ``None`` if none does."""
    return next((k for k in sorted(curve) if curve[k] >= target), None)


# ---------------------------------------------------------------------- rule


def conditions(
    total_vs_pca1: Interval, fast_ev: float, fast_vs_pca1: Interval, fast_vs_fixed: Interval
) -> dict[str, bool]:
    """The four conditions of protocol section 7."""
    return {
        "T": total_vs_pca1.low > 0,
        "H1": fast_ev >= FAST_FLOOR,
        "H2": fast_vs_pca1.low > 0,
        "H3": fast_vs_fixed.low > 0,
    }


def reconstruction_outcome(gates: Mapping[str, bool], met: Mapping[str, bool]) -> str:
    """``blocked`` when a gate fails; otherwise ``stop``, ``slow_only`` or ``sweep`` (protocol section 7)."""
    if not all(gates.values()):
        return "blocked"
    if not met["T"]:
        return "stop"
    if not (met["H1"] and met["H2"] and met["H3"]):
        return "slow_only"
    return "sweep"
