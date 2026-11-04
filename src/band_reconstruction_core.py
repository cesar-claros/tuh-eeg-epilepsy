"""Pure logic of the band-limited reconstruction (``documentation/separable_band_limited_reconstruction_protocol.md``).

It reuses ``src/reconstruction_baselines_core.py`` (the PCA, the band split, the bootstrap, the fixed bank) and adds
the 1-30 Hz band-pass, the rank-r ceiling, the scoring of one batch by the arms of this protocol, the edge energy and
the two-arm rule. The baselines core stays unchanged, as the record of its run. No corpus; the tensor functions run on
CPU or GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from src.reconstruction_baselines_core import (
    BANDS,
    CEILING_TOLERANCE,
    PCA_KS,
    POOL_LEN,
    SFREQ,
    Interval,
    band_energy,
    bank_frequencies,
    gabor_bank,
    pca_reconstructions,
    unwindow,
    window_energy,
    windows,
)

# The domain (protocol section 3): a zero-phase 4th-order Butterworth band-pass on every cached segment.
BAND_LOW_HZ, BAND_HIGH_HZ, BAND_ORDER = 1.0, 30.0, 4
RS = (1, 4)  # atoms per pooling window of the two deciding arms
TRAINED = {r: f"B{r}-trained" for r in RS}
UNTRAINED = {r: f"B{r}-untrained" for r in RS}
FIXED = {r: f"B{r}-fixed" for r in RS}
PASS1 = "S4-pass1"  # the pass-1 readout dictionaries, trained on the full-band rows (R=1)
MODEL_ARMS = (*TRAINED.values(), *UNTRAINED.values(), *FIXED.values(), PASS1)
ARMS = (*MODEL_ARMS, *(f"PCA-{k}" for k in PCA_KS), *(f"SVD-{r}" for r in RS))
BUDGET_ARMS = tuple(TRAINED.values())
BAR = {r: f"PCA-{r}" for r in RS}  # the budget-matched low bar: one real number per atom and window

# Columns of BANDS: 1-4, 4-8, 8-13 and 13-30 Hz are in the band; 4-30 Hz decides.
IN_BAND = (1, 2, 3, 4)
FAST = (2, 3, 4)
if (BANDS[IN_BAND[0]][0], BANDS[FAST[0]][0], BANDS[FAST[-1]][1]) != (BAND_LOW_HZ, 4.0, BAND_HIGH_HZ):
    raise ValueError("the band columns do not match BANDS")

FAST_FLOOR = 0.05  # F1: the least EV in 4-30 Hz of a trained arm
IN_BAND_MIN = 0.90  # gate 2: the least share of the row energy in 1-30 Hz
EDGE_SECONDS = 1.0
SLOW_ATOM_HZ = 4.0


def band_pass(
    x: np.ndarray,
    sfreq: float = SFREQ,
    low: float = BAND_LOW_HZ,
    high: float = BAND_HIGH_HZ,
    order: int = BAND_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (applied forward and backward), in the input dtype."""
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(x.dtype)


def band_bank() -> np.ndarray:
    """The fixed waveforms ``(64, L)`` of the baselines protocol, re-spaced to 1-30 Hz (protocol section 4)."""
    return gabor_bank(frequencies=bank_frequencies(low=BAND_LOW_HZ, high=BAND_HIGH_HZ))


def rank_reconstruction(rows: torch.Tensor, rank: int, pool: int = POOL_LEN) -> torch.Tensor:
    """The best rank-``rank`` approximation ``U U^T X`` of every window ``X`` (Eckart and Young), as rows ``(B, C, T)``.

    ``U`` holds the leading eigenvectors of ``X X^T``, computed in float64.

    Raises
    ------
    ValueError
        If ``rank`` is not between 1 and the number of channels.
    """
    if not 1 <= rank <= rows.shape[1]:
        raise ValueError(f"rank {rank} must be between 1 and the {rows.shape[1]} channels")
    parts = windows(rows, pool).double()
    lead = torch.linalg.eigh(parts @ parts.transpose(-1, -2))[1][..., -rank:]
    return unwindow(lead @ (lead.transpose(-1, -2) @ parts)).to(rows.dtype)


def edge_energy(
    rows: torch.Tensor, sfreq: float = SFREQ, seconds: float = EDGE_SECONDS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy per second ``(B,)`` of the first and last ``seconds`` of each segment, and of the rest, in float64.

    Raises
    ------
    ValueError
        If the two edges cover the whole segment.
    """
    n = int(round(seconds * sfreq))
    if 2 * n >= rows.shape[-1]:
        raise ValueError(f"edges of {n} samples cover a segment of {rows.shape[-1]}")
    per_sample = rows.double().pow(2).sum(dim=1)
    edge = per_sample[:, :n].sum(dim=-1) + per_sample[:, -n:].sum(dim=-1)
    inner = per_sample[:, n:-n].sum(dim=-1)
    return edge / (2 * n / sfreq), inner / ((rows.shape[-1] - 2 * n) / sfreq)


@torch.no_grad()
def score_arms(
    rows: torch.Tensor,
    valid: torch.Tensor,
    models: Mapping[str, Any],
    basis: torch.Tensor,
    weights: torch.Tensor,
    slow_atoms: Mapping[str, torch.Tensor],
    pool: int = POOL_LEN,
    sfreq: float = SFREQ,
) -> dict[str, torch.Tensor]:
    """Reconstruct encoded rows ``(B, C, T)`` with every arm of ``ARMS`` and return their energies, all float64.

    ``models`` maps each of ``MODEL_ARMS`` to a ``SeparableSAE``; its ``top_r`` names its ceiling ``SVD-r``.
    ``slow_atoms`` maps each of ``BUDGET_ARMS`` to its slow-atom mask ``(K,)``. Every reconstruction is zero on a
    rejected row (``valid`` ``(B, C)`` false).

    Returns
    -------
    dict[str, torch.Tensor]
        Per segment: ``signal_bands`` ``(B, n_bands)``, ``signal_total``, ``usable_rows``, ``edge_energy`` and
        ``interior_energy`` ``(B,)``; ``residual_bands`` ``(n_arms, B, n_bands)`` and ``residual_total``
        ``(n_arms, B)``; per budget arm and segment ``(n_budget, B)``: ``windows_won`` (windows with a winner),
        ``windows_slow`` (windows whose winners are all slow), ``slots_won`` and ``slots_slow`` (winning atoms),
        ``explained_all`` and ``explained_slow`` (explained energy of all windows and of ``windows_slow``). Per batch:
        ``ceiling_violations`` and ``ceiling_excess`` ``(n_model_arms,)``, and ``atom_wins`` ``(n_budget, K)``.

    Raises
    ------
    ValueError
        If a model's ``top_r`` has no ceiling.
    """
    mask = valid.unsqueeze(-1).to(rows.dtype)
    reconstructions: dict[str, torch.Tensor] = {}
    codes: dict[str, torch.Tensor] = {}
    for name in MODEL_ARMS:
        model = models[name]
        if model.spec.top_r not in RS:
            raise ValueError(f"{name} keeps {model.spec.top_r} atoms per window; the ceilings cover {RS}")
        code, _, reconstruction, _, _ = model._encode_reconstruct(rows, valid, model._dictionary())
        reconstructions[name] = reconstruction * mask
        if name in BUDGET_ARMS:
            codes[name] = code
    for k, reconstruction in pca_reconstructions(rows, basis, PCA_KS, pool).items():
        reconstructions[f"PCA-{k}"] = reconstruction * mask
    for r in RS:
        reconstructions[f"SVD-{r}"] = rank_reconstruction(rows, r, pool) * mask
    bands, totals, per_window = [], [], {}
    for name in ARMS:
        residual = rows - reconstructions[name]
        bands.append(band_energy(residual, weights))
        totals.append(residual.double().pow(2).sum(dim=(1, 2)))
        per_window[name] = window_energy(residual, pool)
    signal_windows = window_energy(rows, pool)
    gaps = torch.stack([per_window[f"SVD-{models[name].spec.top_r}"] - per_window[name] for name in MODEL_ARMS])
    budget: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("windows_won", "windows_slow", "slots_won", "slots_slow", "explained_all", "explained_slow")
    }
    atom_wins = []
    for name in BUDGET_ARMS:
        won = codes[name].unflatten(-1, (-1, pool)).amax(dim=-1) > 0
        slots = won.sum(dim=1)
        slow_slots = (won & slow_atoms[name].to(rows.device)[None, :, None]).sum(dim=1)
        all_slow = (slots > 0) & (slow_slots == slots)
        explained = signal_windows - per_window[name]
        budget["windows_won"].append((slots > 0).sum(dim=-1))
        budget["windows_slow"].append(all_slow.sum(dim=-1))
        budget["slots_won"].append(slots.sum(dim=-1))
        budget["slots_slow"].append(slow_slots.sum(dim=-1))
        budget["explained_all"].append(explained.sum(dim=-1))
        budget["explained_slow"].append((explained * all_slow).sum(dim=-1))
        atom_wins.append(won.sum(dim=(0, 2)))
    edge, interior = edge_energy(rows, sfreq)
    return {
        "signal_bands": band_energy(rows, weights),
        "signal_total": rows.double().pow(2).sum(dim=(1, 2)),
        "usable_rows": valid.sum(dim=-1),
        "edge_energy": edge,
        "interior_energy": interior,
        "residual_bands": torch.stack(bands),
        "residual_total": torch.stack(totals),
        **{key: torch.stack(values) for key, values in budget.items()},
        "ceiling_violations": (gaps > CEILING_TOLERANCE * signal_windows).sum(dim=(1, 2)),
        "ceiling_excess": (gaps / signal_windows.clamp_min(1e-12)).amax(dim=(1, 2)),
        "atom_wins": torch.stack(atom_wins),
    }


def arm_conditions(
    total_vs_bar: Interval, fast_ev: float, fast_vs_bar: Interval, fast_vs_fixed: Interval
) -> dict[str, bool]:
    """The four conditions of one arm (protocol section 7): ``T``, ``F1``, ``F2`` and ``F3``."""
    return {
        "T": total_vs_bar.low > 0,
        "F1": fast_ev >= FAST_FLOOR,
        "F2": fast_vs_bar.low > 0,
        "F3": fast_vs_fixed.low > 0,
    }


def band_outcome(gates: Mapping[str, bool], met: Mapping[int, Mapping[str, bool]]) -> str:
    """``blocked`` when a gate fails; ``sweep`` when either arm meets its four conditions; else ``stop`` when the
    R=1 code does not beat ``PCA-1`` in total; else ``delta_only`` (protocol section 7)."""
    if not all(gates.values()):
        return "blocked"
    if any(all(conditions.values()) for conditions in met.values()):
        return "sweep"
    if not met[1]["T"]:
        return "stop"
    return "delta_only"
