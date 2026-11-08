"""Pure logic of the frozen readout of the band-limited code (``documentation/band_code_readout_protocol.md``).

The code details of one batch (sparse code, winning projections, best response per atom and window, reconstruction),
the features of section 4 (hard and soft counts per atom and per template, response tails, channel energy and
left-right asymmetry, the power level), the count check and the rule of section 7. No corpus; torch and numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))  # the band-power arms (section 4)
# The 10 homologous left-right pairs of the 20-channel TCP montage (section 4).
LEFT_RIGHT = (
    ("Fp1-F7", "Fp2-F8"),
    ("F7-T3", "F8-T4"),
    ("T3-T5", "T4-T6"),
    ("T5-O1", "T6-O2"),
    ("Fp1-F3", "Fp2-F4"),
    ("F3-C3", "F4-C4"),
    ("C3-P3", "C4-P4"),
    ("P3-O1", "P4-O2"),
    ("T3-C3", "C4-T4"),
    ("C3-Cz", "Cz-C4"),
)
TAIL_QUANTILE = 0.9
MAD_TO_SCALE = 1.4826
N_PERMUTATIONS = 20
PERMUTATION_MARGIN = 0.03  # gate 2
EV_TOLERANCE = 1e-6  # gate 3, relative
REPEAT_TOLERANCE = 1e-6  # gate 5
N_REPEAT = 20
TOP_ATOMS = 10
RECUR_WAVEFORM, RECUR_PATTERN = 0.9, 0.8  # the recurrence rule of pass 1
CODE_SETS = ("code", "counts", "templates", "hard", "soft", "tails", "channels", "asymmetry")


@dataclass
class CodeDetails:
    """One batch of segments ``(B, C, T)`` under one dictionary: the code ``(B, K, T)``, the winning projection at
    every time ``(B, K, T)``, the best response per atom and window before selection ``(B, K, M)``, and the
    reconstruction ``(B, C, T)``."""

    code: torch.Tensor
    projection: torch.Tensor
    candidates: torch.Tensor
    reconstruction: torch.Tensor


@torch.no_grad()
def encode_details(model: Any, rows: torch.Tensor, valid: torch.Tensor) -> CodeDetails:
    """The code details of a ``SeparableSAE`` on encoded rows, as ``_encode_reconstruct`` computes them."""
    atoms = model._dictionary()
    padded = F.pad(rows, model.padding)
    responses, projection = model._correlate(padded, atoms).max(dim=2)
    code, _ = model._selection(responses)
    candidates = F.relu(responses).unflatten(-1, (-1, model.spec.pool_len)).amax(dim=-1)
    reconstruction = model._encode_reconstruct(rows, valid, atoms)[2]
    return CodeDetails(code, projection, candidates, reconstruction)


def channel_index(names: Sequence[str]) -> tuple[list[int], list[int]]:
    """The left and right channel indices of ``LEFT_RIGHT`` in the order of ``names``.

    Raises
    ------
    ValueError
        If a channel of a pair is missing.
    """
    position = {name: i for i, name in enumerate(names)}
    missing = [n for pair in LEFT_RIGHT for n in pair if n not in position]
    if missing:
        raise ValueError(f"channels missing for the asymmetry pairs: {missing}")
    return [position[a] for a, _ in LEFT_RIGHT], [position[b] for _, b in LEFT_RIGHT]


@torch.no_grad()
def code_features(
    details: CodeDetails,
    patterns: torch.Tensor,
    pool: int,
    top_r: int,
    minutes: float,
    left: Sequence[int],
    right: Sequence[int],
) -> dict[str, np.ndarray]:
    """The code features of section 4 per segment, as float64 numpy arrays, with the counts that gate 4 checks.

    ``patterns`` holds the unit spatial patterns ``(K, C, D)``. A winner counts only with an amplitude above 0. The
    channel energy counts every win with its full template, so a support cut at the window edge counts in full.
    ``counts`` holds the hard and soft counts per minute without ``log1p``, for the ``Code-hydra`` arm.
    """
    values, position = details.code.unflatten(-1, (-1, pool)).max(dim=-1)  # (B, K, M)
    won = values > 0
    n_patterns = patterns.shape[-1]
    winning = details.projection.unflatten(-1, (-1, pool)).gather(-1, position.unsqueeze(-1)).squeeze(-1)
    one_hot = F.one_hot(winning, n_patterns).to(values.dtype) * won.unsqueeze(-1).to(values.dtype)  # (B, K, M, D)
    hard = won.sum(dim=-1).to(torch.float64) / minutes
    soft = values.double().sum(dim=-1) / minutes
    template_hard = one_hot.double().sum(dim=2) / minutes  # (B, K, D)
    template_soft = (one_hot.double() * values.double().unsqueeze(-1)).sum(dim=2) / minutes
    energy = torch.einsum(
        "bkmd,kcd->bkc", one_hot.double() * values.double().pow(2).unsqueeze(-1), patterns.double().pow(2)
    )
    energy = energy / minutes
    e_left, e_right = energy[:, :, list(left)].sum(-1), energy[:, :, list(right)].sum(-1)
    total = e_left + e_right
    asymmetry = torch.where(total > 0, (e_left - e_right).abs() / total.clamp_min(1e-300), torch.zeros_like(total))
    candidates = details.candidates.double()
    tails = torch.stack([torch.quantile(candidates, TAIL_QUANTILE, dim=-1), candidates.amax(dim=-1)], dim=-1)

    def out(t: torch.Tensor) -> np.ndarray:
        return t.flatten(1).cpu().numpy()

    log_hard, log_soft = torch.log1p(hard), torch.log1p(soft)
    return {
        "code": out(torch.cat([log_hard, log_soft], dim=1)),
        "counts": out(torch.cat([hard, soft], dim=1)),  # per minute, without log1p (the Code-hydra arm)
        "templates": out(torch.cat([torch.log1p(template_hard).flatten(1), torch.log1p(template_soft).flatten(1)], 1)),
        "hard": out(log_hard),
        "soft": out(log_soft),
        "tails": out(torch.log1p(tails)),
        "channels": out(torch.log1p(energy)),
        "asymmetry": out(asymmetry),
        "won_total": won.sum(dim=(1, 2)).cpu().numpy(),
        "positive_total": (details.code > 0).sum(dim=(1, 2)).cpu().numpy(),
        "slots": np.full(values.shape[0], top_r * values.shape[-1]),
    }


def level_features(x: torch.Tensor) -> np.ndarray:
    """``log10(1.4826 MAD)`` of every channel of segments ``(B, C, T)`` before the row scaling, with the lower median
    of the row pipeline (``ShapeConvSAE._row_median``)."""
    from src.models.components.shapeconv_sae import ShapeConvSAE

    centered = x - ShapeConvSAE._row_median(x)
    mad = ShapeConvSAE._row_median(centered.abs()).squeeze(-1) * MAD_TO_SCALE
    return torch.log10(mad.double().clamp_min(1e-12)).cpu().numpy()


@torch.no_grad()
def segment_energy(
    rows: torch.Tensor, reconstruction: torch.Tensor, valid: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Residual and signal energy per segment, float64, as ``band_sweep_core.score_setting`` computes them: the
    reconstruction of an unusable channel (``valid`` false, ``(B, C)``) counts as 0."""
    residual = rows - reconstruction * valid.unsqueeze(-1).to(rows.dtype)
    energy = residual.double().pow(2).sum(dim=(1, 2))
    return energy.cpu().numpy(), rows.double().pow(2).sum(dim=(1, 2)).cpu().numpy()


def count_check(won_total: np.ndarray, positive_total: np.ndarray, slots: int) -> dict[str, Any]:
    """Gate 4: the hard counts add up to the positive amplitudes of the code and to at most ``slots`` per segment."""
    return {
        "matches_code": bool(np.array_equal(won_total, positive_total)),
        "within_slots": bool((won_total <= slots).all()),
        "zero_slot_share": float(1.0 - won_total.sum() / max(slots * len(won_total), 1)),
    }


def recur(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> bool:
    """Whether an atom (waveform ``(L,)``, patterns ``(C, D)``) matches another, by the rule of pass 1: waveform
    ``|cos|`` at the best lag at least 0.9, and some pair of spatial patterns with ``|cos|`` at least 0.8."""
    a, b = first[0] / np.linalg.norm(first[0]), second[0] / np.linalg.norm(second[0])
    lag = np.abs(np.correlate(np.pad(a, len(b)), b, mode="valid")).max()
    patterns = np.abs(first[1].T @ second[1]).max()
    return bool(lag >= RECUR_WAVEFORM and patterns >= RECUR_PATTERN)


def readout_outcome(gates_ok: bool, learning: bool, band: bool, drug: bool) -> str:
    """The outcome of section 7 from the three conditions L, B and G."""
    if not gates_ok:
        return "blocked"
    if learning and band:
        return "code_adds" if drug else "beats_band_power"
    return "review" if drug else "no_advantage"
