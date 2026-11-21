"""Responses, ranks, simple measures and scores of the Events corpus protocol (pure logic; torch and numpy).

Protocol: ``documentation/event_corpus_dictionary_protocol.md``, sections 5 to 7. A record is one piece of a file
(5 minutes) as ``(channels, samples)`` at 256 Hz. A feature of a channel and a second is the response of a waveform,
or a simple measure, in that second. Its rank is the share of the other seconds of the same record and channel with
a lower value. ``src/event_dictionary.py`` holds the code that reads files.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from src.band_reconstruction_core import band_pass
from src.models.components.shapeconv_sae import ShapeConvSAE

SFREQ = 256
ATOM_LEN = 128
# A band: the low and the high edge in Hz, and whether the first difference follows (protocol section 4).
BANDS = {"30": (1.0, 30.0, False), "70": (1.0, 70.0, False), "70-diff": (1.0, 70.0, True)}
POWER_BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 70))
SIMPLE = (*(f"power_{low}_{high}" for low, high in POWER_BANDS), "line_length", "peak_to_peak", "max_abs")
N_RANDOM = 4  # random dictionaries per band, as many atoms as the candidates of a class in stage 2
S_FLOOR, STRONG, GAIN = 0.65, 0.80, 0.03
DECIDING = ("spsw", "gped", "pled")
LEVEL = 1 - 0.05 / len(DECIDING)  # the level of the intervals of L and B


def simple_names(band: str) -> tuple[str, ...]:
    """The simple measures of a band: the power above 30 Hz exists on the wide bands only."""
    return SIMPLE if BANDS[band][1] > 30.0 else tuple(name for name in SIMPLE if name != "power_30_70")


def band_rows(record: np.ndarray, band: str) -> torch.Tensor:
    """The rows of a record in a band: band-pass, the first difference if the band has it, then each row centered
    on its median, scaled by 1.4826 MAD and clipped at 20, as the model does. A constant row gives zeros."""
    low, high, difference = BANDS[band]
    rows = torch.from_numpy(band_pass(np.asarray(record, dtype=np.float32), SFREQ, low, high).copy())
    valid = ShapeConvSAE._valid_rows(rows)
    if difference:
        rows = rows.diff(dim=-1, prepend=rows[:, :1])
    rows = ShapeConvSAE._robust_scale(rows)
    return torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))


def random_atoms(band: str, index: int, n_atoms: int = 64) -> np.ndarray:
    """One dictionary of random waveforms of unit norm, fixed by the band and its index."""
    seed = 1000 * list(BANDS).index(band) + index
    atoms = np.random.default_rng(seed).normal(size=(n_atoms, ATOM_LEN))
    return (atoms / np.linalg.norm(atoms, axis=1, keepdims=True)).astype(np.float32)


def responses(rows: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """The absolute response ``(channels, atoms, samples)`` of every waveform on every row. Sample ``t`` holds the
    response of the patch that is centered on ``t`` (samples ``t - 64`` to ``t + 63``; zeros beyond the record)."""
    out = F.conv1d(rows.unsqueeze(1), atoms.unsqueeze(1), padding=atoms.shape[-1] // 2)
    return out[..., : rows.shape[-1]].abs()


def measures(windows: torch.Tensor, band: str) -> torch.Tensor:
    """The simple measures ``(..., measures)`` of windows of one second ``(..., 256)``, in the order of
    ``simple_names``: the log power in each frequency band, the line length, the peak-to-peak amplitude and the
    largest absolute value."""
    taper = torch.hann_window(windows.shape[-1], device=windows.device, dtype=windows.dtype)
    power = torch.fft.rfft(windows * taper, dim=-1).abs().pow(2)  # bins of 1 Hz
    out = []
    for low, high in POWER_BANDS:
        if f"power_{low}_{high}" in simple_names(band):
            out.append(torch.log10(power[..., low:high].sum(-1) + 1e-12))
    out.append(windows.diff(dim=-1).abs().sum(-1))
    out.append(windows.amax(-1) - windows.amin(-1))
    out.append(windows.abs().amax(-1))
    return torch.stack(out, dim=-1)


def record_features(
    rows: torch.Tensor, atoms: torch.Tensor, band: str, starts: Sequence[int], chunk: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """The features of one record in one band: the response of every atom, then the simple measures.

    Returns ``grid`` ``(channels, features, seconds)`` for the whole seconds of the record, and ``at``
    ``(starts, channels, features)`` for windows of one second that start at the given samples.
    """
    n_seconds = rows.shape[-1] // SFREQ
    grid_parts, at_parts = [], []
    for first in range(0, atoms.shape[0], chunk):
        response = responses(rows, atoms[first : first + chunk])
        used = response[..., : n_seconds * SFREQ]
        grid_parts.append(used.reshape(*used.shape[:2], n_seconds, SFREQ).amax(-1))
        at_parts.append(torch.stack([response[..., s : s + SFREQ].amax(-1) for s in starts]))
    grid_windows = rows[:, : n_seconds * SFREQ].reshape(rows.shape[0], n_seconds, SFREQ)
    grid_parts.append(measures(grid_windows, band).transpose(1, 2))
    at_parts.append(torch.stack([measures(rows[:, s : s + SFREQ], band) for s in starts]))
    return torch.cat(grid_parts, dim=1), torch.cat(at_parts, dim=2)


def rank_in_record(value: torch.Tensor, grid: torch.Tensor, start: int) -> torch.Tensor:
    """The rank ``(channels, features)`` of a window that starts at sample ``start``: the share of the whole
    seconds of the record that do not overlap the window and have a lower value, with half of the ties."""
    seconds = torch.arange(grid.shape[-1], device=grid.device) * SFREQ
    keep = (seconds + SFREQ <= start) | (seconds >= start + SFREQ)
    others = grid[..., keep]
    lower = (others < value.unsqueeze(-1)).sum(-1)
    ties = (others == value.unsqueeze(-1)).sum(-1)
    return (lower + 0.5 * ties) / max(int(keep.sum()), 1)


# ---------------------------------------------------------------------- scores


def auroc_columns(features: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """The AUROC of every column of ``features`` ``(units, columns)`` for the units in ``positive`` against the
    others; a row with a NaN is not used for its column. NaN where a side has no unit."""
    out = np.full(features.shape[1], np.nan)
    positive = np.asarray(positive, dtype=bool)
    for j in range(features.shape[1]):
        column = features[:, j]
        known = np.isfinite(column)
        n_pos, n_neg = int((positive & known).sum()), int((~positive & known).sum())
        if n_pos and n_neg:
            ranks = rankdata(column[known])
            out[j] = (ranks[positive[known]].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def pair_counts(values: np.ndarray, positive: np.ndarray, patients: np.ndarray) -> dict[str, Any]:
    """For the bootstrap of patients: ``wins[p, q]`` is the number of pairs of a positive unit of patient ``p`` and
    another unit of patient ``q`` in which the positive unit has the higher value, with half of the ties. Units
    with a NaN are left out."""
    known = np.isfinite(values)
    values, positive, patients = values[known], np.asarray(positive, dtype=bool)[known], np.asarray(patients)[known]
    names, index = np.unique(patients, return_inverse=True)
    wins = np.zeros((len(names), len(names)))
    pos_values, pos_index = values[positive], index[positive]
    for q in range(len(names)):
        others = np.sort(values[~positive & (index == q)])
        if len(others) and len(pos_values):
            left = np.searchsorted(others, pos_values, side="left")
            right = np.searchsorted(others, pos_values, side="right")
            wins[:, q] = np.bincount(pos_index, weights=left + 0.5 * (right - left), minlength=len(names))
    return {
        "patients": names,
        "wins": wins,
        "n_pos": np.bincount(index[positive], minlength=len(names)).astype(float),
        "n_neg": np.bincount(index[~positive], minlength=len(names)).astype(float),
    }


def auroc_of(counts: dict[str, Any], weights: np.ndarray | None = None) -> np.ndarray | float:
    """The AUROC from pair counts; with ``weights`` ``(draws, patients)``, one value per draw of patients (NaN for
    a draw with no unit on a side)."""
    if weights is None:
        total = counts["n_pos"].sum() * counts["n_neg"].sum()
        return float(counts["wins"].sum() / total) if total else float("nan")
    top = np.einsum("bp,pq,bq->b", weights, counts["wins"], weights)
    bottom = (weights @ counts["n_pos"]) * (weights @ counts["n_neg"])
    return np.divide(top, bottom, out=np.full(len(weights), np.nan), where=bottom > 0)


def patient_draws(patients: Sequence[str], n_boot: int, seed: int) -> np.ndarray:
    """How often each patient is in each bootstrap draw ``(draws, patients)``; patients in sorted order."""
    names = np.unique(np.asarray(patients))
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=n_boot).astype(float)


def choose_and_score(
    features: np.ndarray,
    columns: Sequence[Sequence[int]],
    label: np.ndarray,
    patient: np.ndarray,
    fold: np.ndarray,
    target: str,
    n_folds: int,
) -> list[dict[str, Any] | None]:
    """Per fold: the column with the highest AUROC for ``target`` against all other units on the training folds,
    among ``columns[fold]``, and its pair counts on the held-out fold. ``None`` for a fold with no unit of the
    target on a side. No held-out unit enters the choice."""
    positive = label == target
    out: list[dict[str, Any] | None] = []
    for f in range(n_folds):
        train, held = fold != f, fold == f
        candidates = np.asarray(columns[f], dtype=int)
        if not (positive & train).any() or not (positive & held).any() or not len(candidates):
            out.append(None)
            continue
        scores = auroc_columns(features[train][:, candidates], positive[train])
        if not np.isfinite(scores).any():
            out.append(None)
            continue
        best = int(candidates[int(np.nanargmax(scores))])
        counts = pair_counts(features[held, best], positive[held], patient[held])
        out.append({"column": best, "train_auroc": float(np.nanmax(scores)), "counts": counts})
    return out


def summarize(folds: list[dict[str, Any] | None], draws: list[dict[str, Any] | None]) -> dict[str, Any]:
    """The mean-fold score of an arm with the bootstrap values of that mean (one per draw). ``draws`` holds, per
    fold, the sorted held-out patients and their counts per draw (``patient_draws``)."""
    points, boots = [], []
    for entry, weights in zip(folds, draws, strict=True):
        if entry is None or weights is None:
            continue
        points.append(auroc_of(entry["counts"]))
        boots.append(aligned_draws(entry["counts"], weights))
    if not points:
        return {"score": None, "boot": None, "folds": 0}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a draw can lack the class in every fold
        boot = np.nanmean(np.stack(boots), axis=0)
    return {"score": float(np.mean(points)), "boot": boot, "folds": len(points)}


def aligned_draws(counts: dict[str, Any], weights: dict[str, Any]) -> np.ndarray:
    """The bootstrap AUROCs of an arm for the draws of a fold. ``weights`` holds the sorted held-out patients of
    the fold and their counts per draw; an arm can lack a patient whose values are all NaN."""
    position = {name: i for i, name in enumerate(weights["patients"])}
    return np.asarray(auroc_of(counts, weights["counts"][:, [position[name] for name in counts["patients"]]]))


def interval(boot: np.ndarray, level: float = 0.95) -> list[float]:
    """The central interval of bootstrap values at a level."""
    tail = (1 - level) / 2
    return [float(np.nanquantile(boot, tail)), float(np.nanquantile(boot, 1 - tail))]


def class_outcome(score: float, learning: dict[str, float], simple: dict[str, float]) -> str:
    """The outcome of a class from S, L and B (protocol section 7). ``learning`` and ``simple`` hold the gain over
    the random waveforms and over the best simple measure, with the lower limit of its interval."""
    if score < S_FLOOR:
        return "not_identified"
    passed = all(gain["gain"] >= GAIN and gain["low"] > 0 for gain in (learning, simple))
    if not passed:
        return "detected_not_specific"
    return "identified (strong)" if score >= STRONG else "identified (weak)"


# ---------------------------------------------------------------------- the scan of a record (protocol section 8)


def second_scores(grid: torch.Tensor) -> torch.Tensor:
    """The score of every second of a record from the values ``(channels, seconds)`` of one feature: each channel in
    robust units of its own record (minus the median, over 1.4826 MAD), then the second-largest channel value, so
    that two channels must agree."""
    median = grid.median(dim=-1, keepdim=True).values
    spread = 1.4826 * (grid - median).abs().median(dim=-1, keepdim=True).values
    units = (grid - median) / spread.clamp_min(1e-12)
    return units.topk(2, dim=0).values[1]


def alarm_starts(score: np.ndarray, threshold: float) -> np.ndarray:
    """The first second of every alarm: a run of consecutive seconds above the threshold."""
    above = np.asarray(score) > threshold
    return np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))


def alarms_per_record(scores: Sequence[np.ndarray], threshold: float, record_s: float = 300.0) -> float:
    """The alarms per record length (5 minutes) over a set of records, each given by its score per second."""
    seconds = sum(len(score) for score in scores)
    alarms = sum(len(alarm_starts(score, threshold)) for score in scores)
    return float(alarms * record_s / seconds) if seconds else float("nan")


def threshold_for_rate(scores: Sequence[np.ndarray], rate: float, record_s: float = 300.0) -> float:
    """The lowest threshold at which the records give at most ``rate`` alarms per record length. The candidates are
    the scores themselves, from the highest down; NaN without a record."""
    if not scores:
        return float("nan")
    candidates = np.unique(np.concatenate(scores))[::-1]
    low, high = 0, len(candidates) - 1  # the rate rises as the threshold falls, apart from merged runs
    best = float(candidates[0])
    while low <= high:
        middle = (low + high) // 2
        if alarms_per_record(scores, float(candidates[middle]), record_s) <= rate:
            best, low = float(candidates[middle]), middle + 1
        else:
            high = middle - 1
    return best


def found_marks(score: np.ndarray, marks: Sequence[int], threshold: float) -> int:
    """How many marked seconds have a second above the threshold within 1 s."""
    above = np.asarray(score) > threshold
    return int(sum(above[max(mark - 1, 0) : mark + 2].any() for mark in marks))
