"""Pure logic of the class-conditional dictionaries (``documentation/class_dictionary_score_protocol.md``).

The dictionary patients of each arm, the stratified random halves and the split checks of gate 2, the capped segment
draw, the residual energies of one batch under one dictionary, the patient scores, the recurrence share of two
dictionaries, and the rule of section 7. No corpus; torch and numpy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from src.band_code_readout_core import recur
from src.frozen_readout_core import dictionary_split, patient_mean, segment_draw
from src.reconstruction_baselines_core import band_energy

TRAINED = ("epi", "non", "A", "B")  # the trained arms (section 4)
STARTS = ("epi-start", "non-start")  # one epoch at learning rate 0, from the class's own patches
SEEDS = (1, 2)
EPOCHS = 56  # about 392 steps at 7 steps per epoch, against 390 for the adopted dictionary (section 2)
SEGMENT_CAP = 12  # distinct training segments per dictionary patient (section 3)
HALVES_SEED = 20261112  # the fixed seed of the random halves, with the fold
# The patient scores of sections 5 and 8: (first arm, second arm), score = r_first - r_second.
PAIRS = {"Score": ("non", "epi"), "Score-halves": ("B", "A"), "Score-start": ("non-start", "epi-start")}
BAND_COLUMNS = {"1_4": 1, "4_8": 2, "8_13": 3, "13_30": 4}  # columns of reconstruction_baselines_core.BANDS
ROW_TOLERANCE = 1e-9  # gate 3, relative
REPEAT_TOLERANCE = 1e-6  # gate 5
N_REPEAT = 20  # gate 5: at least this many segments, in whole batches
OUTCOMES = ("score_adds", "beats_band_power", "review", "separates_only", "no_advantage", "blocked")


def dictionary_names() -> list[str]:
    """The 12 dictionaries of a fold, as ``{arm}_s{seed}``: 8 trained, then 4 starts."""
    return [f"{arm}_s{seed}" for arm in (*TRAINED, *STARTS) for seed in SEEDS]


def parse_name(name: str) -> tuple[str, int]:
    """The arm and seed of a dictionary name.

    Raises
    ------
    ValueError
        If the arm or the seed is not one of the protocol's.
    """
    arm, _, seed = name.rpartition("_s")
    if arm not in (*TRAINED, *STARTS) or not seed.isdigit() or int(seed) not in SEEDS:
        raise ValueError(f"unknown dictionary {name!r}")
    return arm, int(seed)


def halves(patients: Sequence[str], labels: Mapping[str, int], fold: int) -> tuple[list[str], list[str]]:
    """Two random halves of ``patients``, each stratified by class; an odd class puts its extra patient first."""
    rng = np.random.default_rng([HALVES_SEED, fold])
    first: list[str] = []
    second: list[str] = []
    for label in (0, 1):
        members = sorted(p for p in patients if labels[p] == label)
        order = rng.permutation(len(members))
        cut = (len(members) + 1) // 2
        first += [members[i] for i in order[:cut]]
        second += [members[i] for i in order[cut:]]
    return sorted(first), sorted(second)


def class_split(labels: Mapping[str, int], folds: Mapping[str, int], fold: int, seed: int) -> dict[str, Any]:
    """The training and validation patients of every arm of one outer fold (section 3).

    The dictionary split of pass 1 gives the dictionary patients and the monitoring patients. The class arms take one
    class of each; the halves split the dictionary patients and monitor on all monitoring patients; a start uses the
    patients of its class arm.
    """
    train, val = dictionary_split(dict(labels), dict(folds), fold, seed)
    first, second = halves(train, labels, fold)
    by_class = {
        arm: ([p for p in train if labels[p] == label], [p for p in val if labels[p] == label])
        for arm, label in (("epi", 1), ("non", 0))
    }
    arms = {
        "epi": by_class["epi"],
        "non": by_class["non"],
        "A": (first, list(val)),
        "B": (second, list(val)),
        "epi-start": by_class["epi"],
        "non-start": by_class["non"],
    }
    return {
        "fold": fold,
        "dictionary_patients": list(train),
        "monitor_patients": list(val),
        "arms": {arm: {"train": list(t), "val": list(v)} for arm, (t, v) in arms.items()},  # no shared lists
    }


def check_split(split: Mapping[str, Any], labels: Mapping[str, int], folds: Mapping[str, int]) -> dict[str, bool]:
    """Gate 2 on the split of one fold: held-out patients nowhere, pure class arms, and halves that are disjoint,
    complete and stratified within one patient per class."""
    fold = split["fold"]
    arms = split["arms"]
    used = {p for arm in arms.values() for part in ("train", "val") for p in arm[part]}
    dictionary = set(split["dictionary_patients"])
    first, second = set(arms["A"]["train"]), set(arms["B"]["train"])
    balance = all(
        abs(sum(labels[p] == label for p in first) - sum(labels[p] == label for p in second)) <= 1 for label in (0, 1)
    )
    return {
        "known_patients": used <= set(labels),
        "no_held_out_patient": all(folds[p] != fold for p in used),
        "class_arms_pure": all(
            labels[p] == label
            for arm, label in (("epi", 1), ("non", 0), ("epi-start", 1), ("non-start", 0))
            for part in ("train", "val")
            for p in arms[arm][part]
        ),
        "starts_match_their_arms": arms["epi-start"] == arms["epi"] and arms["non-start"] == arms["non"],
        "halves_disjoint": not (first & second),
        "halves_complete": first | second == dictionary,
        "halves_stratified": balance,
        "monitor_apart": not (dictionary & set(split["monitor_patients"])),
    }


def capped_positions(subject: str, keys: Sequence[tuple[Any, ...]], cap: int = SEGMENT_CAP) -> list[int]:
    """Positions of at most ``cap`` segments of a patient: all of them, or the seeded per-patient draw of pass 1."""
    if len(keys) <= cap:
        return list(range(len(keys)))
    return segment_draw(subject, keys, cap)


@torch.no_grad()
def segment_energies(rows: torch.Tensor, valid: torch.Tensor, model: Any, weights: torch.Tensor) -> dict[str, Any]:
    """Residual and signal energies of one batch under one dictionary, as ``band_sweep_core.score_setting`` computes
    them without its ceiling: totals ``(B,)``, bands ``(B, n_bands)``, all float64, and the won windows per atom."""
    pool = model.spec.pool_len
    code, _, reconstruction, _, _ = model._encode_reconstruct(rows, valid, model._dictionary())
    residual = rows - reconstruction * valid.unsqueeze(-1).to(rows.dtype)
    won = code.unflatten(-1, (-1, pool)).amax(dim=-1) > 0
    return {
        "residual_total": residual.double().pow(2).sum(dim=(1, 2)).cpu().numpy(),
        "residual_bands": band_energy(residual, weights).cpu().numpy(),
        "signal_total": rows.double().pow(2).sum(dim=(1, 2)).cpu().numpy(),
        "signal_bands": band_energy(rows, weights).cpu().numpy(),
        "atom_wins": won.sum(dim=(0, 2)).cpu().numpy(),
    }


def residual_shares(residual: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """``residual / signal`` per segment (and band); a segment without signal gives NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(signal > 0, residual / signal, np.nan)


def pair_difference(shares: Mapping[str, np.ndarray], first: str, second: str, seeds: Sequence[int]) -> np.ndarray:
    """Per segment, the mean over ``seeds`` of ``r_first - r_second`` (section 5)."""
    return np.mean([shares[f"{first}_s{s}"] - shares[f"{second}_s{s}"] for s in seeds], axis=0)


def patient_scores(difference: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted patients and the mean of their segments' differences.

    Raises
    ------
    ValueError
        If a difference is not finite.
    """
    if not np.isfinite(difference).all():
        raise ValueError("a residual share is not finite")
    patients, means, _ = patient_mean(difference, subject)
    return patients, means


def recurrence_share(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> float:
    """The share of the atoms of ``first`` (waveforms ``(K, L)``, patterns ``(K, C, D)``) that match an atom of
    ``second`` by the rule of pass 1."""
    waveforms, patterns = first
    others = list(zip(*second, strict=True))
    found = [any(recur((w, u), other) for other in others) for w, u in zip(waveforms, patterns, strict=True)]
    return float(np.mean(found))


def training_moved(val_residual: Sequence[float]) -> bool:
    """Gate 4: the last validation residual of a training curve lies below the first."""
    return len(val_residual) >= 2 and float(val_residual[-1]) < float(val_residual[0])


def score_outcome(gates_ok: bool, separates: bool, band: bool, drug: bool) -> str:
    """The outcome of section 7 from the conditions S, B and G."""
    if not gates_ok:
        return "blocked"
    if separates and band:
        return "score_adds" if drug else "beats_band_power"
    if drug:
        return "review"
    return "separates_only" if separates else "no_advantage"
