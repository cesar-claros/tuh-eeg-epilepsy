"""Pure logic of the class-masked separable dictionary: arms, permuted labels, group counts, scores and the rule.

Protocol: ``documentation/masked_separable_sae_protocol.md``. ``src/masked_dictionary.py`` holds the code that reads
signals and trains. The constants are fixed before any result (protocol sections 4 to 7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from src.frozen_readout_core import dictionary_split

N_SHARED, N_EPILEPSY_ONLY = 64, 16  # the two atom groups (section 4)
HINGE = {"sup": 1.0, "perm": 1.0, "mask": 0.0, "strong": 10.0}  # the hinge weight of each arm
SEEDS = {"sup": (1, 2), "perm": (1, 2), "mask": (1, 2), "strong": (1,)}
EPOCHS = 30  # about 390 steps on about 100 dictionary patients, as the adopted dictionary
SEGMENT_CAP = 12  # distinct training segments per dictionary patient (section 3)
PERMUTATION_SEED = 20261115  # the fixed seed of the permuted labels, with the fold
# The patient scores of sections 5 and 8: the arm and the seeds that are averaged.
SCORES = {
    "Score": ("sup", (1, 2)),
    "Score-perm": ("perm", (1, 2)),
    "Score-mask": ("mask", (1, 2)),
    "Score-strong": ("strong", (1,)),
    "Score-s1": ("sup", (1,)),
    "Score-s2": ("sup", (2,)),
}
GAP_MARGIN = 0.05  # condition C: the lower bound of Score minus Spectral must be above its negative
ROW_TOLERANCE = 1e-9  # gate 3, relative
REPEAT_TOLERANCE = 1e-6  # gate 6
N_REPEAT = 20  # gate 6: at least this many segments, in whole batches
OUTCOMES = ("closes_gap", "adds_to_drug", "separates_only", "review", "no_signal", "blocked")


def dictionary_names() -> list[str]:
    """The 7 dictionaries of one fold: ``sup``, ``perm`` and ``mask`` at seeds 1 and 2, and ``strong`` at seed 1."""
    return [f"{arm}_s{seed}" for arm, seeds in SEEDS.items() for seed in seeds]


def parse_name(name: str) -> tuple[str, int]:
    """The arm and the seed of a dictionary name.

    Raises
    ------
    ValueError
        If the name is not one of ``dictionary_names``.
    """
    if name not in dictionary_names():
        raise ValueError(f"unknown dictionary {name!r}")
    arm, _, seed = name.rpartition("_s")
    return arm, int(seed)


def permuted_labels(patients: Sequence[str], labels: Mapping[str, int], fold: int, part: int) -> dict[str, int]:
    """The labels of ``patients`` shuffled between them, so that the class counts stay; never the true labels.

    Raises
    ------
    ValueError
        If the patients hold one class only, so that every permutation equals the true labels.
    """
    members = sorted(patients)
    true = np.array([labels[p] for p in members])
    if true.min() == true.max():
        raise ValueError("a permutation needs patients of both classes")
    rng = np.random.default_rng([PERMUTATION_SEED, fold, part])
    shuffled = rng.permutation(true)
    while np.array_equal(shuffled, true):
        shuffled = rng.permutation(true)
    return {p: int(v) for p, v in zip(members, shuffled, strict=True)}


def masked_split(labels: Mapping[str, int], folds: Mapping[str, int], fold: int, seed: int) -> dict[str, Any]:
    """The dictionary and monitoring patients of one outer fold (the dictionary split of pass 1), with the permuted
    labels of the control arm. The labels are permuted within each part, so both parts keep their class counts."""
    train, val = dictionary_split(dict(labels), dict(folds), fold, seed)
    permuted = permuted_labels(train, labels, fold, 0) | permuted_labels(val, labels, fold, 1)
    return {
        "fold": fold,
        "dictionary_patients": list(train),
        "monitor_patients": list(val),
        "permuted_labels": permuted,
    }


def check_split(split: Mapping[str, Any], labels: Mapping[str, int], folds: Mapping[str, int]) -> dict[str, bool]:
    """Gate 2 on the split of one fold."""
    fold = split["fold"]
    train, val = list(split["dictionary_patients"]), list(split["monitor_patients"])
    permuted = split["permuted_labels"]
    used = set(train) | set(val)

    cover = set(permuted) == used

    def kept(part: list[str]) -> bool:
        return cover and sum(permuted[p] for p in part) == sum(labels[p] for p in part)

    return {
        "known_patients": used <= set(labels),
        "no_held_out_patient": all(folds[p] != fold for p in used),
        "monitor_apart": not (set(train) & set(val)),
        "both_classes": all(len({labels[p] for p in part}) == 2 for part in (train, val)),
        "permuted_cover": cover,
        "permuted_counts_kept": kept(train) and kept(val),
        "permuted_differ": cover and any(permuted[p] != labels[p] for p in train),
    }


@torch.no_grad()
def group_counts(code: torch.Tensor, n_shared: int, pool: int) -> dict[str, np.ndarray]:
    """What each atom group wins in a code ``(B, K, T)``: placements and squared amplitudes per segment.

    One placement is one atom in one pooling window with a positive amplitude. ``leak_windows`` counts the windows
    of a segment in which an epilepsy-only atom is placed, and ``atom_wins`` the placements of each atom per segment.
    """
    values = code.unflatten(-1, (-1, pool)).amax(dim=-1).double()  # (B, K, M)
    won = values > 0
    energy = values.square()
    return {
        "wins_shared": won[:, :n_shared].sum(dim=(1, 2)).cpu().numpy(),
        "wins_epilepsy": won[:, n_shared:].sum(dim=(1, 2)).cpu().numpy(),
        "energy_shared": energy[:, :n_shared].sum(dim=(1, 2)).cpu().numpy(),
        "energy_epilepsy": energy[:, n_shared:].sum(dim=(1, 2)).cpu().numpy(),
        "leak_windows": won[:, n_shared:].any(dim=1).sum(dim=1).cpu().numpy(),
        "windows": np.full(code.shape[0], values.shape[-1]),
        "atom_wins": won.sum(dim=2).cpu().numpy(),
    }


def share(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``first / (first + second)`` per segment, and 0 where both are 0 (section 5)."""
    total = first + second
    return np.divide(first, total, out=np.zeros(len(total), dtype=np.float64), where=total > 0)


def seed_mean(shares: Mapping[str, np.ndarray], arm: str, seeds: Sequence[int]) -> np.ndarray:
    """The mean over ``seeds`` of the segment shares of an arm."""
    return np.mean([shares[f"{arm}_s{seed}"] for seed in seeds], axis=0)


def patient_scores(values: np.ndarray, subject: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted patients and the mean of ``values`` over the segments of each."""
    patients, inverse, counts = np.unique(subject, return_inverse=True, return_counts=True)
    return patients, np.bincount(inverse, weights=values) / counts


def training_moved(val_residual: Sequence[float]) -> bool:
    """Gate 5: the validation residual after the last epoch is below the one after the first."""
    return len(val_residual) >= 2 and val_residual[-1] < val_residual[0]


def masked_outcome(gates_ok: bool, separates: bool, drug: bool, gap: bool) -> str:
    """The rule of section 7."""
    if not gates_ok:
        return "blocked"
    if separates:
        if gap:
            return "closes_gap"
        return "adds_to_drug" if drug else "separates_only"
    return "review" if drug else "no_signal"
