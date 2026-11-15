"""Pure logic of the spectral decomposition and the score fusion: column layout, arms, rank-mean and rule.

Protocol: ``documentation/spectral_decomposition_fusion_protocol.md``. This module needs numpy and scipy only.
``src/spectral_decomposition.py`` holds the code that reads the saved files. The constants are fixed before any
result (protocol sections 4 to 7).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata

# Column layout of ``SpectralFeatures`` (src/models/components/spectral_features.py): per channel, the log10 absolute
# power of each band, then the relative power of each band.
BANDS = ("1_4", "4_8", "8_13", "13_30", "30_45")
N_BANDS = len(BANDS)
PER_CHANNEL = 2 * N_BANDS
# The five chains of the TCP montage without the ear pairs (protocol section 5).
CHAINS: dict[str, tuple[str, ...]] = {
    "left_temporal": ("Fp1-F7", "F7-T3", "T3-T5", "T5-O1"),
    "right_temporal": ("Fp2-F8", "F8-T4", "T4-T6", "T6-O2"),
    "left_parasagittal": ("Fp1-F3", "F3-C3", "C3-P3", "P3-O1"),
    "right_parasagittal": ("Fp2-F4", "F4-C4", "C4-P4", "P4-O2"),
    "central": ("T3-C3", "C3-Cz", "Cz-C4", "C4-T4"),
}
# The left and right channels of each pair, as ``band_code_readout_core.LEFT_RIGHT`` (a test checks the equality).
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
N_CHANNELS = sum(len(chain) for chain in CHAINS.values())
FULL = "Spec-full"
NULL_ARMS = ("Spec-chanmean", "Spec-level")  # the smallest arms: the permuted-label read of section 8
FUSED = ("Spectral", "HYDRA", "REVE")  # the arms of the deciding fusion (section 4), in a fixed order
# Gates (section 6).
SCORE_TOLERANCE = 1e-4  # gate 3: the largest gap of a patient score from the saved ``Spectral`` score
AUROC_TOLERANCE = 5e-4  # gate 3: the mean-fold AUROC equals the saved one to three decimals
LAYOUT_TOLERANCE = 1e-4  # gate 4: the relative powers recomputed from the absolute columns


def arm_names() -> list[str]:
    """The 26 arms of section 5, in the order of its table."""
    return (
        [FULL, "Spec-abs", "Spec-rel", "Spec-level"]
        + [f"Spec-band-{b}" for b in BANDS]
        + [f"Spec-drop-{b}" for b in BANDS]
        + ["Spec-chanmean"]
        + [f"Spec-chain-{g}" for g in CHAINS]
        + [f"Spec-drop-{g}" for g in CHAINS]
        + ["Spec-asym"]
    )


def needs_channels(name: str) -> bool:
    """Whether an arm needs the channel order of the columns (the chain arms and ``Spec-asym``)."""
    return (
        name == "Spec-asym" or name.removeprefix("Spec-chain-") in CHAINS or name.removeprefix("Spec-drop-") in CHAINS
    )


def split_columns(x: np.ndarray) -> np.ndarray:
    """The features as ``(rows, channels, PER_CHANNEL)``.

    Raises
    ------
    ValueError
        If the number of columns is not a multiple of ``PER_CHANNEL``.
    """
    if x.ndim != 2 or x.shape[1] % PER_CHANNEL:
        raise ValueError(f"expected rows of a multiple of {PER_CHANNEL} columns, got shape {x.shape}")
    return x.reshape(x.shape[0], -1, PER_CHANNEL)


def layout_gap(x: np.ndarray) -> float:
    """Gate 4: the largest gap between the saved relative powers and those recomputed from the absolute columns."""
    cube = split_columns(np.asarray(x, dtype=np.float64))
    power = 10.0 ** cube[:, :, :N_BANDS]
    relative = power / power.sum(axis=2, keepdims=True)
    return float(np.abs(relative - cube[:, :, N_BANDS:]).max())


def channel_problems(names: Sequence[str]) -> list[str]:
    """Gate 5: what is wrong with a channel order; an empty list means that every chain and pair can be found."""
    problems = []
    if len(names) != N_CHANNELS or len(set(names)) != len(names):
        problems.append(f"expected {N_CHANNELS} distinct channels, got {len(names)} ({len(set(names))} distinct)")
    wanted = {c for chain in CHAINS.values() for c in chain} | {c for pair in LEFT_RIGHT for c in pair}
    missing = sorted(wanted - set(names))
    if missing:
        problems.append(f"channels missing: {missing}")
    return problems


def _index(names: Sequence[str], wanted: Sequence[str]) -> list[int]:
    position = {name: i for i, name in enumerate(names)}
    return [position[name] for name in wanted]


def arm_features(name: str, x: np.ndarray, channels: Sequence[str] | None = None) -> np.ndarray:
    """The feature matrix of one arm of section 5, from the saved ``Spectral`` columns.

    The result keeps the dtype of ``x``. Pass 1 fitted its head on the saved float32 values, and a float64 copy
    gives slightly different scores, so ``Spec-full`` would not reproduce ``Spectral`` (gate 3). The derived arms
    are computed in float64 and then cast back.

    Raises
    ------
    ValueError
        If the arm is unknown, or if it needs the channel order and ``channels`` is missing or wrong.
    """
    if name not in arm_names():
        raise ValueError(f"unknown arm {name!r}")
    x = np.asarray(x)
    cube = split_columns(x)
    rows = cube.shape[0]
    absolute, relative = cube[:, :, :N_BANDS], cube[:, :, N_BANDS:]
    if needs_channels(name):
        if channels is None or channel_problems(channels) or len(channels) != cube.shape[1]:
            raise ValueError(f"{name} needs the channel order of the columns")
    if name == FULL:
        return cube.reshape(rows, -1)
    if name == "Spec-abs":
        return absolute.reshape(rows, -1)
    if name == "Spec-rel":
        return relative.reshape(rows, -1)
    if name == "Spec-level":
        return np.log10((10.0 ** absolute.astype(np.float64)).sum(axis=2)).astype(x.dtype)
    if name == "Spec-chanmean":
        return cube.astype(np.float64).mean(axis=1).astype(x.dtype)
    if name == "Spec-asym":
        left = _index(channels, [pair[0] for pair in LEFT_RIGHT])
        right = _index(channels, [pair[1] for pair in LEFT_RIGHT])
        wide = absolute.astype(np.float64)
        return (wide[:, left, :] - wide[:, right, :]).reshape(rows, -1).astype(x.dtype)
    kind, _, key = name.removeprefix("Spec-").partition("-")
    if key in BANDS:
        band = BANDS.index(key)
        own = [band, N_BANDS + band]
        columns = own if kind == "band" else [k for k in range(PER_CHANNEL) if k not in own]
        return cube[:, :, columns].reshape(rows, -1)
    own = _index(channels, CHAINS[key])
    chosen = own if kind == "chain" else [c for c in range(cube.shape[1]) if c not in own]
    return cube[:, chosen, :].reshape(rows, -1)


def rank_mean(scores: Sequence[np.ndarray], fold: np.ndarray) -> np.ndarray:
    """Section 4: within each fold, the rank of every score divided by the fold size, averaged over the arms.

    Ties share their mean rank. No label is used.

    Raises
    ------
    ValueError
        If no scores are given or their lengths differ from ``fold``.
    """
    if not scores or any(len(s) != len(fold) for s in scores):
        raise ValueError("the scores and the folds must have the same length")
    fused = np.zeros(len(fold))
    for f in np.unique(fold):
        where = fold == f
        fused[where] = np.mean([rankdata(s[where], method="average") / where.sum() for s in scores], axis=0)
    return fused


def fusion_outcome(gates_ok: bool, gains: bool) -> str:
    """The rule of section 7: ``blocked``, ``fusion_gains`` or ``no_gain``."""
    if not gates_ok:
        return "blocked"
    return "fusion_gains" if gains else "no_gain"


# ---------------------------------------------------------------------- the small-arm lead
# Protocol: ``documentation/small_arm_lead_check_protocol.md``. The constants are fixed before any result.
LEAD_ARM = "Spec-chanmean"  # the arm of check 2
LEAD_ARMS = (
    LEAD_ARM,
    "Spec-chain-left_temporal",
    "Spec-chain-right_temporal",
    "Spec-chain-left_parasagittal",
    "Spec-chain-right_parasagittal",
)  # the five arms of the lead (section 1)
LEAD_SEEDS = tuple(20261115 + i for i in range(1, 21))  # the fold assignments of check 2
N_LEAD_PERMUTATIONS = 10_000
LEAD_P = 0.05  # condition P: the selection p-value must be below it
LEAD_MEDIAN = 0.65  # condition M: the median cell statistic over the assignments must reach it
LEAD_SHARE = 0.60  # descriptive: the share of assignments at or above it
CELL_TOLERANCE = 1e-9  # gate 3: the cell statistic against the saved value of the read
REFIT_TOLERANCE = 1e-4  # gate 3: the refit of ``LEAD_ARM`` against its saved scores


def cell_statistic(score: np.ndarray, label: np.ndarray) -> float:
    """The share of the pairs (label 1, label 0) in which the label-1 patient has the higher score, ties at one half.

    Raises
    ------
    ValueError
        If a class is empty.
    """
    positive, negative = score[label == 1], score[label == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("the cell needs patients of both classes")
    wins = (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])
    return float(wins.mean())


def permute_within_fold(label: np.ndarray, fold: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The labels shuffled between the patients of each fold, so that every fold keeps its class counts."""
    out = label.copy()
    for f in np.unique(fold):
        where = np.flatnonzero(fold == f)
        out[where] = label[where][rng.permutation(len(where))]
    return out


def selection_null(
    scores: np.ndarray, label: np.ndarray, fold: np.ndarray, n_perm: int = N_LEAD_PERMUTATIONS, seed: int = 0
) -> dict[str, np.ndarray]:
    """Check 1: the cell statistic of every arm, and its values under labels permuted within fold.

    ``scores`` holds one row per arm over the patients of the cell. The statistic of an arm is its rank sum, so one
    permutation costs one matrix product. Returns ``observed`` ``(arms,)`` and ``null`` ``(n_perm, arms)``.

    Raises
    ------
    ValueError
        If the shapes disagree or a class is empty.
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    if scores.shape[1] != len(label) or len(label) != len(fold):
        raise ValueError("the scores, the labels and the folds must cover the same patients")
    n_pos, n_neg = int((label == 1).sum()), int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("the cell needs patients of both classes")
    ranks = np.stack([rankdata(row, method="average") for row in scores])

    def statistic(y: np.ndarray) -> np.ndarray:
        return (ranks @ (y == 1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    rng = np.random.default_rng(seed)
    null = np.stack([statistic(permute_within_fold(label, fold, rng)) for _ in range(n_perm)])
    return {"observed": statistic(label), "null": null}


def permutation_p(observed: float, null: np.ndarray) -> float:
    """The share of the draws at or above ``observed``, with the observed arrangement counted as one draw."""
    return float((1 + (null >= observed - 1e-12).sum()) / (1 + len(null)))


def lead_outcome(gates_ok: bool, selection: bool, median: bool) -> str:
    """The rule of section 5: ``blocked``, ``survives`` or ``dropped``."""
    if not gates_ok:
        return "blocked"
    return "survives" if selection and median else "dropped"
