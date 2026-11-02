"""Pure logic of the frozen dictionary feature readout: folds, segment draw, peak-rate features, head and rule.

Protocol: ``documentation/frozen_feature_readout_protocol.md``. This module needs numpy, scipy and
scikit-learn only, not torch or the corpus. ``src/frozen_readout.py`` holds the code that reads signals.
The constants are fixed before any result (protocol sections 3 to 7).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

# Data (section 3).
N_FOLDS = 5
DICTIONARY_VAL_SHARE = 0.2
PASS_SEEDS = {1: 20261101, 2: 20261102}  # fold assignment and dictionary seeds of pass 1 and pass 2
MAX_SEGMENTS = 9  # the fixed read: at most this, and never more than the smallest bag of the audit
LOW_PASS_HZ, LOW_PASS_ORDER = 100.0, 4  # applied forward and backward on every cached segment
# Features (section 5).
Z_GRID = (3.0, 4.0, 5.0, 6.0, 8.0)
TARGET_RATE = 0.5  # peaks per minute: the label-free target of the z rule
MAD_TO_SCALE = 1.4826
# Head (section 6).
C_GRID = (0.1, 1.0, 10.0)
INNER_FOLDS = 5
SPARSE_TEMPLATES = 10
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
N_PERMUTATIONS = 20
# Rule (section 7).
HYDRA_MARGIN = 0.15
PLANTED_FRACTION_MIN = 0.9  # of the patients, for each template of the positive control
MIN_RATE_GROUP = 10  # patients per class for a native-rate group to enter the guard
PRIMARY = "S4-proj"
DECIDING = {"Spectral": 0.0, "S4-untrained": 0.0, "HYDRA": -HYDRA_MARGIN}  # the lower bound must exceed these


def assign_folds(labels: dict[str, int], seed: int, n_folds: int = N_FOLDS) -> dict[str, int]:
    """Outer fold of every patient, stratified by class. It depends only on the patient IDs, labels and seed."""
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    for label in sorted(set(labels.values())):
        members = sorted(subject for subject, value in labels.items() if value == label)
        for position, index in enumerate(rng.permutation(len(members))):
            folds[members[index]] = position % n_folds
    return folds


def dictionary_split(
    labels: dict[str, int], folds: dict[str, int], fold: int, seed: int, share: float = DICTIONARY_VAL_SHARE
) -> tuple[list[str], list[str]]:
    """Dictionary training and validation patients of an outer fold, stratified; the fold's patients are in neither."""
    rng = np.random.default_rng([seed, fold])
    train: list[str] = []
    val: list[str] = []
    for label in sorted(set(labels.values())):
        members = sorted(s for s, value in labels.items() if value == label and folds[s] != fold)
        order = rng.permutation(len(members))
        n_val = round(share * len(members))
        val += [members[i] for i in order[:n_val]]
        train += [members[i] for i in order[n_val:]]
    return sorted(train), sorted(val)


def segment_draw(subject: str, keys: Sequence[tuple[Any, ...]], n: int) -> list[int]:
    """Positions in ``keys`` of ``n`` segments, seeded by the patient ID; the input order of ``keys`` does not matter.

    Raises
    ------
    ValueError
        If the patient has fewer than ``n`` segments.
    """
    if len(keys) < n:
        raise ValueError(f"patient {subject} has {len(keys)} segments, fewer than {n}")
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    seed = int.from_bytes(hashlib.sha256(subject.encode()).digest()[:8], "big")
    chosen = np.random.default_rng(seed).choice(len(keys), size=n, replace=False)
    return sorted(order[int(i)] for i in chosen)


def low_pass(x: np.ndarray, sfreq: float, cutoff: float = LOW_PASS_HZ) -> np.ndarray:
    """Zero-phase Butterworth low-pass along the last axis; a flat row stays flat."""
    sos = butter(LOW_PASS_ORDER, cutoff, fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(x.dtype)


@dataclass
class PeakTable:
    """Every response peak of one patient: its segment position, its template and its height."""

    segment: np.ndarray
    template: np.ndarray
    height: np.ndarray
    n_segments: int
    n_templates: int


def patient_scale(table: PeakTable) -> tuple[np.ndarray, np.ndarray]:
    """Median and scaled MAD of the peak heights of every template; a template without peaks has infinite median."""
    median = np.full(table.n_templates, np.inf)
    scale = np.zeros(table.n_templates)
    order = np.argsort(table.template, kind="stable")
    templates, starts = np.unique(table.template[order], return_index=True)
    for template, values in zip(templates, np.split(table.height[order], starts[1:])):
        median[template] = np.median(values)
        scale[template] = MAD_TO_SCALE * np.median(np.abs(values - median[template]))
    return median, scale


def peak_counts(table: PeakTable, z: float) -> np.ndarray:
    """Peaks above the patient-relative threshold ``m + z a``, per segment and template ``(n_segments, n_templates)``.

    A template with ``a = 0`` counts nothing.
    """
    median, scale = patient_scale(table)
    usable = scale[table.template] > 0
    above = usable & (table.height > median[table.template] + z * scale[table.template])
    counts = np.zeros((table.n_segments, table.n_templates))
    np.add.at(counts, (table.segment[above], table.template[above]), 1.0)
    return counts


def rate_features(table: PeakTable, z: float, minutes_per_segment: float) -> np.ndarray:
    """The feature of section 5: ``log(1 + peaks above the threshold per minute)``, per segment and template."""
    return np.log1p(peak_counts(table, z) / minutes_per_segment)


def choose_z(
    tables: Sequence[PeakTable],
    minutes_per_segment: float,
    grid: Sequence[float] = Z_GRID,
    target: float = TARGET_RATE,
) -> tuple[float, dict[float, float]]:
    """The grid value whose median rate over templates, on these patients, is closest to ``target``; no labels.

    Exact ties go to the smaller ``z``. Returns ``z`` and the median rate at every grid value.
    """
    rates: dict[float, float] = {}
    for z in grid:
        per_patient = [peak_counts(t, z).sum(axis=0) / (t.n_segments * minutes_per_segment) for t in tables]
        rates[z] = float(np.median(np.mean(per_patient, axis=0)))
    return min(grid, key=lambda z: (abs(rates[z] - target), z)), rates


def patient_weights(subjects: np.ndarray) -> np.ndarray:
    """Window weights ``1 / n_i``, normalized to mean 1, so that every patient has the same total weight."""
    _, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * len(weights) / weights.sum()


def patient_mean(scores: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted patients, the mean score of each, and their window counts."""
    unique, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    return unique, np.bincount(inverse, weights=scores) / counts, counts


def _patient_labels(labels: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    unique, first = np.unique(subjects, return_index=True)
    return labels[first]


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    scaler_factory: Callable[[], Any],
    weights: np.ndarray | None,
    scaler_mask: np.ndarray | None,
) -> tuple[Any, LogisticRegression]:
    scaler = scaler_factory()
    scaler.fit(x if scaler_mask is None else x[scaler_mask])
    clf = LogisticRegression(C=c, max_iter=5000, solver="lbfgs")
    clf.fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, clf


def select_c(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    scaler_factory: Callable[[], Any],
    weights: np.ndarray | None = None,
    scaler_mask: np.ndarray | None = None,
    seed: int = 0,
    grid: Sequence[float] = C_GRID,
    inner_folds: int = INNER_FOLDS,
) -> tuple[float, dict[float, float]]:
    """``C`` by patient-grouped inner cross-validation on mean patient AUROC; exact ties go to the smaller ``C``."""
    splits = list(StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(x, y, subjects))
    means: dict[float, float] = {}
    for c in grid:
        values = []
        for train, val in splits:
            scaler, clf = _fit(
                x[train],
                y[train],
                c,
                scaler_factory,
                None if weights is None else weights[train],
                None if scaler_mask is None else scaler_mask[train],
            )
            _, pooled, _ = patient_mean(clf.decision_function(scaler.transform(x[val])), subjects[val])
            values.append(roc_auc_score(_patient_labels(y[val], subjects[val]), pooled))
        means[c] = float(np.mean(values))
    return max(grid, key=lambda c: (means[c], -c)), means


@dataclass
class HeadResult:
    """The held-out patients of one outer fold with their scores, and the head that produced them."""

    subject: np.ndarray
    label: np.ndarray
    score: np.ndarray
    n_windows: np.ndarray
    c: float
    coef: np.ndarray


def fit_head(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    held_out: np.ndarray,
    scaler_factory: Callable[[], Any],
    weighted: bool = False,
    scaler_mask: np.ndarray | None = None,
    seed: int = 0,
) -> HeadResult:
    """Fit the L2 head on the rows outside ``held_out`` and score the held-out patients (protocol section 6).

    ``weighted`` gives every training patient the same total weight. ``scaler_mask`` (over all rows) selects the
    rows that fit the scaler.

    Raises
    ------
    ValueError
        If a patient has rows on both sides of the fold.
    """
    train = ~held_out
    if set(subjects[train]) & set(subjects[held_out]):
        raise ValueError("a patient is on both sides of the fold")
    weights = patient_weights(subjects[train]) if weighted else None
    mask = None if scaler_mask is None else scaler_mask[train]
    c, _ = select_c(x[train], y[train], subjects[train], scaler_factory, weights, mask, seed)
    scaler, clf = _fit(x[train], y[train], c, scaler_factory, weights, mask)
    scores = clf.decision_function(scaler.transform(x[held_out]))
    unique, pooled, counts = patient_mean(scores, subjects[held_out])
    return HeadResult(unique, _patient_labels(y[held_out], subjects[held_out]), pooled, counts, c, clf.coef_[0])


def fit_sparse_head(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    held_out: np.ndarray,
    scaler_factory: Callable[[], Any],
    max_templates: int = SPARSE_TEMPLATES,
) -> tuple[HeadResult, np.ndarray]:
    """An L1 head with at most ``max_templates`` nonzero weights: the largest ``C`` of a path that keeps to the limit.

    Returns the result and the kept templates. With no usable ``C``, all scores are 0 and no template is kept.
    """
    train = ~held_out
    scaler = scaler_factory()
    scaler.fit(x[train])
    scaled_train, scaled_test = scaler.transform(x[train]), scaler.transform(x[held_out])
    best_c, best_coef, best_scores = 0.0, np.zeros(x.shape[1]), np.zeros(int(held_out.sum()))
    for c in np.geomspace(1e-3, 1.0, 20):
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=float(c), max_iter=2000)
        clf.fit(scaled_train, y[train])
        n_kept = int(np.count_nonzero(clf.coef_[0]))
        if n_kept > max_templates:
            break
        if n_kept > 0:
            best_c, best_coef, best_scores = float(c), clf.coef_[0].copy(), clf.decision_function(scaled_test)
    unique, pooled, counts = patient_mean(best_scores, subjects[held_out])
    result = HeadResult(unique, _patient_labels(y[held_out], subjects[held_out]), pooled, counts, best_c, best_coef)
    return result, np.flatnonzero(best_coef)


def permuted_scores(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    held_out: np.ndarray,
    c: float,
    scaler_factory: Callable[[], Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out patient scores of a head fitted on training labels shuffled between patients (the null, section 8)."""
    train = ~held_out
    unique, first = np.unique(subjects[train], return_index=True)
    shuffled = dict(zip(unique, rng.permutation(y[train][first])))
    scaler, clf = _fit(x[train], np.array([shuffled[s] for s in subjects[train]]), c, scaler_factory, None, None)
    patients, pooled, _ = patient_mean(clf.decision_function(scaler.transform(x[held_out])), subjects[held_out])
    return patients, pooled


@dataclass
class ArmScores:
    """Held-out patient scores of one arm over the outer folds, sorted by patient."""

    subject: np.ndarray
    fold: np.ndarray
    label: np.ndarray
    score: np.ndarray

    @classmethod
    def from_rows(cls, subject: np.ndarray, fold: np.ndarray, label: np.ndarray, score: np.ndarray) -> ArmScores:
        """The scores in sorted patient order, so that two arms align row by row."""
        order = np.argsort(subject)
        return cls(subject[order], fold[order], label[order], score[order])


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC with ties at one half; NaN if a class is empty."""
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return math.nan
    return float(_wins(positive, negative).mean())


def _wins(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])


def mean_fold_auroc(arm: ArmScores) -> float:
    """The statistic of section 6: the AUROC of each outer fold, averaged over the folds."""
    return float(np.mean([auroc(arm.label[arm.fold == f], arm.score[arm.fold == f]) for f in np.unique(arm.fold)]))


def pooled_auroc(arm: ArmScores) -> float:
    """The AUROC of all held-out scores together (reported, not used by the rule)."""
    return auroc(arm.label, arm.score)


def paired_interval(
    first: ArmScores, second: ArmScores, n_boot: int = N_BOOTSTRAP, seed: int = 0, confidence: float = CONFIDENCE
) -> tuple[float, float, float]:
    """Mean-fold AUROC of ``first`` minus ``second``, with a percentile interval from a patient bootstrap.

    Patients are resampled within fold and class, the same draw for both arms.

    Raises
    ------
    ValueError
        If the arms do not hold the same patients in the same folds.
    """
    if not (np.array_equal(first.subject, second.subject) and np.array_equal(first.fold, second.fold)):
        raise ValueError("the arms do not hold the same patients in the same folds")
    rng = np.random.default_rng(seed)
    folds = np.unique(first.fold)
    draws = np.zeros((n_boot, len(folds)))
    point = np.zeros(len(folds))
    for f, fold in enumerate(folds):
        where = first.fold == fold
        positive, negative = where & (first.label == 1), where & (first.label == 0)
        difference = _wins(first.score[positive], first.score[negative]) - _wins(
            second.score[positive], second.score[negative]
        )
        point[f] = difference.mean()
        n_pos, n_neg = difference.shape
        pos_draw = rng.integers(0, n_pos, size=(n_boot, n_pos))
        neg_draw = rng.integers(0, n_neg, size=(n_boot, n_neg))
        for b in range(n_boot):
            draws[b, f] = difference[np.ix_(pos_draw[b], neg_draw[b])].mean()
    means = draws.mean(axis=1)
    tail = 100 * (1 - confidence) / 2
    return float(point.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def conditions(intervals: dict[str, tuple[float, float, float]]) -> dict[str, bool | None]:
    """Whether each deciding comparison passes, from its interval; ``None`` when it is missing."""
    return {name: (intervals[name][1] > bound if name in intervals else None) for name, bound in DECIDING.items()}


def readout_outcome(
    passed: dict[str, bool | None], pass_index: int, gates_ok: bool | None, rate_guard_ok: bool | None
) -> str:
    """The outcome of section 7: ``blocked``, ``incomplete``, ``stop``, ``confirm`` or ``review``.

    The gates come first, because no classification result is read while they fail. A failing condition then
    decides at once, so a missing run cannot hide it. ``review`` means pass 2 passed: the explanation review
    decides between ``continue`` and ``stop``, and that verdict is recorded by hand.
    """
    if gates_ok is False:
        return "blocked"
    if gates_ok is None:
        return "incomplete"
    values = list(passed.values())
    if any(v is False for v in values):
        return "stop"
    if any(v is None for v in values):
        return "incomplete"
    if pass_index == 1:
        return "confirm"
    if rate_guard_ok is None:
        return "incomplete"
    return "review" if rate_guard_ok else "stop"


def rate_guard(
    first: ArmScores, second: ArmScores, rates: dict[str, float], min_per_class: int = MIN_RATE_GROUP
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Section 7's guard: the pooled difference within each native-rate group with enough patients of each class.

    Returns whether no such group has a wrong sign, and the groups, including the small ones that do not count.
    """
    group = np.array([rates[s] for s in first.subject])
    report: dict[str, dict[str, Any]] = {}
    ok = True
    for rate in np.unique(group):
        where = group == rate
        n_pos, n_neg = int((first.label[where] == 1).sum()), int((first.label[where] == 0).sum())
        difference = auroc(first.label[where], first.score[where]) - auroc(second.label[where], second.score[where])
        counts = min(n_pos, n_neg) >= min_per_class
        report[str(float(rate))] = {
            "epilepsy": n_pos,
            "non_epilepsy": n_neg,
            "difference": difference,
            "counts": counts,
        }
        if counts and difference < 0:
            ok = False
    return ok, report


def bag_size_link(arm: ArmScores, segments: dict[str, int]) -> dict[str, float]:
    """Spearman correlation of score and log segment count, within each class (protocol section 8)."""
    size = np.log([segments[s] for s in arm.subject])
    links = {}
    for label, name in ((1, "epilepsy"), (0, "non_epilepsy")):
        where = arm.label == label
        constant = np.ptp(size[where]) == 0 or np.ptp(arm.score[where]) == 0
        links[name] = math.nan if constant else float(spearmanr(arm.score[where], size[where]).statistic)
    return links


def planted_fraction(planted: np.ndarray, plain: np.ndarray) -> float:
    """Share of the patients whose rate is higher with planted events (the positive control, section 7)."""
    return float(np.mean(planted > plain))


def training_passed(trained: Sequence[float], untrained: Sequence[float]) -> bool:
    """Every trained dictionary leaves less validation residual than its untrained start (section 7)."""
    return all(a < b for a, b in zip(trained, untrained, strict=True))
