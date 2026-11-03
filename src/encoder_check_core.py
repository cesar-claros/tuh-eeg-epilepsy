"""Pure logic of the frozen encoder check: windows, scaling, heads, the gain over the drug, gates and rule.

Protocol: ``documentation/frozen_encoder_check_protocol.md``. This module needs numpy, scipy and scikit-learn only,
not torch or the corpus. ``src/encoder_check.py`` reads the signals and runs the encoder. The constants are fixed
before any result (protocol sections 3 to 7).
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.covariate_audit_core import _wins, pair_bootstrap, standardize_within_fold
from src.frozen_readout_core import C_GRID, CONFIDENCE, N_BOOTSTRAP, ArmScores, auroc, paired_interval

# Signal (section 3). Electrodes of the TCP montage common to all recordings; Fz and Pz are not in that montage.
ELECTRODES = ("C3", "C4", "Cz", "F3", "F4", "F7", "F8", "Fp1", "Fp2", "O1", "O2", "P3", "P4", "T3", "T4", "T5", "T6")
SFREQ = 200.0
BAND = (0.5, 99.5)
SEGMENT_SECONDS, SKIP_START_SECONDS = 120, 40
SEGMENT_SAMPLES = int(SEGMENT_SECONDS * SFREQ)
WINDOW_SAMPLES = int(10 * SFREQ)
PROBE_WINDOW_SAMPLES = int(5 * SFREQ)
PATCH_SAMPLES, PATCH_STEP = 200, 180  # the REVE defaults: 1 s patches with 0.1 s overlap
CLIP = 15.0
EMBED_DIM = 512
# Reads (section 5).
N_PROBE_WINDOWS = 60
N_SPLITS = 1000
STEP0_MARGIN = 0.10
TEST_SHARE, MLP_VAL_SHARE = 0.25, 0.15
MLP_HIDDEN = 256
MLP_ALPHAS = (1e-4, 1e-2, 1.0)
MLP_MAX_EPOCHS, MLP_PATIENCE = 200, 20
HIGH_RATES = (400.0, 512.0, 1000.0)
# The probe of Helwan et al. (section 2.9 of their paper), for read H only.
HELWAN_C = tuple(10.0**k for k in range(-6, 4))
HELWAN_VAL_SHARE = 0.2
HELWAN_INTERVAL = (0.596, 0.748)
# Gates (section 7).
N_PERMUTATIONS = 20
PERMUTATION_MARGIN = 0.03
AGE_MARGIN = 0.10
REPEAT_TOLERANCE = 1e-4
LOAD_FAILURE_MAX = 0.01
AGE_ALPHAS = tuple(10.0**k for k in range(-2, 5))


# ---------------------------------------------------------------------- windows and scaling


def tiled_length(n_samples: int, patch: int = PATCH_SAMPLES, step: int = PATCH_STEP) -> int:
    """The longest length of at most ``n_samples`` that REVE's patches tile exactly.

    Raises
    ------
    ValueError
        If not even one patch fits.
    """
    if n_samples < patch:
        raise ValueError(f"{n_samples} samples hold no patch of {patch}")
    return patch + (n_samples - patch) // step * step


def common_average(x: np.ndarray) -> np.ndarray:
    """Subtract the mean over channels (section 3, step 2); ``x`` is ``(channels, samples)``."""
    return x - x.mean(axis=0, keepdims=True)


def zscore_channels(x: np.ndarray, clip: float = CLIP) -> np.ndarray:
    """Z-score each channel over the segment and clip (step 4); a flat channel becomes 0."""
    sd = x.std(axis=-1, keepdims=True)
    scaled = (x - x.mean(axis=-1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    return np.clip(scaled, -clip, clip)


def scale_level(x: np.ndarray, mean: np.ndarray, sd: np.ndarray, clip: float = CLIP) -> np.ndarray:
    """The ``D-level`` scale: one mean and standard deviation per channel for all patients, so level survives."""
    return np.clip((x - mean[:, None]) / np.where(sd > 0, sd, 1.0)[:, None], -clip, clip)


def level_scale(sums: np.ndarray, squares: np.ndarray, counts: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean and standard deviation per channel over the training patients, from per-patient sums.

    ``sums`` and ``squares`` are ``(patients, channels)``, and ``counts`` the samples per patient.
    """
    total = counts[train].sum()
    mean = sums[train].sum(axis=0) / total
    variance = squares[train].sum(axis=0) / total - mean**2
    return mean, np.sqrt(np.maximum(variance, 0.0))


def segment_windows(segment: np.ndarray, window: int = WINDOW_SAMPLES) -> np.ndarray:
    """Windows ``(windows, channels, samples)`` without overlap, each cropped to a length that the patches tile."""
    n_windows = segment.shape[-1] // window
    length = tiled_length(window)
    return np.stack([segment[..., i * window : i * window + length] for i in range(n_windows)])


def embed_segment(
    segment: np.ndarray, encode: Callable[[np.ndarray], np.ndarray], window: int = WINDOW_SAMPLES
) -> np.ndarray:
    """The mean embedding of the windows of one scaled segment (step 6)."""
    return encode(segment_windows(segment, window).astype(np.float32)).mean(axis=0)


def random_windows(
    subject: str,
    n_segments: int,
    n: int = N_PROBE_WINDOWS,
    segment_samples: int = SEGMENT_SAMPLES,
    window_samples: int = PROBE_WINDOW_SAMPLES,
) -> list[tuple[int, int]]:
    """``(segment, start sample)`` of the windows of read ``H``, seeded by the patient ID, sorted by segment."""
    seed = int.from_bytes(hashlib.sha256(f"probe-{subject}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    segments = rng.integers(0, n_segments, size=n)
    starts = rng.integers(0, segment_samples - window_samples + 1, size=n)
    return sorted((int(a), int(b)) for a, b in zip(segments, starts))


def extra_recordings(
    rows: Sequence[Mapping[str, Any]],
    subjects: set[str],
    campaign_paths: set[str],
    locked: set[str],
    minimum_seconds: float = SKIP_START_SECONDS + SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """The recordings of the patients that the campaign left out for a seizure or a high native rate.

    Raises
    ------
    ValueError
        If a locked patient is among ``subjects``.
    """
    if subjects & locked:
        raise ValueError("a locked test patient is in the read")
    chosen = []
    for row in rows:
        seizure, high_rate = float(row["n_seizure"]) > 0, float(row["sfreq"]) in HIGH_RATES
        if (
            str(row["subject"]) in subjects
            and str(row["path"]) not in campaign_paths
            and (seizure or high_rate)
            and float(row["duration"]) >= minimum_seconds
        ):
            chosen.append({"row": row, "seizure": seizure, "high_rate": high_rate})
    return chosen


def unloaded_tensors(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray], head_prefix: str = "final_layer."
) -> list[str]:
    """Encoder tensors that differ between two loads of a checkpoint under different random seeds.

    A tensor read from the file is equal in both loads. A tensor that the file did not fill keeps its random start,
    so it differs. The loader does not raise for a missing weight, and a random encoder would read like a real one.
    The classification head is not used, so it is not checked. The check does not depend on how weights are named.
    """
    return sorted(k for k in first if not k.startswith(head_prefix) and not np.array_equal(first[k], second[k]))


def text_array(values: Iterable[Any]) -> np.ndarray:
    """A numpy array of plain strings; a pandas column gives an object array, which ``np.load`` refuses to read."""
    return np.array([str(v) for v in values])


def segment_starts(duration: float) -> list[float]:
    """The 2-minute grid of the campaign on one recording, after the skipped start."""
    count = max(0, int((duration - SKIP_START_SECONDS) // SEGMENT_SECONDS))
    return [float(SKIP_START_SECONDS + i * SEGMENT_SECONDS) for i in range(count)]


# ---------------------------------------------------------------------- heads


def patient_rows(x: np.ndarray, subject: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """The mean row of each of ``patients``, in that order.

    Raises
    ------
    ValueError
        If a patient has no row.
    """
    out = np.zeros((len(patients), x.shape[1]), dtype=x.dtype)
    order = {s: i for i, s in enumerate(patients)}
    counts = np.zeros(len(patients))
    for row, name in zip(x, subject):
        out[order[str(name)]] += row
        counts[order[str(name)]] += 1
    if (counts == 0).any():
        raise ValueError("a patient has no row")
    return out / counts[:, None]


def stratified_split(y: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    """A boolean mask of about ``share`` of the rows, within each class."""
    held = np.zeros(len(y), bool)
    for label in np.unique(y):
        where = rng.permutation(np.flatnonzero(y == label))
        held[where[: max(1, round(share * len(where)))]] = True
    return held


def linear_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int = 0,
    grid: Sequence[float] = C_GRID,
    val_share: float = MLP_VAL_SHARE,
    standardize: bool = True,
) -> np.ndarray:
    """L2 logistic regression with ``C`` chosen on a held-out share of the training rows; rows are patients.

    Pass ``standardize=False`` for rows from ``rotate_rows``, which a second scale would whiten.
    """
    held = stratified_split(y_train, val_share, np.random.default_rng(seed))
    if standardize:
        scaler = StandardScaler().fit(x_train)
        a, b = scaler.transform(x_train), scaler.transform(x_test)
    else:
        a, b = x_train, x_test
    best_c, best = grid[0], -math.inf
    for c in grid:
        model = LogisticRegression(C=c, max_iter=2000).fit(a[~held], y_train[~held])
        value = auroc(y_train[held], model.decision_function(a[held]))
        if value > best:
            best_c, best = c, value
    return LogisticRegression(C=best_c, max_iter=2000).fit(a, y_train).decision_function(b)


def mlp_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int = 0,
    max_epochs: int = MLP_MAX_EPOCHS,
    patience: int = MLP_PATIENCE,
) -> np.ndarray:
    """A 2-layer MLP: weight decay and stopping epoch chosen on a held-out share of the training rows (section 5.3).

    Ties go to the larger weight decay and the earlier epoch.
    """
    held = stratified_split(y_train, MLP_VAL_SHARE, np.random.default_rng(seed))
    scaler = StandardScaler().fit(x_train)
    a, b = scaler.transform(x_train), scaler.transform(x_test)
    best_model, best = None, -math.inf
    for alpha in sorted(MLP_ALPHAS, reverse=True):
        model = MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), alpha=alpha, random_state=seed, batch_size=32)
        since = 0
        for _ in range(max_epochs):
            model.partial_fit(a[~held], y_train[~held], classes=np.array([0, 1]))
            value = auroc(y_train[held], model.predict_proba(a[held])[:, 1])
            if value > best:
                best_model, best, since = copy.deepcopy(model), value, 0
            else:
                since += 1
                if since >= patience:
                    break
    return best_model.predict_proba(b)[:, 1]


def helwan_scores(
    x: np.ndarray, y: np.ndarray, subject: np.ndarray, held_out: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """The probe of read ``H``: class-balanced L2 logistic regression on windows, with ``C`` from 20% of the patients.

    Returns the held-out patients and the mean probability of their windows.
    """
    rows = np.flatnonzero(~held_out)
    patients = np.unique(subject[rows])
    first = {s: y[rows][subject[rows] == s][0] for s in patients}
    labels = np.array([first[s] for s in patients])
    validation = set(patients[stratified_split(labels, HELWAN_VAL_SHARE, np.random.default_rng(seed))])
    val_row = np.array([s in validation for s in subject[rows]])
    scaler = StandardScaler().fit(x[rows])
    a = scaler.transform(x[rows])
    best_c, best = HELWAN_C[0], -math.inf
    for c in HELWAN_C:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(a[~val_row], y[rows][~val_row])
        value = auroc(y[rows][val_row], model.decision_function(a[val_row]))
        if value > best:
            best_c, best = c, value
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(a, y[rows])
    probability = model.predict_proba(scaler.transform(x[held_out]))[:, 1]
    held_patients = np.unique(subject[held_out])
    return held_patients, np.array([probability[subject[held_out] == s].mean() for s in held_patients])


def fold_scores(
    x: np.ndarray, y: np.ndarray, fold: np.ndarray, fit: Callable[..., np.ndarray], seed: int
) -> np.ndarray:
    """Held-out scores of a head on patient rows, over the outer folds."""
    out = np.zeros(len(y))
    for f in np.unique(fold):
        test = fold == f
        out[test] = fit(x[~test], y[~test], x[test], seed=seed + int(f))
    return out


def split_spread(
    x: np.ndarray,
    y: np.ndarray,
    fit: Callable[..., np.ndarray],
    n_splits: int = N_SPLITS,
    seed: int = 0,
    test_share: float = TEST_SHARE,
) -> dict[str, float]:
    """Test AUROC over random patient splits, as in Cao et al.; it shows how much one split can read."""
    rng = np.random.default_rng(seed)
    values = []
    for i in range(n_splits):
        test = stratified_split(y, test_share, rng)
        values.append(auroc(y[test], fit(x[~test], y[~test], x[test], seed=seed + i)))
    draws = np.array(values)
    return {
        "splits": n_splits,
        "mean": float(draws.mean()),
        "sd": float(draws.std()),
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "highest": float(draws.max()),
    }


def rotate_rows(x: np.ndarray) -> np.ndarray:
    """Rows in the basis of their own span: an L2 logistic head on rows that are few and wide gives the same scores.

    The new columns are principal components, so they must not be standardized again: that would whiten them, give
    noise directions the weight of the signal, and bring a held-out AUROC to chance. Scale before this call only.
    """
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u * s


def rotated_linear_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int = 0) -> np.ndarray:
    """``linear_scores`` for rows from ``rotate_rows``, with no second scale."""
    return linear_scores(x_train, y_train, x_test, seed=seed, standardize=False)


def spread_is_consistent(fold_value: float, spread_mean: float, margin: float = STEP0_MARGIN) -> bool:
    """Whether a read over random splits agrees with the same head family over folds; a large gap means a fault."""
    return bool(abs(fold_value - spread_mean) <= margin)


def age_correlation(x: np.ndarray, subject: np.ndarray, age: Mapping[str, float], fold: Mapping[str, int]) -> float:
    """Spearman correlation of age with its held-out prediction, by a ridge head on rows, at patient level (gate 3)."""
    rows = np.array([math.isfinite(age[str(s)]) for s in subject])
    x, subject = x[rows], subject[rows].astype(str)
    target = np.array([age[s] for s in subject])
    folds = np.array([fold[s] for s in subject])
    predicted = np.zeros(len(subject))
    for f in np.unique(folds):
        test = folds == f
        scaler = StandardScaler().fit(x[~test])
        model = RidgeCV(alphas=AGE_ALPHAS).fit(scaler.transform(x[~test]), target[~test])
        predicted[test] = model.predict(scaler.transform(x[test]))
    patients = np.unique(subject)
    means = np.array([predicted[subject == s].mean() for s in patients])
    return float(spearmanr(means, [age[s] for s in patients]).statistic)


# ---------------------------------------------------------------------- reads


def alone_interval(arm: ArmScores, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float, float]:
    """Mean-fold AUROC of an arm with its patient bootstrap interval."""
    chance = ArmScores(arm.subject, arm.fold, arm.label, np.zeros(len(arm.score)))
    value, low, high = paired_interval(arm, chance, n_boot=n_boot, seed=seed)
    return value + 0.5, low + 0.5, high + 0.5


def gain_ceiling(label: np.ndarray, on: np.ndarray, known: np.ndarray) -> float:
    """The largest possible gain over the drug: half of the share of pairs that the drug ties (section 6)."""
    p, n = (label == 1) & known, (label == 0) & known
    ties = int(on[p].sum()) * int(on[n].sum()) + int((~on[p]).sum()) * int((~on[n]).sum())
    return 0.5 * ties / max(int(p.sum()) * int(n.sum()), 1)


def drug_gain(
    score: np.ndarray,
    label: np.ndarray,
    fold: np.ndarray,
    status: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """The gain over the drug on patients of known status: the drug ranks first and the score breaks ties.

    No model is fitted. Scores are standardized within fold over all patients, as in the report.
    """
    z = standardize_within_fold(score, fold)
    known = status != "unknown"
    on = (status == "on").astype(float)
    p, n = (label == 1) & known, (label == 0) & known
    combined = on * 1000.0 + z
    difference = _wins(combined[p], combined[n]).astype(float) - _wins(on[p], on[n]).astype(float)
    low, high = pair_bootstrap(difference, np.ones_like(difference, bool), fold[p], fold[n], n_boot, seed, confidence)
    return float(difference.mean()), low, high


# ---------------------------------------------------------------------- gates and rule


def gate_report(
    permuted_mean: float,
    age_encoder: float,
    age_spectral: float,
    repeat_max: float,
    load_failures: float,
    helwan_value: float,
) -> dict[str, Any]:
    """The gates of section 7. ``status`` is ``blocked``, ``review`` (only read ``H`` missed) or ``pass``."""
    report: dict[str, Any] = {
        "permuted_mean": permuted_mean,
        "permuted_ok": bool(abs(permuted_mean - 0.5) <= PERMUTATION_MARGIN),
        "age_encoder": age_encoder,
        "age_spectral": age_spectral,
        "age_ok": bool(age_encoder >= age_spectral - AGE_MARGIN),
        "repeat_max": repeat_max,
        "repeat_ok": bool(repeat_max <= REPEAT_TOLERANCE),
        "load_failures": load_failures,
        "load_ok": bool(load_failures <= LOAD_FAILURE_MAX),
        "helwan_value": helwan_value,
        "helwan_ok": bool(HELWAN_INTERVAL[0] <= helwan_value <= HELWAN_INTERVAL[1]),
    }
    hard = report["permuted_ok"] and report["age_ok"] and report["repeat_ok"] and report["load_ok"]
    report["status"] = "blocked" if not hard else ("pass" if report["helwan_ok"] else "review")
    return report


def encoder_outcome(difference: tuple[float, float, float], gain: tuple[float, float, float]) -> str:
    """The outcome of section 1, from the intervals of ``REVE`` minus ``Spectral`` and of the gain over the drug."""
    higher, adds = difference[1] > 0, gain[1] > 0
    if higher:
        return "encoder_adds" if adds else "chart_reader"
    return "review" if adds else "no_advantage"
