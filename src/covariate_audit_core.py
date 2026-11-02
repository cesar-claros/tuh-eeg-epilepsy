"""Pure logic of the covariate audit: patient covariates, matched pairs, pair bootstrap, medication status and rule.

Protocol: ``documentation/covariate_audit_protocol.md``. This module needs numpy, scipy and scikit-learn only, not
torch or the corpus. ``src/covariate_audit.py`` reads the files. The constants are fixed before any result
(protocol sections 6 to 9).
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.frozen_readout_core import CONFIDENCE, N_BOOTSTRAP, auroc

# Statistics (section 6).
CALIPER_YEARS = 5.0
CHANCE = 0.5
EXCLUSION_BOUND = 0.65  # only an upper bound below this lets a report exclude a signal above it
SECOND_LEVEL_C = 1e6  # the logistic models of the added-value read are not meant to shrink
# Rule and gates (section 7).
MIN_MATCHED_PER_CLASS = 30
HEADER_COVERAGE_MIN = 0.9
AGE_VALIDITY_MIN = 0.9
AGE_TOLERANCE_YEARS = 1.0
AGE_FIT_MIN = 0.9  # section 15: the second age gate, on the read age that the match uses
AGE_FIT_YEARS = CALIPER_YEARS
MAX_AGE = 120.0
MISSING_SEX = ""
# Medication (section 8).
MIN_STRATUM_PER_CLASS = 10
STRATA = ("off", "on")  # the first eligible group is read: the group without the drug comes first
# Controls (section 9).
N_CONTROL_DRAWS = 20
CONTROL_MARGIN = 0.03
CONTROL_AUROC = 0.70
CONTROL_RECOVERY = 0.05
# Drug read (section 16): three deciding reads share the error rate, and the control keeps the plain interval.
DRUG_FAMILY_CONFIDENCE = 1 - (1 - CONFIDENCE) / 3
MIN_DRUG_GROUP = MIN_STRATUM_PER_CLASS
# Descriptive reads (section 10).
SENSITIVITY_CALIPERS = (2.5, 10.0)
BAG_RATIO = 1.5
AGE_BANDS = (("below 18", 0.0, 18.0), ("18 to 39", 18.0, 40.0), ("40 to 64", 40.0, 65.0), ("65 and above", 65.0, 200.0))
# Inventory (section 5): text is listed only without digits and these words, so no age or sex can reach the output.
BLOCKED_NAME_TOKENS = frozenset({"age", "sex", "gender", "birth", "born", "dob"})
BLOCKED_TEXT_WORDS = frozenset(
    {"male", "female", "man", "men", "woman", "women", "boy", "girl", "year", "years", "old"}
)
TOP_VALUES, TOP_TOKENS, MAX_VALUE_LENGTH = 200, 400, 60


# ---------------------------------------------------------------------- patient covariates


def session_of(path: str) -> tuple[str, int]:
    """The session and year of an EDF path, such as ``("s001", 2003)`` from ``.../s001_2003/...``.

    Raises
    ------
    ValueError
        If the path does not hold exactly one session directory.
    """
    found = re.findall(r"/(s\d{3})_(\d{4})/", path)
    if len(found) != 1:
        raise ValueError(f"no session directory in {path}")
    return found[0][0], int(found[0][1])


def clean_age(value: float) -> float:
    """The age, or NaN if the header held none (the parser gives -1) or it is not plausible."""
    return float(value) if 0 <= value <= MAX_AGE else math.nan


def clean_sex(value: str) -> str:
    """``F`` or ``M``; anything else is missing."""
    return value if value in ("F", "M") else MISSING_SEX


def patient_age(values: Sequence[float]) -> float:
    """The mean of the usable ages of the patient's read segments; NaN if none is usable."""
    usable = [v for v in values if math.isfinite(v)]
    return float(np.mean(usable)) if usable else math.nan


def patient_sex(values: Sequence[str]) -> str:
    """The sex of the patient's read segments; a conflict or a missing value counts as missing."""
    return next(iter(set(values))) if len(set(values)) == 1 else MISSING_SEX


def age_validity(sessions: Mapping[str, Sequence[tuple[float, float]]]) -> tuple[float, int]:
    """Share of same-patient session pairs whose age difference matches their year difference, and the pair count.

    ``sessions`` holds ``(year, age)`` per session of each patient. The share is NaN without any pair.
    """
    within, total = 0, 0
    for items in sessions.values():
        usable = [(year, age) for year, age in items if math.isfinite(age)]
        for (year_a, age_a), (year_b, age_b) in itertools.combinations(usable, 2):
            total += 1
            within += abs((age_b - age_a) - (year_b - year_a)) <= AGE_TOLERANCE_YEARS
    return (within / total if total else math.nan), total


def patient_age_validity(sessions: Mapping[str, Sequence[tuple[float, float]]]) -> tuple[float, int]:
    """The mean over patients of each patient's share from ``age_validity``, and how many patients have a pair.

    Each patient counts once, so that a patient with many sessions cannot decide the gate (section 15). The value is
    NaN without any pair.
    """
    shares = []
    for subject, items in sessions.items():
        share, pairs = age_validity({subject: items})
        if pairs:
            shares.append(share)
    return (float(np.mean(shares)) if shares else math.nan), len(shares)


def median_birth_year(items: Sequence[tuple[float, float]]) -> float:
    """The median over a patient's usable sessions of year minus age; NaN if none is usable."""
    births = [year - age for year, age in items if math.isfinite(age)]
    return float(np.median(births)) if births else math.nan


def repaired_ages(
    subjects: np.ndarray, read_year: np.ndarray, sessions: Mapping[str, Sequence[tuple[float, float]]]
) -> np.ndarray:
    """Read year minus median birth year, the age of the repaired read (section 15)."""
    return read_year - np.array([median_birth_year(sessions.get(s, [])) for s in subjects])


def read_age_fit(
    subjects: np.ndarray,
    age: np.ndarray,
    repaired: np.ndarray,
    sessions: Mapping[str, Sequence[tuple[float, float]]],
) -> tuple[float, int]:
    """Share of checkable patients whose read age lies within the caliper of the repaired age, and their count.

    A patient is checkable with at least two usable sessions. The value is NaN without such a patient.
    """
    checkable = np.array([sum(math.isfinite(a) for _, a in sessions.get(s, [])) >= 2 for s in subjects])
    if not checkable.any():
        return math.nan, 0
    return float(np.mean(np.abs(age[checkable] - repaired[checkable]) <= AGE_FIT_YEARS)), int(checkable.sum())


def header_gates(age: np.ndarray, sex: np.ndarray, validity: float, fit: float) -> dict[str, Any]:
    """The header coverage gate of section 7 and the two age gates of section 15."""
    coverage = float(np.mean(np.isfinite(age) & (sex != MISSING_SEX)))
    return {
        "header_coverage": coverage,
        "header_coverage_ok": coverage >= HEADER_COVERAGE_MIN,
        "age_validity": validity,
        "age_validity_ok": bool(validity >= AGE_VALIDITY_MIN),
        "age_read_fit": fit,
        "age_read_fit_ok": bool(fit >= AGE_FIT_MIN),
    }


def gates_pass(gates: Mapping[str, Any]) -> bool:
    """Whether every gate holds."""
    return bool(gates["header_coverage_ok"] and gates["age_validity_ok"] and gates["age_read_fit_ok"])


# ---------------------------------------------------------------------- matched pairs


def pair_mask(
    label: np.ndarray, age: np.ndarray, sex: np.ndarray, caliper: float = CALIPER_YEARS, extra: np.ndarray | None = None
) -> np.ndarray:
    """The pairs of section 6, shape ``(epilepsy patients, non-epilepsy patients)``, in array order.

    A pair holds the same sex and an age difference of at most ``caliper``. A patient with a missing age or sex has
    no pair. ``extra`` is a further boolean matrix of the same shape.
    """
    pos, neg = label == 1, label == 0
    close = np.abs(age[pos][:, None] - age[neg][None, :]) <= caliper
    same = (sex[pos][:, None] == sex[neg][None, :]) & (sex[pos][:, None] != MISSING_SEX)
    return close & same if extra is None else close & same & extra


def all_pairs(label: np.ndarray, usable: np.ndarray | None = None) -> np.ndarray:
    """Every pair of usable patients, the mask of an unmatched read."""
    keep = np.ones(len(label), bool) if usable is None else usable
    return keep[label == 1][:, None] & keep[label == 0][None, :]


def group_pairs(label: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Pairs whose two patients share a value, such as a native rate."""
    return values[label == 1][:, None] == values[label == 0][None, :]


def ratio_pairs(label: np.ndarray, size: np.ndarray, ratio: float = BAG_RATIO) -> np.ndarray:
    """Pairs whose two sizes are within ``ratio`` of each other."""
    pos, neg = label == 1, label == 0
    return np.abs(np.log(size[pos])[:, None] - np.log(size[neg])[None, :]) <= math.log(ratio)


def standardize_within_fold(score: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Center each fold's scores and divide by their standard deviation (section 3); missing scores stay missing."""
    out = np.full(len(score), math.nan)
    for f in np.unique(fold):
        where = (fold == f) & np.isfinite(score)
        if where.any():
            sd = score[where].std()
            out[where] = (score[where] - score[where].mean()) / (sd if sd > 0 else 1.0)
    return out


def _wins(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return (positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])


@dataclass
class MatchedRead:
    """A statistic over pairs: its value and interval, the pair count and the patients per class in those pairs."""

    value: float
    low: float
    high: float
    pairs: int
    patients_epilepsy: int
    patients_non_epilepsy: int

    def as_dict(self) -> dict[str, Any]:
        """The read as plain values."""
        return asdict(self)


def _draw_counts(groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """How often each patient is drawn in each bootstrap draw, resampling within its group."""
    counts = np.zeros((n_boot, len(groups)))
    for group in np.unique(groups):
        where = np.flatnonzero(groups == group)
        counts[:, where] = rng.multinomial(len(where), np.full(len(where), 1.0 / len(where)), size=n_boot)
    return counts


def pair_bootstrap(
    values: np.ndarray,
    mask: np.ndarray,
    fold_pos: np.ndarray,
    fold_neg: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """A percentile interval of the mean of ``values`` over the pairs of ``mask``, by patient bootstrap (section 6).

    Patients are resampled within fold and class. A pair counts by the product of how often each of its two patients
    was drawn. Draws that hold no pair are left out. The interval is NaN if no pair exists.
    """
    if not mask.any():
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    pos, neg = _draw_counts(fold_pos, n_boot, rng), _draw_counts(fold_neg, n_boot, rng)
    weight = mask.astype(float)
    total = ((pos @ weight) * neg).sum(axis=1)
    summed = ((pos @ (values * weight)) * neg).sum(axis=1)
    draws = summed[total > 0] / total[total > 0]
    tail = 100 * (1 - confidence) / 2
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def matched_read(
    score: np.ndarray,
    label: np.ndarray,
    fold: np.ndarray,
    mask: np.ndarray,
    standardize: bool = True,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> MatchedRead:
    """The share of the pairs of ``mask`` in which the epilepsy patient has the higher score, ties at one half.

    Scores from the heads are standardized within each fold, so that pairs can be pooled over folds; pass
    ``standardize=False`` for a value that means the same in every fold, such as a band power.
    """
    z = standardize_within_fold(score, fold) if standardize else score
    pos, neg = label == 1, label == 0
    wins = _wins(z[pos], z[neg])
    value = float(wins[mask].mean()) if mask.any() else math.nan
    low, high = pair_bootstrap(wins, mask, fold[pos], fold[neg], n_boot, seed)
    return MatchedRead(value, low, high, int(mask.sum()), int(mask.any(axis=1).sum()), int(mask.any(axis=0).sum()))


def matched_difference(
    first: np.ndarray,
    second: np.ndarray,
    label: np.ndarray,
    fold: np.ndarray,
    mask: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> MatchedRead:
    """Matched AUROC of ``first`` minus ``second``, with the same bootstrap draws for both."""
    a, b = standardize_within_fold(first, fold), standardize_within_fold(second, fold)
    pos, neg = label == 1, label == 0
    difference = _wins(a[pos], a[neg]) - _wins(b[pos], b[neg])
    value = float(difference[mask].mean()) if mask.any() else math.nan
    low, high = pair_bootstrap(difference, mask, fold[pos], fold[neg], n_boot, seed)
    return MatchedRead(value, low, high, int(mask.sum()), int(mask.any(axis=1).sum()), int(mask.any(axis=0).sum()))


# ---------------------------------------------------------------------- models of the demographics


def _demographics(age: np.ndarray, sex: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Age, age squared and sex; the age scale comes from the training rows only."""
    mean, sd = age[train].mean(), age[train].std()
    scaled = (age - mean) / (sd if sd > 0 else 1.0)
    return np.column_stack([scaled, scaled**2, (sex == "M").astype(float)])


def residualize(x: np.ndarray, age: np.ndarray, sex: np.ndarray, train: np.ndarray) -> np.ndarray:
    """``x`` with age, age squared and sex regressed out of each feature; the coefficients use the training rows only.

    The refit read of section 6. All arrays hold one row per window.
    """
    design = np.column_stack([np.ones(len(x)), _demographics(age, sex, train)])
    beta, *_ = np.linalg.lstsq(design[train], x[train], rcond=None)
    return (x - design @ beta).astype(x.dtype)


def second_level_scores(
    label: np.ndarray, fold: np.ndarray, age: np.ndarray, sex: np.ndarray, score: np.ndarray | None = None
) -> np.ndarray:
    """Held-out values of ``M0`` (age, age squared, sex), or of ``M1`` when ``score`` is given (section 6).

    In each fold, the model fits on the patients of the other folds. ``score`` must be standardized within fold.
    All patients must have their covariates.
    """
    out = np.zeros(len(label))
    for f in np.unique(fold):
        train = fold != f
        x = _demographics(age, sex, train)
        x = x if score is None else np.column_stack([x, score])
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=SECOND_LEVEL_C, max_iter=5000).fit(scaler.transform(x[train]), label[train])
        out[~train] = model.decision_function(scaler.transform(x[~train]))
    return out


# ---------------------------------------------------------------------- controls and rule


def _pair_mean(score: np.ndarray, label: np.ndarray, fold: np.ndarray, mask: np.ndarray) -> float:
    z = standardize_within_fold(score, fold)
    return float(_wins(z[label == 1], z[label == 0])[mask].mean()) if mask.any() else math.nan


def control_reads(
    label: np.ndarray,
    fold: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    seed: int = 0,
    n_draws: int = N_CONTROL_DRAWS,
) -> dict[str, Any]:
    """The three controls of section 9, on the real pairs and with no EEG.

    A score of age plus noise must read near 0.5, a permuted score too, and a score with a known AUROC must be
    recovered. The AUROC of the age score over all pairs is reported, because it shows the age route that matching
    closes; it does not count, since age may hold no class signal.
    """
    mask = pair_mask(label, age, sex)
    usable = np.isfinite(age)
    scale = age[usable].std() if usable.any() else 1.0
    shift = math.sqrt(2.0) * NormalDist().inv_cdf(CONTROL_AUROC)
    rng = np.random.default_rng(seed)
    age_matched, age_unmatched, known, permuted = [], [], [], []
    for _ in range(n_draws):
        by_age = np.where(usable, age, 0.0) / (scale if scale > 0 else 1.0) + rng.standard_normal(len(label))
        age_matched.append(_pair_mean(by_age, label, fold, mask))
        age_unmatched.append(auroc(label[usable], by_age[usable]))
        planted = shift * label + rng.standard_normal(len(label))
        known.append(_pair_mean(planted, label, fold, mask))
        shuffled = planted.copy()
        for f in np.unique(fold):
            where = np.flatnonzero(fold == f)
            shuffled[where] = planted[rng.permutation(where)]
        permuted.append(_pair_mean(shuffled, label, fold, mask))
    report = {
        "age_plus_noise_matched": float(np.mean(age_matched)),
        "age_plus_noise_unmatched": float(np.mean(age_unmatched)),
        "permuted_matched": float(np.mean(permuted)),
        "known_matched": float(np.mean(known)),
        "known_target": CONTROL_AUROC,
    }
    report["passed"] = bool(
        abs(report["age_plus_noise_matched"] - CHANCE) <= CONTROL_MARGIN
        and abs(report["permuted_matched"] - CHANCE) <= CONTROL_MARGIN
        and abs(report["known_matched"] - CONTROL_AUROC) <= CONTROL_RECOVERY
    )
    return report


def audit_outcome(
    gates_ok: bool,
    patients_epilepsy: int,
    patients_non_epilepsy: int,
    primary: MatchedRead | None = None,
    refit: MatchedRead | None = None,
    hydra: MatchedRead | None = None,
    repaired: MatchedRead | None = None,
) -> str:
    """The outcome of section 7: ``blocked``, ``too_few``, ``holds``, ``review``, ``not_resolved`` or ``incomplete``.

    Only ``primary`` can give ``holds``. ``refit`` and ``hydra`` can only prevent ``not_resolved``. If ``repaired``
    disagrees with ``primary`` on ``holds``, the outcome is ``review`` (section 15).
    """
    if not gates_ok:
        return "blocked"
    if min(patients_epilepsy, patients_non_epilepsy) < MIN_MATCHED_PER_CLASS:
        return "too_few"
    if primary is None or repaired is None:
        return "incomplete"
    if (primary.low > CHANCE) != (repaired.low > CHANCE):
        return "review"
    if primary.low > CHANCE:
        return "holds"
    if refit is None or hydra is None:
        return "incomplete"
    return "review" if max(refit.low, hydra.low) > CHANCE else "not_resolved"


def excludes_above_bound(read: MatchedRead) -> bool:
    """Whether the upper bound lets a report write that a signal above the bound is excluded (section 7)."""
    return bool(read.high < EXCLUSION_BOUND)


# ---------------------------------------------------------------------- medication


def medication_status(entries: Sequence[str], terms: Sequence[str], empty_means: str) -> str:
    """``on``, ``off`` or ``unknown`` from one text per read session, empty if the session lists nothing (section 8).

    ``on`` if any session names a term of the mapping, as a whole word. Otherwise ``off`` if all sessions list
    medication. The addendum's ``empty_means`` (``off`` or ``unknown``) decides what an empty field means.

    Raises
    ------
    ValueError
        If ``terms`` is empty or ``empty_means`` is not known.
    """
    if not terms or empty_means not in ("off", "unknown"):
        raise ValueError("the mapping needs terms, and empty_means of off or unknown")
    if any(names_any(entry, terms) for entry in entries):
        return "on"
    if not entries or (empty_means == "unknown" and any(not entry.strip() for entry in entries)):
        return "unknown"
    return "off"


def medication_read(
    score: np.ndarray,
    label: np.ndarray,
    fold: np.ndarray,
    mask: np.ndarray,
    status: np.ndarray,
    min_per_class: int = MIN_STRATUM_PER_CLASS,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """The limit of section 8: ``survives``, ``unresolved`` or ``inseparable``, from the first eligible group.

    A group is eligible with at least ``min_per_class`` patients per class in its matched pairs. Scores are
    standardized over all patients of a fold, as for the other reads.
    """
    for group in STRATA:
        inside = status == group
        pairs = mask & inside[label == 1][:, None] & inside[label == 0][None, :]
        if min(pairs.any(axis=1).sum(), pairs.any(axis=0).sum()) >= min_per_class:
            read = matched_read(score, label, fold, pairs, n_boot=n_boot, seed=seed)
            result = "survives" if read.low > CHANCE else "unresolved"
            return {"result": result, "group": group, "read": read.as_dict()}
    return {"result": "inseparable", "group": None, "read": None}


# ---------------------------------------------------------------------- drug read


def _pattern(terms: Sequence[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


def names_any(text: str, terms: Sequence[str]) -> bool:
    """Whether the text names any of the terms, as whole words."""
    return bool(terms) and _pattern(terms).search(text) is not None


def drugs_named(text: str, drugs: Mapping[str, Sequence[str]]) -> set[str]:
    """The drugs whose spellings the text names; two spellings of one drug count once."""
    return {drug for drug, spellings in drugs.items() if names_any(text, spellings)}


@dataclass
class DrugExposure:
    """A patient's maintenance drugs at the read sessions (section 16)."""

    drugs_per_session: float
    older: bool
    newer_only: bool
    named: set[str]


def drug_exposure(
    entries: Sequence[str],
    older: Mapping[str, Sequence[str]],
    newer: Mapping[str, Sequence[str]],
    unnamed: Sequence[str],
) -> DrugExposure:
    """The exposure from one text per read session; a session that lists nothing does not count.

    The count is a mean per listing session, so that more sessions do not raise it. Bag size predicts the label, and
    a count of distinct drugs over all sessions would carry it. A drug named only as ``seizure medication`` counts as
    one drug and enters neither class.
    """
    counts, named_older, named_newer = [], set(), set()
    for entry in (e for e in entries if e.strip()):
        found_older, found_newer = drugs_named(entry, older), drugs_named(entry, newer)
        counts.append(len(found_older) + len(found_newer) + int(names_any(entry, unnamed)))
        named_older |= found_older
        named_newer |= found_newer
    return DrugExposure(
        float(np.mean(counts)) if counts else math.nan,
        bool(named_older),
        bool(named_newer) and not named_older,
        named_older | named_newer,
    )


@dataclass
class GroupRead:
    """The AUROC of a first group of patients above a second, with its interval and the group sizes."""

    value: float
    low: float
    high: float
    first: int
    second: int

    def as_dict(self) -> dict[str, Any]:
        """The read as plain values."""
        return asdict(self)


def group_read(
    score: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    fold: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> GroupRead:
    """AUROC of ``first`` above ``second``, ties at one half; patients are resampled within fold and group."""
    n_first, n_second = int(first.sum()), int(second.sum())
    if n_first == 0 or n_second == 0:
        return GroupRead(math.nan, math.nan, math.nan, n_first, n_second)
    wins = _wins(score[first], score[second])
    low, high = pair_bootstrap(wins, np.ones_like(wins, bool), fold[first], fold[second], n_boot, seed, confidence)
    return GroupRead(float(wins.mean()), low, high, n_first, n_second)


def rank_read(
    x: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """Spearman correlation with a percentile interval, from a patient bootstrap within fold."""
    rng = np.random.default_rng(seed)
    index = np.empty((n_boot, len(x)), dtype=int)
    for f in np.unique(fold):
        where = np.flatnonzero(fold == f)
        index[:, where] = rng.choice(where, size=(n_boot, len(where)))
    a, b = rankdata(x[index], axis=1), rankdata(y[index], axis=1)
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    scale = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    draws = (a * b).sum(axis=1)[scale > 0] / scale[scale > 0]
    tail = 100 * (1 - confidence) / 2
    return float(spearmanr(x, y).statistic), float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def partial_rank(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """The correlation of the ranks of ``x`` and ``y`` after a line on the ranks of ``z`` is taken out of each."""
    design = np.column_stack([np.ones(len(z)), rankdata(z)])

    def residual(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values)
        return ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]

    return float(np.corrcoef(residual(x), residual(y))[0, 1])


def drug_outcome(
    on_off: GroupRead | None,
    count: tuple[float, float, float] | None,
    drug_class: GroupRead | None,
    control: GroupRead | None,
) -> str:
    """The outcome of section 16: ``tracks``, ``nonspecific``, ``no_measured_tracking`` or ``incomplete``.

    On against off must lie above 0.5 and the count correlation above 0, because more drug can only raise the score
    if the score reads the drug. The two classes can differ in either direction. A read that is ``None`` had a group
    too small to read. If a drug class passes and listing a cardio-metabolic drug also raises the score, the read is
    ``nonspecific``: it follows drug listing or illness, not antiseizure treatment.
    """
    if count is None:
        return "incomplete"
    passed = count[1] > 0
    passed = passed or (on_off is not None and on_off.low > CHANCE)
    passed = passed or (drug_class is not None and (drug_class.low > CHANCE or drug_class.high < CHANCE))
    if not passed:
        return "no_measured_tracking"
    return "nonspecific" if control is not None and control.low > CHANCE else "tracks"


# ---------------------------------------------------------------------- spreadsheet inventory


def is_missing(value: Any) -> bool:
    """Whether a spreadsheet cell is empty: none, NaN or blank text."""
    return value is None or value != value or (isinstance(value, str) and not value.strip())


def is_flagged(value: Any) -> bool:
    """Whether a yes-or-no cell of the spreadsheet says yes."""
    return not is_missing(value) and str(value).strip().upper() == "Y"


def subject_token(value: Any, subjects: set[str]) -> str | None:
    """The first of ``subjects`` that the cell names, as a whole token of eight letters."""
    if is_missing(value):
        return None
    tokens = re.findall(r"(?<![a-z])[a-z]{8}(?![a-z])", str(value).lower())
    return next((token for token in tokens if token in subjects), None)


def session_token(value: Any) -> str | None:
    """The session that a cell names, as ``s001``: from ``s001``, ``s001_2003``, ``x_s001_t000`` or the number 1."""
    if is_missing(value):
        return None
    text = str(value).lower().strip()
    found = re.search(r"(?<![a-z0-9])s(\d{3})(?!\d)", text)
    if found:
        return f"s{found.group(1)}"
    try:
        number = float(text)
    except ValueError:
        return None
    return f"s{int(number):03d}" if number.is_integer() and 0 <= number < 1000 else None


def find_subject_column(columns: Mapping[str, Sequence[Any]], subjects: set[str]) -> str | None:
    """The column that names the most of ``subjects``, if it names at least 90% of them."""
    found = {name: {subject_token(v, subjects) for v in values} - {None} for name, values in columns.items()}
    best = max(found, key=lambda name: len(found[name]), default=None)
    return best if best is not None and len(found[best]) >= 0.9 * len(subjects) else None


def find_session_column(
    columns: Mapping[str, Sequence[Any]],
    rows: Sequence[int],
    owners: Sequence[str],
    read_sessions: set[tuple[str, str]],
) -> tuple[str | None, float]:
    """The column that best joins the read sessions, and the share it joins; a name with ``session`` wins a tie.

    ``rows`` are the kept rows and ``owners`` their patients. A column that joins under half of the read sessions
    does not count.
    """
    shares = {}
    for name, values in columns.items():
        joined = {(owner, session_token(values[row])) for row, owner in zip(rows, owners)}
        shares[name] = len(joined & read_sessions) / max(len(read_sessions), 1)
    best = max(shares, key=lambda name: (shares[name], "sess" in name.lower()), default=None)
    if best is None or shares[best] < 0.5:
        return None, 0.0
    return best, shares[best]


def _name_is_blocked(name: str) -> bool:
    return bool(set(re.split(r"[^a-z]+", name.lower())) & BLOCKED_NAME_TOKENS)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool | int | float | np.integer | np.floating):
        return True
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-]{2,}", text.lower())


def _listable_text(text: str) -> bool:
    return len(text) <= MAX_VALUE_LENGTH and not re.search(r"\d", text) and not set(_words(text)) & BLOCKED_TEXT_WORDS


def column_summary(name: str, values: Sequence[Any], listable: bool = True) -> dict[str, Any]:
    """Name, kind and fill rate of a column; for a text column, also its values and tokens, without any digit.

    The output of section 5 must hold no value of age or sex. So a column whose name says so, a number column, a date,
    and every listed value that holds a digit or a word such as ``male`` stay out.
    """
    filled = [v for v in values if not is_missing(v)]
    texts = [v for v in filled if isinstance(v, str) and not _is_number(v)]
    numbers = [v for v in filled if _is_number(v)]
    kind = "text" if len(texts) > len(filled) / 2 else ("number" if len(numbers) > len(filled) / 2 else "other")
    summary: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "filled": len(filled) / max(len(values), 1),
        "distinct": len({str(v) for v in filled}),
    }
    if kind == "text" and listable and not _name_is_blocked(name):
        cells = Counter(v.strip() for v in texts)
        tokens = Counter(token for v in texts for token in _words(v) if token not in BLOCKED_TEXT_WORDS)
        shown = [(v, n) for v, n in cells.most_common() if _listable_text(v)]
        summary["values"] = dict(shown[:TOP_VALUES])
        summary["tokens"] = dict(tokens.most_common(TOP_TOKENS))
    return summary


def inventory_sheet(
    name: str,
    columns: Mapping[str, Sequence[Any]],
    subjects: set[str],
    read_sessions: set[tuple[str, str]],
    subject_column: str | None = None,
    session_column: str | None = None,
) -> dict[str, Any]:
    """The inventory of one sheet (section 5), on the rows of ``subjects`` only.

    Other patients are dropped before any summary. Without a subject column no row can be kept, so the sheet reports
    only its column names.

    Raises
    ------
    ValueError
        If a named column does not exist.
    """
    for wanted in (subject_column, session_column):
        if wanted is not None and wanted not in columns:
            raise ValueError(f"sheet {name} has no column {wanted}")
    subject_column = subject_column or find_subject_column(columns, subjects)
    report: dict[str, Any] = {"sheet": name, "subject_column": subject_column, "session_column": None}
    if subject_column is None:
        report["column_names"] = list(columns)
        return report
    owners = [subject_token(v, subjects) for v in columns[subject_column]]
    rows = [i for i, owner in enumerate(owners) if owner is not None]
    kept = [owners[i] for i in rows]
    if session_column is None:
        session_column, share = find_session_column(columns, rows, kept, read_sessions)
    else:
        joined = {(owner, session_token(columns[session_column][i])) for i, owner in zip(rows, kept)}
        share = len(joined & read_sessions) / max(len(read_sessions), 1)
    report |= {
        "session_column": session_column,
        "rows_kept": len(rows),
        "patients_kept": len(set(kept)),
        "read_sessions_joined": share,
    }
    report["columns"] = [
        column_summary(column, [values[i] for i in rows], listable=column not in (subject_column, session_column))
        for column, values in columns.items()
    ]
    return report
