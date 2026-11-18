"""Pure logic of the within-patient drug protocol: the side of a session, the eligible patients and contrasts, the
measures, the paired tests and the rule.

Protocol: ``documentation/drug_signature_detection_protocol.md``. ``src/drug_signature.py`` holds the code that reads
the spreadsheet and the signals. The constants are fixed before any result (protocol sections 5, 8 and 10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# A contrast and what makes a session "with": a mechanism group, or any maintenance drug (section 5).
CONTRASTS = {"W-sodium": "sodium", "W-gaba": "gaba", "W-any": "any", "W-lev": "sv2a"}
# The deciding tests: the measure and the expected sign of "with" minus "without" (section 8). W-lev decides nothing.
DECIDING = {"W-sodium": ("M-alpha-frequency", -1), "W-gaba": ("M-beta", 1), "W-any": ("M-level", 1)}
MEASURES = ("M-alpha-frequency", "M-beta", "M-level", "M-slow")
MIN_SEGMENTS = 4  # segments on each side for a patient to enter a contrast
MIN_PATIENTS = 15  # eligible patients for a contrast to be tested
MIN_TIME_GROUP = 10  # "added" patients and control patients for the test against time
N_FLIPS = 10_000  # sign flips, and permutations of the test against time
ALPHA = 0.05
N_CHANNELS, N_BANDS = 20, 5  # the band-power layout: per channel, 5 log absolute powers and then 5 relative powers
BAND_NAMES = ("1_4", "4_8", "8_13", "13_30", "30_45")
ALPHA_RANGE = (6.0, 13.0)  # Hz: the range of the alpha frequency measure
POSTERIOR = ("T5-O1", "T6-O2", "P3-O1", "P4-O2")  # the channels of the alpha frequency measure
OUTCOMES = ("within_patient_effect", "time_or_drug", "no_effect_measured", "too_few_patients", "blocked")
# Amendment A1 (section 20.3): the strata of the test against time.
STRATA = (
    "same_year|same_acquisition",
    "same_year|other_acquisition",
    "later_year|same_acquisition",
    "later_year|other_acquisition",
)


# ---------------------------------------------------------------------- sessions


def drug_groups(mechanisms: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """The mechanism group of every drug.

    Raises
    ------
    ValueError
        If a drug is in two groups.
    """
    owner: dict[str, str] = {}
    for group, drugs in mechanisms.items():
        for drug in drugs:
            if owner.setdefault(drug, group) != group:
                raise ValueError(f"{drug} is in the groups {owner[drug]} and {group}")
    return owner


def session_side(contrast: str, listed: bool, maintenance: bool, unnamed: bool, groups: set[str]) -> str:
    """The side of a session for a contrast: ``with``, ``without`` or ``unknown`` (section 5).

    ``listed``: the session lists at least one drug of any kind. ``maintenance``: it lists a maintenance drug.
    ``unnamed``: it lists a maintenance drug without its name ("seizure medication"), which can belong to any
    group. ``groups``: the mechanism groups of the named drugs.

    Raises
    ------
    KeyError
        If the contrast is not one of ``CONTRASTS``.
    """
    target = CONTRASTS[contrast]
    if not listed:
        return "unknown"
    if target == "any":
        return "with" if maintenance else "without"
    if target in groups:
        return "with"
    return "unknown" if unnamed else "without"


def direction(with_orders: Sequence[float], without_orders: Sequence[float]) -> str:
    """``added`` if the sessions with the group come later on average than those without it, else ``removed``."""
    return "added" if float(np.mean(with_orders)) > float(np.mean(without_orders)) else "removed"


def consistent(sides: Sequence[str]) -> bool:
    """Whether the known sides, in session order, change exactly once: the drug comes and stays, or leaves and
    stays away (section 6, item 5)."""
    known = [side for side in sides if side != "unknown"]
    return sum(a != b for a, b in zip(known, known[1:], strict=False)) == 1


def patient_entry(
    sides: Sequence[str],
    segments: Sequence[int | None],
    orders: Sequence[int],
    years: Sequence[int],
    acquisitions: Sequence[Any],
) -> dict[str, Any] | None:
    """A patient in a contrast, from its sessions in order; ``None`` if the patient does not have both sides.

    ``segments`` holds the segments of each session, or ``None`` where the count is not known. ``eligible`` is then
    ``None``: the patient has both sides, and the floor of ``MIN_SEGMENTS`` is not checked.
    """
    with_index = [i for i, side in enumerate(sides) if side == "with"]
    without_index = [i for i, side in enumerate(sides) if side == "without"]
    if not with_index or not without_index:
        return None
    counted = all(segments[i] is not None for i in with_index + without_index)
    n_with = sum(segments[i] for i in with_index) if counted else None  # type: ignore[misc]
    n_without = sum(segments[i] for i in without_index) if counted else None  # type: ignore[misc]
    used = with_index + without_index
    return {
        "sessions_with": len(with_index),
        "sessions_without": len(without_index),
        "segments_with": n_with,
        "segments_without": n_without,
        "eligible": (n_with >= MIN_SEGMENTS and n_without >= MIN_SEGMENTS) if counted else None,
        "direction": direction([orders[i] for i in with_index], [orders[i] for i in without_index]),
        "consistent": consistent(sides),
        "same_acquisition": len({acquisitions[i] for i in used}) == 1,
        "gap_years": abs(float(np.mean([years[i] for i in with_index]) - np.mean([years[i] for i in without_index]))),
    }


def control_halves(n: int) -> tuple[list[int], list[int]]:
    """The earlier and the later half of ``n`` sessions in order; with an odd count the middle session is earlier.

    Raises
    ------
    ValueError
        If there are fewer than two sessions.
    """
    if n < 2:
        raise ValueError("a control patient needs two sessions")
    cut = (n + 1) // 2
    return list(range(cut)), list(range(cut, n))


def control_entry(sides: Sequence[str], segments: Sequence[int | None]) -> dict[str, Any] | None:
    """A control patient of a contrast: two or more sessions of a known side, all on the same side (section 6,
    item 6). ``None`` otherwise. Sessions of an unknown side are not counted."""
    known = [i for i, side in enumerate(sides) if side != "unknown"]
    if len(known) < 2 or len({sides[i] for i in known}) != 1:
        return None
    earlier, later = control_halves(len(known))
    earlier, later = [known[i] for i in earlier], [known[i] for i in later]
    counted = all(segments[i] is not None for i in known)
    n_earlier = sum(segments[i] for i in earlier) if counted else None  # type: ignore[misc]
    n_later = sum(segments[i] for i in later) if counted else None  # type: ignore[misc]
    return {
        "side": sides[known[0]],
        "earlier": earlier,
        "later": later,
        "segments_earlier": n_earlier,
        "segments_later": n_later,
        "eligible": (n_earlier >= MIN_SEGMENTS and n_later >= MIN_SEGMENTS) if counted else None,
    }


def stratum(gap_years: float, same_acquisition: bool) -> str:
    """The stratum of a patient for the test against time (amendment A1): whether the two sides, or the two halves
    of a control patient, are in the same calendar year, and whether all their sessions have the same acquisition."""
    year = "same_year" if gap_years == 0 else "later_year"
    return f"{year}|{'same' if same_acquisition else 'other'}_acquisition"


def control_stratum(entry: Mapping[str, Any], years: Sequence[int], acquisitions: Sequence[Any]) -> str:
    """The stratum of a control patient, from its entry of ``control_entry`` and its sessions in order."""
    earlier, later = entry["earlier"], entry["later"]
    gap = abs(float(np.mean([years[i] for i in later]) - np.mean([years[i] for i in earlier])))
    return stratum(gap, len({acquisitions[i] for i in earlier + later}) == 1)


def contrast_eligible(n_patients: int) -> bool:
    """Whether a contrast has enough eligible patients to be tested."""
    return n_patients >= MIN_PATIENTS


# ---------------------------------------------------------------------- measures


def band_cube(features: np.ndarray) -> np.ndarray:
    """The band-power features ``(segments, 200)`` as ``(segments, channels, 10)``: 5 log absolute powers and
    5 relative powers per channel.

    Raises
    ------
    ValueError
        If the features do not have ``N_CHANNELS * 2 * N_BANDS`` columns.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != N_CHANNELS * 2 * N_BANDS:
        raise ValueError(f"band-power features need {N_CHANNELS * 2 * N_BANDS} columns, got {features.shape}")
    return features.reshape(len(features), N_CHANNELS, 2 * N_BANDS)


def alpha_frequency(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float] = ALPHA_RANGE) -> np.ndarray:
    """The mean frequency in ``band``, weighted by power, of each spectrum ``(segments, frequencies)``; NaN for a
    spectrum with no power in the band."""
    psd, freqs = np.asarray(psd, dtype=np.float64), np.asarray(freqs, dtype=np.float64)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    power = psd[:, inside]
    total = power.sum(axis=1)
    weighted = (power * freqs[inside]).sum(axis=1)
    return np.divide(weighted, total, out=np.full(len(psd), np.nan), where=total > 0)


def measure_values(features: np.ndarray, alpha: np.ndarray) -> dict[str, np.ndarray]:
    """The four measures of section 8, one value per segment."""
    absolute = band_cube(features)[:, :, :N_BANDS]
    return {
        "M-alpha-frequency": np.asarray(alpha, dtype=np.float64),
        "M-beta": absolute[:, :, 3].mean(axis=1),
        "M-level": absolute[:, :, :4].mean(axis=(1, 2)),
        "M-slow": (absolute[:, :, 1] - absolute[:, :, 2]).mean(axis=1),
    }


def side_mean(values: np.ndarray, session: np.ndarray, keep: np.ndarray) -> float:
    """The mean over the sessions in ``keep`` of the session means of ``values``; sessions count equally."""
    names = np.unique(session[keep])
    return float(np.mean([np.nanmean(values[keep & (session == name)]) for name in names]))


def patient_difference(values: np.ndarray, session: np.ndarray, side: np.ndarray) -> float:
    """``with`` minus ``without`` for the segments of one patient (section 8).

    Raises
    ------
    ValueError
        If a side has no segment.
    """
    values, session, side = np.asarray(values, dtype=np.float64), np.asarray(session), np.asarray(side)
    first, second = side == "with", side == "without"
    if not first.any() or not second.any():
        raise ValueError("a difference needs segments on both sides")
    return side_mean(values, session, first) - side_mean(values, session, second)


# ---------------------------------------------------------------------- tests


def t_value(differences: np.ndarray) -> np.ndarray:
    """The t-value of each row of differences ``(..., patients)``; 0 where the differences are all 0."""
    d = np.asarray(differences, dtype=np.float64)
    mean = d.mean(axis=-1)
    error = d.std(axis=-1, ddof=1) / np.sqrt(d.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(error > 0, mean / error, np.where(mean == 0, 0.0, np.sign(mean) * np.inf))


def sign_flip_test(
    differences: Sequence[float], expected: int, n_flips: int = N_FLIPS, seed: int = 0
) -> dict[str, Any]:
    """The test against no change (section 8): the t-value of the patient differences against random sign flips,
    one-sided in the ``expected`` direction (1 or -1).

    Raises
    ------
    ValueError
        If there are fewer than two patients, or ``expected`` is not 1 or -1.
    """
    d = np.asarray(differences, dtype=np.float64)
    if len(d) < 2 or expected not in (1, -1):
        raise ValueError("the test needs two or more patients and an expected sign of 1 or -1")
    observed = float(t_value(d))
    signs = np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=(n_flips, len(d)))
    flipped = t_value(signs * d)
    beyond = int((expected * flipped >= expected * observed).sum())
    return {
        "patients": len(d),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "share_expected": float((expected * d > 0).mean()),
        "t": observed,
        "p": (1 + beyond) / (1 + n_flips),
    }


def time_test(
    added: Sequence[float], control: Sequence[float], expected: int, n_perm: int = N_FLIPS, seed: int = 0
) -> dict[str, Any]:
    """The test against time as section 8 wrote it: the differences of the ``added`` patients against the
    later-minus-earlier differences of the control patients, by a permutation of the two groups, one-sided. ``ran``
    is false with fewer than ``MIN_TIME_GROUP`` patients in a group. Since amendment A1 it is a descriptive read;
    ``stratified_time_test`` decides."""
    a, c = np.asarray(added, dtype=np.float64), np.asarray(control, dtype=np.float64)
    record: dict[str, Any] = {"ran": False, "added": len(a), "control": len(c)}
    if len(a) < MIN_TIME_GROUP or len(c) < MIN_TIME_GROUP:
        return record
    pooled = np.concatenate([a, c])
    observed = float(a.mean() - c.mean())
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_perm, len(pooled))), axis=1)
    shuffled = pooled[order]
    null = shuffled[:, : len(a)].mean(axis=1) - shuffled[:, len(a) :].mean(axis=1)
    beyond = int((expected * null >= expected * observed).sum())
    return record | {
        "ran": True,
        "mean_added": float(a.mean()),
        "mean_control": float(c.mean()),
        "difference": observed,
        "p": (1 + beyond) / (1 + n_perm),
    }


def stratified_time_test(
    added: Sequence[float],
    added_strata: Sequence[str],
    control: Sequence[float],
    control_strata: Sequence[str],
    expected: int,
    n_perm: int = N_FLIPS,
    seed: int = 0,
) -> dict[str, Any]:
    """The test against time of amendment A1 (section 20.3): each ``added`` patient against the control patients of
    its stratum. The statistic is the mean over the ``added`` patients of their difference minus the mean control
    difference of their stratum. The permutation exchanges patients inside a stratum only, one-sided. A stratum
    with no ``added`` or no control patient leaves the test. ``ran`` is false with fewer than ``MIN_TIME_GROUP``
    patients of a group in the strata that enter.

    Raises
    ------
    ValueError
        If the values and the strata of a group differ in length.
    """
    a, c = np.asarray(added, dtype=np.float64), np.asarray(control, dtype=np.float64)
    a_strata, c_strata = np.asarray(added_strata, dtype=str), np.asarray(control_strata, dtype=str)
    if len(a) != len(a_strata) or len(c) != len(c_strata):
        raise ValueError("each patient needs one stratum")
    names = sorted(set(a_strata.tolist()) | set(c_strata.tolist()))
    sizes = {name: [int((a_strata == name).sum()), int((c_strata == name).sum())] for name in names}
    entering = [name for name in names if min(sizes[name]) > 0]
    n_added = sum(sizes[name][0] for name in entering)
    n_control = sum(sizes[name][1] for name in entering)
    record: dict[str, Any] = {
        "ran": False,
        "stratified": True,
        "added": n_added,
        "control": n_control,
        "strata": sizes,
        "strata_left_out": [name for name in names if name not in entering],
    }
    if n_added < MIN_TIME_GROUP or n_control < MIN_TIME_GROUP:
        return record
    rng = np.random.default_rng(seed)
    observed, null = 0.0, np.zeros(n_perm)
    for name in entering:
        mine, theirs = a[a_strata == name], c[c_strata == name]
        weight = len(mine) / n_added
        observed += weight * float(mine.mean() - theirs.mean())
        pooled = np.concatenate([mine, theirs])
        shuffled = pooled[np.argsort(rng.random((n_perm, len(pooled))), axis=1)]
        null += weight * (shuffled[:, : len(mine)].mean(axis=1) - shuffled[:, len(mine) :].mean(axis=1))
    beyond = int((expected * null >= expected * observed).sum())
    return record | {
        "ran": True,
        "mean_added": float(a[np.isin(a_strata, entering)].mean()),
        "mean_control": float(c[np.isin(c_strata, entering)].mean()),
        "difference": observed,
        "p": (1 + beyond) / (1 + n_perm),
    }


def passes_change(p: float, n_contrasts: int) -> bool:
    """Whether a deciding contrast passes the test against no change, with ``n_contrasts`` eligible ones."""
    return p <= ALPHA / n_contrasts


def passes_time(record: Mapping[str, Any]) -> bool:
    """Whether the test against time ran and passed at ``ALPHA``."""
    return bool(record.get("ran")) and record["p"] <= ALPHA


def signature_outcome(
    gates_ok: bool, eligible: Sequence[str], change: Mapping[str, bool], time: Mapping[str, bool]
) -> str:
    """The rule of section 10, on the eligible deciding contrasts."""
    if not gates_ok:
        return "blocked"
    if not eligible:
        return "too_few_patients"
    passed = [name for name in eligible if change[name]]
    if not passed:
        return "no_effect_measured"
    return "within_patient_effect" if any(time[name] for name in passed) else "time_or_drug"
