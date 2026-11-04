"""Pure logic of the band-limited hyperparameter sweep (``documentation/separable_band_sweep_protocol.md``).

Settings and their budgets, the 2^4 factorial and its contrasts, the scoring of one setting on a batch of rows, atom
health, and the selection and confirmation rules. It reuses ``src/band_reconstruction_core.py`` and
``src/reconstruction_baselines_core.py``, which stay unchanged. No corpus; the tensor functions run on CPU or GPU.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from src.band_reconstruction_core import FAST, rank_reconstruction
from src.reconstruction_baselines_core import (
    CEILING_TOLERANCE,
    SFREQ,
    Interval,
    band_energy,
    pooled_ev,
    window_energy,
)


@dataclass(frozen=True)
class Setting:
    """One configuration of the separable dictionary: atoms, waveform length, projections, window, atoms per window,
    epochs and seed (1 or 2)."""

    K: int
    L: int
    D: int
    P: int
    R: int
    epochs: int = 30
    seed: int = 1

    @property
    def name(self) -> str:
        """For example ``K64_L256_D4_P512_R4_E30_s1``."""
        return f"K{self.K}_L{self.L}_D{self.D}_P{self.P}_R{self.R}_E{self.epochs}_s{self.seed}"

    @property
    def amplitudes_per_second(self) -> float:
        """The amplitude budget ``R fs / P``."""
        return self.R * SFREQ / self.P

    @property
    def bits_per_second(self) -> float:
        """The choice bits ``R (log2 K + log2 P + log2 D) fs / P``: atom, position and projection of every amplitude."""
        return self.R * (math.log2(self.K) + math.log2(self.P) + math.log2(self.D)) * SFREQ / self.P

    def with_seed(self, seed: int) -> Setting:
        """The same setting at another seed."""
        return replace(self, seed=seed)


_NAME = re.compile(r"K(\d+)_L(\d+)_D(\d+)_P(\d+)_R(\d+)_E(\d+)_s([12])")


def parse_setting(name: str) -> Setting:
    """The setting of a name written by ``Setting.name``.

    Raises
    ------
    ValueError
        If the name does not match.
    """
    match = _NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"{name!r} is not a setting name")
    return Setting(*(int(v) for v in match.groups()))


BASE = Setting(K=64, L=256, D=4, P=512, R=4)  # the R=4 arm of the band-limited run
SECOND_SEED_OFFSET = 100  # seed 2 is the plan seed plus the fold plus this
# The 2^4 factorial (protocol section 4): each factor's low level is the base setting.
FACTORS: dict[str, tuple[dict[str, int], dict[str, int]]] = {
    "K": ({"K": 64}, {"K": 128}),
    "L": ({"L": 256}, {"L": 128}),
    "D": ({"D": 4}, {"D": 8}),
    "PR": ({"P": 512, "R": 4}, {"P": 256, "R": 2}),
}
DESCRIPTIVE = {
    "R8": replace(BASE, R=8),
    "R16": replace(BASE, R=16),
    "E60": replace(BASE, epochs=60),
}
SMOKE = (replace(BASE, K=128, D=8), replace(BASE, K=128, D=8, L=128))
SCREEN_FOLDS = (0, 1)
CONFIRM_FOLDS = (2, 3, 4)
MARGIN = 0.01  # the least mean gain in 4-30 Hz that adopts a setting
DEAD_SHARE_MAX = 0.10  # health: at most this share of atoms may win no held-out window, in each fold
NEAR_COPY_THRESHOLD = 0.9


def factorial() -> list[tuple[tuple[int, ...], Setting]]:
    """The 16 settings of the factorial with their codes (-1 low, +1 high, in ``FACTORS`` order); the base first."""
    out = []
    for codes in itertools.product((-1, 1), repeat=len(FACTORS)):
        fields: dict[str, int] = {}
        for code, (low, high) in zip(codes, FACTORS.values()):
            fields |= high if code > 0 else low
        out.append((codes, replace(BASE, **fields)))
    if out[0][1] != BASE or len({s for _, s in out}) != 2 ** len(FACTORS):
        raise ValueError("the factorial must hold 16 distinct settings with the base first")
    return out


def screen_settings() -> list[Setting]:
    """The new settings of the screen: 15 factorial settings and the descriptive arms (the base exists)."""
    return [s for _, s in factorial()[1:]] + list(DESCRIPTIVE.values())


def effect_names() -> list[str]:
    """Every main effect and interaction of the factorial, for example ``K`` and ``K:L:D``."""
    names = list(FACTORS)
    return [":".join(c) for size in range(1, len(names) + 1) for c in itertools.combinations(names, size)]


def contrast_signs() -> dict[str, np.ndarray]:
    """For every effect, the sign of each of the 16 settings (in ``factorial`` order) in its contrast."""
    codes = np.array([c for c, _ in factorial()])
    index = {name: i for i, name in enumerate(FACTORS)}
    return {e: np.prod(codes[:, [index[f] for f in e.split(":")]], axis=1) for e in effect_names()}


def contrasts(values: np.ndarray) -> dict[str, np.ndarray]:
    """Effects of values ``(..., 16)`` in ``factorial`` order: the mean at the high sign minus the mean at the low."""
    half = values.shape[-1] / 2
    return {e: (values * signs).sum(axis=-1) / half for e, signs in contrast_signs().items()}


def effect_intervals(
    residuals: np.ndarray, signal: np.ndarray, counts: np.ndarray, confidence: float = 0.95
) -> dict[str, Interval]:
    """Effects of the pooled EV of 16 settings, with percentile intervals from the patient bootstrap.

    ``residuals`` is ``(16, n_patients)`` in ``factorial`` order, ``signal`` ``(n_patients,)`` and ``counts``
    ``(n_boot, n_patients)``.
    """
    point = contrasts(np.array([pooled_ev(r, signal) for r in residuals]))
    draws = contrasts(np.stack([pooled_ev(r, signal, counts) for r in residuals], axis=-1))
    tail = (1.0 - confidence) / 2
    out = {}
    for name, value in point.items():
        low, high = np.quantile(draws[name], [tail, 1.0 - tail])
        out[name] = Interval(float(value), float(low), float(high))
    return out


# ---------------------------------------------------------------------- models and scoring


def separable_spec(setting: Setting, waveform_spec: Any) -> Any:
    """The ``SeparableSpec`` of a setting: the campaign's waveform fields with the setting's geometry, as
    ``separable_synthetic.build_model`` builds it (extended loss support), with ``D`` projections."""
    from src.models.components.separable_sae import SeparableSpec
    from src.models.components.waveform_sae import WaveformSpec

    fields = {name: getattr(waveform_spec, name) for name in WaveformSpec.__dataclass_fields__}
    fields |= {"n_atoms": setting.K, "atom_len": setting.L, "pool_len": setting.P, "top_r": setting.R}
    return SeparableSpec(**fields, n_projections=setting.D, loss_support="extended")


def check_setting(model: Any, setting: Setting) -> None:
    """Refuse a model whose geometry differs from its setting.

    Raises
    ------
    ValueError
        If ``K``, ``L``, ``D``, ``P``, ``R`` or the row domain differ.
    """
    spec = model.spec
    found = (spec.n_atoms, spec.atom_len, spec.n_projections, spec.pool_len, spec.top_r, spec.pre_emphasis)
    wanted = (setting.K, setting.L, setting.D, setting.P, setting.R, "none")
    if found != wanted:
        raise ValueError(f"model geometry {found} differs from {setting.name} {wanted}")


@torch.no_grad()
def score_setting(
    rows: torch.Tensor, valid: torch.Tensor, model: Any, weights: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Energies of one setting on encoded rows ``(B, C, T)``, with its own ceiling ``SVD-R`` on windows of its ``P``.

    Returns per segment ``signal_bands`` ``(B, n_bands)``, ``signal_total``, ``residual_bands``, ``residual_total``,
    ``ceiling_bands`` and ``ceiling_total``; and per batch ``ceiling_violations`` and ``ceiling_excess`` (scalars) and
    ``atom_wins`` ``(K,)``. All float64, except the counts.
    """
    pool, r = model.spec.pool_len, model.spec.top_r
    mask = valid.unsqueeze(-1).to(rows.dtype)
    code, _, reconstruction, _, _ = model._encode_reconstruct(rows, valid, model._dictionary())
    residual = rows - reconstruction * mask
    ceiling = rows - rank_reconstruction(rows, r, pool) * mask
    signal_windows = window_energy(rows, pool)
    gap = window_energy(ceiling, pool) - window_energy(residual, pool)
    won = code.unflatten(-1, (-1, pool)).amax(dim=-1) > 0
    return {
        "signal_bands": band_energy(rows, weights),
        "signal_total": rows.double().pow(2).sum(dim=(1, 2)),
        "residual_bands": band_energy(residual, weights),
        "residual_total": residual.double().pow(2).sum(dim=(1, 2)),
        "ceiling_bands": band_energy(ceiling, weights),
        "ceiling_total": ceiling.double().pow(2).sum(dim=(1, 2)),
        "ceiling_violations": (gap > CEILING_TOLERANCE * signal_windows).sum(),
        "ceiling_excess": (gap / signal_windows.clamp_min(1e-12)).amax(),
        "atom_wins": won.sum(dim=(0, 2)),
    }


def largest_near_copy_group(waveforms: np.ndarray, threshold: float = NEAR_COPY_THRESHOLD) -> int:
    """Size of the largest group of near-copy waveforms ``(K, L)``.

    Two waveforms are near-copies when their largest absolute normalized cross-correlation over all lags exceeds
    ``threshold``. Groups are the connected components of that relation. A lone waveform is a group of 1.
    """
    w = waveforms / np.maximum(np.linalg.norm(waveforms, axis=1, keepdims=True), 1e-12)
    n, length = w.shape
    spectra = np.fft.rfft(w, n=2 * length, axis=1)
    linked = np.zeros((n, n), dtype=bool)
    for i in range(n):
        cross = np.fft.irfft(spectra[i][None] * np.conj(spectra), n=2 * length, axis=1)
        linked[i] = np.abs(cross).max(axis=1) > threshold
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(linked)):
        parent[root(int(i))] = root(int(j))
    sizes = np.bincount([root(i) for i in range(n)], minlength=n)
    return int(sizes.max()) if n else 0


def healthy(atom_wins_per_fold: Sequence[np.ndarray]) -> bool:
    """At most ``DEAD_SHARE_MAX`` of the atoms win no held-out window, in each fold."""
    return all(float((wins == 0).mean()) <= DEAD_SHARE_MAX for wins in atom_wins_per_fold)


# ---------------------------------------------------------------------- rules


def select(
    ranked: Sequence[tuple[str, float]], versus_base: Mapping[str, Interval], health: Mapping[str, bool]
) -> str | None:
    """The selected setting (protocol section 7): the first, in descending order of screen EV in 4-30 Hz, whose
    interval against the base lies above 0 and that is healthy; ``None`` when none is."""
    for name, _ in sorted(ranked, key=lambda item: -item[1]):
        if versus_base[name].low > 0 and health[name]:
            return name
    return None


def confirmation_outcome(per_seed: Mapping[int, Interval], health_ok: bool, margin: float = MARGIN) -> str:
    """``adopt`` when, at both seeds, the selected setting beats the base (interval above 0), the mean point gain
    reaches ``margin`` and the setting is healthy; ``base_holds`` otherwise."""
    if set(per_seed) != {1, 2}:
        raise ValueError("the confirmation needs seeds 1 and 2")
    beats = all(interval.low > 0 for interval in per_seed.values())
    gain = float(np.mean([interval.value for interval in per_seed.values()]))
    return "adopt" if beats and gain >= margin and health_ok else "base_holds"


def fast_columns() -> list[int]:
    """The band columns of 4-30 Hz."""
    return list(FAST)
