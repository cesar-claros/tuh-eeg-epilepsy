"""Checks of the pure logic of the spectral decomposition and the fusion, ``src/spectral_decomposition_core.py`` (no
corpus; CPU).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure. Covered:
the 26 arms and their dimensions, the columns of every arm on a known layout, the level, the channel mean and the
asymmetry on known values, the refusal of a channel arm without a channel order, the layout gap of gate 4 on a right
and a wrong layout, the channel checks of gate 5, the rank-mean with ties and folds, and the rule. For the small-arm
lead: the cell statistic, the permutation that keeps the class counts of every fold, the selection null on a planted
effect and on scores without one, the permutation p-value, and the rule.
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402

import src.spectral_decomposition_core as core  # noqa: E402

CHANNELS = sorted(c for chain in core.CHAINS.values() for c in chain)  # the recorded order is alphabetical


def _expect(error: type[Exception], function, message: str) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError(message)


def _coded(rows: int = 3) -> np.ndarray:
    """Features whose value names its place: ``row * 10000 + channel * 100 + kind``."""
    cube = (
        np.arange(rows)[:, None, None] * 10000
        + np.arange(core.N_CHANNELS)[None, :, None] * 100
        + np.arange(core.PER_CHANNEL)[None, None, :]
    )
    return cube.reshape(rows, -1).astype(np.float64)


def _spectral(rows: int = 4, seed: int = 0) -> np.ndarray:
    """Features with the layout of ``SpectralFeatures``: log10 absolute powers, then relative powers."""
    power = np.random.default_rng(seed).uniform(0.5, 5.0, (rows, core.N_CHANNELS, core.N_BANDS))
    cube = np.concatenate([np.log10(power), power / power.sum(axis=2, keepdims=True)], axis=2)
    return cube.reshape(rows, -1)


def test_arms_and_dimensions() -> None:
    names = core.arm_names()
    assert len(names) == 26 and len(set(names)) == 26 and names[0] == core.FULL
    assert core.N_CHANNELS == 20 and core.PER_CHANNEL == 10
    x = _spectral()
    expected = {"Spec-full": 200, "Spec-abs": 100, "Spec-rel": 100, "Spec-level": 20, "Spec-chanmean": 10}
    expected |= {f"Spec-band-{b}": 40 for b in core.BANDS} | {f"Spec-drop-{b}": 160 for b in core.BANDS}
    expected |= {f"Spec-chain-{g}": 40 for g in core.CHAINS} | {f"Spec-drop-{g}": 160 for g in core.CHAINS}
    expected |= {"Spec-asym": 50}
    assert set(expected) == set(names)
    for name in names:
        assert core.arm_features(name, x, CHANNELS).shape == (4, expected[name]), name
    assert sum(core.needs_channels(n) for n in names) == 11
    _expect(ValueError, lambda: core.arm_features("Spec-other", x, CHANNELS), "an unknown arm must be refused")
    _expect(ValueError, lambda: core.arm_features(core.FULL, x[:, :-1], CHANNELS), "a wrong width must be refused")


def test_columns_of_every_arm() -> None:
    x = _coded()
    kind = lambda values: np.unique(values % 100).tolist()  # noqa: E731
    channel = lambda values: np.unique(values // 100 % 100).astype(int).tolist()  # noqa: E731
    assert np.array_equal(core.arm_features(core.FULL, x), x)
    assert kind(core.arm_features("Spec-abs", x)) == [0, 1, 2, 3, 4]
    assert kind(core.arm_features("Spec-rel", x)) == [5, 6, 7, 8, 9]
    assert kind(core.arm_features("Spec-band-8_13", x)) == [2, 7]
    assert kind(core.arm_features("Spec-drop-8_13", x)) == [0, 1, 3, 4, 5, 6, 8, 9]
    own = [CHANNELS.index(c) for c in core.CHAINS["central"]]
    assert channel(core.arm_features("Spec-chain-central", x, CHANNELS)) == sorted(own)
    rest = channel(core.arm_features("Spec-drop-central", x, CHANNELS))
    assert rest == sorted(set(range(core.N_CHANNELS)) - set(own))
    chains = [c for chain in core.CHAINS.values() for c in chain]
    assert sorted(chains) == CHANNELS  # the five chains hold every channel once
    # Every arm that selects columns keeps each row's own values.
    derived = {"Spec-level", "Spec-chanmean", "Spec-asym"}
    for name in set(core.arm_names()) - derived:
        assert np.all(core.arm_features(name, x, CHANNELS) // 10000 == np.arange(3)[:, None]), name


def test_level_mean_and_asymmetry() -> None:
    x = _spectral()
    cube = core.split_columns(x)
    power = 10.0 ** cube[:, :, : core.N_BANDS]
    assert np.allclose(core.arm_features("Spec-level", x), np.log10(power.sum(axis=2)))
    assert np.allclose(core.arm_features("Spec-chanmean", x), cube.mean(axis=1))
    asym = core.arm_features("Spec-asym", x, CHANNELS).reshape(4, len(core.LEFT_RIGHT), core.N_BANDS)
    left, right = (CHANNELS.index(name) for name in core.LEFT_RIGHT[2])
    assert np.allclose(asym[:, 2, :], cube[:, left, : core.N_BANDS] - cube[:, right, : core.N_BANDS])
    # A mirrored recording changes the sign of every asymmetry.
    mirror = cube.copy()
    for a, b in core.LEFT_RIGHT:
        i, j = CHANNELS.index(a), CHANNELS.index(b)
        mirror[:, [i, j]] = mirror[:, [j, i]]
    assert np.allclose(core.arm_features("Spec-asym", mirror.reshape(4, -1), CHANNELS), -asym.reshape(4, -1))


def test_arms_keep_the_saved_dtype() -> None:
    saved = _spectral().astype(np.float32)  # pass 1 fitted its head on float32 values
    for name in core.arm_names():
        assert core.arm_features(name, saved, CHANNELS).dtype == np.float32, name
    assert np.array_equal(core.arm_features(core.FULL, saved), saved)
    assert np.allclose(core.arm_features("Spec-level", saved), core.arm_features("Spec-level", _spectral()), atol=1e-5)


def test_channel_arms_need_the_order() -> None:
    x = _spectral()
    for name in ("Spec-asym", "Spec-chain-central", "Spec-drop-left_temporal"):
        _expect(ValueError, lambda name=name: core.arm_features(name, x), f"{name} must need the channel order")
        _expect(
            ValueError,
            lambda name=name: core.arm_features(name, x, CHANNELS[:-1] + ["Fz-Cz"]),
            f"{name} must refuse a wrong channel order",
        )
    assert core.arm_features("Spec-band-1_4", x).shape == (4, 40)  # a band arm needs no order


def test_layout_gap() -> None:
    x = _spectral()
    assert core.layout_gap(x) < 1e-12
    assert core.layout_gap(x.astype(np.float32)) < core.LAYOUT_TOLERANCE  # the saved features are float32
    cube = core.split_columns(x)
    swapped = np.concatenate([cube[:, :, core.N_BANDS :], cube[:, :, : core.N_BANDS]], axis=2).reshape(4, -1)
    assert core.layout_gap(swapped) > core.LAYOUT_TOLERANCE  # relative powers first: the wrong layout
    by_band = cube.reshape(4, core.N_CHANNELS, 2, core.N_BANDS).transpose(0, 1, 3, 2).reshape(4, -1)
    assert core.layout_gap(by_band) > core.LAYOUT_TOLERANCE  # absolute and relative power interleaved


def test_channel_problems() -> None:
    assert core.channel_problems(CHANNELS) == []
    assert core.channel_problems(CHANNELS[:-1])  # one channel short
    assert core.channel_problems(CHANNELS[:-1] + CHANNELS[:1])  # a repeated channel
    assert core.channel_problems(CHANNELS[:-1] + ["A1-T3"])  # an ear pair in place of a chain channel


def test_left_right_pairs_match_the_band_code_readout() -> None:
    try:
        from src.band_code_readout_core import LEFT_RIGHT
    except ImportError:  # torch is not installed: the equality is checked in the project container
        return
    assert core.LEFT_RIGHT == LEFT_RIGHT


def test_rank_mean() -> None:
    fold = np.array([0, 0, 0, 0, 1, 1])
    one = np.array([0.1, 0.4, 0.2, 0.3, 9.0, 5.0])
    assert np.allclose(core.rank_mean([one], fold), [0.25, 1.0, 0.5, 0.75, 1.0, 0.5])  # ranks inside each fold
    two = np.array([3.0, 3.0, 1.0, 2.0, 0.0, 1.0])  # a tie shares its mean rank
    assert np.allclose(core.rank_mean([two], fold), [0.875, 0.875, 0.25, 0.5, 0.5, 1.0])
    fused = core.rank_mean([one, two], fold)
    assert np.allclose(fused, (core.rank_mean([one], fold) + core.rank_mean([two], fold)) / 2)
    # Any increasing change of one arm's scale leaves the fusion unchanged.
    assert np.allclose(core.rank_mean([np.exp(one), 100 * two - 7], fold), fused)
    _expect(ValueError, lambda: core.rank_mean([], fold), "no scores must be refused")
    _expect(ValueError, lambda: core.rank_mean([one[:-1]], fold), "a wrong length must be refused")


def test_rule() -> None:
    assert core.fusion_outcome(False, True) == "blocked"
    assert core.fusion_outcome(False, False) == "blocked"
    assert core.fusion_outcome(True, True) == "fusion_gains"
    assert core.fusion_outcome(True, False) == "no_gain"
    assert core.FUSED == ("Spectral", "HYDRA", "REVE")


def _cell(seed: int, effect: float = 0.0, arms: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An untreated cell of 68 patients in 5 folds (10 of label 1), with ``arms`` score rows; arm 0 holds the effect."""
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(5), [16, 14, 13, 13, 12])
    label = np.zeros(68, dtype=int)
    for f, k in enumerate((3, 1, 3, 2, 1)):
        label[np.flatnonzero(fold == f)[:k]] = 1
    scores = rng.standard_normal((arms, 68))
    scores[0] += effect * label
    return scores, label, fold


def test_cell_statistic() -> None:
    score, label = np.array([3.0, 1.0, 2.0, 2.0, 0.0]), np.array([1, 1, 0, 0, 0])
    assert core.cell_statistic(score, label) == (3 + 1) / 6  # pairs won: (3;2,2,0) and (1;0)
    assert core.cell_statistic(np.array([1.0, 1.0]), np.array([1, 0])) == 0.5  # a tie counts one half
    _expect(ValueError, lambda: core.cell_statistic(score, np.zeros(5, dtype=int)), "an empty class must be refused")


def test_permutation_keeps_fold_counts() -> None:
    _, label, fold = _cell(0)
    rng = np.random.default_rng(1)
    changed = 0
    for _ in range(50):
        shuffled = core.permute_within_fold(label, fold, rng)
        for f in np.unique(fold):
            assert shuffled[fold == f].sum() == label[fold == f].sum()
        changed += int(not np.array_equal(shuffled, label))
    assert changed > 40 and label.sum() == 10  # the input is not changed, and most draws differ from it


def test_selection_null() -> None:
    scores, label, fold = _cell(2, effect=3.0)
    out = core.selection_null(scores, label, fold, n_perm=2000, seed=0)
    assert out["observed"].shape == (6,) and out["null"].shape == (2000, 6)
    for arm in range(6):  # the rank form equals the pair form
        assert abs(out["observed"][arm] - core.cell_statistic(scores[arm], label)) < 1e-12
    maxima = out["null"].max(axis=1)
    assert int(np.argmax(out["observed"])) == 0 and core.permutation_p(out["observed"].max(), maxima) < 0.01
    # The arms without an effect have a null near one half. The planted arm does not: its high scores sit in the
    # folds with the most label-1 patients, and the permutation keeps the folds.
    assert abs(out["null"][:, 1:].mean() - 0.5) < 0.03 and out["null"][:, 0].mean() > 0.5
    assert np.median(maxima) > 0.55  # the largest of six arms is biased upward
    # Without an effect, the selection p-value is spread over the unit interval.
    ps = []
    for seed in range(40):
        scores, label, fold = _cell(100 + seed)
        out = core.selection_null(scores, label, fold, n_perm=300, seed=seed)
        ps.append(core.permutation_p(out["observed"].max(), out["null"].max(axis=1)))
    assert 0.3 < np.mean(ps) < 0.7 and np.mean(np.array(ps) < 0.05) < 0.2
    _expect(ValueError, lambda: core.selection_null(scores[:, :-1], label, fold), "a wrong shape must be refused")
    _expect(ValueError, lambda: core.selection_null(scores, np.zeros(68, dtype=int), fold), "one class must be refused")


def test_permutation_p_and_lead_rule() -> None:
    null = np.array([0.4, 0.5, 0.6, 0.7])
    assert core.permutation_p(0.9, null) == 1 / 5  # the observed arrangement counts as one draw
    assert core.permutation_p(0.6, null) == 3 / 5 and core.permutation_p(0.1, null) == 1.0
    assert core.lead_outcome(False, True, True) == "blocked"
    assert core.lead_outcome(True, True, True) == "survives"
    assert core.lead_outcome(True, True, False) == "dropped" and core.lead_outcome(True, False, True) == "dropped"
    assert len(core.LEAD_SEEDS) == 20 and len(set(core.LEAD_SEEDS)) == 20 and core.LEAD_ARM == "Spec-chanmean"
    assert set(core.LEAD_ARMS) <= set(core.arm_names()) and core.LEAD_P == 0.05 and core.LEAD_MEDIAN == 0.65


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
