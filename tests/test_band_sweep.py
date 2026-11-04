"""Checks of the band-limited sweep driver ``src/band_sweep.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the dictionary paths (the base at seed 1 comes from the band-limited root), the sweep root record and its
refusals, the settings of each launcher stage, and ``select`` with ``read`` end to end on written score files: the
outcomes `adopt` and `base_holds` (by the margin, and with no selected setting), and `blocked` by the row gate.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import src.band_reconstruction_core as band  # noqa: E402
import src.band_sweep as driver  # noqa: E402
import src.band_sweep_core as core  # noqa: E402
import src.reconstruction_baselines_core as base  # noqa: E402

N_SEGMENTS = 2
WINNER = next(s for c, s in core.factorial() if c == (1, -1, -1, -1))  # K=128, the rest at the base


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _curve(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"epoch": [1, 2], "val_residual_frac": [0.95, 0.9]}).write_csv(folder / "sae_training.csv")


def _write(
    tmp: Path,
    screen_fast: dict[str, float],
    confirm_fast: float,
    *,
    shift_rows: bool = False,
) -> tuple[Path, Path, Path]:
    """A readout plan, a finished band-limited root and a sweep root with every screen and confirmation score.

    The residual over signal is 0.7 outside 4-30 Hz and, in 4-30 Hz, 0.92 for the base, ``screen_fast[name]`` for a
    screen setting (0.93 when absent) and ``confirm_fast`` for the selected setting on the confirmation folds.
    """
    readout, band_root, root = tmp / "readout", tmp / "band", tmp / "sweep"
    labels = {f"p{i:03d}": int(i % 2 == 0) for i in range(20)}
    folds = {s: i % 5 for i, s in enumerate(sorted(labels))}
    readout.mkdir(parents=True)
    plan = {"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds}
    (readout / "folds.json").write_text(json.dumps(plan))
    (band_root / "scores").mkdir(parents=True)
    (band_root / "band_decision.json").write_text("{}")
    driver.check_sweep_root(root, band_root)
    rng = np.random.default_rng(0)
    fast = list(band.FAST)
    signals = {}
    for fold in range(5):
        subject = np.repeat([s for s in sorted(labels) if folds[s] == fold], N_SEGMENTS)
        signal = rng.uniform(1.0, 2.0, (len(subject), len(base.BANDS)))
        signals[fold] = (subject, signal)
        residual = np.full((len(band.ARMS), *signal.shape), 0.9) * signal[None]
        reference = signal * (1.0 + 1e-6) if shift_rows else signal
        driver._save_npz(
            band_root / "scores" / f"fold{fold}.npz",
            subject=subject,
            arms=np.array(band.ARMS),
            signal_bands=reference,
            signal_total=reference.sum(axis=1),
            residual_bands=residual,
            residual_total=residual.sum(axis=2),
        )
        _curve(band_root / "dictionaries" / f"B4-trained_f{fold}")

    def score(setting: core.Setting, fold: int, fast_factor: float) -> None:
        subject, signal = signals[fold]
        factor = np.full(len(base.BANDS), 0.7)
        factor[fast] = fast_factor
        residual = signal * factor
        driver._save_npz(
            driver.score_file(root, setting, fold),
            subject=subject,
            fold=np.array(fold),
            setting=np.array(setting.name),
            bands=np.array(base.BAND_NAMES),
            ceiling_violations=np.array(0),
            ceiling_excess=np.array(-0.1),
            atom_wins=np.ones(setting.K, dtype=np.int64),
            waveforms=rng.standard_normal((setting.K, 16)),
            dominant_hz=np.linspace(1.0, 30.0, setting.K),
            signal_bands=signal,
            signal_total=signal.sum(axis=1),
            residual_bands=residual,
            residual_total=residual.sum(axis=1),
            ceiling_bands=0.3 * signal,
            ceiling_total=0.3 * signal.sum(axis=1),
        )
        if setting != core.BASE:
            _curve(driver.dictionary_path(root, band_root, setting, fold))

    for fold in core.SCREEN_FOLDS:
        score(core.BASE, fold, 0.92)
        for setting in core.screen_settings():
            score(setting, fold, screen_fast.get(setting.name, 0.93))
    for fold in core.CONFIRM_FOLDS:
        for setting in (core.BASE, core.BASE.with_seed(2)):
            score(setting, fold, 0.92)
        for setting in (WINNER, WINNER.with_seed(2)):
            score(setting, fold, confirm_fast)
    return root, band_root, readout


def test_dictionary_paths_and_stage_settings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root = Path(tmp) / "sweep", Path(tmp) / "band"
        assert driver.dictionary_path(root, band_root, core.BASE, 3) == band_root / "dictionaries" / "B4-trained_f3"
        assert driver.dictionary_path(root, band_root, core.BASE.with_seed(2), 3).parent == root / "dictionaries"
        assert driver.dictionary_path(root, band_root, WINNER, 0, smoke=True).parent == root / "smoke"
        assert len(driver.stage_settings("screen-fit")) == 18 and driver.stage_settings("screen-score")[0] == core.BASE
        root.mkdir()
        (root / "selection.json").write_text(json.dumps({"selected": None}))
        assert driver.stage_settings("confirm-fit", root) == [] and driver.stage_settings("confirm-score", root) == []
        (root / "selection.json").write_text(json.dumps({"selected": WINNER.name}))
        assert driver.stage_settings("confirm-fit", root) == [WINNER, WINNER.with_seed(2), core.BASE.with_seed(2)]
        assert driver.stage_settings("confirm-score", root)[0] == core.BASE


def test_sweep_root_record_and_refusals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, other = Path(tmp) / "sweep", Path(tmp) / "band", Path(tmp) / "other"
        driver.check_sweep_root(root, band_root)
        assert (root / "domain.json").is_file() and (root / "sweep.json").is_file()
        driver.check_sweep_root(root, band_root)
        _expect_value_error(lambda: driver.check_sweep_root(root, other), "another band-limited root was accepted")
        band_root.mkdir()
        (band_root / "band_decision.json").write_text("{}")
        _expect_value_error(lambda: driver.check_sweep_root(band_root, band_root), "a band-limited root was accepted")


def test_select_and_read_adopt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), {WINNER.name: 0.88}, confirm_fast=0.89)
        selection = driver.select_stage(root, band_root, readout)
        assert all(selection["gates"].values()), selection["gates"]
        assert selection["selected"] == WINNER.name
        effects = selection["effects"]["4_30"]
        assert effects["D"]["value"] < 0 < effects["K"]["value"], (
            "only the K=128 cell is better, so K helps and D hurts"
        )
        assert {"screen_summary.csv", "screen_effects.csv", "screen_budget_curve.csv"} <= {
            p.name for p in root.iterdir()
        }
        record = driver.read(root, band_root, readout)
        assert record["outcome"] == "adopt" and abs(record["mean_gain_4_30"] - 0.03) < 1e-9
        assert json.loads((root / "sweep_decision.json").read_text())["outcome"] == "adopt"


def test_read_base_holds_by_the_margin_or_without_a_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), {WINNER.name: 0.88}, confirm_fast=0.915)
        driver.select_stage(root, band_root, readout)
        record = driver.read(root, band_root, readout)
        assert record["outcome"] == "base_holds" and record["mean_gain_4_30"] < core.MARGIN
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), {}, confirm_fast=0.89)
        selection = driver.select_stage(root, band_root, readout)
        assert selection["selected"] is None and all(selection["gates"].values())
        assert driver.read(root, band_root, readout)["outcome"] == "base_holds"


def test_read_is_blocked_when_the_rows_differ() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), {WINNER.name: 0.88}, confirm_fast=0.89, shift_rows=True)
        selection = driver.select_stage(root, band_root, readout)
        assert not selection["gates"]["rows_identical"] and selection["selected"] is None
        assert driver.read(root, band_root, readout)["outcome"] == "blocked"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
