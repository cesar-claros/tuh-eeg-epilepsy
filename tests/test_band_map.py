"""Checks of the map stage of ``src/band_sweep.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the map list stage at both seeds, ``map_stage`` end to end on written score files (`shifts`, `stable` with no
gain at R=16 or with an unhealthy large dictionary, and `blocked` by the row gate, with the files it writes), and the
two-seed read of amendment A1 (`shifts` when both seeds agree, `stable` when the large dictionary is unhealthy at one
seed, the seed noise and the replication of effects).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import src.band_map_core as core  # noqa: E402
import src.band_reconstruction_core as band  # noqa: E402
import src.band_sweep as driver  # noqa: E402
import src.band_sweep_core as sweep  # noqa: E402
import src.reconstruction_baselines_core as base  # noqa: E402

N_SEGMENTS = 2


def _curve(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"epoch": [1, 2], "val_residual_frac": [0.95, 0.9]}).write_csv(folder / "sae_training.csv")


def _fast_factor(setting: sweep.Setting, k128_gain_at_16: float) -> float:
    """The 4-30 Hz residual over signal of a cell: lower for larger R, L=128 and D=8; K=128 hurts at R=4."""
    factor = 0.92 - 0.03 * np.log2(setting.R / 4) - 0.01 * (setting.L == 128) - 0.003 * (setting.D == 8)
    factor += 0.004 * (setting.K == 128) * (setting.R == 4) + 0.002 * (setting.K == 32)
    factor -= k128_gain_at_16 * (setting.K == 128) * (setting.R == 16)
    return float(factor)


def _write(
    tmp: Path,
    k128_gain_at_16: float,
    *,
    dead_k128_at_16: bool = False,
    shift_rows: bool = False,
    seed2: dict[str, float | bool] | None = None,
) -> tuple[Path, Path, Path]:
    """A readout plan, a band-limited root and a sweep root with the map cells at seed 1, and with ``seed2`` (its
    ``k128_gain_at_16``, ``dead_k128_at_16`` and ``shift``, a constant added to the 4-30 Hz residual factor) also at
    seed 2."""
    readout, band_root, root = tmp / "readout", tmp / "band", tmp / "sweep"
    labels = {f"p{i:03d}": int(i % 2 == 0) for i in range(20)}
    folds = {s: i % 5 for i, s in enumerate(sorted(labels))}
    readout.mkdir(parents=True)
    (readout / "folds.json").write_text(
        json.dumps({"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds})
    )
    (band_root / "scores").mkdir(parents=True)
    (band_root / "band_decision.json").write_text("{}")
    driver.check_sweep_root(root, band_root)
    rng = np.random.default_rng(0)
    signals = {}
    for fold in sweep.SCREEN_FOLDS:
        subject = np.repeat([s for s in sorted(labels) if folds[s] == fold], N_SEGMENTS)
        signal = rng.uniform(1.0, 2.0, (len(subject), len(base.BANDS)))
        signals[fold] = (subject, signal)
        reference = signal * (1.0 + 1e-6) if shift_rows else signal
        residual = np.full((len(band.ARMS), *signal.shape), 0.9) * signal[None]
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
    plans = [(core.map_settings(1), k128_gain_at_16, dead_k128_at_16, 0.0)]
    if seed2 is not None:
        plans.append(
            (
                core.map_settings(2),
                float(seed2.get("k128_gain_at_16", k128_gain_at_16)),
                bool(seed2.get("dead_k128_at_16", False)),
                float(seed2.get("shift", 0.0)),
            )
        )
    for settings, gain, dead, shift in plans:
        _write_cells(root, band_root, signals, settings, gain, dead, shift, rng)
    return root, band_root, readout


def _write_cells(
    root: Path,
    band_root: Path,
    signals: dict[int, tuple[np.ndarray, np.ndarray]],
    settings: list[sweep.Setting],
    gain: float,
    dead: bool,
    shift: float,
    rng: np.random.Generator,
) -> None:
    for setting in settings:
        for fold in sweep.SCREEN_FOLDS:
            subject, signal = signals[fold]
            factor = np.full(len(base.BANDS), 0.7)
            factor[list(band.FAST)] = _fast_factor(setting, gain) + shift
            residual = signal * factor
            wins = np.ones(setting.K, dtype=np.int64)
            if dead and setting.K == 128 and setting.R == 16:
                wins[:20] = 0
            driver._save_npz(
                driver.score_file(root, setting, fold),
                subject=subject,
                fold=np.array(fold),
                setting=np.array(setting.name),
                bands=np.array(base.BAND_NAMES),
                ceiling_violations=np.array(0),
                ceiling_excess=np.array(-0.1),
                atom_wins=wins,
                waveforms=rng.standard_normal((setting.K, 16)),
                dominant_hz=np.linspace(1.0, 30.0, setting.K),
                signal_bands=signal,
                signal_total=signal.sum(axis=1),
                residual_bands=residual,
                residual_total=residual.sum(axis=1),
                ceiling_bands=0.3 * signal,
                ceiling_total=0.3 * signal.sum(axis=1),
            )
            if setting != sweep.BASE:
                _curve(driver.dictionary_path(root, band_root, setting, fold))


def test_map_list_stage() -> None:
    assert driver.stage_settings("map") == core.map_settings()
    second = driver.stage_settings("map", seed=2)
    assert len(second) == 36 and all(s.seed == 2 for s in second), "seed 2 repeats the 36 cells without convergence"


def test_map_shifts_when_a_larger_dictionary_wins_at_r16() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.02)
        record = driver.map_stage(root, band_root, readout)
        assert all(record["gates"].values()), record["gates"]
        assert record["best_healthy"]["R4"] == "K64_L128_D8_P512_R4_E30_s1"
        assert record["best_healthy"]["R16"] == "K128_L128_D8_P512_R16_E30_s1"
        assert record["answer"] == "shifts" and record["reads"]["R8_vs_R4"]["answer"] == "stable"
        assert record["effects"]["4_30"]["R16-R4:K128"]["value"] > 0
        assert {"map.json", "map_summary.csv", "map_effects.csv", "map_front.csv"} <= {p.name for p in root.iterdir()}
        assert pl.read_csv(root / "map_summary.csv").height == 36


def test_map_stable_without_gain_or_with_unhealthy_large_dictionaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.0)
        assert driver.map_stage(root, band_root, readout)["answer"] == "stable"
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.02, dead_k128_at_16=True)
        record = driver.map_stage(root, band_root, readout)
        assert record["best_healthy"]["R16"] == "K64_L128_D8_P512_R16_E30_s1" and record["answer"] == "stable"


def test_map_is_blocked_when_the_rows_differ() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.02, shift_rows=True)
        record = driver.map_stage(root, band_root, readout)
        assert record["answer"] == "blocked" and not record["gates"]["rows_identical"]


def test_map_at_seed_two_writes_its_own_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.02, seed2={})
        record = driver.map_stage(root, band_root, readout, seed=2)
        assert record["seed"] == 2 and record["answer"] == "shifts" and "convergence_E60_minus_E30_at_R16" not in record
        assert record["best_healthy"]["R16"] == "K128_L128_D8_P512_R16_E30_s2"
        assert (root / "map_s2.json").is_file() and not (root / "map.json").exists(), "seed 1 outputs must stay apart"


def test_two_seed_read_shifts_when_both_seeds_agree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.02, seed2={"shift": 0.001})
        record = driver.map_seeds_stage(root, band_root, readout)
        assert all(record["gates"].values()), record["gates"]
        assert record["answer"] == "shifts" and record["answer_each_seed"] == {"s1": "shifts", "s2": "shifts"}
        assert abs(record["seed_noise_4_30"]["median_abs"] - 0.001) < 1e-9
        assert record["K128_minus_K64_at_R16_L128_D8_4_30"]["value"] > 0
        rep = record["replication"]["4_30"]
        assert rep["R16-R4:K128"]["replicates"] and rep["R4:L128"]["replicates"]
        assert not rep["R4:K*L"]["replicates"], "a null effect must not replicate"
        assert {"map_seeds.json", "map_seeds_summary.csv", "map_seeds_effects.csv"} <= {p.name for p in root.iterdir()}
        assert pl.read_csv(root / "map_seeds_summary.csv").height == 36


def test_two_seed_read_needs_health_at_both_seeds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, band_root, readout = _write(Path(tmp), k128_gain_at_16=0.02, seed2={"dead_k128_at_16": True})
        record = driver.map_seeds_stage(root, band_root, readout)
        assert record["answer_each_seed"] == {"s1": "shifts", "s2": "stable"}
        assert record["best_healthy"]["R16"] == "K64_L128_D8_P512_R16_E30_s1" and record["answer"] == "stable"
    try:
        driver.map_seeds_stage(Path("unused"), Path("unused"), Path("unused"), seeds=(1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("one seed twice was accepted")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
