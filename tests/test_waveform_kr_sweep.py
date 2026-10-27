"""Decision rules of the K x R x initialization reconstruction sweep (src/waveform_kr_sweep.py); synthetic, CPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.waveform_kr_sweep import (  # noqa: E402
    EV_TOLERANCE,
    health,
    init_verdict,
    knee,
    parse_arm,
    summarize,
    tolerance,
)


def _arm(k: int, r: int, explained: float, init: str = "patches", seed: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "arm": f"K{k}_R{r}_{init}_s{seed}",
        "k": k,
        "r": r,
        "init": init,
        "seed": seed,
        "frozen": False,
        "val_explained": explained,
        "health": "healthy",
        **extra,
    }


def test_arm_names() -> None:
    assert parse_arm("K64_R1_patches_s0") == {
        "arm": "K64_R1_patches_s0",
        "k": 64,
        "r": 1,
        "init": "patches",
        "seed": 0,
        "frozen": False,
    }
    assert parse_arm("K16_R4_random_s1_frozen")["frozen"]
    for bad in ("K64_R1_patch_s0", "f5_r1_s0", "K4_R8_patches_s0"):
        with pytest.raises(ValueError):
            parse_arm(bad)


def test_health_thresholds_scale_with_k() -> None:
    assert health(64, 8, 0.10, False) == "healthy"
    assert health(64, 12, 0.05, False) == "watch"
    assert health(64, 16, 0.05, False) == "starving"
    assert health(64, 0, 0.25, False) == "starving"
    assert health(16, 2, 0.40, False) == "healthy"  # 6.4 / 16
    assert health(16, 2, 1.0, False) == "starving"  # 16 / 16
    assert health(128, 0, 0.02, True) == "starving"
    assert health(64, float("nan"), float("nan"), False) == "watch"


def test_tolerance_grows_with_the_seed_spread() -> None:
    assert tolerance([_arm(64, 1, 0.20)]) == (EV_TOLERANCE, None)
    tol, spread = tolerance([_arm(64, 1, 0.200), _arm(64, 1, 0.205, seed=1)])
    assert spread == pytest.approx(0.005) and tol == pytest.approx(0.015)


def test_knee_takes_the_smallest_k_within_tolerance_and_skips_starving_arms() -> None:
    arms = [_arm(16, 1, 0.15), _arm(32, 1, 0.19), _arm(64, 1, 0.20), _arm(128, 1, 0.205)]
    arms += [_arm(16, 2, 0.20), _arm(64, 2, 0.24), _arm(128, 2, 0.30, health="starving")]
    arms += [_arm(32, 4, 0.20), _arm(64, 4, 0.22), _arm(128, 4, 0.30)]
    result = knee(arms, 0.01)
    assert result[1]["knee_k"] == 64 and result[1]["best_k"] == 128 and result[1]["saturated"]
    assert result[2]["knee_k"] == 64 and result[2]["starving_k"] == [128]
    assert result[4]["knee_k"] == 128 and not result[4]["saturated"]  # still rising at the largest K
    assert knee([_arm(64, 1, 0.2, health="starving")], 0.01)[1]["knee_k"] is None


def test_initialization_verdict_needs_every_matched_pair() -> None:
    patches = [_arm(32, 1, 0.18), _arm(64, 1, 0.20)]
    better = [_arm(32, 1, 0.20, init="random"), _arm(64, 1, 0.22, init="random")]
    assert init_verdict(patches + better, 0.01)["verdict"] == "random_better"
    mixed = [_arm(32, 1, 0.20, init="random"), _arm(64, 1, 0.20, init="random")]
    assert init_verdict(patches + mixed, 0.01)["verdict"] == "no_difference"
    worse = [_arm(32, 1, 0.15, init="random"), _arm(64, 1, 0.17, init="random")]
    assert init_verdict(patches + worse, 0.01)["verdict"] == "patches_better"
    starving = [_arm(32, 1, 0.20, init="random", health="starving"), _arm(64, 1, 0.22, init="random")]
    assert init_verdict(patches + starving, 0.01)["verdict"] == "no_difference"  # better but less healthy
    assert init_verdict(patches, 0.01)["verdict"] is None


def _write(root: Path, name: str, explained: list[float], never_won: list[float], k: int) -> None:
    folder = root / name
    folder.mkdir()
    epochs = len(explained)
    pl.DataFrame(
        {
            "epoch": [float(e + 1) for e in range(epochs)],
            "val_objective": [1.0 - v for v in explained],
            "val_full_residual_frac": [1.0 - v for v in explained],
            "val_full_gain": [0.97] * epochs,
            "n_atoms_never_won": never_won,
            "max_won_share": [2.0 / k] * epochs,
        }
    ).write_csv(folder / "sae_training.csv")
    pl.DataFrame({"atom": list(range(k)), "energy_share": [1.0 / k] * k, "won_windows": [5] * k}).write_csv(
        folder / "sae_atom_stats.csv"
    )


def test_summarize_reads_a_sweep_root(tmp_path: Path) -> None:
    for k, ev in ((16, 0.15), (32, 0.19), (64, 0.20), (128, 0.205)):
        _write(tmp_path, f"K{k}_R1_patches_s0", [ev - 0.02, ev - 0.01, ev], [0.0, 0.0, 1.0], k)
    _write(tmp_path, "K64_R1_random_s0", [0.14, 0.16, 0.17], [0.0, 2.0, 3.0], 64)
    _write(tmp_path, "K64_R1_patches_s1", [0.18, 0.19, 0.201], [0.0, 0.0, 1.0], 64)
    _write(tmp_path, "K64_R1_patches_s0_frozen", [0.11], [0.0], 64)
    (tmp_path / "K128_R2_patches_s0").mkdir()  # an unfinished arm is skipped
    decision = summarize(tmp_path)
    assert decision["knee_by_r"]["1"]["knee_k"] == 64 and decision["knee_by_r"]["1"]["saturated"]
    assert decision["init"]["verdict"] == "patches_better"
    assert decision["tolerance"] == pytest.approx(0.01) and decision["seed_spread"] == pytest.approx(0.001)
    assert decision["frozen_explained"] == {"K64_R1_patches_s0_frozen": pytest.approx(0.11)}
    assert set(decision["not_converged"]) >= {"K64_R1_patches_s0", "K128_R1_patches_s0"}
    assert decision["campaign"].startswith("K=64 is the knee at R=1")
    assert json.loads((tmp_path / "kr_sweep_decision.json").read_text())["arms"] == decision["arms"]
    assert pl.read_csv(tmp_path / "kr_sweep_summary.csv").height == 7
    assert (tmp_path / "kr_sweep.png").stat().st_size > 0
