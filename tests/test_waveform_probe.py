"""Decision rules of the first format-5 probe (src/waveform_probe.py); synthetic CSVs, no corpus, CPU."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.waveform_probe import (  # noqa: E402
    NEXT_STEPS,
    ArmResult,
    next_step,
    reconstruction_verdict,
    summarize,
    usage_verdict,
)


def _arm(rule: str, seed: int, explained: float, objective: float, **kwargs: object) -> ArmResult:
    values = {
        "arm": f"{rule}_r1_s{seed}",
        "rule": rule,
        "top_r": 1,
        "frozen": False,
        "seed": seed,
        "epochs": 5,
        "selected_epoch": 5,
        "val_objective": objective,
        "val_explained": explained,
        "val_gain": 1.0,
        "n_effective_atoms": 40.0,
        "val_atoms_never_won": 0.0 if rule == "f5" else math.nan,
        "never_won_last": 2.0 if rule == "f5" else math.nan,
        "max_won_share_last": 0.05 if rule == "f5" else math.nan,
        "never_won_rising": False,
    }
    return ArmResult(**{**values, **kwargs})


def test_reconstruction_needs_every_seed_and_the_relative_gain() -> None:
    better = [
        (_arm("f5", 0, 0.33, 1.0), _arm("f4", 0, 0.30, 1.1)),
        (_arm("f5", 1, 0.34, 1.0), _arm("f4", 1, 0.30, 1.1)),
    ]
    assert reconstruction_verdict(better) == "f5_better"  # mean gain 0.035 >= 0.1 * 0.30
    small = [(_arm("f5", 0, 0.31, 1.0), _arm("f4", 0, 0.30, 1.1))]
    assert reconstruction_verdict(small) == "no_difference"  # 0.01 < 0.03
    split = [better[0], (_arm("f5", 1, 0.29, 1.0), _arm("f4", 1, 0.30, 1.1))]
    assert reconstruction_verdict(split) == "no_difference"  # the seeds disagree
    objective = [(_arm("f5", 0, 0.40, 1.2), _arm("f4", 0, 0.30, 1.1))]
    assert reconstruction_verdict(objective) == "no_difference"  # the patient objective disagrees
    worse = [(_arm("f5", 0, 0.20, 1.2), _arm("f4", 0, 0.30, 1.1))]
    assert reconstruction_verdict(worse) == "f4_better"
    near_zero = [(_arm("f5", 0, 0.012, 1.0), _arm("f4", 0, 0.0, 1.1))]
    assert reconstruction_verdict(near_zero) == "f5_better"  # the scale floor avoids a zero threshold
    with pytest.raises(ValueError):
        reconstruction_verdict([])


def test_usage_thresholds() -> None:
    healthy = _arm("f5", 0, 0.3, 1.0)
    assert usage_verdict([healthy]) == "healthy"
    assert usage_verdict([replace(healthy, never_won_last=12.0)]) == "watch"
    assert usage_verdict([replace(healthy, max_won_share_last=0.2)]) == "watch"
    assert usage_verdict([replace(healthy, never_won_last=16.0)]) == "starving"
    assert usage_verdict([replace(healthy, max_won_share_last=0.25)]) == "starving"
    assert usage_verdict([healthy, replace(healthy, never_won_rising=True)]) == "starving"


def test_every_verdict_combination_has_a_next_step() -> None:
    for reconstruction in ("f5_better", "no_difference", "f4_better"):
        for usage in ("healthy", "watch", "starving"):
            assert next_step(reconstruction, usage, trained=True)
    assert next_step("f5_better", "healthy", trained=False) == NEXT_STEPS["undertrained"]
    assert next_step("f4_better", "starving", trained=True) == NEXT_STEPS["f4_better"]


def _write_arm(root: Path, name: str, explained: list[float], objective: list[float], f5: bool) -> None:
    folder = root / name
    folder.mkdir()
    history = {
        "epoch": [float(e + 1) for e in range(len(explained))],
        "val_objective": objective,
        "val_full_residual_frac": [1.0 - v for v in explained],
        "val_full_gain": [0.9] * len(explained),
    }
    if f5:
        history["n_atoms_never_won"] = [3.0, 2.0, 2.0][: len(explained)]
        history["max_won_share"] = [0.04] * len(explained)
    pl.DataFrame(history).write_csv(folder / "sae_training.csv")
    stats = {"atom": [0, 1], "energy_share": [0.5, 0.5]}
    if f5:
        stats["won_windows"] = [10, 0]
    pl.DataFrame(stats).write_csv(folder / "sae_atom_stats.csv")


def test_summarize_selects_the_minimum_objective_epoch_and_decides(tmp_path: Path) -> None:
    _write_arm(tmp_path, "f5_r1_s0", [0.20, 0.36, 0.35], [1.2, 1.0, 1.05], f5=True)
    _write_arm(tmp_path, "f4_r1_s0", [0.20, 0.29, 0.30], [1.2, 1.1, 1.08], f5=False)
    _write_arm(tmp_path, "f5_r1_frozen_s0", [0.10], [1.3], f5=True)
    _write_arm(tmp_path, "f4_r1_frozen_s0", [0.10], [1.3], f5=False)
    decision = summarize(tmp_path)
    assert decision["reconstruction"] == "f5_better" and decision["usage"] == "healthy" and decision["trained"]
    assert decision["next_step"] == NEXT_STEPS[("f5_better", "healthy")]
    assert decision["paired_explained_gain"]["0"] == pytest.approx(0.36 - 0.30)
    summary = pl.read_csv(tmp_path / "probe_summary.csv")
    f5 = summary.filter(pl.col("arm") == "f5_r1_s0").row(0, named=True)
    assert f5["selected_epoch"] == 2 and f5["val_atoms_never_won"] == 1.0 and f5["n_effective_atoms"] == 2.0
    assert json.loads((tmp_path / "probe_decision.json").read_text())["seeds"] == [0]


def test_summarize_reports_missing_controls_as_incomplete(tmp_path: Path) -> None:
    _write_arm(tmp_path, "f5_r1_s0", [0.3], [1.0], f5=True)
    _write_arm(tmp_path, "f4_r1_s0", [0.3], [1.0], f5=False)
    (tmp_path / "f5_r1_frozen_s0").mkdir()  # an arm that failed before writing its files
    assert summarize(tmp_path)["next_step"] == NEXT_STEPS["incomplete"]


def test_summarize_flags_no_training_gain(tmp_path: Path) -> None:
    _write_arm(tmp_path, "f5_r1_s0", [0.36], [1.0], f5=True)
    _write_arm(tmp_path, "f4_r1_s0", [0.10], [1.1], f5=False)
    _write_arm(tmp_path, "f5_r1_frozen_s0", [0.10], [1.3], f5=True)
    _write_arm(tmp_path, "f4_r1_frozen_s0", [0.12], [1.3], f5=False)
    decision = summarize(tmp_path)
    assert decision["trained"] is False and decision["next_step"] == NEXT_STEPS["undertrained"]
