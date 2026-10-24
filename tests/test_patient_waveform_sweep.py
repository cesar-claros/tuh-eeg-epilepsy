"""Campaign pairing and resume guards; execute in the HPC container."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.run_patient_waveform_sweep import completed, finish, make_config, shared_center  # noqa: E402


def _probes(folder: Path, values: list[float]) -> Path:
    """Save one seed's synthetic calibration output."""
    folder.mkdir()
    with (folder / "patient_gradient_calibration.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ratio"])
        writer.writerows([value] for value in values)
    return folder


def test_center_pools_ratios_not_seed_medians(tmp_path: Path) -> None:
    """Unequal within-seed distributions distinguish pooling from a median of medians."""
    paths = [_probes(tmp_path / str(seed), [float(seed)] * 17 + [100.0] * 15) for seed in (1, 2, 3)]
    assert shared_center(paths) == 3.0  # The seed medians would give 2.0.


@pytest.mark.parametrize("invalid", [[1.0] * 31, [1.0] * 31 + [0.0], [1.0] * 31 + [float("nan")]])
def test_center_refuses_incomplete_or_degenerate_calibration(tmp_path: Path, invalid: list[float]) -> None:
    """Do not silently change a seed's weight or floor an undefined gradient ratio."""
    paths = [_probes(tmp_path / str(seed), [1.0] * 32) for seed in (0, 1)]
    paths.append(_probes(tmp_path / "2", invalid))
    with pytest.raises(ValueError, match="32 finite positive"):
        shared_center(paths)
    with pytest.raises(ValueError, match="all three seeds"):
        shared_center(paths[:2])


def test_resume_requires_untampered_completed_artifacts(tmp_path: Path) -> None:
    """Partial attempts are retried; altered completed outputs are refused."""
    attempt = tmp_path / "attempt_first"
    attempt.mkdir()
    artifact = attempt / "reference.txt"
    artifact.write_text("saved reference")
    assert completed(tmp_path) is None
    finish(tmp_path, attempt, ("reference.txt",), {"r0": 1.0})
    record = completed(tmp_path)
    assert record is not None and record["metrics"]["r0"] == 1.0
    artifact.write_text("different reference")
    with pytest.raises(ValueError, match="missing or changed"):
        completed(tmp_path)


def test_sweep_config_preserves_patient_split_and_complete_bags(tmp_path: Path) -> None:
    """Changing a model seed cannot redraw the fixed split or impose a manifest cap."""
    cfg = make_config(tmp_path, tmp_path, workers=0)
    for seed in (0, 1, 2):
        cfg.feature.random_state = seed
        assert cfg.data.seed == 42 and cfg.data.max_windows_per_subject is None
        assert cfg.data.windows_train_csv == str(tmp_path / "windows_train.csv")
    assert (cfg.feature.spec.atom_len, cfg.feature.spec.pool_len, cfg.feature.spec.top_r) == (256, 512, 1)
    assert cfg.feature.train_spec.sampling_unit == "patient" and cfg.feature.train_spec.crop_sampling == "full"
    assert cfg.feature.train_spec.max_crop_energy_ratio is None
    assert (cfg.feature.train_spec.epochs, cfg.supervised.warmup_epochs, cfg.supervised.joint_epochs) == (30, 50, 30)
    assert cfg.supervised.calibration_batches == 32 and cfg.supervised.ramp_epochs == 5
    assert list(cfg.supervised.hidden_dims) == []
