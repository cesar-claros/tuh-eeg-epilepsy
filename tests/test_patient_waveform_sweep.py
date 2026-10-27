"""Campaign pairing and resume guards; execute in the HPC container."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.run_patient_waveform_sweep import (  # noqa: E402
    check_audit,
    completed,
    digest,
    finish,
    make_config,
    shared_center,
)


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
    cfg = make_config(tmp_path, tmp_path, workers=0, skip_start_s=40.0)
    for seed in (0, 1, 2):
        cfg.feature.random_state = seed
        assert cfg.data.seed == 42 and cfg.data.max_windows_per_subject is None and cfg.data.skip_start_s == 40.0
        assert cfg.data.windows_train_csv == str(tmp_path / "windows_train.csv")
    assert (cfg.feature.spec.atom_len, cfg.feature.spec.pool_len, cfg.feature.spec.top_r) == (256, 512, 1)
    assert cfg.feature.train_spec.sampling_unit == "patient" and cfg.feature.train_spec.crop_sampling == "full"
    assert cfg.feature.train_spec.max_crop_energy_ratio is None
    assert (cfg.feature.train_spec.epochs, cfg.supervised.warmup_epochs, cfg.supervised.joint_epochs) == (30, 50, 30)
    assert cfg.supervised.calibration_batches == 32 and cfg.supervised.ramp_epochs == 5
    assert list(cfg.supervised.hidden_dims) == []


def _audit(folder: Path, **overrides: Any) -> Path:
    """A synthetic audit directory: three manifests and a provenance like count_pool.py writes."""
    folder.mkdir()
    for split in ("train", "val", "test"):
        (folder / f"windows_{split}.csv").write_text(
            f"subject,path,start,end,epilepsy\n{split}_p,{split}.edf,0,120,1\n"
        )
    provenance = {
        "scope": "all_metadata_eligible_complete_segments",
        "balance_per_subject": False,
        "max_windows_per_subject": None,
        "window_s": 120.0,
        "stride_s": 120.0,
        "skip_start_s": 0.0,
        "patient_split": {"mode": "fresh_stratified", "seed": 42},
        "source_manifests": {},
        "reference_subjects_without_eligible_recordings": [],
        "recordings": 1994,
        "subjects": 195,
        "output_sha256": {path.name: digest(path) for path in folder.glob("windows_*.csv")},
    }
    provenance.update(overrides)
    (folder / "audit_provenance.json").write_text(json.dumps(provenance))
    return folder


def _offset(folder: Path, reference: Path, **overrides: Any) -> Path:
    """An offset audit that preserves the split of ``reference`` (count_pool.py --split_manifest_dir)."""
    sources = {
        split: {"path": str(reference / f"windows_{split}.csv"), "sha256": digest(reference / f"windows_{split}.csv")}
        for split in ("train", "val", "test")
    }
    values = {
        "skip_start_s": 40.0,
        "patient_split": {"mode": "preserved"},
        "recordings": 1950,
        "source_manifests": sources,
    }
    return _audit(folder, **{**values, **overrides})


def test_check_audit_accepts_the_reference_and_its_offset_audit(tmp_path: Path) -> None:
    reference = _audit(tmp_path / "reference")
    assert check_audit(reference)["skip_start_s"] == 0
    assert check_audit(_offset(tmp_path / "offset", reference))["skip_start_s"] == 40.0


@pytest.mark.parametrize(
    "case",
    ["changed_reference", "lost_patient", "fresh_split", "foreign_reference", "reference_counts", "more_recordings"],
)
def test_check_audit_refuses_offset_audits_that_leave_the_reference_split(tmp_path: Path, case: str) -> None:
    reference = _audit(
        tmp_path / "reference",
        **({"patient_split": {"mode": "fresh_stratified", "seed": 7}} if case == "foreign_reference" else {}),
        **({"recordings": 1990} if case == "reference_counts" else {}),
    )
    overrides: dict[str, Any] = {
        "lost_patient": {"reference_subjects_without_eligible_recordings": ["p1"], "subjects": 194},
        "fresh_split": {"patient_split": {"mode": "fresh_stratified", "seed": 42}},
        "more_recordings": {"recordings": 1995},
    }.get(case, {})
    offset = _offset(tmp_path / "offset", reference, **overrides)
    if case == "changed_reference":
        (reference / "windows_train.csv").write_text("subject,path,start,end,epilepsy\nother,x.edf,0,120,0\n")
    with pytest.raises(ValueError):
        check_audit(offset)
    if case in ("foreign_reference", "reference_counts"):
        with pytest.raises(ValueError):
            check_audit(reference)
