"""HPC orchestration of the P=512 linear patient sweep; reuse existing training functions.

Pretrain once per seed, pool 32 gradient ratios from each of three references,
then run five joint arms per seed with that fixed center. Every arm restores the
same saved normalization, warmed-up head and calibration without setup scans.
Strict deterministic execution and reference checks protect the paired starts.
The training objective and sampling are unchanged. Test signals are never evaluated.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import hydra
import lightning
import polars as pl
import rootutils
import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.patient_waveform import (  # noqa: E402
    PatientPreparation,
    SupervisedSpec,
    fit_patient_supervised,
    prepare_patient_supervised,
)
from src.models.components.waveform_sae import WaveformSAE  # noqa: E402
from src.utils import check_split_disjoint, dump_window_metadata, split_provenance  # noqa: E402

SEEDS = (0, 1, 2)
MULTIPLIERS = (0.03, 0.1, 0.3, 1.0, 3.0)


def digest(path: Path) -> str:
    """Hash a saved file without loading model tensors or signals."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, value: object) -> None:
    """Atomically publish metadata only after its contents are complete."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def completed(folder: Path) -> dict[str, Any] | None:
    """Read a completion marker; refuse missing or altered saved stage artifacts."""
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    for name, checksum in record["files"].items():
        path = folder / record["attempt"] / name
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"completed artifact is missing or changed: {path}")
    return record


def finish(folder: Path, attempt: Path, files: tuple[str, ...], metrics: dict[str, Any]) -> None:
    """Mark a successful stage, retaining all earlier incomplete attempts."""
    save_json(
        folder / "complete.json",
        {"attempt": attempt.name, "files": {name: digest(attempt / name) for name in files}, "metrics": metrics},
    )


def make_config(manifests: Path, data_dir: Path, workers: int) -> DictConfig:
    """Compose the fixed protocol while keeping patient split seed 42 independent of model seeds."""
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train_patient_waveform")
    cfg.data.data_dir = str(data_dir)
    for split in ("train", "val", "test"):
        cfg.data[f"windows_{split}_csv"] = str(manifests / f"windows_{split}.csv")
    cfg.data.seed, cfg.data.num_workers = 42, workers
    cfg.data.lazy_loading, cfg.data.window_len_min, cfg.data.target_sfreq = True, 2, 256
    cfg.data.include_seizures = False
    cfg.data.exclude_native_sfreqs = [400.0, 512.0, 1000.0]
    cfg.data.signal_mode, cfg.data.bipolar = "bipolar", False
    cfg.data.filter_freq, cfg.data.notch_freqs = [0.1, None], [60.0]
    cfg.data.interpolate_bad_channels, cfg.data.apply_aas = True, False
    cfg.data.drop_bad_segments, cfg.data.drop_seizure_segments = False, False
    cfg.data.require_keep_labels, cfg.data.exclude_recordings_file = None, None
    cfg.data.skip_start_s, cfg.data.overlap_pct = 0.0, 0.0
    cfg.data.max_windows_per_subject = None
    cfg.feature.device, cfg.feature.pretrained = "cuda", None
    cfg.feature.spec.n_atoms = 64
    cfg.feature.spec.atom_len, cfg.feature.spec.pool_len, cfg.feature.spec.top_r = 256, 512, 1
    cfg.feature.spec.pre_emphasis, cfg.feature.spec.atom_norm, cfg.feature.spec.center_atoms = "none", "unit", False
    cfg.feature.train_spec.epochs, cfg.feature.train_spec.lr = 30, 1e-3
    cfg.feature.train_spec.sampling_unit, cfg.feature.train_spec.crop_sampling = "patient", "full"
    cfg.feature.train_spec.max_crop_energy_ratio, cfg.feature.train_spec.init = None, "patches"
    cfg.feature.train_spec.full_window_val = True
    cfg.feature.train_spec.patients_per_batch, cfg.feature.train_spec.segments_per_patient = 8, 4
    cfg.feature.train_spec.segment_batch = 2
    cfg.feature.train_spec.recenter_atoms = False
    cfg.supervised.warmup_epochs, cfg.supervised.joint_epochs = 50, 30
    cfg.supervised.warmup_lr, cfg.supervised.dictionary_lr, cfg.supervised.head_lr = 1e-2, 1e-4, 1e-3
    cfg.supervised.calibration_batches, cfg.supervised.ramp_epochs = 32, 5
    cfg.supervised.hidden_dims, cfg.supervised.lambda_center = [], None
    cfg.supervised.gamma, cfg.supervised.epsilon = 1e-3, 1e-6
    return cfg


def shared_center(prepared: list[Path]) -> float:
    """Pool exactly 32 positive finite calibration ratios per seed, without seed reweighting."""
    if len(prepared) != len(SEEDS):
        raise ValueError("shared calibration requires all three seeds")
    ratios: list[float] = []
    for path in prepared:
        with (path / "patient_gradient_calibration.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = [float(row["ratio"]) for row in rows]
        if len(values) != 32 or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"expected 32 finite positive probe ratios: {path}")
        ratios.extend(values)
    return statistics.median(ratios)


def check_audit(manifests: Path) -> dict[str, Any]:
    """Require the fresh uncapped eligible cohort before constructing any signal loaders."""
    audit = json.loads((manifests / "audit_provenance.json").read_text())
    checks = (
        audit["scope"] == "all_metadata_eligible_complete_segments",
        audit["balance_per_subject"] is False and audit["max_windows_per_subject"] is None,
        audit["patient_split"]["mode"] == "fresh_stratified" and audit["patient_split"]["seed"] == 42,
        audit["window_s"] == audit["stride_s"] == 120 and audit["skip_start_s"] == 0,
        audit["recordings"] == 1994 and audit["subjects"] == 195,
    )
    if not all(checks):
        raise ValueError(
            "need fresh seed-42 stratified, uncapped 120-second manifests for 1994 recordings/195 patients"
        )
    return audit


def run_campaign(args: argparse.Namespace) -> None:
    """Execute or resume complete stages; reject changed inputs instead of mixing experiments."""
    manifests, data_dir, output = args.manifests.resolve(), args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = check_audit(manifests)
    if not torch.cuda.is_available():
        raise RuntimeError("run inside the HPC container on an allocated CUDA GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = make_config(manifests, data_dir, args.num_workers)
    cache_root = args.signal_cache_dir.expanduser().resolve() if args.signal_cache_dir else None
    cfg.signal_cache_dir = str(cache_root) if cache_root is not None else None
    sources = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/**/*.yaml"), ROOT / "run_patient_wsae_sweep.sh"])
    identity = {
        "protocol": 2,
        "seeds": SEEDS,
        "multipliers": MULTIPLIERS,
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "feature": OmegaConf.to_container(cfg.feature, resolve=True),
        "supervised": OmegaConf.to_container(cfg.supervised, resolve=True),
        "signal_cache_dir": str(cache_root) if cache_root is not None else None,
        "manifests": {
            name: digest(manifests / name)
            for name in ("windows_train.csv", "windows_val.csv", "windows_test.csv", "audit_provenance.json")
        },
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "container": os.environ.get("APPTAINER_CONTAINER", os.environ.get("SINGULARITY_CONTAINER")),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "polars",
                    "pandas",
                    "mne",
                    "braindecode",
                    "lightning",
                    "hydra-core",
                    "scikit-learn",
                )
            },
        },
    }
    # Canonical JSON converts tuples before comparing a resumed invocation.
    identity = json.loads(json.dumps(identity))
    identity_path = output / "campaign.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("campaign inputs, code, runtime or settings changed; use a new RUN_DIR")
    save_json(identity_path, identity)
    OmegaConf.save(cfg, output / "config.yaml", resolve=False)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    expected_patients = {"train": 117, "val": 39, "test": 39}
    for split, expected in expected_patients.items():
        with (manifests / f"windows_{split}.csv").open(newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        frame = getattr(datamodule, f"{split}_df")
        if len(frame) != count or frame.subject.nunique() != expected:
            raise ValueError(f"{split} manifest rows or patient counts changed during loading; regenerate audit")
    dump_window_metadata(output, datamodule)
    if cache_root is not None:
        cache_identity = {}
        for split in ("train", "val"):
            dataset = getattr(datamodule, f"data_{split}")
            dataset.enable_signal_cache(cache_root)
            cache_identity[split] = dataset.signal_cache.identity
        cache_record = output / "signal_cache.json"
        if cache_record.exists() and json.loads(cache_record.read_text()) != cache_identity:
            raise ValueError("preprocessing or signal cache sources changed; use a new RUN_DIR")
        save_json(cache_record, cache_identity)
    provenance = split_provenance(datamodule, cfg.data)
    loaders = datamodule.train_dataloader(), datamodule.val_dataloader()
    spec: SupervisedSpec = hydra.utils.instantiate(cfg.supervised, _convert_="all")
    logger.info(
        "Verified {} patients / {} segments; test signals will not be evaluated", audit["subjects"], audit["segments"]
    )
    prepared = []
    for seed in SEEDS:
        folder = output / f"seed_{seed}" / "reference"
        folder.mkdir(parents=True, exist_ok=True)
        record = completed(folder)
        if record is None:
            if args.stage == "sweep":
                raise ValueError(f"prepare all three references first; missing {folder}")
            attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=folder))
            logger.info("Preparing seed {} in {}", seed, attempt)
            lightning.seed_everything(seed, workers=True)
            cfg.feature.random_state = seed
            model: WaveformSAE = hydra.utils.instantiate(cfg.feature)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model.fit_unsupervised(*loaders, provenance=provenance)
            model.save_artifacts(attempt)
            preparation = prepare_patient_supervised(model, loaders, spec, attempt)
            metrics = {
                "r0": preparation.r0,
                "val_reconstruction": preparation.val_reconstruction,
                "val_zero": preparation.val_zero,
                "seed_lambda_center": preparation.lambda_center,
                "frozen_val_patient_auroc": preparation.frozen_val_patient_auroc,
            }
            metrics.update(
                seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated()
            )
            finish(
                folder,
                attempt,
                (
                    "sae_state.pt",
                    "patient_gradient_calibration.csv",
                    "patient_warmup.csv",
                    "patient_preparation.pt",
                    "frozen_patient_scores_val.csv",
                ),
                metrics,
            )
            del model, preparation
            gc.collect()
            torch.cuda.empty_cache()
            record = completed(folder)
        assert record is not None
        prepared.append(folder / record["attempt"])
    center = shared_center(prepared)
    save_json(
        output / "lambda_calibration.json",
        {
            "center": center,
            "ratios": 96,
            "seeds": SEEDS,
            "weights": [{"alpha": alpha, "lambda": center * alpha} for alpha in MULTIPLIERS],
        },
    )
    logger.info("Shared P=512 lambda center: {:.10g} (96 ratios)", center)
    if args.stage == "prepare":
        return
    summaries = []
    for seed, reference in zip(SEEDS, prepared):
        preparation = PatientPreparation.load(reference / "patient_preparation.pt")
        for alpha in MULTIPLIERS:
            folder = output / f"seed_{seed}" / f"alpha_{alpha:g}"
            folder.mkdir(parents=True, exist_ok=True)
            record = completed(folder)
            if record is None:
                attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=folder))
                logger.info("Joint seed {}, alpha {}, lambda {:.10g}: {}", seed, alpha, center * alpha, attempt)
                lightning.seed_everything(seed, workers=True)
                cfg.feature.random_state = seed
                model = hydra.utils.instantiate(cfg.feature, pretrained=str(reference / "sae_state.pt"))
                if model.provenance != provenance:
                    raise ValueError("saved reference provenance differs from the current complete manifests")
                arm_spec = replace(spec, lambda_center=center, lambda_multiplier=alpha)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                head, arm_metrics = fit_patient_supervised(model, loaders, arm_spec, attempt, prepared=preparation)
                # The saved setup is copied into each arm, never recalibrated or trained in place.
                for name in ("mean", "scale"):
                    torch.testing.assert_close(getattr(head, name).cpu(), preparation.head[name], rtol=0, atol=0)
                if (
                    arm_metrics["r0"] != preparation.r0
                    or arm_metrics["frozen_val_patient_auroc"] != preparation.frozen_val_patient_auroc
                ):
                    raise ValueError("arm's frozen starting reference differs from the saved preparation")
                if (reference / "patient_gradient_calibration.csv").read_bytes() != (
                    attempt / "patient_gradient_calibration.csv"
                ).read_bytes():
                    raise ValueError("copied gradient calibration report differs from the reference")
                checkpoint = torch.load(attempt / "patient_model.pt", map_location="cpu", weights_only=True)
                if checkpoint["probe_draws"] != preparation.probe_draws:
                    raise ValueError("probe draws differ across paired arms")
                previous = completed(output / f"seed_{seed}" / f"alpha_{MULTIPLIERS[0]:g}")
                if previous is not None:
                    first_path = output / f"seed_{seed}" / f"alpha_{MULTIPLIERS[0]:g}" / previous["attempt"]
                    first = torch.load(first_path / "patient_model.pt", map_location="cpu", weights_only=True)
                    if checkpoint["joint_draws"] != first["joint_draws"]:
                        raise ValueError("joint patient/segment draws differ across paired arms")
                curve = pl.read_csv(attempt / "patient_joint_training.csv")
                arm_metrics.update(
                    seconds=time.perf_counter() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    eligible_joint_epochs=int(curve["eligible"].sum()),
                )
                finish(
                    folder,
                    attempt,
                    (
                        "patient_model.pt",
                        "sae_state.pt",
                        "patient_joint_training.csv",
                        "patient_gradient_calibration.csv",
                        "patient_warmup.csv",
                        "patient_scores_val.csv",
                        "patient_scores_train.csv",
                    ),
                    arm_metrics,
                )
                del model, head, checkpoint
                gc.collect()
                torch.cuda.empty_cache()
                record = completed(folder)
            assert record is not None
            summaries.append(
                {"seed": seed, "alpha": alpha, **record["metrics"], "output": str(folder / record["attempt"])}
            )
            pl.DataFrame(summaries).write_csv(output / "sweep_summary.csv")
    logger.info("Completed 15 joint arms. U controls, fresh-head refits and held-out testing remain separate.")


def main() -> None:
    """Parse paths and stage selection for the HPC shell entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "prepare", "sweep"), default="all")
    parser.add_argument("--signal-cache-dir", type=Path, help="persistent sensor cache shared by all seeds/arms")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is already using this RUN_DIR") from exc
        run_campaign(args)


if __name__ == "__main__":
    main()
