"""Two reconstruction epochs with cold/warm validation-cache auditing on the HPC.

Reuse the sweep configuration and the production training loop. Instrumentation
counts preprocessing calls across forked loader workers without changing signals,
draws, losses or selection. Timing is observational: the atoms change between epochs.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Any
from unittest.mock import patch

import hydra
import lightning
import numpy as np
import pandas as pd
import rootutils
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components.window_dataset import WindowDataset  # noqa: E402
from src.models.components import patient_waveform  # noqa: E402
from src.models.components.patient_waveform import PatientSummary  # noqa: E402
from src.models.components.waveform_sae import WaveformSAE  # noqa: E402
from src.run_patient_waveform_sweep import check_audit, digest, make_config, save_json  # noqa: E402
from src.utils import check_split_disjoint, dump_window_metadata, split_provenance  # noqa: E402


@contextmanager
def count_preprocessing(identity: str, workers: int) -> Iterator[Synchronized]:
    """Count actual validation preprocessing, including calls in Linux fork workers.

    Only the matching cache namespace is counted. The original method runs once
    per call, unchanged. Reject unsupported worker contexts before training;
    NUM_WORKERS=0 also works. No dataset attributes or cache identities change.
    """
    if workers and mp.get_start_method() != "fork":
        raise RuntimeError("cache smoke audit requires fork workers; use NUM_WORKERS=0 on this platform")
    counter = mp.Value("q", 0)
    original = WindowDataset._process

    def counted(source: WindowDataset, row: pd.Series) -> np.ndarray | None:
        cache = source.signal_cache
        if cache is not None and cache.identity == identity:
            with counter.get_lock():
                counter.value += 1
        return original(source, row)

    with patch.object(WindowDataset, "_process", counted):
        yield counter


class ValidationAudit:
    """Measure two full passes and require complete cache fill followed by zero preprocessing."""

    def __init__(self, source: WindowDataset, counter: Synchronized, output: Path) -> None:
        if source.signal_cache is None:
            raise ValueError("attach the validation cache before auditing")
        self.directory = source.signal_cache.directory
        self.expected = {f"{index:08d}.npz" for index in range(len(source))}
        self.counter, self.output = counter, output
        self.evaluate = patient_waveform.summarize_patients
        self.records: list[dict[str, Any]] = []

    def _entries(self) -> dict[str, tuple[int, int, int]]:
        entries = {}
        for path in self.directory.glob("*.npz"):
            stat = path.stat()
            entries[path.name] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return entries

    def summarize(self, model: WaveformSAE, loader: DataLoader, description: str) -> PatientSummary:
        """Wrap one existing validation call; save evidence and fail if cache reuse is incomplete."""
        epoch = len(self.records) + 1
        if epoch > 2 or description != f"Reconstruction validation {epoch}/2":
            raise RuntimeError(f"unexpected validation call: {description}")
        before, calls_before = self._entries(), self.counter.value
        started = time.perf_counter()
        summary = self.evaluate(model, loader, description)
        elapsed = time.perf_counter() - started
        after, calls = self._entries(), self.counter.value - calls_before
        visited = int(summary.counts.sum()) + summary.skipped
        verified = (
            visited == len(self.expected)
            and set(after) == self.expected
            and (not before and calls == visited if epoch == 1 else before == after and calls == 0)
        )
        self.records.append(
            {
                "epoch": epoch,
                "cache_state": "cold" if epoch == 1 else "warm",
                "segments": visited,
                "cache_entries_before": len(before),
                "cache_entries_after": len(after),
                "preprocessing_calls": calls,
                "validation_seconds": elapsed,
                "segments_per_second": visited / max(elapsed, 1e-9),
                "skipped_segments": summary.skipped,
                "verified": verified,
            }
        )
        with (self.output / "validation_cache_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)
        logger.info(
            "Cache audit epoch {}: {} -> {} entries, {} preprocessing calls, verified={}",
            epoch,
            len(before),
            len(after),
            calls,
            verified,
        )
        if not verified:
            raise RuntimeError("validation cache audit failed; inspect comparison CSV and loader warnings")
        return summary

    def finish(self) -> None:
        """Save the completed comparison; speedup is reported, never required for success."""
        if len(self.records) != 2 or not all(row["verified"] for row in self.records):
            raise RuntimeError("expected two verified full validation passes")
        cold, warm = (row["validation_seconds"] for row in self.records)
        result = {
            "status": "passed",
            "validation": self.records,
            "cold_over_warm": cold / max(warm, 1e-9),
            "seconds_saved": cold - warm,
            "caveat": "First pass includes cache writes; atoms and OS/GPU cache state change between epochs.",
        }
        save_json(self.output / "cache_smoke_summary.json", result)
        logger.info(
            "Cache reuse verified: cold {:.1f}s, warm {:.1f}s, cold/warm {:.2f}x, saved {:.1f}s",
            cold,
            warm,
            result["cold_over_warm"],
            result["seconds_saved"],
        )


def run_smoke(args: argparse.Namespace) -> None:
    """Run the fixed seed-0 P=512 reconstruction smoke using the uncapped seed-42 manifests."""
    manifests, output = args.manifests.resolve(), args.output.resolve()
    audit = check_audit(manifests)
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("run inside the HPC container on an allocated CUDA GPU")
    if args.num_workers and mp.get_start_method() != "fork":
        raise RuntimeError("cache smoke audit requires fork workers; use NUM_WORKERS=0 on this platform")
    output.mkdir(parents=True, exist_ok=True)
    # A non-resumable diagnostic must start cold. Never erase an existing cache.
    cache_root = args.signal_cache_dir.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    cfg = make_config(manifests, args.data_dir.resolve(), args.num_workers)
    cfg.seed, cfg.feature.random_state = 0, 0
    cfg.feature.train_spec.epochs, cfg.supervised = 2, None
    cfg.signal_cache_dir = str(cache_root)
    OmegaConf.save(cfg, output / "config.yaml", resolve=False)
    sources = sorted(ROOT.glob("src/**/*.py")) + [ROOT / "run_patient_wsae_cache_smoke.sh"]
    save_json(
        output / "smoke_provenance.json",
        {
            "audit": audit,
            "manifests": {path.name: digest(path) for path in manifests.glob("windows_*.csv")},
            "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources},
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "seed": 0,
            "split_seed": 42,
            "signal_cache_dir": str(cache_root),
        },
    )
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    for split, patients in (("train", 117), ("val", 39), ("test", 39)):
        with (manifests / f"windows_{split}.csv").open(newline="") as handle:
            segments = sum(1 for _ in csv.DictReader(handle))
        frame = getattr(datamodule, f"{split}_df")
        if len(frame) != segments or frame.subject.nunique() != patients:
            raise ValueError(f"{split} manifest rows or patient counts changed during loading")
    dump_window_metadata(output, datamodule)
    for split in ("train", "val"):
        getattr(datamodule, f"data_{split}").enable_signal_cache(cache_root)
    validation = datamodule.data_val
    save_json(
        output / "signal_cache.json",
        {split: getattr(datamodule, f"data_{split}").signal_cache.identity for split in ("train", "val")},
    )
    lightning.seed_everything(0, workers=True)
    model: WaveformSAE = hydra.utils.instantiate(cfg.feature)
    logger.info("Two reconstruction epochs; complete validation each epoch. Test signals are not evaluated.")
    with count_preprocessing(validation.signal_cache.identity, args.num_workers) as counter:
        observer = ValidationAudit(validation, counter, output)
        with patch.object(patient_waveform, "summarize_patients", observer.summarize):
            model.fit_unsupervised(
                datamodule.train_dataloader(),
                datamodule.val_dataloader(),
                provenance=split_provenance(datamodule, cfg.data),
            )
        model.save_artifacts(output)
        observer.finish()
    logger.info("Retained signal cache for a subsequent sweep: {}", cache_root)


def main() -> None:
    """Parse HPC locations and loader concurrency; the experiment always uses two epochs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-cache-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
