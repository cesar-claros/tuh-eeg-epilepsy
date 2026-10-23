"""Write the frozen window manifests of a data configuration without loading any signal.

Builds the subject-level split and window plan exactly as a training run would
(``data.lazy_loading=true`` is required: the plan is the same on both paths and
the eager path would load every window into RAM), asserts that the splits are
non-empty, subject-disjoint and single-label, and writes to ``output_dir``:

- ``windows_{train,val,test}.csv``: the manifests that later runs reuse through
  ``data.windows_*_csv`` (the Phase 2 arms and the Phase 3 controls);
- ``manifest_provenance.json``: the data settings that define the signal and the
  plan (the same keys the pretrained-dictionary guard compares), the SHA-256 of each
  plan, and per split the number of windows, subjects and positive subjects;
- ``segment_counts_by_{subject,split_class,split,class}.csv``: retained segment,
  recording and subject counts, plus segment-hours. These describe the effective
  plans after selection and filtering, before any signal-validity checks.

The repair flags (``interpolate_bad_channels``, ``drop_bad_segments``) filter the
plan, so they must be chosen before this run and repeated by every consumer. Run on
the HPC from ``code/``::

    python src/dump_windows.py data.lazy_loading=true data.signal_mode=bipolar \\
        data.filter_freq=[1,45] data.seed=42 output_dir=logs/manifests/phase2_seed42
"""

from __future__ import annotations

import json
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, pythonpath=True)

from src.utils import (  # noqa: E402
    RankedLogger,
    check_split_disjoint,
    dump_window_metadata,
    extras,
    split_provenance,
    task_wrapper,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def dump(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the plan, check it, and write the manifests with their provenance.

    Raises
    ------
    ValueError
        If ``data.lazy_loading`` is not set, or the plan fails the split checks.
        Also raised if a retained segment's duration differs from the configured window.
    """
    if not cfg.data.get("lazy_loading"):
        raise ValueError("set data.lazy_loading=true: the manifests are built without loading signal")
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # noqa: G004
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    windows = pd.concat(
        [getattr(datamodule, f"{split}_df").assign(split=split) for split in ("train", "val", "test")],
        ignore_index=True,
    )
    windows["epilepsy"] = windows["epilepsy"].astype(int)
    window_s = float(cfg.data.window_len_min) * 60.0
    durations = windows["end"].astype(float) - windows["start"].astype(float)
    if not durations.map(lambda duration: isclose(duration, window_s, rel_tol=0.0, abs_tol=1e-6)).all():
        raise ValueError(f"Retained manifests must contain only {window_s:g}-second segments")
    dump_window_metadata(output_dir, datamodule)

    provenance = split_provenance(datamodule, cfg.data)
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        df = getattr(datamodule, f"{split}_df")
        per_subject = df.groupby("subject")["epilepsy"].first().astype(int)
        splits[split] = {
            "windows": int(len(df)),
            "subjects": int(per_subject.shape[0]),
            "positive_subjects": int(per_subject.sum()),
            "windows_sha256": provenance[f"{split}_windows_sha256"],
        }
    record = {"data": provenance["data"], "splits": splits}
    (output_dir / "manifest_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    for name, group_keys in (
        ("subject", ["split", "epilepsy", "subject"]),
        ("split_class", ["split", "epilepsy"]),
        ("split", ["split"]),
        ("class", ["epilepsy"]),
    ):
        counts = (
            windows.groupby(group_keys, sort=True)
            .agg(subjects=("subject", "nunique"), recordings=("path", "nunique"), segments=("path", "size"))
            .reset_index()
        )
        counts["segment_hours"] = counts["segments"] * window_s / 3600.0
        if name == "subject":
            counts = counts.drop(columns="subjects")
        counts.to_csv(output_dir / f"segment_counts_by_{name}.csv", index=False)
        if name != "subject":
            log.info(f"Retained segment counts by {name} (epilepsy: 0=no, 1=yes):\n{counts.to_string(index=False)}")  # noqa: G004
    log.info(
        f"Retained totals: {len(windows)} segments, {windows['path'].nunique()} recordings, "  # noqa: G004
        f"{windows['subject'].nunique()} subjects. Counts precede signal-validity checks."
    )
    log.info(f"Manifests written to {output_dir}: {splits}")  # noqa: G004
    return {f"{split}_windows": v["windows"] for split, v in splits.items()}, {"cfg": cfg, "datamodule": datamodule}


@hydra.main(version_base="1.3", config_path="../configs", config_name="dump_windows.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for the manifest dump.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.
    """
    extras(cfg)
    dump(cfg)


if __name__ == "__main__":
    main()
