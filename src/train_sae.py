"""Train an SAE dictionary, optionally followed by pooled-patient waveform supervision.

``--config-name train_patient_waveform`` adds head warm-up and joint fine-tuning;
the other presets remain reconstruction-only.

Works for every feature extractor that exposes ``fit_unsupervised``:
``feature=shapeconv_sae`` (the default), ``feature=waveform_sae``, and
``feature=local_competition_sae``. Fits the atoms
without labels on the training windows, scores the validation windows for the loss
curve, and writes ``sae_state.pt`` (plus ``sae_atoms.npy``, ``sae_training.csv`` and
the ``windows_*.csv`` split files) to the run directory. A later
``src/train.py feature=shapeconv_sae feature.pretrained=<run>/sae_state.pt`` run
reuses a ShapeConv dictionary and skips the fit. Keep ``data.seed`` (and the split
ratios) identical between the two runs, or ``train.py`` refuses the checkpoint
because its training subjects would land in the test set.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import lightning
import rootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

rootutils.setup_root(__file__, pythonpath=True)

# The project root must be on sys.path before the `src` imports (same as train.py).
from src.models.components.shapeconv_sae import CHECKPOINT_NAME  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    calibration_provenance,
    check_split_disjoint,
    dump_window_metadata,
    extras,
    instantiate_feature,
    split_provenance,
    task_wrapper,
    threshold_calibration,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train_sae(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the unsupervised dictionary of ``cfg.feature`` on the training windows and save it.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra (``configs/train_sae.yaml``).

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The last training-history record (losses, dead atoms) and the instantiated objects.

    Raises
    ------
    TypeError
        If ``cfg.feature`` does not expose ``fit_unsupervised`` (e.g. HYDRA),
        or calibration is requested for a model without that capability.
    ValueError
        If ``feature.pretrained`` is set: this entry point trains a new dictionary.
    """
    if cfg.get("seed") is not None:
        lightning.seed_everything(cfg.seed, workers=True)

    if cfg.get("require_window_manifests", False):
        for split in ("train", "val", "test"):
            if not cfg.data.get(f"windows_{split}_csv"):
                raise ValueError(
                    "provide all complete windows_{train,val,test}_csv from the uncapped eligibility audit"
                )
        if not cfg.data.lazy_loading:
            raise ValueError("complete manifest training requires data.lazy_loading=true")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # noqa: G004
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    if cfg.get("require_window_manifests", False):
        for split in ("train", "val", "test"):
            with Path(cfg.data[f"windows_{split}_csv"]).open(newline="") as handle:
                original_count = sum(1 for _ in csv.DictReader(handle))
            if len(getattr(datamodule, f"{split}_df")) != original_count:
                raise ValueError(
                    f"{split} manifest changed during eligibility filtering; regenerate the complete audit"
                )
    output_dir = Path(cfg.paths.output_dir)
    dump_window_metadata(output_dir, datamodule)

    log.info(f"Instantiating feature extractor <{cfg.feature._target_}>")  # noqa: G004
    feature_extractor = instantiate_feature(cfg.feature)
    if not callable(getattr(feature_extractor, "fit_unsupervised", None)):
        raise TypeError(
            f"{cfg.feature._target_} has no fit_unsupervised; use feature=shapeconv_sae, "
            "feature=waveform_sae, or feature=local_competition_sae"
        )
    if getattr(feature_extractor, "fitted", False):
        raise ValueError("train_sae.py trains a new dictionary; unset feature.pretrained")

    supervised = cfg.get("supervised")
    if supervised is not None:
        from src.models.components.patient_waveform import fit_patient_supervised, validate_supervised
        from src.models.components.waveform_sae import WaveformSAE

        if not isinstance(feature_extractor, WaveformSAE):
            raise TypeError("pooled patient supervision is implemented for waveform_sae only")
        supervised_spec = hydra.utils.instantiate(supervised, _convert_="all")
        validate_supervised(feature_extractor, supervised_spec)

    calibration = threshold_calibration(cfg)
    calibrate = getattr(feature_extractor, "calibrate_amp_min", None)
    if calibration is not None and not callable(calibrate):
        raise TypeError(f"{cfg.feature._target_} does not support threshold calibration; unset feature.calibrate_fa")

    feature_extractor.fit_unsupervised(
        datamodule.train_dataloader(), datamodule.val_dataloader(), provenance=split_provenance(datamodule, cfg.data)
    )
    if calibration is not None:
        assert callable(calibrate)
        calibrate(
            datamodule.train_dataloader(),
            calibration["false_alarms_per_channel_minute"],
            calibration["sfreq"],
            keep_label=calibration["keep_label"],
            provenance=calibration_provenance(datamodule, calibration["keep_label"]),
        )
    if supervised is not None:
        pretraining_dir = output_dir / "pretraining"
        pretraining_dir.mkdir(parents=True, exist_ok=True)
        feature_extractor.save_artifacts(pretraining_dir)
        head, supervised_metrics = fit_patient_supervised(
            feature_extractor,
            (datamodule.train_dataloader(), datamodule.val_dataloader()),
            supervised_spec,
            output_dir,
        )
    else:
        feature_extractor.save_artifacts(output_dir)
    log.info(  # noqa: G004
        f"Reuse with: feature={HydraConfig.get().runtime.choices['feature']} "
        f"feature.pretrained={output_dir / CHECKPOINT_NAME} data.seed={cfg.data.seed}"
    )

    history = feature_extractor.history
    metric_dict: dict[str, Any] = dict(history[-1]) if history else {}
    object_dict = {"cfg": cfg, "datamodule": datamodule, "feature_extractor": feature_extractor}
    if supervised is not None:
        metric_dict.update(supervised_metrics)
        object_dict["patient_head"] = head
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_sae.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for standalone dictionary training.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.
    """
    extras(cfg)
    train_sae(cfg)


if __name__ == "__main__":
    main()
