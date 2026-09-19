"""Entry point that trains a ShapeConv SAE dictionary alone (no features, no classifier).

Fits the atoms without labels on the training windows, scores the validation
windows for the loss curve, and writes ``sae_state.pt`` (plus ``sae_atoms.npy``,
``sae_training.csv`` and the ``windows_*.csv`` split files) to the run directory.
A later ``src/train.py feature=shapeconv_sae feature.pretrained=<run>/sae_state.pt``
run reuses the dictionary and skips the fit. Keep ``data.seed`` (and the split
ratios) identical between the two runs, or ``train.py`` refuses the checkpoint
because its training subjects would land in the test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import lightning
import rootutils
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
        If ``cfg.feature`` does not expose ``fit_unsupervised`` (e.g. HYDRA).
    ValueError
        If ``feature.pretrained`` is set: this entry point trains a new dictionary.
    """
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # noqa: G004
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    output_dir = Path(cfg.paths.output_dir)
    dump_window_metadata(output_dir, datamodule)

    log.info(f"Instantiating feature extractor <{cfg.feature._target_}>")  # noqa: G004
    feature_extractor = instantiate_feature(cfg.feature)
    if not hasattr(feature_extractor, "fit_unsupervised"):
        raise TypeError(f"{cfg.feature._target_} has no fit_unsupervised; use feature=shapeconv_sae")
    if getattr(feature_extractor, "fitted", False):
        raise ValueError("train_sae.py trains a new dictionary; unset feature.pretrained")

    feature_extractor.fit_unsupervised(
        datamodule.train_dataloader(), datamodule.val_dataloader(), provenance=split_provenance(datamodule, cfg.data)
    )
    calibration = threshold_calibration(cfg)
    if calibration is not None:
        feature_extractor.calibrate_amp_min(
            datamodule.train_dataloader(), calibration["false_alarms_per_channel_minute"], calibration["sfreq"],
            keep_label=calibration["keep_label"],
            provenance=calibration_provenance(datamodule, calibration["keep_label"]),
        )
    feature_extractor.save_artifacts(output_dir)
    log.info(  # noqa: G004
        "Reuse with: python src/train.py feature=shapeconv_sae "
        f"feature.pretrained={output_dir / CHECKPOINT_NAME} data.seed={cfg.data.seed}"
    )

    history = feature_extractor.history
    metric_dict: dict[str, Any] = dict(history[-1]) if history else {}
    object_dict = {"cfg": cfg, "datamodule": datamodule, "feature_extractor": feature_extractor}
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
