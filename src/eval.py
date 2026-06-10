"""Main entry point for evaluation."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import joblib
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, pythonpath=True)

from src.utils import (
    RankedLogger,
    Trainer,
    extras,
    task_wrapper,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a trained HYDRA + sklearn pipeline on the test split.

    Loads the artifacts saved by `Trainer.fit` (`model.joblib`, the fitted
    scaler+classifier pipeline, and `feature_extractor.joblib`) and scores the
    datamodule's test set, reporting window-level and subject-level accuracy.

    This method is wrapped in the optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A dict with metrics and dict with all instantiated objects.

    """
    if not cfg.ckpt_path:
        msg = "No checkpoint path provided!"
        raise ValueError(msg)

    # Accept either the run directory or a direct path to model.joblib.
    ckpt_dir = Path(cfg.ckpt_path)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # noqa: G004
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # noqa: G004
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info(f"Loading trained pipeline and feature extractor from <{ckpt_dir}>")  # noqa: G004
    pipeline = joblib.load(ckpt_dir / "model.joblib")
    feature_extractor = joblib.load(ckpt_dir / "feature_extractor.joblib")
    # The saved pipeline is `make_pipeline(scaler, classifier)`; recover its steps.
    scaler, model = pipeline[0], pipeline[1]

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "feature_extractor": feature_extractor,
        "model": model,
        "scaler": scaler,
        "trainer": trainer,
    }

    log.info("Starting testing!")
    metric_dict = trainer.test(
        model=model,
        feature_extractor=feature_extractor,
        scaler=scaler,
        datamodule=datamodule,
    )

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for evaluation.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.

    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
