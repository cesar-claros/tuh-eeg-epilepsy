"""Main entry point for training."""

from typing import TYPE_CHECKING, Any

import hydra
import lightning
import rootutils
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, pythonpath=True)
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    task_wrapper,
    Trainer,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule, LightningModule
    from torch import nn



log = RankedLogger(__name__, rank_zero_only=True)



@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior duringfailure. Useful for multiruns, saving info about the crash, etc.


    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A dict with metrics and dict with all instantiated objects.

    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # noqa: G004
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating feature extractor <{cfg.feature._target_}>")  # noqa: G004
    feature_extractor: nn.Module = hydra.utils.instantiate(cfg.feature)
    
    log.info(f"Instantiating sparse scaler <{cfg.scaler._target_}>")  # noqa: G004
    sparse_scaler: nn.Module = hydra.utils.instantiate(cfg.scaler)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")  # noqa: G004
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # noqa: G004
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "feature_extractor": feature_extractor,
        "model": model,
        "scaler": sparse_scaler,
        "trainer": trainer,
    }

    if cfg.get("train"):
        train_metrics = trainer.fit(
            model=model, 
            feature_extractor=feature_extractor, 
            scaler=sparse_scaler, 
            datamodule=datamodule,
            output_path=cfg.paths.output_dir
        )

    if cfg.get("test"):
        test_metrics = trainer.test(
            model=model, 
            feature_extractor=feature_extractor, 
            scaler=sparse_scaler, 
            datamodule=datamodule
        )
    
    # Return metrics (placeholder as we didn't capture them in a dict yet)
    # logic could be expanded to return dict from test
    
    # return object_dict

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict
    # return object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.

    Returns
    -------
    float | None
        The optimized metric value.

    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)
    # _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    # return optimized metric
    return get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )


if __name__ == "__main__":
    main()
