"""Main entry point for training."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import lightning
import polars as pl
import rootutils
from omegaconf import DictConfig
from tqdm import tqdm

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


def _resolve_feature_seeds(spec: int | list[int] | None) -> list[int] | None:
    """Resolve the ``feature_seeds`` config value into an explicit seed list.

    Parameters
    ----------
    spec : int | list[int] | None
        ``None`` requests a single run, an ``int`` ``N`` expands to seeds
        ``0..N-1``, and a list of ints is used verbatim.

    Returns
    -------
    list[int] | None
        The HYDRA kernel seeds to evaluate, or ``None`` for a single run.

    Raises
    ------
    ValueError
        If ``spec`` is an int that is not strictly positive.
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("`feature_seeds` given as an int must be positive")
        return list(range(spec))
    return [int(s) for s in spec]


def _run_seed_sweep(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    trainer: Trainer,
    seeds: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Quantify accuracy variation across multiple HYDRA kernel seeds.

    The data is loaded and windowed once. For every seed a fresh feature
    extractor, scaler, and classifier are built, fitted, and scored, so the only
    quantity that changes between runs is the random kernel seed. Per-seed
    results and a summary are written to the run output directory.

    Parameters
    ----------
    cfg : DictConfig
        The composed Hydra configuration.
    datamodule : LightningDataModule
        The already-instantiated data module.
    trainer : Trainer
        The custom sklearn-style trainer.
    seeds : list[int]
        The HYDRA ``random_state`` values to evaluate.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A dict of aggregated metrics (per-metric mean and std) and a dict of the
        instantiated objects plus the per-seed results table.
    """
    log.info(f"Evaluating HYDRA feature variability over {len(seeds)} seeds: {seeds}")  # noqa: G004

    rows: list[dict[str, float]] = []
    for seed in tqdm(seeds, desc="HYDRA feature seed sweep"):
        feature_extractor = hydra.utils.instantiate(cfg.feature, random_state=seed)
        scaler = hydra.utils.instantiate(cfg.scaler)
        model = hydra.utils.instantiate(cfg.model)

        train_scores = trainer.fit(
            model=model,
            feature_extractor=feature_extractor,
            scaler=scaler,
            datamodule=datamodule,
            output_path=cfg.paths.output_dir,
            save=False,
        )
        test_scores = trainer.test(
            model=model,
            feature_extractor=feature_extractor,
            scaler=scaler,
            datamodule=datamodule,
        )
        rows.append(
            {
                "seed": seed,
                "train_accuracy_by_window": float(train_scores["accuracy_by_window"]),
                "train_accuracy_by_subject": float(train_scores["accuracy_by_subject"]),
                "test_accuracy_by_window": float(test_scores["accuracy_by_window"]),
                "test_accuracy_by_subject": float(test_scores["accuracy_by_subject"]),
            }
        )

    results = pl.DataFrame(rows)
    metric_columns = [c for c in results.columns if c != "seed"]

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(output_dir / "seed_sweep_results.csv")
    results.drop("seed").describe().write_csv(output_dir / "seed_sweep_summary.csv")

    log.info("HYDRA feature accuracy variation (mean +/- std [min, max]):")
    metric_dict: dict[str, Any] = {}
    for column in metric_columns:
        series = results[column]
        mean = float(series.mean())
        std = float(series.std()) if len(seeds) > 1 else 0.0
        metric_dict[f"{column}_mean"] = mean
        metric_dict[f"{column}_std"] = std
        log.info(  # noqa: G004
            f"  {column:32s}: {mean:.4f} +/- {std:.4f} "
            f"[{float(series.min()):.4f}, {float(series.max()):.4f}]"
        )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "trainer": trainer,
        "results": results,
    }
    return metric_dict, object_dict



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

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # noqa: G004
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # Optional multi-seed evaluation of the random HYDRA kernels: quantifies how
    # much accuracy varies with feature.random_state while the data split is held
    # fixed. Triggered by the `feature_seeds` config (see configs/train.yaml).
    feature_seeds = _resolve_feature_seeds(cfg.get("feature_seeds"))
    if feature_seeds is not None:
        return _run_seed_sweep(cfg, datamodule, trainer, feature_seeds)

    log.info(f"Instantiating feature extractor <{cfg.feature._target_}>")  # noqa: G004
    feature_extractor: nn.Module = hydra.utils.instantiate(cfg.feature)

    log.info(f"Instantiating sparse scaler <{cfg.scaler._target_}>")  # noqa: G004
    sparse_scaler: nn.Module = hydra.utils.instantiate(cfg.scaler)

    log.info(f"Instantiating model <{cfg.model._target_}>")  # noqa: G004
    model: LightningModule = hydra.utils.instantiate(cfg.model)

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
            output_path=cfg.paths.output_dir,
        )

    if cfg.get("test"):
        test_metrics = trainer.test(
            model=model,
            feature_extractor=feature_extractor,
            scaler=sparse_scaler,
            datamodule=datamodule,
        )

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


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
