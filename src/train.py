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

# The datamodule encodes the epilepsy target as 1 and no-epilepsy as 0.
EPILEPSY_CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}


def _dump_window_metadata(cfg: DictConfig, datamodule: Any) -> None:
    """Save the per-split window metadata to CSV in the run output directory.

    Each ``windows_<split>.csv`` lists the windows (subject, path, start, end,
    ...) used in that split. Because the windowing is deterministic in
    ``data.seed`` and independent of ``signal_mode`` / ``ica_keep_labels``, these
    files let you confirm that two runs evaluated on exactly the same windows
    (diff the files) or see precisely which windows differ.
    """
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = getattr(datamodule, f"{split}_df", None)
        if df is not None and len(df):
            df.to_csv(output_dir / f"windows_{split}.csv", index=False)
            log.info(f"Saved {len(df)} {split} window rows to windows_{split}.csv")  # noqa: G004


def _write_performance_csv(cfg: DictConfig, scores_by_split: dict) -> None:
    """Write window- and subject-level performance metrics to ``performance.csv``.

    One row per (split, level) with all metrics as columns (accuracy,
    balanced_accuracy, sensitivity, specificity, precision, f1, mcc, roc_auc,
    average_precision, and the n / n_pos / n_neg counts).
    """
    rows: list[dict] = []
    for split_name, scores in scores_by_split.items():
        for level in ("window", "subject"):
            if isinstance(scores, dict) and level in scores:
                rows.append({"split": split_name, "level": level, **scores[level]})
    if not rows:
        return
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(output_dir / "performance.csv")
    log.info("Saved performance metrics to performance.csv")


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
        row: dict[str, float] = {"seed": seed}
        for split_name, scores in (("train", train_scores), ("test", test_scores)):
            for level in ("window", "subject"):
                for metric, value in scores[level].items():
                    row[f"{split_name}_{level}_{metric}"] = value
        rows.append(row)

    results = pl.DataFrame(rows)
    metric_columns = [c for c in results.columns if c != "seed"]

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(output_dir / "seed_sweep_results.csv")
    results.drop("seed").describe().write_csv(output_dir / "seed_sweep_summary.csv")

    metric_dict: dict[str, Any] = {}
    for column in metric_columns:
        series = results[column]
        metric_dict[f"{column}_mean"] = float(series.mean())
        metric_dict[f"{column}_std"] = float(series.std()) if len(seeds) > 1 else 0.0

    # Log only the key test metrics; the CSVs hold every metric for all splits.
    log.info(f"HYDRA feature performance variation over {len(seeds)} seeds (mean +/- std [min, max]):")  # noqa: G004
    key_metrics = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc")
    for level in ("window", "subject"):
        for metric in key_metrics:
            column = f"test_{level}_{metric}"
            if column in results.columns:
                series = results[column]
                std = float(series.std()) if len(seeds) > 1 else 0.0
                log.info(  # noqa: G004
                    f"  test_{level}_{metric:18s}: {float(series.mean()):.4f} +/- {std:.4f} "
                    f"[{float(series.min()):.4f}, {float(series.max()):.4f}]"
                )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "trainer": trainer,
        "results": results,
    }
    return metric_dict, object_dict


def _kernel_rows(infos: list, fractions_for: tuple | None = None) -> list[dict]:
    """Flatten KernelInfo / DiscriminativeKernel objects into CSV-ready rows."""
    rows: list[dict] = []
    for info in infos:
        row: dict[str, Any] = {"rank": info.rank}
        if fractions_for is None:
            row["count"] = info.count
        else:
            class_a, class_b = fractions_for
            row["score"] = info.score
            row["favors"] = info.favors
            row[f"frac_{class_a}"] = info.fractions[class_a]
            row[f"frac_{class_b}"] = info.fractions[class_b]
        row["dilation"] = info.dilation
        row["peak_freq_hz"] = info.peak_freq_hz
        row["representation"] = info.representation
        row["group"] = info.group
        row["kernel"] = info.kernel
        for i, w in enumerate(info.weight.tolist()):
            row[f"w{i}"] = w
        rows.append(row)
    return rows


def _kernel_sfreq(cfg: DictConfig, feature_extractor: Any) -> float | None:
    """Resampled sampling rate (Hz) of the windows, or None if not derivable."""
    n_timepoints = getattr(feature_extractor, "n_timepoints", None)
    window_len_min = cfg.data.get("window_len_min") if "data" in cfg else None
    if n_timepoints and window_len_min:
        return n_timepoints / (window_len_min * 60.0)
    return None


def _report_top_kernels(cfg: DictConfig, feature_extractor: Any) -> None:
    """Log, save, and plot the most-used and most class-discriminative kernels.

    Only meaningful when ``feature.track_counts`` is enabled (so win counts were
    accumulated during feature extraction). Writes ``top_kernels.csv`` /
    ``top_discriminative_kernels.csv`` (with a ``peak_freq_hz`` column) and, when
    a sampling rate is derivable, waveform-plus-spectrum plots and a peak-frequency
    histogram split by favored class, to the run output directory. The variants
    (n, weighting, competition, metric) come from the ``top_kernels`` config block.
    """
    from src.models.components import kernel_viz

    spec = cfg.get("top_kernels") or {}
    n = spec.get("n", 20)
    n_per_class = spec.get("n_per_class", 5)
    n_hist = spec.get("n_hist_per_class", 100)
    by = spec.get("by", "max")
    weighting = spec.get("weighting", "frequency")
    metric = spec.get("metric", "difference")

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sfreq = _kernel_sfreq(cfg, feature_extractor)
    if sfreq is not None:
        log.info(f"Kernel frequencies reported at sfreq={sfreq:.1f} Hz")  # noqa: G004

    log.info(f"Top {n} kernels by global '{by}/{weighting}' count:")  # noqa: G004
    top = feature_extractor.top_kernels(n, by=by, weighting=weighting, sfreq=sfreq)
    pl.DataFrame(_kernel_rows(top)).write_csv(output_dir / "top_kernels.csv")
    for info in top[:5]:
        freq = f" peak={info.peak_freq_hz:.1f}Hz" if info.peak_freq_hz is not None else ""
        log.info(  # noqa: G004
            f"  #{info.rank} d{info.dilation} {info.representation} "
            f"g{info.group}k{info.kernel} count={info.count:.2f}{freq}"
        )
    if sfreq is not None:
        kernel_viz.plot_kernels(
            top, sfreq, output_dir / "top_kernels.png", "Top HYDRA kernels (global)"
        )

    labels = feature_extractor.class_labels()
    if len(labels) == 2:
        class_a, class_b = labels[0], labels[1]
        class_names = {c: EPILEPSY_CLASS_NAMES.get(c, str(c)) for c in labels}
        # Balanced view: the most-favoring kernels for EACH class (the global
        # |score| ranking is dominated by whichever class has the peakier wins).
        per_class = feature_extractor.top_discriminative_kernels_per_class(
            n_hist, by=by, weighting=weighting, metric=metric, sfreq=sfreq
        )
        flat = per_class[class_a] + per_class[class_b]
        pl.DataFrame(_kernel_rows(flat, fractions_for=(class_a, class_b))).write_csv(
            output_dir / "top_discriminative_kernels.csv"
        )
        log.info(  # noqa: G004
            f"Top class-discriminative kernels (metric={metric}, "
            f"{class_names[class_a]} vs {class_names[class_b]}):"
        )
        for c in labels:
            for info in per_class[c][:5]:
                freq = f" peak={info.peak_freq_hz:.1f}Hz" if info.peak_freq_hz is not None else ""
                log.info(  # noqa: G004
                    f"  favors {class_names[c]} #{info.rank} score={info.score:+.3f} "
                    f"d{info.dilation} {info.representation} g{info.group}k{info.kernel}{freq}"
                )
        if sfreq is not None:
            plot_set = {c: per_class[c][:n_per_class] for c in labels}
            kernel_viz.plot_kernels_by_class(
                plot_set, sfreq, output_dir / "top_discriminative_kernels.png",
                "Top discriminative HYDRA kernels", class_names=class_names,
            )
            kernel_viz.plot_peak_freq_hist(
                flat, sfreq, output_dir / "discriminative_peak_freq_hist.png",
                class_names=class_names,
            )
    else:
        log.info(f"Skipping discriminative ranking (need 2 classes, have {labels}).")  # noqa: G004


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

    # Load/window the data once and save the per-split window metadata, so runs
    # with different signal_mode / ica_keep_labels (same data.seed) can be
    # confirmed to use the same windows.
    datamodule.setup()
    _dump_window_metadata(cfg, datamodule)

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

    # Report the most-used / most discriminative HYDRA kernels when the feature
    # extractor accumulated win counts (feature.track_counts=true). Requires
    # training to have run, so the extractor has processed (and counted) data.
    if cfg.get("train") and cfg.feature.get("track_counts"):
        _report_top_kernels(cfg, feature_extractor)

    # Save the full window- and subject-level performance metrics to CSV.
    scores_by_split = {}
    if cfg.get("train"):
        scores_by_split["train"] = train_metrics
    if cfg.get("test"):
        scores_by_split["test"] = test_metrics
    _write_performance_csv(cfg, scores_by_split)

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
