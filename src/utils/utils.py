"""Utility functions."""

import hashlib
import warnings
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.

    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    -------
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Parameters
    ----------
    task_func : Callable
        The task function to be wrapped.

    Returns
    -------
    Callable
        The wrapped task function.

    """  # noqa: E501, D401

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
            # object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory
            # errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex  # noqa: TRY201

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")  # noqa: G004

        return metric_dict, object_dict
        # return object_dict

    return wrap


def get_metric_value(
    metric_dict: dict[str, Any],
    metric_name: str | None,
) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    Parameters
    ----------
    metric_dict : dict[str, Any]
        A dict containing metric values.
    metric_name : str | None
        If provided, the name of the metric to retrieve.

    Returns
    -------
    float | None
        If a metric name was provided, the value of the metric.

    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        msg = (
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
        raise Exception(  # noqa: TRY002
            msg,
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")  # noqa: G004

    return metric_value


_PROVENANCE_DATA_KEYS = (
    "version", "signal_mode", "target_sfreq", "filter_freq", "notch_freqs", "window_len_min",
    "interpolate_bad_channels", "drop_bad_segments", "apply_aas", "seed", "train_val_test_split",
)
_PROVENANCE_MATCH_KEYS = ("signal_mode", "target_sfreq", "filter_freq", "notch_freqs")


def _plain(value: Any) -> Any:
    """Convert an OmegaConf node (or any value) to plain Python containers."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def split_provenance(datamodule: Any, data_cfg: DictConfig) -> dict[str, Any]:
    """Provenance of a learned feature extractor: training subjects, data settings, window-plan fingerprint.

    Recorded in the extractor's checkpoint by ``fit_unsupervised`` and checked by
    ``check_pretrained_provenance`` when the checkpoint is reused.

    Parameters
    ----------
    datamodule : Any
        A set-up datamodule exposing ``train_df`` (columns ``subject``, ``path``,
        ``start``, ``end``).
    data_cfg : DictConfig
        The ``data`` config of the run.

    Returns
    -------
    dict[str, Any]
        ``train_subjects`` (sorted), ``data`` (the settings that define the signal),
        and ``train_windows_sha256`` (a hash of the training window plan).
    """
    df = datamodule.train_df
    subjects = sorted(str(s) for s in df["subject"].unique())
    plan = "\n".join(f"{p}|{s}|{e}" for p, s, e in zip(df["path"], df["start"], df["end"]))
    return {
        "train_subjects": subjects,
        "data": {k: _plain(data_cfg.get(k)) for k in _PROVENANCE_DATA_KEYS if k in data_cfg},
        "train_windows_sha256": hashlib.sha256(plan.encode()).hexdigest(),
    }


def check_pretrained_provenance(
    feature_extractor: Any, datamodule: Any, data_cfg: DictConfig, check_val: bool = False
) -> None:
    """Refuse a fitted extractor whose training subjects or data settings do not fit this run.

    A dictionary trained by ``src/train_sae.py`` (or inside an earlier ``train``
    run) records its provenance. Reusing it on a split drawn with another
    ``data.seed`` can put its training subjects in the test set, which leaks them
    into the features even though no label was used; reusing it on another
    montage, rate or filter changes what its atoms mean. Extractors without
    provenance (HYDRA, or a not-yet-fitted SAE) pass.

    Parameters
    ----------
    feature_extractor : Any
        The instantiated (or loaded) feature extractor.
    datamodule : Any
        A set-up datamodule exposing ``val_df`` and ``test_df``.
    data_cfg : DictConfig
        The ``data`` config of the current run.
    check_val : bool
        Also refuse overlap with the validation subjects (when val is held out).

    Raises
    ------
    ValueError
        On subject overlap with the test (or val) split, or on a data-setting mismatch.
    """
    provenance = getattr(feature_extractor, "provenance", None) or {}
    trained_on = set(provenance.get("train_subjects", []))
    if not trained_on:
        return
    splits = ["test", "val"] if check_val else ["test"]
    for split in splits:
        df = getattr(datamodule, f"{split}_df", None)
        if df is None or not len(df):
            continue
        overlap = sorted(trained_on & {str(s) for s in df["subject"].unique()})
        if overlap:
            raise ValueError(
                f"{len(overlap)} {split} subjects were used to train the pretrained feature extractor "
                f"(e.g. {overlap[:5]}); rerun with the data.seed of the train_sae run"
            )
    saved = provenance.get("data", {})
    mismatch = {
        k: (saved.get(k), _plain(data_cfg.get(k)))
        for k in _PROVENANCE_MATCH_KEYS
        if k in saved and saved.get(k) != _plain(data_cfg.get(k))
    }
    if mismatch:
        raise ValueError(f"pretrained feature extractor data settings differ from this run (saved, run): {mismatch}")


def dump_window_metadata(output_dir: Path, datamodule: Any) -> None:
    """Save the per-split window metadata to ``windows_<split>.csv`` in ``output_dir``.

    Each file lists the windows (subject, path, start, end, ...) used in that
    split. The windowing is deterministic in ``data.seed`` and independent of
    ``signal_mode``, so the files let you confirm that two runs used exactly the
    same windows, or fix the split of a later run (``data.windows_*_csv``).

    Parameters
    ----------
    output_dir : Path
        Run output directory; created if absent.
    datamodule : Any
        A set-up datamodule exposing ``train_df`` / ``val_df`` / ``test_df``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df = getattr(datamodule, f"{split}_df", None)
        if df is not None and len(df):
            df.to_csv(output_dir / f"windows_{split}.csv", index=False)
            log.info(f"Saved {len(df)} {split} window rows to windows_{split}.csv")  # noqa: G004
