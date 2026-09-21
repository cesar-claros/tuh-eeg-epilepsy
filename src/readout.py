"""Phase 3 development readout: train / validation, fixed C grid, subject-mean pooling, test untouched.

Implements the four steps of the experimental proposal's Phase 3 for one arm:

1. the feature extractor is fitted on training subjects only (a pretrained SAE
   dictionary already is; a frozen k-means dictionary is fitted here with
   ``feature.train_spec.epochs=0``; HYDRA and the spectral control have no fit);
   the feature scaler is fitted on the training windows;
2. an ordinary L2 logistic regression is fitted on the training windows for every
   ``C`` of the declared grid, the same grid for every arm, with inverse
   window-count subject weights (uniform when every subject has the same number of
   windows, which the manifests are checked for and reported);
3. decision values are pooled by the subject mean, and the arm's ``C`` is selected by
   validation subject AUROC with exact ties broken toward stronger regularization
   (smaller ``C``); a validation balanced accuracy at the validation-optimal
   threshold is reported separately and takes no part in the selection;
4. the test split is never loaded.

Several dictionaries can be read out as an equal-weight ensemble (one training-only
classifier per dictionary at each shared ``C``, subject decision scores averaged).
Writes to ``output_dir``: ``readout_grid.csv`` (one row per ``C``), ``readout_summary.json``
(the selection, the feature dimension, the subjects, the provenance of every
dictionary and of the manifests), ``subject_scores_train.csv`` and
``subject_scores_val.csv`` (subject, label, windows, pooled score at the selected
``C``) and ``window_scores_val.csv``. Run on the HPC from ``code/``::

    python src/readout.py arm=D05 feature.pretrained=logs/sae_phase2/<root>/D05/sae_state.pt \\
        data.lazy_loading=true data.windows_train_csv=... data.windows_val_csv=... data.windows_test_csv=... \\
        data.signal_mode=bipolar data.filter_freq=[1,45] output_dir=logs/readout/<root>/D05
    python src/readout.py arm=hydra feature=hydra_transformer ...
    python src/readout.py arm=spectral feature=spectral ...
    python src/readout.py arm=kmeans_diff feature.train_spec.epochs=0 feature.calibrate_fa=0.1 ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import polars as pl
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

rootutils.setup_root(__file__, pythonpath=True)

from src.models.components.hydra_transformer import _SparseScaler  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    calibration_provenance,
    check_pretrained_provenance,
    check_split_disjoint,
    extras,
    instantiate_feature,
    split_provenance,
    task_wrapper,
    threshold_calibration,
    window_plan_sha256,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)


def _extract(extractor: Any, loader: Any, name: str) -> tuple[torch.Tensor, np.ndarray]:
    """Features ``(n, d)`` on CPU and labels ``(n,)`` of a loader, in loader order."""
    parts, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            parts.append(extractor(x, y).detach().cpu().float())
            labels.append(torch.as_tensor(y).reshape(-1))
    features = torch.cat(parts)
    log.info(f"{name}: features {tuple(features.shape)}, finite {bool(torch.isfinite(features).all())}")  # noqa: G004
    return features, torch.cat(labels).numpy().astype(int)


def _subject_mean(scores: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean decision value per subject: (subjects sorted, pooled scores, window counts)."""
    unique, inverse, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    pooled = np.bincount(inverse, weights=scores) / counts
    return unique, pooled, counts


def _subject_labels(labels: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """One label per sorted subject; refuses a subject with mixed labels."""
    unique, inverse = np.unique(subjects, return_inverse=True)
    per_subject = np.full(len(unique), -1)
    for i, label in zip(inverse, labels):
        if per_subject[i] not in (-1, label):
            raise ValueError(f"subject {unique[i]} has mixed labels")
        per_subject[i] = label
    return per_subject


def _best_threshold_balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Balanced accuracy at the threshold that maximizes it on the given scores (reported, not used for selection)."""
    best, best_threshold = -1.0, 0.0
    for threshold in np.unique(scores):
        value = balanced_accuracy_score(labels, (scores >= threshold).astype(int))
        if value > best:
            best, best_threshold = value, float(threshold)
    return float(best), best_threshold


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The grid row with the highest validation subject AUROC; exact ties go to the smaller ``C``."""
    return max(rows, key=lambda r: (round(float(r["val_subject_auroc"]), 12), -float(r["C"])))


def _make_scaler(kind: str) -> Any:
    """The scaler of a feature family: sparse-aware for counts, standard for dense spectral features."""
    return StandardScaler() if kind == "spectral" else _SparseScaler()


def _to_numpy(scaled: Any) -> np.ndarray:
    return scaled.numpy() if isinstance(scaled, torch.Tensor) else np.asarray(scaled)


def _build_extractors(cfg: DictConfig, datamodule: Any, output_dir: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    """The extractor(s) of the arm and their provenance records; a frozen k-means dictionary is fitted here."""
    kind = cfg.features
    if kind == "sae":
        paths = list(cfg.dictionaries) if cfg.get("dictionaries") else [cfg.feature.pretrained]
        if not paths or not all(paths):
            raise ValueError("features=sae needs feature.pretrained or a dictionaries list")
        extractors, records = [], []
        for path in paths:
            extractor = instantiate_feature(cfg.feature, pretrained=str(path))
            check_pretrained_provenance(extractor, datamodule, cfg.data, check_val=True)
            if extractor.calibrated_thresholds is None:
                log.warning(f"{path}: no calibrated thresholds; the count block uses amp_min")  # noqa: G004
            extractors.append(extractor)
            records.append({"path": str(path), "fit_meta": extractor.fit_meta, "provenance": extractor.provenance})
        return extractors, records
    if kind == "kmeans":
        if int(cfg.feature.train_spec.epochs) != 0:
            raise ValueError("features=kmeans is the initialized dictionary: set feature.train_spec.epochs=0")
        extractor = instantiate_feature(cfg.feature)
        provenance = split_provenance(datamodule, cfg.data)
        extractor.fit_unsupervised(datamodule.train_dataloader(), None, provenance=provenance)
        calibration = threshold_calibration(cfg)
        if calibration is not None:
            keep = calibration["keep_label"]
            extractor.calibrate_amp_min(
                datamodule.train_dataloader(), calibration["false_alarms_per_channel_minute"], calibration["sfreq"],
                keep_label=keep, provenance=calibration_provenance(datamodule, keep),
            )
        extractor.save_artifacts(output_dir)
        return [extractor], [{"path": str(output_dir), "fit_meta": extractor.fit_meta, "provenance": provenance}]
    if kind in ("hydra", "spectral"):
        extractor = instantiate_feature(cfg.feature)
        check_pretrained_provenance(extractor, datamodule, cfg.data, check_val=True)
        return [extractor], [{"path": None, "settings": OmegaConf.to_container(cfg.feature, resolve=True)}]
    raise ValueError(f"features must be sae, kmeans, hydra or spectral, got {kind!r}")


@task_wrapper
def readout(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the development readout of one arm.

    Raises
    ------
    ValueError
        On a wrong arm specification, a split defect, or a subject with mixed labels.
    """
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    # The test split is never loaded here; only its plan hash is recorded.
    train_loader, val_loader = datamodule.train_dataloader(), datamodule.val_dataloader()
    train_df, val_df = datamodule.train_df, datamodule.val_df
    if len(train_df) != len(train_loader.dataset) or len(val_df) != len(val_loader.dataset):
        raise ValueError("window metadata and loaders are not aligned")

    extractors, records = _build_extractors(cfg, datamodule, output_dir)
    c_grid = [float(c) for c in cfg.c_grid]
    subjects_train = train_df["subject"].astype(str).to_numpy()
    subjects_val = val_df["subject"].astype(str).to_numpy()
    windows_per_subject = np.unique(subjects_train, return_counts=True)[1]
    weights = None
    if cfg.subject_weights:
        _, inverse, counts = np.unique(subjects_train, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
        weights = weights * len(weights) / weights.sum()

    # One scaler and one classifier per dictionary; subject scores averaged over dictionaries.
    features_train, features_val, y_train, y_val = [], [], None, None
    for i, extractor in enumerate(extractors):
        ft, y_train = _extract(extractor, train_loader, f"train (extractor {i})")
        fv, y_val = _extract(extractor, val_loader, f"val (extractor {i})")
        scaler = _make_scaler(cfg.features)
        features_train.append(_to_numpy(scaler.fit_transform(ft)))
        features_val.append(_to_numpy(scaler.transform(fv)))
    labels_train_subject = _subject_labels(y_train, subjects_train)
    labels_val_subject = _subject_labels(y_val, subjects_val)

    rows, pooled_by_c = [], {}
    for c in c_grid:
        train_scores = np.zeros(len(y_train))
        val_scores = np.zeros(len(y_val))
        for ft, fv in zip(features_train, features_val):
            clf = LogisticRegression(C=c, max_iter=5000, solver="lbfgs")  # L2 penalty (the default)
            clf.fit(ft, y_train, sample_weight=weights)
            train_scores += clf.decision_function(ft) / len(features_train)
            val_scores += clf.decision_function(fv) / len(features_val)
        _, pooled_train, _ = _subject_mean(train_scores, subjects_train)
        subjects_sorted, pooled_val, counts_val = _subject_mean(val_scores, subjects_val)
        val_bal_acc, val_threshold = _best_threshold_balanced_accuracy(labels_val_subject, pooled_val)
        row = {
            "C": c,
            "train_window_auroc": float(roc_auc_score(y_train, train_scores)),
            "val_window_auroc": float(roc_auc_score(y_val, val_scores)),
            "train_subject_auroc": float(roc_auc_score(labels_train_subject, pooled_train)),
            "val_subject_auroc": float(roc_auc_score(labels_val_subject, pooled_val)),
            "val_subject_balanced_accuracy_at_val_threshold": val_bal_acc,
            "val_threshold": val_threshold,
        }
        rows.append(row)
        pooled_by_c[c] = (subjects_sorted, pooled_val, counts_val, pooled_train, val_scores)
        log.info(f"C={c:g}: {row}")  # noqa: G004

    # Selection: maximum validation subject AUROC; exact ties toward stronger regularization (smaller C).
    best = _select(rows)
    selected_c = best["C"]
    subjects_sorted, pooled_val, counts_val, pooled_train, val_scores = pooled_by_c[selected_c]
    grid = pl.DataFrame(rows)
    grid.write_csv(output_dir / "readout_grid.csv")
    train_subjects_sorted, _, counts_train = _subject_mean(np.zeros(len(y_train)), subjects_train)
    pl.DataFrame({
        "subject": train_subjects_sorted, "label": labels_train_subject, "windows": counts_train, "score": pooled_train,
    }).write_csv(output_dir / "subject_scores_train.csv")
    pl.DataFrame({
        "subject": subjects_sorted, "label": labels_val_subject, "windows": counts_val, "score": pooled_val,
    }).write_csv(output_dir / "subject_scores_val.csv")
    pl.DataFrame({
        "subject": subjects_val, "window": np.arange(len(y_val)), "label": y_val, "score": val_scores,
    }).write_csv(output_dir / "window_scores_val.csv")

    summary = {
        "arm": cfg.arm,
        "features": cfg.features,
        "feature_dim": int(features_train[0].shape[1]),
        "n_dictionaries": len(extractors),
        "c_grid": c_grid,
        "selected_C": selected_c,
        "selection_rule": "max validation subject AUROC, exact ties toward smaller C",
        "selected": best,
        "subject_weights": "inverse window count" if cfg.subject_weights else "none",
        "windows_per_train_subject": {"min": int(windows_per_subject.min()), "max": int(windows_per_subject.max())},
        "train_subjects": int(len(labels_train_subject)), "train_positive": int(labels_train_subject.sum()),
        "val_subjects": int(len(labels_val_subject)), "val_positive": int(labels_val_subject.sum()),
        "manifests": {
            "train_windows_sha256": window_plan_sha256(train_df), "val_windows_sha256": window_plan_sha256(val_df),
            "test_windows_sha256": (
                window_plan_sha256(datamodule.test_df) if getattr(datamodule, "test_df", None) is not None else None
            ),
        },
        "extractors": records,
        "data": OmegaConf.to_container(cfg.data, resolve=True),
    }
    (output_dir / "readout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    log.info(  # noqa: G004
        f"{cfg.arm}: selected C={selected_c:g}, validation subject AUROC {best['val_subject_auroc']:.3f} "
        f"(train {best['train_subject_auroc']:.3f}; window {best['val_window_auroc']:.3f}); test untouched"
    )
    metrics = {f"val_subject_auroc_C{c:g}": r["val_subject_auroc"] for c, r in zip(c_grid, rows)}
    metrics["val_subject_auroc_selected"] = best["val_subject_auroc"]
    return metrics, {"cfg": cfg, "datamodule": datamodule, "extractors": extractors, "grid": grid}


@hydra.main(version_base="1.3", config_path="../configs", config_name="readout.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for the Phase 3 development readout.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.
    """
    extras(cfg)
    readout(cfg)


if __name__ == "__main__":
    main()
