import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger as log
from pathlib import Path
from pytorch_lightning import LightningDataModule
from sklearn import metrics as skm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

from src.models.components.hydra_transformer import _SparseScaler


def _classification_metrics(y_true, y_pred, y_score) -> dict:
    """Compute binary-classification metrics.

    Parameters
    ----------
    y_true : array-like
        True 0/1 labels.
    y_pred : array-like
        Predicted 0/1 labels.
    y_score : array-like
        Signed decision values (for ROC-AUC / average precision).

    Returns
    -------
    dict
        n / n_pos / n_neg counts plus accuracy, balanced_accuracy, sensitivity,
        specificity, precision, f1, mcc, roc_auc, and average_precision. The
        AUC-based metrics and MCC are NaN when only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    both = n_pos > 0 and n_neg > 0
    return {
        "n": int(y_true.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(skm.recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(skm.recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(skm.precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(y_true, y_pred)) if both else float("nan"),
        "roc_auc": float(skm.roc_auc_score(y_true, y_score)) if both else float("nan"),
        "average_precision": (
            float(skm.average_precision_score(y_true, y_score)) if both else float("nan")
        ),
    }


def _best_subject_threshold(subject_true, subject_score) -> float:
    """Decision threshold on subject scores that maximizes balanced accuracy (val calibration)."""
    subject_true = np.asarray(subject_true).astype(int)
    subject_score = np.asarray(subject_score, dtype=float)
    if len(np.unique(subject_true)) < 2:
        return 0.0
    edges = np.unique(subject_score)
    cands = np.concatenate([[edges[0] - 1e-6], (edges[:-1] + edges[1:]) / 2.0, [edges[-1] + 1e-6]])
    best_thr, best_bal = 0.0, -1.0
    for t in cands:
        bal = skm.balanced_accuracy_score(subject_true, (subject_score > t).astype(int))
        if bal > best_bal:
            best_bal, best_thr = bal, float(t)
    return best_thr


def _dump_prediction_npz(out_path: Path, window_score, window_y, window_subject,
                         subject_ids, subject_score, subject_true, counts) -> None:
    """Write per-window and per-subject (label, score, subject) arrays for offline ROC curves.

    Uses the SAME npz schema as the LuMamba eval dump (``win_prob``/``win_y``/``win_subject`` and
    ``subj_prob``/``subj_y``/``win_per_subj``), so ``scripts/plot_roc_variants.py`` can overlay
    HYDRA against the foundation models as just another 'variant'. The scores are the classifier's
    signed decision-function margin (not a probability in [0, 1]); ROC / AUROC are rank-based, so
    the score SCALE does not matter for the comparison, and each model's curve uses its own scores.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        win_prob=np.asarray(window_score, dtype=np.float32),
        win_y=np.asarray(window_y, dtype=np.int8),
        win_subject=np.asarray(window_subject),
        subj_id=np.asarray(subject_ids),
        subj_prob=np.asarray(subject_score, dtype=np.float32),
        subj_y=np.asarray(subject_true, dtype=np.int8),
        win_per_subj=np.asarray(counts, dtype=np.int32),
    )
    log.info(f"Dumped predictions to {out_path} "
             f"({len(window_score)} windows, {len(subject_ids)} subjects)")


class Trainer:
    """
    Trainer class to handle feature extraction, training, and testing.
    Operates similarly to PyTorch Lightning's Trainer.

    Parameters
    ----------
    merge_train_val : bool, default=True
        If True, fit the classifier on train + val combined (original behavior). If False,
        fit on train only and keep val held out (e.g. for threshold calibration).
    calibrate_threshold : bool, default=False
        If True (and ``merge_train_val`` is False), pick the subject-level decision threshold
        that maximizes balanced accuracy on the held-out val set, and apply it at test time
        (instead of the default 0). Enables a fair same-calibration comparison with LuMamba.
    dump_predictions_dir : str | None, default=None
        If set, write each scored split's raw per-window / per-subject (label, score, subject)
        arrays to ``<dir>/<dump_tag>_<split>.npz`` (LuMamba-compatible schema), so ROC curves can
        be drawn offline and overlaid on the foundation models (scripts/plot_roc_variants.py).
    dump_tag : str, default="hydra"
        Filename stem for the dumps; the launcher sets it to ``w<ws>_s<seed>_hydra_full`` so the
        plotter parses it as the 'hydra' variant at that window / seed.
    group_inner_cv : bool, default=False
        If True, replace an integer ``cv`` of the classifier (e.g. ``LogisticRegressionCV(cv=5)``)
        by subject-grouped, class-stratified folds over the training windows, so windows of one
        subject never sit on both sides of an inner model-selection fold. False keeps sklearn's
        default sample-level folds (the historical behaviour, which leaks subjects across inner
        folds; the outer subject split is unaffected either way).
    inner_cv_seed : int, default=0
        Shuffle seed of the grouped inner folds.
    """
    def __init__(self, merge_train_val: bool = True, calibrate_threshold: bool = False,
                 dump_predictions_dir: str | None = None, dump_tag: str = "hydra",
                 group_inner_cv: bool = False, inner_cv_seed: int = 0):
        self.feature_extractor = None
        self.trained_pipeline = None
        self.merge_train_val = merge_train_val
        self.calibrate_threshold = calibrate_threshold
        self.subject_threshold = 0.0
        self.dump_predictions_dir = dump_predictions_dir
        self.dump_tag = dump_tag
        self.group_inner_cv = group_inner_cv
        self.inner_cv_seed = inner_cv_seed

    def _set_grouped_inner_cv(self, model, y, subjects) -> None:
        """Replace an integer ``cv`` on ``model`` by precomputed subject-grouped stratified folds."""
        n_splits = getattr(model, "cv", None)
        if not isinstance(n_splits, int) or n_splits < 2:
            return
        y = np.asarray(y).astype(int)
        groups = np.asarray(subjects)
        n_splits = min(n_splits, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.inner_cv_seed)
        model.set_params(cv=list(splitter.split(np.zeros(len(y)), y, groups)))
        log.info(f"Inner model-selection folds: {n_splits} subject-grouped stratified folds over "
                 f"{len(np.unique(groups))} training subjects")

    def _get_scores(self, pipeline, data, metadata_df, split_name, subject_threshold=0.0, dump=False):
        log.info(f"Calculating scores for {split_name} data")
        # Window-level: signed decision value per window, thresholded at 0.
        window_score = pipeline.decision_function(data["X"])
        window_pred = (window_score > 0).astype(int)
        metrics_window = _classification_metrics(data["y"], window_pred, window_score)

        # Subject-level: average the signed decision value over each subject's
        # windows, thresholded at 0.
        df = metadata_df[['subject', 'epilepsy']].copy()
        df['score'] = window_score
        grouped = df.groupby('subject')
        subj_mean = grouped['score'].mean()
        subject_score = subj_mean.to_numpy()
        subject_ids = subj_mean.index.to_numpy()
        subject_true = grouped['epilepsy'].first().astype(int).to_numpy()
        subject_pred = (subject_score > subject_threshold).astype(int)
        metrics_subject = _classification_metrics(subject_true, subject_pred, subject_score)

        # Optional raw-score dump for offline ROC curves (same schema as the LuMamba eval dump).
        if dump and self.dump_predictions_dir:
            y_win = data["y"].cpu().numpy() if hasattr(data["y"], "cpu") else np.asarray(data["y"])
            _dump_prediction_npz(
                Path(self.dump_predictions_dir) / f"{self.dump_tag}_{split_name}.npz",
                window_score, y_win, df['subject'].to_numpy(),
                subject_ids, subject_score, subject_true, grouped.size().to_numpy())

        name = split_name.capitalize()
        log.info(
            f"{name} window:  acc={metrics_window['accuracy']:.4f} "
            f"bal_acc={metrics_window['balanced_accuracy']:.4f} "
            f"sens={metrics_window['sensitivity']:.4f} spec={metrics_window['specificity']:.4f} "
            f"auc={metrics_window['roc_auc']:.4f}"
        )
        log.info(
            f"{name} subject: acc={metrics_subject['accuracy']:.4f} "
            f"bal_acc={metrics_subject['balanced_accuracy']:.4f} "
            f"sens={metrics_subject['sensitivity']:.4f} spec={metrics_subject['specificity']:.4f} "
            f"auc={metrics_subject['roc_auc']:.4f}"
        )
        return {
            "window": metrics_window,
            "subject": metrics_subject,
            "accuracy_by_window": metrics_window["accuracy"],
            "accuracy_by_subject": metrics_subject["accuracy"],
        }


    def _extract_features(
            self, 
            feature_extractor: nn.Module,
            dataloader, 
            split_name
        ):
        log.info(f"Extracting features for {split_name} data")
        X_split_batches = []
        y_split_batches = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Feature extraction for {split_name} data"):
                X, y = batch
                # The feature extractor moves X to its device (e.g. GPU) internally.
                # `y` is passed so it can bin per-kernel win counts per class when
                # track_counts is enabled; it does not change the features. Features
                # are moved back to CPU for the downstream sklearn pipeline.
                X_transformed = feature_extractor(X, y).cpu()
                X_split_batches.append(X_transformed)
                y_split_batches.append(y)

        return {
            "X": torch.cat(X_split_batches, dim=0),
            "y": torch.cat(y_split_batches, dim=0).squeeze(),
        }

    def fit(
            self,
            model: nn.Module,
            feature_extractor: nn.Module,
            scaler: nn.Module,
            datamodule: LightningDataModule,
            output_path: str,
            save: bool = True,
            provenance: dict | None = None,
            calibration: dict | None = None,
        ):
        """Extract features, fit the scaler+classifier pipeline, and score on train.

        Parameters
        ----------
        model : nn.Module
            The (sklearn) classifier to fit.
        feature_extractor : nn.Module
            The feature transform applied to each window (HYDRA, or a learned
            extractor exposing ``fit_unsupervised`` such as ShapeConvSAE).
        scaler : nn.Module
            The sparse scaler placed before the classifier in the pipeline.
        datamodule : LightningDataModule
            Provides the train/val dataloaders and per-split metadata.
        output_path : str
            Directory where the fitted artifacts are written when ``save`` is True.
        save : bool, default=True
            If True, dump the pipeline, feature extractor, and scaler to disk. Set
            False for seed sweeps where the per-seed artifacts are not needed.
        provenance : dict | None, default=None
            Recorded by a learned feature extractor when it fits itself here
            (training subjects, data settings, window fingerprint; see
            ``src.utils.split_provenance``). Required when the extractor exposes
            ``fit_unsupervised`` and is not already fitted.
        calibration : dict | None, default=None
            If set and the extractor exposes ``calibrate_amp_min``, calibrate its
            per-atom extraction thresholds on the TRAINING windows with label
            ``keep_label`` (0 = non-epileptic subjects) to a background activation
            rate: ``{"false_alarms_per_channel_minute": r, "sfreq": f, "keep_label": 0}``.
            Skipped for a pretrained extractor that already carries calibrated
            thresholds.

        Returns
        -------
        dict
            Window-level and subject-level training accuracy.
        """
        log.info("Starting feature extraction for training!")
        datamodule.setup()

        # Extract features for train (and val if needed by model, though sklearn pipeline usually just uses train)
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        # Extractors with learnable parameters (e.g. ShapeConvSAE) fit themselves on
        # the training windows first (val is only scored for the loss curve). Labels
        # are never read. A pretrained extractor (feature.pretrained) skips this.
        if hasattr(feature_extractor, "fit_unsupervised"):
            log.info("Fitting the feature extractor without labels on the training windows")
            feature_extractor.fit_unsupervised(train_dataloader, val_dataloader, provenance=provenance)
        if calibration and hasattr(feature_extractor, "calibrate_amp_min"):
            if feature_extractor.calibrated_thresholds is None:
                from src.utils.utils import calibration_provenance

                keep_label = calibration.get("keep_label", 0)
                log.info(f"Calibrating extraction thresholds on training windows with label "
                         f"{keep_label} at {calibration['false_alarms_per_channel_minute']} per channel-minute")
                feature_extractor.calibrate_amp_min(
                    train_dataloader, calibration["false_alarms_per_channel_minute"], calibration["sfreq"],
                    keep_label=keep_label, provenance=calibration_provenance(datamodule, keep_label),
                )
            else:
                log.info("Pretrained extractor already carries calibrated thresholds; keeping them")

        train_data = self._extract_features(feature_extractor, train_dataloader, "train")
        val_data = self._extract_features(feature_extractor, val_dataloader, "val")

        if self.merge_train_val:
            # Original behavior: fit on train + val combined (no held-out val).
            log.info("Combining training and validation data for final training!")
            train_data["X"] = torch.cat([train_data["X"], val_data["X"]], dim=0)
            train_data["y"] = torch.cat([train_data["y"], val_data["y"]], dim=0)
            log.info(f"Combined train+val: X={train_data['X'].shape}, y={train_data['y'].shape}")
            metadata_df = pd.concat([datamodule.train_df, datamodule.val_df], ignore_index=True)
        else:
            # Keep val held out (for threshold calibration / a proper val split).
            log.info("Fitting on TRAIN only; validation held out.")
            metadata_df = datamodule.train_df

        log.info("Starting classifier training!")
        if self.group_inner_cv:
            self._set_grouped_inner_cv(model, train_data["y"], metadata_df["subject"].to_numpy())
        pipeline = make_pipeline(
            scaler,
            model
        )
        pipeline.fit(train_data["X"], train_data["y"])
        log.info("Classifier training completed!")
        # Which subjects the scaler and classifier were fitted on (and, below, calibrated on),
        # so eval.py can refuse a test split that contains any of them.
        pipeline.fit_provenance_ = {
            "fit_subjects": sorted(str(s) for s in metadata_df["subject"].unique()),
            "calibration_subjects": [],
        }

        # Subject-level threshold calibration on the held-out val set (max balanced accuracy),
        # applied at test time; window-level scoring stays at the default 0 threshold.
        if self.calibrate_threshold:
            if self.merge_train_val:
                log.warning("calibrate_threshold ignored: val was merged into train (set merge_train_val=false).")
            else:
                val_score = pipeline.decision_function(val_data["X"])
                vdf = datamodule.val_df[['subject', 'epilepsy']].copy()
                vdf['score'] = val_score
                vg = vdf.groupby('subject')
                self.subject_threshold = _best_subject_threshold(
                    vg['epilepsy'].first().astype(int).to_numpy(), vg['score'].mean().to_numpy()
                )
                pipeline.fit_provenance_["calibration_subjects"] = sorted(str(s) for s in vdf["subject"].unique())
                log.info(f"Calibrated subject threshold on val: {self.subject_threshold:.6f}")

        # Dump the held-out val predictions too (test is dumped in test()), for a val-split ROC.
        if self.dump_predictions_dir:
            self._get_scores(pipeline, val_data, datamodule.val_df, "val",
                             subject_threshold=self.subject_threshold, dump=True)

        train_scores = self._get_scores(pipeline, train_data, metadata_df, "train",
                                        subject_threshold=self.subject_threshold)

        if save:
            self.output_path = Path(output_path)
            self.output_path.mkdir(parents=True, exist_ok=True)
            log.info("Saving trained pipeline!")
            # Save the model
            model_path = self.output_path / "model.joblib"
            log.info(f"Saving model to {model_path}")
            joblib.dump(pipeline, model_path)
            # Save the feature extractor
            feature_extractor_path = self.output_path / "feature_extractor.joblib"
            log.info(f"Saving feature extractor to {feature_extractor_path}")
            joblib.dump(feature_extractor, feature_extractor_path)
            # Learned extractors also write their inspectable artifacts (atoms, history).
            if hasattr(feature_extractor, "save_artifacts"):
                feature_extractor.save_artifacts(self.output_path)
            # Save the scaler
            scaler_path = self.output_path / "scaler.joblib"
            log.info(f"Saving scaler to {scaler_path}")
            joblib.dump(scaler, scaler_path)

        return train_scores

    def test(
            self, 
            model: nn.Module|None = None, 
            feature_extractor: nn.Module|None = None,
            scaler: nn.Module|None = None, 
            datamodule: LightningDataModule|None = None,
        ):

        log.info("Starting testing!")
        pipeline = make_pipeline(
            scaler,
            model
        )
        if datamodule.data_test is None:
            datamodule.setup()
        test_dataloader = datamodule.test_dataloader()
        test_data = self._extract_features(feature_extractor, test_dataloader, "test")
        log.info("Starting evaluation!")
        test_scores = self._get_scores(pipeline, test_data, datamodule.test_df, "test",
                                       subject_threshold=self.subject_threshold, dump=True)
        return test_scores