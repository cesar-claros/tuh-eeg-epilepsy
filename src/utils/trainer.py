import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger as log
from pathlib import Path
from pytorch_lightning import LightningDataModule
from sklearn import metrics as skm
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
    """
    def __init__(self, merge_train_val: bool = True, calibrate_threshold: bool = False):
        self.feature_extractor = None
        self.trained_pipeline = None
        self.merge_train_val = merge_train_val
        self.calibrate_threshold = calibrate_threshold
        self.subject_threshold = 0.0

    def _get_scores(self, pipeline, data, metadata_df, split_name, subject_threshold=0.0):
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
        subject_score = grouped['score'].mean().to_numpy()
        subject_true = grouped['epilepsy'].first().astype(int).to_numpy()
        subject_pred = (subject_score > subject_threshold).astype(int)
        metrics_subject = _classification_metrics(subject_true, subject_pred, subject_score)

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
        ):
        """Extract features, fit the scaler+classifier pipeline, and score on train.

        Parameters
        ----------
        model : nn.Module
            The (sklearn) classifier to fit.
        feature_extractor : nn.Module
            The HYDRA feature transform applied to each window.
        scaler : nn.Module
            The sparse scaler placed before the classifier in the pipeline.
        datamodule : LightningDataModule
            Provides the train/val dataloaders and per-split metadata.
        output_path : str
            Directory where the fitted artifacts are written when ``save`` is True.
        save : bool, default=True
            If True, dump the pipeline, feature extractor, and scaler to disk. Set
            False for seed sweeps where the per-seed artifacts are not needed.

        Returns
        -------
        dict
            Window-level and subject-level training accuracy.
        """
        log.info("Starting feature extraction for training!")
        datamodule.setup()

        # Extract features for train (and val if needed by model, though sklearn pipeline usually just uses train)
        train_dataloader = datamodule.train_dataloader()
        train_data = self._extract_features(feature_extractor, train_dataloader, "train")

        val_dataloader = datamodule.val_dataloader()
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
        pipeline = make_pipeline(
            scaler,
            model
        )
        pipeline.fit(train_data["X"], train_data["y"])
        log.info("Classifier training completed!")

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
                log.info(f"Calibrated subject threshold on val: {self.subject_threshold:.6f}")

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
                                       subject_threshold=self.subject_threshold)
        return test_scores