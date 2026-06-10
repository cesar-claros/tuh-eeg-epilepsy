import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from loguru import logger as log
from tqdm import tqdm
import joblib
from pathlib import Path
from src.models.components.hydra_transformer import _SparseScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

class Trainer:
    """
    Trainer class to handle feature extraction, training, and testing.
    Operates similarly to PyTorch Lightning's Trainer.
    """
    def __init__(self,  ):
        self.feature_extractor = None
        self.trained_pipeline = None

    def _get_scores(self, pipeline, data, metadata_df, split_name):
        log.info(f"Calculating scores for {split_name} data")
        # Scores by quick window-level predictions
        score_by_window = pipeline.score(data["X"], data["y"])
        # Scores by subject-level predictions
        df = metadata_df[['subject','epilepsy']].copy().set_index('subject')
        df.loc[:, 'score'] = pipeline.decision_function(data["X"])
        mean_scores_by_subject = df.groupby(df.index)['score'].mean()
        epilepsy_by_subject = df.groupby(df.index)['epilepsy'].first()  # Get epilepsy label for each subject
        accuracy_by_subject = (mean_scores_by_subject > 0).astype(bool) == epilepsy_by_subject.astype(bool)
        log.info(f"{split_name.capitalize()} accuracy by window: {score_by_window:.4f}")
        log.info(f"{split_name.capitalize()} accuracy by subject: {accuracy_by_subject.mean():.4f}")
        return {
            "accuracy_by_window": score_by_window,
            "accuracy_by_subject": accuracy_by_subject.mean(),
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
        for batch in tqdm(dataloader, desc=f"Feature extraction for {split_name} data"):
            X, y = batch
            # Ensure X is on the same device as feature_extractor if needed (assuming CPU for now based on context)
            # `y` is passed so the feature extractor can bin per-kernel win counts
            # per class when track_counts is enabled; it does not change features.
            X_transformed = feature_extractor(X, y)
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

        # Combine train and val data for training the sklearn model, or use val for early stopping if desired (not implemented here)
        log.info("Combining training and validation data for final training!")
        train_data["X"] = torch.cat([train_data["X"], val_data["X"]], dim=0)
        train_data["y"] = torch.cat([train_data["y"], val_data["y"]], dim=0)
        log.info(f"Training data shape after combining train and val: X={train_data['X'].shape}, y={train_data['y'].shape}")

        log.info("Starting classifier training!")
        pipeline = make_pipeline(
            scaler,
            model
        )
        pipeline.fit(train_data["X"], train_data["y"])

        log.info("Classifier training completed!")
        metadata_df = pd.concat([datamodule.train_df, datamodule.val_df], ignore_index=True)
        train_scores = self._get_scores(pipeline, train_data, metadata_df, "train")

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
        test_scores = self._get_scores(pipeline, test_data, datamodule.test_df, "test")
        return test_scores