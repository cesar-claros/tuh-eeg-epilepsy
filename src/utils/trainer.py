import torch
import joblib
from pathlib import Path
from src.models.components.hydra_transformer import _SparseScaler
from sklearn.pipeline import make_pipeline

class Trainer:
    """
    Trainer class to handle feature extraction, training, and testing.
    Operates similarly to PyTorch Lightning's Trainer.
    """
    def __init__(self,  ):
        self.feature_extractor = None
        self.trained_pipeline = None

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
            X_transformed = feature_extractor(X)
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
            output_path: str
        ):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        log.info("Starting feature extraction for training!")
        datamodule.setup()
        
        # Extract features for train (and val if needed by model, though sklearn pipeline usually just uses train)
        train_dataloader = datamodule.train_dataloader()
        train_data = self._extract_features(feature_extractor, train_dataloader, "train")
        
        log.info("Starting classifier training!")
        pipeline = make_pipeline(
            scaler,
            model
        )
        pipeline.fit(train_data["X"], train_data["y"])
        log.info("Classifier training completed!")
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
        score = pipeline.score(test_data["X"], test_data["y"])
        log.info(f"Test accuracy: {score:.4f}")
        return score