#%%
"""Contains the LightningDataModule for the TUH EEG Epilepsy dataset."""

from __future__ import annotations
from typing import Any
from loguru import logger
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy
from torchvision.transforms import transforms
import torch
import re
import pandas as pd

#%%
class TUHEEGDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """  # noqa: E501

    def __init__(
        self,
        data_dir: str = "../../data/",
        version: str = 'v3.0.0',
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a `MNISTDataModule`.

        Parameters
        ----------
        data_dir : str, default="data/"
            The data directory
        train_val_test_split : tuple[float, float, float], default=(0.8, 0.1, 0.1)
            The train, validation and test split
        batch_size : int, default=64
            The batch size
        num_workers : int, default=0
            The number of workers
        pin_memory : bool, default=False
            Whether to pin memory

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.version = version
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        # )

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        Returns
        -------
        int
            The number of classes in the TUH EGG Epilepsy dataset (2).

        """
        return 2
    

    def prepare_data(self) -> None:
        """Make sure dataset has been placed in the data folder and create dataframes.
        
        Dataset name should be v3.0.0
        Lightning ensures that `self.prepare_data()` is called only within a single process on CPU, so you can safely add your downloading logic within. In case of multi-node training, the execution of this hook depends upon `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """  
        pass

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Parameters
        ----------
        stage : str | None, default=None
            The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.

        """  # noqa: E501
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                msg = f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."  # noqa: E501
                raise RuntimeError(
                    msg,
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            tuh = TUHEEGEpilepsy(
                data_dir=self.data_dir,
                version=self.version,
                add_annotations=True,
            )            

            # Split the dataset
            data = tuh.load_data(
                mode = 'raw',
                target_name = 'epilepsy',
                preload = True,
                rename_channels = True,
                set_montage = False,
                n_jobs = 1,
                # New args for balanced windowing
                window_len_s = 5*60, # 5 minutes
                overlap_pct = 0.0,
                balance_per_subject = True,
                include_seizures = False,
                fix_length_mode = 'resample', # 'resample', 'pad', or None
                shuffle_windows = True,
                seed = 42,
                splits = {'train': self.train_val_test_split[0],
                          'val': self.train_val_test_split[1],
                          'test': self.train_val_test_split[2]},
                stratify_by = 'epilepsy',
            )
            self.data_train = TensorDataset(data['train'][0], data['train'][1].squeeze())
            self.data_val = TensorDataset(data['val'][0], data['val'][1].squeeze())
            self.data_test = TensorDataset(data['test'][0], data['test'][1].squeeze())


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns
        -------
        DataLoader
            The train dataloader

        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader

        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader

        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,`trainer.test()`, and `trainer.predict()`.

        Parameters
        ----------
        stage : str | None, default=None
            The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.

        """  # noqa: E501

    def state_dict(self) -> dict[Any, Any]:
        """Call when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns
        -------
        dict
            A dictionary containing the datamodule state that you want to save.

        """  # noqa: E501
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Call when loading a checkpoint. Implement to reload datamodule state given datamodule `state_dict()`.

        Parameters
        ----------
        state_dict : dict[str, Any]
            The datamodule state returned by `self.state_dict()`.

        """  # noqa: E501
#%%
if __name__ == "__main__":
    _ = TUHEEGDataModule()
