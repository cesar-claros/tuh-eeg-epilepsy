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
    """`LightningDataModule` for the TUH EEG Epilepsy Corpus.

    Loads the Temple University Hospital (TUH) EEG Epilepsy Corpus, slices each
    recording into fixed-length windows (5 minutes by default), balances and
    splits them at the subject level (so no patient leaks across train/val/test),
    and exposes them as `TensorDataset`s of `(channels, time)` windows with a
    binary epilepsy target. The per-split metadata DataFrames (`train_df`,
    `val_df`, `test_df`) are retained so the custom Trainer can aggregate
    window-level predictions into subject-level scores.

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
        window_len_min: int = 5, # 5 minutes
        overlap_pct: float = 0.0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,  # noqa: FBT001, FBT002
        seed: int = 42,
        dict_learning_window_len_s: float = 2.0,
        signal_mode: str = 'raw',
        ica_keep_labels: tuple = ('brain', 'other'),
    ) -> None:
        """Initialize a `TUHEEGDataModule`.

        Parameters
        ----------
        data_dir : str, default="../../data/"
            The data directory containing the dataset version folder.
        version : str, default="v3.0.0"
            The corpus version subdirectory to load.
        train_val_test_split : tuple[float, float, float], default=(0.8, 0.1, 0.1)
            The subject-level train, validation and test split ratios.
        window_len_min : int, default=5
            Window length in minutes.
        overlap_pct : float, default=0.0
            Fractional overlap between consecutive windows.
        batch_size : int, default=64
            The batch size.
        num_workers : int, default=0
            The number of dataloader workers.
        pin_memory : bool, default=False
            Whether to pin memory.
        seed : int, default=42
            Seed for windowing, shuffling, and the subject-level split.
        dict_learning_window_len_s : float, default=2.0
            Window length in seconds for the separate dictionary-learning load
            pass (a short-window view, distinct from the main `window_len_min`).
        signal_mode : str, default='raw'
            'raw' for sensor-space EEG; 'ica_clean' to keep only the ICs whose
            ICLabel is in `ica_keep_labels` (back-projected to sensor space); or
            'brain_ic' to feed the region-binned IC sources directly (each kept IC
            is assigned to a scalp region by its dominant electrode and summed).
        ica_keep_labels : tuple, default=('brain', 'other')
            For 'ica_clean', the ICLabel categories to keep; all others (the
            confident artifacts) are excluded. For 'brain_ic', the categories
            whose ICs build the regional series (set ('brain',) for brain-only).

        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.seed = seed
        self.data_dir = data_dir
        self.version = version
        self.train_val_test_split = train_val_test_split
        self.window_len_min = window_len_min
        self.overlap_pct = overlap_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dict_learning_window_len_s = dict_learning_window_len_s
        self.signal_mode = signal_mode
        self.ica_keep_labels = ica_keep_labels

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        # )

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        
        self.train_df:pd.DataFrame | None = None
        self.val_df:pd.DataFrame | None = None
        self.test_df:pd.DataFrame | None = None

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
                ica_keep_labels=self.ica_keep_labels,
            )

            # Split the dataset
            data = tuh.load_data(
                mode = self.signal_mode,
                target_name = 'epilepsy',
                preload = True,
                rename_channels = True,
                set_montage = False,
                n_jobs = 1,
                # New args for balanced windowing
                window_len_s = self.window_len_min*60, # 5 minutes
                overlap_pct = self.overlap_pct,
                balance_per_subject = True,
                include_seizures = False,
                fix_length_mode = 'resample', # 'resample', 'pad', or None
                shuffle_windows = True,
                seed = self.seed,
                splits = {'train': self.train_val_test_split[0],
                          'val': self.train_val_test_split[1],
                          'test': self.train_val_test_split[2]},
                stratify_by = 'epilepsy',
            )
            self.data_train = TensorDataset(data['train'][0], data['train'][1].squeeze())
            self.data_val = TensorDataset(data['val'][0], data['val'][1].squeeze())
            self.data_test = TensorDataset(data['test'][0], data['test'][1].squeeze())

            self.train_df = data['train'][2]
            self.val_df = data['val'][2]
            self.test_df = data['test'][2]

            # Split the dataset for dictionary learning
            data_dictionary_learning = tuh.load_data(
                mode = self.signal_mode,
                target_name = 'epilepsy',
                preload = True,
                rename_channels = True,
                set_montage = False,
                n_jobs = 1,
                # New args for balanced windowing
                window_len_s = self.dict_learning_window_len_s, # dictionary-learning window length (seconds)
                dictionary_learning=True,
                n_windows_per_subject=10, # limit to 10 windows per subject for dictionary learning to manage memory and training time. These will be randomly selected from the full set of possible windows for each subject.
                overlap_pct = self.overlap_pct,
                balance_per_subject = True,
                include_seizures = False,
                fix_length_mode = 'resample', # 'resample', 'pad', or None
                shuffle_windows = True,
                seed = self.seed,
                idx_list = self.train_df['subject'].unique().tolist(), # only use training subjects for dictionary learning
                stratify_by = 'epilepsy',
            )
            logger.info(f"Dictionary Learning Dataset: Generated {data_dictionary_learning[0].shape} windows for dictionary learning.")
            logger.info(f"Example window shape for dictionary learning:\n {data_dictionary_learning[-1].head(30)}")


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
            shuffle=False, # Shuffling is handled in the TUHEEGEpilepsy windowing, so we set shuffle to False here
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
