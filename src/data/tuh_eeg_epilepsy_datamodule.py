#%%
"""Contains the LightningDataModule for the TUH EEG Epilepsy dataset."""

from __future__ import annotations
from typing import Any
from loguru import logger
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch
import re
import pandas as pd
#%%
# import rootutils
# rootutils.setup_root(__file__, pythonpath=True)
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
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a `MNISTDataModule`.

        Parameters
        ----------
        data_dir : str, default="data/"
            The data directory
        train_val_test_split : tuple[int, int, int], default=(55_000, 5_000, 10_000)
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
    
    # def build_tree(
    #     self,
    #     root: Path,
    #     max_depth: int | None = None,
    #     include_hidden: bool = False,
    # ) -> list[str]:
    #     """
    #     Returns a list of lines representing the folder/file structure under `root`.
    #     """
    #     if not root.exists():
    #         raise FileNotFoundError(f"Path does not exist: {root}")
    #     if not root.is_dir():
    #         # If a file is provided, just show it
    #         return [root.name]

    #     def is_hidden(p: Path) -> bool:
    #         # Cross-platform-ish: dotfiles on Unix, and dot-prefix on Windows too.
    #         return p.name.startswith(".")

    #     lines: list[str] = [root.name]

    #     def walk(dir_path: Path, prefix: str, depth: int) -> None:
    #         if max_depth is not None and depth >= max_depth:
    #             return

    #         try:
    #             entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    #         except PermissionError:
    #             lines.append(prefix + "└── [Permission denied]")
    #             return

    #         if not include_hidden:
    #             entries = [e for e in entries if not is_hidden(e)]

    #         for i, entry in enumerate(entries):
    #             is_last = i == len(entries) - 1
    #             branch = "└── " if is_last else "├── "
    #             lines.append(prefix + branch + entry.name)

    #             if entry.is_dir():
    #                 extension = "    " if is_last else "│   "
    #                 walk(entry, prefix + extension, depth + 1)

    #     walk(root, "", 0)
    #     return lines
    
    # def generate_tree(
    #     self,
    #     root: Path,
    #     max_depth: int | None = None,
    #     include_hidden: bool = False,
    #     output_path: Path | None = None,
    # ) -> None:
    #     """
    #     Generates the tree style directory listing.
    #     """
    #     # Assigns Path for output file
    #     if output_path is None:
    #         filename = root.name+'.txt'
    #         output_path = root.parent/filename
    #     # Verify file existis
    #     if output_path.exists():
    #         logger.info(f'{output_path} file already exists.' )
    #     else:
    #         root = root.expanduser().resolve()
    #         lines = self.build_tree(root, max_depth=max_depth, include_hidden=include_hidden)
    #         text = "\n".join(lines)
    #         # output_path = f'../../data/v3.0.0/0{epilepsy_flag}_no_epilepsy_tree.txt'
    #         out = Path(output_path).expanduser().resolve()
    #         out.write_text(text, encoding="utf-8")
    
    # def parse_tree_file(tree_path: Path):
    #     """
    #     Yield tuples of (root_name, subject_id, session_id, year, montage, filename).
    #     """
    #     lines = tree_path.read_text(encoding="utf-8").splitlines()
    #     tree_line_re = re.compile(r"^(?P<prefix>(?:│   |    )*)(?:├── |└── )?(?P<name>.+)$")

    #     root_name = lines[0].strip()
    #     stack = [root_name]  # level 0
    #     current_subject = None
    #     current_session = None
    #     current_year = None
    #     current_montage = None

    #     for raw in lines[1:]:
    #         m = tree_line_re.match(raw)
    #         if not m:
    #             continue
    #         prefix = m.group("prefix")
    #         name = m.group("name").strip()

    #         depth = len(prefix) // 4 + 1  # depth relative to root
    #         # trim stack to depth
    #         stack = stack[:depth]
    #         stack.append(name)

    #         if depth == 1:  # subject
    #             current_subject = name
    #         elif depth == 2:  # session
    #             session_id, year = name.split("_", 1)
    #             current_session = session_id
    #             current_year = year
    #         elif depth == 3:  # montage
    #             current_montage = name
    #         elif depth == 4:  # files
    #             yield (
    #                 root_name,
    #                 current_subject,
    #                 current_session,
    #                 current_year,
    #                 current_montage,
    #                 name,
    #             )

    # def read_duration_from_csv(csv_path: Path) -> float:
    #     """
    #     Reads the duration from the CSV third line:
    #     '# duration = 1214.0000 secs'
    #     The 4th segment (0-index 3) is the duration.
    #     """
    #     with csv_path.open("r", encoding="utf-8") as f:
    #         for i, line in enumerate(f):
    #             if i == 2:  # third line
    #                 parts = line.strip().split(" ")
    #                 return float(parts[3])
    #     raise ValueError(f"No duration line in {csv_path}")

    # def read_annotations_from_csv(csv_path: Path, n_header:int=5) -> float:
    #     """
    #     Reads the annotations from the CSV starting at n_header line of the csv file:
    #     # version = csv_v1.0.0
    #     # bname = aaaaaanr_s001_t001
    #     # duration = 1214.0000 secs
    #     # montage_file = 02_tcp_le_montage.txt
    #     #
    #     channel,start_time,stop_time,label,confidence
    #     FP1-F7,0.0000,0.9802,bckg,1.0000
    #     """
    #     return pd.read_csv(csv_path, sep=',', header=n_header)

    def prepare_data(self) -> None:
        """Make sure dataset has been placed in the data folder and create dataframes.
        
        Dataset name should be v3.0.0
        Lightning ensures that `self.prepare_data()` is called only within a single process on CPU, so you can safely add your downloading logic within. In case of multi-node training, the execution of this hook depends upon `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """  # noqa: E501
        # EEG_PATH = 'v3.0.0/'
        # EPILEPSY_PATH = '00_epilepsy/'
        # NO_EPILEPSY_PATH = '01_no_epilepsy/'
        # DEPTH = 4 # subject_ID/session/montage/*.edf,*.csv,*.csv_bi
        # dataset_path = Path(self.data_dir+EEG_PATH+'/')
        # if dataset_path.exists():
        #     logger.info("Generating tree-style directory listing...")
        #     self.generate_tree(dataset_path/EPILEPSY_PATH, max_depth=DEPTH)
        #     self.generate_tree(dataset_path/NO_EPILEPSY_PATH, max_depth=DEPTH)
        #     logger.info("Extracting annotations...")
        # else:
        #     raise FileNotFoundError(f"No such directory: '{self.data_dir+EEG_PATH}'")
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

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
            trainset = MNIST(
                self.data_dir,
                train=True,
                transform=self.transforms,
            )
            testset = MNIST(
                self.data_dir,
                train=False,
                transform=self.transforms,
            )
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

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
datamod = TUHEEGDataModule()
#%%
datamod.prepare_data()
#%%
if __name__ == "__main__":
    _ = TUHEEGDataModule()



#%%


# def build_tree(
#     root: Path,
#     max_depth: int | None = None,
#     include_hidden: bool = False,
# ) -> list[str]:
#     """
#     Returns a list of lines representing the folder/file structure under `root`.
#     """
#     if not root.exists():
#         raise FileNotFoundError(f"Path does not exist: {root}")
#     if not root.is_dir():
#         # If a file is provided, just show it
#         return [root.name]

#     def is_hidden(p: Path) -> bool:
#         # Cross-platform-ish: dotfiles on Unix, and dot-prefix on Windows too.
#         return p.name.startswith(".")

#     lines: list[str] = [root.name]

#     def walk(dir_path: Path, prefix: str, depth: int) -> None:
#         if max_depth is not None and depth >= max_depth:
#             return

#         try:
#             entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
#         except PermissionError:
#             lines.append(prefix + "└── [Permission denied]")
#             return

#         if not include_hidden:
#             entries = [e for e in entries if not is_hidden(e)]

#         for i, entry in enumerate(entries):
#             is_last = i == len(entries) - 1
#             branch = "└── " if is_last else "├── "
#             lines.append(prefix + branch + entry.name)

#             if entry.is_dir():
#                 extension = "    " if is_last else "│   "
#                 walk(entry, prefix + extension, depth + 1)

#     walk(root, "", 0)
#     return lines

# %%
# epilepsy_flag = 1
# root = Path(f'../../data/v3.0.0/0{epilepsy_flag}_no_epilepsy/').expanduser().resolve()
# lines = build_tree(root, max_depth=4)
# text = "\n".join(lines)
# output = f'../../data/v3.0.0/0{epilepsy_flag}_no_epilepsy_tree.txt'
# out_path = Path(output).expanduser().resolve()
# out_path.write_text(text, encoding="utf-8")
# %%


# from pathlib import Path
#%%

#%%

#%%
#%%

#%%
def extract_annotations(self, dataset_paths:list) -> None:
    """Extract information from tree structure and annotations from csv files.
    """  # noqa: E501
    records = []
    # TREE_FILES = [
    # Path("00_epilepsy.txt"),
    # Path("01_no_epilepsy.txt"),
    # ]

    # Set these to your actual dataset roots (not the tree files):
    # DATASET_ROOTS = {
    #     "00_epilepsy": Path("../../data/v3.0.0/00_epilepsy"),
    #     "01_no_epilepsy": Path("../../data/v3.0.0/01_no_epilepsy"),
    # }
    filename = root.name+'.txt'
    output_path = root.parent/filename
    for root in dataset_paths:
        filename = root.name+'.txt'
        tree_file = root.parent/filename
        for root_name, subject, session, year, montage, filename in parse_tree_file(tree_file):
            records.append(
                {
                    "root": root_name,
                    "subject": subject,
                    "session": session,
                    "year": year,
                    "montage": montage,
                    "filename": filename,
                }
            )
    df = pd.DataFrame(records)
    # -------------------------
    # DataFrame: number of time series (count of .edf)
    # -------------------------
    edf_df = df[df["filename"].str.endswith(".edf")].copy()
    series_count_df = (
        edf_df.groupby(["subject", "session"])
            .size()
            .unstack()
    )
    # -------------------------
    # DataFrame: duration
    # -------------------------
    # Map edf -> csv and read duration from actual dataset path
    duration_entries = []
    annotated_entries = []
    annotated_bi_entries = []
    for _, row in edf_df.iterrows():
        root_name = row["root"]
        subject = row["subject"]
        session = row["session"]
        montage = row["montage"]
        edf_name = row["filename"]
        csv_name = edf_name.replace(".edf", ".csv")
        csv_bi_name = edf_name.replace(".edf", ".csv_bi")

        # Build real path to CSV in your dataset
        csv_path = DATASET_ROOTS[root_name] / subject / f"{session}_{row['year']}" / montage / csv_name
        csv_bi_path = DATASET_ROOTS[root_name] / subject / f"{session}_{row['year']}" / montage / csv_bi_name
        duration = read_duration_from_csv(csv_path)
        annot = read_annotations_from_csv(csv_path)
        annot[['root','subject','session','montage']] = [root_name,subject,session,montage]
        annot_bi = read_annotations_from_csv(csv_bi_path)
        annot_bi[['root','subject','session','montage']] = [root_name,subject,session,montage]
        # Extract t### from filename
        t_match = re.search(r"_t(\d+)\.edf$", edf_name)
        t_id = f"t{t_match.group(1)}" if t_match else edf_name

        duration_entries.append(
            {
                "subject": subject,
                "session": session,
                "t_id": t_id,
                "duration": duration,
            }
        )
        annotated_entries.append(annot)
        annotated_bi_entries.append(annot_bi)

    duration_df_raw = pd.DataFrame(duration_entries)
    duration_df = (
        duration_df_raw.groupby(["subject", "session"])
                    .apply(lambda g: dict(zip(g["t_id"], g["duration"])))
                    .unstack()
    )
    annotated_df = pd.concat(annotated_entries)
    annotated_bi_df = pd.concat(annotated_bi_entries)

#%%

#%%
# -------------------------
# DataFrame 1: session year
# -------------------------
year_df = (
    df.drop_duplicates(subset=["subject", "session", "year"])
      .pivot(index="subject", columns="session", values="year")
)
#%%
# -------------------------
# DataFrame 2: montage per session
# (if multiple montages per session, join them)
# -------------------------
montage_df = (
    df.drop_duplicates(subset=["subject", "session", "montage"])
      .groupby(["subject", "session"])["montage"]
      .apply(lambda s: sorted(set(s)))
      .apply(lambda s: ", ".join(s))
      .unstack()
)


#%%
# Results:
# - year_df
# - montage_df
# - series_count_df
# - duration_df

# %%
