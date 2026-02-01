#%%
"""
Dataset class for the Temple University Hospital (TUH) EEG Epilipsy Corpus.
"""

# Authors: Claudio Cesar Claros Olivares <cesar.claros@outlook.com>
#
# License: BSD (3-clause)

from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterable, Tuple
# from unittest import mock
from joblib import Parallel, delayed
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from braindecode.datasets.base import BaseConcatDataset, RawDataset
from mne.preprocessing import ICA
from mne_icalabel import label_components
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.util import set_random_seeds
# import glob
# import os
# import re
import warnings
# import numpy as np
import mne
# import numpy as np
import pandas as pd
import utils


#%%
class TUHEEGEpilepsy():
    """Temple University Hospital (TUH) EEG Epilepsy Corpus.

    Parameters
    ----------
    path : str
        Parent directory of the dataset.
    recording_ids : list(int) | int
        A (list of) int of recording id(s) to be read (order matters and will
        overwrite default chronological order, e.g. if recording_ids=[1,0],
        then the first recording returned by this class will be chronologically
        later then the second recording. Provide recording_ids in ascending
        order to preserve chronological order.).
    target_name : str
        Can be 'gender', or 'age'.
    preload : bool
        If True, preload the data of the Raw objects.
    add_physician_reports : bool
        If True, the physician reports will be read from disk and added to the
        description.
    rename_channels : bool
        If True, rename the EEG channels to the standard 10-05 system.
    set_montage : bool
        If True, set the montage to the standard 10-05 system.
    n_jobs : int
        Number of jobs to be used to read files in parallel.
    """

    def __init__(
        self,
        data_dir: str = '../../../data/',
        version:str = 'v3.0.0',
        recording_ids: int|list[int]| None = None,
        add_annotations: bool = False,

    ):
        
        # create an index of all files and gather easily accessible info
        # without actually touching the files
        self.compute_ICA_labels = False
        self.data_dir = data_dir
        self.version = version
        EPILEPSY_PATH = '00_epilepsy/'
        NO_EPILEPSY_PATH = '01_no_epilepsy/'
        
        DEPTH = 4 # subject_ID/session/montage/*.edf,*.csv,*.csv_bi
        self.dataset_path = Path(self.data_dir+self.version)
        if self.dataset_path.exists():
            epilepsy_paths = [self.dataset_path/EPILEPSY_PATH, self.dataset_path/NO_EPILEPSY_PATH]
            logger.info("Generating tree-style directory listing...")
            utils.generate_tree_file(epilepsy_paths[0], max_depth=DEPTH)
            utils.generate_tree_file(epilepsy_paths[1], max_depth=DEPTH)
            logger.info("Extracting info from directory listing ...")
            info_out = utils.extract_info_annotations(epilepsy_paths, add_annotations=add_annotations, recording_ids=recording_ids)    
            if add_annotations:
                self.descriptions, self.annotated_df, self.annotated_bi_df = info_out
            else:
                self.descriptions = info_out
        else:
            raise FileNotFoundError(f"No such directory: '{self.dataset_path}'")
        self.descriptions = self.descriptions.reset_index(drop=True)
        _, self.montages = self.get_metadata(add_montages=True)
        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            if not isinstance(recording_ids, Iterable):
                # Assume it is an integer specifying number
                # of recordings to load
                recording_ids = range(recording_ids)
            self.descriptions = self.descriptions.iloc[recording_ids]

        
        # workaround to ensure warnings are suppressed when running in parallel
    
    def get_metadata(
        self,
        add_montages: bool = False,
    ) -> pd.DataFrame:
        DOCS_PATH = 'DOCS/'
        metadata = utils.get_metadata(self.dataset_path/DOCS_PATH, add_montages=add_montages)
        return metadata

    @staticmethod
    def _create_dataset(
        description: pd.DataFrame,
        montages: dict,
        target_name: str | tuple[str, ...] | None,
        preload: bool,
        rename_channels: bool,
        set_montage: bool,
        mode: str,
        pick_channels: list[str]|None,
        filter_freq: list[float]|None = None,
        compute_ICA_labels: bool = False,
    ):
        if mode == 'ica':
            file_path = description.loc["path_ica"]
            filter_freq = None
            channels = None
        else:
            file_path = description.loc["path"]
            montage_name = description.loc["montage"]
            channels = list(montages[montage_name]['channels'])
        # parse age and gender information from EDF header
        age, gender = utils.parse_age_and_gender_from_edf_header(file_path)
        # Read raw edf file
        raw = mne.io.read_raw_edf(
                                file_path, 
                                include = channels,
                                preload = preload,
                                infer_types = False, 
                                verbose = "error",
        )
        # set measurement date from description if available
        meas_date = datetime(
                            int(description['year']),
                            raw.info["meas_date"].month,
                            raw.info["meas_date"].day,
                            tzinfo=timezone.utc
                            )
        raw.set_meas_date(meas_date)
        if filter_freq is not None:
            LOW_FREQ = filter_freq[0]
            HIGH_FREQ = filter_freq[1]
            raw = raw.copy().filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose='ERROR')
        if rename_channels:
            TUHEEGEpilepsy._rename_channels(raw)
        if set_montage:
            TUHEEGEpilepsy._set_montage(raw)
        if compute_ICA_labels:
            TUHEEGEpilepsy._generate_ICA_labels(
                raw,
                file_path,
            )
            return None

        

        d = {
            "age": int(age),
            "gender": gender,
        }
        d["year"] = raw.info["meas_date"].year
        d["month"] = raw.info["meas_date"].month
        d["day"] = raw.info["meas_date"].day

        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        if pick_channels is not None:
            raw = raw.pick(pick_channels)
        base_dataset = RawDataset(raw, description, target_name=target_name)
        return base_dataset
    
    def load_data(
        self,
        # split : str = 'train',
        mode :str = 'raw',
        filter_freq: list[float]|None = None,
        target_name: str | tuple[str, ...] | None = 'epilepsy',
        pick_channels:list[str]|None = None,
        preload: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
        ):

        if set_montage:
            assert rename_channels, (
                "If set_montage is True, rename_channels must be True."
            )
        if mode == 'ica':
            assert not rename_channels and not set_montage, (
                "If mode is 'ica', rename_channels and set_montage must be False."
            )
            self.descriptions['path_ica'] = self.descriptions['path'].apply(
                                            lambda x: x.parent/x.name.replace('.edf','_ica.edf'))
            assert self.descriptions['path_ica'].apply(lambda x: x.exists()).all(), (
                "Some ICA .edf files do not exist. Please compute ICA files first."
            )

        def create_dataset(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*not in description. '__getitem__'"
                )
                return self._create_dataset(*args, **kwargs)

        # this is the second loop (slow)
        # create datasets gathering more info about the files touching them
        # reading the raws and potentially preloading the data
        # disable joblib for tests. mocking seems to fail otherwise
        n_edf_files = self.descriptions.shape[0]
        logger.info(f'Loading .edf files: {n_edf_files} files will be loaded')
        if n_jobs == 1:
            base_datasets = [
                create_dataset(
                    row,
                    self.montages,
                    target_name,
                    preload,
                    rename_channels,
                    set_montage,
                    mode,
                    pick_channels,
                    filter_freq,
                    self.compute_ICA_labels,
                )
                for _,row in tqdm(self.descriptions.iterrows(), desc="Loading .edf files", total=n_edf_files)
            ]
        else:
            base_datasets = Parallel(n_jobs)(
                delayed(create_dataset)(
                    row,
                    target_name,
                    preload,
                    rename_channels,
                    set_montage,
                    mode,
                    pick_channels,
                    filter_freq,
                    self.compute_ICA_labels,
                )
                for _,row in tqdm(self.descriptions.iterrows(), desc="Loading .edf files", total=n_edf_files)
            )
        # Filter .edf files for which no selected channels was an empty set
        base_datasets = [ base for base in base_datasets if isinstance(base,RawDataset) ]
        if len(base_datasets) > 0:
            logger.info(f'Number of .edf files loaded: {len(base_datasets)}')
            return BaseConcatDataset(base_datasets)
        elif self.compute_ICA_labels:
            return None
        else:
            raise ValueError("No dataset was created. Please check the parameters provided.")
        # super().__init__(base_datasets)
    
    @staticmethod
    def _generate_ICA_labels(
        raw: mne.io.BaseRaw,
        file_path: Path,
        seed: int = 45,
    ):
        ICA_SEED = seed
        filt_raw = raw.set_eeg_reference("average")
        channels = filt_raw.ch_names
        logger.info(f'Channels used for ICA: {channels}')
        ica_path = file_path.parent/file_path.name.replace('.edf','-ica.fif')
        if ica_path.exists():
            logger.info(f'Reading ICA solution from: {ica_path}')
            ica = mne.preprocessing.read_ica(ica_path)
        else:
            logger.info('Computing ICA solution')
            ica = ICA(
                n_components=0.999999,
                max_iter="auto",
                method="infomax",
                random_state=ICA_SEED,
                fit_params=dict(extended=True),
            )
            ica.fit(filt_raw)
            ica.save(ica_path)
                    
        labels_path = file_path.parent/file_path.name.replace('.edf','-ica_labels.csv')
        if labels_path.exists():
            logger.info(f'Reading ICA LAbels from: {labels_path}')
            labels_df = pd.read_csv(labels_path,index_col=0)
        else:
            logger.info('Computing ICA labels')
            ic_labels = label_components(filt_raw, ica, method="iclabel")
            labels_df = pd.DataFrame(ic_labels)
            labels_df.to_csv(labels_path)
        logger.info(f'Number of components labeled "brain" : {(labels_df["labels"]=="brain").sum()}')
        # Create a new Raw object with the ICA components as channels
        ref_comps = ica.get_sources(filt_raw)
        for j, c in enumerate(ref_comps.ch_names):  # they need to have REF_ prefix to be recognised
            label_name = labels_df["labels"][j].split(" ")[0]  # get main label
            ref_comps.rename_channels({c: f'{label_name}-{c}'})
        filt_raw.add_channels([ref_comps])
        filt_raw = filt_raw.drop_channels(channels)
        # Export the Raw object to an EDF file
        output_ica_file = file_path.parent/file_path.name.replace('.edf','_ica.edf')
        if not output_ica_file.exists():
            logger.info(f'Saving ICA components to EDF file: {output_ica_file}')
            mne.export.export_raw(output_ica_file, filt_raw, fmt='edf', physical_range='auto', overwrite=False)
        else:
            logger.info(f'ICA components already saved to EDF file: {output_ica_file}')


    def get_ICA_labels(
        self,
        n_jobs: int = 1,
        filter_freq: list[float]|None = [1.0,100.0],
        ):
        self.compute_ICA_labels = True
        self.load_data(
                        mode  = 'raw',
                        filter_freq = filter_freq,
                        target_name = None,
                        pick_channels = None,
                        preload = True,
                        rename_channels = True,
                        set_montage = True,
                        n_jobs = n_jobs,
                        )
        self.compute_ICA_labels = False
        return

    @staticmethod
    def _rename_channels(raw):
        """
        Renames the EEG channels using mne conventions and sets their type to 'eeg'.

        See https://isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes/
        """
        # remove ref suffix and prefix:
        # TODO: replace with removesuffix and removeprefix when 3.8 is dropped
        mapping_strip = {
            c: c.replace("-REF", "").replace("-LE", "").replace("EEG ", "")
            for c in raw.ch_names
        }
        raw.rename_channels(mapping_strip)

        montage1020 = mne.channels.make_standard_montage("standard_1020")
        mapping_eeg_names = {
            c.upper(): c for c in montage1020.ch_names if c.upper() in raw.ch_names
        }

        # Set channels whose type could not be inferred (defaulted to "eeg") to "misc":
        non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
        if non_eeg_names:
            non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
            mapping_non_eeg_types = {
                c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
            }
            if mapping_non_eeg_types:
                raw.set_channel_types(mapping_non_eeg_types)
        # print(mapping_eeg_names)
        if mapping_eeg_names:
            # Set 1005 channels type to "eeg":
            raw.set_channel_types(
                {c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore"
            )
            # Fix capitalized EEG channel names:
            raw.rename_channels(mapping_eeg_names)

    @staticmethod
    def _set_montage(raw):
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

    

# %%
tuh = TUHEEGEpilepsy(
                    recording_ids=range(30), 
                    # preload=True, 
                    add_annotations=True, 
                    # add_montages=True,  
                    # target_name=('epilepsy','age','gender'),
                    # rename_channels=True,
                    # set_montage=True,
                    # compute_iclabels=True,
                    # add_ica_components=True,
                    # keep_channels=['brain','other']
                    )
#%%
tuh.get_ICA_labels(
                    n_jobs=1,
                    filter_freq = [1.0,100.0],
                    )
#%%
data = tuh.load_data(
                    mode='raw',
                    target_name=('epilepsy','age','gender'),
                    preload=True,
                    rename_channels=False,
                    set_montage=False,
                    n_jobs=1,
                    )
#%%
tuh_windows = create_fixed_length_windows(
    tuh,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=250,
    window_stride_samples=250,
    drop_last_window=True,
)
# # %%
# tuh.montages['01_tcp_ar']['channels']
# %%
# %%

