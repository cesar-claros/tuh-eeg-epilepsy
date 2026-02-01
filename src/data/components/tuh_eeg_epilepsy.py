"""
Dataset class for the Temple University Hospital (TUH) EEG Epilipsy Corpus.
"""

# Authors: Claudio Cesar Claros Olivares <cesar.claros@outlook.com>
#
# License: BSD (3-clause)

from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple, Union
from joblib import Parallel, delayed
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from braindecode.datasets.base import BaseConcatDataset, RawDataset
from mne.preprocessing import ICA
from mne_icalabel import label_components
from braindecode.preprocessing import create_fixed_length_windows
import pandas as pd
import mne
import warnings

# Local imports
try:
    from . import utils
except ImportError:
    import utils  # type: ignore


class TUHEEGEpilepsy:
    """Temple University Hospital (TUH) EEG Epilepsy Corpus.

    Parameters
    ----------
    data_dir : str
        Parent directory of the dataset.
    version : str
        Version of the dataset.
    recording_ids : int | list[int] | None
        A (list of) int of recording id(s) to be read.
    add_annotations : bool
        If True, annotations will be read and added to the description.
    """
    
    EPILEPSY_PATH = '00_epilepsy/'
    NO_EPILEPSY_PATH = '01_no_epilepsy/'
    DOCS_PATH = 'DOCS/'
    DEPTH = 4  # subject_ID/session/montage/*.edf,*.csv,*.csv_bi

    def __init__(
        self,
        data_dir: str = '../../../data/',
        version: str = 'v3.0.0',
        recording_ids: Union[int, List[int], None] = None,
        add_annotations: bool = False,
    ):
        
        self.data_dir = data_dir
        self.version = version
        self.dataset_path = Path(self.data_dir) / self.version
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"No such directory: '{self.dataset_path}'")
            
        epilepsy_paths = [
            self.dataset_path / self.EPILEPSY_PATH,
            self.dataset_path / self.NO_EPILEPSY_PATH
        ]
        
        logger.info("Generating tree-style directory listing...")
        utils.generate_tree_file(epilepsy_paths[0], max_depth=self.DEPTH)
        utils.generate_tree_file(epilepsy_paths[1], max_depth=self.DEPTH)
        
        logger.info("Extracting info from directory listing ...")
        info_out = utils.extract_info_annotations(
            epilepsy_paths, 
            add_annotations=add_annotations, 
            recording_ids=recording_ids
        )
        
        if add_annotations:
            # Type ignore because we know the return type based on add_annotations=True
            self.descriptions: pd.DataFrame = info_out[0].reset_index(drop=True) # type: ignore
            self.annotated_df: pd.DataFrame = info_out[1] # type: ignore
            self.annotated_bi_df: pd.DataFrame = info_out[2] # type: ignore
        else:
            self.descriptions: pd.DataFrame = info_out.reset_index(drop=True) # type: ignore
            
        _, self.montages = self._get_metadata(add_montages=True)

        # limit to specified recording ids before doing slow stuff
        if recording_ids is not None:
            if not isinstance(recording_ids, Iterable):
                # Assume it is an integer specifying number of recordings
                recording_ids = range(recording_ids) # type: ignore
            self.descriptions = self.descriptions.iloc[recording_ids]

    def _get_metadata(self, add_montages: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
        metadata = utils.get_metadata(
            self.dataset_path / self.DOCS_PATH, 
            add_montages=add_montages
        )
        return metadata

    @staticmethod
    def _create_dataset(
        description: pd.Series,
        montages: dict,
        target_name: Optional[Union[str, Tuple[str, ...]]],
        preload: bool,
        rename_channels: bool,
        set_montage: bool,
        mode: str,
        pick_channels: Optional[List[str]],
        filter_freq: Optional[List[float]] = None,
    ) -> Optional[RawDataset]:
        
        if mode == 'ica':
            file_path = description["path_ica"]
            filter_freq = None
            channels = None
        else:
            file_path = description["path"]
            montage_name = description["montage"]
            channels = list(montages[montage_name]['channels'])

        # parse age and gender information from EDF header
        age, gender = utils.parse_age_and_gender_from_edf_header(file_path)
        
        # Read raw edf file
        try:
            raw = mne.io.read_raw_edf(
                file_path, 
                include=channels,
                preload=preload,
                infer_types=False, 
                verbose="error",
            )
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None

        # set measurement date from description if available
        meas_date = datetime(
            int(description['year']),
            raw.info["meas_date"].month,
            raw.info["meas_date"].day,
            tzinfo=timezone.utc
        )
        raw.set_meas_date(meas_date)
        
        if filter_freq is not None:
            raw = raw.copy().filter(l_freq=filter_freq[0], h_freq=filter_freq[1], verbose='ERROR')
            
        if rename_channels:
            TUHEEGEpilepsy._rename_channels(raw)
            
        if set_montage:
            TUHEEGEpilepsy._set_montage(raw)

        d = {
            "age": int(age),
            "gender": gender,
            "year": raw.info["meas_date"].year,
            "month": raw.info["meas_date"].month,
            "day": raw.info["meas_date"].day
        }

        additional_description = pd.Series(d)
        description = pd.concat([description, additional_description])
        
        if pick_channels is not None:
            raw = raw.pick(pick_channels)
            
        base_dataset = RawDataset(raw, description, target_name=target_name)
        return base_dataset
    
    def load_data(
        self,
        mode: str = 'raw',
        filter_freq: Optional[List[float]] = None,
        target_name: Optional[Union[str, Tuple[str, ...]]] = 'epilepsy',
        pick_channels: Optional[List[str]] = None,
        preload: bool = False,
        rename_channels: bool = False,
        set_montage: bool = False,
        n_jobs: int = 1,
    ) -> BaseConcatDataset:

        if set_montage:
            assert rename_channels, (
                "If set_montage is True, rename_channels must be True."
            )
        if mode == 'ica':
            assert not rename_channels and not set_montage, (
                "If mode is 'ica', rename_channels and set_montage must be False."
            )
            # Ensure ICA paths exist
            self.descriptions['path_ica'] = self.descriptions['path'].apply(
                lambda x: x.parent / x.name.replace('.edf', '_ica.edf')
            )
            assert self.descriptions['path_ica'].apply(lambda x: x.exists()).all(), (
                "Some ICA .edf files do not exist. Please compute ICA files first using compute_ica_labels()."
            )

        def safe_create_dataset(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*not in description. '__getitem__'"
                )
                return self._create_dataset(*args, **kwargs)

        n_edf_files = self.descriptions.shape[0]
        logger.info(f'Loading .edf files: {n_edf_files} files will be loaded')
        
        if n_jobs == 1:
            base_datasets = [
                safe_create_dataset(
                    row,
                    self.montages,
                    target_name,
                    preload,
                    rename_channels,
                    set_montage,
                    mode,
                    pick_channels,
                    filter_freq,
                )
                for _, row in tqdm(self.descriptions.iterrows(), desc="Loading .edf files", total=n_edf_files)
            ]
        else:
            base_datasets = Parallel(n_jobs)(
                delayed(safe_create_dataset)(
                    row,
                    self.montages,
                    target_name,
                    preload,
                    rename_channels,
                    set_montage,
                    mode,
                    pick_channels,
                    filter_freq,
                )
                for _, row in tqdm(self.descriptions.iterrows(), desc="Loading .edf files", total=n_edf_files)
            )
            
        # Filter .edf files which failed to load
        valid_datasets = [base for base in base_datasets if isinstance(base, RawDataset)]
        
        if len(valid_datasets) > 0:
            logger.info(f'Number of .edf files loaded: {len(valid_datasets)}')
            return BaseConcatDataset(valid_datasets)
        else:
            raise ValueError("No dataset was created. Please check the parameters provided.")

    @staticmethod
    def _generate_ICA_labels(
        file_path: Path,
        seed: int = 45,
    ) -> None:
        """
        Computes ICA, labels components, and saves the cleaned EDF file.
        Only runs if output files do not already exist.
        """
        try:
            # Define output paths
            ica_path = file_path.parent / file_path.name.replace('.edf', '-ica.fif')
            labels_path = file_path.parent / file_path.name.replace('.edf', '-ica_labels.csv')
            output_ica_file = file_path.parent / file_path.name.replace('.edf', '_ica.edf')

            # Short-circuit if final output exists
            if output_ica_file.exists() and labels_path.exists():
                logger.info(f"Skipping {file_path.name}, outputs already exist.")
                return

            # Read Raw
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            filt_raw = raw.copy().set_eeg_reference("average")
            # Basic filtering for ICA
            filt_raw.filter(l_freq=1.0, h_freq=100.0, verbose='error')
            
            channels = filt_raw.ch_names
            
            # ICA Computation/Loading
            if ica_path.exists():
                ica = mne.preprocessing.read_ica(ica_path)
            else:
                ica = ICA(
                    n_components=0.999999,
                    max_iter="auto",
                    method="infomax",
                    random_state=seed,
                    fit_params=dict(extended=True),
                )
                ica.fit(filt_raw)
                ica.save(ica_path)

            # Labeling
            if labels_path.exists():
                labels_df = pd.read_csv(labels_path, index_col=0)
            else:
                try:
                    ic_labels = label_components(filt_raw, ica, method="iclabel")
                    labels_df = pd.DataFrame(ic_labels)
                    labels_df.to_csv(labels_path)
                except Exception as e:
                    logger.error(f"ICA Labeling failed for {file_path}: {e}")
                    return

            # Create new Raw with ICA components
            ref_comps = ica.get_sources(filt_raw)
            for j, c in enumerate(ref_comps.ch_names):
                label_name = labels_df["labels"][j].split(" ")[0]
                ref_comps.rename_channels({c: f'{label_name}-{c}'})
            
            filt_raw.add_channels([ref_comps])
            filt_raw.drop_channels(channels)
            
            if not output_ica_file.exists():
                mne.export.export_raw(output_ica_file, filt_raw, fmt='edf', physical_range='auto', overwrite=False)
                
        except Exception as e:
            logger.error(f"Error processing ICA for {file_path}: {e}")

    def compute_ica_labels(
        self,
        n_jobs: int = 1,
    ) -> None:
        """
        Iterates over all files in the dataset description and computes ICA labels and cleaned files.
        This is a side-effect heavy operation that writes files to disk.
        """
        logger.info(f"Starting ICA computation for {len(self.descriptions)} files...")
        
        paths = self.descriptions['path'].tolist()
        
        if n_jobs == 1:
            for path in tqdm(paths, desc="Computing ICA Labels"):
                self._generate_ICA_labels(path)
        else:
            Parallel(n_jobs=n_jobs)(
                delayed(self._generate_ICA_labels)(path)
                for path in tqdm(paths, desc="Computing ICA Labels")
            )
        logger.info("ICA computation completed.")

    @staticmethod
    def _rename_channels(raw: mne.io.BaseRaw) -> None:
        """
        Renames the EEG channels using mne conventions and sets their type to 'eeg'.
        """
        mapping_strip = {
            c: c.replace("-REF", "").replace("-LE", "").replace("EEG ", "")
            for c in raw.ch_names
        }
        raw.rename_channels(mapping_strip)

        montage1020 = mne.channels.make_standard_montage("standard_1020")
        mapping_eeg_names = {
            c.upper(): c for c in montage1020.ch_names if c.upper() in raw.ch_names
        }

        non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
        if non_eeg_names:
            non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
            mapping_non_eeg_types = {
                c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
            }
            if mapping_non_eeg_types:
                raw.set_channel_types(mapping_non_eeg_types) # type: ignore

        if mapping_eeg_names:
            raw.set_channel_types(
                {c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore" # type: ignore
            )
            raw.rename_channels(mapping_eeg_names)

    @staticmethod
    def _set_montage(raw: mne.io.BaseRaw) -> None:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")


if __name__ == "__main__":
    # Example usage
    tuh = TUHEEGEpilepsy(
        recording_ids=range(30),
        add_annotations=True,
    )
    
    # Uncomment to compute ICA
    # tuh.compute_ica_labels(n_jobs=4)
    
    data = tuh.load_data(
        mode='raw',
        target_name=('epilepsy', 'age', 'gender'),
        preload=True,
        rename_channels=False,
        set_montage=False,
        n_jobs=1,
    )
    
    tuh_windows = create_fixed_length_windows(
        data,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=250,
        window_stride_samples=250,
        drop_last_window=True,
    )
