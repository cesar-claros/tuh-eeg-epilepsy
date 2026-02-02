#%%
"""
Dataset class for the Temple University Hospital (TUH) EEG Epilipsy Corpus.
"""

# Authors: Claudio Cesar Claros Olivares <cesar.claros@outlook.com>
#
# License: BSD (3-clause)

from __future__ import annotations
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple, Union, Dict
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
import torch
import numpy as np

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
        # New args for balanced windowing
        window_len_s: Optional[float] = None,
        overlap_pct: float = 0.0,
        balance_per_subject: bool = False,
        include_seizures: bool = True,
        fix_length_mode: Optional[str] = 'resample', # 'resample', 'pad', or None
        shuffle_windows: bool = False,
        seed: int = 42,
        splits: Optional[Dict[str, float]] = None,
    ) -> Union[BaseConcatDataset, Tuple[torch.Tensor, torch.Tensor, pd.DataFrame], Dict[str, Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]]]:
        
        # If window_len_s is provided, use the optimized balanced loader
        if window_len_s is not None:
             return self._load_balanced_windows(
                window_len_s=window_len_s,
                overlap_pct=overlap_pct,
                balance_per_subject=balance_per_subject,
                include_seizures=include_seizures,
                fix_length_mode=fix_length_mode,
                shuffle_windows=shuffle_windows,
                seed=seed,
                splits=splits,
                mode=mode,
                filter_freq=filter_freq,
                target_name=target_name,
                pick_channels=pick_channels,
                rename_channels=rename_channels,
                set_montage=set_montage,
                n_jobs=n_jobs,
             )

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

    def _load_balanced_windows(
        self,
        window_len_s: float,
        overlap_pct: float,
        balance_per_subject: bool,
        include_seizures: bool,
        fix_length_mode: Optional[str],
        shuffle_windows: bool,
        seed: int,
        splits: Optional[Dict[str, float]],
        mode: str,
        filter_freq: Optional[List[float]],
        target_name: Optional[Union[str, Tuple[str, ...]]],
        pick_channels: Optional[List[str]],
        rename_channels: bool,
        set_montage: bool,
        n_jobs: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, pd.DataFrame], Dict[str, Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]]]:
        
        # RNG
        rng = np.random.RandomState(seed)

        # 1. Filter Descriptions
        df = self.descriptions.copy()
        if not include_seizures:
            # Assumes 'n_seizure' col exists and > 0 means it contains seizure
            # If the user meant "exclude segments with seizures", that's harder without detailed annotations
            # Assuming here: exclude FILES that have seizures
             df = df[df['n_seizure'] == 0]

        if df.empty:
            raise ValueError("No data available after filtering.")
        
        # 2. Calculate available windows per subject
        stride_s = window_len_s * (1 - overlap_pct)
        if stride_s <= 0:
            raise ValueError("Overlap must be < 1.0")

        # Approx windows per file
        df['n_windows'] = np.maximum(0, np.floor((df['duration'] - window_len_s) / stride_s) + 1).astype(int)
        
        # Windows per subject
        subject_window_counts = df.groupby('subject')['n_windows'].sum()
        
        if balance_per_subject:
            limit_per_subject = int(subject_window_counts.min())
            logger.info(f"Balancing: Limiting to {limit_per_subject} windows per subject.")
            if limit_per_subject == 0:
                 raise ValueError("Balancing requested but at least one subject has 0 valid windows.")
        else:
            limit_per_subject = None # Use all
            logger.info("Using all available windows (unbalanced).")

        # 3. Generate Window List
        window_meta = []
        
        # Group by subject to enforce limits
        # To handle shuffle correctly with balanced limits, we must collect ALL possibilities first for a subject
        
        for subject, group in df.groupby('subject'):
            # Sort for deterministic base order
            group = group.sort_values(['year', 'month', 'day', 'time'] if {'year','month','day','time'}.issubset(group.columns) else ['path'])
            
            subject_windows = []
            
            for _, row in group.iterrows():
                age, gender = utils.parse_age_and_gender_from_edf_header(row['path'])
                n_possible = row['n_windows']
                if n_possible <= 0:
                    continue
                
                # Calculate start times for this file
                for idx in range(n_possible):
                    start_t = idx * stride_s
                    end_t = start_t + window_len_s
                    
                    subject_windows.append({
                        'subject': subject,
                        'path': row['path'],
                        'start': start_t,
                        'end': end_t,
                        't_id': row.get('t_id', 'unknown'),
                        'epilepsy': row.get('epilepsy', False),
                        # Store other useful metadata from row if needed
                        'age': age,
                        'gender': gender,
                        'description_row': row # Keep ref for loading
                    })
                    
            # After collecting all windows for this subject:
            if shuffle_windows:
                rng.shuffle(subject_windows)
                
            # Apply Limit (if balance is ON, limit is min_windows, otherwise None means take all)
            if limit_per_subject is not None:
                subject_windows = subject_windows[:limit_per_subject]
            
            # Assign final index
            for i, w in enumerate(subject_windows):
                w['window_idx_within_subject'] = i
                
            window_meta.extend(subject_windows)

        window_df = pd.DataFrame(window_meta)
        logger.info(f"Found {len(window_df['subject'].unique())} subjects.")
        logger.info(f"Generated {len(window_df)} windows plan.")

        # 3.5 Handle Data Splitting (Assign splits to rows)
        # Default split is 'all' if no splits provided
        split_map = {} # subject -> split_name
        
        unique_subjects = window_df['subject'].unique()
        
        if splits is not None:
            # Check sum
            total_ratio = sum(splits.values())
            if not np.isclose(total_ratio, 1.0):
                logger.warning(f"Split ratios sum to {total_ratio}, not 1.0. This might leave some subjects unused or cause overlap issues if > 1.")
            
            # Shuffle unique subjects for random split
            # Use same seed for consistency
            rng.shuffle(unique_subjects)
            
            n_sub = len(unique_subjects)
            current_idx = 0
            
            for split_name, ratio in splits.items():
                n_split = int(np.floor(ratio * n_sub))
                # Assign
                split_subs = unique_subjects[current_idx : current_idx + n_split]
                for s in split_subs:
                    split_map[s] = split_name
                
                current_idx += n_split
            
            # Assign remaining (rounding errors) to last split or warning?
            # Or just 'test'? Let's assign strict based on ratios. 
            # If sum < 1, some subjects map to None (unused).
            # If sum = 1, last few might be skipped due to floor.
            # Let's add remaining to the last defined split to be safe, or just leave them unused.
            # Common practice: Add remainder to last split.
            if current_idx < n_sub and np.isclose(total_ratio, 1.0):
                 remainder = unique_subjects[current_idx:]
                 last_split = list(splits.keys())[-1]
                 for s in remainder:
                     split_map[s] = last_split
                     
            logger.info("Subject Split Assignment Complete.")

        # 4. Load Data
        # Helper for parallel loading
        def load_window(row_meta):
            desc_row = row_meta['description_row']
            # Re-use _create_dataset logic but we only need the raw array
            # We can't use _create_dataset easily because it loads the WHOLE file.
            # We want to load just crop.
            
            # Manually load logic
             
            if mode == 'ica':
                file_path = desc_row["path_ica"]
                ch_names = None
            else:
                file_path = desc_row["path"]
                montage_name = desc_row["montage"]
                ch_names = list(self.montages[montage_name]['channels'])

            try:
                # MNE read_raw_edf supports 'preload' but we want specific times.
                # read_raw_edf doesn't support tmin/tmax load optimization directly FOR EDF in all versions efficiently,
                # but cropping after loading header might be better than full load.
                # However, for 5 min windows, full load might be huge.
                # Efficient way: read_raw_edf(preload=False), then crop, then load_data()
                
                raw = mne.io.read_raw_edf(
                    file_path, 
                    include=ch_names,
                    preload=False,
                    infer_types=False, 
                    verbose="error",
                )
                
                 # Basic Filter (should be done before cropping ideally to avoid edge artifacts, or crop with margin)
                 # If we filter AFTER cropping, we need margin. 
                 # Given user request for "load only necessary", we accept edge artifacts or assume data is clean/prefiltered,
                 # OR we load with margin. Let's load exact for now to match request strictness, or suggest margin.
                 # Standard practice: Load, (Filter), Crop. 
                 # To save memory: Crop then Filter (bad edges).
                 # To do it right without full load: read_raw_edf allows cropping.
                
                # Filter before crop if we want perfect filter, but that requires full load.
                # Compromise: Crop with margin? 
                # For this implementation, I will just crop exact window.
                
                raw.crop(tmin=row_meta['start'], tmax=row_meta['end'], include_tmax=False)
                raw.load_data()
                
                # Apply processing matching _create_dataset
                if filter_freq is not None:
                    raw.filter(l_freq=filter_freq[0], h_freq=filter_freq[1], verbose='ERROR')
            
                if rename_channels:
                    TUHEEGEpilepsy._rename_channels(raw)
            
                if set_montage:
                    TUHEEGEpilepsy._set_montage(raw)
                    
                if pick_channels:
                    raw.pick(pick_channels)
                    
                # To Epoch/Tensor
                # Resample? Not requested, but raw.get_data() returns numpy
                data = raw.get_data() # (Channels, Time)
                # Ensure fixed length (Float rounding issues might give +/- 1 sample)
                # Check expected samples
                sfreq = raw.info['sfreq']
                logger.info(f"Loading window: {row_meta['path'].name}, start: {row_meta['start']}, end: {row_meta['end']}, sfreq: {sfreq}")
                expected_samples = int(window_len_s * sfreq)
                
                if data.shape[1] > expected_samples:
                    # If strictly cropping, we might do it here, but wait for global check
                    # data = data[:, :expected_samples]
                    pass
                elif data.shape[1] < expected_samples:
                    # If padding is allowed, we keep it. If strict mode (None), we might drop?
                    # For now keep it and let post-process handle it
                    pass
                    
                # Return data and channel names for harmonization
                # Note: raw.ch_names might have been renamed or picked.
                return data, sfreq, raw.ch_names

            except Exception as e:
                logger.error(f"Error loading window {row_meta}: {e}")
                return None

        # Execute Load
        if n_jobs == 1:
            results = [load_window(r) for _, r in tqdm(window_df.iterrows(), total=len(window_df), desc="Loading Windows")]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(load_window)(r) 
                for _, r in tqdm(window_df.iterrows(), total=len(window_df), desc="Loading Windows")
            )
            
        # Filter Nones
        valid_indices = [i for i, r in enumerate(results) if r is not None]
        # Unzip data, sfreqs, and channel names
        valid_data_raw = [results[i][0] for i in valid_indices] # List[np.ndarray (Ch, Time)]
        valid_sfreqs = [results[i][1] for i in valid_indices]   # List[float]
        valid_channels = [results[i][2] for i in valid_indices] # List[List[str]]
        valid_meta = window_df.iloc[valid_indices].reset_index(drop=True)
        
        if not valid_data_raw:
             raise ValueError("No valid windows loaded.")

        # 5a. Harmonize Channels
        # Find intersection of all channel lists
        if not valid_channels:
            raise ValueError("No channel information available.")
   
        common_channels = set(valid_channels[0])
        for ch_list in valid_channels[1:]:
            common_channels.intersection_update(ch_list)
            
        if not common_channels:
             raise ValueError("No common channels found across the loaded set.")
             
        # Sort for deterministic order (e.g. alphabetical)
        # Or standard_1020 if possible, but alphabetical is safe default
        common_channels = sorted(list(common_channels))
        logger.info(f"Harmonizing to {len(common_channels)} common channels: {common_channels}")
        
        # Reorder/Submit data
        # data shape: (Channels, Time). We need to select rows corresponding to common_channels.
        for i in range(len(valid_data_raw)):
            d = valid_data_raw[i]
            ch_names = valid_channels[i]
            
            # Map channel name to index
            ch_idx_map = {name: idx for idx, name in enumerate(ch_names)}
            
            # Find indices for common channels in correct order
            try:
                indices = [ch_idx_map[name] for name in common_channels]
            except KeyError as e:
                # Should not happen since we took intersection, but good for safety
                logger.error(f"Logic error: Channel {e} missing in file despite intersection check.")
                valid_data_raw[i] = None # Drop this one
                continue
                
            # Slice
            valid_data_raw[i] = d[indices, :]
            
        # Filter dropped items from harmonization
        # (Technically shouldn't be any unless logic error above)
        valid_indices_2 = [i for i, d in enumerate(valid_data_raw) if d is not None]
        valid_data_raw = [valid_data_raw[i] for i in valid_indices_2]
        valid_sfreqs = [valid_sfreqs[i] for i in valid_indices_2]
        valid_meta = valid_meta.iloc[valid_indices_2].reset_index(drop=True)

        # 5b. Handle variable sampling rates / lengths
        # Determine target parameters
        unique_sfreqs = np.unique(valid_sfreqs)
        
        if len(unique_sfreqs) > 1:
            logger.warning(f"Found multiple sampling frequencies: {unique_sfreqs}")
            
            if fix_length_mode == 'resample':
                target_sfreq = float(np.min(unique_sfreqs))
                logger.info(f"Resampling all windows to {target_sfreq} Hz")
                
                # Resample items that need it
                for i in range(len(valid_data_raw)):
                    d = valid_data_raw[i]
                    sf = valid_sfreqs[i]
                    
                    if not np.isclose(sf, target_sfreq):
                        # Calculate resampling factor
                        # up/down = sf_target / sf_current
                        # mne.filter.resample operates on last axis by default (Time)
                        # data shape (Channels, Time)
                        resampled_d = mne.filter.resample(d, up=target_sfreq, down=sf, axis=-1)
                        valid_data_raw[i] = resampled_d
                        valid_sfreqs[i] = target_sfreq
                        
            elif fix_length_mode == 'pad':
               # Padding doesn't fix sampling rate, it just fixes shape.
               # If sampling rates are different, padding makes them same size but different time duration.
               # The user request said: "If pad is chosen, then we will pad the signals with zeros to the largest number of samples."
               # This implies we accept different time durations in the same tensor (Time dimension), 
               # but usually tensors represent fixed time.
               # If SF is different, samples=100 @ 10Hz = 10s, samples=100 @ 100Hz = 1s.
               # We just follow instructions to pad to largest SAMPLE count.
               pass 
            
            else: 
                raise ValueError(f"Function found multiple sampling frequencies {unique_sfreqs} but fix_length_mode is {fix_length_mode}.")

        # Now ensure sample lengths are identical
        lengths = [d.shape[1] for d in valid_data_raw]
        max_len =  max(lengths)
        min_len = min(lengths)
        
        target_len = max_len if fix_length_mode == 'pad' else int(window_len_s * min(valid_sfreqs)) # Approx target
        
        # Refine target_len:
        # If 'pad', target is max_len.
        # If 'resample', target should ideally be window_len_s * target_sfreq
        if fix_length_mode == 'resample':
             # We forced sfreq to min. Expected samples = window_len * min_sfreq
             target_sfreq = min(valid_sfreqs)
             target_len = int(window_len_s * target_sfreq)
        
        final_data_list = []
        
        for i, d in enumerate(valid_data_raw):
            current_len = d.shape[1]
            diff = target_len - current_len
            
            if diff == 0:
                final_data_list.append(d)
            elif diff > 0:
                # Need to pad
                if fix_length_mode in ['pad', 'resample']: 
                    # Pad right with zeros
                    # shape (Channels, Time)
                    padding = np.zeros((d.shape[0], diff), dtype=d.dtype)
                    padded_d = np.concatenate([d, padding], axis=1)
                    final_data_list.append(padded_d)
                else:
                    # Mismatch and no fix mode
                     # If diff is small (rounding error), maybe we allow it?
                     # For now, strict.
                     logger.warning(f"Window {i} has {current_len} samples, expected {target_len}. Skipping.")
                     final_data_list.append(None) 
            elif diff < 0:
                # Need to trim
                # Usually happens in resample mode due to rounding or if data was slightly longer
                trimmed_d = d[:, :target_len]
                final_data_list.append(trimmed_d)

        # Filter dropped items
        # Re-sync metadata
        valid_final_indices = [i for i, d in enumerate(final_data_list) if d is not None]
        final_valid_data = [final_data_list[i] for i in valid_final_indices]
        valid_meta = valid_meta.iloc[valid_final_indices].reset_index(drop=True)

        if not final_valid_data:
            raise ValueError("All windows filtered out due to length mismatch.")

        # Stack Data
        # formatted as (Batch, Channels, Time) -> Transpose to (Batch, Time, Channels) if user asked (windows*samples*channels)?
        # User asked: (windows*samples*channels)
        logger.info(f"Loaded {len(final_valid_data)} valid windows.")
        tensor_data = np.stack(final_valid_data) # (Batch, Channels, Time)
        tensor_data = np.transpose(tensor_data, (0, 2, 1)) # (Batch, Time, Channels)
        tensor_data = torch.from_numpy(tensor_data).float()
        
        # Stack Targets
        # "Target can be a tuple ... epilepsy or not, but it can also contain age and gender"
        # We'll support the basic epilepsy target, plus age/gender if in target_name
        
        target_list = []
        # Support tuple target_name
        if isinstance(target_name, str):
            target_cols = [target_name]
        elif isinstance(target_name, tuple):
             target_cols = list(target_name)
        else:
             target_cols = ['epilepsy'] # Default
             
        for _, row in valid_meta.iterrows():
            vals = []
            for col in target_cols:
                val = row.get(col, -1)
                # Simple encoding
                if col == 'gender':
                    val = 0 if val == 'M' else (1 if val == 'F' else 2)
                elif col == 'epilepsy':
                    val = 1 if val else 0
                vals.append(val)
            target_list.append(vals)
            
        tensor_targets = torch.tensor(target_list)
        
        # Clean metadata for return (drop the heavy object)
        if 'description_row' in valid_meta.columns:
            valid_meta = valid_meta.drop(columns=['description_row'])
            
        # 6. Organize Output based on Splits
        if splits is None:
             # Return just the valid tuple
             return tensor_data, tensor_targets, valid_meta
        else:
             # Partition the tensors and metadata back into dictionary
             output_dict = {}
             
             # We need to map the valid_meta rows back to their subjects
             # Note: valid_meta is aligned with tensor_data (indices match)
             
             # Create a mask for each split
             
             distinct_split_names = set(splits.keys())
             
             for split_name in distinct_split_names:
                  # Find subjects belonging to this split
                  # Helper: get indices in valid_meta
                  # Optimization: Add 'split' column to window_df earlier?
                  # Yes, but we did it on subjects. 
                  # Let's just iterate valid_meta
                  
                  # Which subjects are in this split?
                  subjects_in_split = {s for s, param in split_map.items() if param == split_name}
                  
                  # Bool mask
                  mask = valid_meta['subject'].isin(subjects_in_split)
                  indices = np.where(mask)[0]
                  
                  if len(indices) == 0:
                      logger.warning(f"Split '{split_name}' resulted in 0 windows.")
                      output_dict[split_name] = (torch.tensor([]), torch.tensor([]), pd.DataFrame())
                  else:
                      split_data = tensor_data[indices]
                      split_targets = tensor_targets[indices]
                      split_meta = valid_meta.iloc[indices].reset_index(drop=True)
                      
                      output_dict[split_name] = (split_data, split_targets, split_meta)
                      
             return output_dict
#%%

if __name__ == "__main__":
    # Example usage
    tuh = TUHEEGEpilepsy(
        recording_ids=range(30),
        add_annotations=True,
    )
    
    # Uncomment to compute ICA
    # tuh.compute_ica_labels(n_jobs=4)

