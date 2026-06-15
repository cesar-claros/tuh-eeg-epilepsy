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
import re
import traceback
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
    # Coarse scalp regions for Option 3 (brain_ic) region binning. Each kept IC is
    # assigned to one of these by its dominant electrode; this fixed order is the
    # channel order of the regional time series.
    CANONICAL_REGIONS = (
        'frontal_left', 'frontal_right',
        'temporal_left', 'temporal_right',
        'posterior_left', 'posterior_right',
        'central_midline',
    )
    # Thresholds for region_from_dipole, in MNE head-coordinate meters (+x toward
    # the right preauricular point, +y toward the nasion, +z up, origin near the
    # head center). Calibrated against the fitted-dipole distribution across the
    # corpus (see documentation/brain_ic_dipoles.md, Section 8): the brain-IC
    # cloud sits at z > 0 (median ~0.045), so the temporal cut is positive, and
    # the midline band is narrowed so central_midline does not swallow ~half the
    # sources. Re-run src/dipole_diagnostics.py if the montage / corpus changes.
    _DIP_X_MID = 0.015         # |x| below this -> midline (central_midline)
    _DIP_X_LATERAL = 0.035     # |x| above this, if inferior, -> temporal
    _DIP_Z_TEMPORAL = 0.030    # z below this, if lateral, -> temporal lobe
    _DIP_Y_FRONTAL = 0.030     # y above this -> frontal
    _DIP_Y_POSTERIOR = -0.020  # y below this -> posterior

    def __init__(
        self,
        data_dir: str = '../../../data/',
        version: str = 'v3.0.0',
        recording_ids: Union[int, List[int], None] = None,
        add_annotations: bool = False,
        ica_keep_labels: tuple = ('brain', 'other'),
        brain_ic_min_gof: float = 0.0,
        brain_ic_use_dipoles: bool = True,
        ic_bag_max_k: int = 20,
        ic_bag_sign_normalize: bool = True,
        ic_bag_rank_by: str = 'variance',
    ):

        self.data_dir = data_dir
        self.version = version
        # ICLabel categories to keep when reconstructing in 'ica_clean' mode; all
        # other components (the confident artifacts) are excluded.
        self.ica_keep_labels = ica_keep_labels
        # 'brain_ic' options: minimum dipole goodness of fit (percent) to keep an
        # IC (0 = keep all), and whether to assign regions from cached dipoles
        # ('-ica_dipoles.csv') rather than the dominant-electrode heuristic.
        self.brain_ic_min_gof = brain_ic_min_gof
        self.brain_ic_use_dipoles = brain_ic_use_dipoles
        # 'ic_bag' options: max ICs per window (pad/truncate), whether to
        # sign-normalize sources by topography, and the ranking used to pick the
        # top ICs when there are more than ic_bag_max_k.
        self.ic_bag_max_k = ic_bag_max_k
        self.ic_bag_sign_normalize = ic_bag_sign_normalize
        self.ic_bag_rank_by = ic_bag_rank_by
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
            raw = TUHEEGEpilepsy._read_raw_edf(file_path, include=channels, preload=preload)
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
        # Args for dictionary learning
        idx_list: Optional[List[str]] = None,
        dictionary_learning: bool = False,
        n_windows_per_subject: Optional[int] = None,
        max_windows_per_subject: Optional[int] = None,
        overlap_pct: float = 0.0,
        balance_per_subject: bool = False,
        include_seizures: bool = True,
        fix_length_mode: Optional[str] = 'resample', # 'resample', 'pad', or None
        shuffle_windows: bool = False,
        seed: int = 42,
        splits: Optional[Dict[str, float]] = None,
        stratify_by: str = 'epilepsy',
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
                    idx_list=idx_list,
                    dictionary_learning=dictionary_learning,
                    n_windows_per_subject=n_windows_per_subject,
                    max_windows_per_subject=max_windows_per_subject,
                    stratify_by=stratify_by,
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

    @staticmethod
    def _generate_ic_dipoles(
        file_path: Path,
        montage_name: str = 'standard_1020',
    ) -> None:
        """Fit one equivalent-current dipole per IC topography and cache it.

        Offline, side-effect-heavy pass mirroring ``_generate_ICA_labels``. Reads
        the saved ICA solution (``-ica.fif``), fits a single dipole to each
        component's scalp map (a column of the mixing matrix) with a template
        spherical head model, and writes ``-ica_dipoles.csv`` (one row per IC:
        ``ic, x, y, z, ori_x, ori_y, ori_z, gof``; positions in MNE head
        coordinates, meters; ``gof`` in percent). Window loading reads this CSV
        instead of refitting, since the topographies are fixed per recording.
        Skips work if the CSV already exists. Requires ``compute_ica_labels`` to
        have produced the ``.fif`` first; no individual MRI is used, so locations
        are template-space approximations comparable across subjects.

        Parameters
        ----------
        file_path : Path
            Path to the original ``.edf`` (used to locate the sibling ICA files).
        montage_name : str, default='standard_1020'
            Standard montage supplying the template electrode positions.
        """
        try:
            ica_path = file_path.parent / file_path.name.replace('.edf', '-ica.fif')
            dipoles_path = file_path.parent / file_path.name.replace('.edf', '-ica_dipoles.csv')
            if dipoles_path.exists():
                logger.info(f"Skipping dipoles for {file_path.name}, already exist.")
                return
            if not ica_path.exists():
                logger.error(
                    f"Missing ICA solution for {file_path.name}; run compute_ica_labels first."
                )
                return

            ica = mne.preprocessing.read_ica(ica_path, verbose='ERROR')

            # Topographies as an Evoked: each "time sample" is one component's
            # scalp map, so fit_dipole returns one dipole per component at once.
            evoked = mne.EvokedArray(
                ica.get_components(), ica.info.copy(), tmin=0.0, verbose='ERROR'
            )
            # EDF EEG-only recordings carry no device->head transform, so the info
            # inherited from the ICA has dev_head_t=None; fit_dipole dereferences
            # it while logging. Set an identity transform (EEG is already in head
            # coordinates) so the fit proceeds.
            with evoked.info._unlock():
                evoked.info['dev_head_t'] = mne.transforms.Transform('meg', 'head')
            # Canonical 10-20 names + eeg type, then template electrode positions.
            TUHEEGEpilepsy._rename_channels(evoked)
            evoked.set_montage(montage_name, on_missing='ignore', verbose='ERROR')

            # Keep only channels that received a position from the montage; the
            # forward model and the sphere fit both need electrode locations.
            has_pos = [
                ch
                for ch, info_ch in zip(evoked.ch_names, evoked.info['chs'])
                if np.any(info_ch['loc'][:3]) and np.all(np.isfinite(info_ch['loc'][:3]))
            ]
            if len(has_pos) < 4:
                logger.error(
                    f"Too few localizable channels for {file_path.name}; skipping dipoles."
                )
                return
            evoked.pick(has_pos)
            evoked.set_eeg_reference('average', projection=True, verbose='ERROR')

            # Fixed adult-head spherical conductor in head coordinates (no MRI,
            # no headshape digitization). The electrodes are a fixed template, so
            # an auto-fitted sphere would be identical every time and only adds a
            # fragile dependency on dig points that the saved ICA does not carry.
            sphere = mne.make_sphere_model(
                r0=(0.0, 0.0, 0.04), head_radius=0.09, verbose='ERROR'
            )
            cov = mne.make_ad_hoc_cov(evoked.info, verbose='ERROR')
            dip, _ = mne.fit_dipole(evoked, cov, sphere, verbose='ERROR')

            pos = np.asarray(dip.pos)  # (n_components, 3), meters, head coords
            ori = np.asarray(dip.ori)  # (n_components, 3)
            gof = np.asarray(dip.gof)  # (n_components,), percent
            pd.DataFrame(
                {
                    'ic': np.arange(pos.shape[0]),
                    'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2],
                    'ori_x': ori[:, 0], 'ori_y': ori[:, 1], 'ori_z': ori[:, 2],
                    'gof': gof,
                }
            ).to_csv(dipoles_path, index=False)
        except Exception as e:
            logger.error(
                f"Error fitting IC dipoles for {file_path}: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

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

    def compute_ic_dipoles(
        self,
        n_jobs: int = 1,
    ) -> None:
        """Fit and cache per-IC dipoles for every recording.

        Run after ``compute_ica_labels`` (it needs the ``-ica.fif`` files). Writes
        one ``-ica_dipoles.csv`` next to each ``.edf``; existing files are skipped.

        Parameters
        ----------
        n_jobs : int, default=1
            Number of parallel workers (joblib) for the per-file dipole fits.
        """
        logger.info(f"Starting IC dipole fitting for {len(self.descriptions)} files...")
        paths = self.descriptions['path'].tolist()
        if n_jobs == 1:
            for path in tqdm(paths, desc="Fitting IC dipoles"):
                self._generate_ic_dipoles(path)
        else:
            Parallel(n_jobs=n_jobs)(
                delayed(self._generate_ic_dipoles)(path)
                for path in tqdm(paths, desc="Fitting IC dipoles")
            )
        logger.info("IC dipole fitting completed.")

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
        # Map each raw channel that is a 10-20 electrode to its canonical name,
        # matching case-insensitively so both stripped EDF names (e.g. "FP1") and
        # already-clean names (e.g. "Fp1") are recognised as EEG.
        upper_to_canonical = {c.upper(): c for c in montage1020.ch_names}
        mapping_eeg_names = {
            c: upper_to_canonical[c.upper()]
            for c in raw.ch_names
            if c.upper() in upper_to_canonical
        }

        non_eeg_names = [c for c in raw.ch_names if c not in mapping_eeg_names]
        if non_eeg_names:
            non_eeg_types = raw.get_channel_types(picks=non_eeg_names)
            mapping_non_eeg_types = {
                c: "misc" for c, t in zip(non_eeg_names, non_eeg_types) if t == "eeg"
            }
            if mapping_non_eeg_types:
                raw.set_channel_types(
                    mapping_non_eeg_types, on_unit_change="ignore" # type: ignore
                )

        if mapping_eeg_names:
            raw.set_channel_types(
                {c: "eeg" for c in mapping_eeg_names}, on_unit_change="ignore" # type: ignore
            )
            rename_map = {c: v for c, v in mapping_eeg_names.items() if c != v}
            if rename_map:
                raw.rename_channels(rename_map)

    @staticmethod
    def _set_montage(raw: mne.io.BaseRaw) -> None:
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

    @staticmethod
    def _apply_ica_cleaning(
        raw: mne.io.BaseRaw,
        raw_path: Path,
        keep_labels: tuple = ('brain', 'other'),
    ) -> Optional[mne.io.BaseRaw]:
        """Back-project the raw keeping only ICs whose ICLabel is in ``keep_labels``.

        Loads the ICA solution (``<name>-ica.fif``) and IC labels
        (``<name>-ica_labels.csv``) saved by ``compute_ica_labels``, sets an
        average reference to match how the ICA was fitted, then applies the ICA
        with every component whose label is NOT in ``keep_labels`` excluded. The
        default keeps 'brain' and 'other' (so only the confident artifact classes,
        muscle / eye / heart / line noise / channel noise, are removed). The result
        is a sensor-space signal, so the usual cross-file channel harmonization
        still applies. Returns ``None`` if the ICA files are missing or the
        application fails.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The loaded (cropped) raw recording, with its original channel names.
        raw_path : Path
            Path to the original ``.edf`` (used to locate the sibling ICA files).
        keep_labels : tuple, default=('brain', 'other')
            ICLabel categories to keep; all others are excluded.

        Returns
        -------
        mne.io.BaseRaw | None
            The cleaned raw, or ``None`` on missing files / failure.
        """
        ica_path = raw_path.parent / raw_path.name.replace('.edf', '-ica.fif')
        labels_path = raw_path.parent / raw_path.name.replace('.edf', '-ica_labels.csv')
        if not ica_path.exists() or not labels_path.exists():
            logger.error(f"Missing ICA solution/labels for {raw_path.name}; skipping window.")
            return None
        try:
            ica = mne.preprocessing.read_ica(ica_path, verbose='ERROR')
            labels_df = pd.read_csv(labels_path, index_col=0)
            keep = {str(k).strip().lower() for k in keep_labels}
            labels = labels_df['labels'].astype(str).str.strip().str.lower()
            exclude = labels_df.index[~labels.isin(keep)].tolist()

            raw = TUHEEGEpilepsy._align_raw_to_ica(raw, ica, raw_path)
            if raw is None:
                return None
            ica.apply(raw, exclude=exclude, verbose='ERROR')
            return raw
        except Exception as e:
            logger.error(f"Failed to apply ICA cleaning for {raw_path.name}: {e}")
            return None

    @staticmethod
    def _match_ica_channels(raw_names: list, ica_names: list) -> Optional[dict]:
        """Map raw channel names onto the ICA's channel names.

        Returns an empty dict if the raw already contains every ICA channel
        (no renaming needed), a ``{raw_name: ica_name}`` rename map when a
        normalized match aligns them, or ``None`` if some ICA channel has no
        counterpart in the raw.
        """
        if all(c in set(raw_names) for c in ica_names):
            return {}

        def norm(name: str) -> str:
            return (
                name.upper()
                .replace("EEG ", "")
                .replace("-REF", "")
                .replace("-LE", "")
                .strip()
            )

        raw_by_norm: dict = {}
        for c in raw_names:
            raw_by_norm.setdefault(norm(c), c)

        rename: dict = {}
        for ic in ica_names:
            match = raw_by_norm.get(norm(ic))
            if match is None:
                return None
            if match != ic:
                rename[match] = ic
        return rename
    
    @staticmethod
    def _align_raw_to_ica(
        raw: mne.io.BaseRaw, ica, raw_path: Path
    ) -> Optional[mne.io.BaseRaw]:
        """Align the raw to the ICA's channels (rename, pick, eeg-type, avg-ref).

        Renames the raw channels onto the ICA's names (exact or normalized match),
        restricts to them, forces the 'eeg' type, and sets an average reference to
        match how the ICA was fitted. Returns the aligned raw, or ``None`` if the
        channels cannot be aligned.
        """
        rename = TUHEEGEpilepsy._match_ica_channels(raw.ch_names, ica.ch_names)
        if rename is None:
            logger.error(
                f"Cannot align raw to ICA channels for {raw_path.name}: "
                f"raw={raw.ch_names[:4]}..., ica={ica.ch_names[:4]}..."
            )
            return None
        if rename:
            raw.rename_channels(rename)
        raw.pick(ica.ch_names)
        # Force the EEG type so the average reference and ICA picks resolve
        # (read_raw_edf does not always tag channels as 'eeg').
        raw.set_channel_types({c: 'eeg' for c in raw.ch_names}, verbose='ERROR')
        raw.set_eeg_reference('average', verbose='ERROR')
        return raw

    @staticmethod
    def _electrode_region(name: str) -> str:
        """Map a 10-20 electrode name to one of ``CANONICAL_REGIONS``.

        Lobe is inferred from the letter prefix and hemisphere from the trailing
        digit (odd = left, even = right, 'z' = midline). Frontal and temporal keep
        their hemisphere; parietal / occipital map to 'posterior'; central and all
        midline electrodes map to 'central_midline'. Unknown names fall back to
        'central_midline'.
        """
        n = name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "").strip()
        match = re.search(r'(\d+|Z)$', n)
        if match is None:
            return 'central_midline'
        suffix = match.group(1)
        hemi = 'mid' if suffix == 'Z' else ('left' if int(suffix) % 2 == 1 else 'right')
        prefix = n[:match.start()]
        lobe = 'central'
        for pre, lb in (
            ('FP', 'frontal'), ('AF', 'frontal'), ('FC', 'central'), ('FT', 'temporal'),
            ('TP', 'temporal'), ('CP', 'parietal'), ('PO', 'occipital'),
            ('F', 'frontal'), ('T', 'temporal'), ('C', 'central'),
            ('P', 'parietal'), ('O', 'occipital'),
        ):
            if prefix.startswith(pre):
                lobe = lb
                break
        if hemi == 'mid':
            return 'central_midline'
        if lobe in ('frontal', 'temporal'):
            return f'{lobe}_{hemi}'
        if lobe in ('parietal', 'occipital'):
            return f'posterior_{hemi}'
        return 'central_midline'

    @staticmethod
    def region_from_dipole(x: float, y: float, z: float) -> str:
        """Map a fitted dipole location to one of ``CANONICAL_REGIONS``.

        Coordinates are MNE head coordinates in meters (+x toward the right
        preauricular point, +y toward the nasion, +z up, origin near the head
        center). The folding mirrors ``_electrode_region`` (frontal and temporal
        keep their hemisphere; parietal and occipital collapse into 'posterior';
        central, vertex and midline sources go to 'central_midline') but uses the
        source location instead of the dominant electrode. Thresholds are the
        ``_DIP_*`` class constants.
        """
        hemi = 'left' if x < 0 else 'right'
        # Temporal lobes sit low and lateral.
        if z < TUHEEGEpilepsy._DIP_Z_TEMPORAL and abs(x) > TUHEEGEpilepsy._DIP_X_LATERAL:
            return f'temporal_{hemi}'
        # Close to the midline -> central / vertex / midline.
        if abs(x) < TUHEEGEpilepsy._DIP_X_MID:
            return 'central_midline'
        if y > TUHEEGEpilepsy._DIP_Y_FRONTAL:
            return f'frontal_{hemi}'
        if y < TUHEEGEpilepsy._DIP_Y_POSTERIOR:
            return f'posterior_{hemi}'
        # Lateral but central in y: no lateral-central region, fold to midline.
        return 'central_midline'

    @staticmethod
    def _brain_ic_regional(
        raw: mne.io.BaseRaw,
        raw_path: Path,
        keep_labels: tuple = ('brain',),
        min_gof: float = 0.0,
        use_dipoles: bool = True,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Build the K-region time series from the kept brain ICs (Option 3, R1).

        Selects the components whose ICLabel is in ``keep_labels``, sign-normalizes
        each by its topography (dominant projection made positive), assigns each to
        a scalp region, and sums the sign-normalized sources within each region.
        When ``use_dipoles`` is True, region assignment prefers a cached per-IC
        dipole location (``-ica_dipoles.csv`` via ``region_from_dipole``) and falls
        back to the dominant-electrode heuristic (``_electrode_region``) when no
        dipole file is present; with ``use_dipoles=False`` the electrode heuristic
        is always used.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The loaded (cropped) raw recording, with its original channel names.
        raw_path : Path
            Path to the original ``.edf`` (used to locate the sibling ICA files).
        keep_labels : tuple, default=('brain',)
            ICLabel categories whose components are used.
        min_gof : float, default=0.0
            If > 0 and dipoles are cached, drop ICs whose dipole goodness of fit
            (percent) is below this, keeping only confidently localized sources.
        use_dipoles : bool, default=True
            If True, assign regions from the cached dipole locations when present;
            if False, always use the dominant-electrode heuristic.

        Returns
        -------
        (np.ndarray, list[str]) | None
            ``(data, region_names)`` with ``data`` of shape (n_regions, n_times),
            or ``None`` on missing files / failure / no kept components.
        """
        ica_path = raw_path.parent / raw_path.name.replace('.edf', '-ica.fif')
        labels_path = raw_path.parent / raw_path.name.replace('.edf', '-ica_labels.csv')
        if not ica_path.exists() or not labels_path.exists():
            logger.error(f"Missing ICA solution/labels for {raw_path.name}; skipping window.")
            return None
        try:
            ica = mne.preprocessing.read_ica(ica_path, verbose='ERROR')
            labels_df = pd.read_csv(labels_path, index_col=0)
            keep = {str(k).strip().lower() for k in keep_labels}
            labels = labels_df['labels'].astype(str).str.strip().str.lower()
            keep_idx = labels_df.index[labels.isin(keep)].tolist()
            if not keep_idx:
                logger.error(
                    f"No kept ICs ({sorted(keep)}) for {raw_path.name}; skipping window."
                )
                return None

            raw = TUHEEGEpilepsy._align_raw_to_ica(raw, ica, raw_path)
            if raw is None:
                return None

            sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
            topographies = ica.get_components()         # (n_channels, n_components)
            regions = TUHEEGEpilepsy.CANONICAL_REGIONS
            region_index = {r: i for i, r in enumerate(regions)}
            regional = np.zeros((len(regions), sources.shape[1]), dtype=np.float64)

            # Prefer cached per-IC dipole locations (source geometry) over the
            # dominant-electrode heuristic; fall back to the electrode when dipoles
            # are disabled, the dipole file is absent, or a given IC's row is missing.
            dipoles_path = raw_path.parent / raw_path.name.replace('.edf', '-ica_dipoles.csv')
            dipoles = (
                pd.read_csv(dipoles_path).set_index('ic')
                if use_dipoles and dipoles_path.exists()
                else None
            )

            for j in keep_idx:
                topo = topographies[:, j]
                dom = int(np.argmax(np.abs(topo)))
                sign = 1.0 if topo[dom] >= 0 else -1.0
                region = None
                if dipoles is not None and j in dipoles.index:
                    row = dipoles.loc[j]
                    if min_gof > 0.0 and float(row['gof']) < min_gof:
                        continue  # poorly localized IC: not a confident dipole
                    region = TUHEEGEpilepsy.region_from_dipole(
                        float(row['x']), float(row['y']), float(row['z'])
                    )
                if region is None:
                    region = TUHEEGEpilepsy._electrode_region(ica.ch_names[dom])
                regional[region_index[region]] += sign * sources[j]

            return regional, list(regions)
        except Exception as e:
            logger.error(f"Failed to build brain-IC regions for {raw_path.name}: {e}")
            return None

    @staticmethod
    def _ic_bag_sources(
        raw: mne.io.BaseRaw,
        raw_path: Path,
        keep_labels: tuple = ('brain',),
        max_k: int = 20,
        sign_normalize: bool = True,
        rank_by: str = 'variance',
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Build a padded bag of kept IC sources for univariate-HYDRA pooling.

        Selects the components whose ICLabel is in ``keep_labels``, optionally
        sign-normalizes each by its topography (dominant projection made positive),
        ranks them, takes the top ``max_k``, and stacks them into a fixed
        ``(max_k, n_times)`` array (zero-padded if fewer kept ICs). Channel names
        are positional placeholders (``ic_0`` ...); the downstream
        ``ICBagTransformer`` pools over them, so slot order does not matter.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The loaded (cropped) raw recording, with its original channel names.
        raw_path : Path
            Path to the original ``.edf`` (used to locate the sibling ICA files).
        keep_labels : tuple, default=('brain',)
            ICLabel categories whose components are used.
        max_k : int, default=20
            Number of IC slots per window; more kept ICs are truncated to the top
            ``max_k`` by ``rank_by``, fewer are zero-padded.
        sign_normalize : bool, default=True
            If True, flip each source so its dominant topography projection is
            positive (polarity consistent across recordings).
        rank_by : str, default='variance'
            Ranking used to pick the top ``max_k`` ICs: 'variance' (source
            variance) or 'prob' (ICLabel probability, if available).

        Returns
        -------
        (np.ndarray, list[str]) | None
            ``(data, names)`` with ``data`` of shape (max_k, n_times), or ``None``
            on missing files / failure / no kept components.
        """
        ica_path = raw_path.parent / raw_path.name.replace('.edf', '-ica.fif')
        labels_path = raw_path.parent / raw_path.name.replace('.edf', '-ica_labels.csv')
        if not ica_path.exists() or not labels_path.exists():
            logger.error(f"Missing ICA solution/labels for {raw_path.name}; skipping window.")
            return None
        try:
            ica = mne.preprocessing.read_ica(ica_path, verbose='ERROR')
            labels_df = pd.read_csv(labels_path, index_col=0)
            keep = {str(k).strip().lower() for k in keep_labels}
            labels = labels_df['labels'].astype(str).str.strip().str.lower()
            keep_idx = labels_df.index[labels.isin(keep)].tolist()
            if not keep_idx:
                logger.error(
                    f"No kept ICs ({sorted(keep)}) for {raw_path.name}; skipping window."
                )
                return None

            raw = TUHEEGEpilepsy._align_raw_to_ica(raw, ica, raw_path)
            if raw is None:
                return None

            sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
            topographies = ica.get_components()         # (n_channels, n_components)
            n_times = sources.shape[1]
            use_prob = rank_by == 'prob' and 'y_pred_proba' in labels_df.columns

            keys: List[float] = []
            srcs: List[np.ndarray] = []
            for j in keep_idx:
                src = sources[j]
                if sign_normalize:
                    topo = topographies[:, j]
                    if topo[int(np.argmax(np.abs(topo)))] < 0:
                        src = -src
                keys.append(float(labels_df['y_pred_proba'].loc[j]) if use_prob else float(np.var(src)))
                srcs.append(src)

            # Rank descending by key, keep the top max_k.
            order = np.argsort(keys)[::-1][:max_k]
            bag = np.zeros((max_k, n_times), dtype=np.float64)
            for i, idx in enumerate(order):
                bag[i] = srcs[int(idx)]
            names = [f'ic_{i}' for i in range(max_k)]
            return bag, names
        except Exception as e:
            logger.error(f"Failed to build IC bag for {raw_path.name}: {e}")
            return None

    @staticmethod
    def _windowed_tensor_gb(
        n_windows: int, n_channels: int, n_samples: int, bytes_per: int = 4
    ) -> float:
        """Estimate the windowed-tensor size in GB for a given shape and dtype width."""
        return n_windows * n_channels * n_samples * bytes_per / 1e9

    @staticmethod
    def _read_raw_edf(
        file_path: Path,
        include: Optional[List[str]] = None,
        preload: bool = False,
    ) -> mne.io.BaseRaw:
        """Read an EDF, silencing the benign mixed-sampling-frequency warning.

        With ``preload=False`` MNE upsamples any lower-rate channels on the fly and
        warns about edge artifacts. In our pipeline the EEG channels are at the
        native (maximum) rate, so they are never resampled, and any resampled
        non-EEG channels are dropped before features are computed; the warning does
        not affect the result. Suppress it so the per-window logs (one read per
        window in lazy mode) stay clean.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*mixed sampling frequencies.*",
                category=RuntimeWarning,
            )
            return mne.io.read_raw_edf(
                file_path,
                include=include,
                preload=preload,
                infer_types=False,
                verbose="error",
            )

    def _calculate_limit_per_subject(
            self,
            df:pd.DataFrame,
            window_len_s:int,
            stride:float,
            balance_per_subject:bool,
            max_windows_per_subject: Optional[int] = None,
            unit: str = 'seconds',
        ) -> Optional[int]:
        # Approx windows per file
        df = df.copy() # Avoid modifying original
        if unit == 'seconds':
            df['n_windows'] = np.maximum(0, np.floor((df['duration'] - window_len_s) / stride) + 1).astype(int)        
        elif unit == 'samples':
            assert isinstance(stride, int), "When unit is 'samples', stride must be an integer number of samples."
            df['n_windows'] = np.maximum(0, np.floor((df['duration'] - window_len_s)*df['sfreq'] / stride) + 1).astype(int)
        # Windows per subject
        subject_window_counts = df.groupby('subject')['n_windows'].sum().sort_index()
        smallest_n_windows_count = subject_window_counts.nsmallest(10)
        # logger.info(f"Subjects with the lowest number of windows:\n{smallest_n_windows_count}")
        df_smallest = df[df["subject"].isin(smallest_n_windows_count.index)][["subject", "duration","n_windows"]].sort_values(["subject","n_windows" ])
        logger.info(f'Durations of subjects with the lowest number of windows:\n{df_smallest}')
        
        total_available = int(subject_window_counts.sum())
        n_subjects = int(subject_window_counts.shape[0])
        strict_min = int(subject_window_counts.min())

        if balance_per_subject:
            if max_windows_per_subject is not None:
                # Soft balance: cap each subject at max_windows_per_subject;
                # subjects with fewer keep all they have (the slice in
                # _generate_windows_list takes min(actual, limit)). Uses far more
                # data than clamping every subject to the global minimum (strict
                # balance), at the cost of mild per-subject imbalance.
                limit_per_subject = int(max_windows_per_subject)
            else:
                # Strict balance: every subject contributes the global-minimum count.
                limit_per_subject = strict_min
            mode_str = "soft cap" if max_windows_per_subject is not None else "strict global-min"
            logger.info(
                f"Balancing ({mode_str}): limiting to {limit_per_subject} windows per subject "
                f"(fewest-window subject '{subject_window_counts.idxmin()}' has {strict_min})."
            )
            if limit_per_subject == 0:
                raise ValueError("Balancing requested but the per-subject limit is 0.")
        else:
            limit_per_subject = (
                int(max_windows_per_subject) if max_windows_per_subject is not None else None
            )
            logger.info(
                "Unbalanced: "
                + (
                    f"capping at {limit_per_subject} windows per subject."
                    if limit_per_subject is not None
                    else "using all available windows."
                )
            )

        # Quantify how much of the available window budget the limit keeps, so the
        # cost of balancing is visible (the strict global-min clamp can discard the
        # large majority of windows).
        kept = (
            int(subject_window_counts.clip(upper=limit_per_subject).sum())
            if limit_per_subject is not None
            else total_available
        )
        logger.info(
            f"Window budget: {kept} / {total_available} kept "
            f"({100 * kept / max(total_available, 1):.1f}%) across {n_subjects} subjects; "
            f"per-subject window counts min={strict_min}, "
            f"median={int(subject_window_counts.median())}, max={int(subject_window_counts.max())}."
        )
        return df, limit_per_subject

    def _generate_windows_list(
            self, 
            df: pd.DataFrame,
            window_len_s: float,
            stride: float,
            shuffle_windows: bool,
            limit_per_subject: Optional[int],
            rng: np.random.RandomState,
            unit: str = 'seconds',
        ) -> pd.DataFrame:
        window_meta = []
        
        # Group by subject to enforce limits
        # To handle shuffle correctly with balanced limits, we must collect ALL possibilities first for a subject
        
        for subject, group in tqdm( df.groupby('subject'), desc="Generating windows list per subject"):
            # Sort for deterministic base order
            group = group.sort_values(['year', 'month', 'day', 'time'] if {'year','month','day','time'}.issubset(group.columns) else ['path'])
            
            subject_windows = []
            
            for _, row in group.iterrows():
                # age, gender = utils.parse_age_and_gender_from_edf_header(row['path'])
                n_possible = row['n_windows']
                if n_possible <= 0:
                    continue
                indices = np.arange(n_possible)
                if unit == 'samples':
                    if limit_per_subject is not None and limit_per_subject < n_possible:
                        indices = rng.permutation(indices)[:limit_per_subject] 
                # Calculate start times for this file
                for idx in tqdm(indices, desc=f"Windows for subject {subject}", leave=False):
                    if unit == 'seconds':
                        start_t = idx * stride
                    elif unit == 'samples':
                        start_t = (idx * stride)/row['sfreq']
                    end_t = start_t + window_len_s
                    
                    subject_windows.append({
                        'subject': subject,
                        'path': row['path'],
                        'start': start_t,
                        'end': end_t,
                        't_id': row.get('t_id', 'unknown'),
                        'epilepsy': row.get('epilepsy', False),
                        # Store other useful metadata from row if needed
                        # 'age': row['age'],
                        # 'gender': row['gender'],
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
        return window_df

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
        idx_list: Optional[List[str]],
        dictionary_learning: bool,
        n_windows_per_subject: Optional[int],
        max_windows_per_subject: Optional[int],
        stratify_by: str,
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
        # Filter by idx_list if provided (e.g. specific subjects)
        if idx_list is not None:
            df = df[df['subject'].isin(idx_list)]

        if df.empty:
            raise ValueError("No data available after filtering.")
        
        # 2. Calculate available windows per subject
        
        if dictionary_learning:
            df['duration'] = df['duration']/2
            stride = 1 # For dictionary learning, we want all possible windows with no stride (fully overlapping) to maximize data for learning. The actual windowing will be handled in the dictionary learning step, and we will generate a separate set of windows for that.
            df, limit_per_subject = self._calculate_limit_per_subject(df, window_len_s, stride, balance_per_subject, unit='samples')
            if n_windows_per_subject is not None and n_windows_per_subject < limit_per_subject:
                limit_per_subject = n_windows_per_subject
            logger.info(f"Dictionary Learning Mode: Generating fully overlapping windows with stride {stride}s. Adjusted limit per subject: {limit_per_subject} windows.")
            # 3. Generate Window List
            window_df = self._generate_windows_list(df, window_len_s, stride, shuffle_windows, limit_per_subject, rng, unit='samples')
        else:
            stride_s = window_len_s * (1 - overlap_pct)
            if stride_s <= 0:
                raise ValueError("Overlap must be < 1.0")
            logger.info(f"Window length: {window_len_s}s, Overlap: {overlap_pct*100}%, Stride: {stride_s}s")

            df, limit_per_subject = self._calculate_limit_per_subject(
                df, window_len_s, stride_s, balance_per_subject,
                max_windows_per_subject=max_windows_per_subject, unit='seconds',
            )

            # 3. Generate Window List
            window_df = self._generate_windows_list(df, window_len_s, stride_s, shuffle_windows, limit_per_subject, rng, unit='seconds')

        # Pre-flight memory estimate: the loader materializes every window in RAM
        # (float64 while processing, float32 once stacked), so memory scales with
        # the planned window count. Log the projected size before allocating so an
        # oversized configuration is visible up front rather than via an OOM kill.
        n_planned = len(window_df)
        est_sfreq = float(df['sfreq'].min()) if 'sfreq' in df.columns and len(df) else 0.0
        est_samples = int(window_len_s * est_sfreq)
        if mode == 'brain_ic':
            est_channels = len(TUHEEGEpilepsy.CANONICAL_REGIONS)
        elif mode == 'ic_bag':
            est_channels = self.ic_bag_max_k
        elif 'montage' in df.columns:
            montage_sets = [
                set(self.montages[m]['channels'])
                for m in df['montage'].unique()
                if m in self.montages
            ]
            est_channels = len(set.intersection(*montage_sets)) if montage_sets else 0
        else:
            est_channels = 0
        if est_samples > 0 and est_channels > 0:
            gb_resident = self._windowed_tensor_gb(n_planned, est_channels, est_samples, 4)
            gb_peak = self._windowed_tensor_gb(n_planned, est_channels, est_samples, 8) * 2.5
            logger.info(
                f"Pre-flight memory estimate: {n_planned} windows x {est_channels} ch x "
                f"{est_samples} samp (~{est_sfreq:.0f} Hz) -> ~{gb_resident:.1f} GB resident "
                f"(float32); est. peak ~{gb_peak:.1f} GB during load (float64 + copies). "
                f"Lower data.max_windows_per_subject or data.window_len_min if this exceeds node RAM."
            )
        else:
            logger.warning("Pre-flight memory estimate skipped (missing sfreq/montage info).")

        # 3.5 Handle Data Splitting (Assign splits to rows)
        # Default split is 'all' if no splits provided
        split_map = {} # subject -> split_name
        
        unique_subjects = window_df['subject'].unique()
        
        if splits is not None: 
            # Check sum
            total_ratio = sum(splits.values())
            if not np.isclose(total_ratio, 1.0):
                logger.warning(f"Split ratios sum to {total_ratio}, not 1.0. This might leave some subjects unused or cause overlap issues if > 1.")

            # Stratification Logic
            # Get label per subject
            # We need to look back at original or window_df to find the stratify_by value for each subject
            
            # Simple aggregation: take max value (works for boolean/binary epilepsy) or mode
            # window_df has the columns.
            if stratify_by not in window_df.columns:
                 logger.warning(f"Stratification column '{stratify_by}' not found. Falling back to random splitting.")
                 subject_labels = None
            else:
                 # Group by subject and take the first value (assuming constant per subject) or max
                 # For epilepsy: if any window is epilepsy, subject is epilepsy? 
                 # Usually epilepsy status is subject-level constant.
                 # Let's use max() to be safe for binary tags.
                 # Optimization: drop duplicates first
                 # Get a dataframe of unique subjects and their labels
                 subj_label_df = window_df[['subject', stratify_by]].drop_duplicates('subject')
                 # If duplicates exist (e.g. subject has changing age?), max might be safer? 
                 # Generally assume constant. 
                 subject_labels = subj_label_df.set_index('subject')[stratify_by]
            
            # Divide subjects by label
            if subject_labels is not None:
                # Group subjects by their label
                # label -> list of subjects
                label_groups = {}
                for s in unique_subjects:
                    lbl = subject_labels.loc[s]
                    if lbl not in label_groups:
                        label_groups[lbl] = []
                    label_groups[lbl].append(s)
                    
                logger.info(f"Stratification Groups ({stratify_by}): { {k: len(v) for k, v in label_groups.items()} }")
                
                # Split each group
                for lbl, subjects in label_groups.items():
                    subjects = np.array(subjects)
                    rng.shuffle(subjects)
                    
                    n_sub_group = len(subjects)
                    current_idx = 0
                    
                    for split_name, ratio in splits.items():
                        n_split = int(np.floor(ratio * n_sub_group))
                        split_subs = subjects[current_idx : current_idx + n_split]
                        for s in split_subs:
                            split_map[s] = split_name
                        current_idx += n_split
                        
                    # Handle remainder for this group
                    if current_idx < n_sub_group and np.isclose(total_ratio, 1.0):
                        remainder = subjects[current_idx:]
                        last_split = list(splits.keys())[-1]
                        for s in remainder:
                             split_map[s] = last_split
            else:
                 # No stratification, random shuffle
                rng.shuffle(unique_subjects)
                n_sub = len(unique_subjects)
                current_idx = 0
                for split_name, ratio in splits.items():
                    n_split = int(np.floor(ratio * n_sub))
                    split_subs = unique_subjects[current_idx : current_idx + n_split]
                    for s in split_subs:
                        split_map[s] = split_name
                    current_idx += n_split
                
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
            elif mode in ('ica_clean', 'brain_ic', 'ic_bag'):
                # ica_clean / brain_ic / ic_bag: read ALL channels so the saved
                # ICA's channels are present, then back-project to sensor space
                # (ica_clean), build the regional IC series (brain_ic), or build the
                # padded bag of IC sources (ic_bag).
                file_path = desc_row["path"]
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
                
                raw = TUHEEGEpilepsy._read_raw_edf(file_path, include=ch_names, preload=False)

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

                # Option 3 (brain_ic): build the fixed K-region time series from
                # the kept brain ICs and return it directly. The regions are the
                # channels, so no further sensor-space processing applies and we
                # bypass the rename / montage / pick steps below.
                if mode == 'brain_ic':
                    out = TUHEEGEpilepsy._brain_ic_regional(
                        raw,
                        desc_row['path'],
                        self.ica_keep_labels,
                        min_gof=self.brain_ic_min_gof,
                        use_dipoles=self.brain_ic_use_dipoles,
                    )
                    if out is None:
                        return None
                    regional_data, region_names = out
                    return regional_data, raw.info['sfreq'], region_names

                # Option (ic_bag): build the padded bag of kept IC sources and
                # return directly (IC slots are the channels; the ICBagTransformer
                # pools over them). Bypasses the sensor-space steps below.
                if mode == 'ic_bag':
                    out = TUHEEGEpilepsy._ic_bag_sources(
                        raw,
                        desc_row['path'],
                        self.ica_keep_labels,
                        max_k=self.ic_bag_max_k,
                        sign_normalize=self.ic_bag_sign_normalize,
                        rank_by=self.ic_bag_rank_by,
                    )
                    if out is None:
                        return None
                    bag_data, bag_names = out
                    return bag_data, raw.info['sfreq'], bag_names

                # Option 1 (ica_clean): keep only ICs whose ICLabel is in
                # self.ica_keep_labels (default brain + other), back-projected.
                if mode == 'ica_clean':
                    raw = TUHEEGEpilepsy._apply_ica_cleaning(
                        raw, desc_row['path'], self.ica_keep_labels
                    )
                    if raw is None:
                        return None

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
        # tensor_data = np.transpose(tensor_data, (0, 2, 1)) # (Batch, Time, Channels)
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
