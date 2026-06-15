"""Lazy (streaming) windowed dataset for the TUH EEG Epilepsy corpus.

This is the memory-light counterpart to the eager ``_load_balanced_windows`` path
in ``tuh_eeg_epilepsy.py``. Instead of materializing every window into one big
``(N, C, T)`` tensor in RAM, it builds only the window *plan* up front and reads /
processes each window on demand in ``WindowDataset.__getitem__`` (so a DataLoader
with ``num_workers`` streams batches). Resident memory becomes O(batch) instead of
O(N), which is what the custom sklearn-style Trainer actually needs: it extracts
HYDRA features in a single streaming pass, so the raw windows never all need to be
resident, only the much smaller feature matrix.

Scope of this draft:

- Supports ``mode='raw'`` and ``mode='brain_ic'``. ``ica`` / ``ica_clean`` raise
  ``NotImplementedError`` (their harmonization target depends on per-file ICA
  channels; use the eager path for those).
- Harmonization targets (the common channel set and the resample rate) are fixed
  *up front* (the eager path derives them from the loaded batch). The channel set
  is found by a header-only scan of the unique files in the plan, matching the
  eager intersection; the resample rate is the minimum sfreq.
- A window that fails to load is zero-filled (not dropped), so ``__len__`` is
  fixed and the per-split metadata stays aligned with the feature order. For
  ``raw`` this never triggers (no ICA dependency), so lazy and eager produce
  identical windows; for ``brain_ic`` a failed window becomes a zero row instead
  of being dropped (a small, documented difference from eager).

The plan + split assignment reuse the engine's own helpers and RNG order, so the
selected windows and the subject-level split match the eager path. Validate with
the windows_*.csv dump (lazy vs eager) before relying on it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy


def _assign_splits(
    window_df: pd.DataFrame,
    splits: dict,
    stratify_by: str,
    rng: np.random.RandomState,
) -> dict:
    """Assign whole subjects to splits (stratified), mirroring the eager path.

    Returns a ``{subject: split_name}`` map. The RNG is consumed in the same order
    as ``_load_balanced_windows`` so, given the same plan and seed, the split is
    identical.
    """
    split_map: dict = {}
    unique_subjects = window_df['subject'].unique()
    total_ratio = sum(splits.values())
    if not np.isclose(total_ratio, 1.0):
        logger.warning(f"Split ratios sum to {total_ratio}, not 1.0.")

    if stratify_by not in window_df.columns:
        logger.warning(f"Stratification column '{stratify_by}' not found; random split.")
        subject_labels = None
    else:
        subj_label_df = window_df[['subject', stratify_by]].drop_duplicates('subject')
        subject_labels = subj_label_df.set_index('subject')[stratify_by]

    if subject_labels is not None:
        label_groups: dict = {}
        for s in unique_subjects:
            label_groups.setdefault(subject_labels.loc[s], []).append(s)
        logger.info(f"Stratification groups ({stratify_by}): { {k: len(v) for k, v in label_groups.items()} }")
        for _lbl, subjects in label_groups.items():
            subjects = np.array(subjects)
            rng.shuffle(subjects)
            n_sub_group = len(subjects)
            current_idx = 0
            for split_name, ratio in splits.items():
                n_split = int(np.floor(ratio * n_sub_group))
                for s in subjects[current_idx:current_idx + n_split]:
                    split_map[s] = split_name
                current_idx += n_split
            if current_idx < n_sub_group and np.isclose(total_ratio, 1.0):
                last_split = list(splits.keys())[-1]
                for s in subjects[current_idx:]:
                    split_map[s] = last_split
    else:
        rng.shuffle(unique_subjects)
        n_sub = len(unique_subjects)
        current_idx = 0
        for split_name, ratio in splits.items():
            n_split = int(np.floor(ratio * n_sub))
            for s in unique_subjects[current_idx:current_idx + n_split]:
                split_map[s] = split_name
            current_idx += n_split
        if current_idx < n_sub and np.isclose(total_ratio, 1.0):
            last_split = list(splits.keys())[-1]
            for s in unique_subjects[current_idx:]:
                split_map[s] = last_split
    return split_map


def _scan_common_channels(
    window_df: pd.DataFrame, montages: dict, rename_channels: bool
) -> List[str]:
    """Header-only scan of the plan's unique files to find the common channel set.

    Mirrors the eager harmonization (intersection of renamed channel names across
    all windows), but reads only EDF headers (``preload=False``), so it is cheap.
    """
    common: Optional[set] = None
    seen: set = set()
    for _, row in window_df.iterrows():
        desc = row['description_row']
        path = desc['path']
        if path in seen:
            continue
        seen.add(path)
        include = list(montages[desc['montage']]['channels'])
        try:
            raw = mne.io.read_raw_edf(
                path, include=include, preload=False, infer_types=False, verbose='error'
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Channel scan skipped {path}: {e}")
            continue
        if rename_channels:
            TUHEEGEpilepsy._rename_channels(raw)
        names = set(raw.ch_names)
        common = names if common is None else (common & names)
    if not common:
        raise ValueError("No common channels found in the lazy channel scan.")
    return sorted(common)


def _harmonize_one(
    data: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    target_channels: List[str],
    target_sfreq: float,
    target_len: int,
) -> Optional[np.ndarray]:
    """Reindex to ``target_channels``, resample to ``target_sfreq``, fix length.

    Per-window version of the eager global harmonization. Returns ``None`` if a
    target channel is missing (the eager intersection guarantees it is present for
    ``raw``, so this only guards against surprises).
    """
    idx_map = {name: i for i, name in enumerate(ch_names)}
    try:
        indices = [idx_map[c] for c in target_channels]
    except KeyError:
        return None
    data = data[indices, :]
    if not np.isclose(sfreq, target_sfreq):
        data = mne.filter.resample(data, up=target_sfreq, down=sfreq, axis=-1)
    cur = data.shape[1]
    if cur < target_len:
        pad = np.zeros((data.shape[0], target_len - cur), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=1)
    elif cur > target_len:
        data = data[:, :target_len]
    return data.astype(np.float32, copy=False)


class WindowDataset(Dataset):
    """Streaming dataset: reads and processes one window per ``__getitem__``.

    Holds only the window plan (metadata) plus the fixed harmonization targets, so
    it is light to construct and picklable for DataLoader workers. Returns
    ``(x, y)`` with ``x`` of shape ``(len(target_channels), target_len)``.
    """

    def __init__(
        self,
        plan: pd.DataFrame,
        mode: str,
        target_channels: List[str],
        target_sfreq: float,
        target_len: int,
        montages: dict,
        filter_freq: Optional[List[float]],
        rename_channels: bool,
        set_montage: bool,
        pick_channels: Optional[List[str]],
        ica_keep_labels: tuple,
        brain_ic_min_gof: float,
        brain_ic_use_dipoles: bool,
        target_cols: List[str],
    ) -> None:
        self.plan = plan.reset_index(drop=True)
        self.mode = mode
        self.target_channels = list(target_channels)
        self.target_sfreq = float(target_sfreq)
        self.target_len = int(target_len)
        self.montages = montages
        self.filter_freq = filter_freq
        self.rename_channels = rename_channels
        self.set_montage = set_montage
        self.pick_channels = pick_channels
        self.ica_keep_labels = ica_keep_labels
        self.brain_ic_min_gof = brain_ic_min_gof
        self.brain_ic_use_dipoles = brain_ic_use_dipoles
        self.target_cols = list(target_cols)

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.plan.iloc[i]
        data = self._process(row)
        if data is None:
            data = np.zeros((len(self.target_channels), self.target_len), dtype=np.float32)
        # squeeze so a single-target label is a per-sample scalar, matching the
        # eager TensorDataset (which squeezes the stacked target tensor).
        return torch.from_numpy(data).float(), self._encode_label(row).squeeze()

    def _process(self, row: pd.Series) -> Optional[np.ndarray]:
        desc = row['description_row']
        if self.mode == 'brain_ic':
            include = None
        else:
            include = list(self.montages[desc['montage']]['channels'])
        try:
            raw = mne.io.read_raw_edf(
                desc['path'], include=include, preload=False, infer_types=False, verbose='error'
            )
            raw.crop(tmin=row['start'], tmax=row['end'], include_tmax=False)
            raw.load_data()

            if self.mode == 'brain_ic':
                out = TUHEEGEpilepsy._brain_ic_regional(
                    raw, desc['path'], self.ica_keep_labels,
                    min_gof=self.brain_ic_min_gof, use_dipoles=self.brain_ic_use_dipoles,
                )
                if out is None:
                    return None
                data, ch_names = out
                sfreq = raw.info['sfreq']
            else:
                if self.filter_freq is not None:
                    raw.filter(l_freq=self.filter_freq[0], h_freq=self.filter_freq[1], verbose='ERROR')
                if self.rename_channels:
                    TUHEEGEpilepsy._rename_channels(raw)
                if self.set_montage:
                    TUHEEGEpilepsy._set_montage(raw)
                if self.pick_channels:
                    raw.pick(self.pick_channels)
                data = raw.get_data()
                ch_names = raw.ch_names
                sfreq = raw.info['sfreq']

            return _harmonize_one(
                data, ch_names, sfreq, self.target_channels, self.target_sfreq, self.target_len
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Lazy window load failed ({desc['path']} @ {row['start']}s): {e}")
            return None

    def _encode_label(self, row: pd.Series) -> torch.Tensor:
        vals = []
        for col in self.target_cols:
            v = row[col] if col in row.index else -1
            if col == 'gender':
                v = 0 if v == 'M' else (1 if v == 'F' else 2)
            elif col == 'epilepsy':
                v = 1 if v else 0
            vals.append(int(v))
        return torch.tensor(vals)


def build_lazy_datasets(
    engine: TUHEEGEpilepsy,
    *,
    window_len_s: float,
    overlap_pct: float,
    balance_per_subject: bool,
    max_windows_per_subject: Optional[int],
    include_seizures: bool,
    shuffle_windows: bool,
    seed: int,
    splits: dict,
    stratify_by: str,
    mode: str,
    filter_freq: Optional[List[float]],
    target_name: str,
    pick_channels: Optional[List[str]],
    rename_channels: bool,
    set_montage: bool,
    idx_list: Optional[List[str]] = None,
) -> dict:
    """Build ``{split: (WindowDataset, metadata_df)}`` without loading any signal.

    Reuses the engine's plan helpers and RNG order so the window selection and the
    subject-level split match the eager path; only the per-window signal load is
    deferred to ``WindowDataset``.
    """
    if mode in ('ica', 'ica_clean'):
        raise NotImplementedError(
            f"lazy_loading does not support mode='{mode}' yet; use the eager path."
        )

    rng = np.random.RandomState(seed)
    df = engine.descriptions.copy()
    if not include_seizures:
        df = df[df['n_seizure'] == 0]
    if idx_list is not None:
        df = df[df['subject'].isin(idx_list)]
    if df.empty:
        raise ValueError("No data available after filtering.")

    stride_s = window_len_s * (1 - overlap_pct)
    if stride_s <= 0:
        raise ValueError("Overlap must be < 1.0")

    df, limit_per_subject = engine._calculate_limit_per_subject(
        df, window_len_s, stride_s, balance_per_subject,
        max_windows_per_subject=max_windows_per_subject, unit='seconds',
    )
    window_df = engine._generate_windows_list(
        df, window_len_s, stride_s, shuffle_windows, limit_per_subject, rng, unit='seconds',
    )
    logger.info(
        f"lazy_loading: planned {len(window_df)} windows; resident memory is O(batch), "
        f"not O(N) (windows are read on demand)."
    )

    split_map = _assign_splits(window_df, splits, stratify_by, rng)

    target_sfreq = float(df['sfreq'].min())
    target_len = int(window_len_s * target_sfreq)
    if mode == 'brain_ic':
        # Eager harmonization sorts the common channels alphabetically; match that
        # ordering here so the HYDRA features are identical (channel order matters).
        target_channels = sorted(TUHEEGEpilepsy.CANONICAL_REGIONS)
    else:
        target_channels = _scan_common_channels(window_df, engine.montages, rename_channels)
    logger.info(
        f"lazy_loading harmonization targets: {len(target_channels)} channels, "
        f"{target_sfreq:.0f} Hz, {target_len} samples."
    )

    if isinstance(target_name, str):
        target_cols = [target_name]
    elif target_name:
        target_cols = list(target_name)
    else:
        target_cols = ['epilepsy']

    out: dict = {}
    for split_name in splits:
        subjects = {s for s, sp in split_map.items() if sp == split_name}
        split_plan = window_df[window_df['subject'].isin(subjects)].reset_index(drop=True)
        dataset = WindowDataset(
            split_plan, mode, target_channels, target_sfreq, target_len,
            engine.montages, filter_freq, rename_channels, set_montage, pick_channels,
            engine.ica_keep_labels, engine.brain_ic_min_gof, engine.brain_ic_use_dipoles,
            target_cols,
        )
        meta = (
            split_plan.drop(columns=['description_row'])
            if 'description_row' in split_plan.columns
            else split_plan
        )
        out[split_name] = (dataset, meta)
    return out
