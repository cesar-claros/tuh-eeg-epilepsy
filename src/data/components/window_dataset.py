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

- Supports ``mode='raw'``, ``'ica_clean'``, ``'brain_ic'`` and ``'ic_bag'``.
  ``'ica_clean'`` back-projects each window keeping only the ICs whose ICLabel is
  in ``ica_keep_labels`` (set ``[brain]`` for a brain-IC-only reconstruction) and
  then runs the same sensor-space pipeline as ``'raw'``, so the two share the SAME
  channel target and differ only by denoising. ``'ica'`` (raw IC sources) still
  raises ``NotImplementedError``; use the eager path for it.
- Harmonization targets (the common channel set and the resample rate) are fixed
  *up front* (the eager path derives them from the loaded batch). The channel set
  is found by a header-only scan of the unique files in the plan, matching the
  eager intersection; the resample rate is the minimum sfreq. ``'ica_clean'`` uses
  the same montage-based sensor target as ``'raw'`` (the montage channels are a
  subset of the channels the ICA was fit on, so they survive the back-projection);
  this keeps it in raw's channel space and may differ slightly from the eager
  ``ica_clean`` target, which intersects all post-ICA channels.
- A window that fails to load is zero-filled (not dropped), so ``__len__`` is
  fixed and the per-split metadata stays aligned with the feature order. For
  ``raw`` this never triggers (no ICA dependency), so lazy and eager produce
  identical windows; for ``brain_ic`` / ``ic_bag`` / ``ica_clean`` a failed window
  (missing ICA files, alignment failure) becomes a zero row instead of being
  dropped (a small, documented difference from eager).

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
            raw = TUHEEGEpilepsy._read_raw_edf(path, include=include, preload=False)
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
        ic_bag_max_k: int = 20,
        ic_bag_sign_normalize: bool = True,
        ic_bag_rank_by: str = 'variance',
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
        self.ic_bag_max_k = ic_bag_max_k
        self.ic_bag_sign_normalize = ic_bag_sign_normalize
        self.ic_bag_rank_by = ic_bag_rank_by

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
        # brain_ic / ic_bag build IC series; ica_clean back-projects via the saved
        # ICA. All three need every channel the ICA was fit on, so read all of them.
        if self.mode in ('brain_ic', 'ic_bag', 'ica_clean'):
            include = None
        else:
            include = list(self.montages[desc['montage']]['channels'])
        try:
            raw = TUHEEGEpilepsy._read_raw_edf(desc['path'], include=include, preload=False)
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
            elif self.mode == 'ic_bag':
                out = TUHEEGEpilepsy._ic_bag_sources(
                    raw, desc['path'], self.ica_keep_labels,
                    max_k=self.ic_bag_max_k, sign_normalize=self.ic_bag_sign_normalize,
                    rank_by=self.ic_bag_rank_by,
                )
                if out is None:
                    return None
                data, ch_names = out
                sfreq = raw.info['sfreq']
            else:
                # ica_clean: back-project keeping only the ICs whose ICLabel is in
                # ica_keep_labels, yielding a sensor-space raw; then run the exact
                # same pipeline as 'raw' so both harmonize to the same channels.
                if self.mode == 'ica_clean':
                    raw = TUHEEGEpilepsy._apply_ica_cleaning(
                        raw, desc['path'], self.ica_keep_labels
                    )
                    if raw is None:
                        return None
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


def _plan_from_csv(engine: TUHEEGEpilepsy, csv_path: str) -> pd.DataFrame:
    """Rebuild a window plan from a dumped ``windows_*.csv``, re-attaching descriptions.

    Each window's recording is looked up in the engine's descriptions (by path) to
    recover the full description row the loader needs (montage, sfreq, ...), so a
    fixed window set can be reused across runs / signal modes for reproducibility.
    """
    plan_csv = pd.read_csv(csv_path)
    desc_by_path = {str(row['path']): row for _, row in engine.descriptions.iterrows()}
    has_idx = 'window_idx_within_subject' in plan_csv.columns
    records = []
    missing = 0
    for _, r in plan_csv.iterrows():
        desc = desc_by_path.get(str(r['path']))
        if desc is None:
            missing += 1
            continue
        records.append({
            'subject': desc['subject'],
            'path': desc['path'],
            'start': float(r['start']),
            'end': float(r['end']),
            'epilepsy': desc['epilepsy'],
            't_id': desc.get('t_id', 'unknown'),
            'window_idx_within_subject': (
                int(r['window_idx_within_subject']) if has_idx else len(records)
            ),
            'description_row': desc,
        })
    if missing:
        logger.warning(
            f"{missing} of {len(plan_csv)} rows in {csv_path} had no matching recording; skipped."
        )
    if not records:
        raise ValueError(f"No usable windows loaded from {csv_path}.")
    return pd.DataFrame(records)


def _build_lazy_from_csv(
    engine: TUHEEGEpilepsy,
    window_csvs: dict,
    *,
    window_len_s: float,
    mode: str,
    filter_freq: Optional[List[float]],
    target_name: str,
    pick_channels: Optional[List[str]],
    rename_channels: bool,
    set_montage: bool,
    ic_bag_max_k: int,
    ic_bag_sign_normalize: bool,
    ic_bag_rank_by: str,
) -> dict:
    """Build per-split lazy datasets from fixed window CSVs (one path per split).

    The split is given by which CSV a window came from (no re-splitting). Targets
    (channel set, resample rate) are derived from the recordings the CSVs name, so
    runs sharing the same CSVs get identical windows and targets.
    """
    plans = {sp: _plan_from_csv(engine, path) for sp, path in window_csvs.items()}
    all_rows = pd.concat(plans.values(), ignore_index=True)
    target_sfreq = float(min(d['sfreq'] for d in all_rows['description_row']))
    target_len = int(window_len_s * target_sfreq)
    if mode == 'brain_ic':
        target_channels = sorted(TUHEEGEpilepsy.CANONICAL_REGIONS)
    elif mode == 'ic_bag':
        target_channels = sorted(f'ic_{i}' for i in range(ic_bag_max_k))
    else:
        target_channels = _scan_common_channels(all_rows, engine.montages, rename_channels)
    logger.info(
        f"lazy_loading (fixed window CSVs): {len(all_rows)} windows across {len(plans)} splits; "
        f"{len(target_channels)} channels, {target_sfreq:.0f} Hz, {target_len} samples."
    )
    if isinstance(target_name, str):
        target_cols = [target_name]
    elif target_name:
        target_cols = list(target_name)
    else:
        target_cols = ['epilepsy']

    out: dict = {}
    for split_name, plan in plans.items():
        dataset = WindowDataset(
            plan, mode, target_channels, target_sfreq, target_len,
            engine.montages, filter_freq, rename_channels, set_montage, pick_channels,
            engine.ica_keep_labels, engine.brain_ic_min_gof, engine.brain_ic_use_dipoles,
            target_cols,
            ic_bag_max_k=ic_bag_max_k,
            ic_bag_sign_normalize=ic_bag_sign_normalize,
            ic_bag_rank_by=ic_bag_rank_by,
        )
        out[split_name] = (dataset, plan.drop(columns=['description_row']))
    return out


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
    ic_bag_max_k: int = 20,
    ic_bag_sign_normalize: bool = True,
    ic_bag_rank_by: str = 'variance',
    window_csvs: Optional[dict] = None,
) -> dict:
    """Build ``{split: (WindowDataset, metadata_df)}`` without loading any signal.

    Reuses the engine's plan helpers and RNG order so the window selection and the
    subject-level split match the eager path; only the per-window signal load is
    deferred to ``WindowDataset``.
    """
    if mode == 'ica':
        raise NotImplementedError(
            f"lazy_loading does not support mode='{mode}' yet; use the eager path."
        )

    # Fixed window set (reproducibility / identical windows across signal modes):
    # load the plan from the provided per-split CSVs instead of generating one.
    if window_csvs is not None:
        return _build_lazy_from_csv(
            engine, window_csvs,
            window_len_s=window_len_s, mode=mode, filter_freq=filter_freq,
            target_name=target_name, pick_channels=pick_channels,
            rename_channels=rename_channels, set_montage=set_montage,
            ic_bag_max_k=ic_bag_max_k, ic_bag_sign_normalize=ic_bag_sign_normalize,
            ic_bag_rank_by=ic_bag_rank_by,
        )

    rng = np.random.RandomState(seed)
    df = engine.descriptions.copy()
    if not include_seizures:
        df = df[df['n_seizure'] == 0]
    if idx_list is not None:
        df = df[df['subject'].isin(idx_list)]
    if engine.require_keep_labels is not None:
        df = TUHEEGEpilepsy._filter_recordings_by_labels(df, engine.require_keep_labels)
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
    elif mode == 'ic_bag':
        # Positional IC slots; the ICBagTransformer pools over them, so order does
        # not matter (sorted to match the eager alphabetical harmonization).
        target_channels = sorted(f'ic_{i}' for i in range(ic_bag_max_k))
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
            ic_bag_max_k=ic_bag_max_k,
            ic_bag_sign_normalize=ic_bag_sign_normalize,
            ic_bag_rank_by=ic_bag_rank_by,
        )
        meta = (
            split_plan.drop(columns=['description_row'])
            if 'description_row' in split_plan.columns
            else split_plan
        )
        out[split_name] = (dataset, meta)
    return out
