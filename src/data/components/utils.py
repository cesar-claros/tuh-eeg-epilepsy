#%%
from __future__ import annotations
# from datetime import datetime, timezone
from typing import Iterable, Tuple
from unittest import mock
from pathlib import Path
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from braindecode.datasets.base import BaseConcatDataset, RawDataset

import glob
import os
import re
import warnings

import mne
import numpy as np
import pandas as pd
#%%
def read_edf_header(file_path):
    f = open(file_path, "rb")
    header = f.read(88)
    f.close()
    return header


def parse_age_and_gender_from_edf_header(file_path):
    header = read_edf_header(file_path)
    # bytes 8 to 88 contain ascii local patient identification
    # see https://www.teuniz.net/edfbrowser/edf%20format%20description.html
    patient_id = header[8:].decode("ascii")
    age = -1
    found_age = re.findall(r"Age:(\d+)", patient_id)
    if len(found_age) == 1:
        age = int(found_age[0])
    gender = "X"
    found_gender = re.findall(r"\s([F|M])\s", patient_id)
    if len(found_gender) == 1:
        gender = found_gender[0]
    return age, gender

def parse_montage_file(path: str | Path) -> pd.DataFrame:
    rows = []
    channels = []
    in_section = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip empty/comment lines
            if not line or line.startswith("#"):
                continue

            if line == "[Montage]":
                in_section = True
                continue

            if not in_section:
                continue

            # expected format:
            # montage =  0, FP1-F7: EEG FP1-LE  --  EEG F7-LE
            m = re.match(r"^montage\s*=\s*(\d+),\s*([^:]+):\s*(.+?)\s*--\s*(.+)$", line)
            if not m:
                # skip any unexpected lines in the section
                continue

            montage_idx = int(m.group(1))
            bipole = m.group(2).strip()
            pole_a = m.group(3).strip()
            pole_b = m.group(4).strip()

            rows.append(
                {
                    "montage": montage_idx,
                    "bipole": bipole,
                    "pole A": pole_a,
                    "pole B": pole_b,
                }
            )
            
    montage_info = pd.DataFrame(rows)
    channels = pd.unique(montage_info[['pole A','pole B']].values.ravel('K'))
    return {'montage_info': montage_info, 'channels':channels}
#%%
def build_tree(
    # self,
    root: Path,
    max_depth: int | None = None,
    include_hidden: bool = False,
) -> list[str]:
    """
    Returns a list of lines representing the folder/file structure under `root`.
    """
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        # If a file is provided, just show it
        return [root.name]

    def is_hidden(p: Path) -> bool:
        # Cross-platform-ish: dotfiles on Unix, and dot-prefix on Windows too.
        return p.name.startswith(".")

    lines: list[str] = [root.name]

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth >= max_depth:
            return

        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            lines.append(prefix + "└── [Permission denied]")
            return

        if not include_hidden:
            entries = [e for e in entries if not is_hidden(e)]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            lines.append(prefix + branch + entry.name)

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension, depth + 1)

    walk(root, "", 0)
    return lines

def get_metadata(
    # self,
    root: Path,
    add_montages: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    """
    Generates the tree style directory listing.
    """
    METADATA_FILE = 'metadata_v00r.xlsx'
    TCP_AR = '01_tcp_ar_montage.txt'
    TCP_LE = '02_tcp_le_montage.txt'
    TCP_AR_A = '03_tcp_ar_a_montage.txt'
    TCP_LE_A = '04_tcp_le_a_montage.txt'
    root = root.expanduser().resolve()
    metadata_df = pd.read_excel(root/METADATA_FILE)
    if not add_montages:
        return metadata_df
    montages = {
        '01_tcp_ar': parse_montage_file(root/TCP_AR),
        '02_tcp_le': parse_montage_file(root/TCP_LE),
        '03_tcp_ar_a': parse_montage_file(root/TCP_AR_A),
        '04_tcp_le_a': parse_montage_file(root/TCP_LE_A)
        }
    return metadata_df, montages


def generate_tree_file(
    # self,
    root: Path,
    max_depth: int | None = None,
    include_hidden: bool = False,
    output_path: Path | None = None,
) -> None:
    """
    Generates the tree style directory listing.
    """
    # Assigns Path for output file
    if output_path is None:
        filename = root.name+'.txt'
        output_path = root.parent/filename
    # Verify file existis
    if output_path.exists():
        logger.info(f'{output_path} file already exists.' )
    else:
        root = root.expanduser().resolve()
        lines = build_tree(root, max_depth=max_depth, include_hidden=include_hidden)
        text = "\n".join(lines)
        # output_path = f'../../data/v3.0.0/0{epilepsy_flag}_no_epilepsy_tree.txt'
        out = Path(output_path).expanduser().resolve()
        out.write_text(text, encoding="utf-8")

def parse_tree_file(tree_path: Path):
    """
    Yield tuples of (root_name, subject_id, session_id, year, montage, filename).
    """
    lines = tree_path.read_text(encoding="utf-8").splitlines()
    tree_line_re = re.compile(r"^(?P<prefix>(?:│   |    )*)(?:├── |└── )?(?P<name>.+)$")

    root_name = lines[0].strip()
    stack = [root_name]  # level 0
    current_subject = None
    current_session = None
    current_year = None
    current_montage = None

    for raw in tqdm(lines[1:], desc=f'Parsing tree-style directory listing {root_name}'):
        m = tree_line_re.match(raw)
        if not m:
            continue
        prefix = m.group("prefix")
        name = m.group("name").strip()

        depth = len(prefix) // 4 + 1  # depth relative to root
        # trim stack to depth
        stack = stack[:depth]
        stack.append(name)

        if depth == 1:  # subject
            current_subject = name
        elif depth == 2:  # session
            session_id, year = name.split("_", 1)
            current_session = session_id
            current_year = year
        elif depth == 3:  # montage
            current_montage = name
        elif depth == 4:  # files
            yield (
                root_name,
                current_subject,
                current_session,
                current_year,
                current_montage,
                name,
            )

def read_duration_annotations_from_csv(
    csv_path: Path,
    n_header: int = 5
) -> Tuple[float,pd.DataFrame]:
    """
    Reads the duration from the CSV third line:
    '# duration = 1214.0000 secs'
    The 4th segment (0-index 3) is the duration.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        duration = None
        for i, line in enumerate(f):
            if i == 2:  # third line
                assert 'duration' in line, f"No duration line in {csv_path}"
                parts = line.strip().split(' ')
                duration = float(parts[3]) 
                # return duration
        return duration, pd.read_csv(csv_path, sep=',', header=n_header)
    raise ValueError(f"No duration line in {csv_path}")


def extract_info_annotations(
    dataset_paths:list, 
    add_annotations:bool = False,
    recording_ids:list = None,
) -> pd.DataFrame|Tuple[pd.DataFrame, ...]:
    """Extract information from tree structure and annotations from csv files.
        dataset_paths = 
        [Path('/data_dir/version/00_epilepsy'), Path('/data_dir/version/01_no_epilepsy')]
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
    dataset_dicts = {p.name:p for p in dataset_paths}
    for path in dataset_paths:
        filename = path.name+'.txt'
        tree_file = path.parent/filename
        for root_name, subject, session, year, montage, filename in parse_tree_file(tree_file):
            # Extract t### from filename
            t_match = re.search(r"_t(\d+)\.edf$", filename)
            t_id = f"t{t_match.group(1)}" if t_match else filename
            edf_path = dataset_dicts[root_name] / subject / f"{session}_{year}" / montage / filename
            records.append(
                {
                    "root": root_name,
                    "subject": subject,
                    "session": session,
                    "year": year,
                    "t_id" : t_id,
                    "montage": montage,
                    "epilepsy": path.name=="00_epilepsy",
                    "filename": filename,
                    "path": edf_path,
                }
            )
    df = pd.DataFrame(records)
    # -------------------------
    # DataFrame: number of time series (count of .edf)
    # -------------------------
    edf_df = df[df["path"].apply(lambda x:x.name.endswith(".edf"))].copy()
    # series_count_df = (
    #     edf_df.groupby(["subject", "session"])
    #         .size()
    #         .unstack()
    # )
    if add_annotations:
        logger.info("Extracting annotations ...")
        if recording_ids is not None:
            if not isinstance(recording_ids, Iterable):
                # Assume it is an integer specifying number
                # of recordings to load
                recording_ids = range(recording_ids)
            edf_df = edf_df.iloc[recording_ids]
        # -------------------------
        # DataFrame: duration
        # -------------------------
        # Map edf -> csv and read duration from actual dataset path
        duration_entries = []
        annotated_entries = []
        annotated_bi_entries = []
        n_csv_files = edf_df.shape[0]
        for _, row in tqdm(edf_df.iterrows(), desc="Reading .csv files", total=n_csv_files):
            root_name = row["root"]
            epilepsy_flag = root_name=='00_epilepsy'
            subject = row["subject"]
            session = row["session"]
            montage = row["montage"]
            year = row["year"]
            edf_name = row["filename"]
            csv_name = edf_name.replace(".edf", ".csv")
            csv_bi_name = edf_name.replace(".edf", ".csv_bi")

            # Build real path to CSV in your dataset
            edf_path = dataset_dicts[root_name] / subject / f"{session}_{row['year']}" / montage / edf_name
            csv_path = dataset_dicts[root_name] / subject / f"{session}_{row['year']}" / montage / csv_name
            csv_bi_path = dataset_dicts[root_name] / subject / f"{session}_{row['year']}" / montage / csv_bi_name
            duration, annot = read_duration_annotations_from_csv(csv_path)
            _, annot_bi = read_duration_annotations_from_csv(csv_bi_path)
            annot[['root','subject','session','montage','path']] = [root_name,subject,session,montage,edf_name]
            annot_bi[['root','subject','session','montage','path']] = [root_name,subject,session,montage,edf_name]
            # Extract t### from filename
            t_match = re.search(r"_t(\d+)\.edf$", edf_name)
            t_id = f"t{t_match.group(1)}" if t_match else edf_name

            duration_entries.append(
                {
                    "subject": subject,
                    "session": session,
                    "year": year,
                    "montage": montage,
                    "t_id": t_id,
                    "epilepsy": epilepsy_flag,
                    "duration": duration,
                    "n_seizure": annot_bi['label'].isin(['seiz']).sum(),
                    # "filename": edf_name,
                    "path": edf_path,
                }
            )
            annotated_entries.append(annot)
            annotated_bi_entries.append(annot_bi)

        duration_df_raw = pd.DataFrame(duration_entries)
        # duration_df = (
        #     duration_df_raw.groupby(["subject", "session"])
        #                 .apply(lambda g: dict(zip(g["t_id"], g["duration"])))
        #                 .unstack()
        # )
        annotated_df = pd.concat(annotated_entries)
        annotated_bi_df = pd.concat(annotated_bi_entries)
        return duration_df_raw, annotated_df, annotated_bi_df 
    # else:
    return edf_df.drop(columns=['root','filename'])