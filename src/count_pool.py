"""Count patients / sessions / EDF files eligible before window sampling, per class.

Applies native-rate exclusions, seizure-recording exclusions, require_keep_labels,
an explicit path exclusion list, minimum duration, and the recording-level bad-channel
exclusion used by the lazy loader. Counts the whole eligible pool, not the smaller set
of recordings selected by subject balancing or frozen window manifests. Signal samples
are not loaded, so load failures and flat/non-finite signals are not assessed.

Defaults reproduce the benchmark pool. Run in the container, from code/ (it reads the
corpus descriptions, the -ica_labels.csv files, and the exclusion list)::

    python src/count_pool.py --exclude_recordings_file diagnostics/psd/exclude_recordings.txt

For the current waveform pool (no ICLabel requirement or anomaly exclusion list)::

    python src/count_pool.py --require_keep_labels --exclude_native_sfreqs 400 512 1000 \\
        --interpolate_bad_channels --out logs/waveform_pool_counts.md

Add ``--audit_segments_dir NEW_DIR --split_manifest_dir OLD_DIR`` to enumerate all
complete, non-overlapping segments. The old manifests supply patient assignments
only; their recording paths, selected windows and caps are not reused.
Alternatively, ``--audit_segments_dir NEW_DIR --fresh_splits --split_seed 42``
creates fresh stratified 60/20/20 patient splits after eligibility filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import rootutils
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.data.components.window_dataset import _read_bads  # noqa: E402


def _subject_splits(directory: Path) -> tuple[dict[str, tuple[str, int]], dict[str, dict[str, str]]]:
    """Read patient assignments and source hashes; reject split or label conflicts."""
    assignments: dict[str, tuple[str, int]] = {}
    sources: dict[str, dict[str, str]] = {}
    label_values = {"false": 0, "0": 0, "0.0": 0, "true": 1, "1": 1, "1.0": 1}
    for split in ("train", "val", "test"):
        source = directory / f"windows_{split}.csv"
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        sources[split] = {"path": str(source.resolve()), "sha256": digest}
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"subject", "epilepsy"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{source} must contain subject and epilepsy columns")
            for row in reader:
                subject = str(row["subject"] or "").strip()
                label = label_values.get(str(row["epilepsy"]).strip().lower())
                if not subject or label is None:
                    raise ValueError(f"Invalid subject or epilepsy label in {source}")
                assignment = (split, label)
                if subject in assignments and assignments[subject] != assignment:
                    raise ValueError(f"Conflicting split or label assignments for subject {subject}")
                assignments[subject] = assignment
    return assignments, sources


def _fresh_subject_splits(eligible: pd.DataFrame, seed: int) -> dict[str, tuple[str, int]]:
    """Stratify unique eligible patients 60/20/20, independently of recording order or duration.

    Patient IDs are sorted before each seeded split. Raises ValueError for mixed
    patient labels, nonbinary labels, or a cohort too small for stratification.
    """
    subjects = eligible[["subject", "epilepsy"]].copy()
    subjects["subject"] = subjects["subject"].astype(str)
    if not subjects["epilepsy"].isin([False, True, 0, 1]).all():
        raise ValueError("Eligible epilepsy labels must be binary")
    subjects["epilepsy"] = subjects["epilepsy"].astype(int)
    if (subjects.groupby("subject")["epilepsy"].nunique() > 1).any():
        raise ValueError("An eligible subject carries conflicting epilepsy labels")
    subjects = subjects.drop_duplicates("subject").sort_values("subject").reset_index(drop=True)
    if subjects["epilepsy"].nunique() != 2:
        raise ValueError("Fresh stratification requires eligible subjects in both classes")
    train, held_out = train_test_split(subjects, train_size=0.6, stratify=subjects["epilepsy"], random_state=seed)
    held_out = held_out.sort_values("subject")
    val, test = train_test_split(held_out, test_size=0.5, stratify=held_out["epilepsy"], random_state=seed)
    return {
        str(row.subject): (split, int(row.epilepsy))
        for split, frame in (("train", train), ("val", val), ("test", test))
        for row in frame.itertuples(index=False)
    }


def _audit_segments(
    eligible: pd.DataFrame,
    split_source: Path | int,
    output_dir: Path,
    window_s: float,
    skip_start_s: float,
) -> str:
    """Enumerate the entire eligible pool with patient splits fixed before enumeration.

    ``split_source`` is a manifest directory to preserve assignments, or an integer
    seed for fresh stratified 60/20/20 splits. Writes patient assignments, segment
    manifests, recording/subject/aggregate counts and audit provenance.
    Raises ValueError for invalid metadata or missing/conflicting patient assignments;
    refuses an existing output directory. No signal reader or window sampler is called.
    """
    if not math.isfinite(window_s) or window_s <= 0 or not math.isfinite(skip_start_s) or skip_start_s < 0:
        raise ValueError("Audit window must be positive and offset nonnegative, both finite")
    records = eligible.copy()
    if records.empty:
        raise ValueError("No eligible recordings to audit")
    records["subject"] = records["subject"].astype(str)
    records["path"] = records["path"].map(lambda path: str(Path(path)))
    if records["path"].duplicated().any():
        raise ValueError("Eligible metadata contains duplicate recording paths")
    if not records["epilepsy"].isin([False, True, 0, 1]).all():
        raise ValueError("Eligible epilepsy labels must be binary")
    records["epilepsy"] = records["epilepsy"].astype(int)
    split_details: dict[str, object]
    if isinstance(split_source, Path):
        assignments, sources = _subject_splits(split_source)
        split_details = {"mode": "preserved"}
        split_note = "Source manifests supply patient splits only."
    else:
        assignments = _fresh_subject_splits(records, split_source)
        sources = {}
        split_details = {
            "mode": "fresh_stratified",
            "seed": split_source,
            "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
            "stratify_by": "epilepsy",
            "algorithm": "train_test_split: 60/40, then held-out 50/50; same seed, sorted unique IDs",
            "sklearn_version": sklearn_version,
        }
        split_note = f"Fresh stratified patient splits: 60/20/20, seed {split_source}."
    missing = sorted(set(records["subject"]) - assignments.keys())
    if missing:
        raise ValueError(f"Missing split assignments for {len(missing)} eligible subjects: {missing}")
    labels_match = records["epilepsy"] == records["subject"].map(lambda subject: assignments[subject][1])
    if not labels_match.all():
        mismatched = sorted(records.loc[~labels_match, "subject"].unique())
        raise ValueError(f"Corpus and split manifests disagree on labels for subjects: {mismatched}")
    records["split"] = records["subject"].map(lambda subject: assignments[subject][0])
    durations = records["duration"].astype(float)
    if not (durations.map(math.isfinite) & (durations >= skip_start_s + window_s)).all():
        raise ValueError("Each eligible recording must have finite duration and at least one complete segment")
    records["segments"] = ((durations - skip_start_s) / window_s).map(math.floor)
    records["segment_hours"] = records["segments"] * window_s / 3600.0
    records["unused_tail_s"] = durations - skip_start_s - records["segments"] * window_s
    records = records.sort_values(["split", "subject", "path"])

    # A new directory prevents accidental changes to the reference or training manifests.
    output_dir.mkdir(parents=True, exist_ok=False)
    records[["subject", "epilepsy", "split"]].drop_duplicates("subject").sort_values("subject").to_csv(
        output_dir / "patient_splits.csv", index=False
    )
    fields = ["subject", "path", "start", "end", "epilepsy", "t_id", "window_idx_within_subject"]
    with ExitStack() as stack:
        writers = {
            split: csv.DictWriter(
                stack.enter_context((output_dir / f"windows_{split}.csv").open("w", newline="")), fieldnames=fields
            )
            for split in ("train", "val", "test")
        }
        for writer in writers.values():
            writer.writeheader()
        next_index: dict[str, int] = {}
        for _, recording in records.iterrows():
            subject = str(recording["subject"])
            first_index = next_index.get(subject, 0)
            count = int(recording["segments"])
            for index in range(count):
                writers[recording["split"]].writerow(
                    {
                        "subject": subject,
                        "path": recording["path"],
                        "start": skip_start_s + index * window_s,
                        "end": skip_start_s + (index + 1) * window_s,
                        "epilepsy": int(recording["epilepsy"]),
                        "t_id": recording.get("t_id", "unknown"),
                        "window_idx_within_subject": first_index + index,
                    }
                )
            next_index[subject] = first_index + count
    records[
        ["split", "epilepsy", "subject", "path", "sfreq", "duration", "segments", "segment_hours", "unused_tail_s"]
    ].to_csv(output_dir / "segment_counts_by_recording.csv", index=False)

    summary = [
        "\n## All Complete Segments (No Balancing or Caps)\n",
        f"Window and stride: {window_s:g} s; offset: {skip_start_s:g} s. "
        f"{split_note} Labels: 0=no epilepsy, 1=epilepsy.\n",
        "| Split | Epilepsy | Subjects | Recordings | Segments | Hours |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, group_keys in (
        ("subject", ["split", "epilepsy", "subject"]),
        ("split_class", ["split", "epilepsy"]),
        ("split", ["split"]),
        ("class", ["epilepsy"]),
    ):
        counts = (
            records.groupby(group_keys, sort=True)
            .agg(subjects=("subject", "nunique"), recordings=("path", "nunique"), segments=("segments", "sum"))
            .reset_index()
        )
        counts["segment_hours"] = counts["segments"] * window_s / 3600.0
        if name == "subject":
            counts = counts.drop(columns="subjects")
        counts.to_csv(output_dir / f"segment_counts_by_{name}.csv", index=False)
        if name == "split_class":
            for _, row in counts.iterrows():
                summary.append(
                    f"| {row['split']} | {int(row['epilepsy'])} | {int(row['subjects'])} | "
                    f"{int(row['recordings'])} | {int(row['segments'])} | {row['segment_hours']:.2f} |"
                )
    total_segments = int(records["segments"].sum())
    total_subjects = int(records["subject"].nunique())
    summary.append(
        f"| **All** | **All** | **{total_subjects}** | **{len(records)}** | "
        f"**{total_segments}** | **{total_segments * window_s / 3600.0:.2f}** |\n"
    )
    summary.append("Incomplete tails are excluded. Signal validity is not checked; training settings are unchanged.\n")
    output_hashes = {}
    for path in sorted(output_dir.glob("*.csv")):
        with path.open("rb") as handle:
            output_hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = {
        "scope": "all_metadata_eligible_complete_segments",
        "balance_per_subject": False,
        "max_windows_per_subject": None,
        "window_s": window_s,
        "stride_s": window_s,
        "skip_start_s": skip_start_s,
        "patient_split": split_details,
        "source_manifests": sources,
        "reference_subjects_without_eligible_recordings": sorted(assignments.keys() - set(records["subject"])),
        "subjects": total_subjects,
        "recordings": len(records),
        "segments": total_segments,
        "output_sha256": output_hashes,
        "eligibility_report": "audit_report.md",
        "signal_validity_checked": False,
    }
    (output_dir / "audit_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return "\n".join(summary)


def _counts(df: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    """{'epilepsy'|'no-epilepsy'|'total': (patients, sessions, files)} for a descriptions df.

    A session is a (subject, session, year) folder, matching the corpus directory layout
    (the AAREADME 'sessions' count); a file is one unique EDF path.
    """
    out = {}
    is_epi = df["epilepsy"].astype(bool)
    for label, sub in (("epilepsy", df[is_epi]), ("no-epilepsy", df[~is_epi]), ("total", df)):
        pats = int(sub["subject"].nunique())
        sess = int(sub.drop_duplicates(["subject", "session", "year"]).shape[0])
        files = int(sub["path"].nunique())
        out[label] = (pats, sess, files)
    return out


def _md_table(c: dict[str, tuple[int, int, int]]) -> str:
    rows = [("patients", 0), ("sessions", 1), ("files (EDF)", 2)]
    lines = [
        "| count       | epilepsy (00) | no-epilepsy (01) | total |",
        "| ----------- | ------------- | ---------------- | ----- |",
    ]
    for name, idx in rows:
        e, n, t = c["epilepsy"][idx], c["no-epilepsy"][idx], c["total"][idx]
        lines.append(f"| {name:<11} | {e:<13,} | {n:<16,} | {t:<5,} |")
    return "\n".join(lines)


def main() -> None:
    """Parse the pool filters and print/save metadata counts without loading signals."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(root / "data"), help="Parent of the version folder.")
    parser.add_argument("--version", default="v3.0.0", help="Corpus version subfolder.")
    parser.add_argument(
        "--window_len_min",
        type=float,
        default=2.0,
        help="Training window (min); recordings shorter than this are dropped (default 2).",
    )
    parser.add_argument(
        "--include_seizures",
        action="store_true",
        help="Keep seizure recordings (default: drop them, matching include_seizures=false).",
    )
    parser.add_argument(
        "--require_keep_labels",
        nargs="*",
        default=["brain", "other"],
        help="Keep only recordings with >=1 IC in these ICLabel classes (default: brain other). "
        "Pass with no values to disable.",
    )
    parser.add_argument(
        "--exclude_recordings_file",
        default=None,
        help="Anomaly exclusion list (e.g. diagnostics/psd/exclude_recordings.txt).",
    )
    parser.add_argument(
        "--exclude_native_sfreqs",
        nargs="*",
        type=float,
        default=[],
        help="Exclude these native EDF rates in both classes, before resampling.",
    )
    parser.add_argument(
        "--interpolate_bad_channels",
        action="store_true",
        help="Drop recordings flagged too_many_bad_channels, matching the lazy repair path.",
    )
    parser.add_argument(
        "--skip_start_s",
        type=float,
        default=0.0,
        help="Recording start offset; duration must cover this plus one complete window.",
    )
    parser.add_argument("--out", default=None, help="Also save the filters, funnel and tables as Markdown.")
    parser.add_argument("--audit_segments_dir", type=Path, help="New directory for an uncapped complete-segment audit.")
    split_choice = parser.add_mutually_exclusive_group()
    split_choice.add_argument(
        "--split_manifest_dir", type=Path, help="Use only patient splits/labels from its windows_*.csv."
    )
    split_choice.add_argument(
        "--fresh_splits", action="store_true", help="Create stratified 60/20/20 splits of eligible subjects."
    )
    parser.add_argument("--split_seed", type=int, default=42, help="Fresh patient split seed (default 42).")
    parser.add_argument("--expected_recordings", type=int, help="Fail if the filtered recording count differs.")
    parser.add_argument("--expected_subjects", type=int, help="Fail if the filtered patient count differs.")
    args = parser.parse_args()
    if bool(args.audit_segments_dir) != bool(args.split_manifest_dir or args.fresh_splits):
        parser.error("--audit_segments_dir requires exactly one of --split_manifest_dir or --fresh_splits")
    if not 0 <= args.split_seed < 2**32:
        parser.error("--split_seed must be in [0, 2**32)")
    if not math.isfinite(args.window_len_min) or args.window_len_min <= 0:
        parser.error("--window_len_min must be finite and positive")
    if not math.isfinite(args.skip_start_s) or args.skip_start_s < 0:
        parser.error("--skip_start_s must be finite and nonnegative")
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.exclude_native_sfreqs):
        parser.error("--exclude_native_sfreqs must contain finite positive rates")

    tuh = TUHEEGEpilepsy(data_dir=args.data_dir, version=args.version, add_annotations=True)
    df0 = tuh.descriptions.copy()

    funnel = [("full corpus", len(df0))]
    df = df0
    excluded_rates = sorted(set(args.exclude_native_sfreqs))
    if excluded_rates:
        native_sfreq = df["sfreq"].astype(float)
        if not (native_sfreq.map(math.isfinite) & (native_sfreq > 0)).all():
            raise ValueError("Native sfreq metadata must be finite and positive for every recording")
        df = df[~native_sfreq.isin(excluded_rates)]
        funnel.append((f"after excluding native rates {excluded_rates} Hz", len(df)))
    if not args.include_seizures:
        df = df[df["n_seizure"] == 0]
        funnel.append(("after seizure drop (n_seizure == 0)", len(df)))
    if args.require_keep_labels:
        df = TUHEEGEpilepsy._filter_recordings_by_labels(df, args.require_keep_labels)
        funnel.append((f"after require_keep_labels={args.require_keep_labels}", len(df)))
    if args.exclude_recordings_file:
        lines = [ln.strip() for ln in Path(args.exclude_recordings_file).read_text().splitlines()]
        paths = [ln for ln in lines if ln and not ln.startswith("#")]
        df = TUHEEGEpilepsy._filter_excluded_paths(df, {str(Path(p)) for p in paths})
        funnel.append((f"after anomaly exclusion list ({len(paths)} paths)", len(df)))
    win_s = args.window_len_min * 60.0
    min_duration_s = args.skip_start_s + win_s
    df = df[df["duration"].astype(float) >= min_duration_s]
    funnel.append((f"after duration >= {min_duration_s:g}s (offset + complete window)", len(df)))
    sidecar_note = ""
    if args.interpolate_bad_channels:
        paths = df["path"].astype(str)
        missing = sum(not Path(path.replace(".edf", "-bads.json")).is_file() for path in paths.unique())
        too_many = paths.map(lambda path: _read_bads(path)[2])
        df = df[~too_many.astype(bool)]
        funnel.append(("after excluding unrepairable bad-channel recordings", len(df)))
        sidecar_note = (
            f"\nMissing bad-channel sidecars among duration-eligible recordings: {missing}. "
            "Missing/unreadable sidecars do not exclude recordings, matching the loader; "
            "unreadable sidecars also produce a warning.\n"
        )

    before, after = _counts(df0), _counts(df)
    if args.expected_recordings is not None and after["total"][2] != args.expected_recordings:
        parser.error(f"Expected {args.expected_recordings} eligible recordings, found {after['total'][2]}")
    if args.expected_subjects is not None and after["total"][0] != args.expected_subjects:
        parser.error(f"Expected {args.expected_subjects} eligible subjects, found {after['total'][0]}")
    funnel_text = "\n".join(f"{n:6d}  {name}" for name, n in funnel)
    block = (
        "# Recording Pool Counts\n\n"
        f"Corpus: `{Path(args.data_dir) / args.version}`. "
        f"Native-rate exclusions (Hz): `{excluded_rates}`; include seizures: `{args.include_seizures}`; "
        f"required IC labels: `{args.require_keep_labels or None}`; "
        f"path exclusion list: `{args.exclude_recordings_file}`; "
        f"window: `{win_s:g}` s; start offset: `{args.skip_start_s:g}` s; "
        f"bad-channel exclusion: `{args.interpolate_bad_channels}`.\n\n"
        "Metadata eligibility only, before balancing/window sampling; signal validity is not checked.\n\n"
        "## Filter Funnel (Recordings Remaining)\n\n```text\n"
        + funnel_text
        + "\n```\n"
        + sidecar_note
        + "\n## Before Any Exclusion\n\n"
        + _md_table(before)
        + "\n\n## After Exclusions (Eligible Pool)\n\n"
        + _md_table(after)
        + "\n"
    )
    if args.audit_segments_dir:
        split_source = args.split_manifest_dir if args.split_manifest_dir is not None else args.split_seed
        block += _audit_segments(df, split_source, args.audit_segments_dir, win_s, args.skip_start_s)
        (args.audit_segments_dir / "audit_report.md").write_text(block)
    print(block)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(block)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
