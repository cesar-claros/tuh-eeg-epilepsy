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
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.data.components.window_dataset import _read_bads  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd


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
    args = parser.parse_args()
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
    print(block)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(block)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
