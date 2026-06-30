"""Count patients / sessions / EDF files in the training pool, before and after the
recording exclusions, per epilepsy class.

Mirrors the pre-windowing filters the engine applies in _load_balanced_windows /
build_lazy_datasets, in the same order: drop seizure recordings (include_seizures=false),
require_keep_labels, the anomaly exclusion list (exclude_recordings_file), and the
sub-window-length drop (duration < window_len_min * 60 s yields no window). Prints the
per-filter funnel and two markdown tables (before / after) matching the Dataset table in
recording_exclusion_and_bipolar_icaclean_benchmark.md.

Defaults reproduce the benchmark pool. Run in the container, from code/ (it reads the
corpus descriptions, the -ica_labels.csv files, and the exclusion list)::

    python src/count_pool.py --exclude_recordings_file diagnostics/psd/exclude_recordings.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402


def _counts(df) -> dict:
    """{'epilepsy'|'no-epilepsy'|'total': (patients, sessions, files)} for a descriptions df.

    A session is a (subject, session, year) folder, matching the corpus directory layout
    (the AAREADME 'sessions' count); a file is one EDF recording (one descriptions row).
    """
    out = {}
    is_epi = df["epilepsy"].astype(bool)
    for label, sub in (("epilepsy", df[is_epi]), ("no-epilepsy", df[~is_epi]), ("total", df)):
        pats = int(sub["subject"].nunique())
        sess = int(sub.drop_duplicates(["subject", "session", "year"]).shape[0])
        files = int(sub.shape[0])
        out[label] = (pats, sess, files)
    return out


def _md_table(c: dict) -> str:
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(root / "data"), help="Parent of the version folder.")
    parser.add_argument("--version", default="v3.0.0", help="Corpus version subfolder.")
    parser.add_argument("--window_len_min", type=float, default=2.0,
                        help="Training window (min); recordings shorter than this are dropped (default 2).")
    parser.add_argument("--include_seizures", action="store_true",
                        help="Keep seizure recordings (default: drop them, matching include_seizures=false).")
    parser.add_argument("--require_keep_labels", nargs="*", default=["brain", "other"],
                        help="Keep only recordings with >=1 IC in these ICLabel classes (default: brain other). "
                        "Pass with no values to disable.")
    parser.add_argument("--exclude_recordings_file", default=None,
                        help="Anomaly exclusion list (e.g. diagnostics/psd/exclude_recordings.txt).")
    parser.add_argument("--out", default=None, help="Also write the two markdown tables to this file.")
    args = parser.parse_args()

    tuh = TUHEEGEpilepsy(data_dir=args.data_dir, version=args.version, add_annotations=True)
    df0 = tuh.descriptions.copy()

    funnel = [("full corpus", len(df0))]
    df = df0
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
    df = df[df["duration"].astype(float) >= win_s]
    funnel.append((f"after duration >= {win_s:.0f}s (>= {args.window_len_min:g} min window)", len(df)))

    before, after = _counts(df0), _counts(df)
    print("\nFunnel (recordings remaining after each filter):")
    for name, n in funnel:
        print(f"  {n:6d}  {name}")
    block = (
        "\n### Before any exclusion (full corpus)\n\n" + _md_table(before)
        + "\n\n### After exclusions (benchmark pool)\n\n" + _md_table(after) + "\n"
    )
    print(block)
    if args.out:
        Path(args.out).write_text(block)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
