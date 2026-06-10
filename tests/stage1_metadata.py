"""Stage 1 - Corpus metadata extraction.

INPUT  : a dataset directory `<data_dir>/<version>` (the unpacked TUH EEG
         Epilepsy corpus, with 00_epilepsy/, 01_no_epilepsy/, DOCS/).
OUTPUT : the `descriptions` DataFrame built by `TUHEEGEpilepsy.__init__`
         (one row per .edf: subject, session, montage, sfreq, duration,
         n_seizure, epilepsy, path, ...) plus the parsed montage definitions.

This is the table every later stage filters and windows over. Requires the real
corpus to be present; if it is missing the script explains and exits cleanly.

Run:
    uv run python tests/stage1_metadata.py --limit 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)


def _banner(t: str) -> None:
    print("\n" + "=" * 88 + "\n" + t + "\n" + "=" * 88)


def _sec(t: str) -> None:
    print("\n" + t + "\n" + "-" * max(len(t), 8))


def _kv(k: str, v) -> None:
    print(f"  {k:<28}: {v}")


def _desc_df(name: str, df, n: int = 8) -> None:
    _sec(f"[dataframe] {name}")
    _kv("shape", df.shape)
    _kv("columns", list(df.columns))
    if len(df):
        print(df.head(n).to_string())


def _need(import_fn, pkgs: str):
    try:
        return import_fn()
    except ImportError as exc:
        _sec("missing dependency")
        _kv("error", exc)
        print(f"  # needs: {pkgs}")
        print("  # run `uv sync` in code/ (also add aeon/mne/braindecode/"
              "mne-icalabel/scikit-learn/joblib/pandas if not declared)")
        return None


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--version", default="v3.0.0")
    p.add_argument("--limit", type=int, default=None,
                   help="restrict to the first N recordings (fast smoke test; "
                        "note: the first records are all epilepsy-class)")
    p.add_argument("--no-annotations", action="store_true",
                   help="skip reading .csv annotations (no duration / n_seizure)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 1 - Corpus metadata extraction")

    def _imports():
        from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy
        return TUHEEGEpilepsy

    TUHEEGEpilepsy = _need(_imports, "braindecode, mne, mne-icalabel, pandas")
    if TUHEEGEpilepsy is None:
        return 1

    ds = Path(args.data_dir) / args.version
    needed = ["00_epilepsy", "01_no_epilepsy", "DOCS"]
    present = ds.exists() and all((ds / s).is_dir() for s in needed)

    _sec("INPUT")
    _kv("dataset path", ds)
    _kv("add_annotations", not args.no_annotations)
    _kv("limit (recording_ids)", args.limit)

    if not present:
        _sec("dataset not available")
        _kv("looked in", ds)
        print("  # The EDF class folders (00_epilepsy/, 01_no_epilepsy/) and DOCS/")
        print("  # are required here but are missing. Run on a machine/HPC where")
        print("  # the corpus is unpacked under data/<version>/.")
        return 0

    tuh = TUHEEGEpilepsy(
        data_dir=args.data_dir,
        version=args.version,
        recording_ids=args.limit,
        add_annotations=not args.no_annotations,
    )

    _banner("OUTPUT")
    desc = tuh.descriptions
    _desc_df("tuh.descriptions", desc)

    if "epilepsy" in desc.columns:
        _sec("[distribution] epilepsy")
        print(desc["epilepsy"].value_counts(dropna=False).to_string())
    if "sfreq" in desc.columns:
        _sec("[distribution] sfreq (Hz)")
        print(desc["sfreq"].value_counts(dropna=False).to_string())
    if "duration" in desc.columns:
        _sec("[stats] duration (s)")
        print(desc["duration"].describe().to_string())
    if "n_seizure" in desc.columns:
        _sec("[stats] n_seizure per file")
        _kv("files with >=1 seizure", int((desc["n_seizure"] > 0).sum()))
        _kv("total seizures", int(desc["n_seizure"].sum()))

    _sec("[montages] tuh.montages")
    _kv("montage keys", list(tuh.montages.keys()))
    first = next(iter(tuh.montages))
    _kv(f"{first} -> n_channels", len(tuh.montages[first]["channels"]))
    _kv(f"{first} -> channels", list(tuh.montages[first]["channels"]))

    if not args.no_annotations:
        _kv("annotated_df shape", tuh.annotated_df.shape)
        _kv("annotated_bi_df shape", tuh.annotated_bi_df.shape)

    _sec("-> flows to Stage 2")
    print("  # `descriptions` is filtered (drop seizures, pick subjects) and")
    print("  # windowed by `load_data` / `_load_balanced_windows` next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
