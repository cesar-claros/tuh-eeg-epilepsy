"""Stage 2 - Balanced windowing and subject-level splitting.

INPUT  : the `descriptions` table from Stage 1, plus windowing parameters
         (window length, overlap, split ratios, stratification).
OUTPUT : the dict returned by `TUHEEGEpilepsy.load_data(window_len_s=..., splits=...)`
         -> {split_name: (X, y, meta_df)} where
           X    : float32 tensor (n_windows, n_channels, n_timepoints)
           y    : int tensor (n_windows, n_targets)   (epilepsy encoded 0/1)
           meta : DataFrame aligned row-for-row with X (subject, start, end, ...)

It also checks the central correctness property: subjects do NOT leak across
train/val/test. Requires the real corpus.

Run:
    uv run python tests/stage2_windowing.py --subjects-per-class 3 --window-len-s 60
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


def _desc_tensor(name: str, t) -> None:
    import torch

    _sec(f"[tensor] {name}")
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        _kv("value", "empty / not a tensor")
        return
    tf = t.detach().float()
    _kv("shape", tuple(t.shape))
    _kv("dtype", t.dtype)
    _kv("min/max", f"{tf.min():.4g} / {tf.max():.4g}")
    _kv("mean/std", f"{tf.mean():.4g} / {tf.std():.4g}")


def _need(import_fn, pkgs: str):
    try:
        return import_fn()
    except ImportError as exc:
        _sec("missing dependency")
        _kv("error", exc)
        print(f"  # needs: {pkgs}; run `uv sync` in code/ (add aeon/mne/braindecode/"
              "scikit-learn/joblib/pandas if not declared)")
        return None


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--version", default="v3.0.0")
    p.add_argument("--subjects-per-class", type=int, default=3,
                   help="load windows for K subjects per class (0 = all). Keeps "
                        "both classes present so the stratified split is meaningful.")
    p.add_argument("--window-len-s", type=float, default=60.0)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--splits", default="0.6,0.2,0.2", help="train,val,test ratios")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 2 - Balanced windowing and subject-level splitting")

    def _imports():
        from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy
        return TUHEEGEpilepsy

    TUHEEGEpilepsy = _need(_imports, "braindecode, mne, mne-icalabel, torch, pandas")
    if TUHEEGEpilepsy is None:
        return 1

    ds = Path(args.data_dir) / args.version
    needed = ["00_epilepsy", "01_no_epilepsy", "DOCS"]
    if not (ds.exists() and all((ds / s).is_dir() for s in needed)):
        _sec("dataset not available")
        _kv("looked in", ds)
        print("  # Requires the unpacked TUH corpus. Run where data/<version>/ exists.")
        return 0

    ratios = [float(x) for x in args.splits.split(",")]
    splits = dict(zip(["train", "val", "test"], ratios))

    _sec("INPUT")
    _kv("window_len_s", args.window_len_s)
    _kv("overlap_pct", args.overlap)
    _kv("splits", splits)
    _kv("subjects_per_class", args.subjects_per_class or "all")
    _kv("balance_per_subject", True)
    _kv("include_seizures", False)
    _kv("stratify_by", "epilepsy")

    tuh = TUHEEGEpilepsy(data_dir=args.data_dir, version=args.version, add_annotations=True)

    # Build a balanced subject subset so loading stays small and both classes appear.
    idx_list = None
    if args.subjects_per_class:
        df = tuh.descriptions
        epi = df[df["epilepsy"]]["subject"].drop_duplicates().tolist()[: args.subjects_per_class]
        non = df[~df["epilepsy"]]["subject"].drop_duplicates().tolist()[: args.subjects_per_class]
        idx_list = epi + non
        _kv("idx_list (subjects)", idx_list)

    data = tuh.load_data(
        mode="raw",
        target_name="epilepsy",
        preload=True,
        rename_channels=True,
        set_montage=False,
        n_jobs=args.n_jobs,
        window_len_s=args.window_len_s,
        overlap_pct=args.overlap,
        balance_per_subject=True,
        include_seizures=False,
        fix_length_mode="resample",
        shuffle_windows=True,
        seed=args.seed,
        idx_list=idx_list,
        splits=splits,
        stratify_by="epilepsy",
    )

    _banner("OUTPUT")
    _kv("return type", type(data).__name__)
    _kv("split keys", list(data.keys()))

    subject_sets = {}
    for name in ["train", "val", "test"]:
        if name not in data:
            continue
        x, y, meta = data[name]
        _sec(f"=== split: {name} ===")
        _desc_tensor(f"{name}.X", x)
        _desc_tensor(f"{name}.y", y)
        if hasattr(x, "shape") and len(getattr(x, "shape", [])) == 3:
            _kv("channels (C)", x.shape[1])
            _kv("timepoints (T)", x.shape[2])
        if len(meta):
            subjects = sorted(meta["subject"].unique().tolist())
            subject_sets[name] = set(subjects)
            _kv("n_windows", len(meta))
            _kv("n_subjects", len(subjects))
            _kv("subjects", subjects)
            if "epilepsy" in meta.columns:
                bal = meta["epilepsy"].value_counts(dropna=False).to_dict()
                _kv("window label balance", bal)

    # Central invariant: no subject appears in more than one split.
    _sec("INVARIANT: subject-disjoint splits (no patient leakage)")
    leaks = []
    names = list(subject_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = subject_sets[names[i]] & subject_sets[names[j]]
            if overlap:
                leaks.append((names[i], names[j], overlap))
    if leaks:
        for a, b, ov in leaks:
            _kv(f"LEAK {a} & {b}", ov)
        _kv("result", "FAIL")
    else:
        _kv("result", "PASS (splits are subject-disjoint)")

    _sec("-> flows to Stage 3 / Stage 4")
    print("  # The DataModule wraps these tensors into TensorDatasets + dataloaders;")
    print("  # each window X then feeds the HYDRA feature extractor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
