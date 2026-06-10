"""Stage 3 - DataModule and dataloader batches.

INPUT  : a `TUHEEGDataModule` configuration (data dir, window length, split,
         batch size, seed).
OUTPUT : after `setup()`,
           - data_train / data_val / data_test : TensorDatasets
           - train_df / val_df / test_df        : per-split metadata
           - one (X, y) batch from each dataloader (the exact tuple the custom
             Trainer feeds into the HYDRA feature extractor).

NOTE: unlike Stage 2, the real DataModule.setup() windows the FULL corpus (all
subjects) and also runs a second 2-second "dictionary-learning" load pass. That
is heavy (lots of EDF I/O + RAM), so this script refuses to run without
`--yes-full`. For quick subset exploration of the windowing logic, use Stage 2.

Run:
    uv run python tests/stage3_datamodule.py --yes-full --window-len-min 1 --batch-size 8
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


def _desc_df(name: str, df, n: int = 5) -> None:
    _sec(f"[dataframe] {name}")
    if df is None:
        _kv("value", "None")
        return
    _kv("shape", df.shape)
    _kv("columns", list(df.columns))
    if len(df):
        print(df.head(n).to_string())


def _desc_batch(name: str, batch) -> None:
    _sec(f"[batch] {name}")
    x, y = batch
    _kv("X shape / dtype", f"{tuple(x.shape)}  {x.dtype}")
    _kv("y shape / dtype", f"{tuple(y.shape)}  {y.dtype}")
    _kv("X min/max", f"{x.float().min():.4g} / {x.float().max():.4g}")
    _kv("y values (first 16)", y.flatten()[:16].tolist())


def _need(import_fn, pkgs: str):
    try:
        return import_fn()
    except ImportError as exc:
        _sec("missing dependency")
        _kv("error", exc)
        print(f"  # needs: {pkgs}; run `uv sync` in code/")
        return None


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--version", default="v3.0.0")
    p.add_argument("--window-len-min", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--splits", default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--yes-full", action="store_true",
                   help="acknowledge that setup() processes the FULL corpus")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    _banner("STAGE 3 - DataModule and dataloader batches")

    def _imports():
        from src.data.tuh_eeg_epilepsy_datamodule import TUHEEGDataModule
        return TUHEEGDataModule

    TUHEEGDataModule = _need(_imports, "lightning, braindecode, mne, torch, pandas")
    if TUHEEGDataModule is None:
        return 1

    ds = Path(args.data_dir) / args.version
    needed = ["00_epilepsy", "01_no_epilepsy", "DOCS"]
    if not (ds.exists() and all((ds / s).is_dir() for s in needed)):
        _sec("dataset not available")
        _kv("looked in", ds)
        print("  # Requires the unpacked TUH corpus. Run where data/<version>/ exists.")
        return 0

    if not args.yes_full:
        _sec("refusing to run the full pipeline")
        print("  # DataModule.setup() windows ALL subjects and runs a second")
        print("  # dictionary-learning load pass. Re-run with --yes-full to proceed,")
        print("  # or use Stage 2 (--subjects-per-class) for a fast subset view.")
        return 0

    ratios = [float(x) for x in args.splits.split(",")]

    _sec("INPUT")
    _kv("window_len_min", args.window_len_min)
    _kv("train_val_test_split", tuple(ratios))
    _kv("batch_size", args.batch_size)
    _kv("seed", args.seed)

    dm = TUHEEGDataModule(
        data_dir=args.data_dir,
        version=args.version,
        train_val_test_split=tuple(ratios),
        window_len_min=args.window_len_min,
        overlap_pct=0.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print("  # running setup() ... (full corpus; this can take a while)")
    dm.setup()

    _banner("OUTPUT")
    _kv("num_classes", dm.num_classes)
    _kv("len(data_train)", len(dm.data_train))
    _kv("len(data_val)", len(dm.data_val))
    _kv("len(data_test)", len(dm.data_test))

    _desc_df("train_df", dm.train_df)
    _desc_df("val_df", dm.val_df)
    _desc_df("test_df", dm.test_df)

    _desc_batch("next(train_dataloader)", next(iter(dm.train_dataloader())))
    _desc_batch("next(val_dataloader)", next(iter(dm.val_dataloader())))
    _desc_batch("next(test_dataloader)", next(iter(dm.test_dataloader())))

    _sec("-> flows to Stage 4")
    print("  # Trainer._extract_features iterates these dataloaders and pushes each")
    print("  # X batch through the HYDRA feature extractor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
