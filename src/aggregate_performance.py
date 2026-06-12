"""Collect ``performance.csv`` across training runs into one comparison table.

Each training run writes a ``performance.csv`` (one row per split x level with
accuracy, balanced accuracy, sensitivity, specificity, precision, f1, mcc,
roc_auc, average_precision, and the n / n_pos / n_neg counts) into its Hydra run
directory. This script walks one or more run roots, reads every
``performance.csv``, attaches the run's config (signal_mode, GOF gate, dipole
toggle, model, seed, and the raw CLI overrides), and produces:

- a combined long table (every run x split x level) written to ``--out``;
- a focused comparison (one row per run for a chosen split and level, default
  test / subject) sorted by balanced accuracy, printed and saved alongside.

It only reads CSV / config files, so it is safe to run anywhere the run
directories are visible.

Example
-------
From the ``code/`` directory::

    python src/aggregate_performance.py --runs_root logs/train/runs
    python src/aggregate_performance.py --split test --level subject --metrics balanced_accuracy roc_auc mcc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import rootutils
from loguru import logger
from omegaconf import OmegaConf

root = rootutils.setup_root(__file__, pythonpath=True)

_CONFIG_COLS = (
    "signal_mode",
    "ica_keep_labels",
    "brain_ic_min_gof",
    "brain_ic_use_dipoles",
    "model",
    "seed",
    "overrides",
)
_DEFAULT_METRICS = (
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "n",
)


def _run_config(run_dir: Path) -> dict:
    """Read a run's Hydra config to label it (overrides plus key fields)."""
    info: dict = {c: None for c in _CONFIG_COLS}
    info["overrides"] = ""
    hydra_dir = run_dir / ".hydra"

    overrides_path = hydra_dir / "overrides.yaml"
    if overrides_path.exists():
        try:
            info["overrides"] = "; ".join(str(o) for o in OmegaConf.load(overrides_path))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read overrides for {run_dir.name}: {e}")

    config_path = hydra_dir / "config.yaml"
    if config_path.exists():
        try:
            cfg = OmegaConf.load(config_path)
            info["signal_mode"] = OmegaConf.select(cfg, "data.signal_mode")
            keep = OmegaConf.select(cfg, "data.ica_keep_labels")
            info["ica_keep_labels"] = list(keep) if keep is not None else None
            info["brain_ic_min_gof"] = OmegaConf.select(cfg, "data.brain_ic_min_gof")
            info["brain_ic_use_dipoles"] = OmegaConf.select(cfg, "data.brain_ic_use_dipoles")
            info["seed"] = OmegaConf.select(cfg, "seed")
            target = OmegaConf.select(cfg, "model._target_")
            info["model"] = target.split(".")[-1] if isinstance(target, str) else target
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not read config for {run_dir.name}: {e}")
    return info


def _collect(runs_roots: list[Path]) -> pd.DataFrame:
    """Build the combined long table over every ``performance.csv`` found."""
    frames: list[pd.DataFrame] = []
    for rr in runs_roots:
        for perf_path in sorted(rr.rglob("performance.csv")):
            run_dir = perf_path.parent
            try:
                df = pd.read_csv(perf_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Skipping unreadable {perf_path}: {e}")
                continue
            try:
                run_name = str(run_dir.relative_to(rr))
            except ValueError:
                run_name = run_dir.name
            df.insert(0, "run", run_name)
            for key, value in _run_config(run_dir).items():
                df[key] = value if not isinstance(value, list) else str(value)
            df["path"] = str(run_dir)
            frames.append(df)

    if not frames:
        roots = ", ".join(str(r) for r in runs_roots)
        raise FileNotFoundError(f"No 'performance.csv' found under: {roots}")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Collected {combined['run'].nunique()} runs "
        f"({len(combined)} split x level rows)."
    )
    return combined


def _focused(combined: pd.DataFrame, split: str, level: str, metrics: list[str]) -> pd.DataFrame:
    """Pivot to one row per run for a chosen split / level, sorted by bal. acc."""
    sel = combined[(combined["split"] == split) & (combined["level"] == level)].copy()
    if sel.empty:
        logger.warning(f"No rows for split='{split}', level='{level}'.")
        return sel
    available = [m for m in metrics if m in sel.columns]
    cols = ["run", *[c for c in _CONFIG_COLS if c != "overrides"], *available, "overrides"]
    cols = [c for c in cols if c in sel.columns]
    sel = sel[cols]
    if "balanced_accuracy" in sel.columns:
        sel = sel.sort_values("balanced_accuracy", ascending=False)
    return sel.reset_index(drop=True)


def main() -> None:
    """Parse arguments, aggregate, and write / print the comparison tables."""
    parser = argparse.ArgumentParser(
        description="Aggregate performance.csv across run directories into one table.",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        nargs="+",
        default=[str(root / "logs" / "train" / "runs")],
        help="Run root(s) searched recursively (default: <PROJECT_ROOT>/logs/train/runs).",
    )
    parser.add_argument("--split", type=str, default="test", help="Split for the focused table.")
    parser.add_argument("--level", type=str, default="subject", help="Level for the focused table.")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(_DEFAULT_METRICS),
        help="Metrics to show in the focused table.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(root / "logs" / "train" / "performance_comparison.csv"),
        help="Path for the combined long table (focused table saved alongside).",
    )
    args = parser.parse_args()

    runs_roots = [Path(r) for r in args.runs_root]
    combined = _collect(runs_roots)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    logger.info(f"Wrote combined table to {out}")

    focused = _focused(combined, args.split, args.level, args.metrics)
    if not focused.empty:
        focused_path = out.with_name(f"{out.stem}_{args.split}_{args.level}.csv")
        focused.to_csv(focused_path, index=False)
        logger.info(f"Wrote focused table to {focused_path}")
        logger.info(
            f"Comparison ({args.split} / {args.level}, best first):\n"
            f"{focused.round(4).to_string(index=False)}"
        )


if __name__ == "__main__":
    main()
