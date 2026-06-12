"""Diagnostics for the cached IC dipoles, to calibrate ``brain_ic`` thresholds.

Aggregates every ``-ica_dipoles.csv`` across the corpus (joined with its sibling
``-ica_labels.csv`` so components can be split by ICLabel category), then:

- plots histograms of the fitted dipole coordinates (x, y, z) and goodness of fit
  (GOF), with the current ``region_from_dipole`` thresholds (the ``_DIP_*``
  constants) drawn on the coordinate panels and candidate GOF cut-offs on the GOF
  panel;
- shows how many brain ICs land in each of the seven ``CANONICAL_REGIONS`` under
  the current thresholds, plus an axial (x vs y) scatter of the partition;
- writes a percentile summary and the per-region counts to CSV, and logs the
  numbers needed to pick ``data.brain_ic_min_gof`` and adjust the ``_DIP_*``
  thresholds.

This reads only the cached CSVs (no EDF / GPU), so it is safe to run anywhere the
dipole files are visible.

Example
-------
From the ``code/`` directory::

    python src/dipole_diagnostics.py --keep_labels brain --out_dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rootutils  # noqa: E402
from loguru import logger  # noqa: E402

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402

_COORD_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _collect_dipoles(data_dir: Path, version: str) -> pd.DataFrame:
    """Gather every ``-ica_dipoles.csv`` and attach each IC's ICLabel category.

    Parameters
    ----------
    data_dir : Path
        Parent directory of the version folder.
    version : str
        Corpus version subfolder (e.g. ``v3.0.0``).

    Returns
    -------
    pd.DataFrame
        One row per fitted dipole with columns ``ic, x, y, z, gof, label,
        recording`` (``label`` lower-cased; ``recording`` the file stem).
    """
    base = data_dir / version
    csv_paths = sorted(base.rglob("*-ica_dipoles.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No '-ica_dipoles.csv' files found under {base}")

    frames: list[pd.DataFrame] = []
    n_missing_labels = 0
    for dip_path in csv_paths:
        labels_path = dip_path.parent / dip_path.name.replace(
            "-ica_dipoles.csv", "-ica_labels.csv"
        )
        dip = pd.read_csv(dip_path)
        if labels_path.exists():
            lab = pd.read_csv(labels_path, index_col=0)
            label = lab["labels"].astype(str).str.strip().str.lower()
            dip["label"] = dip["ic"].map(label)
        else:
            n_missing_labels += 1
            dip["label"] = "unknown"
        dip["recording"] = dip_path.name.replace("-ica_dipoles.csv", "")
        frames.append(dip)

    if n_missing_labels:
        logger.warning(f"{n_missing_labels} dipole files had no sibling labels CSV.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["x", "y", "z", "gof"])
    logger.info(
        f"Collected {len(df)} dipoles from {len(csv_paths)} recordings "
        f"({df['label'].eq('brain').sum()} labelled brain)."
    )
    return df


def _summarize(
    df: pd.DataFrame, keep_labels: tuple[str, ...], gof_thresholds: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute coordinate/GOF percentiles and per-region counts for kept ICs.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        A percentile table (rows = percentiles, columns = x/y/z/gof) and the
        per-region counts reindexed to ``CANONICAL_REGIONS``.
    """
    kept = df[df["label"].isin(keep_labels)]
    if kept.empty:
        raise ValueError(f"No dipoles with label in {keep_labels}; nothing to summarize.")
    pct = pd.DataFrame(
        {
            col: np.percentile(kept[col], _COORD_PERCENTILES)
            for col in ("x", "y", "z", "gof")
        },
        index=[f"p{p}" for p in _COORD_PERCENTILES],
    )
    regions = [
        TUHEEGEpilepsy.region_from_dipole(float(x), float(y), float(z))
        for x, y, z in zip(kept["x"], kept["y"], kept["z"])
    ]
    region_counts = (
        pd.Series(regions, name="count")
        .value_counts()
        .reindex(TUHEEGEpilepsy.CANONICAL_REGIONS, fill_value=0)
    )

    logger.info(f"Kept ICs ({'/'.join(keep_labels)}): {len(kept)}")
    logger.info(f"Coordinate / GOF percentiles (meters, percent):\n{pct.round(4)}")
    logger.info(f"Brain ICs per region (current thresholds):\n{region_counts}")
    for tau in gof_thresholds:
        frac = float((kept["gof"] >= tau).mean())
        logger.info(f"GOF >= {tau:g}: keeps {frac * 100:.1f}% of kept ICs")
    return pct, region_counts


def _plot(
    df: pd.DataFrame,
    keep_labels: tuple[str, ...],
    region_counts: pd.Series,
    out_path: Path,
    bins: int,
    gof_thresholds: tuple[float, ...],
) -> None:
    """Render the 2x3 diagnostic figure and save it to ``out_path``."""
    kept = df[df["label"].isin(keep_labels)]
    other = df[~df["label"].isin(keep_labels)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # x: left-right, with midline / lateral thresholds (both signs).
    axes[0, 0].hist(kept["x"], bins=bins, color="steelblue")
    for thr in (
        -TUHEEGEpilepsy._DIP_X_LATERAL,
        -TUHEEGEpilepsy._DIP_X_MID,
        TUHEEGEpilepsy._DIP_X_MID,
        TUHEEGEpilepsy._DIP_X_LATERAL,
    ):
        axes[0, 0].axvline(thr, color="crimson", ls="--", lw=1)
    axes[0, 0].set(title="Dipole x (left -> right)", xlabel="x (m)", ylabel="count")

    # y: posterior-anterior, with frontal / posterior thresholds.
    axes[0, 1].hist(kept["y"], bins=bins, color="steelblue")
    for thr in (TUHEEGEpilepsy._DIP_Y_POSTERIOR, TUHEEGEpilepsy._DIP_Y_FRONTAL):
        axes[0, 1].axvline(thr, color="crimson", ls="--", lw=1)
    axes[0, 1].set(title="Dipole y (post -> ant)", xlabel="y (m)")

    # z: inferior-superior, with the temporal threshold.
    axes[0, 2].hist(kept["z"], bins=bins, color="steelblue")
    axes[0, 2].axvline(
        TUHEEGEpilepsy._DIP_Z_TEMPORAL, color="crimson", ls="--", lw=1, label="z thresh"
    )
    axes[0, 2].set(title="Dipole z (inf -> sup)", xlabel="z (m)")

    # GOF: kept (brain) vs other, with candidate cut-offs.
    gof_lo = float(np.percentile(df["gof"], 1))
    rng = (min(gof_lo, 0.0), 100.0)
    axes[1, 0].hist(
        kept["gof"], bins=bins, range=rng, color="seagreen", alpha=0.7,
        density=True, label="/".join(keep_labels),
    )
    if len(other):
        axes[1, 0].hist(
            other["gof"], bins=bins, range=rng, color="gray", alpha=0.5,
            density=True, label="other",
        )
    for tau in gof_thresholds:
        axes[1, 0].axvline(tau, color="crimson", ls="--", lw=1)
    axes[1, 0].set(title="Goodness of fit", xlabel="GOF (%)", ylabel="density")
    axes[1, 0].legend(fontsize=8)

    # Per-region counts for kept ICs under the current thresholds.
    axes[1, 1].bar(range(len(region_counts)), region_counts.to_numpy(), color="slateblue")
    axes[1, 1].set_xticks(range(len(region_counts)))
    axes[1, 1].set_xticklabels(region_counts.index, rotation=45, ha="right", fontsize=7)
    axes[1, 1].set(title="Kept ICs per region", ylabel="count")

    # Axial scatter (x vs y) showing the partition, with both sets of thresholds.
    axes[1, 2].scatter(kept["x"], kept["y"], s=3, alpha=0.2, color="steelblue")
    for thr in (
        -TUHEEGEpilepsy._DIP_X_LATERAL,
        -TUHEEGEpilepsy._DIP_X_MID,
        TUHEEGEpilepsy._DIP_X_MID,
        TUHEEGEpilepsy._DIP_X_LATERAL,
    ):
        axes[1, 2].axvline(thr, color="crimson", ls="--", lw=0.8)
    for thr in (TUHEEGEpilepsy._DIP_Y_POSTERIOR, TUHEEGEpilepsy._DIP_Y_FRONTAL):
        axes[1, 2].axhline(thr, color="crimson", ls="--", lw=0.8)
    axes[1, 2].set(title="Axial x vs y (kept ICs)", xlabel="x (m)", ylabel="y (m)")

    fig.suptitle(
        f"IC dipole diagnostics ({len(kept)} kept / {len(df)} total dipoles)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved diagnostic figure to {out_path}")


def main() -> None:
    """Parse CLI arguments, collect dipoles, and write the diagnostics."""
    parser = argparse.ArgumentParser(
        description="Histogram fitted IC dipole coordinates and GOF to calibrate brain_ic.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(root / "data"),
        help="Parent directory of the version folder (default: <PROJECT_ROOT>/data).",
    )
    parser.add_argument("--version", type=str, default="v3.0.0", help="Corpus version subfolder.")
    parser.add_argument(
        "--keep_labels",
        type=str,
        nargs="+",
        default=["brain"],
        help="ICLabel categories to treat as kept ICs (default: brain).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(root / "figures"),
        help="Directory for the figure and summary CSVs (default: <PROJECT_ROOT>/figures).",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins (default: 60).")
    parser.add_argument(
        "--gof_thresholds",
        type=float,
        nargs="+",
        default=[50.0, 70.0, 80.0, 90.0],
        help="Candidate GOF cut-offs to mark and report (default: 50 70 80 90).",
    )
    args = parser.parse_args()

    keep_labels = tuple(s.strip().lower() for s in args.keep_labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_dipoles(Path(args.data_dir), args.version)
    pct, region_counts = _summarize(df, keep_labels, tuple(args.gof_thresholds))

    pct.to_csv(out_dir / "dipole_diagnostics_percentiles.csv")
    region_counts.to_csv(out_dir / "dipole_diagnostics_region_counts.csv", header=True)
    _plot(
        df,
        keep_labels,
        region_counts,
        out_dir / "dipole_diagnostics.png",
        args.bins,
        tuple(args.gof_thresholds),
    )


if __name__ == "__main__":
    main()
