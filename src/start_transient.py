"""Recording-start transient of the processed signal, measured on cached train/validation windows.

Protocol: ``documentation/recording_start_transient_protocol.md``. For every train or validation
recording with at least two segments, read its first window (0-120 s) and second window (120-240 s)
from the campaign signal cache: the signal after the 0.1 Hz high-pass, the 60 Hz notch, bad-channel
interpolation, the TCP bipolar montage and resampling to 256 Hz. Test signals are never read.

For each second t of a window:

- ``rms``: root mean square over usable channels and samples;
- ``line_length``: mean over usable channels of the summed absolute differences of the samples;
- ``clip_frac``: share of usable channel-samples at the encoder clip after the model's row scaling
  (median/MAD per window and channel, clip at +-20 robust units).

For rms and line length, the paired log ratio ``d(t) = log s1(t) - log s2(t)`` compares the same
second of the first and the second window of one recording. Its median over seconds 90-119 measures
a level difference between the two windows that is unrelated to the start; subtracting it gives the
start excess ``e(t)``. A group curve is the median of ``e(t)`` over recordings, with a patient-level
bootstrap interval. For the clip fraction the curve is the mean of ``c1(t) - c2(t)``.

Decision (fixed before the first result, constants below): the transient of a statistic ends at the
first second after which its curve stays inside the tolerance until second 90. The proposed skip is
that end, maximized over the three statistics, rounded up to 10 s, and never shorter than the filter
edge (the context radius of the high-pass and notch filters, where MNE's reflection padding defines
the signal). Class and native-rate differences of the mean excess over the first 30 s are reported
with patient-level bootstrap intervals, because a start transient that differs by class is a
confound for the classifier.

Run on the HPC inside the container (CPU suffices)::

    python src/start_transient.py --manifests <audit dir> --data-dir <corpus parent> \\
        --signal-cache-dir <campaign cache> --output logs/start_transient/<name>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

# Decision constants, fixed before the first result (protocol section 3).
EXCESS_TOLERANCE = math.log(1.10)  # |median start excess| of rms and line length: within 10 percent
CLIP_TOLERANCE_ABS = 1e-4  # clip-fraction excess tolerance, absolute ...
CLIP_TOLERANCE_REL = 0.5  # ... or this fraction of the second window's mean clip fraction, if larger
LATE_SECONDS = (90, 120)  # seconds of the window that define the level offset between the windows
SKIP_STEP_S = 10  # the proposed skip is rounded up to a multiple of this
CONFOUND_WINDOW_S = 30  # class and rate comparisons use the mean excess over these first seconds
CONFOUND_MIN_EFFECT = math.log(1.05)  # a flagged difference must also reach 5 percent
STATISTICS = ("rms", "line_length", "clip_frac")
COLORS = {"all": "#000000", "class 0": "#0072B2", "class 1": "#D55E00", "rate 250": "#009E73", "rate 256": "#CC79A7"}


def per_second_stats(signal: np.ndarray, rows: np.ndarray, valid: np.ndarray, sfreq: int) -> np.ndarray | None:
    """``(3, seconds)`` rms, line length and clip fraction of one window; ``None`` without a usable channel.

    ``signal`` is the processed window ``(C, T)``, ``rows`` the model's scaled rows ``(C, T)`` and
    ``valid`` the usable-channel mask ``(C,)``. ``T`` must be a multiple of ``sfreq``.
    """
    if not valid.any():
        return None
    from src.models.components.shapeconv_sae import ROBUST_CLIP

    x = signal[valid].astype(np.float64).reshape(int(valid.sum()), -1, sfreq)
    scaled = rows[valid].reshape(x.shape)
    rms = np.sqrt((x**2).mean(axis=(0, 2)))
    line_length = np.abs(np.diff(x, axis=2)).sum(axis=2).mean(axis=0)
    clip = (np.abs(scaled) >= ROBUST_CLIP).mean(axis=(0, 2))
    return np.stack([rms, line_length, clip])


def start_excess(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Start excess ``e(t)`` per recording ``(R, S)`` from paired per-second statistics ``(R, S)``.

    ``log first - log second``, minus its median over ``LATE_SECONDS`` of the same recording.
    """
    paired = np.log(first) - np.log(second)
    offset = np.median(paired[:, LATE_SECONDS[0] : LATE_SECONDS[1]], axis=1, keepdims=True)
    return paired - offset


def bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    reducer: Callable[..., np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced curve over recordings ``(S,)`` and its patient-level 95 percent percentile interval.

    ``values`` is ``(R, S)``, ``clusters`` the patient of each row, ``reducer`` ``np.median`` or
    ``np.mean`` (called with ``axis=0``). Patients are resampled with replacement, with all their rows.
    """
    estimate = reducer(values, axis=0)
    members = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    draws = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        chosen = rng.integers(0, len(members), len(members))
        draws[b] = reducer(values[np.concatenate([members[i] for i in chosen])], axis=0)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return estimate, low, high


def transient_end(curve: np.ndarray, tolerance: float, stop: int = LATE_SECONDS[0], two_sided: bool = True) -> int:
    """First second after which ``curve`` stays within ``tolerance`` up to ``stop`` (exclusive).

    Returns 0 when no second before ``stop`` exceeds the tolerance, and ``stop`` when the last
    second before ``stop`` still exceeds it (the transient is not resolved).
    """
    deviation = np.abs(curve[:stop]) if two_sided else curve[:stop]
    outside = np.flatnonzero(deviation > tolerance)
    return int(outside[-1]) + 1 if outside.size else 0


def clip_tolerance(second_window_clip: np.ndarray) -> float:
    """Tolerance of the clip-fraction excess, given the second window's per-second clip fractions ``(R, S)``."""
    return max(CLIP_TOLERANCE_ABS, CLIP_TOLERANCE_REL * float(second_window_clip.mean()))


def skip_length(ends: dict[str, int], filter_edge_s: float) -> int:
    """Proposed skip: the latest transient end rounded up to ``SKIP_STEP_S``, at least the filter edge."""
    latest = max(max(ends.values()), math.ceil(filter_edge_s))
    return int(math.ceil(latest / SKIP_STEP_S) * SKIP_STEP_S)


def group_difference(
    burden: np.ndarray,
    in_a: np.ndarray,
    in_b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Mean burden of group ``a`` minus group ``b`` with a stratified patient bootstrap interval and a flag."""
    if not in_a.any() or not in_b.any():
        return {"difference": None, "low": None, "high": None, "n_a": int(in_a.sum()), "n_b": int(in_b.sum())}
    estimate = float(burden[in_a].mean() - burden[in_b].mean())
    groups = []
    for mask in (in_a, in_b):
        patients = clusters[mask]
        groups.append([np.flatnonzero(mask & (clusters == c)) for c in np.unique(patients)])
    draws = np.empty(n_boot)
    for b in range(n_boot):
        means = []
        for members in groups:
            chosen = rng.integers(0, len(members), len(members))
            means.append(burden[np.concatenate([members[i] for i in chosen])].mean())
        draws[b] = means[0] - means[1]
    low, high = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {
        "difference": estimate,
        "low": low,
        "high": high,
        "n_a": int(in_a.sum()),
        "n_b": int(in_b.sum()),
        "flag": (low > 0 or high < 0) and abs(estimate) >= CONFOUND_MIN_EFFECT,
    }


def analyze(
    first: np.ndarray,
    second: np.ndarray,
    meta: pl.DataFrame,
    filter_edge_s: float,
    n_boot: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Curves, per-recording burdens and the decision from paired statistics ``(R, 3, S)``.

    ``meta`` holds one row per recording with ``subject``, ``epilepsy`` (0/1) and ``sfreq``.
    Returns the per-second table, the per-recording table and the decision.
    """
    rng = np.random.default_rng(seed)
    clusters = meta["subject"].to_numpy()
    labels = meta["epilepsy"].to_numpy().astype(int)
    rates = meta["sfreq"].to_numpy().round().astype(int)
    groups = {"all": np.ones(len(meta), dtype=bool), "class 0": labels == 0, "class 1": labels == 1}
    groups.update({f"rate {r}": rates == r for r in sorted(set(rates.tolist()))})
    excess = {name: start_excess(first[:, i], second[:, i]) for i, name in enumerate(STATISTICS[:2])}
    clip_diff = first[:, 2] - second[:, 2]
    seconds = np.arange(first.shape[2])
    rows: list[pl.DataFrame] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for group, mask in groups.items():
        if not mask.any():
            continue
        curves[group] = {}
        for name, values, reducer in (
            ("rms", excess["rms"], np.median),
            ("line_length", excess["line_length"], np.median),
            ("clip_frac", clip_diff, np.mean),
        ):
            estimate, low, high = bootstrap(values[mask], clusters[mask], reducer, n_boot, rng)
            curves[group][name] = estimate
            rows.append(
                pl.DataFrame(
                    {
                        "group": group,
                        "statistic": name,
                        "second": seconds,
                        "recordings": int(mask.sum()),
                        "estimate": estimate,
                        "low": low,
                        "high": high,
                    }
                )
            )
    tolerance = {"rms": EXCESS_TOLERANCE, "line_length": EXCESS_TOLERANCE, "clip_frac": clip_tolerance(second[:, 2])}
    ends = {
        name: transient_end(curves["all"][name], tolerance[name], two_sided=name != "clip_frac") for name in STATISTICS
    }
    skip = skip_length(ends, filter_edge_s)
    window = slice(0, CONFOUND_WINDOW_S)
    burden = {name: excess[name][:, window].mean(axis=1) for name in ("rms", "line_length")}
    comparisons: dict[str, Any] = {}
    for name, values in burden.items():
        comparisons[name] = {
            "class_1_minus_class_0": group_difference(values, labels == 1, labels == 0, clusters, n_boot, rng),
            "class_1_minus_class_0_within_rate": {
                str(r): group_difference(
                    values, (labels == 1) & (rates == r), (labels == 0) & (rates == r), clusters, n_boot, rng
                )
                for r in sorted(set(rates.tolist()))
            },
        }
        if {250, 256} <= set(rates.tolist()):
            comparisons[name]["rate_250_minus_256"] = group_difference(
                values, rates == 250, rates == 256, clusters, n_boot, rng
            )
    recordings = meta.with_columns(
        pl.Series("rms_burden_first_30s", burden["rms"]),
        pl.Series("line_length_burden_first_30s", burden["line_length"]),
        pl.Series("clip_frac_first_30s", first[:, 2, window].mean(axis=1)),
        pl.Series("clip_frac_second_window_first_30s", second[:, 2, window].mean(axis=1)),
    )
    flags = [
        f"{name}: {key}"
        for name, block in comparisons.items()
        for key, value in block.items()
        if isinstance(value, dict) and value.get("flag")
    ]
    flags += [
        f"{name}: class difference within rate {rate}"
        for name, block in comparisons.items()
        for rate, value in block["class_1_minus_class_0_within_rate"].items()
        if value.get("flag")
    ]
    decision = {
        "recordings": len(meta),
        "filter_edge_s": filter_edge_s,
        "tolerance": tolerance,
        "transient_end_s": ends,
        "unresolved": any(end >= LATE_SECONDS[0] for end in ends.values()),
        "proposed_skip_s": skip,
        "comparisons_first_30s": comparisons,
        "confound_flags": flags,
    }
    return pl.concat(rows), recordings, decision


def plot(table: pl.DataFrame, decision: dict[str, Any], path: Path) -> None:
    """Three panels (rms, line length, clip fraction) of the start excess against time since the recording start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "rms": "RMS: median start excess (log ratio)",
        "line_length": "Line length: median start excess (log ratio)",
        "clip_frac": "Clip fraction: mean excess over the second window",
    }
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, name in zip(axes, STATISTICS):
        for group, color in COLORS.items():
            curve = table.filter((pl.col("group") == group) & (pl.col("statistic") == name)).sort("second")
            if curve.is_empty():
                continue
            t = curve["second"].to_numpy() + 0.5
            ax.plot(t, curve["estimate"].to_numpy(), color=color, lw=1.6 if group == "all" else 1.0, label=group)
            if group == "all":
                ax.fill_between(t, curve["low"].to_numpy(), curve["high"].to_numpy(), color=color, alpha=0.15, lw=0)
        tolerance = decision["tolerance"][name]
        ax.axhspan(-tolerance if name != "clip_frac" else 0.0, tolerance, color="#999999", alpha=0.12, lw=0)
        ax.axhline(0.0, color="#999999", lw=0.6)
        ax.axvline(decision["filter_edge_s"], color="#555555", ls=":", lw=1.0, label="filter edge")
        ax.axvline(decision["proposed_skip_s"], color="#E69F00", ls="--", lw=1.2, label="proposed skip")
        ax.set_title(titles[name], fontsize=9)
    axes[0].legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time since recording start (s)")
    fig.suptitle(
        f"Recording-start transient, {decision['recordings']} train/validation recordings; "
        f"proposed skip {decision['proposed_skip_s']} s",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _enable_cached(dataset: Any, cache_root: Path, existing: set[str], split: str) -> str:
    """Attach the campaign cache and refuse to rebuild it; return the cache identity."""
    dataset.enable_signal_cache(cache_root)
    identity = dataset.signal_cache.identity
    if identity not in existing:
        folder = cache_root / identity
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
        raise ValueError(
            f"{split} signals have no entries under {cache_root} (identity {identity}); pass the campaign cache"
        )
    return identity


def measure(args: argparse.Namespace) -> None:
    """Read the paired windows from the cache, compute the statistics and write the outputs."""
    import hydra
    import torch
    from torch.utils.data import DataLoader, Subset

    from src.data.components.temporal_filtering import filter_context_seconds
    from src.run_patient_waveform_sweep import check_audit, digest, make_config, save_json

    manifests, output = args.manifests.resolve(), args.output.resolve()
    audit = check_audit(manifests)
    if audit["skip_start_s"] != 0:
        raise ValueError(
            "the start transient is measured on the offset-0 reference audit, whose first windows start at 0 s"
        )
    cache_root = args.signal_cache_dir.expanduser().resolve()
    if not cache_root.is_dir():
        raise ValueError(f"signal cache root not found: {cache_root}")
    existing = {p.name for p in cache_root.iterdir() if p.is_dir()}
    output.mkdir(parents=True, exist_ok=False)
    cfg = make_config(manifests, args.data_dir.resolve(), args.num_workers, skip_start_s=audit["skip_start_s"])
    cfg.supervised = None
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    model = hydra.utils.instantiate(cfg.feature, device="cpu")  # only for the model's row scaling
    sfreq = int(cfg.data.target_sfreq)
    identities, first, second, meta = {}, [], [], []
    excluded = {"single_segment": 0, "unusable": 0}
    for split in ("train", "val"):
        dataset = getattr(datamodule, f"data_{split}")
        identities[split] = _enable_cached(dataset, cache_root, existing, split)
        plan = dataset.plan.reset_index(drop=True)
        index = {(str(p), float(s)): i for i, (p, s) in enumerate(zip(plan["path"], plan["start"]))}
        pairs = []
        for path in plan["path"].astype(str).unique():
            if (path, 0.0) in index and (path, 120.0) in index:
                pairs.append((index[(path, 0.0)], index[(path, 120.0)]))
            else:
                excluded["single_segment"] += 1
        order = [i for pair in pairs for i in pair]
        loader = DataLoader(Subset(dataset, order), batch_size=2 * args.pairs_per_batch, num_workers=args.num_workers)
        position = 0
        for x, _ in loader:
            rows, valid = model._rows(x)
            for b in range(0, x.shape[0], 2):
                i1, _ = pairs[position]
                position += 1
                stats = [
                    per_second_stats(x[b + k].numpy(), rows[b + k].numpy(), valid[b + k].numpy(), sfreq) for k in (0, 1)
                ]
                if stats[0] is None or stats[1] is None or (stats[0][:2] <= 0).any() or (stats[1][:2] <= 0).any():
                    excluded["unusable"] += 1
                    continue
                first.append(stats[0])
                second.append(stats[1])
                row = plan.iloc[i1]
                meta.append(
                    {
                        "split": split,
                        "subject": str(row["subject"]),
                        "path": str(row["path"]),
                        "epilepsy": int(bool(row["epilepsy"])),
                        "sfreq": float(row["description_row"]["sfreq"]),
                    }
                )
        logger.info(f"{split}: {len(pairs)} paired recordings read")
    rates = sorted({m["sfreq"] for m in meta})
    filter_edge_s = max(filter_context_seconds(r, cfg.data.filter_freq, cfg.data.notch_freqs) for r in rates)
    table, recordings, decision = analyze(
        np.stack(first), np.stack(second), pl.DataFrame(meta), filter_edge_s, args.bootstrap, args.seed
    )
    decision["excluded"] = excluded
    decision["native_rates"] = rates
    table.write_csv(output / "start_transient_seconds.csv")
    recordings.write_csv(output / "start_transient_recordings.csv")
    save_json(output / "start_transient_decision.json", decision)
    save_json(
        output / "start_transient_provenance.json",
        {
            "audit": audit,
            "manifests": {path.name: digest(path) for path in manifests.glob("windows_*.csv")},
            "signal_cache": {"root": str(cache_root), **identities},
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "torch": str(torch.__version__),
        },
    )
    plot(table, decision, output / "start_transient.png")
    logger.info(f"Decision: {json.dumps(decision)}")


def main() -> None:
    """Parse the locations and run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--signal-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pairs-per-batch", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
