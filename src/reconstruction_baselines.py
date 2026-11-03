"""Reconstruction baselines of the separable dictionary: does its R=1 code beat simple codes, in total and above 4 Hz?

Protocol: ``documentation/separable_reconstruction_baselines_protocol.md``. Run on the HPC from ``code/``; a root
holds one read of the five pass-1 dictionaries of the frozen readout, which it does not change.
``src/reconstruction_baselines_core.py`` holds the pure logic and the rule. Subcommands, in order:

- ``fit`` computes the spatial patterns and the window PCA of one outer fold from the patients that trained its
  dictionary, on their fixed segments.
- ``score`` reconstructs the fixed segments of the fold's own patients with every arm, and saves the energies per
  segment and band.
- ``read`` applies the gates and the rule, and writes ``reconstruction_decision.json``,
  ``reconstruction_summary.csv`` and ``pca_curve.csv``.

The 39 test patients stay locked: the readout's cohort refuses them. No label enters a fit or a score; ``read``
uses the labels only to report EV per class.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.frozen_readout_core import N_FOLDS, dictionary_split  # noqa: E402
from src.reconstruction_baselines_core import (  # noqa: E402
    ARMS,
    ATOM_LEN,
    BAND_NAMES,
    CONFIDENCE,
    FAST_FLOOR,
    N_ATOMS,
    N_BOOTSTRAP,
    N_SPATIAL,
    PARSEVAL_TOLERANCE,
    PCA_KS,
    POOL_LEN,
    ROW_MATCH_TOLERANCE,
    S4_ARMS,
    SFREQ,
    SLOW_ATOM_HZ,
    TOP_R,
    BatchScore,
    Interval,
    accumulate_moments,
    band_weights,
    bootstrap_counts,
    conditions,
    ev_interval,
    fast_bands,
    gabor_bank,
    paired_interval,
    parseval_error,
    pooled_ev,
    reconstruction_outcome,
    score_batch,
    set_fixed_atoms,
    smallest_k,
    top_components,
)
from src.separable_factor_analysis import dominant_frequency  # noqa: E402

PREDICTION = "slow_only, about 85% (protocol section 7)"
# Fields of BatchScore with one value per segment; the others are per batch.
SEGMENT_FIELDS = (
    "signal_bands",
    "signal_total",
    "residual_bands",
    "residual_total",
    "usable_rows",
    "windows_won",
    "windows_slow",
    "explained_all",
    "explained_slow",
)


# ---------------------------------------------------------------------- files


def fit_path(root: Path, fold: int) -> Path:
    """The PCA and spatial patterns of one fold."""
    return root / "fits" / f"fold{fold}.npz"


def score_path(root: Path, fold: int) -> Path:
    """The energies of one fold's scored patients."""
    return root / "scores" / f"fold{fold}.npz"


def _plain(value: Any) -> Any:
    """``value`` with numpy scalars made plain and NaN floats made ``None``, at any depth."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(value), indent=2) + "\n")


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write through a temporary file, so a finished file means a finished stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _check_geometry(model: Any) -> None:
    """Refuse a dictionary whose geometry differs from the protocol's.

    Raises
    ------
    ValueError
        If ``K``, ``L``, ``P``, ``R``, ``D`` or the row domain differ.
    """
    spec = model.spec
    found = (spec.n_atoms, spec.atom_len, spec.pool_len, spec.top_r, spec.n_projections, spec.pre_emphasis)
    wanted = (N_ATOMS, ATOM_LEN, POOL_LEN, TOP_R, N_SPATIAL, "none")
    if found != wanted:
        raise ValueError(f"dictionary geometry {found} differs from the protocol's {wanted}")


# ---------------------------------------------------------------------- data


def _context(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    """The readout's plan, sources and cohort, and the trained dictionary of a fold, on the GPU.

    Raises
    ------
    RuntimeError
        Without a CUDA GPU.
    ValueError
        If the cohort differs from the readout's patients, the rate is not 256 Hz, or the geometry differs.
    """
    from src.frozen_readout import _load_dictionary, build_cohort, dictionary_dir, load_plan, load_sources

    if not torch.cuda.is_available():
        raise RuntimeError("run inside the HPC container on an allocated CUDA GPU")
    readout = args.readout_root.resolve()
    plan = load_plan(readout)
    seed = plan["seed"] + fold
    cfg, _, sources, locked, _ = load_sources(args, seed)
    sfreq = float(cfg.data.target_sfreq)
    if sfreq != SFREQ:
        raise ValueError(f"the protocol reads {SFREQ} Hz rows, the configuration gives {sfreq} Hz")
    cohort = build_cohort(sources, locked)
    if set(cohort.labels) != set(plan["labels"]):
        raise ValueError("the cohort differs from the patients of the readout plan")

    def load(arm: str) -> Any:
        model = _load_dictionary(dictionary_dir(readout, arm, fold), cfg, seed, "cuda")
        _check_geometry(model)
        return model

    return {"plan": plan, "sources": sources, "cohort": cohort, "sfreq": sfreq, "load": load}


def _segments(context: dict[str, Any], patients: Iterable[str]) -> Any:
    """The band-limited fixed segments of ``patients``, patient by patient."""
    from src.frozen_readout import BandLimited, PooledSegments, fixed_rows

    rows = fixed_rows(context["cohort"], sorted(patients), context["plan"]["n_segments"])
    return BandLimited(PooledSegments(context["sources"], rows), context["sfreq"])


def _loader(view: Any, num_workers: int) -> Any:
    from torch.utils.data import DataLoader

    from src.frozen_readout import EXTRACT_BATCH

    return DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------------- fit


def fit_fold(args: argparse.Namespace) -> None:
    """The spatial patterns and the uncentered window PCA of one fold, from its dictionary's training patients.

    A finished fold is skipped.

    Raises
    ------
    ValueError
        If a fit patient belongs to the fold.
    """
    root, fold = args.root.resolve(), args.fold
    path = fit_path(root, fold)
    if path.is_file():
        logger.info(f"fit fold {fold}: finished")
        return
    context = _context(args, fold)
    plan = context["plan"]
    patients, _ = dictionary_split(plan["labels"], plan["folds"], fold, plan["seed"])
    if any(plan["folds"][s] == fold for s in patients):
        raise ValueError(f"a fit patient belongs to fold {fold}")
    model = context["load"]("trained")
    n_channels = model.n_channels
    size = n_channels * POOL_LEN
    window_moment = torch.zeros(size, size, dtype=torch.float64, device="cuda")
    channel_moment = torch.zeros(n_channels, n_channels, dtype=torch.float64, device="cuda")
    n_windows = n_samples = 0
    with torch.no_grad():
        for x, _ in _loader(_segments(context, patients), args.num_workers):
            rows, _ = model._full_rows(x)
            added = accumulate_moments(rows, window_moment, channel_moment)
            n_windows, n_samples = n_windows + added[0], n_samples + added[1]
    window_moment /= n_windows
    basis, values = top_components(window_moment, max(PCA_KS))
    del window_moment
    patterns, channel_values = top_components(channel_moment / n_samples, N_SPATIAL)
    share = (values.cumsum(0) / values.sum()).cpu().numpy()
    _save_npz(
        path,
        basis=basis.float().cpu().numpy(),
        values=values.cpu().numpy(),
        patterns=patterns.cpu().numpy(),
        channel_values=channel_values.cpu().numpy(),
        patients=np.array(sorted(patients), dtype=str),
        n_windows=np.array(n_windows),
    )
    _save_json(
        path.with_suffix(".json"),
        {
            "fold": fold,
            "fit_patients": len(patients),
            "windows": n_windows,
            "training_energy_share": {f"PCA-{k}": float(share[k - 1]) for k in PCA_KS},
            "spatial_energy_share": float((channel_values[:N_SPATIAL].sum() / channel_values.sum()).item()),
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted": _git("status", "--porcelain").splitlines(),
        },
    )
    logger.info(f"fit fold {fold}: {len(patients)} patients, {n_windows} windows, PCA-1 share {share[0]:.3f}")


# ---------------------------------------------------------------------- score


def score_fold(args: argparse.Namespace) -> None:
    """Reconstruct the fixed segments of one fold's patients with every arm; save energies per segment and band.

    A finished fold is skipped.

    Raises
    ------
    FileNotFoundError
        If the fold's fit is missing.
    ValueError
        If a fit patient is among the scored patients.
    """
    root, fold = args.root.resolve(), args.fold
    path = score_path(root, fold)
    if path.is_file():
        logger.info(f"score fold {fold}: finished")
        return
    if not fit_path(root, fold).is_file():
        raise FileNotFoundError(f"no fit for fold {fold}: run fit --fold {fold} first")
    with np.load(fit_path(root, fold)) as fit:
        basis, patterns, fit_patients = fit["basis"], fit["patterns"], set(fit["patients"].tolist())
    context = _context(args, fold)
    plan, sfreq = context["plan"], context["sfreq"]
    scored = sorted(s for s, f in plan["folds"].items() if f == fold)
    if fit_patients & set(scored):
        raise ValueError(f"a fit patient is among the scored patients of fold {fold}")
    models = {"S4-trained": context["load"]("trained"), "S4-untrained": context["load"]("untrained")}
    models["S4-fixed"] = context["load"]("trained")
    set_fixed_atoms(models["S4-fixed"], gabor_bank(), patterns)
    trained = models["S4-trained"]
    waveforms = trained._split(trained._dictionary().detach())[0].cpu().numpy()
    dominant = dominant_frequency(waveforms, sfreq)
    slow = torch.as_tensor(dominant < SLOW_ATOM_HZ, device="cuda")
    basis_t = torch.as_tensor(basis, device="cuda")
    view = _segments(context, scored)
    subjects = view.plan["subject"].astype(str).to_numpy()
    parts: dict[str, list[np.ndarray]] = {name: [] for name in SEGMENT_FIELDS}
    violations = np.zeros(len(S4_ARMS), dtype=np.int64)
    excess = np.full(len(S4_ARMS), -np.inf)
    atom_wins = np.zeros(N_ATOMS, dtype=np.int64)
    weights, n_times = None, 0
    for x, _ in _loader(view, args.num_workers):
        with torch.no_grad():
            rows, valid = trained._full_rows(x)
        if weights is None:
            n_times = rows.shape[-1]
            weights = band_weights(n_times, sfreq).to(rows.device)
        result: BatchScore = score_batch(rows, valid, models, basis_t, weights, slow)
        values = {f.name: getattr(result, f.name).cpu().numpy() for f in fields(BatchScore)}
        for name in SEGMENT_FIELDS:
            parts[name].append(values[name])
        violations += values["ceiling_violations"]
        excess = np.maximum(excess, values["ceiling_excess"])
        atom_wins += values["atom_wins"]
    arrays = {
        name: np.concatenate(chunks, axis=1 if name.startswith("residual") else 0) for name, chunks in parts.items()
    }
    _save_npz(
        path,
        subject=subjects,
        fold=np.array(fold),
        arms=np.array(ARMS),
        bands=np.array(BAND_NAMES),
        n_channels=np.array(trained.n_channels),
        windows_per_segment=np.array(n_times // POOL_LEN),
        ceiling_violations=violations,
        ceiling_excess=excess,
        atom_wins=atom_wins,
        dominant_hz=dominant,
        **arrays,
    )
    trained_ev = pooled_ev(arrays["residual_total"][0], arrays["signal_total"])
    logger.info(f"score fold {fold}: {len(scored)} patients, {len(subjects)} segments, S4-trained EV {trained_ev:.3f}")


# ---------------------------------------------------------------------- read


def _by_patient(values: np.ndarray, index: np.ndarray, n_patients: int) -> np.ndarray:
    """Sums over the segments of each patient; the segment axis is the first."""
    out = np.zeros((n_patients, *values.shape[1:]), dtype=np.float64)
    np.add.at(out, index, values)
    return out


def _validation_ev(readout_root: Path, fold: int) -> float:
    """The best validation EV of a fold's trained dictionary, from its ``sae_training.csv``."""
    path = readout_root / "dictionaries" / f"trained_f{fold}" / "sae_training.csv"
    return 1.0 - float(pl.read_csv(path)["val_residual_frac"].min())


def _interval_record(interval: Interval) -> dict[str, float]:
    return {"value": interval.value, "low": interval.low, "high": interval.high}


def read(root: Path, readout_root: Path) -> dict[str, Any]:
    """Apply the gates and the rule of protocol sections 6 and 7 to a root with five scored folds.

    Raises
    ------
    FileNotFoundError
        If a fold is not scored.
    ValueError
        If a fold file holds other arms or bands.
    """
    plan = json.loads((readout_root / "folds.json").read_text())
    folds_of, labels = plan["folds"], plan["labels"]
    data = []
    for fold in range(N_FOLDS):
        path = score_path(root, fold)
        if not path.is_file():
            raise FileNotFoundError(f"fold {fold} is not scored: run score --fold {fold}")
        with np.load(path) as stored:
            item = {key: stored[key] for key in stored.files}
        if item["arms"].tolist() != list(ARMS) or item["bands"].tolist() != list(BAND_NAMES):
            raise ValueError(f"{path.name} holds other arms or bands")
        data.append(item)
    segment_subject = np.concatenate([d["subject"] for d in data]).astype(str)
    segment_fold = np.concatenate([np.full(len(d["subject"]), int(d["fold"])) for d in data])
    patients = np.unique(segment_subject)
    index = np.searchsorted(patients, segment_subject)
    n = len(patients)

    def stack(name: str) -> np.ndarray:
        return np.concatenate([d[name] for d in data], axis=1 if name.startswith("residual") else 0)

    signal_bands, signal_total = stack("signal_bands"), stack("signal_total")
    residual_bands, residual_total = stack("residual_bands"), stack("residual_total")
    e_bands, e_total = _by_patient(signal_bands, index, n), _by_patient(signal_total, index, n)
    r_bands = _by_patient(residual_bands.transpose(1, 0, 2), index, n).transpose(1, 0, 2)
    r_total = _by_patient(residual_total.T, index, n).T
    fold = np.array([folds_of[s] for s in patients])
    label = np.array([labels[s] for s in patients])
    segments = np.bincount(index, minlength=n)
    fast = fast_bands()
    columns = {"total": (e_total, r_total)}
    columns |= {name: (e_bands[:, b], r_bands[:, :, b]) for b, name in enumerate(BAND_NAMES)}
    columns["above_4"] = (e_bands[:, fast].sum(axis=1), r_bands[:, :, fast].sum(axis=2))
    arm = {name: i for i, name in enumerate(ARMS)}
    counts = bootstrap_counts(fold)

    def ev(name: str, column: str, where: np.ndarray | None = None) -> float:
        signal, residual = columns[column]
        keep = np.ones(n, dtype=bool) if where is None else where
        return pooled_ev(residual[arm[name]][keep], signal[keep])

    def diff(a: str, b: str, column: str) -> Interval:
        signal, residual = columns[column]
        return paired_interval(residual[arm[a]], residual[arm[b]], signal, counts)

    summary = []
    for name in ARMS:
        for column, (signal, residual) in columns.items():
            interval = ev_interval(residual[arm[name]], signal, counts)
            summary.append({"arm": name, "column": column, **_interval_record(interval)})
    pl.DataFrame(summary).write_csv(root / "reconstruction_summary.csv")

    # Gates (section 6); the unit tests run in the launcher before any stage.
    held_out = {f: ev("S4-trained", "total", fold == f) for f in range(N_FOLDS)}
    validation = {f: _validation_ev(readout_root, f) for f in range(N_FOLDS)}
    parseval = max(
        parseval_error(signal_bands, signal_total),
        *(parseval_error(residual_bands[i], residual_total[i]) for i in range(len(ARMS))),
    )
    violations = sum(int(d["ceiling_violations"].sum()) for d in data)
    complete = (
        set(patients.tolist()) == set(labels)
        and all(folds_of[s] == f for s, f in zip(segment_subject, segment_fold))
        and bool((segments == plan["n_segments"]).all())
    )
    gates = {
        "complete": complete,
        "ceiling": violations == 0,
        "rows_match_training": all(abs(held_out[f] - validation[f]) <= ROW_MATCH_TOLERANCE for f in range(N_FOLDS)),
        "training": ev("S4-trained", "total") > ev("S4-untrained", "total"),
        "parseval": parseval <= PARSEVAL_TOLERANCE,
    }

    # Rule (section 7).
    total_vs_pca1 = diff("S4-trained", "PCA-1", "total")
    fast_vs_pca1 = diff("S4-trained", "PCA-1", "above_4")
    fast_vs_fixed = diff("S4-trained", "S4-fixed", "above_4")
    fast_ev = ev("S4-trained", "above_4")
    met = conditions(total_vs_pca1, fast_ev, fast_vs_pca1, fast_vs_fixed)
    outcome = reconstruction_outcome(gates, met)

    # Descriptive reads (section 8).
    curve = [{"k": k, **{column: ev(f"PCA-{k}", column) for column in columns}} for k in PCA_KS]
    pl.DataFrame(curve).write_csv(root / "pca_curve.csv")
    windows = int(data[0]["windows_per_segment"])
    won, slow = stack("windows_won").sum(), stack("windows_slow").sum()
    explained_all, explained_slow = stack("explained_all").sum(), stack("explained_slow").sum()
    ceiling = {c: ev("SVD-1", c) for c in columns}
    record = {
        "rule": "protocol section 7",
        "outcome": outcome,
        "prediction": PREDICTION,
        "gates": gates,
        "gate_details": {
            "held_out_ev": held_out,
            "validation_ev": validation,
            "row_match_tolerance": ROW_MATCH_TOLERANCE,
            "ceiling_violations": violations,
            "ceiling_largest_excess": float(max(float(d["ceiling_excess"].max()) for d in data)),
            "parseval_largest_error": parseval,
        },
        "conditions": met,
        "deciding": {
            "T_total_S4_trained_minus_PCA_1": _interval_record(total_vs_pca1),
            "H1_above_4_S4_trained_ev": fast_ev,
            "H1_floor": FAST_FLOOR,
            "H2_above_4_S4_trained_minus_PCA_1": _interval_record(fast_vs_pca1),
            "H3_above_4_S4_trained_minus_S4_fixed": _interval_record(fast_vs_fixed),
        },
        "bootstrap": {"draws": N_BOOTSTRAP, "confidence": CONFIDENCE, "resampled": "patients within fold"},
        "descriptive": {
            "training_total_S4_trained_minus_S4_untrained": _interval_record(
                diff("S4-trained", "S4-untrained", "total")
            ),
            "bit_matched_S4_trained_minus_PCA_2": {
                c: _interval_record(diff("S4-trained", "PCA-2", c)) for c in ("total", "above_4")
            },
            "smallest_pca_k_reaching_S4_trained": {
                c: smallest_k({k: ev(f"PCA-{k}", c) for k in PCA_KS}, ev("S4-trained", c)) for c in ("total", "above_4")
            },
            "share_of_ceiling": {c: ev("S4-trained", c) / ceiling[c] if ceiling[c] > 0 else None for c in columns},
            "selection_optimism": {f: validation[f] - held_out[f] for f in range(N_FOLDS)},
            "budget": {
                "share_of_windows_with_a_winner": float(won / (len(segment_subject) * windows)),
                "share_of_won_windows_won_by_slow_atoms": float(slow / max(won, 1)),
                "share_of_explained_energy_in_slow_wins": float(explained_slow / explained_all)
                if explained_all
                else None,
                "slow_atom_hz": SLOW_ATOM_HZ,
                "atoms_without_held_out_wins": [int((d["atom_wins"] == 0).sum()) for d in data],
                "slow_atoms": [int((d["dominant_hz"] < SLOW_ATOM_HZ).sum()) for d in data],
            },
            "ev_per_class": {
                group: {name: {c: ev(name, c, label == value) for c in ("total", "above_4")} for name in ARMS}
                for group, value in (("epilepsy", 1), ("non_epilepsy", 0))
            },
        },
        "counts": {
            "patients": n,
            "segments": len(segment_subject),
            "per_fold_and_class": {
                f: {
                    "epilepsy": int(((fold == f) & (label == 1)).sum()),
                    "non_epilepsy": int(((fold == f) & (label == 0)).sum()),
                }
                for f in range(N_FOLDS)
            },
            "rejected_rows": int(sum((d["n_channels"] - d["usable_rows"]).sum() for d in data)),
        },
    }
    _save_json(root / "reconstruction_decision.json", record)
    logger.info(f"Reconstruction baselines: {outcome} (gates {gates}, conditions {met})")
    return record


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("fit", "the spatial patterns and the window PCA of one fold"),
        ("score", "the energies of one fold's patients under every arm"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    reading = commands.add_parser("read", help="apply the gates and the rule")
    reading.add_argument("--root", type=Path, required=True)
    reading.add_argument("--readout-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit_fold(args)
    elif args.command == "score":
        score_fold(args)
    else:
        read(args.root.resolve(), args.readout_root.resolve())


if __name__ == "__main__":
    main()
