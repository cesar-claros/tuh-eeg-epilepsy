"""Read-only analysis of the saved factors of the stage-1b dictionaries of one gate-1 root.

Protocol: ``documentation/separable_atom_sae_protocol.md``, section 10.8. The script reads only the
result files of every ``1b_*`` arm (``sae_atoms.npy``, ``sae_time_atoms.npy``,
``sae_channel_atoms.npy``, ``sae_atom_stats.csv``, ``sae_channels.txt``, ``synthetic_readout.json``)
and imports no other project module, so it runs anywhere with numpy and polars::

    python src/separable_factor_analysis.py --root logs/separable_synthetic/<gate1 root>

It writes ``factor_summary.csv`` (one row per arm: dominant frequency, band energy, waveform
coherence, rank-1 share or projection similarity) and ``factor_event_atoms.csv`` (one row per
amplitude-6 arm and focus: the signed waveform match and the signed cosine of every projection)
to the root. Format-5 atoms are reduced to their best rank-1 approximation. The template and the
topography repeat the formulas of ``src/separable_synthetic.py`` (``event_template``,
``focal_topography``); ``tests/test_separable_factor_analysis.py`` checks that they agree.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

SFREQ = 256.0
TEMPLATE_SECONDS = 0.3
FOCI = ("T3", "C4")
FOCUS_POTENTIAL, NEIGHBOR_POTENTIAL = -1.0, -0.5
LOWEST_FREQUENCY = 0.5  # Hz; a 1-s atom does not resolve slower periodicity
SPECTRUM_LENGTH = 2048  # zero-padded FFT length (0.125 Hz bins at 256 Hz)
BANDS = (("delta", 0.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0), ("gamma", 30.0, 1e9))
DUPLICATE_COS, DISTINCT_COS = 0.9, 0.5


def spike_wave(length: int) -> np.ndarray:
    """The stage-7 spike-and-wave template of gate 1, unit norm (``event_template(...)[0]`` of the driver)."""
    t = np.linspace(0.0, 1.0, length)
    template = np.exp(-(((t - 0.2) / 0.03) ** 2)) - 0.5 * np.exp(-(((t - 0.6) / 0.15) ** 2))
    template = template - template.mean()
    return template / np.linalg.norm(template)


def focal_topography(channels: list[str], focus: str) -> np.ndarray:
    """Unit bipolar pattern of a negative potential at ``focus`` and half of it at its montage neighbors."""
    pairs = [tuple(name.split("-")) for name in channels]
    neighbors = {e for pair in pairs if focus in pair for e in pair} - {focus}
    potential = {focus: FOCUS_POTENTIAL, **dict.fromkeys(neighbors, NEIGHBOR_POTENTIAL)}
    pattern = np.array([potential.get(a, 0.0) - potential.get(c, 0.0) for a, c in pairs])
    return pattern / np.linalg.norm(pattern)


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """``x`` scaled to unit norm along ``axis``; a zero vector stays zero."""
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-30)


def dominant_frequency(waveforms: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Frequency of the largest spectral power of every waveform ``(K, L)``, at or above ``LOWEST_FREQUENCY``."""
    power = np.abs(np.fft.rfft(waveforms, n=SPECTRUM_LENGTH, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(SPECTRUM_LENGTH, 1.0 / sfreq)
    keep = freqs >= LOWEST_FREQUENCY
    return freqs[keep][power[:, keep].argmax(axis=1)]


def lag_coherence(atoms: np.ndarray) -> float:
    """Median over atom pairs of the largest absolute normalized cross-correlation over lags; atoms ``(K, C, L)``."""
    atoms = unit(atoms.reshape(len(atoms), -1)).reshape(atoms.shape)
    length = atoms.shape[-1]
    spectra = np.fft.rfft(atoms, n=2 * length, axis=-1)
    values: list[float] = []
    for i in range(len(atoms) - 1):
        cross = np.fft.irfft((spectra[i][None] * np.conj(spectra[i + 1 :])).sum(axis=1), n=2 * length, axis=-1)
        values.extend(np.abs(cross).max(axis=-1).tolist())
    return float(np.median(values)) if values else 0.0


def best_lag(waveform: np.ndarray, template: np.ndarray) -> tuple[float, int]:
    """Signed correlation at the lag of the largest absolute correlation, and the template's offset in the waveform."""
    full = np.correlate(waveform, template, mode="full")
    best = int(np.abs(full).argmax())
    return float(full[best] / (np.linalg.norm(waveform) * np.linalg.norm(template))), best - (len(template) - 1)


def arm_factors(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Unit waveforms ``(K, L)``, unit projections ``(K, C, D)`` and, for format 5, the rank-1 energy share ``(K,)``."""
    if (folder / "sae_time_atoms.npy").is_file():
        return (
            unit(np.load(folder / "sae_time_atoms.npy")),
            unit(np.load(folder / "sae_channel_atoms.npy"), axis=1),
            None,
        )
    left, singular, right = np.linalg.svd(np.load(folder / "sae_atoms.npy"), full_matrices=False)
    return right[:, 0], left[:, :, :1], singular[:, 0] ** 2 / (singular**2).sum(axis=1)


def summarize_arm(folder: Path) -> dict[str, Any]:
    """The factor summary of one arm (one row of ``factor_summary.csv``)."""
    waveforms, projections, rank_one = arm_factors(folder)
    with (folder / "sae_atom_stats.csv").open(newline="") as handle:
        stats = {int(row["atom"]): float(row["energy_share"]) for row in csv.DictReader(handle)}
    energy = np.array([stats[k] for k in range(len(waveforms))])
    freq = dominant_frequency(waveforms)
    row: dict[str, Any] = {
        "arm": folder.name,
        "median_dominant_hz": float(np.median(freq)),
        "energy_weighted_hz": float((energy * freq).sum() / max(energy.sum(), 1e-30)),
        **{f"energy_{name}": float(energy[(freq >= lo) & (freq < hi)].sum()) for name, lo, hi in BANDS},
        "waveform_coherence": lag_coherence(waveforms[:, None, :]),
    }
    if rank_one is not None:
        row["multichannel_coherence"] = lag_coherence(np.load(folder / "sae_atoms.npy"))
        row["median_rank_one_share"] = float(np.median(rank_one))
        return row
    cos = np.einsum("kcd,kce->kde", projections, projections)
    upper = np.triu_indices(projections.shape[2], 1)
    pairs = cos[:, upper[0], upper[1]]
    row.update(
        {
            "median_abs_projection_cos": float(np.median(np.abs(pairs))),
            "atoms_with_duplicate_projections": int((pairs.max(axis=1) > DUPLICATE_COS).sum()),
            "atoms_with_opposite_projections": int((pairs.min(axis=1) < -DUPLICATE_COS).sum()),
            "atoms_with_distinct_projections": int((np.abs(pairs).max(axis=1) < DISTINCT_COS).sum()),
        }
    )
    return row


def event_atoms(folder: Path) -> list[dict[str, Any]]:
    """Signed waveform match and signed projection cosines of the focus atoms of an amplitude-6 arm."""
    readout = json.loads((folder / "synthetic_readout.json").read_text())
    if readout["amplitude"] != 6:
        return []
    waveforms, projections, _ = arm_factors(folder)
    channels = (folder / "sae_channels.txt").read_text().split()
    template = spike_wave(round(TEMPLATE_SECONDS * SFREQ))
    rows = []
    for focus in FOCI:
        k = int(readout[f"{focus}_atom"])
        correlation, offset = best_lag(waveforms[k], template)
        cosines = projections[k].T @ focal_topography(channels, focus)
        effective = np.sign(correlation) * cosines
        rows.append(
            {
                "arm": folder.name,
                "focus": focus,
                "atom": k,
                "waveform_correlation": correlation,
                "offset": offset,
                "projection_cosines": json.dumps(np.round(cosines, 3).tolist()),
                "best_effective_cosine": float(effective.max()),
                "worst_effective_cosine": float(effective.min()),
            }
        )
    return rows


def main() -> None:
    """Analyze every stage-1b arm of ``--root`` and write the two CSV files to it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="a gate-1 root with stage-1b arms")
    root = parser.parse_args().root.resolve()
    folders = sorted(p for p in root.glob("1b_*") if p.is_dir() and (p / "sae_atom_stats.csv").is_file())
    if not folders:
        raise SystemExit(f"no finished stage-1b arms under {root}")
    summary = pl.DataFrame([summarize_arm(folder) for folder in folders], infer_schema_length=None)
    events = pl.DataFrame([row for folder in folders for row in event_atoms(folder)], infer_schema_length=None)
    summary.write_csv(root / "factor_summary.csv")
    events.write_csv(root / "factor_event_atoms.csv")
    logger.info(
        f"Wrote factor_summary.csv ({len(summary)} arms) and factor_event_atoms.csv ({len(events)} rows) to {root}"
    )


if __name__ == "__main__":
    main()
