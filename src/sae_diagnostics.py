"""Full-window diagnostics of a fitted ShapeConv SAE on one split (Phase 2 measurements).

Loads a dictionary (``feature.pretrained=<run>/sae_state.pt``) and the datamodule of the
run, encodes every window of ``split`` (default ``val``) in full, and writes to
``output_dir``:

- ``sae_diagnostics.csv``: per atom, activations per valid channel-minute at two views
  (every NMS peak, and above the extraction threshold: calibrated per-atom thresholds
  when present, else ``amp_min``), the median and 99th percentile of the peak
  magnitudes, the measured background response scale and the threshold-to-response
  ratio, the peak frequency of the signal-domain shape, and the maximum absolute lag
  coherence with any other atom;
- ``sae_atom_coherence.npy``: the ``(n_atoms, n_atoms)`` maximum absolute cross-correlation
  over lags;
- ``sae_atoms_snippets.png``: the signal-domain shape of the most active atoms next to
  their highest-magnitude activations in the split, with the surrounding channels;
- a log summary (flat rows, total rates, coherent pairs).

Rates are activation rates on the chosen development split, not event rates. Run on
the HPC from ``code/``::

    python src/sae_diagnostics.py feature.pretrained=logs/train_sae/runs/<ts>/sae_state.pt \\
        data.signal_mode=bipolar data.filter_freq=[1,45] split=val output_dir=logs/sae_diag/<name>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import polars as pl
import rootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

rootutils.setup_root(__file__, pythonpath=True)

from src.utils import (  # noqa: E402
    RankedLogger,
    check_pretrained_provenance,
    extras,
    instantiate_feature,
    task_wrapper,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)


def _coherence(atoms: np.ndarray) -> np.ndarray:
    """Maximum absolute cross-correlation over lags between unit-norm atoms, ``(K, K)``."""
    bank = torch.from_numpy(atoms).unsqueeze(1)
    return F.conv1d(bank, bank, padding=atoms.shape[1] - 1).abs().amax(-1).numpy()


_HIST_BINS, _HIST_MAX = 4096, 128.0


def _quantile(hist: torch.Tensor, q: float) -> float:
    """Quantile of peak magnitudes from a fixed-bin histogram over ``[0, _HIST_MAX]``; nan when empty."""
    total = float(hist.sum())
    if total == 0:
        return float("nan")
    edge = int(torch.searchsorted(hist.cumsum(0), torch.tensor(q * total)))
    return (min(edge, _HIST_BINS - 1) + 0.5) * _HIST_MAX / _HIST_BINS


def _peak_frequency(shapes: np.ndarray, sfreq: float) -> np.ndarray:
    """Frequency of the largest spectral magnitude of each signal-domain shape, in Hz."""
    spectrum = np.abs(np.fft.rfft(shapes, n=4 * shapes.shape[1], axis=1))
    freqs = np.fft.rfftfreq(4 * shapes.shape[1], d=1.0 / sfreq)
    return freqs[spectrum[:, 1:].argmax(1) + 1]


def _snippet_figure(path: Path, sae: Any, shapes: np.ndarray, snippets: dict[int, list], sfreq: float) -> None:
    """One row per atom: the signal-domain shape, then its top activations with neighbouring channels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atoms = sorted(snippets)
    fig, axes = plt.subplots(len(atoms), 4, figsize=(14, 2.2 * len(atoms)), squeeze=False)
    for row, atom in enumerate(atoms):
        t_shape = np.arange(shapes.shape[1]) / sfreq * 1000.0
        axes[row, 0].plot(t_shape, shapes[atom], color="black")
        axes[row, 0].set_title(f"atom {atom}: signal-domain shape (ms)", fontsize=9)
        for col, (magnitude, segment, channel) in enumerate(snippets[atom][:3], start=1):
            t_seg = np.arange(segment.shape[1]) / sfreq * 1000.0
            offsets = np.arange(segment.shape[0])[:, None] * 4.0
            axes[row, col].plot(t_seg, (segment - offsets).T, color="0.6", linewidth=0.6)
            axes[row, col].plot(t_seg, segment[channel] - offsets[channel], color="tab:red", linewidth=0.9)
            axes[row, col].set_title(f"|a| = {magnitude:.1f}, channel {channel}", fontsize=9)
            axes[row, col].set_yticks([])
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@task_wrapper
def diagnose(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode one split in full and write the per-atom diagnostics.

    Raises
    ------
    ValueError
        If ``feature.pretrained`` is not set or the split is unknown.
    """
    if not cfg.feature.get("pretrained"):
        raise ValueError("set feature.pretrained=<run>/sae_state.pt")
    if cfg.split not in ("train", "val", "test"):
        raise ValueError(f"split must be train, val or test, got {cfg.split!r}")
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    sae = instantiate_feature(cfg.feature)
    check_pretrained_provenance(sae, datamodule, cfg.data, check_val=False)
    sfreq = float(cfg.data.target_sfreq)
    loader = getattr(datamodule, f"{cfg.split}_dataloader")()
    n_atoms = sae.spec.n_atoms
    threshold = sae._event_threshold().flatten()
    threshold_source = "calibrated" if sae.calibrated_thresholds is not None else "amp_min"

    peaks_all = torch.zeros(n_atoms, dtype=torch.long)
    peaks_thr = torch.zeros(n_atoms, dtype=torch.long)
    hist = torch.zeros(n_atoms, _HIST_BINS)
    top: dict[int, list] = {atom: [] for atom in range(n_atoms)}
    rows_total, rows_flat, minutes = 0, 0, 0.0
    margin = sae.spec.atom_len
    with torch.no_grad():
        for x, _ in loader:
            n_windows, n_channels, n_times = x.shape
            rows_total += n_windows * n_channels
            rows_flat += int((x.abs().amax(-1) == 0).sum())
            minutes += n_windows * n_channels * n_times / sfreq / 60.0
            for start, code, _ in sae._row_chunks(x):
                magnitude = code.abs().cpu()
                peaks_all += (magnitude > 0).sum(dim=(0, 2))
                peaks_thr += (magnitude > threshold.cpu().view(1, -1, 1)).sum(dim=(0, 2))
                for atom in range(n_atoms):
                    values = magnitude[:, atom][magnitude[:, atom] > 0]
                    if values.numel():
                        hist[atom] += torch.histc(values.clamp(max=_HIST_MAX), bins=_HIST_BINS, min=0.0, max=_HIST_MAX)
                    for local, sample in zip(*torch.nonzero(magnitude[:, atom] > threshold[atom].cpu(), as_tuple=True)):
                        value = float(magnitude[local, atom, sample])
                        if len(top[atom]) < 3 or value > top[atom][-1][0]:
                            row = start + int(local)
                            w, c = divmod(row, n_channels)
                            lo, hi = max(int(sample) - margin, 0), min(int(sample) + 2 * margin, n_times)
                            top[atom].append((value, x[w, :, lo:hi].numpy(), c))
                            top[atom] = sorted(top[atom], key=lambda item: -item[0])[:3]

    shapes = sae.atoms_signal_domain
    coherence = _coherence(sae.atoms_numpy)
    off = coherence - np.eye(n_atoms)
    response = sae.atom_response_std if sae.atom_response_std is not None else np.full(n_atoms, np.nan)
    thresholds = threshold.cpu().numpy()
    table = pl.DataFrame(
        {
            "atom": np.arange(n_atoms),
            "rate_all_per_channel_minute": peaks_all.numpy() / max(minutes, 1e-9),
            "rate_thresholded_per_channel_minute": peaks_thr.numpy() / max(minutes, 1e-9),
            "peak_median": [_quantile(hist[atom], 0.5) for atom in range(n_atoms)],
            "peak_p99": [_quantile(hist[atom], 0.99) for atom in range(n_atoms)],
            "response_std": response,
            "threshold": thresholds,
            "threshold_over_response": thresholds / np.where(response > 0, response, np.nan),
            "peak_freq_hz": _peak_frequency(shapes, sfreq),
            "max_coherence": off.max(1),
        }
    )
    table.write_csv(output_dir / "sae_diagnostics.csv")
    np.save(output_dir / "sae_atom_coherence.npy", coherence)
    most_active = [int(a) for a in table.sort("rate_thresholded_per_channel_minute", descending=True)["atom"][:8]]
    snippets = {atom: top[atom] for atom in most_active if top[atom]}
    if snippets:
        _snippet_figure(output_dir / "sae_atoms_snippets.png", sae, shapes, snippets, sfreq)

    summary = {
        "split": cfg.split,
        "channel_minutes": minutes,
        "rows_flat_fraction": rows_flat / max(rows_total, 1),
        "threshold_source": threshold_source,
        "total_rate_all": float(table["rate_all_per_channel_minute"].sum()),
        "total_rate_thresholded": float(table["rate_thresholded_per_channel_minute"].sum()),
        "atoms_silent_thresholded": int((peaks_thr == 0).sum()),
        "pairs_coherence_over_0.9": int((np.triu(off, 1) > 0.9).sum()),
        "median_threshold_over_response": float(np.nanmedian(table["threshold_over_response"].to_numpy())),
    }
    log.info(f"SAE diagnostics on {cfg.split}: {summary}")  # noqa: G004
    pl.DataFrame([summary]).write_csv(output_dir / "sae_diagnostics_summary.csv")
    return summary, {"cfg": cfg, "datamodule": datamodule, "feature_extractor": sae, "table": table}


@hydra.main(version_base="1.3", config_path="../configs", config_name="sae_diagnostics.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for the full-window SAE diagnostics.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.
    """
    extras(cfg)
    diagnose(cfg)


if __name__ == "__main__":
    main()
