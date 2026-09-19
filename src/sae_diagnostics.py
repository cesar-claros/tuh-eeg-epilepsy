"""Full-window diagnostics of a fitted ShapeConv SAE on one split (Phase 2 measurements).

Loads a dictionary (``feature.pretrained=<run>/sae_state.pt``) and the datamodule of the
run, checks that the splits are subject-disjoint and that the dictionary's training
subjects are not in the diagnosed held-out split, encodes every window of ``split``
(default ``val``) in full, and writes to ``output_dir``:

- ``sae_diagnostics.csv``: per atom, activations per usable channel-minute at two views
  (every NMS peak, and above the extraction threshold: calibrated per-atom thresholds
  when present, else ``amp_min``), the thresholded rate on the negative and on the
  positive windows separately, the median and 99th percentile of the peak magnitudes,
  the measured background response scale (RMS of the pre-threshold correlation on the
  fit's validation crops), the threshold-to-response ratio and, in ``shrink`` mode, the
  effective pre-shrink ratio ``(theta + tau) / response``, the peak frequency of the
  signal-domain shape, the maximum absolute lag coherence with any other atom and, under
  ``diff_ar``, the energy fraction of the inverse-whitening tail of the display shape;
- ``sae_subject_rates.csv``: per subject of the split, its label, usable channel-minutes
  and thresholded activation rate summed over atoms;
- ``sae_atom_coherence.npy``: the ``(n_atoms, n_atoms)`` maximum absolute cross-correlation
  over lags; ``sae_encoded_autocorr.npy``: ``(2, 17)``, the normalized autocorrelation of
  the encoded (pre-emphasized, scaled) rows at lags 0 to 16, pooled over rows (dominated
  by high-variance rows) and as the median over usable rows (robust), a whitening check;
- ``sae_atoms_snippets.png``: the signal-domain shape of the most active atoms next to
  their highest-magnitude activation in each of up to three subjects, all channels in
  microvolts with a stated spacing and fixed axis limits, channel names in loader order,
  and the subject, window, recording and time of each snippet;
- ``sae_diagnostics_summary.csv`` and a log line: exposure (attempted, usable, excluded
  channel-minutes), threshold source, total rates (all windows, negative windows,
  positive windows), the median and quartiles of the per-subject rate over negative
  subjects, silent atoms, coherent pairs, the median threshold ratios, the edge
  activation fraction (startup region of the causal filters) and the maximum encoded
  autocorrelation beyond lag 0.

Rates are activation rates on the chosen development split, not event rates, and rows
that the extractor rejects (failed loads, flat channels, non-finite samples) count
neither activations nor exposure. Run on the HPC from ``code/``::

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
    check_split_disjoint,
    extras,
    instantiate_feature,
    task_wrapper,
)

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)

_HIST_BINS, _HIST_MAX = 4096, 128.0
_AUTOCORR_LAGS = 16
_SNIPPETS_PER_ATOM = 3
_SNIPPET_ATOMS = 8


def _coherence(atoms: np.ndarray) -> np.ndarray:
    """Maximum absolute cross-correlation over lags between unit-norm atoms, ``(K, K)``."""
    bank = torch.from_numpy(atoms).unsqueeze(1)
    return F.conv1d(bank, bank, padding=atoms.shape[1] - 1).abs().amax(-1).numpy()


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


def _unwhiten_tail_energy(sae: Any) -> np.ndarray:
    """Energy fraction of the inverse-whitening tail (beyond ``atom_len``) of each display shape; nan unless diff_ar."""
    n_atoms, length = sae.spec.n_atoms, sae.spec.atom_len
    if sae.spec.pre_emphasis != "diff_ar" or sae.spec.ar_order == 0:
        return np.full(n_atoms, np.nan)
    with torch.no_grad():
        shape = sae._unwhiten(sae._project(sae.atoms.detach())).squeeze(1)
    energy = shape.pow(2)
    return (energy[:, length:].sum(1) / energy.sum(1).clamp_min(1e-12)).cpu().numpy()


def _robust_std(segment: np.ndarray) -> np.ndarray:
    """Per-channel ``1.4826 * MAD`` of a segment ``(C, n)``, floored at a small value."""
    centered = segment - np.median(segment, axis=1, keepdims=True)
    return np.maximum(1.4826 * np.median(np.abs(centered), axis=1), 1e-9)


def _snippet_figure(path: Path, shapes: np.ndarray, atom_len: int, snippets: dict[int, list], sfreq: float,
                    channel_names: list[str] | None, units: str) -> None:
    """One row per atom: the signal-domain shape, then its top activations (one per subject) with every channel.

    Channels are drawn in loader order (alphabetical in this pipeline), which is
    not montage adjacency. The spacing between channels is four times the median
    robust standard deviation of the segment, stated in the panel title, and the
    axis limits are fixed to that spacing, so a channel with a large excursion runs
    off the panel instead of compressing the others; the activated channel is red.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atoms = sorted(snippets)
    fig, axes = plt.subplots(len(atoms), 1 + _SNIPPETS_PER_ATOM, figsize=(16, 2.8 * len(atoms)), squeeze=False)
    for row, atom in enumerate(atoms):
        t_shape = np.arange(shapes.shape[1]) / sfreq * 1000.0
        axes[row, 0].plot(t_shape, shapes[atom], color="black")
        if shapes.shape[1] > atom_len + 1:
            axes[row, 0].axvline(atom_len / sfreq * 1000.0, color="0.5", linewidth=0.6, linestyle=":")
            axes[row, 0].set_title(f"atom {atom}: signal-domain shape (ms; right of the line: unwhitening tail)",
                                   fontsize=7)
        else:
            axes[row, 0].set_title(f"atom {atom}: signal-domain shape (ms, unit norm)", fontsize=7)
        for col, snippet in enumerate(snippets[atom][:_SNIPPETS_PER_ATOM], start=1):
            segment, channel = snippet["segment"], snippet["channel"]
            spacing = 4.0 * float(np.median(_robust_std(segment)))
            t_seg = (np.arange(segment.shape[1]) + snippet["lo"]) / sfreq
            offsets = np.arange(segment.shape[0])[:, None] * spacing
            ax = axes[row, col]
            ax.plot(t_seg, (segment - offsets).T, color="0.6", linewidth=0.5)
            ax.plot(t_seg, segment[channel] - offsets[channel], color="tab:red", linewidth=0.9)
            ax.axvline(snippet["sample"] / sfreq, color="tab:blue", linewidth=0.6, linestyle=":")
            ax.set_ylim(-(segment.shape[0]) * spacing, spacing)
            names = channel_names if channel_names and len(channel_names) == segment.shape[0] else None
            ax.set_yticks(-offsets[:, 0])
            ax.set_yticklabels(names if names else [str(c) for c in range(segment.shape[0])], fontsize=5)
            name = names[channel] if names else str(channel)
            ax.set_title(
                f"|a| = {snippet['magnitude']:.1f} on {name}, spacing {spacing:.0f} {units}\n{snippet['where']}",
                fontsize=6,
            )
            ax.set_xlabel("window time (s)", fontsize=6)
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    fig.suptitle(
        f"most active atoms, top activation per subject; channels in loader order (not montage adjacency); {units}",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


@task_wrapper
def diagnose(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode one split in full and write the per-atom diagnostics.

    Raises
    ------
    ValueError
        If ``feature.pretrained`` is not set, the split is unknown, or the split's
        metadata is not aligned with its loader.
    """
    if not cfg.feature.get("pretrained"):
        raise ValueError("set feature.pretrained=<run>/sae_state.pt")
    if cfg.split not in ("train", "val", "test"):
        raise ValueError(f"split must be train, val or test, got {cfg.split!r}")
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    sae = instantiate_feature(cfg.feature)
    # The diagnosed split must be held out from the dictionary: val when diagnosing val
    # (test is always checked); diagnosing train is in-sample by construction.
    check_pretrained_provenance(sae, datamodule, cfg.data, check_val=cfg.split == "val")
    sfreq = float(cfg.data.target_sfreq)
    signal_scale = float(cfg.signal_scale)
    loader = getattr(datamodule, f"{cfg.split}_dataloader")()
    meta = getattr(datamodule, f"{cfg.split}_df")
    if len(meta) != len(loader.dataset):
        raise ValueError(f"{cfg.split}_df has {len(meta)} rows but the loader has {len(loader.dataset)} windows")
    channel_names = list(getattr(loader.dataset, "target_channels", None) or [])
    n_atoms, atom_len = sae.spec.n_atoms, sae.spec.atom_len
    threshold = sae._event_threshold().flatten().cpu()
    threshold_source = "calibrated" if sae.calibrated_thresholds is not None else "amp_min"
    edge_len = atom_len + (sae.spec.ar_order if sae.spec.pre_emphasis == "diff_ar" else 0)

    peaks_all = torch.zeros(n_atoms, dtype=torch.long)
    peaks_thr = torch.zeros(2, n_atoms, dtype=torch.long)  # by window label
    peaks_edge = 0
    hist = torch.zeros(n_atoms, _HIST_BINS)
    top: dict[int, list] = {atom: [] for atom in range(n_atoms)}
    autocorr = torch.zeros(_AUTOCORR_LAGS + 1, dtype=torch.float64)
    row_autocorr: list[torch.Tensor] = []  # per usable row, lags 0 to 16
    minutes = torch.zeros(2)  # usable, by window label
    subjects_of = meta["subject"].astype(str).to_numpy()
    paths_of = meta["path"].astype(str).to_numpy()
    starts_of = meta["start"].astype(float).to_numpy()
    window_counts: list[int] = []
    window_minutes: list[float] = []
    rows_total, rows_usable, n_positions = 0, 0, 0
    attempted_minutes, offset = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            n_windows, n_channels, n_times = x.shape
            labels = torch.as_tensor(y).reshape(-1).long()
            valid = sae._valid_rows(x.flatten(0, 1)).view(n_windows, n_channels)
            row_minutes = n_times / sfreq / 60.0
            rows_total += n_windows * n_channels
            rows_usable += int(valid.sum())
            attempted_minutes += n_windows * n_channels * row_minutes
            for w in range(n_windows):
                minutes[int(labels[w])] += float(valid[w].sum()) * row_minutes
                window_minutes.append(float(valid[w].sum()) * row_minutes)
            encoded = sae._rows(x).double()
            per_row = torch.stack(
                [(encoded[:, lag:] * encoded[:, : encoded.shape[1] - lag]).sum(1) for lag in range(_AUTOCORR_LAGS + 1)],
                dim=1,
            ).cpu()
            autocorr += per_row.sum(0)
            usable_rows = per_row[:, 0] > 0
            row_autocorr.append(per_row[usable_rows] / per_row[usable_rows, :1])
            counts_per_window = torch.zeros(n_windows, dtype=torch.long)
            for start, code, _ in sae._row_chunks(x):
                magnitude = code.abs().cpu()
                above = magnitude > threshold.view(1, -1, 1)
                n_positions = magnitude.shape[2]
                peaks_all += (magnitude > 0).sum(dim=(0, 2))
                peaks_edge += int(above[..., :edge_len].sum())
                for local in range(magnitude.shape[0]):
                    w = (start + local) // n_channels
                    peaks_thr[int(labels[w])] += above[local].sum(dim=-1)
                    counts_per_window[w] += int(above[local].sum())
                for atom in range(n_atoms):
                    values = magnitude[:, atom][magnitude[:, atom] > 0]
                    if values.numel():
                        hist[atom] += torch.histc(values.clamp(max=_HIST_MAX), bins=_HIST_BINS, min=0.0, max=_HIST_MAX)
                    for local, sample in zip(*torch.nonzero(above[:, atom], as_tuple=True)):
                        value = float(magnitude[local, atom, sample])
                        w, c = divmod(start + int(local), n_channels)
                        subject = subjects_of[offset + w]
                        # One snippet per subject and atom: the largest activation of that subject.
                        current = next((s for s in top[atom] if s["subject"] == subject), None)
                        if current is not None and value <= current["magnitude"]:
                            continue
                        full = len(top[atom]) >= _SNIPPETS_PER_ATOM
                        if current is None and full and value <= top[atom][-1]["magnitude"]:
                            continue
                        lo, hi = max(int(sample) - atom_len, 0), min(int(sample) + 2 * atom_len, n_times)
                        t_event = starts_of[offset + w] + int(sample) / sfreq
                        recording = Path(paths_of[offset + w]).name
                        where = f"{subject}, window {offset + w}, {recording}, t = {t_event:.2f} s"
                        if current is not None:
                            top[atom].remove(current)
                        top[atom].append({
                            "magnitude": value, "channel": c, "sample": int(sample), "lo": lo, "where": where,
                            "subject": subject,
                            "segment": np.array(x[w, :, lo:hi].numpy(), copy=True) * signal_scale,
                        })
                        top[atom] = sorted(top[atom], key=lambda item: -item["magnitude"])[:_SNIPPETS_PER_ATOM]
            window_counts.extend(int(v) for v in counts_per_window)
            offset += n_windows

    usable_minutes = float(minutes.sum())
    denominator = max(usable_minutes, 1e-9)
    shapes = sae.atoms_signal_domain
    coherence = _coherence(sae.atoms_numpy)
    off = coherence - np.eye(n_atoms)
    response = sae.atom_response_std if sae.atom_response_std is not None else np.full(n_atoms, np.nan)
    thresholds = threshold.numpy()
    safe_response = np.where(response > 0, response, np.nan)
    effective = thresholds + sae.thresholds if sae.spec.mode == "shrink" else thresholds
    table = pl.DataFrame(
        {
            "atom": np.arange(n_atoms),
            "rate_all_per_channel_minute": peaks_all.numpy() / denominator,
            "rate_thresholded_per_channel_minute": peaks_thr.sum(0).numpy() / denominator,
            "rate_thresholded_neg_per_channel_minute": peaks_thr[0].numpy() / max(float(minutes[0]), 1e-9),
            "rate_thresholded_pos_per_channel_minute": peaks_thr[1].numpy() / max(float(minutes[1]), 1e-9),
            "peak_median": [_quantile(hist[atom], 0.5) for atom in range(n_atoms)],
            "peak_p99": [_quantile(hist[atom], 0.99) for atom in range(n_atoms)],
            "response_rms": response,
            "threshold": thresholds,
            "threshold_over_response": thresholds / safe_response,
            "effective_threshold_over_response": effective / safe_response,
            "peak_freq_hz": _peak_frequency(shapes, sfreq),
            "max_coherence": off.max(1),
            "unwhiten_tail_energy_frac": _unwhiten_tail_energy(sae),
        }
    )
    table.write_csv(output_dir / "sae_diagnostics.csv")
    np.save(output_dir / "sae_atom_coherence.npy", coherence)
    autocorr_norm = (autocorr / autocorr[0].clamp_min(1e-12)).numpy()
    per_row_median = (torch.cat(row_autocorr).median(0).values.numpy() if row_autocorr
                      else np.full(_AUTOCORR_LAGS + 1, np.nan))
    np.save(output_dir / "sae_encoded_autocorr.npy", np.stack([autocorr_norm, per_row_median]))

    subjects = pl.DataFrame({
        "subject": [str(s) for s in meta["subject"]],
        "label": [int(v) for v in meta["epilepsy"]],
        "usable_channel_minutes": window_minutes,
        "thresholded_activations": window_counts,
    }).group_by("subject", "label").agg(pl.col("usable_channel_minutes").sum(), pl.col("thresholded_activations").sum())
    subjects = subjects.with_columns(
        (pl.col("thresholded_activations") / pl.col("usable_channel_minutes").clip(lower_bound=1e-9))
        .alias("rate_thresholded_per_channel_minute")
    ).sort("subject")
    subjects.write_csv(output_dir / "sae_subject_rates.csv")
    negative = subjects.filter(pl.col("label") == 0)["rate_thresholded_per_channel_minute"].to_numpy()

    ranked = table.sort("rate_thresholded_per_channel_minute", descending=True)["atom"]
    most_active = [int(a) for a in ranked[:_SNIPPET_ATOMS]]
    snippets = {atom: top[atom] for atom in most_active if top[atom]}
    if snippets:
        _snippet_figure(output_dir / "sae_atoms_snippets.png", shapes, atom_len, snippets, sfreq, channel_names,
                        cfg.signal_units)

    summary = {
        "split": cfg.split,
        "channel_minutes_attempted": attempted_minutes,
        "channel_minutes_usable": usable_minutes,
        "channel_minutes_excluded": attempted_minutes - usable_minutes,
        "rows_excluded_fraction": (rows_total - rows_usable) / max(rows_total, 1),
        "threshold_source": threshold_source,
        "total_rate_all": float(table["rate_all_per_channel_minute"].sum()),
        "total_rate_thresholded": float(table["rate_thresholded_per_channel_minute"].sum()),
        "total_rate_thresholded_neg": float(table["rate_thresholded_neg_per_channel_minute"].sum()),
        "total_rate_thresholded_pos": float(table["rate_thresholded_pos_per_channel_minute"].sum()),
        "neg_subjects": int(len(negative)),
        "neg_subject_rate_median": float(np.median(negative)) if len(negative) else float("nan"),
        "neg_subject_rate_q25": float(np.quantile(negative, 0.25)) if len(negative) else float("nan"),
        "neg_subject_rate_q75": float(np.quantile(negative, 0.75)) if len(negative) else float("nan"),
        "atoms_silent_thresholded": int((peaks_thr.sum(0) == 0).sum()),
        "pairs_coherence_over_0.9": int((np.triu(off, 1) > 0.9).sum()),
        "median_threshold_over_response": float(np.nanmedian(table["threshold_over_response"].to_numpy())),
        "median_effective_threshold_over_response": float(
            np.nanmedian(table["effective_threshold_over_response"].to_numpy())
        ),
        "edge_activation_fraction": peaks_edge / max(int(peaks_thr.sum()), 1),
        "edge_expected_fraction": edge_len / max(n_positions, 1),
        "encoded_autocorr_max_abs_lag1_16": float(np.abs(autocorr_norm[1:]).max()),
        "encoded_autocorr_row_median_max_abs_lag1_16": float(np.nanmax(np.abs(per_row_median[1:]))),
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
