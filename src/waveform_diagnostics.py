"""Whole-window activation diagnostics of a fitted Waveform SAE on one split.

Loads a dictionary (``feature.pretrained=<run>/sae_state.pt``) and the datamodule of the
run, checks that the splits are subject-disjoint and that the dictionary's training
subjects are not in the diagnosed held-out split, encodes every window of ``split``
(default ``val``) in full, and writes to ``output_dir``:

- ``waveform_diagnostics.csv``: per atom, the dominant channel and its energy fraction,
  the number of channels that hold 90 percent of the atom's energy, the rank-one energy
  fraction (how close the atom is to one waveform times one spatial pattern), the peak
  frequency of the encoded atom, the norm, the checkpoint's energy share and mean
  amplitude, the strong threshold, activations and strong activations per window-minute
  (all windows, negative windows, positive windows), the median, 90th and 99th percentile
  and maximum of the positive coefficients, and the median and interquartile range of the
  intervals between consecutive strong activations;
- ``waveform_subject_rates.csv``: per subject, label, minutes, windows and the strong
  activation rate summed over atoms; ``waveform_subject_atom_rates.csv``: the same per
  subject and atom (long format);
- ``waveform_snippets.png``: for the selected atoms, the signal-domain shape (all channels,
  dominant channel highlighted) next to the largest strong activation of up to three
  subjects, all channels in microvolts with the placement marked;
- ``waveform_diagnostics_summary.csv`` and a log line: windows, subjects and minutes by
  label, the total and strong rates by label, the atoms without a strong activation, the
  selected atoms.

Under format-5 selection, at most ``top_r`` atoms are active in each pooling window, and
each atom at most once per window. An atom's plain rate counts the windows it wins and is
at most ``60 * sfreq / pool_len`` per minute; the plain rates summed over atoms are at most
``60 * sfreq * top_r / pool_len``, a bound reached when every window fills its budget.
The strong rate uses coefficients above ``strong_factor`` times the atom's mean
coefficient on the fit's validation windows. Both are activation rates on a development
split, not event rates. Full-atom shapes and norms ignore window truncation; snippet
shading marks the placement restricted to its pooling window. Run on the HPC from ``code/``::

    python src/waveform_diagnostics.py feature.pretrained=logs/waveform_sae/<root>/P32_free/sae_state.pt \\
        feature.spec.atom_norm=free data.windows_train_csv=... \\
        data.windows_val_csv=... data.windows_test_csv=... split=val 'snippet_atoms=[49,32,42]' \\
        output_dir=logs/waveform_diag/P32_free
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import polars as pl
import rootutils
import torch
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

_HIST_BINS, _HIST_MAX = 4096, 256.0  # coefficient histogram (robust units of the encoded row)
_INTERVAL_BINS, _INTERVAL_MAX_S = 512, 4.0  # strong-activation intervals


def atom_descriptors(atoms: np.ndarray, channel_names: list[str], sfreq: float) -> dict[str, list]:
    """Spatial and spectral descriptors of a dictionary ``(K, C, L)`` in the encoded domain.

    Returns, per atom: ``dominant_channel`` (name or index), ``dominant_energy_frac``,
    ``n_channels_90`` (channels that hold 90 percent of the energy), ``rank_one_frac``
    (energy of the first singular component), ``peak_hz_encoded`` (Hann-windowed
    spectrum of the encoded atom summed over channels, at and above 1 Hz) and ``norm``.
    """
    n_atoms, n_channels, length = atoms.shape
    names = channel_names if len(channel_names) == n_channels else [str(c) for c in range(n_channels)]
    energy = (atoms**2).sum(-1)
    total = np.maximum(energy.sum(-1, keepdims=True), 1e-12)
    fraction = energy / total
    sorted_desc = -np.sort(-fraction, axis=-1)
    n_90 = (np.cumsum(sorted_desc, axis=-1) < 0.9).sum(-1) + 1
    rank_one = np.array([np.linalg.svd(a, compute_uv=False)[0] ** 2 / max(float((a**2).sum()), 1e-12) for a in atoms])
    n_fft = 1024
    spectrum = (np.abs(np.fft.rfft(atoms * np.hanning(length), n=n_fft, axis=-1)) ** 2).sum(1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
    first = int((freqs < 1.0).sum())
    peak = freqs[spectrum[:, first:].argmax(-1) + first]
    dominant = fraction.argmax(-1)
    return {
        "dominant_channel": [names[int(c)] for c in dominant],
        "dominant_energy_frac": fraction.max(-1).tolist(),
        "n_channels_90": n_90.astype(int).tolist(),
        "rank_one_frac": rank_one.tolist(),
        "peak_hz_encoded": peak.tolist(),
        "norm": np.sqrt(total[:, 0]).tolist(),
    }


def quantile_from_hist(hist: np.ndarray, q: float, upper: float) -> float:
    """Value at quantile ``q`` of a histogram over ``[0, upper]`` with equal bins; NaN when empty."""
    total = float(hist.sum())
    if total <= 0:
        return float("nan")
    cumulative = np.cumsum(hist) / total
    index = int(np.searchsorted(cumulative, q))
    return float((min(index, hist.size - 1) + 0.5) * upper / hist.size)


def _robust_std(segment: np.ndarray) -> np.ndarray:
    """Per-channel ``1.4826 * MAD`` of a segment ``(C, n)``, floored at a small value."""
    centered = segment - np.median(segment, axis=1, keepdims=True)
    return np.maximum(1.4826 * np.median(np.abs(centered), axis=1), 1e-9)


def _snippet_figure(
    path: Path, shapes: np.ndarray, dominant: list[int], snippets: dict[int, list], per_atom: int,
    sfreq: float, channel_names: list[str], units: str, padding: tuple[int, int], pool_len: int,
) -> None:
    """One row per atom: the signal-domain shape (all channels), then its largest strong activation per subject.

    Channels are drawn in loader order (alphabetical in this pipeline), which is
    not montage adjacency. The spacing between channels is four times the median
    robust standard deviation of the segment, stated in the panel title, and the
    axis limits are fixed to that spacing; the atom's dominant channel is drawn in
    blue, and the shaded span is the atom placement clipped to its pooling window.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atoms = sorted(snippets)
    fig, axes = plt.subplots(len(atoms), 1 + per_atom, figsize=(4.0 * (1 + per_atom), 3.0 * len(atoms)), squeeze=False)
    for row, atom in enumerate(atoms):
        shape = shapes[atom] - shapes[atom].mean(-1, keepdims=True)
        step = 2.2 * float(np.abs(shape).max()) + 1e-12
        t_shape = np.arange(shape.shape[1]) / sfreq * 1000.0
        ax = axes[row, 0]
        for c in range(shape.shape[0]):
            ax.plot(t_shape, shape[c] - c * step, lw=0.9 if c == dominant[atom] else 0.6,
                    color="#1f6fb2" if c == dominant[atom] else "#333333")
        ax.set_yticks([-c * step for c in range(shape.shape[0])])
        ax.set_yticklabels(channel_names, fontsize=5)
        ax.set_title(f"atom {atom}: signal-domain shape (ms), mean removed per channel", fontsize=7)
        for col, snippet in enumerate(snippets[atom][:per_atom], start=1):
            segment = snippet["segment"]
            spacing = 4.0 * float(np.median(_robust_std(segment)))
            t_seg = (np.arange(segment.shape[1]) + snippet["lo"]) / sfreq
            offsets = np.arange(segment.shape[0])[:, None] * spacing
            ax = axes[row, col]
            ax.plot(t_seg, (segment - offsets).T, color="0.6", linewidth=0.5)
            ax.plot(t_seg, segment[dominant[atom]] - offsets[dominant[atom]], color="#1f6fb2", linewidth=0.9)
            pool_start = snippet["sample"] // pool_len * pool_len
            placement_start = max(snippet["sample"] - padding[0], pool_start)
            placement_end = min(snippet["sample"] + padding[1] + 1, pool_start + pool_len)
            ax.axvspan(placement_start / sfreq, placement_end / sfreq,
                       color="#1f6fb2", alpha=0.08, lw=0)
            ax.set_ylim(-(segment.shape[0]) * spacing, spacing)
            ax.set_yticks(-offsets[:, 0])
            ax.set_yticklabels(channel_names, fontsize=5)
            ax.set_title(f"a = {snippet['value']:.1f}, spacing {spacing:.0f} {units}\n{snippet['where']}", fontsize=6)
            ax.set_xlabel("window time (s)", fontsize=6)
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    fig.suptitle(
        f"selected atoms, largest strong activation per subject; channels in loader order; {units}", fontsize=8
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


@task_wrapper
def diagnose(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Encode one split in full and write the per-atom and per-subject diagnostics.

    Raises
    ------
    ValueError
        If ``feature.pretrained`` is not set, the split is unknown, the checkpoint
        carries no atom statistics, or the split's metadata is not aligned with its
        loader.
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
    check_pretrained_provenance(sae, datamodule, cfg.data, check_val=cfg.split == "val")
    if not sae.atom_stats.get("mean_amplitude"):
        raise ValueError("the checkpoint carries no atom statistics (fit without a validation loader?)")
    sfreq = float(cfg.data.target_sfreq)
    signal_scale = float(cfg.signal_scale)
    loader = getattr(datamodule, f"{cfg.split}_dataloader")()
    meta = getattr(datamodule, f"{cfg.split}_df")
    if len(meta) != len(loader.dataset):
        raise ValueError(f"{cfg.split}_df has {len(meta)} rows but the loader has {len(loader.dataset)} windows")
    channel_names = [str(c) for c in (getattr(loader.dataset, "target_channels", None) or [])]

    atoms = sae._dictionary().detach()
    n_atoms = atoms.shape[0]
    padding = sae.padding
    mean_amplitude = torch.as_tensor(sae.atom_stats["mean_amplitude"], dtype=atoms.dtype, device=atoms.device)
    strong_threshold = float(cfg.strong_factor) * mean_amplitude
    share = np.asarray(sae.atom_stats["energy_share"], dtype=float)
    if cfg.get("snippet_atoms"):
        selected = [int(a) for a in cfg.snippet_atoms]
    else:
        selected = [int(a) for a in np.argsort(-share)[: int(cfg.n_snippet_atoms)]]
    per_atom_snippets = int(cfg.snippets_per_atom)

    counts = torch.zeros(2, n_atoms, dtype=torch.float64)  # positive activations by window label
    strong = torch.zeros(2, n_atoms, dtype=torch.float64)
    hist = torch.zeros(n_atoms, _HIST_BINS, dtype=torch.float64)
    amp_max = torch.zeros(n_atoms, dtype=torch.float64)
    intervals = torch.zeros(n_atoms, _INTERVAL_BINS, dtype=torch.float64)
    minutes = torch.zeros(2, dtype=torch.float64)
    n_windows_by_label = [0, 0]
    subject_strong: dict[str, np.ndarray] = {}
    subject_minutes: dict[str, float] = {}
    subject_windows: dict[str, int] = {}
    subject_label: dict[str, int] = {}
    top: dict[int, list] = {atom: [] for atom in selected}
    subjects_of = meta["subject"].astype(str).to_numpy()
    paths_of = meta["path"].astype(str).to_numpy()
    starts_of = meta["start"].astype(float).to_numpy()
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            labels = torch.as_tensor(y).reshape(-1).long()
            rows, valid = sae._rows(x)
            n_windows, _, n_times = rows.shape
            window_minutes = n_times / sfreq / 60.0
            usable = valid.any(1).cpu()
            for start in range(0, n_windows, sae.chunk_windows):
                chunk = slice(start, min(start + sae.chunk_windows, n_windows))
                code = sae._select(sae._responses(rows[chunk], atoms))
                positive = code > 0
                is_strong = code > strong_threshold.view(1, -1, 1)
                counts_w = positive.sum(-1).double().cpu()  # (b, K)
                strong_w = is_strong.sum(-1).double().cpu()
                amp_max = torch.maximum(amp_max, code.amax(dim=(0, 2)).double().cpu())
                for local in range(code.shape[0]):
                    w = start + local
                    if not bool(usable[w]):
                        continue
                    label = int(labels[w])
                    subject = subjects_of[offset + w]
                    counts[label] += counts_w[local]
                    strong[label] += strong_w[local]
                    minutes[label] += window_minutes
                    n_windows_by_label[label] += 1
                    subject_strong.setdefault(subject, np.zeros(n_atoms))
                    subject_strong[subject] += strong_w[local].numpy()
                    subject_minutes[subject] = subject_minutes.get(subject, 0.0) + window_minutes
                    subject_windows[subject] = subject_windows.get(subject, 0) + 1
                    subject_label[subject] = label
                    values = code[local]
                    for atom in range(n_atoms):
                        kept = values[atom][positive[local, atom]]
                        if kept.numel():
                            hist[atom] += torch.histc(
                                kept.clamp(max=_HIST_MAX).float().cpu(), bins=_HIST_BINS, min=0.0, max=_HIST_MAX
                            ).double()
                        where_strong = torch.nonzero(is_strong[local, atom]).flatten()
                        if where_strong.numel() > 1:
                            gaps = (where_strong[1:] - where_strong[:-1]).float().cpu() / sfreq
                            intervals[atom] += torch.histc(
                                gaps.clamp(max=_INTERVAL_MAX_S), bins=_INTERVAL_BINS, min=0.0, max=_INTERVAL_MAX_S
                            ).double()
                    for atom in selected:
                        if not bool(is_strong[local, atom].any()):
                            continue
                        value, sample = values[atom].max(-1)
                        value, sample = float(value), int(sample)
                        current = next((s for s in top[atom] if s["subject"] == subject), None)
                        if current is not None and value <= current["value"]:
                            continue
                        full = len(top[atom]) >= per_atom_snippets
                        if current is None and full and value <= top[atom][-1]["value"]:
                            continue
                        length = atoms.shape[-1]
                        lo, hi = max(sample - 2 * length, 0), min(sample + 3 * length, x.shape[-1])
                        t_event = starts_of[offset + w] + sample / sfreq
                        recording = Path(paths_of[offset + w]).name
                        where = f"{subject}, window {offset + w}, {recording}, t = {t_event:.2f} s"
                        if current is not None:
                            top[atom].remove(current)
                        top[atom].append({
                            "value": value, "sample": sample, "lo": lo, "where": where, "subject": subject,
                            "segment": np.array(x[w, :, lo:hi].numpy(), copy=True) * signal_scale,
                        })
                        top[atom] = sorted(top[atom], key=lambda item: -item["value"])[:per_atom_snippets]
            offset += n_windows

    total_minutes = float(minutes.sum())
    descriptors = atom_descriptors(sae.atoms_numpy, channel_names, sfreq)
    hist_np, intervals_np = hist.numpy(), intervals.numpy()
    table = pl.DataFrame({
        "atom": list(range(n_atoms)),
        **descriptors,
        "energy_share": share.tolist(),
        "mean_amplitude": [float(v) for v in mean_amplitude.cpu()],
        "strong_threshold": [float(v) for v in strong_threshold.cpu()],
        "rate_per_min": (counts.sum(0) / max(total_minutes, 1e-9)).tolist(),
        "rate_neg_per_min": (counts[0] / max(float(minutes[0]), 1e-9)).tolist(),
        "rate_pos_per_min": (counts[1] / max(float(minutes[1]), 1e-9)).tolist(),
        "strong_rate_per_min": (strong.sum(0) / max(total_minutes, 1e-9)).tolist(),
        "strong_rate_neg_per_min": (strong[0] / max(float(minutes[0]), 1e-9)).tolist(),
        "strong_rate_pos_per_min": (strong[1] / max(float(minutes[1]), 1e-9)).tolist(),
        "amp_q50": [quantile_from_hist(h, 0.5, _HIST_MAX) for h in hist_np],
        "amp_q90": [quantile_from_hist(h, 0.9, _HIST_MAX) for h in hist_np],
        "amp_q99": [quantile_from_hist(h, 0.99, _HIST_MAX) for h in hist_np],
        "amp_max": amp_max.tolist(),
        "strong_interval_median_s": [quantile_from_hist(h, 0.5, _INTERVAL_MAX_S) for h in intervals_np],
        "strong_interval_iqr_s": [
            quantile_from_hist(h, 0.75, _INTERVAL_MAX_S) - quantile_from_hist(h, 0.25, _INTERVAL_MAX_S)
            for h in intervals_np
        ],
        "n_strong_intervals": intervals_np.sum(-1).astype(int).tolist(),
    })
    table.write_csv(output_dir / "waveform_diagnostics.csv")
    subjects = sorted(subject_strong)
    pl.DataFrame({
        "subject": subjects,
        "label": [subject_label[s] for s in subjects],
        "minutes": [subject_minutes[s] for s in subjects],
        "windows": [subject_windows[s] for s in subjects],
        "strong_rate_per_min": [float(subject_strong[s].sum() / max(subject_minutes[s], 1e-9)) for s in subjects],
    }).write_csv(output_dir / "waveform_subject_rates.csv")
    pl.DataFrame({
        "subject": [s for s in subjects for _ in range(n_atoms)],
        "label": [subject_label[s] for s in subjects for _ in range(n_atoms)],
        "atom": [a for _ in subjects for a in range(n_atoms)],
        "strong_rate_per_min": [
            float(subject_strong[s][a] / max(subject_minutes[s], 1e-9)) for s in subjects for a in range(n_atoms)
        ],
    }).write_csv(output_dir / "waveform_subject_atom_rates.csv")
    snippets = {atom: top[atom] for atom in selected if top[atom]}
    if snippets:
        dominant = [int(np.argmax((sae.atoms_numpy[a] ** 2).sum(-1))) for a in range(n_atoms)]
        names = channel_names if len(channel_names) == atoms.shape[1] else [str(c) for c in range(atoms.shape[1])]
        _snippet_figure(
            output_dir / "waveform_snippets.png", sae.atoms_signal_domain, dominant, snippets, per_atom_snippets,
            sfreq, names, str(cfg.signal_units), padding, sae.spec.pool_len,
        )
    summary = {
        "split": cfg.split,
        "n_windows": int(sum(n_windows_by_label)),
        "n_windows_neg": n_windows_by_label[0],
        "n_windows_pos": n_windows_by_label[1],
        "n_subjects": len(subjects),
        "minutes_neg": float(minutes[0]),
        "minutes_pos": float(minutes[1]),
        "strong_factor": float(cfg.strong_factor),
        "rate_per_min_total": float(counts.sum() / max(total_minutes, 1e-9)),
        "strong_rate_per_min_total": float(strong.sum() / max(total_minutes, 1e-9)),
        "strong_rate_per_min_neg": float(strong[0].sum() / max(float(minutes[0]), 1e-9)),
        "strong_rate_per_min_pos": float(strong[1].sum() / max(float(minutes[1]), 1e-9)),
        "atoms_without_strong_activation": int((strong.sum(0) == 0).sum()),
        "snippet_atoms": " ".join(str(a) for a in selected),
    }
    log.info(f"Waveform SAE diagnostics on {cfg.split}: {summary}")  # noqa: G004
    pl.DataFrame([summary]).write_csv(output_dir / "waveform_diagnostics_summary.csv")
    return summary, {"cfg": cfg, "datamodule": datamodule, "feature_extractor": sae, "table": table}


@hydra.main(version_base="1.3", config_path="../configs", config_name="waveform_diagnostics.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point for the whole-window Waveform SAE diagnostics.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.
    """
    extras(cfg)
    diagnose(cfg)


if __name__ == "__main__":
    main()
