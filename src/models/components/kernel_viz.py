"""Plotting helpers for inspecting HYDRA kernels (waveform + frequency response).

These render the time-domain dilated kernel (the matched template, on a real ms
axis) next to its magnitude frequency response in Hz, and a histogram of peak
frequencies split by favored class. matplotlib is imported lazily so importing
this module stays cheap; the functions return ``None`` if it is unavailable.
"""

from __future__ import annotations

from pathlib import Path

from src.models.components.hydra_transform import HydraTransform


def _label(info) -> str:
    """Compact one-line label for a KernelInfo / DiscriminativeKernel."""
    base = f"#{info.rank} d{info.dilation} {info.representation} g{info.group}k{info.kernel}"
    favors = getattr(info, "favors", None)
    return f"{base} favors={favors}" if favors is not None else base


def _class_name(label, class_names) -> str:
    """Human-readable class name, falling back to the raw label."""
    if class_names and label in class_names:
        return str(class_names[label])
    return str(label)


def _draw_kernel(ax_time, ax_freq, info, sfreq) -> float:
    """Draw one kernel's waveform (ms) on ax_time and its |H(f)| (Hz) on ax_freq."""
    weight = info.weight
    dilation = info.dilation
    taps_ms = [j * dilation / sfreq * 1000.0 for j in range(weight.numel())]
    ax_time.stem(taps_ms, weight.tolist())
    ax_time.axhline(0.0, color="gray", linewidth=0.5)
    ax_time.tick_params(labelsize=6)

    freqs, mag = HydraTransform.kernel_frequency_response(weight, dilation, sfreq)
    peak = info.peak_freq_hz
    if peak is None:
        peak = HydraTransform.peak_frequency(weight, dilation, sfreq)
    ax_freq.plot(freqs.tolist(), mag.tolist())
    ax_freq.axvline(peak, color="0.3", linestyle="--", linewidth=0.7)
    ax_freq.tick_params(labelsize=6)
    return peak


def plot_kernels(infos, sfreq: float, out_path, title: str = "Top HYDRA kernels"):
    """Plot each kernel's time-domain waveform (ms) beside its ``|H(f)|`` (Hz).

    Returns the saved Path, or None if matplotlib is unavailable or ``infos`` empty.
    """
    if not infos:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(infos)
    fig, axes = plt.subplots(n, 2, figsize=(8, 1.7 * n), squeeze=False)
    for i, info in enumerate(infos):
        peak = _draw_kernel(axes[i][0], axes[i][1], info, sfreq)
        axes[i][0].set_title(f"{_label(info)}  (time)", fontsize=7)
        axes[i][1].set_title(f"peak = {peak:.1f} Hz  (freq)", fontsize=7)
        if i == n - 1:
            axes[i][0].set_xlabel("ms", fontsize=7)
            axes[i][1].set_xlabel("Hz", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_kernels_by_class(
    per_class, sfreq: float, out_path, title: str = "Top discriminative HYDRA kernels",
    class_names: dict | None = None,
):
    """Plot the top kernels for each class side by side (waveform + spectrum).

    Each class gets a pair of columns (time, frequency); rows are the per-class
    ranking. Use this to compare, e.g., the top-5 epilepsy-favoring kernels next
    to the top-5 no-epilepsy-favoring ones.

    Parameters
    ----------
    per_class : dict
        ``{class_label: [DiscriminativeKernel, ...]}`` from
        ``top_discriminative_kernels_per_class``.
    sfreq : float
        Sampling rate in Hz.
    out_path : str | Path
        Where to save the PNG.
    title : str
        Figure title.
    class_names : dict | None
        Optional ``{class_label: name}`` for readable column titles.

    Returns
    -------
    Path | None
        The saved path, or None if matplotlib is unavailable or there is nothing
        to plot.
    """
    classes = [c for c in per_class if per_class[c]]
    if not classes:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = max(len(per_class[c]) for c in classes)
    n_cols = 2 * len(classes)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * len(classes), 1.7 * n_rows), squeeze=False
    )
    for ci, c in enumerate(classes):
        name = _class_name(c, class_names)
        infos = per_class[c]
        for r in range(n_rows):
            ax_t = axes[r][2 * ci]
            ax_f = axes[r][2 * ci + 1]
            if r >= len(infos):
                ax_t.axis("off")
                ax_f.axis("off")
                continue
            info = infos[r]
            peak = _draw_kernel(ax_t, ax_f, info, sfreq)
            ax_t.set_title(
                f"favors {name} #{info.rank} d{info.dilation} {info.representation}",
                fontsize=7,
            )
            ax_f.set_title(f"peak = {peak:.1f} Hz", fontsize=7)
            if r == n_rows - 1 or r == len(infos) - 1:
                ax_t.set_xlabel("ms", fontsize=7)
                ax_f.set_xlabel("Hz", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_peak_freq_hist(
    disc_infos, sfreq: float, out_path, bins: int = 20,
    title: str = "Peak frequency by favored class", class_names: dict | None = None,
):
    """Histogram of discriminative kernels' peak frequencies, split by favored class.

    Returns the saved Path, or None if matplotlib is unavailable or ``disc_infos``
    empty.
    """
    if not disc_infos:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_class: dict = {}
    for info in disc_infos:
        peak = info.peak_freq_hz
        if peak is None:
            peak = HydraTransform.peak_frequency(info.weight, info.dilation, sfreq)
        by_class.setdefault(info.favors, []).append(peak)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, peaks in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        ax.hist(peaks, bins=bins, alpha=0.5, label=f"favors {_class_name(label, class_names)}")
    ax.set_xlabel("peak frequency (Hz)")
    ax.set_ylabel("number of kernels")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
