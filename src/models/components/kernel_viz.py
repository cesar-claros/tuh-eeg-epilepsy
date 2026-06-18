"""Plotting helpers for inspecting HYDRA kernels (waveform + frequency response).

These render the time-domain dilated kernel (the matched template, on a real ms
axis) next to its magnitude frequency response in Hz, and a histogram of peak
frequencies split by favored class. matplotlib is imported lazily so importing
this module stays cheap; the functions return ``None`` if it is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
    representation = getattr(info, "representation", "raw")
    taps_ms = [j * dilation / sfreq * 1000.0 for j in range(weight.numel())]
    ax_time.stem(taps_ms, weight.tolist())
    ax_time.axhline(0.0, color="gray", linewidth=0.5)
    ax_time.tick_params(labelsize=6)

    freqs, mag = HydraTransform.kernel_frequency_response(
        weight, dilation, sfreq, representation=representation
    )
    peak = info.peak_freq_hz
    if peak is None:
        peak = HydraTransform.peak_frequency(weight, dilation, sfreq, representation=representation)
    ax_freq.plot(freqs.tolist(), mag.tolist())
    ax_freq.axvline(peak, color="0.3", linestyle="--", linewidth=0.7)
    ax_freq.tick_params(labelsize=6)
    return peak


def plot_kernels(
    infos, sfreq: float, out_path, title: str = "Top HYDRA kernels",
    max_kernels: int = 10, kernels_per_row: int = 2,
):
    """Plot each kernel's time-domain waveform (ms) beside its ``|H(f)|`` (Hz).

    Shows up to ``max_kernels`` kernels (highest-ranked first), laid out
    ``kernels_per_row`` kernels per row; each kernel uses a (time, frequency)
    column pair. The defaults (10 kernels, 2 per row) give a 5x4 grid of panels.

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
    infos = infos[:max_kernels]
    n = len(infos)
    n_rows = int(np.ceil(n / kernels_per_row))
    n_cols = 2 * kernels_per_row
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * kernels_per_row, 1.7 * n_rows), squeeze=False
    )
    for i, info in enumerate(infos):
        r, c = i // kernels_per_row, (i % kernels_per_row) * 2
        ax_t, ax_f = axes[r][c], axes[r][c + 1]
        peak = _draw_kernel(ax_t, ax_f, info, sfreq)
        ax_t.set_title(f"{_label(info)}  (time)", fontsize=7)
        ax_f.set_title(f"peak = {peak:.3g} Hz  (freq)", fontsize=7)
        # Label the x-axis only on the lowest kernel in each column.
        if i + kernels_per_row >= n:
            ax_t.set_xlabel("ms", fontsize=7)
            ax_f.set_xlabel("Hz", fontsize=7)
    # Hide any unused cells when the last row is only partly filled.
    for j in range(n, n_rows * kernels_per_row):
        r, c = j // kernels_per_row, (j % kernels_per_row) * 2
        axes[r][c].axis("off")
        axes[r][c + 1].axis("off")
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
            ax_f.set_title(f"peak = {peak:.3g} Hz", fontsize=7)
            if r == n_rows - 1 or r == len(infos) - 1:
                ax_t.set_xlabel("ms", fontsize=7)
                ax_f.set_xlabel("Hz", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_peak_freq_hist(
    disc_infos, sfreq: float, out_path, bin_width: float = 5.0, fmax: float | None = None,
    title: str = "Peak frequency by favored class", class_names: dict | None = None,
):
    """Histogram of discriminative kernels' peak frequencies, split by favored class.

    Both classes share the same fixed-width bin edges (``bin_width`` Hz, from 0 to
    ``fmax`` rounded up to a whole bin), so the two distributions are directly
    comparable. ``fmax`` defaults to the Nyquist frequency (``sfreq / 2``), the
    largest peak frequency a kernel can have.

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
            peak = HydraTransform.peak_frequency(
                info.weight, info.dilation, sfreq,
                representation=getattr(info, "representation", "raw"),
            )
        by_class.setdefault(info.favors, []).append(peak)

    # Shared fixed-width bin edges from 0 to (rounded-up) fmax, so the two classes
    # use identical bins and can be compared directly. fmax defaults to Nyquist.
    if fmax is None:
        fmax = sfreq / 2.0
    top = float(np.ceil(fmax / bin_width) * bin_width)
    edges = np.arange(0.0, top + bin_width / 2.0, bin_width)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, peaks in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        ax.hist(peaks, bins=edges, alpha=0.5, label=f"favors {_class_name(label, class_names)}")
    ax.set_xlim(0.0, top)
    ax.set_xlabel("peak frequency (Hz)")
    ax.set_ylabel("number of kernels")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_peak_freq_hists(
    panels, sfreq: float, out_path, bin_width: float = 5.0, fmax: float | None = None,
    suptitle: str = "Peak frequency of top kernels by ranking",
    class_names: dict | None = None,
):
    """One figure stacking a peak-frequency histogram per ranking, on shared bins.

    ``panels`` is a list of ``(subtitle, infos)``. If a panel's infos carry a
    ``favors`` attribute they are split by favored class (overlaid, like the
    discriminative ranking); otherwise they are drawn as a single distribution.
    Every panel uses the same fixed-width bins (``bin_width`` Hz, from 0 to ``fmax``
    rounded up to a whole bin; ``fmax`` defaults to Nyquist) and a shared x-axis, so
    the rankings are directly comparable.

    Returns the saved Path, or None if matplotlib is unavailable or no panel has
    any kernels.
    """
    panels = [(t, infos) for t, infos in panels if infos]
    if not panels:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmax is None:
        fmax = sfreq / 2.0
    top = float(np.ceil(fmax / bin_width) * bin_width)
    edges = np.arange(0.0, top + bin_width / 2.0, bin_width)

    def _peak(info):
        p = info.peak_freq_hz
        if p is not None:
            return p
        return HydraTransform.peak_frequency(
            info.weight, info.dilation, sfreq,
            representation=getattr(info, "representation", "raw"),
        )

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.3 * n), sharex=True, squeeze=False)
    for ax, (subtitle, infos) in zip(axes[:, 0], panels):
        if any(getattr(i, "favors", None) is not None for i in infos):
            by_class: dict = {}
            for i in infos:
                by_class.setdefault(i.favors, []).append(_peak(i))
            for label, peaks in sorted(by_class.items(), key=lambda kv: str(kv[0])):
                ax.hist(peaks, bins=edges, alpha=0.5,
                        label=f"favors {_class_name(label, class_names)}")
            ax.legend(fontsize=7)
        else:
            ax.hist([_peak(i) for i in infos], bins=edges, color="steelblue", alpha=0.8)
        ax.set_xlim(0.0, top)
        ax.set_ylabel("kernels")
        ax.set_title(subtitle, fontsize=9)
    axes[-1, 0].set_xlabel("peak frequency (Hz)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
