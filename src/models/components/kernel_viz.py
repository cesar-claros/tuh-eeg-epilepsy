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


def plot_kernels(infos, sfreq: float, out_path, title: str = "Top HYDRA kernels"):
    """Plot each kernel's time-domain waveform (ms) beside its ``|H(f)|`` (Hz).

    Parameters
    ----------
    infos : list
        KernelInfo or DiscriminativeKernel objects.
    sfreq : float
        Sampling rate in Hz, used to place taps in ms and label frequencies.
    out_path : str | Path
        Where to save the PNG.
    title : str, default="Top HYDRA kernels"
        Figure title.

    Returns
    -------
    Path | None
        The saved path, or None if matplotlib is unavailable or ``infos`` empty.
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
        weight = info.weight
        dilation = info.dilation

        # Time domain: taps at their dilated sample positions, x-axis in ms.
        taps_ms = [j * dilation / sfreq * 1000.0 for j in range(weight.numel())]
        ax_t = axes[i][0]
        ax_t.stem(taps_ms, weight.tolist())
        ax_t.axhline(0.0, color="gray", linewidth=0.5)
        ax_t.set_title(f"{_label(info)}  (time)", fontsize=7)
        ax_t.tick_params(labelsize=6)
        if i == n - 1:
            ax_t.set_xlabel("ms", fontsize=7)

        # Frequency domain: magnitude response in Hz, peak marked.
        freqs, mag = HydraTransform.kernel_frequency_response(weight, dilation, sfreq)
        peak = info.peak_freq_hz
        if peak is None:
            peak = HydraTransform.peak_frequency(weight, dilation, sfreq)
        ax_f = axes[i][1]
        ax_f.plot(freqs.tolist(), mag.tolist())
        ax_f.axvline(peak, color="0.3", linestyle="--", linewidth=0.7)
        ax_f.set_title(f"peak = {peak:.1f} Hz  (freq)", fontsize=7)
        ax_f.tick_params(labelsize=6)
        if i == n - 1:
            ax_f.set_xlabel("Hz", fontsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_peak_freq_hist(
    disc_infos, sfreq: float, out_path, bins: int = 20,
    title: str = "Peak frequency by favored class",
):
    """Histogram of discriminative kernels' peak frequencies, split by favored class.

    Parameters
    ----------
    disc_infos : list
        DiscriminativeKernel objects (need ``favors`` and ``weight``/``dilation``).
    sfreq : float
        Sampling rate in Hz.
    out_path : str | Path
        Where to save the PNG.
    bins : int, default=20
        Histogram bins.
    title : str
        Figure title.

    Returns
    -------
    Path | None
        The saved path, or None if matplotlib is unavailable or ``disc_infos`` empty.
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
        ax.hist(peaks, bins=bins, alpha=0.5, label=f"favors {label}")
    ax.set_xlabel("peak frequency (Hz)")
    ax.set_ylabel("number of kernels")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
