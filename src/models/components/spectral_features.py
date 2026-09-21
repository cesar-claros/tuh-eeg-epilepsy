"""Simple spectral features: log band power per channel, a control for the learned dictionaries.

Welch power spectral density per (window, channel) row (Hann window, 50 percent
overlap, one-sided), integrated over fixed bands, as log10 absolute band power and
as the band's share of the total power over all bands. With five bands and 20
channels this gives 200 features per window. No parameters are fitted; the
downstream scaler is fitted on training windows by the readout.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.components.hydra_transform import HydraTransform

_DEFAULT_BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


class SpectralFeatures(nn.Module):
    """Log absolute and relative band power per channel from a Welch PSD.

    Parameters
    ----------
    sfreq : float
        Sampling rate of the windows in Hz.
    bands : tuple[tuple[float, float], ...]
        Frequency bands ``(low, high)`` in Hz.
    nperseg : int
        Welch segment length in samples (512 = 2 s at 256 Hz).
    device : str | None
        ``cpu`` | ``cuda`` | ``auto``.
    """

    def __init__(
        self,
        sfreq: float = 256.0,
        bands: tuple[tuple[float, float], ...] = _DEFAULT_BANDS,
        nperseg: int = 512,
        device: str | None = "auto",
    ) -> None:
        super().__init__()
        self.sfreq = float(sfreq)
        self.bands = tuple((float(lo), float(hi)) for lo, hi in bands)
        self.nperseg = int(nperseg)
        self.device = HydraTransform._resolve_device(device)
        self.register_buffer("window", torch.hann_window(self.nperseg, periodic=False))

    @property
    def n_features_per_channel(self) -> int:
        """Two numbers per band: log absolute power and relative power."""
        return 2 * len(self.bands)

    @torch.no_grad()
    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Features ``(B, C * 2 * n_bands)`` from a window batch ``(B, C, T)``; ``y`` is unused."""
        x = X.to(self.device).float()
        n_windows, n_channels, _ = x.shape
        rows = x.flatten(0, 1)
        spectrum = torch.stft(
            rows, n_fft=self.nperseg, hop_length=self.nperseg // 2, win_length=self.nperseg,
            window=self.window.to(x.device), center=False, return_complex=True,
        )
        psd = spectrum.abs().pow(2).mean(-1)  # (rows, n_fft // 2 + 1), averaged over frames
        freqs = torch.arange(psd.shape[1], device=x.device) * self.sfreq / self.nperseg
        powers = torch.stack(
            [psd[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in self.bands], dim=1
        )  # (rows, n_bands)
        total = powers.sum(1, keepdim=True).clamp_min(1e-20)
        features = torch.cat([torch.log10(powers.clamp_min(1e-20)), powers / total], dim=1)
        return features.view(n_windows, n_channels * features.shape[1])

    def __repr__(self) -> str:
        bands = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in self.bands)
        return f"SpectralFeatures(sfreq={self.sfreq:g}, bands=[{bands}] Hz, nperseg={self.nperseg}, log10 + relative)"


__all__ = ["SpectralFeatures"]
