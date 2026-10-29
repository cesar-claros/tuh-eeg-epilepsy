"""Separable atom SAE: one waveform per atom shared by all channels, and D unit spatial projections.

Implements ``documentation/Separable Atom Encoder Mathematics.md`` (the equation numbers
below refer to that note); the probe design is ``documentation/separable_atom_sae_protocol.md``.

Every template ``u_{k,d} w_k^T`` is rank-1. The encoder projects the encoded rows onto
every column of ``U_k``, correlates each projection with ``w_k`` (zero padding only at
the segment ends), keeps the largest projection, applies ReLU, takes one maximum per
atom and pooling window, and keeps the top-R atoms of each window with the format-5
rule and tie rules. The tied decoder synthesizes each pooling window on its own, with
its full extension of ``P + L - 1`` samples. With ``loss_support="extended"`` the loss
compares that extension with the segment cut into overlapping windows (equation 22);
with ``"central"`` it scores the window's own ``P`` samples, as format 5 does. The
segment reconstruction behind the statistics is always the central part, which equals
the format-5 decoder output for the same code.

The two factors share one parameter ``atoms`` of shape ``(K, L + C*D)``: ``w_k`` first,
then ``U_k`` flattened in ``(C, D)`` order. The training loops of ``waveform_sae.py`` and
``patient_waveform.py`` handle ``atoms`` as one tensor (one Adam group, one constraint
step, one best-state copy), so they run unchanged. With ``step_matched`` the parameter
stores ``w_k / sqrt(C)`` and ``U_k / sqrt(L)``. Adam moves every stored entry by about
``lr`` per step, so the factors move by ``lr sqrt(C)`` and ``lr sqrt(L)``: the same
relative step as a format-5 entry of size ``1 / sqrt(C L)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from torch import nn
from torch.utils.data import DataLoader

from src.models.components.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec

__all__ = ["CHECKPOINT_FORMAT", "MODEL_TAG", "SeparableSAE", "SeparableSpec"]

_EPS = 1e-8
_LOSS_SUPPORTS = ("extended", "central")
MODEL_TAG = "separable_sae"
CHECKPOINT_FORMAT = 2  # format 1 stored the factors unscaled


@dataclass(frozen=True)
class SeparableSpec(WaveformSpec):
    """The format-5 geometry plus the spatial projections and the loss support.

    Parameters
    ----------
    n_projections : int
        Spatial projections ``D`` per atom (equation 3).
    loss_support : str
        ``extended``: score every decoded window over its ``P + L - 1`` samples against
        the overlapping target windows (equations 22 to 24). ``central``: score the
        window's own ``P`` samples, as format 5 does.
    step_matched : bool
        Store ``w_k / sqrt(C)`` and ``U_k / sqrt(L)``, so that Adam gives the factors the
        relative step of a format-5 entry at the same learning rate (learning rates
        ``lr sqrt(C)`` and ``lr sqrt(L)``). The model function does not change.

    ``atom_norm`` must stay ``unit`` and ``center_atoms`` false: equation (4) normalizes
    ``w_k`` and every ``u_{k,d}`` separately.
    """

    n_projections: int = 4
    loss_support: str = "extended"
    step_matched: bool = True


class SeparableSAE(WaveformSAE):
    """Rank-1 templates with a maximum over spatial projections, format-5 selection and per-window synthesis.

    Same hooks, artifacts and training loops as ``WaveformSAE`` (patient or segment
    sampling, ``save_artifacts``, ``validate_artifact``, provenance), with its own
    checkpoint tag and format. ``forward`` returns the format-5 features. ``decode`` is
    unavailable, because synthesis needs the winning projection of every activation.

    Parameters
    ----------
    spec : SeparableSpec | None
        Geometry, projections and loss support; ``None`` uses the defaults.
    train_spec : WaveformTrainSpec | None
        Unsupervised schedule; must score complete segments without re-centering.
        ``None`` uses the defaults with ``recenter_atoms=False``.
    random_state : int
        Seed for the initialization and the batch order.
    device : str | None
        ``cpu`` | ``cuda`` | ``cuda:N`` | ``auto``.
    pretrained : str | Path | None
        A ``sae_state.pt`` written by this class.
    chunk_windows : int
        Complete segments encoded per pass during validation.

    Raises
    ------
    TypeError
        If ``spec`` is not a ``SeparableSpec``.
    ValueError
        As ``WaveformSAE``, and if ``n_projections`` is not a positive integer,
        ``loss_support`` is unknown, the atoms are not unit-normed or are centered,
        training does not score complete segments, or re-centering is enabled.
    """

    spec: SeparableSpec
    model_tag = MODEL_TAG
    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(
        self,
        spec: SeparableSpec | None = None,
        train_spec: WaveformTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
        chunk_windows: int = 8,
    ) -> None:
        super().__init__(
            spec or SeparableSpec(),
            train_spec or WaveformTrainSpec(recenter_atoms=False),
            random_state,
            device,
            pretrained,
            chunk_windows,
        )

    def _check_specs(self) -> None:
        """Also refuse the options that the separable factors cannot use."""
        super()._check_specs()
        spec, train_spec = self.spec, self.train_spec
        if not isinstance(spec, SeparableSpec):
            raise TypeError(f"SeparableSAE needs a SeparableSpec, got {type(spec).__name__}")
        n_projections = spec.n_projections
        if isinstance(n_projections, bool) or not isinstance(n_projections, int) or n_projections < 1:
            raise ValueError(f"spec.n_projections must be a positive integer, got {n_projections!r}")
        if spec.loss_support not in _LOSS_SUPPORTS:
            raise ValueError(f"spec.loss_support must be one of {_LOSS_SUPPORTS}, got {spec.loss_support!r}")
        if not isinstance(spec.step_matched, bool):
            raise ValueError(f"spec.step_matched must be a boolean, got {spec.step_matched!r}")
        if spec.atom_norm != "unit" or spec.center_atoms:
            raise ValueError("the separable SAE needs atom_norm='unit' and center_atoms=False (equation 4)")
        if train_spec.crop_sampling != "full" or train_spec.recenter_atoms:
            raise ValueError("the separable SAE needs crop_sampling='full' and recenter_atoms=False")

    # ------------------------------------------------------------------ factors

    def _scales(self, n_channels: int) -> tuple[float, float]:
        """Factors per stored unit of the waveforms and of the projections: ``(sqrt(C), sqrt(L))`` or ``(1, 1)``."""
        if not self.spec.step_matched:
            return 1.0, 1.0
        return math.sqrt(n_channels), math.sqrt(self.spec.atom_len)

    def _split(self, atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Waveforms ``(K, L)`` and spatial projections ``(K, C, D)`` of packed atoms ``(K, L + C*D)``."""
        length, n_projections = self.spec.atom_len, self.spec.n_projections
        n_channels = (atoms.shape[1] - length) // n_projections
        time_scale, channel_scale = self._scales(n_channels)
        channel = atoms[:, length:].unflatten(1, (n_channels, n_projections))
        return time_scale * atoms[:, :length], channel_scale * channel

    def _pack(self, time: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        """Packed atoms ``(K, L + C*D)`` of waveforms ``(K, L)`` and projections ``(K, C, D)``: ``_split`` inverted."""
        time_scale, channel_scale = self._scales(channel.shape[1])
        return torch.cat([time / time_scale, channel.flatten(1) / channel_scale], dim=1)

    def _constrain(self, atoms: torch.Tensor) -> torch.Tensor:
        """Unit norm for every waveform and every spatial projection (equation 4); a zero factor stays zero."""
        time, channel = self._split(atoms)
        time = time / torch.linalg.vector_norm(time, dim=1, keepdim=True).clamp_min(_EPS)
        channel = channel / torch.linalg.vector_norm(channel, dim=1, keepdim=True).clamp_min(_EPS)
        return self._pack(time, channel)

    @property
    def n_channels(self) -> int | None:
        """Channels of the spatial projections, ``None`` before they exist."""
        if self.atoms is None:
            return None
        return (int(self.atoms.shape[1]) - self.spec.atom_len) // self.spec.n_projections

    def _atom_norms(self) -> torch.Tensor:
        """Largest template norm ``||w_k|| max_d ||u_{k,d}||`` of every atom ``(K,)``: 1 under equation (4)."""
        time, channel = self._split(self._dictionary().detach())
        return torch.linalg.vector_norm(time, dim=1) * torch.linalg.vector_norm(channel, dim=1).amax(-1)

    def _median_coherence(self) -> float:
        """Median lag coherence of the unit waveforms ``w_k``, the factors that all channels share."""
        return self._lag_coherence(self._split(self._dictionary().detach())[0].unsqueeze(1))

    def _templates(self, integrated: bool) -> np.ndarray:
        """Effective templates ``u_{k,d} w_k^T`` ``(K, D, C, L)``; integrated waveforms give ``L + 1`` samples."""
        time, channel = self._split(self._dictionary().detach())
        if integrated:
            time = F.pad(time.cumsum(-1), (1, 0))
        return torch.einsum("kcd,kl->kdcl", channel, time).cpu().numpy()

    @property
    def atoms_numpy(self) -> np.ndarray:
        """Effective templates ``(K, D, C, L)`` in the encoded domain."""
        return self._templates(integrated=False)

    @property
    def atoms_signal_domain(self) -> np.ndarray:
        """Templates in the scaled signal domain: integrated waveforms ``(K, D, C, L + 1)`` under ``diff``."""
        return self._templates(integrated=self.spec.pre_emphasis == "diff")

    def save_artifacts(self, output_dir: Path) -> None:
        """``WaveformSAE.save_artifacts`` with ``(K, D, C, L)`` templates, plus both factors.

        ``sae_time_atoms.npy`` holds the waveforms ``(K, L)`` and
        ``sae_channel_atoms.npy`` the spatial projections ``(K, C, D)``.
        """
        super().save_artifacts(output_dir)
        time, channel = self._split(self._dictionary().detach().cpu())
        np.save(Path(output_dir) / "sae_time_atoms.npy", time.numpy())
        np.save(Path(output_dir) / "sae_channel_atoms.npy", channel.numpy())

    # ------------------------------------------------------------------ encoder and decoder

    def _correlate(self, padded: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Valid correlations ``(B, K, D, T_in - L + 1)`` of rows ``(B, C, T_in)`` with every template (equation 12).

        The projection comes first, which is cheaper while ``D < C`` (section 6.2 of the
        note): the rows are projected onto every column of ``U_k``, then one grouped
        ``conv1d`` correlates each projection with ``w_k``.
        """
        time, channel = self._split(atoms)
        n_atoms, _, n_projections = channel.shape
        projected = torch.einsum("kcd,bct->bkdt", channel, padded).flatten(1, 2)
        kernels = time.repeat_interleave(n_projections, dim=0).unsqueeze(1)
        responses = F.conv1d(projected, kernels, groups=n_atoms * n_projections)
        return responses.unflatten(1, (n_atoms, n_projections))

    def _responses(self, rows: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Largest projection response ``(B, K, T)`` before ReLU (equations 7 to 9), padded at the segment ends."""
        return self._correlate(F.pad(rows, self.padding), atoms).max(dim=2).values

    def _decode(self, code: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Unavailable: synthesis needs the winning projection of every activation.

        Raises
        ------
        NotImplementedError
            Always; ``_encode_reconstruct`` decodes together with the encoder.
        """
        raise NotImplementedError("the separable decoder needs the winning projection of every activation")

    def _decode_extended(self, code: torch.Tensor, projection: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """Per-window synthesis ``(B, C, M, P + L - 1)`` without cropping (equation 23).

        ``projection`` ``(B, K, T)`` holds ``d_k^star`` (equation 10). The code is lifted
        to ``(B, K, D, T)`` at the winning projection, one grouped ``conv_transpose1d``
        synthesizes every window with ``w_k``, and ``U_k`` mixes the channels.
        """
        time, channel = self._split(atoms)
        n_atoms, _, n_projections = channel.shape
        shape = (code.shape[0], n_atoms, n_projections, code.shape[-1])
        lifted = code.new_zeros(shape).scatter(2, projection.unsqueeze(2), code.unsqueeze(2))
        kernels = time.repeat_interleave(n_projections, dim=0).unsqueeze(1)
        temporal = F.conv_transpose1d(self._pool_batch(lifted.flatten(1, 2)), kernels, groups=n_atoms * n_projections)
        synthesis = torch.einsum("kcd,nkdj->ncj", channel, temporal.unflatten(1, (n_atoms, n_projections)))
        return synthesis.unflatten(0, (code.shape[0], -1)).transpose(1, 2)

    def _encode_reconstruct(
        self, rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Code, selection, segment reconstruction, and the target and estimate on the loss support.

        The segment reconstruction ``(B, C, T)`` is the central part of every extended
        window. Under ``extended`` the target is the segment cut into overlapping
        windows of ``P + L - 1`` samples (equation 22) and the estimate is equation
        (23); under ``central`` both are the segment samples.
        """
        padded = F.pad(rows, self.padding)
        responses, projection = self._correlate(padded, atoms).max(dim=2)
        code, selected = self._selection(responses)
        extended = self._decode_extended(code, projection, atoms) * valid.to(rows.dtype)[..., None, None]
        left, pool = self.padding[0], self.spec.pool_len
        reconstruction = extended[..., left : left + pool].flatten(-2)
        if self.spec.loss_support == "central":
            return code, selected, reconstruction, rows, reconstruction
        target = padded.unfold(-1, pool + self.spec.atom_len - 1, pool)
        return code, selected, reconstruction, target, extended

    # ------------------------------------------------------------------ initialization

    def _init_atoms(self, dataloader: DataLoader, generator: torch.Generator) -> None:
        """Create the packed factors from data patches or Gaussian draws, then impose equation (4).

        ``patches``: draw ``K * D`` patches with the format-5 sampler, whose first ``K``
        are the format-5 patches under the same seed. ``w_k`` is the top right singular
        vector of patch ``k``, and ``u_{k,d} = X w_k / ||X w_k||`` for patch
        ``X = d * K + k``, so every template responds positively at its source patch.
        ``random``: Gaussian factors.
        """
        spec = self.spec
        n_atoms, n_projections = spec.n_atoms, spec.n_projections
        if self.train_spec.init == "random":
            n_channels = int(dataloader.dataset[0][0].shape[0])
            time = torch.randn(n_atoms, spec.atom_len, generator=generator)
            channel = torch.randn(n_atoms, n_channels, n_projections, generator=generator)
            dtype = time.dtype
        else:
            patches = self._sample_patches(dataloader, generator, count=n_atoms * n_projections)
            dtype = patches.dtype
            sources = patches.detach().cpu().double().unflatten(0, (n_projections, n_atoms))
            time = torch.linalg.svd(sources[0], full_matrices=False).Vh[:, 0]
            channel = torch.einsum("dkcl,kl->kcd", sources, time)
        atoms = self._pack(time, channel).to(self.device, dtype)
        self.atoms = nn.Parameter(self._constrain(atoms))
        log.info(
            f"Initialized {n_atoms} separable atoms ({self.n_channels} channels, {n_projections} projections, "
            f"{self.train_spec.init})"
        )
