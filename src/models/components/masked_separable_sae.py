"""Class-masked separable dictionary: shared atoms, epilepsy-only atoms, a mask on the pooled scores and a hinge term.

Mathematics: ``documentation/Separable Atom Encoder Mathematics.md``, section 8. Protocol:
``documentation/masked_separable_sae_protocol.md``. The model is a ``SeparableSAE`` whose atoms split into two groups:

- atoms ``0 .. n_shared - 1`` are shared: every segment may use them;
- the other atoms are epilepsy-only: in training, a segment with label 0 cannot select them (equation 28).

On a label-0 segment, the hinge term (equation 31) penalizes every epilepsy-only atom whose pooled score exceeds the
score of the weakest selected shared atom. Without labels the model selects among all atoms, as at test time (section
8.4), so every function that scores a fitted dictionary works unchanged.

The patient training loop passes the labels of each chunk with ``set_batch_labels`` and reads the weighted hinge with
``batch_penalty``; both are hooks that only this class provides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.components.separable_sae import SeparableSAE, SeparableSpec
from src.models.components.waveform_sae import WaveformTrainSpec

__all__ = ["CHECKPOINT_FORMAT", "MODEL_TAG", "MaskedSeparableSAE", "MaskedSeparableSpec"]

_EPS = 1e-8
MODEL_TAG = "masked_separable_sae"
CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class MaskedSeparableSpec(SeparableSpec):
    """The separable geometry plus the atom groups and the hinge term.

    Parameters
    ----------
    n_shared : int
        Number of shared atoms, ``K_0``. They come first, so they win a tie against an epilepsy-only atom. It must
        be at least ``top_r`` and below ``n_atoms``.
    hinge_weight : float
        The weight ``beta`` of the hinge term in the training loss; 0 gives the mask alone.
    hinge_margin : float
        The margin ``nu`` of the hinge, in the unit of the pooled scores.
    """

    n_shared: int = 0
    hinge_weight: float = 1.0
    hinge_margin: float = 0.0


class MaskedSeparableSAE(SeparableSAE):
    """A separable dictionary with shared and epilepsy-only atoms.

    Same hooks, artifacts and training loop as ``SeparableSAE``, with its own checkpoint tag. ``label_override``
    (patient ID to label) replaces the labels of the loader in training, for the permuted-label control.

    Raises
    ------
    TypeError
        If ``spec`` is not a ``MaskedSeparableSpec``.
    ValueError
        As ``SeparableSAE``, and if ``n_shared`` is not in ``top_r .. n_atoms - 1`` or the hinge weight or margin is
        negative.
    """

    spec: MaskedSeparableSpec
    model_tag = MODEL_TAG
    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(
        self,
        spec: MaskedSeparableSpec | None = None,
        train_spec: WaveformTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
        chunk_windows: int = 8,
    ) -> None:
        super().__init__(spec, train_spec, random_state, device, pretrained, chunk_windows)
        self.label_override: Mapping[str, int] | None = None
        self._batch_labels: torch.Tensor | None = None
        self._penalty: torch.Tensor | None = None
        self._epoch_sums = dict.fromkeys(("hinge", "negative", "leak", "windows", "violations"), 0.0)

    def _check_specs(self) -> None:
        """Also check the atom groups and the hinge."""
        super()._check_specs()
        spec = self.spec
        if not isinstance(spec, MaskedSeparableSpec):
            raise TypeError(f"MaskedSeparableSAE needs a MaskedSeparableSpec, got {type(spec).__name__}")
        shared = spec.n_shared
        if isinstance(shared, bool) or not isinstance(shared, int) or not spec.top_r <= shared < spec.n_atoms:
            raise ValueError(
                f"spec.n_shared must be an integer in top_r .. n_atoms - 1 ({spec.top_r} .. {spec.n_atoms - 1}), "
                f"got {shared!r}"
            )
        if spec.hinge_weight < 0 or spec.hinge_margin < 0:
            raise ValueError("spec.hinge_weight and spec.hinge_margin must not be negative")

    # ------------------------------------------------------------------ hooks of the patient loop

    def resolve_labels(self, labels: torch.Tensor, subjects: Sequence[str]) -> torch.Tensor:
        """The labels that the mask uses: those of the loader, or ``label_override`` by patient ID.

        Raises
        ------
        KeyError
            If ``label_override`` is set and misses a patient.
        """
        if self.label_override is None:
            return labels
        return torch.tensor([float(self.label_override[str(s)]) for s in subjects], dtype=labels.dtype)

    def set_batch_labels(self, labels: torch.Tensor | None) -> None:
        """Set the labels of the segments of the next encoding pass (1 = epilepsy), or ``None`` for no mask."""
        self._batch_labels = None if labels is None else labels.to(self.device)
        self._penalty = None

    def batch_penalty(self) -> torch.Tensor | float:
        """The weighted hinge of the last masked pass, one value per segment; 0 without one."""
        return 0.0 if self._penalty is None else self.spec.hinge_weight * self._penalty

    def epoch_extras(self) -> dict[str, float]:
        """Per-epoch records of the masked passes since the last call: the mean hinge of the label-0 segments, the
        share of their windows in which an epilepsy-only atom would be selected without the mask, and the number of
        code entries of epilepsy-only atoms on label-0 segments (0 by construction; gate 4)."""
        sums = self._epoch_sums
        record = {
            "hinge": sums["hinge"] / max(sums["negative"], 1.0),
            "leak_rate": sums["leak"] / max(sums["windows"], 1.0),
            "mask_violations": sums["violations"],
            "masked_negative_segments": sums["negative"],
        }
        self._epoch_sums = dict.fromkeys(sums, 0.0)
        return record

    # ------------------------------------------------------------------ selection

    def _selection(self, responses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The format-5 selection; with batch labels, label-0 segments cannot select the epilepsy-only atoms.

        Raises
        ------
        ValueError
            If the batch labels do not match the batch of responses.
        """
        labels = self._batch_labels
        if labels is None:
            return super()._selection(responses)
        if responses.ndim != 3 or labels.shape != responses.shape[:1]:
            raise ValueError("batch labels need one value per segment of a (B, K, T) response")
        if responses.shape[-1] == 0 or responses.shape[-1] % self.spec.pool_len:
            raise ValueError("response length must be a positive multiple of pool_len")
        shared = self.spec.n_shared
        rectified = F.relu(responses).unflatten(-1, (-1, self.spec.pool_len))
        values, positions = rectified.max(dim=-1)  # (B, K, M)
        negative = labels == 0
        mask = torch.ones_like(values[:, :, 0])
        mask[negative, shared:] = 0.0
        masked = values * mask.unsqueeze(-1)
        selected = self._select_atoms(masked)
        kept = masked * selected
        code = torch.zeros_like(rectified).scatter(-1, positions.unsqueeze(-1), kept.unsqueeze(-1)).flatten(-2)
        self._penalty = self._hinge(values, negative)
        with torch.no_grad():
            self._epoch_sums["violations"] += float((code[negative][:, shared:] > 0).sum())
        return code, selected

    def _hinge(self, values: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Equation (31) per segment, zero for the segments that are not label 0.

        ``theta`` is the score of the weakest selected shared atom in each window. It and the denominator carry no
        gradient, so the term acts on the epilepsy-only atoms alone.
        """
        shared, top_r = self.spec.n_shared, self.spec.top_r
        theta = values[:, :shared].detach().topk(top_r, dim=1).values[:, -1]  # (B, M)
        excess = F.relu(values[:, shared:] - theta.unsqueeze(1) + self.spec.hinge_margin)
        denominator = (values.shape[1] - shared) * theta.square().sum(-1) + _EPS
        hinge = excess.square().sum((1, 2)) / denominator * negative.to(values.dtype)
        with torch.no_grad():
            leak = (values[:, shared:] > theta.unsqueeze(1)).any(1)  # (B, M): an epilepsy-only atom would enter
            self._epoch_sums["hinge"] += float(hinge.sum())
            self._epoch_sums["negative"] += float(negative.sum())
            self._epoch_sums["leak"] += float(leak[negative].sum())
            self._epoch_sums["windows"] += float(negative.sum()) * values.shape[-1]
        return hinge
