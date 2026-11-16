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

__all__ = [
    "CHECKPOINT_FORMAT",
    "MODEL_TAG",
    "SLOW_MODEL_TAG",
    "MaskedSeparableSAE",
    "MaskedSeparableSpec",
    "SlowHingeSAE",
    "SlowHingeSpec",
]

_EPS = 1e-8
MODEL_TAG = "masked_separable_sae"
SLOW_MODEL_TAG = "slow_hinge_separable_sae"
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


@dataclass(frozen=True)
class SlowHingeSpec(MaskedSeparableSpec):
    """The masked geometry with the hinge of the redesign (``documentation/masked_hinge_redesign_protocol.md``).

    Parameters
    ----------
    hinge_strength : float
        The strength ``eta`` of the slow hinge step, in Adam steps for an atom that leaks in every label-0 window.
    hinge_start_epoch : int
        The first epoch (1-based) with a hinge step; the epochs before it train with the mask alone.
    hinge_ramp_epochs : int
        The hinge reaches its full strength in this many epochs, linearly; 0 gives full strength at once.

    ``hinge_weight`` is not used: the hinge is not a term of the loss.
    """

    hinge_strength: float = 1.0
    hinge_start_epoch: int = 11
    hinge_ramp_epochs: int = 5


class SlowHingeSAE(MaskedSeparableSAE):
    """A masked dictionary whose hinge acts through a step of its own (section 3 of the redesign protocol).

    The reconstruction loss keeps its Adam step, without a hinge term. During the chunks of a batch the model
    collects the gradient of the hinge (summed over the label-0 segments, without the patient weights of the
    reconstruction loss) and how often each epilepsy-only atom leaks. After the optimizer step,
    ``after_optimizer_step`` moves each such atom against its hinge gradient by
    ``eta * ramp * lr * sqrt(n) * leak fraction``, where ``n`` is the number of stored entries of an atom. The loop
    then applies the unit-norm projection, as after every step.

    Raises
    ------
    TypeError
        If ``spec`` is not a ``SlowHingeSpec``.
    ValueError
        As ``MaskedSeparableSAE``, and if the strength or the ramp is negative or the start epoch is below 1.
    """

    spec: SlowHingeSpec
    model_tag = SLOW_MODEL_TAG
    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(
        self,
        spec: SlowHingeSpec | None = None,
        train_spec: WaveformTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
        chunk_windows: int = 8,
    ) -> None:
        super().__init__(spec, train_spec, random_state, device, pretrained, chunk_windows)
        self._epoch = 1
        self._hinge_grad: torch.Tensor | None = None
        self._leak_count: torch.Tensor | None = None
        self._leak_windows = 0.0
        self._step_sums = dict.fromkeys(("steps", "moved", "norm"), 0.0)

    def _check_specs(self) -> None:
        """Also check the fields of the slow hinge."""
        super()._check_specs()
        spec = self.spec
        if not isinstance(spec, SlowHingeSpec):
            raise TypeError(f"SlowHingeSAE needs a SlowHingeSpec, got {type(spec).__name__}")
        if spec.hinge_strength < 0 or spec.hinge_ramp_epochs < 0 or spec.hinge_start_epoch < 1:
            raise ValueError("the slow hinge needs strength >= 0, ramp epochs >= 0 and a start epoch >= 1")

    def ramp(self, epoch: int | None = None) -> float:
        """The share of the full hinge strength in an epoch (1-based): 0 before the start, then a linear rise."""
        spec = self.spec
        done = (self._epoch if epoch is None else epoch) - (spec.hinge_start_epoch - 1)
        if done <= 0:
            return 0.0
        return 1.0 if spec.hinge_ramp_epochs == 0 else min(1.0, done / spec.hinge_ramp_epochs)

    def selectable_epoch(self, epoch: int) -> bool:
        """Whether the state after an epoch (1-based) can be the selected dictionary: only at full hinge strength.

        The loop selects the epoch with the smallest validation loss. Without this floor it could select an epoch
        before the hinge acts.
        """
        return self.ramp(epoch) >= 1.0

    def _reset_step(self) -> None:
        self._hinge_grad, self._leak_count, self._leak_windows = None, None, 0.0

    def begin_epoch(self, epoch: int) -> None:
        """Set the epoch (1-based) of the next training steps and drop what earlier passes collected."""
        self._epoch = epoch
        self._reset_step()

    def _hinge(self, values: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Also count, per epilepsy-only atom, the label-0 windows in which it would be selected without the mask."""
        hinge = super()._hinge(values, negative)
        shared, top_r = self.spec.n_shared, self.spec.top_r
        with torch.no_grad():
            theta = values[:, :shared].topk(top_r, dim=1).values[:, -1]
            leaks = (values[:, shared:] > theta.unsqueeze(1)) & negative[:, None, None]
            count = leaks.sum((0, 2)).double()
            self._leak_count = count if self._leak_count is None else self._leak_count + count
            self._leak_windows += float(negative.sum()) * values.shape[-1]
        return hinge

    def batch_penalty(self) -> float:
        """No term of the loss. While the hinge is on, collect the gradient of the hinge of the last masked pass."""
        penalty = self._penalty
        if penalty is not None and penalty.requires_grad and self.ramp() > 0 and self.spec.hinge_strength > 0:
            gradient = torch.autograd.grad(penalty.sum(), self.atoms, retain_graph=True, allow_unused=True)[0]
            if gradient is not None:
                self._hinge_grad = gradient if self._hinge_grad is None else self._hinge_grad + gradient
        return 0.0

    @torch.no_grad()
    def after_optimizer_step(self, lr: float) -> None:
        """The slow hinge step of the batch; the caller applies the unit-norm projection afterwards."""
        gradient, count, windows = self._hinge_grad, self._leak_count, self._leak_windows
        self._reset_step()
        scale = self.spec.hinge_strength * self.ramp() * lr
        if gradient is None or count is None or windows == 0 or scale == 0:
            return
        shared = self.spec.n_shared
        rows = gradient[shared:]
        norms = rows.norm(dim=1)
        fraction = (count / windows).to(rows.dtype)
        size = scale * rows.shape[1] ** 0.5 * fraction  # the length of each atom's step
        move = (norms > 0) & (fraction > 0)
        step = torch.zeros_like(rows)
        step[move] = rows[move] / norms[move, None] * size[move, None]
        self.atoms[shared:] -= step
        self._step_sums["steps"] += 1.0
        self._step_sums["moved"] += float(move.sum())
        self._step_sums["norm"] += float(size[move].sum())

    def epoch_extras(self) -> dict[str, float]:
        """The records of ``MaskedSeparableSAE``, with the ramp and the slow steps of the epoch: the mean number of
        atoms moved per step and the mean length of their steps."""
        sums = self._step_sums
        record = super().epoch_extras() | {
            "hinge_ramp": self.ramp(),
            "hinge_atoms_moved": sums["moved"] / max(sums["steps"], 1.0),
            "hinge_step_norm": sums["norm"] / max(sums["moved"], 1.0),
        }
        self._step_sums = dict.fromkeys(sums, 0.0)
        return record
