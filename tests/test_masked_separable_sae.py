"""Checks of the class-masked separable dictionary, ``src/models/components/masked_separable_sae.py`` (no corpus; CPU).

Runs under pytest. Covered: the checks of the specification; the model without labels against ``SeparableSAE`` with
the same factors; the masked selection against a brute-force selection, with ties; the hinge on known scores, its
zero exactly when no epilepsy-only atom would be selected without the mask, and its gradient; no gradient of the
reconstruction to a masked atom; the hooks of the patient loop (labels per chunk, the weighted penalty, the permuted
labels, the epoch records).

The slow hinge of the redesign (``SlowHingeSAE``; ``documentation/masked_hinge_redesign_protocol.md``, section 3): the
checks of its specification, the ramp against the protocol's formula, the step on a known gradient and leak count (its
length, no step for an atom that does not leak, no step on a shared atom), the leak count of each atom on known
scores, the unchanged reconstruction gradient, no step before the first hinge epoch, and the patient loop end to end
(the warm epochs equal the training with the mask alone, the selected epoch has the full hinge, the checkpoint tag).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rootutils
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.masked_separable_sae import (  # noqa: E402
    MaskedSeparableSAE,
    MaskedSeparableSpec,
    SlowHingeSAE,
    SlowHingeSpec,
)
from src.models.components.patient_waveform import reconstruction_backward, segment_values  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.waveform_sae import WaveformTrainSpec  # noqa: E402

N_CHANNELS = 3
GEOMETRY = {"n_atoms": 6, "atom_len": 8, "pool_len": 8, "n_projections": 2, "top_r": 2}
SHARED = 4


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _model(**kwargs: object) -> MaskedSeparableSAE:
    """A CPU model with K=6 (4 shared, 2 epilepsy-only), L=8, P=8, D=2, R=2 and Gaussian factors over 3 channels."""
    spec = MaskedSeparableSpec(**{**GEOMETRY, "n_shared": SHARED, **kwargs})
    model = MaskedSeparableSAE(
        spec=spec, train_spec=WaveformTrainSpec(recenter_atoms=False, segment_batch=2), device="cpu"
    )
    width = spec.atom_len + N_CHANNELS * spec.n_projections
    model.atoms = nn.Parameter(model._constrain(_randn(spec.n_atoms, width, seed=7)))
    return model


def _responses(values: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """Responses ``(B, K, M * pool)`` whose pooled score in every window is ``values`` ``(B, K, M)``."""
    out = torch.full((*values.shape, pool), -1.0, dtype=values.dtype)
    out[..., 3] = values
    return out.flatten(-2)


def _brute_force(values: np.ndarray, labels: np.ndarray, shared: int, top_r: int) -> np.ndarray:
    """The pooled values that the masked selection keeps: per window, the ``top_r`` best atoms by (value, index)."""
    kept = np.zeros_like(values)
    for b in range(values.shape[0]):
        for m in range(values.shape[2]):
            column = np.maximum(values[b, :, m], 0.0)
            if labels[b] == 0:
                column[shared:] = 0.0
            order = sorted(range(len(column)), key=lambda k: (-column[k], k))[:top_r]
            kept[b, order, m] = column[order]
    return kept


def test_spec_checks() -> None:
    for bad in (1, 6, 7, True, 2.0):  # below top_r, not below n_atoms, not an integer
        with pytest.raises(ValueError, match="n_shared"):
            _model(n_shared=bad)
    with pytest.raises(ValueError, match="hinge"):
        _model(hinge_weight=-1.0)
    with pytest.raises(ValueError, match="hinge"):
        _model(hinge_margin=-0.1)
    with pytest.raises(TypeError):
        MaskedSeparableSAE(spec=SeparableSpec(**GEOMETRY), device="cpu")
    assert MaskedSeparableSAE.model_tag != SeparableSAE.model_tag  # a separable checkpoint is refused, and the reverse


def test_without_labels_equals_the_separable_model() -> None:
    model = _model()
    plain = SeparableSAE(spec=SeparableSpec(**GEOMETRY), device="cpu")
    plain.atoms = nn.Parameter(model.atoms.detach().clone())
    rows = _randn(3, N_CHANNELS, 4 * GEOMETRY["pool_len"], seed=2)
    valid = torch.ones(3, N_CHANNELS, dtype=torch.bool)
    ours = model._encode_reconstruct(rows, valid, model._dictionary())
    theirs = plain._encode_reconstruct(rows, valid, plain._dictionary())
    for a, b in zip(ours, theirs, strict=True):
        assert torch.equal(a, b)
    assert model.batch_penalty() == 0.0  # no labels: no hinge


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_masked_selection_matches_brute_force(seed: int) -> None:
    model = _model()
    values = _randn(5, 6, 7, seed=seed)
    values[0, 4, 0] = values[0, :4, 0].max() + 1.0  # an epilepsy-only atom that would win in a label-0 segment
    values[1, 1, 2] = values[1, 0, 2] = values[1, 5, 2] = 9.0  # a tie: the lower index wins, shared before the rest
    values[2, :, 3] = -1.0  # a window without a positive candidate
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    model.set_batch_labels(labels)
    code, selected = model._selection(_responses(values))
    kept = code.unflatten(-1, (-1, 8)).amax(-1)
    expected = _brute_force(values.numpy(), labels.numpy(), SHARED, 2)
    assert np.array_equal(kept.numpy(), expected)
    assert torch.equal(selected.sum(1), torch.full((5, 7), 2))  # R atoms per window, zero candidates included
    assert not selected[:3, SHARED:].any() and float(code[:3, SHARED:].abs().sum()) == 0.0  # the mask
    assert bool((code[3:, SHARED:] > 0).any())  # label-1 segments can use the epilepsy-only atoms
    assert kept[1, 0, 2] == 9.0 and kept[1, 1, 2] == 9.0 and kept[1, 5, 2] == 0.0
    assert model.epoch_extras()["mask_violations"] == 0.0
    with pytest.raises(ValueError, match="one value per segment"):
        model.set_batch_labels(labels[:4])
        model._selection(_responses(values))


def test_hinge_on_known_scores() -> None:
    model = _model(hinge_margin=0.0)
    values = torch.tensor(
        [
            [[5.0, 1.0], [4.0, 2.0], [3.0, 0.5], [1.0, 0.1], [6.0, 1.5], [2.0, 3.0]],  # label 0
            [[5.0, 1.0], [4.0, 2.0], [3.0, 0.5], [1.0, 0.1], [9.0, 9.0], [9.0, 9.0]],  # label 1
        ],
        dtype=torch.float64,
    )
    model.set_batch_labels(torch.tensor([0.0, 1.0]))
    model._selection(_responses(values))
    # Window 0: theta = 4 (second best shared), excess (6 - 4, 0). Window 1: theta = 1, excess (0.5, 2).
    expected = ((6 - 4) ** 2 + 0.5**2 + 2**2) / (2 * (4**2 + 1**2) + 1e-8)
    penalty = model._penalty
    assert abs(float(penalty[0]) - expected) < 1e-12 and float(penalty[1]) == 0.0
    assert torch.equal(torch.as_tensor(model.batch_penalty()), model.spec.hinge_weight * penalty)
    extras = model.epoch_extras()
    assert abs(extras["hinge"] - expected) < 1e-12 and extras["leak_rate"] == 1.0  # both windows of the label-0 segment
    assert extras["masked_negative_segments"] == 1.0 and model.epoch_extras()["masked_negative_segments"] == 0.0
    wide = _model(hinge_margin=1.0)
    wide.set_batch_labels(torch.tensor([0.0, 1.0]))
    wide._selection(_responses(values))
    assert float(wide._penalty[0]) > expected  # a margin adds the atoms that come close to the threshold


def test_hinge_is_zero_exactly_without_a_leak() -> None:
    model = _model()
    seen = {True: 0, False: 0}
    for seed in range(60):
        values = _randn(4, 6, 5, seed=seed)
        values[:, SHARED:] *= 0.1 if seed % 2 else 1.5  # rare leaks in odd draws, frequent leaks in even draws
        responses = _responses(values)
        model.set_batch_labels(torch.zeros(4))
        model._selection(responses)
        penalty = model._penalty
        model.set_batch_labels(None)
        unmasked, _ = model._selection(responses)  # the test-time code: all atoms compete
        leaks = (unmasked[:, SHARED:] > 0).flatten(1).any(1)
        assert torch.equal(penalty > 0, leaks)
        for flag in leaks.tolist():
            seen[flag] += 1
    assert seen[True] > 10 and seen[False] > 10  # both cases occur


def test_gradients() -> None:
    model = _model(hinge_weight=0.0)
    rows = _randn(4, N_CHANNELS, 4 * GEOMETRY["pool_len"], seed=5)
    valid = torch.ones(4, N_CHANNELS, dtype=torch.bool)
    model.set_batch_labels(torch.zeros(4))
    _, _, _, target, estimate = model._encode_reconstruct(rows, valid, model._dictionary())
    gradient = torch.autograd.grad((target - estimate).square().sum(), model.atoms, retain_graph=True)[0]
    assert float(gradient[SHARED:].abs().sum()) == 0.0 and float(gradient[:SHARED].abs().sum()) > 0.0
    hinge = torch.autograd.grad(model._penalty.sum(), model.atoms, allow_unused=True)[0]
    assert float(hinge[:SHARED].abs().sum()) == 0.0  # the threshold carries no gradient
    assert float(model._penalty.detach().sum()) == 0.0 or float(hinge[SHARED:].abs().sum()) > 0.0


def test_patient_loop_hooks() -> None:
    model = _model(hinge_weight=2.0)
    model.atoms = nn.Parameter(model.atoms.detach().float())  # the loop reads float32 segments
    x = _randn(5, N_CHANNELS, 4 * GEOMETRY["pool_len"], seed=9).float()
    patient = torch.tensor([0, 0, 1, 1, 2])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    plain, _ = reconstruction_backward(model, x, patient)  # no labels: the unmasked loss
    model.atoms.grad = None
    masked, _ = reconstruction_backward(model, x, patient, labels=labels)
    assert model._batch_labels is None  # the hook is cleared after every chunk
    extras = model.epoch_extras()
    assert extras["masked_negative_segments"] == 3.0 and extras["mask_violations"] == 0.0
    # The masked loss is the reconstruction under the mask plus the weighted hinge of the label-0 segments.
    model.atoms.grad = None
    total = 0.0
    _, inverse = patient.unique(sorted=True, return_inverse=True)
    weights = 1.0 / torch.bincount(inverse)[inverse] / 3
    for start in range(0, 5, 2):
        model.set_batch_labels(labels[start : start + 2])
        losses, _, _, _ = segment_values(model, x[start : start + 2])
        total += float(((losses + model.batch_penalty()) * weights[start : start + 2]).sum().detach())
    model.set_batch_labels(None)
    assert abs(masked - total) < 1e-6 and masked != plain
    model.label_override = {"a": 1, "b": 0}
    assert model.resolve_labels(torch.tensor([0.0, 1.0, 0.0]), ["a", "b", "a"]).tolist() == [1.0, 0.0, 1.0]
    with pytest.raises(KeyError):
        model.resolve_labels(torch.tensor([0.0]), ["c"])
    model.label_override = None
    assert model.resolve_labels(labels, ["x"] * 5) is labels


# ---------------------------------------------------------------------- the slow hinge of the redesign

WIDTH = GEOMETRY["atom_len"] + N_CHANNELS * GEOMETRY["n_projections"]  # the stored entries of one atom


def _slow(**kwargs: object) -> SlowHingeSAE:
    """The model of ``_model`` with the slow hinge (default: first hinge epoch 11, ramp over 5 epochs)."""
    spec = SlowHingeSpec(**{**GEOMETRY, "n_shared": SHARED, "hinge_weight": 0.0, **kwargs})
    model = SlowHingeSAE(spec=spec, train_spec=WaveformTrainSpec(recenter_atoms=False, segment_batch=2), device="cpu")
    model.atoms = nn.Parameter(model._constrain(_randn(spec.n_atoms, WIDTH, seed=7)))
    return model


def test_slow_spec_checks_and_ramp() -> None:
    for bad in ({"hinge_strength": -1.0}, {"hinge_ramp_epochs": -1}, {"hinge_start_epoch": 0}):
        with pytest.raises(ValueError, match="slow hinge"):
            _slow(**bad)
    with pytest.raises(TypeError):
        SlowHingeSAE(spec=MaskedSeparableSpec(**GEOMETRY, n_shared=SHARED), device="cpu")
    assert len({SlowHingeSAE.model_tag, MaskedSeparableSAE.model_tag, SeparableSAE.model_tag}) == 3
    model = _slow()
    for epoch in range(1, 31):  # the factor r_e of the protocol
        assert model.ramp(epoch) == pytest.approx(min(1.0, max(0.0, (epoch - 10) / 5)))
        assert model.selectable_epoch(epoch) == (epoch >= 15)
    model.begin_epoch(12)
    assert model.ramp() == pytest.approx(0.4)
    at_once = _slow(hinge_start_epoch=3, hinge_ramp_epochs=0)
    assert [at_once.ramp(e) for e in (2, 3, 4)] == [0.0, 1.0, 1.0]


def test_slow_step_on_a_known_gradient() -> None:
    eta, lr = 3.0, 0.01
    model = _slow(hinge_strength=eta, hinge_start_epoch=1, hinge_ramp_epochs=0)
    model.begin_epoch(1)
    before = model.atoms.detach().clone()
    gradient = _randn(6, WIDTH, seed=3)
    model._hinge_grad = gradient.clone()
    model._leak_count = torch.tensor([3.0, 0.0], dtype=torch.float64)  # atom 4 leaks in 3 of 12 windows, atom 5 in none
    model._leak_windows = 12.0
    model.after_optimizer_step(lr)
    size = eta * lr * WIDTH**0.5 * 3 / 12
    torch.testing.assert_close(model.atoms[4], before[4] - size * gradient[4] / gradient[4].norm(), rtol=0, atol=1e-14)
    assert float((model.atoms.detach()[4] - before[4]).norm()) == pytest.approx(size, abs=1e-14)
    assert torch.equal(model.atoms[5], before[5])  # an atom that does not leak does not move
    assert torch.equal(model.atoms[:SHARED], before[:SHARED])  # a shared atom never moves
    assert model._hinge_grad is None and model._leak_count is None and model._leak_windows == 0.0
    extras = model.epoch_extras()
    assert extras["hinge_atoms_moved"] == 1.0 and extras["hinge_step_norm"] == pytest.approx(size)
    assert extras["hinge_ramp"] == 1.0 and model.epoch_extras()["hinge_atoms_moved"] == 0.0

    # An atom with a leak count and no gradient does not move; the ramp scales the step.
    ramped = _slow(hinge_strength=eta)  # first hinge epoch 11, ramp over 5 epochs
    ramped.begin_epoch(12)
    before = ramped.atoms.detach().clone()
    gradient[4] = 0.0
    ramped._hinge_grad, ramped._leak_windows = gradient.clone(), 8.0
    ramped._leak_count = torch.tensor([8.0, 8.0], dtype=torch.float64)  # both leak in every window
    ramped.after_optimizer_step(lr)
    assert torch.equal(ramped.atoms[4], before[4])
    step = float((ramped.atoms.detach()[5] - before[5]).norm())
    assert step == pytest.approx(0.4 * eta * lr * WIDTH**0.5, abs=1e-14)

    # Before the first hinge epoch, and at strength 0, nothing moves.
    for quiet, epoch in ((_slow(hinge_strength=eta), 10), (_slow(hinge_strength=0.0, hinge_start_epoch=1), 5)):
        quiet.begin_epoch(epoch)
        before = quiet.atoms.detach().clone()
        quiet._hinge_grad, quiet._leak_windows = _randn(6, WIDTH, seed=4), 8.0
        quiet._leak_count = torch.tensor([8.0, 8.0], dtype=torch.float64)
        quiet.after_optimizer_step(lr)
        assert torch.equal(quiet.atoms, before)


def test_slow_hinge_counts_the_leaks_of_each_atom() -> None:
    model = _slow(hinge_start_epoch=1)
    values = torch.tensor(
        [
            [[5.0, 1.0], [4.0, 2.0], [3.0, 0.5], [1.0, 0.1], [6.0, 1.5], [2.0, 3.0]],  # label 0
            [[5.0, 1.0], [4.0, 2.0], [3.0, 0.5], [1.0, 0.1], [9.0, 9.0], [9.0, 9.0]],  # label 1: not counted
        ],
        dtype=torch.float64,
    )
    model.set_batch_labels(torch.tensor([0.0, 1.0]))
    model._selection(_responses(values))
    # Window 0: theta = 4, atom 4 leaks (6). Window 1: theta = 1, atoms 4 (1.5) and 5 (3) leak.
    assert model._leak_count.tolist() == [2.0, 1.0] and model._leak_windows == 2.0
    assert model.batch_penalty() == 0.0 and model._hinge_grad is None  # no graph: nothing to collect
    assert float(model._penalty[0]) > 0  # the hinge is still measured, for the training table


def test_slow_hinge_keeps_the_reconstruction_gradient() -> None:
    slow = _slow(hinge_strength=2.0, hinge_start_epoch=1, hinge_ramp_epochs=0)
    slow.atoms = nn.Parameter(slow.atoms.detach().float())
    plain = _model(hinge_weight=0.0)  # the mask alone
    plain.atoms = nn.Parameter(slow.atoms.detach().clone())
    x = _randn(5, N_CHANNELS, 4 * GEOMETRY["pool_len"], seed=9).float()
    patient = torch.tensor([0, 0, 1, 1, 2])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    slow.begin_epoch(1)
    loss, _ = reconstruction_backward(slow, x, patient, labels=labels)
    reference, _ = reconstruction_backward(plain, x, patient, labels=labels)
    assert loss == reference and torch.equal(slow.atoms.grad, plain.atoms.grad)  # the hinge is no term of the loss
    expected = torch.zeros_like(plain.atoms)
    for start in range(0, 5, 2):  # the hinge of each chunk, summed over its label-0 segments
        plain.set_batch_labels(labels[start : start + 2])
        segment_values(plain, x[start : start + 2])
        gradient = torch.autograd.grad(plain._penalty.sum(), plain.atoms, allow_unused=True)[0]
        expected += 0 if gradient is None else gradient
    plain.set_batch_labels(None)
    torch.testing.assert_close(slow._hinge_grad, expected)
    assert float(slow._hinge_grad[:SHARED].abs().sum()) == 0.0 and float(slow._hinge_grad[SHARED:].abs().sum()) > 0.0
    assert slow._leak_windows == 3 * 4 and float(slow._leak_count.max()) <= 12  # 3 label-0 segments of 4 windows
    before = slow.atoms.detach().clone()
    slow.after_optimizer_step(0.01)
    moved = (slow.atoms.detach() - before).norm(dim=1)
    assert float(moved[:SHARED].sum()) == 0.0 and float(moved[SHARED:].sum()) > 0.0

    late = _slow(hinge_strength=2.0)  # the first hinge epoch is 11
    late.atoms = nn.Parameter(late.atoms.detach().float())
    late.begin_epoch(10)
    reconstruction_backward(late, x, patient, labels=labels)
    assert late._hinge_grad is None  # the mask alone: no hinge gradient is collected


class _Planned(TensorDataset):
    """Segments in memory with the metadata plan that the patient loop reads."""

    def __init__(self, patients: dict[str, int], seed: int) -> None:
        subjects = [name for name in patients for _ in range(4)]
        labels = torch.tensor([patients[name] for name in subjects])
        signals = torch.randn(
            len(subjects), N_CHANNELS, 4 * GEOMETRY["pool_len"], generator=torch.Generator().manual_seed(seed)
        )
        super().__init__(signals, labels)
        self.plan = pd.DataFrame({"subject": subjects, "epilepsy": labels})
        self.target_len = signals.shape[-1]
        self.target_channels = ["a", "b", "c"]


def _loop_spec(epochs: int) -> WaveformTrainSpec:
    return WaveformTrainSpec(
        epochs=epochs,
        lr=1e-2,
        sampling_unit="patient",
        patients_per_batch=2,
        segments_per_patient=2,
        segment_batch=2,
        init="random",
        recenter_atoms=False,
    )


def test_slow_hinge_in_the_patient_loop(tmp_path: Path) -> None:
    train = {"ta": 0, "tb": 1, "tc": 0, "td": 1}
    loaders = (DataLoader(_Planned(train, seed=1)), DataLoader(_Planned({"va": 0, "vb": 1}, seed=2)))
    provenance = {"train_subjects": sorted(train)}
    hinge = {"hinge_weight": 0.0, "hinge_strength": 5.0, "hinge_start_epoch": 3, "hinge_ramp_epochs": 2}
    spec = SlowHingeSpec(**GEOMETRY, n_shared=SHARED, **hinge)
    model = SlowHingeSAE(spec=spec, train_spec=_loop_spec(5), random_state=3, device="cpu")
    model.fit_unsupervised(*loaders, provenance=provenance)
    history = model.history
    assert [record["hinge_ramp"] for record in history] == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert all(record["hinge_atoms_moved"] == 0.0 for record in history[:2])  # the mask alone
    assert sum(record["hinge_atoms_moved"] for record in history[2:]) > 0.0  # then the atoms that leak move
    assert all(record["mask_violations"] == 0.0 for record in history)
    assert model.fit_meta["selected_epoch"] >= 4  # only an epoch at full hinge strength is selected
    norms = model.atoms.detach()[:, : GEOMETRY["atom_len"]].norm(dim=1)
    torch.testing.assert_close(norms, norms[:1].expand_as(norms))  # the projection follows every hinge step

    # The warm epochs equal the training with the mask alone: same seed, same draws, no hinge.
    warm = SlowHingeSAE(spec=spec, train_spec=_loop_spec(2), random_state=3, device="cpu")
    warm.fit_unsupervised(*loaders, provenance=provenance)
    alone = MaskedSeparableSAE(
        spec=MaskedSeparableSpec(**GEOMETRY, n_shared=SHARED, hinge_weight=0.0),
        train_spec=_loop_spec(2),
        random_state=3,
        device="cpu",
    )
    alone.fit_unsupervised(*loaders, provenance=provenance)
    for key in ("objective", "val_objective", "leak_rate"):
        assert [r[key] for r in warm.history] == [r[key] for r in alone.history], key
    assert [r["objective"] for r in history[:2]] == [r["objective"] for r in alone.history]
    assert warm.fit_meta["selected_epoch"] == 2  # no epoch at full strength: the last state is kept

    model.save_artifacts(tmp_path)
    restored = SlowHingeSAE(spec=spec, train_spec=_loop_spec(5), pretrained=tmp_path / "sae_state.pt", device="cpu")
    torch.testing.assert_close(restored.atoms, model.atoms)
    with pytest.raises(ValueError, match="not a masked_separable_sae checkpoint"):
        MaskedSeparableSAE(
            spec=MaskedSeparableSpec(**GEOMETRY, n_shared=SHARED), pretrained=tmp_path / "sae_state.pt", device="cpu"
        )
    with pytest.raises(ValueError, match="spec differs"):
        other = SlowHingeSpec(**GEOMETRY, n_shared=SHARED, **{**hinge, "hinge_strength": 50.0})
        SlowHingeSAE(spec=other, train_spec=_loop_spec(5), pretrained=tmp_path / "sae_state.pt", device="cpu")
