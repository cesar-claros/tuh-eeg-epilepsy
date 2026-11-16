"""Checks of the class-masked separable dictionary, ``src/models/components/masked_separable_sae.py`` (no corpus; CPU).

Runs under pytest. Covered: the checks of the specification; the model without labels against ``SeparableSAE`` with
the same factors; the masked selection against a brute-force selection, with ties; the hinge on known scores, its
zero exactly when no epilepsy-only atom would be selected without the mask, and its gradient; no gradient of the
reconstruction to a masked atom; the hooks of the patient loop (labels per chunk, the weighted penalty, the permuted
labels, the epoch records).
"""

from __future__ import annotations

import numpy as np
import pytest
import rootutils
import torch
from torch import nn

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.masked_separable_sae import MaskedSeparableSAE, MaskedSeparableSpec  # noqa: E402
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
