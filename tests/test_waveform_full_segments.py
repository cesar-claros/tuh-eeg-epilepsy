"""Full-segment waveform training: coverage, R=1 atom per pooling window, masked gradients and validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import rootutils
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec  # noqa: E402


def _model(train_spec: WaveformTrainSpec | None = None) -> WaveformSAE:
    """Small full-segment model with the default of one atom per pooling window."""
    return WaveformSAE(
        spec=WaveformSpec(n_atoms=3, atom_len=5, pool_len=4),
        train_spec=train_spec,
        device="cpu",
        chunk_windows=2,
    )


def _signals() -> torch.Tensor:
    """Five distinct usable segments, one unusable channel and one completely failed load."""
    signals = torch.randn(6, 2, 32, generator=torch.Generator().manual_seed(17))
    signals[1, 1] = float("nan")
    signals[2, :, 24:] += 100.0
    signals[-1] = 0.0
    return signals


def _one_atom_per_window(rectified: torch.Tensor, pool_len: int = 4) -> torch.Tensor:
    """Reference R=1 code: per atom and window the first maximum, then the first maximal atom per window."""
    windows = rectified.unflatten(-1, (-1, pool_len))
    amplitudes, positions = windows.max(-1)
    winner = torch.zeros_like(amplitudes, dtype=torch.bool).scatter(1, amplitudes.argmax(1, keepdim=True), True)
    code = torch.zeros_like(windows).scatter(-1, positions.unsqueeze(-1), (amplitudes * winner).unsqueeze(-1))
    return code.flatten(-2)


def test_full_mode_defaults_and_no_energy_ratio_gate() -> None:
    """Full scoring is the Python default and keeps high-energy initialization candidates."""
    model = _model()
    assert model.train_spec.crop_sampling == "full"
    assert model.train_spec.segment_batch == 2
    assert model.train_spec.max_crop_energy_ratio is None
    assert model.spec.top_r == 1
    assert model._gate(torch.tensor([0.0, 1.0, 2.0, 1e9])).tolist() == [False, True, True, True]
    with pytest.raises(ValueError, match="does not construct crops"):
        model._crops(_signals(), torch.Generator())


@pytest.mark.parametrize("segment_batch", [0, -1, True, 1.5])
def test_invalid_segment_batch_is_refused(segment_batch: int) -> None:
    """Batch configuration must describe a positive integer number of whole segments."""
    with pytest.raises(ValueError, match="segment_batch"):
        _model(WaveformTrainSpec(segment_batch=segment_batch))


def test_full_mode_refuses_energy_gating_and_partial_pooling_windows() -> None:
    """Neither a hidden energy filter nor silent tail trimming can change full-mode coverage."""
    with pytest.raises(ValueError, match="max_crop_energy_ratio=None"):
        _model(WaveformTrainSpec(max_crop_energy_ratio=10.0))
    model = _model(WaveformTrainSpec(crop_len=3))  # Legacy geometry is irrelevant in full mode.
    with pytest.raises(ValueError, match="no samples may be trimmed"):
        model._full_rows(_signals()[..., :31])
    rows, valid = model._full_rows(_signals())
    assert rows.shape == (6, 2, 32)
    assert not bool(valid[-1].any()) and not bool(valid[1, 1])


def test_full_step_matches_independent_r1_loss_and_gradients() -> None:
    """A direct per-window atom competition and per-window transpose convolution give the same loss/gradient."""
    model = _model()
    rows = torch.randn(3, 2, 24, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    valid = torch.ones(3, 2, dtype=torch.bool)
    valid[0, 1], rows[0, 1] = False, 0.0
    atoms = torch.randn(3, 2, 5, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    atoms.requires_grad_()

    loss, totals, per_atom = model._segment_step(rows, valid, atoms)
    gradient = torch.autograd.grad(loss, atoms)[0]

    reference_atoms = atoms.detach().clone().requires_grad_()
    responses = F.conv1d(F.pad(rows, (2, 2)), reference_atoms)
    code = _one_atom_per_window(responses.relu())
    reconstruction = torch.cat(
        [
            F.conv_transpose1d(code[..., start : start + 4], reference_atoms)[..., 2:6]
            for start in range(0, rows.shape[-1], 4)
        ],
        dim=-1,
    )
    residual = (rows - reconstruction) * valid.unsqueeze(-1)
    expected = 0.5 * residual.square().sum() / (valid.sum() * rows.shape[-1])
    reference_gradient = torch.autograd.grad(expected, reference_atoms)[0]

    assert torch.allclose(loss, expected, atol=1e-12, rtol=1e-10)
    assert torch.allclose(gradient, reference_gradient, atol=1e-12, rtol=1e-10)
    assert torch.allclose(totals[0], residual.square().sum())
    assert float(per_atom[3].sum()) == 3 * 6  # one selected atom in each of the 6 windows of 3 segments
    assert bool((per_atom[2] <= per_atom[3]).all())
    assert torch.equal(per_atom[2], (code > 0).sum((0, 2)).double())


def test_r1_atoms_compete_within_windows_not_across_them() -> None:
    """Events in different windows both survive; a stronger atom suppresses a weaker one in its window."""
    model = WaveformSAE(spec=WaveformSpec(n_atoms=2, atom_len=1, pool_len=4), device="cpu")
    rows = torch.zeros(2, 2, 32)
    rows[0, 0, 0], rows[0, 0, -1] = 3.0, 7.0  # atom 0 at both segment ends: different windows
    rows[1, 0, 0], rows[1, 1, 1] = 9.0, 2.0  # atoms 0 and 1 in the same window
    atoms = torch.eye(2).unsqueeze(-1).requires_grad_()  # atom k reads channel k

    loss, totals, per_atom = model._segment_step(rows, torch.ones(2, 2, dtype=torch.bool), atoms)

    assert torch.allclose(loss, torch.tensor(0.5 * 2.0**2 / 128))
    assert float(totals[4]) == 3.0
    assert per_atom[2].tolist() == [3.0, 0.0]  # won windows
    assert per_atom[3].tolist() == [16.0, 0.0]  # empty windows select atom 0 with a zero (lowest index)


def test_fit_scores_every_usable_segment_once_without_crop_or_gate_calls() -> None:
    """Source shuffling and partial batches preserve coverage, high-energy inputs and loss weighting."""
    signals = _signals()
    loader = DataLoader(TensorDataset(signals, torch.zeros(6)), batch_size=3)
    model = _model(WaveformTrainSpec(epochs=2, lr=0.0, init="random", recenter_atoms=False))
    expected_rows, expected_valid = model._full_rows(signals)
    seen: list[torch.Tensor] = []
    weighted_losses: list[tuple[float, int]] = []
    original = model._segment_step

    def observe(
        rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert 1 <= rows.shape[0] <= model.train_spec.segment_batch
        assert rows.shape[-1] == signals.shape[-1]
        seen.extend(rows.detach().clone().unbind(0))
        result = original(rows, valid, atoms)
        weighted_losses.append((float(result[0].detach()), int(valid.sum()) * rows.shape[-1]))
        return result

    with (
        patch.object(model, "_segment_step", side_effect=observe),
        patch.object(model, "_crops", side_effect=AssertionError("full mode called the crop planner")),
        patch.object(model, "_gate", side_effect=AssertionError("full training called the energy gate")),
    ):
        model.fit_unsupervised(loader, provenance={"train_subjects": ["synthetic"]})

    assert len(seen) == 10
    for row in expected_rows[expected_valid.any(dim=1)]:
        assert sum(torch.equal(row, observed) for observed in seen) == 2
    assert model.fit_meta["scoring_scope"] == "full_encoded_segment"
    offset = 0
    for record in model.history:
        n_steps = int(record["n_steps"])
        calls = weighted_losses[offset : offset + n_steps]
        offset += n_steps
        assert record["segments_seen"] == 6 and record["segments_skipped"] == 1
        assert record["scored_samples"] == int(expected_valid.sum()) * 32
        assert "crops_dropped_frac" not in record
        assert record["objective"] == pytest.approx(
            sum(loss * samples for loss, samples in calls) / sum(samples for _, samples in calls), rel=1e-6
        )


def test_full_validation_is_ungated_and_reports_one_set_of_statistics() -> None:
    """Whole validation does not depend on legacy crop flags or score a second copy of the data."""
    model = _model(WaveformTrainSpec(full_window_val=False))
    model.atoms = nn.Parameter(torch.randn(3, 2, 5, generator=torch.Generator().manual_seed(2)))
    signals = _signals()
    loader = DataLoader(TensorDataset(signals, torch.zeros(6)), batch_size=3)
    rows, valid = model._full_rows(signals)
    usable = valid.any(dim=1)
    _, expected, expected_atoms = model._segment_step(rows[usable], valid[usable], model._dictionary())

    with patch.object(model, "_crops", side_effect=AssertionError("full validation called the crop planner")):
        totals, per_atom, full, full_atom, clipped = model._evaluate(loader)

    assert full is totals and full_atom is per_atom
    assert torch.allclose(totals, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(per_atom, expected_atoms, atol=1e-5, rtol=1e-5)
    assert float(totals[5]) == 5.0
    assert int(clipped[1]) == int(valid.sum()) * 32


def test_full_training_and_validation_refuse_all_unusable_data() -> None:
    """Failed loads cannot silently produce an apparently successful fit or validation."""
    model = _model(WaveformTrainSpec(epochs=1, init="random"))
    loader = DataLoader(TensorDataset(torch.zeros(2, 2, 32), torch.zeros(2)), batch_size=2)
    with pytest.raises(ValueError, match="no usable segments"):
        model.fit_unsupervised(loader, provenance={"train_subjects": ["synthetic"]})
    with pytest.raises(ValueError, match="no usable segments"):
        model._evaluate(loader)


def test_full_fit_matches_independent_adam_update() -> None:
    """Check actual parameter updates against a direct R=1 objective, independently of eventual convergence."""
    spec = WaveformSpec(n_atoms=2, atom_len=5, pool_len=4, top_r=1)
    schedule = WaveformTrainSpec(epochs=1, lr=1e-3, segment_batch=2, init="random", recenter_atoms=False)
    model = WaveformSAE(spec, schedule, random_state=23, device="cpu")
    signals = torch.randn(2, 2, 32, generator=torch.Generator().manual_seed(41))
    loader = DataLoader(TensorDataset(signals, torch.zeros(2)), batch_size=2)

    # Match the declared random initialization and its two unit normalizations;
    # the reference below does not call _segment_step, _select or _decode.
    initial = torch.randn(2, 2, 5, generator=torch.Generator().manual_seed(23))
    initial = initial / torch.linalg.vector_norm(initial, dim=(1, 2), keepdim=True)
    reference = nn.Parameter(initial / torch.linalg.vector_norm(initial, dim=(1, 2), keepdim=True))
    before = reference.detach().clone()
    optimizer = torch.optim.Adam([reference], lr=schedule.lr)
    rows, valid = model._full_rows(signals)
    assert bool(valid.all())
    atoms = reference / torch.linalg.vector_norm(reference, dim=(1, 2), keepdim=True)
    code = _one_atom_per_window(F.conv1d(F.pad(rows, (2, 2)), atoms).relu())
    reconstruction = torch.cat([F.conv_transpose1d(block, atoms)[..., 2:6] for block in code.split(4, dim=-1)], dim=-1)
    loss = 0.5 * (rows - reconstruction).square().mean()
    loss.backward()
    assert reference.grad is not None and float(reference.grad.norm()) > 0
    optimizer.step()
    expected = reference.detach() / torch.linalg.vector_norm(reference.detach(), dim=(1, 2), keepdim=True)
    assert not torch.allclose(before, expected)

    model.fit_unsupervised(loader, provenance={"train_subjects": ["synthetic"]})

    assert model.history[0]["n_steps"] == 1
    assert model.history[0]["objective"] == pytest.approx(float(loss.detach()), rel=1e-6)
    torch.testing.assert_close(model.atoms.detach(), expected, atol=2e-6, rtol=2e-5)
