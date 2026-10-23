"""Unit invariants of the Waveform SAE (no corpus; CPU; seconds).

Runs under pytest (``uv run pytest tests/test_waveform_sae.py``) or as a script
(``uv run python tests/test_waveform_sae.py``), which executes every ``test_*``
function and exits 1 on the first failure. Covered: the old top-one prototype as the
R=M special case, the worked selection example and both tie rules of
the mathematics note, the support bound, the index conventions of equations (5) and
(14) for an even atom length, the encoder/decoder adjoint identity, the autograd
gradient against equation (23), crop / full-window equality of the code and the
reconstruction and gradients using current full-segment window decisions, the atom constraint, the shared row
pipeline and the unusable-channel mask, the energy gate, grid alignment of the
sampled crops, the masked loss, a small fit on planted multichannel waveforms, the
checkpoint round trip with its refusals and the provenance guard, the spec and
unfitted-module guards, atom re-centering, and the whole-window statistics against
``encode`` and ``decode``.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.models.components.shapeconv_sae import ShapeConvSAE  # noqa: E402
from src.models.components.waveform_sae import (  # noqa: E402
    CHECKPOINT_NAME,
    WaveformSAE,
    WaveformSpec,
    WaveformTrainSpec,
    _CropBatch,
)
from src.utils import check_pretrained_provenance  # noqa: E402
from src.waveform_diagnostics import atom_descriptors, quantile_from_hist  # noqa: E402

PROVENANCE = {"train_subjects": ["s1", "s2"], "data": {"signal_mode": "bipolar", "target_sfreq": 256}}


def _model(crop_len: int | None = None, sampling: str = "random", **spec_kwargs) -> WaveformSAE:
    """A CPU model with a small geometry (K=3, L=8, P=4); ``crop_len`` defaults to two pooling windows."""
    spec = WaveformSpec(**{"n_atoms": 3, "atom_len": 8, "pool_len": 4, **spec_kwargs})
    train_spec = WaveformTrainSpec(crop_len=crop_len or 2 * spec.pool_len, crop_sampling=sampling)
    return WaveformSAE(spec=spec, train_spec=train_spec, device="cpu")


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _expect_value_error(fn, message: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(message)


def _prototype_forward(x: torch.Tensor, w: torch.Tensor, top_r: int) -> tuple[torch.Tensor, ...]:
    """The prototype's forward pass (odd L, symmetric padding, ``topk``) on ``x`` of shape ``(B, M, P, C)``."""
    n_batch, n_pool, pool_len, n_channels = x.shape
    pad = (w.shape[-1] - 1) // 2
    y = F.conv1d(x.reshape(n_batch, n_pool * pool_len, n_channels).permute(0, 2, 1), w, padding=pad)
    z = F.relu(y.reshape(n_batch, -1, n_pool, pool_len))
    values, index = torch.topk(z, k=top_r, dim=-1)
    code = torch.zeros_like(z).scatter(-1, index, values).reshape(n_batch, -1, n_pool * pool_len)
    x_hat = F.conv_transpose1d(code, w, padding=pad).permute(0, 2, 1).reshape(x.shape)
    return y, code, x_hat


def test_all_windows_matches_old_top_one_prototype() -> None:
    model = _model(n_atoms=4, atom_len=7, pool_len=10, top_r=6)
    x, w = _randn(2, 6, 10, 2), _randn(4, 2, 7, seed=1)
    y, code, x_hat = _prototype_forward(x, w, top_r=1)
    rows = x.reshape(2, 60, 2).permute(0, 2, 1)
    assert torch.allclose(model._responses(rows, w), y)
    assert torch.allclose(model._select(y), code)
    assert torch.allclose(model._decode(code, w), x_hat.reshape(2, 60, 2).permute(0, 2, 1))


def test_selection_worked_example() -> None:
    y = torch.tensor([[[-3.0, 4.0, 1.0, 2.0, 1.0, 2.0, -1.0, 0.0, -2.0, 3.0, 7.0, 1.0]]])
    expected = torch.zeros_like(y)
    expected[0, 0, 10] = 7.0
    assert torch.equal(_model(top_r=1)._select(y), expected)
    expected[0, 0, 1] = 4.0
    assert torch.equal(_model(top_r=2)._select(y), expected)
    expected[0, 0, 5] = 2.0
    assert torch.equal(_model(top_r=3)._select(y), expected)


def test_selection_ties_keep_the_lowest_index() -> None:
    y = torch.tensor([[[2.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 0.0, -1.0, -2.0, -3.0, -4.0]]])
    one = [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    two = [0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _model(top_r=1)._select(y).flatten().tolist() == one
    assert _model(top_r=2)._select(y).flatten().tolist() == two
    assert _model(top_r=3)._select(y).flatten().tolist() == two
    assert torch.equal(_model(top_r=2)._select(-torch.ones_like(y)), torch.zeros_like(y))


def test_support_bound_and_kept_values() -> None:
    model = _model(n_atoms=5, pool_len=8, top_r=3)
    y = _randn(2, 5, 64)
    code = model._select(y)
    windows, rectified = code.unflatten(-1, (-1, 8)), F.relu(y)
    assert bool((code >= 0).all())
    assert int((windows > 0).sum(-1).max()) <= 1
    assert int((code > 0).sum(-1).max()) <= 3
    assert torch.equal(code[code > 0], rectified[code > 0])
    best = rectified.unflatten(-1, (-1, 8)).amax(-1).topk(3, dim=-1).values.sum(-1)
    assert torch.allclose(code.sum(-1), best)


def test_selection_is_independent_across_segments_and_atoms() -> None:
    model = _model(top_r=2)
    responses = _randn(3, 4, 24)
    joint = model._select(responses)
    for b in range(3):
        for k in range(4):
            assert torch.equal(joint[b:b + 1, k:k + 1], model._select(responses[b:b + 1, k:k + 1]))


def test_selection_gradients_only_reach_retained_positive_maxima() -> None:
    responses = torch.tensor(
        [[[2.0, 5.0, 5.0, 1.0, 3.0, 1.0, 0.0, -1.0, -1.0, -2.0, 0.0, -3.0]]],
        requires_grad=True,
    )
    code = _model(top_r=1)._select(responses)
    code.sum().backward()
    assert torch.equal(responses.grad, (code > 0).to(responses.dtype))


def test_responses_and_decoder_match_equations_5_and_14() -> None:
    n_atoms, n_channels, atom_len, n_times = 2, 3, 4, 8
    model = _model(n_atoms=n_atoms, atom_len=atom_len, pool_len=2)
    x, w = _randn(1, n_channels, n_times), _randn(n_atoms, n_channels, atom_len, seed=1)
    a = _randn(1, n_atoms, n_times, seed=2)
    left = (atom_len - 1) // 2
    y = torch.zeros(n_atoms, n_times, dtype=torch.float64)
    x_hat = torch.zeros(n_channels, n_times, dtype=torch.float64)
    for k in range(n_atoms):
        for c in range(n_channels):
            for t in range(n_times):
                for ell in range(atom_len):
                    if 0 <= t + ell - left < n_times:
                        y[k, t] += w[k, c, ell] * x[0, c, t + ell - left]
                    if 0 <= t - ell + left < n_times:
                        x_hat[c, t] += w[k, c, ell] * a[0, k, t - ell + left]
    assert model.padding == (1, 2)
    assert torch.allclose(model._responses(x, w)[0], y)
    assert torch.allclose(model._decode(a, w)[0], x_hat)


def test_adjoint_identity_even_and_odd_length() -> None:
    for atom_len in (7, 8):
        model = _model(atom_len=atom_len)
        x, w, a = _randn(2, 3, 32), _randn(3, 3, atom_len, seed=1), _randn(2, 3, 32, seed=2)
        assert torch.allclose((model._responses(x, w) * a).sum(), (x * model._decode(a, w)).sum())


def test_gradient_matches_equation_23() -> None:
    n_atoms, n_channels, atom_len, n_times = 3, 2, 6, 16
    model = _model(n_atoms=n_atoms, atom_len=atom_len, pool_len=4, top_r=2)
    x = _randn(1, n_channels, n_times)
    w = _randn(n_atoms, n_channels, atom_len, seed=1).requires_grad_()
    code = model._select(model._responses(x, w))
    x_hat = model._decode(code, w)
    (grad,) = torch.autograd.grad(0.5 * (x - x_hat).pow(2).mean(), w)

    def correlate(maps: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """``D[k, c, l] = sum_j maps[k, j] * signal[c, j + l - p_L]``, zeros outside the segment."""
        patches = F.pad(signal, model.padding).unfold(-1, atom_len, 1)
        return torch.einsum("kj,cjl->kcl", maps, patches)

    code, error = code.detach(), (x_hat - x).detach()
    back = model._responses(error, w.detach())  # G of equation (22)
    active = (code > 0).to(x.dtype)  # Gamma of equation (20)
    expected = (correlate(code[0], error[0]) + correlate((active * back)[0], x[0])) / (n_times * n_channels)
    assert bool(active.any())
    assert torch.allclose(grad, expected, atol=1e-10)


def test_crop_code_matches_full_window_code() -> None:
    for atom_len, pool_len, top_r in ((8, 4, 1), (7, 16, 2), (5, 8, 8)):
        model = _model(atom_len=atom_len, pool_len=pool_len, top_r=top_r)
        crop_len, margin = model.train_spec.crop_len, model.margin
        total = crop_len + 2 * margin
        assert margin % pool_len == 0 and margin >= atom_len + pool_len - 2
        rows, atoms = _randn(2, 3, total + 4 * pool_len), _randn(3, 3, atom_len, seed=1)
        full_code = model._select(model._responses(rows, atoms))
        full_recon = model._decode(full_code, atoms)
        model.chunk_windows = 1
        for start in range(0, rows.shape[-1] - total + 1, pool_len):
            mask = model._crop_window_mask(rows, torch.arange(2), torch.full((2,), start), atoms)
            code = model._select(model._responses(rows[..., start : start + total], atoms), mask)
            center = slice(start + margin, start + margin + crop_len)
            assert torch.allclose(model._center(code), full_code[..., center], atol=1e-12)
            assert torch.allclose(model._center(model._decode(code, atoms)), full_recon[..., center], atol=1e-12)


def test_distant_competitor_suppresses_a_crop_candidate() -> None:
    model = _model(n_atoms=1, atom_len=1, pool_len=4, crop_len=4)
    rows = torch.zeros(1, 1, 32, dtype=torch.float64)
    rows[0, 0, 5], rows[0, 0, 25] = 2.0, 10.0
    atoms = torch.ones(1, 1, 1, dtype=torch.float64)
    window = start = torch.tensor([0])
    inputs = model._crop_inputs(rows, window, start)
    local = model._select(model._responses(inputs, atoms))
    mask = model._crop_window_mask(rows, window, start, atoms)
    consistent = model._select(model._responses(inputs, atoms), mask)
    assert float(model._center(local).sum()) == 2.0
    assert float(model._center(consistent).sum()) == 0.0
    assert not bool(mask.any())


def test_crop_masks_recomputed_after_atom_changes() -> None:
    model = _model(n_atoms=1, atom_len=1, pool_len=4, crop_len=4)
    rows = torch.zeros(1, 2, 32, dtype=torch.float64)
    rows[0, 0, 5], rows[0, 1, 25] = 2.0, 10.0
    window = start = torch.tensor([0])
    atoms = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)
    before = model._crop_window_mask(rows, window, start, atoms)
    atoms[0, 0, 0], atoms[0, 1, 0] = 0.0, 1.0
    after = model._crop_window_mask(rows, window, start, atoms)
    assert bool(before[0, 0, 1]) and not bool(after.any())


def test_crop_loss_and_gradients_match_full_segment_scoring() -> None:
    for sampling in ("random", "tiled"):
        model = _model(top_r=3, sampling=sampling)
        model.chunk_windows = 1
        x = _randn(3, 3, 100)
        x[1, 2] = 0.0
        batch = model._crops(x, torch.Generator().manual_seed(2))
        index = torch.arange(batch.window.numel() - 1, -1, -1)[:17]
        atoms = _randn(3, 3, 8, seed=1).requires_grad_()
        loss, _, _ = model._step(batch, index, atoms)
        (gradient,) = torch.autograd.grad(loss, atoms)
        full_code = model._select(model._responses(batch.rows, atoms))
        full_recon = model._decode(full_code, atoms) * batch.valid.unsqueeze(-1)
        window, start = batch.window[index], batch.start[index]
        target = model._center(model._crop_inputs(batch.rows, window, start))
        reconstruction = model._center(model._crop_inputs(full_recon, window, start))
        denominator = batch.valid[window].sum() * model.train_spec.crop_len
        expected_loss = 0.5 * (target - reconstruction).pow(2).sum() / denominator
        (expected_gradient,) = torch.autograd.grad(expected_loss, atoms)
        assert torch.allclose(loss, expected_loss, atol=1e-10)
        assert torch.allclose(gradient, expected_gradient, atol=1e-10)


def test_atom_constraint() -> None:
    atoms = _randn(4, 3, 10, dtype=torch.float32)
    ones = torch.ones(4)
    assert torch.allclose(torch.linalg.vector_norm(_model()._constrain(atoms), dim=(1, 2)), ones, atol=1e-6)
    centered = _model(center_atoms=True)._constrain(atoms)
    assert float(centered.mean(-1).abs().max()) < 1e-6
    assert torch.allclose(torch.linalg.vector_norm(centered, dim=(1, 2)), ones, atol=1e-6)
    assert torch.equal(_model(atom_norm="free")._constrain(atoms), atoms)
    assert float(_model()._constrain(torch.zeros(1, 3, 10)).abs().sum()) == 0.0


def test_rows_share_the_shapeconv_pipeline_and_zero_unusable_channels() -> None:
    x = _randn(2, 3, 18, dtype=torch.float32)
    x[0, 1] = 5.0
    x[1, 2, 3] = float("nan")
    rows, valid = _model(pre_emphasis="none")._rows(x)
    assert rows.shape == (2, 3, 16)  # 18 samples trimmed to a multiple of pool_len = 4
    assert valid.tolist() == [[True, False, True], [True, True, False]]
    assert float(rows[0, 1].abs().sum()) == 0.0 and float(rows[1, 2].abs().sum()) == 0.0
    assert torch.allclose(rows[0, 0], ShapeConvSAE._robust_scale(x[0, :1])[0, :16], atol=1e-6)
    diffed, _ = _model()._rows(x)
    assert diffed.shape == (2, 3, 16) and bool(torch.isfinite(diffed).all())
    expected = ShapeConvSAE._robust_scale(F.pad(x[0, :1].diff(dim=-1), (1, 0)))[0, :16]
    assert torch.allclose(diffed[0, 0], expected, atol=1e-6)
    _expect_value_error(lambda: _model()._rows(x[0]), "a 2-D batch was not refused")


def test_energy_gate_drops_flat_and_artifact_crops() -> None:
    energy = torch.tensor([1.0, 1.2, 0.8, 0.0, 50.0])
    assert _model()._gate(energy).tolist() == [True, True, True, False, False]
    ungated = WaveformSAE(train_spec=WaveformTrainSpec(max_crop_energy_ratio=None), device="cpu")
    assert ungated._gate(energy).tolist() == [True, True, True, False, True]


def test_crops_are_grid_aligned() -> None:
    model = _model()
    x = _randn(4, 3, 64, dtype=torch.float32)
    total = model.train_spec.crop_len + 2 * model.margin
    batch = model._crops(x, torch.Generator().manual_seed(0))
    crops = model._crop_inputs(batch.rows, batch.window, batch.start)
    assert crops.shape[0] > 0 and crops.shape[1:] == (3, total)
    assert batch.valid[batch.window].shape == (crops.shape[0], 3)
    assert bool((batch.start % model.spec.pool_len == 0).all())
    rows, _ = model._rows(x)
    assert torch.equal(rows, batch.rows)
    assert float(batch.clipped[1]) == 4 * 3 * 64
    assert float(batch.clipped[0]) == float((rows.abs() >= 20.0).sum())
    candidates = rows.unfold(-1, total, model.spec.pool_len).permute(0, 2, 1, 3).reshape(-1, 3, total)
    for crop in crops:
        assert bool((candidates == crop).flatten(1).all(1).any())


def test_tiled_crops_cover_the_window_once() -> None:
    model = _model(sampling="tiled")  # L=8, P=4, crop_len=8: margin 12, crop 32 samples
    x = _randn(3, 3, 100, dtype=torch.float32)
    rows, _ = model._rows(x)
    n_tiles = (100 - 32) // 8 + 1  # 9 tiles per window, slack of 4 samples: two grid phases, 0 and 4
    phases = set()
    for seed in range(12):
        batch = model._crops(x, torch.Generator().manual_seed(seed))
        crops = model._crop_inputs(batch.rows, batch.window, batch.start)
        assert crops.shape == (3 * n_tiles, 3, 32)
        assert batch.valid[batch.window].shape == (3 * n_tiles, 3)
        centers = model._center(crops.view(3, n_tiles, 3, 32)).permute(0, 2, 1, 3).reshape(3, 3, n_tiles * 8)
        matched = [o for o in (0, 4) if torch.equal(centers, rows[..., o + 12 : o + 12 + n_tiles * 8])]
        assert len(matched) == 1  # the scored centers tile the row without gap or overlap, from one grid offset
        phases.add(matched[0])
    assert phases == {0, 4}  # both phases occur across seeds
    _expect_value_error(lambda: _model(sampling="stride"), "an unknown sampling was not refused")


def test_step_loss_masks_unusable_channels() -> None:
    model = _model()
    total = model.train_spec.crop_len + 2 * model.margin
    crops, atoms = _randn(5, 3, total), _randn(3, 3, 8, seed=1)
    valid = torch.ones(5, 3, dtype=torch.bool)
    valid[0, 1] = False
    crops[0, 1] = 0.0
    batch = _CropBatch(crops, valid, torch.arange(5), torch.zeros(5, dtype=torch.long),
                       torch.empty(0), 0, torch.zeros(2))
    loss, stats, per_atom = model._step(batch, torch.arange(5), atoms)
    code = model._select(model._responses(crops, atoms))
    residual = model._center(crops - model._decode(code, atoms))
    residual[0, 1] = 0.0
    n_scored = (5 * 3 - 1) * model.train_spec.crop_len
    assert torch.allclose(loss, 0.5 * residual.pow(2).sum() / n_scored)
    assert torch.allclose(stats[0], residual.pow(2).sum()) and float(stats[5]) == 5.0
    assert float(stats[6]) == 5 * (model.train_spec.crop_len // model.spec.pool_len)
    assert torch.allclose(per_atom[0], model._center(code).pow(2).sum(dim=(0, 2)))


def test_empty_crop_plan_keeps_source_context_and_validation_works() -> None:
    model = _model(n_atoms=1, atom_len=1, pool_len=4, crop_len=4)
    model.atoms = nn.Parameter(torch.ones(1, 1, 1))
    x = torch.zeros(2, 1, 32)
    batch = model._crops(x, torch.Generator().manual_seed(0))
    assert batch.window.numel() == batch.start.numel() == 0
    assert batch.rows.shape == (2, 1, 32) and not bool(batch.valid.any())
    loader = DataLoader(TensorDataset(x, torch.zeros(2)), batch_size=2)
    totals, _, full, full_atom, _ = model._evaluate(loader)
    assert float(totals.sum()) == 0.0
    assert full is not None and full_atom is not None
    assert float(full[4]) == 0.0 and float(full_atom[3].sum()) == 2.0


def test_full_window_statistics_match_encode_decode() -> None:
    model = _model(n_atoms=3, atom_len=8, pool_len=4)
    rows, atoms = _randn(5, 3, 64), _randn(3, 3, 8, seed=1)
    valid = torch.ones(5, 3, dtype=torch.bool)
    valid[2, 0] = False
    rows[2, 0] = 0.0
    model.atoms = nn.Parameter(atoms.float())
    model.chunk_windows = 2  # three chunks, the last one partial
    totals, per_atom = model._window_stats(rows, valid, model._dictionary().double())
    code = model._select(model._responses(rows, model._dictionary().double()))
    reconstruction = model._decode(code, model._dictionary().double())
    reconstruction[2, 0] = 0.0
    assert torch.allclose(totals[0], (rows - reconstruction).pow(2).sum())
    assert torch.allclose(totals[1], rows.pow(2).sum()) and float(totals[5]) == 5.0
    assert float(totals[6]) == 5 * (64 // 4) and float(totals[4]) == float((code > 0).sum())
    assert torch.allclose(per_atom[2], (code > 0).sum(dim=(0, 2)).double())
    table = model._atom_table(totals, per_atom)
    assert torch.equal(per_atom[3], torch.full((3,), 5.0, dtype=torch.float64))
    assert torch.allclose(table["positive_frac"], (code > 0).sum(dim=(0, 2)).double() / 5)
    assert torch.allclose(table["active_window_frac"], (code > 0).sum(dim=(0, 2)).double() / (5 * 16))


def test_usage_counts_only_selected_slots() -> None:
    model = _model(n_atoms=1, top_r=2)
    model.atoms = nn.Parameter(torch.ones(1, 1, 8))
    responses = torch.tensor([[[1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0,
                                -1.0, -2.0, -3.0, -4.0]]])
    selected = model._select_windows(F.relu(responses).unflatten(-1, (-1, 4)).amax(-1))
    code = model._select(responses, selected)
    totals, per_atom = model._statistics(torch.zeros_like(code), torch.zeros_like(code), code, selected)
    table = model._atom_table(totals, per_atom)
    assert float(per_atom[2, 0]) == 1.0 and float(per_atom[3, 0]) == 2.0
    assert float(table["positive_frac"][0]) == 0.5
    assert torch.allclose(table["active_window_frac"], torch.tensor([1 / 3], dtype=torch.float64))


def test_recentering_moves_edge_energy_to_the_center() -> None:
    model = _model(n_atoms=2, atom_len=8)
    atoms = torch.zeros(2, 3, 8)
    atoms[0, :, :3] = 1.0  # centroid 1.0, center 3.5: shift by 2 (the whole-sample part of 2.5)
    atoms[1, :, 3:5] = 1.0  # centroid 3.5: stays
    model.atoms = nn.Parameter(WaveformSAE._unit(atoms))
    optimizer = torch.optim.Adam([model.atoms], lr=1e-3)
    model.atoms.sum().backward()
    optimizer.step()
    before = model.atoms.detach().clone()
    assert model._recenter(optimizer) == 1
    after = model.atoms.detach()
    assert torch.allclose(after[1], WaveformSAE._unit(before[1:2])[0])
    assert float(after[0, :, :2].abs().sum()) == 0.0
    assert torch.allclose(after[0, :, 2:], WaveformSAE._unit(before[0:1])[0, :, :6], atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(after, dim=(1, 2)), torch.ones(2), atol=1e-6)
    moments = optimizer.state[model.atoms]["exp_avg"]
    assert float(moments[0, :, :2].abs().sum()) == 0.0 and bool((moments[0, :, 2:] != 0).all())
    assert model._recenter(optimizer) == 0


def _planted(seed: int, n_windows: int = 24) -> torch.Tensor:
    """Windows ``(n, 3, 512)``: two fixed multichannel templates alternate, one event per 32 samples, plus noise."""
    generator = torch.Generator().manual_seed(seed)
    # The template seed must differ from the model's ``random_state``: ``init="random"`` draws a
    # tensor of this same shape from its own seed, and an equal seed starts the fit at the answer.
    templates = WaveformSAE._unit(torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(123)))
    x = 0.05 * torch.randn(n_windows, 3, 512, generator=generator)
    for window in range(n_windows):
        for block in range(16):
            start = 32 * block + int(torch.randint(0, 16, (1,), generator=generator))
            amplitude = 2.0 + float(torch.rand(1, generator=generator))
            x[window, :, start : start + 16] += amplitude * templates[block % 2]
    return x


def _fitted(epochs: int) -> tuple[WaveformSAE, torch.Tensor]:
    spec = WaveformSpec(n_atoms=2, atom_len=16, pool_len=32, top_r=8, pre_emphasis="none")
    train_spec = WaveformTrainSpec(
        epochs=epochs, lr=1e-2, crop_len=64, crops_per_window=8, crop_batch=32, init="random"
    )
    model = WaveformSAE(spec=spec, train_spec=train_spec, random_state=0, device="cpu")
    x, x_val = _planted(1), _planted(2, n_windows=8)
    loader = DataLoader(TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), batch_size=8)
    val_loader = DataLoader(TensorDataset(x_val, torch.zeros(len(x_val), dtype=torch.long)), batch_size=8)
    model.fit_unsupervised(loader, val_loader, provenance=PROVENANCE)
    return model, x


def test_fit_reduces_the_residual_and_keeps_the_constraint() -> None:
    model, _ = _fitted(epochs=30)
    history = model.history
    assert model.fitted and len(history) == 30
    assert all(np.isfinite(value) for record in history for value in record.values())
    assert history[-1]["residual_frac"] < 0.5 * history[0]["residual_frac"]
    norms = torch.linalg.vector_norm(model.atoms.detach(), dim=(1, 2))
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
    assert 0.0 <= history[-1]["selected_zero_frac"] <= 1.0
    assert 0.0 < history[-1]["n_effective_atoms"] <= 2.0 + 1e-6
    assert model.fit_meta["n_channels"] == 3 and model.fit_meta["margin"] == 64
    assert model.fit_meta["selection_scope"] == "full_encoded_segment"
    assert len(model.atom_stats["atom"]) == 2 and model.fit_meta["atom_stats_split"] == "val_full"
    assert history[-1]["val_full_residual_frac"] < 0.5 * history[0]["val_full_residual_frac"]
    assert 0.0 <= history[-1]["clipped_frac"] <= 1.0 and 0.0 <= history[-1]["val_clipped_frac"] <= 1.0


def test_checkpoint_round_trip_and_refusals() -> None:
    class _Datamodule:
        val_df = pd.DataFrame({"subject": ["s9"]})
        test_df = pd.DataFrame({"subject": ["s2", "s5"]})

    model, x = _fitted(epochs=2)
    code, rows = model.encode(x[:2])
    assert code.shape == (2, 2, 512) and rows.shape == (2, 3, 512)
    _expect_value_error(lambda: model.encode(x[:1, :2]), "a channel-count mismatch was not refused")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model.save_artifacts(out)
        path = out / CHECKPOINT_NAME
        reloaded = WaveformSAE(spec=model.spec, device="cpu", pretrained=path)
        assert reloaded.fitted and reloaded.train_subjects == ["s1", "s2"]
        assert reloaded.history == model.history and reloaded.fit_meta == model.fit_meta
        assert reloaded.atom_stats == model.atom_stats
        assert torch.equal(reloaded.encode(x[:2])[0], code)
        assert np.load(out / "sae_atoms.npy").shape == (2, 3, 16)
        assert np.load(out / "sae_atoms_signal.npy").shape == (2, 3, 16)
        assert (out / "sae_training.csv").exists() and (out / "sae_atom_stats.csv").exists()
        loader = DataLoader(TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), batch_size=8)
        reloaded.fit_unsupervised(loader, provenance={"train_subjects": ["other"]})
        assert reloaded.train_subjects == ["s1", "s2"]
        data_cfg = OmegaConf.create({"signal_mode": "bipolar", "target_sfreq": 256})
        _expect_value_error(
            lambda: check_pretrained_provenance(reloaded, _Datamodule(), data_cfg),
            "test-subject overlap was not refused",
        )
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["format"] == 2
        torch.save({**checkpoint, "format": 1}, out / "legacy.pt")
        torch.save({**checkpoint, "provenance": {}}, out / "bare.pt")
        torch.save({k: v for k, v in checkpoint.items() if k != "model"}, out / "untagged.pt")
        for spec, bad in (
            (model.spec, out / "bare.pt"),
            (model.spec, out / "untagged.pt"),
            (model.spec, out / "legacy.pt"),
            (replace(model.spec, pool_len=16), path),
        ):
            _expect_value_error(
                lambda spec=spec, bad=bad: WaveformSAE(spec=spec, device="cpu", pretrained=bad),
                f"{bad.name} with {spec} was not refused",
            )


def test_spec_and_unfitted_guards() -> None:
    for kwargs in ({"top_r": 0}, {"top_r": 1.5}, {"top_r": True},
                   {"pre_emphasis": "diff_ar"}, {"atom_norm": "max"}):
        _expect_value_error(lambda kwargs=kwargs: _model(**kwargs), f"{kwargs} was not refused")
    _expect_value_error(lambda: _model(crop_len=6), "a crop_len off the pooling grid was not refused")
    assert _model(top_r=5)._select(_randn(1, 3, 24)).shape == (1, 3, 24)
    _expect_value_error(lambda: _model(top_r=5)._select(_randn(1, 3, 16)), "R > M was not refused")
    _expect_value_error(lambda: _model(top_r=5)._rows(_randn(1, 3, 19)), "R > trimmed M was not refused")
    _expect_value_error(lambda: _model()._select(_randn(1, 3, 15)), "a partial pooling window was not refused")
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        for call in (lambda: model.encode(torch.zeros(1, 3, 16)), lambda: model.save_artifacts(Path(tmp))):
            try:
                call()
            except RuntimeError:
                continue
            raise AssertionError("a module without a dictionary was used")
    try:
        model(torch.zeros(1, 3, 16))
    except NotImplementedError:
        return
    raise AssertionError("forward returned features that are not defined")


def test_diagnostics_descriptors_and_histogram_quantiles() -> None:
    sfreq, length = 256.0, 64
    atoms = np.zeros((2, 4, length), dtype=np.float32)
    atoms[0, 2, 10:20] = 1.0  # one channel, one block: rank one, dominant channel 2 with all the energy
    atoms[1] = np.sin(2 * np.pi * 10.0 * np.arange(length) / sfreq)[None, :]  # 10 Hz on every channel equally
    names = ["a", "b", "c", "d"]
    d = atom_descriptors(atoms, names, sfreq)
    assert d["dominant_channel"] == ["c", "a"] and abs(d["dominant_energy_frac"][0] - 1.0) < 1e-6
    assert d["n_channels_90"] == [1, 4] and abs(d["rank_one_frac"][0] - 1.0) < 1e-6 and d["rank_one_frac"][1] > 0.99
    assert abs(d["peak_hz_encoded"][1] - 10.0) < 1.0 and abs(d["norm"][0] - np.sqrt(10.0)) < 1e-5
    hist = np.zeros(100)
    hist[10] = 3.0  # 3 values in the bin [10, 11) of a histogram over [0, 100]
    hist[50] = 1.0
    assert abs(quantile_from_hist(hist, 0.5, 100.0) - 10.5) < 1e-9
    assert abs(quantile_from_hist(hist, 0.99, 100.0) - 50.5) < 1e-9
    assert np.isnan(quantile_from_hist(np.zeros(5), 0.5, 1.0))


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
