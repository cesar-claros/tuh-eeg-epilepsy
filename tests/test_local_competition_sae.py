"""Local SAE invariants and a small in-memory fit, to run in the HPC container.

Run from ``code/`` with ``uv run pytest tests/test_local_competition_sae.py``.
These tests require no corpus and make no waveform-recovery claim.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.components.sae_coverage import plan_coverage  # noqa: E402
from src.models.components.local_competition_sae import (  # noqa: E402
    CHECKPOINT_NAME,
    LocalCompetitionSAE,
    LocalCompetitionSpec,
    LocalCompetitionTrainSpec,
    local_competition,
)
from src.utils import check_pretrained_provenance  # noqa: E402

PROVENANCE = {
    "train_subjects": ["s1", "s2"],
    "data": {"signal_mode": "bipolar", "target_sfreq": 256},
}


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    """Return reproducible double-precision data without changing global RNG state."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _brute_competition(z: torch.Tensor, scale: int, rank: int) -> torch.Tensor:
    """Evaluate the mathematical competitor count directly on small finite signals."""
    result = torch.zeros_like(z)
    for batch in range(z.shape[0]):
        for atom in range(z.shape[1]):
            for sample in range(z.shape[2]):
                competitors = sum(
                    int(z[batch, atom, other] >= z[batch, atom, sample])
                    for other in range(max(0, sample - scale + 1), min(z.shape[2], sample + scale))
                    if other != sample
                )
                if competitors < rank:
                    result[batch, atom, sample] = z[batch, atom, sample]
    return result


def _model(
    weight_tying: str = "tied", atom_len: int = 4, competition_scale: int = 3, top_r: int = 1
) -> LocalCompetitionSAE:
    """Initialize a two-channel, two-atom CPU dictionary for numerical checks."""
    model = LocalCompetitionSAE(
        spec=LocalCompetitionSpec(
            n_atoms=2,
            atom_len=atom_len,
            competition_scale=competition_scale,
            top_r=top_r,
            weight_tying=weight_tying,
            pre_emphasis="none",
        ),
        train_spec=LocalCompetitionTrainSpec(center_len=11),
        random_state=17,
        device="cpu",
    )
    model.initialize_dictionary(2)
    return model.double()


@pytest.mark.parametrize(
    ("length", "scale", "rank"),
    [(1, 1, 1), (2, 7, 1), (2, 7, 3), (9, 3, 1), (19, 4, 2), (19, 4, 4), (33, 8, 3)],
)
def test_competition_matches_finite_neighbor_definition(length: int, scale: int, rank: int) -> None:
    """The optimized rule agrees with counts, including ties and short segments."""
    z = torch.randint(0, 7, (2, 3, length), generator=torch.Generator().manual_seed(6)).double()
    result = local_competition(z, scale, rank)
    assert torch.equal(result, _brute_competition(z, scale, rank))
    assert result.shape == z.shape
    for start in range(length - scale + 1):
        assert bool(((result[..., start : start + scale] > 0).sum(-1) <= rank).all())


def test_rolling_maxima_worked_example() -> None:
    """Recover the two strict peaks from the example in the mathematics note."""
    z = torch.tensor([[[0.0, 2.0, 5.0, 1.0, 0.0, 4.0, 1.0, 0.0, 0.0]]])
    expected = torch.tensor([[[0.0, 0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]]])
    assert torch.equal(local_competition(z, 3, 1), expected)


def test_ties_are_suppressed_without_an_absolute_index_tiebreak() -> None:
    """Equal peaks compete conservatively, including at the general-R cutoff."""
    z = torch.tensor([[[5.0, 4.0, 4.0]]])
    assert torch.equal(local_competition(z, 3, 2), torch.tensor([[[5.0, 0.0, 0.0]]]))
    assert torch.equal(local_competition(torch.ones(1, 1, 7), 3, 1), torch.zeros(1, 1, 7))
    assert torch.equal(local_competition(z, 1, 1), z)


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_integer_shifts_preserve_competition_away_from_edges(rank: int) -> None:
    """Selection has no pooling-grid phase, including when responses are tied."""
    scale, shift = 4, 3
    z = torch.randint(0, 9, (2, 3, 41), generator=torch.Generator().manual_seed(12)).double()
    shifted = torch.cat((torch.zeros_like(z[..., :shift]), z[..., :-shift]), dim=-1)
    original_code = local_competition(z, scale, rank)
    shifted_code = local_competition(shifted, scale, rank)
    start, stop = scale - 1, z.shape[-1] - shift - scale + 1
    assert torch.equal(shifted_code[..., start + shift : stop + shift], original_code[..., start:stop])


def test_coefficient_gradients_pass_only_through_surviving_responses() -> None:
    """The comparison mask is discrete, while retained coefficients remain differentiable."""
    z = torch.tensor([[[1.0, 2.0, 5.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0]]], requires_grad=True)
    upstream = torch.arange(1.0, 10.0).reshape_as(z)
    code = local_competition(z, 3, 1)
    (gradient,) = torch.autograd.grad((code * upstream).sum(), z)
    expected = torch.zeros_like(z)
    expected[..., 2], expected[..., 5] = 3.0, 6.0
    assert torch.equal(gradient, expected)


@pytest.mark.parametrize("atom_len", [3, 4])
def test_encoder_and_decoder_follow_the_sample_index_equations(atom_len: int) -> None:
    """Scalar equations check asymmetric padding and the summed multichannel response."""
    model = _model(atom_len=atom_len)
    rows, atoms, code = _randn(1, 2, 11), _randn(2, 2, atom_len, seed=1), _randn(1, 2, 11, seed=2)
    left = (atom_len - 1) // 2
    response = torch.zeros_like(code)
    reconstruction = torch.zeros_like(rows)
    for atom in range(2):
        for channel in range(2):
            for sample in range(11):
                for tap in range(atom_len):
                    input_sample, code_sample = sample + tap - left, sample - tap + left
                    if 0 <= input_sample < 11:
                        response[0, atom, sample] += atoms[atom, channel, tap] * rows[0, channel, input_sample]
                    if 0 <= code_sample < 11:
                        reconstruction[0, channel, sample] += atoms[atom, channel, tap] * code[0, atom, code_sample]
    torch.testing.assert_close(model._responses(rows, atoms), response)
    torch.testing.assert_close(model._decode(code, atoms), reconstruction)
    torch.testing.assert_close((response * code).sum(), (rows * reconstruction).sum())


def test_relu_bias_precedes_competition_and_decoder_can_reconstruct_both_signs() -> None:
    """Biases apply once in their own domains, without rectifying the reconstruction."""
    model = _model(atom_len=3)
    assert model.beta is not None and model.mu is not None
    with torch.no_grad():
        model.beta.copy_(torch.tensor([0.5, -0.5]))
        model.mu.copy_(torch.tensor([0.5, -0.25]))
    responses = torch.tensor([[[-0.6, -0.1, -0.6], [0.4, 0.6, 0.4]]], dtype=torch.float64)
    expected_code = torch.tensor([[[0.0, 0.4, 0.0], [0.0, 0.1, 0.0]]], dtype=torch.float64)
    torch.testing.assert_close(model._select(responses), expected_code)
    code = torch.zeros(1, 2, 9, dtype=torch.float64)
    code[0, 0, 4] = 1.5
    atoms = torch.zeros(2, 2, 3, dtype=torch.float64)
    atoms[0] = torch.tensor([[1.0, -2.0, 1.0], [-1.0, 0.0, 1.0]])
    expected = model.mu.detach().reshape(1, 2, 1).expand(1, 2, 9).clone()
    expected[0, :, 3:6] += 1.5 * atoms[0]
    torch.testing.assert_close(model._decode(code, atoms), expected)
    assert bool((expected < 0).any()) and bool((expected > 0).any())


@pytest.mark.parametrize("weight_tying", ["tied", "untied"])
def test_factor_constraints_and_parameter_gradients(weight_tying: str) -> None:
    """Every fitted factor and bias receives a finite gradient through reconstruction."""
    model = _model(weight_tying, competition_scale=1)
    assert model.enc_shapes is not None and model.enc_alpha is not None
    assert model.beta is not None and model.mu is not None
    with torch.no_grad():
        model.enc_alpha.mul_(2.0)
        if weight_tying == "untied":
            assert model.dec_alpha is not None
            model.dec_alpha.mul_(3.0)
    model.project_parameters()
    assert model.enc_shapes.shape == (2, 2, 4)
    assert model.enc_alpha.shape == (2, 2) and model.beta.shape == (2,) and model.mu.shape == (2,)
    torch.testing.assert_close(model.enc_shapes.norm(dim=-1), torch.ones(2, 2, dtype=torch.float64))
    assert bool((model.enc_alpha >= 0).all())
    torch.testing.assert_close(model._dictionary("dec").norm(dim=(1, 2)), torch.ones(2, dtype=torch.float64))
    if weight_tying == "tied":
        torch.testing.assert_close(model._dictionary("enc"), model._dictionary("dec"))
    else:
        torch.testing.assert_close(model.enc_alpha.norm(dim=1), torch.full((2,), 2.0, dtype=torch.float64))
        assert model.enc_shapes is not model.dec_shapes and model.enc_alpha is not model.dec_alpha
    rows = _randn(2, 2, 37, seed=5)
    code = model._select(model._responses(rows, model._dictionary("enc")))
    reconstruction = model._decode(code, model._dictionary("dec"))
    assert code.shape == (2, 2, 37) and reconstruction.shape == rows.shape
    loss = 0.5 * (reconstruction - rows).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert float(parameter.grad.abs().sum()) > 1e-12, name


def test_tied_shape_and_channel_weight_gradients_sum_both_paths() -> None:
    """Tying keeps both the encoder and decoder contributions to the same factors."""
    model, rows = _model(competition_scale=1), _randn(2, 2, 23, seed=13)

    def gradient(stop_encoder: bool, stop_decoder: bool) -> tuple[torch.Tensor, ...]:
        enc, dec = model._dictionary("enc"), model._dictionary("dec")
        code = model._select(model._responses(rows, enc.detach() if stop_encoder else enc))
        reconstruction = model._decode(code, dec.detach() if stop_decoder else dec)
        return torch.autograd.grad(0.5 * (reconstruction - rows).square().mean(), (model.enc_shapes, model.enc_alpha))

    complete, encoder, decoder = gradient(False, False), gradient(False, True), gradient(True, False)
    for total, encode_part, decode_part in zip(complete, encoder, decoder, strict=True):
        torch.testing.assert_close(total, encode_part + decode_part)
        assert float(encode_part.abs().sum()) > 0 and float(decode_part.abs().sum()) > 0


@pytest.mark.parametrize("weight_tying", ["tied", "untied"])
@pytest.mark.parametrize("atom_len", [3, 4])
def test_contextualized_partial_chunks_match_full_forward_and_gradients(weight_tying: str, atom_len: int) -> None:
    """Real context and center masks preserve the objective, including padded short tails."""
    model, rows = _model(weight_tying, atom_len), _randn(2, 2, 39, seed=7)
    assert model.beta is not None and model.mu is not None
    valid = torch.tensor([[True, True], [True, False]])
    rows[1, 1] = 0
    with torch.no_grad():
        model.beta.fill_(0.2)
        model.mu.fill_(0.13)
    margin = atom_len + model.spec.competition_scale - 2
    chunks = plan_coverage(rows.shape[-1], 11, margin)
    assert chunks[-1].center_length < 11
    items = [(0, chunks[0]), (0, chunks[-1]), (1, chunks[-1])]
    crops, crop_valid, center_mask = model._chunk_batch(rows, valid, items)
    assert crops.shape[-1] == chunks[0].input_length
    assert center_mask.sum(-1).tolist() == [11, chunks[-1].center_length, chunks[-1].center_length]
    full_mask = torch.zeros(2, 39, dtype=torch.bool)
    for parent, chunk in items:
        full_mask[parent, chunk.center_start : chunk.center_stop] = True
    full_code = model._select(model._responses(rows, model._dictionary("enc")))
    full_reconstruction = model._decode(full_code, model._dictionary("dec"))
    crop_code = model._select(model._responses(crops, model._dictionary("enc")))
    crop_reconstruction = model._decode(crop_code, model._dictionary("dec"))
    for index, (parent, chunk) in enumerate(items):
        local_center = slice(margin, margin + chunk.center_length)
        parent_center = slice(chunk.center_start, chunk.center_stop)
        torch.testing.assert_close(crop_code[index, :, local_center], full_code[parent, :, parent_center])
        torch.testing.assert_close(
            crop_reconstruction[index, :, local_center], full_reconstruction[parent, :, parent_center]
        )
    full_loss, _, _ = model._step(rows, valid, full_mask)
    crop_loss, _, _ = model._step(crops, crop_valid, center_mask)
    torch.testing.assert_close(crop_loss, full_loss)
    full_gradients = torch.autograd.grad(full_loss, tuple(model.parameters()))
    crop_gradients = torch.autograd.grad(crop_loss, tuple(model.parameters()))
    for full_gradient, crop_gradient in zip(full_gradients, crop_gradients, strict=True):
        torch.testing.assert_close(crop_gradient, full_gradient, atol=1e-10, rtol=1e-8)


def test_preprocessing_keeps_nondivisible_lengths_and_masks_unusable_channels() -> None:
    """Full-parent preprocessing retains tail samples and isolates nonfinite or flat channels."""
    model, x = _model(), _randn(2, 2, 37).float()
    x[0, 1] = 3.0
    x[1, 0, 8] = float("nan")
    rows, valid = model._rows(x)
    assert rows.shape == x.shape and valid.tolist() == [[True, False], [False, True]]
    assert bool(torch.isfinite(rows).all())
    assert torch.count_nonzero(rows[0, 1]) == 0 and torch.count_nonzero(rows[1, 0]) == 0


def test_centered_shapes_and_degenerate_decoder_weights_remain_normalized() -> None:
    """Projection has a finite fallback when centering or clamping produces zeros."""
    spec = replace(_model().spec, center_shapes=True)
    model = LocalCompetitionSAE(spec=spec, device="cpu")
    model.initialize_dictionary(2)
    assert model.enc_shapes is not None and model.enc_alpha is not None
    with torch.no_grad():
        model.enc_shapes.fill_(3.0)
        model.enc_alpha.fill_(-1.0)
    model.project_parameters()
    torch.testing.assert_close(model.enc_shapes.mean(-1), torch.zeros(2, 2))
    torch.testing.assert_close(model.enc_shapes.norm(dim=-1), torch.ones(2, 2))
    torch.testing.assert_close(model._dictionary("dec").norm(dim=(1, 2)), torch.ones(2))


@pytest.fixture(scope="module", params=["tied", "untied"])
def fitted_model(request: pytest.FixtureRequest) -> tuple[LocalCompetitionSAE, torch.Tensor, dict[str, torch.Tensor]]:
    """Fit one tiny epoch with a partial final center and held-out monitoring data."""
    x, validation = _randn(4, 2, 35, seed=23).float(), _randn(2, 2, 35, seed=24).float()
    model = LocalCompetitionSAE(
        spec=LocalCompetitionSpec(
            n_atoms=2,
            atom_len=5,
            competition_scale=4,
            weight_tying=str(request.param),
            pre_emphasis="none",
        ),
        train_spec=LocalCompetitionTrainSpec(epochs=1, center_len=8, crop_batch=3),
        random_state=11,
        device="cpu",
    )
    model.initialize_dictionary(2)
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=2)
    validation_loader = DataLoader(TensorDataset(validation, torch.ones(2, dtype=torch.long)), batch_size=2)
    model.fit_unsupervised(loader, validation_loader, provenance=PROVENANCE)
    return model, x, initial


def test_training_updates_parameters_and_accounts_for_actual_coverage(
    fitted_model: tuple[LocalCompetitionSAE, torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    """A pass includes every eligible tail sample, with counts independent of chunk padding."""
    model, x, initial = fitted_model
    assert model.fitted and model.train_subjects == ["s1", "s2"] and len(model.history) == 1
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items())
    record = model.history[0]
    assert all(np.isfinite(value) for value in record.values())
    assert record["n_chunks"] == 12 and record["n_steps"] == 4
    assert record["scored_samples"] == 84 and record["scored_channel_samples"] == 168
    assert record["cumulative_scored_samples"] == 84 and record["cumulative_steps"] == 4
    assert record["boundary_excluded_samples"] == 56 and record["parent_windows"] == 4
    assert record["val_scored_samples"] == 42 and record["val_scored_channel_samples"] == 84
    assert record["scored_seconds"] == pytest.approx(84 / 256)
    assert record["val_n_steps"] == 0
    torch.testing.assert_close(model._dictionary("dec").norm(dim=(1, 2)), torch.ones(2))
    code, rows = model.encode(x[:1])
    assert code.shape == (1, 2, 35) and rows.shape == (1, 2, 35)
    assert model.decode(code).shape == rows.shape
    assert model.fit_meta["margin"] == 7 and model.fit_meta["n_channels"] == 2


def test_history_records_pass_time_dictionary_drift_and_atom_usage(
    fitted_model: tuple[LocalCompetitionSAE, torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    """Per-epoch monitors are bounded: wall time, drift of unit-norm atoms, silent and effective atoms."""
    model, _, _ = fitted_model
    record = model.history[0]
    assert record["epoch_seconds"] >= 0 and record["val_seconds"] >= 0
    assert 0 <= record["atom_drift_median"] <= record["atom_drift_max"] <= 2
    for prefix in ("", "val_"):
        assert 0 <= record[f"{prefix}silent_atoms"] <= model.spec.n_atoms
        assert 0 <= record[f"{prefix}effective_atoms"] <= model.spec.n_atoms + 1e-9


def test_checkpoint_roundtrip_preserves_both_branches_biases_history_and_provenance(
    fitted_model: tuple[LocalCompetitionSAE, torch.Tensor, dict[str, torch.Tensor]], tmp_path: Path
) -> None:
    """The public artifact path preserves the complete learned encoder and decoder."""
    model, x, _ = fitted_model
    model.save_artifacts(tmp_path)
    restored = LocalCompetitionSAE(spec=model.spec, device="cpu", pretrained=tmp_path / CHECKPOINT_NAME)
    assert restored.history == model.history and restored.fit_meta == model.fit_meta
    assert restored.provenance == model.provenance and restored.atom_stats == model.atom_stats
    for name, parameter in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], parameter)
    original_code, _ = model.encode(x)
    restored_code, _ = restored.encode(x)
    assert torch.equal(restored_code, original_code)
    torch.testing.assert_close(restored.decode(restored_code), model.decode(original_code))
    assert np.load(tmp_path / "sae_atoms.npy").shape == (2, 2, 5)
    assert np.load(tmp_path / "sae_encoder_atoms.npy").shape == (2, 2, 5)
    assert (tmp_path / "sae_training.csv").exists() and (tmp_path / "sae_atom_stats.csv").exists()
    loader = DataLoader(TensorDataset(x, torch.ones(len(x), dtype=torch.long)), batch_size=2)
    restored.fit_unsupervised(loader, provenance={"train_subjects": ["different"]})
    assert restored.provenance == model.provenance and restored.history == model.history


def test_checkpoint_rejects_wrong_model_spec_and_missing_provenance(
    fitted_model: tuple[LocalCompetitionSAE, torch.Tensor, dict[str, torch.Tensor]], tmp_path: Path
) -> None:
    """A fixed-window dictionary or changed local rule cannot silently be reused."""
    model, _, _ = fitted_model
    model.save_artifacts(tmp_path)
    checkpoint_path = tmp_path / CHECKPOINT_NAME
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    for name, update in (
        ("wrong_model", {"model": "waveform_sae"}),
        ("wrong_format", {"format": -1}),
        ("missing_provenance", {"provenance": {}}),
    ):
        path = tmp_path / f"{name}.pt"
        torch.save({**checkpoint, **update}, path)
        with pytest.raises(ValueError):
            LocalCompetitionSAE(spec=model.spec, device="cpu", pretrained=path)
    for changed in (
        replace(model.spec, competition_scale=5),
        replace(model.spec, weight_tying="untied" if model.spec.weight_tying == "tied" else "tied"),
    ):
        with pytest.raises(ValueError, match="spec"):
            LocalCompetitionSAE(spec=changed, device="cpu", pretrained=checkpoint_path)


def test_shared_provenance_guard_rejects_held_out_patient_overlap_and_signal_changes(
    fitted_model: tuple[LocalCompetitionSAE, torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    """The new component participates in the existing patient and signal-settings guard."""
    model, _, _ = fitted_model
    data = OmegaConf.create(PROVENANCE["data"])
    disjoint = SimpleNamespace(val_df=pl.DataFrame({"subject": ["s8"]}), test_df=pl.DataFrame({"subject": ["s9"]}))
    check_pretrained_provenance(model, disjoint, data, check_val=True)
    for split in ("val_df", "test_df"):
        overlap = SimpleNamespace(val_df=disjoint.val_df, test_df=disjoint.test_df)
        setattr(overlap, split, pl.DataFrame({"subject": ["s1"]}))
        with pytest.raises(ValueError, match="subjects"):
            check_pretrained_provenance(model, overlap, data, check_val=True)
    with pytest.raises(ValueError, match="settings"):
        check_pretrained_provenance(model, disjoint, OmegaConf.create({"signal_mode": "raw", "target_sfreq": 256}))


@pytest.mark.parametrize(
    "spec",
    [
        LocalCompetitionSpec(n_atoms=0),
        LocalCompetitionSpec(competition_scale=0),
        LocalCompetitionSpec(competition_scale=2, top_r=3),
        LocalCompetitionSpec(weight_tying="unknown"),
        LocalCompetitionSpec(pre_emphasis="diff_ar"),
        LocalCompetitionSpec(atom_len=1, center_shapes=True),
    ],
)
def test_invalid_model_specs_are_rejected(spec: LocalCompetitionSpec) -> None:
    """Configuration errors fail before a data loader can run."""
    with pytest.raises(ValueError):
        LocalCompetitionSAE(spec=spec, device="cpu")


def test_fit_requires_provenance_and_eligible_samples_and_keeps_dictionary_only_interface(tmp_path: Path) -> None:
    """Missing provenance or real context fails explicitly, without inventing classifier features."""
    spec = LocalCompetitionSpec(n_atoms=2, atom_len=5, competition_scale=4, pre_emphasis="none")
    schedule = LocalCompetitionTrainSpec(epochs=1, center_len=3, crop_batch=2)
    model = LocalCompetitionSAE(spec=spec, train_spec=schedule, device="cpu")
    x = _randn(2, 2, 14).float()
    loader = DataLoader(TensorDataset(x, torch.zeros(2)), batch_size=2)
    with pytest.raises(ValueError, match="provenance"):
        model.fit_unsupervised(loader)
    with pytest.raises(RuntimeError):
        model.encode(x)
    with pytest.raises(RuntimeError):
        model.save_artifacts(tmp_path)
    with pytest.raises(NotImplementedError):
        model(x)
    with pytest.raises(ValueError, match="eligible"):
        model.fit_unsupervised(loader, provenance=PROVENANCE)
    assert not model.fitted


def test_systematic_training_rejects_a_loader_that_drops_parent_windows() -> None:
    """A dropped final parent batch would invalidate the advertised coverage."""
    model = LocalCompetitionSAE(
        spec=LocalCompetitionSpec(n_atoms=2, atom_len=3, competition_scale=2),
        train_spec=LocalCompetitionTrainSpec(epochs=1, center_len=5),
        device="cpu",
    )
    loader = DataLoader(TensorDataset(_randn(3, 2, 20).float(), torch.zeros(3)), batch_size=2, drop_last=True)
    with pytest.raises(ValueError, match="drop_last"):
        model.fit_unsupervised(loader, provenance=PROVENANCE)
