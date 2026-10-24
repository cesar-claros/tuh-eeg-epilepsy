"""Patient draws, full-pool reductions and pooled-loss gradients. Run in the HPC container."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import rootutils
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.patient_waveform import (  # noqa: E402
    PatientBatches,
    PatientDataset,
    PatientHead,
    SupervisedSpec,
    _calibrate,
    fit_patient_supervised,
    joint_backward,
    load_patient_model,
    patient_loader,
    patient_means,
    reconstruction_backward,
    segment_values,
    summarize_patients,
    validate_supervised,
)
from src.models.components.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec  # noqa: E402


class PlannedSignals(TensorDataset):
    """In-memory analogue of the lazy dataset, with an aligned immutable metadata plan."""

    def __init__(self, prefix: str = "s", seed: int = 13) -> None:
        signals = torch.randn(7, 2, 32, generator=torch.Generator().manual_seed(seed))
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1])
        super().__init__(signals, labels)
        self.plan = pd.DataFrame({"subject": [prefix + "a"] * 5 + [prefix + "b", prefix + "c"], "epilepsy": labels})
        self.target_len = 32
        self.target_channels = ["a", "b"]


def _model(chunk: int = 2) -> WaveformSAE:
    model = WaveformSAE(
        WaveformSpec(n_atoms=2, atom_len=5, pool_len=4),
        WaveformTrainSpec(
            epochs=2,
            lr=0.0,
            sampling_unit="patient",
            patients_per_batch=2,
            segments_per_patient=2,
            segment_batch=chunk,
            init="random",
            recenter_atoms=False,
        ),
        device="cpu",
    )
    model.atoms = nn.Parameter(model._unit(torch.randn(2, 2, 5, generator=torch.Generator().manual_seed(5))))
    return model


def _batch() -> tuple[torch.Tensor, ...]:
    x = PlannedSignals().tensors[0].clone()
    x[0, 1] = float("nan")  # Per-segment usable-channel denominators differ.
    x[1] = 0  # Failed segment of a patient that still has valid segments.
    patient = torch.tensor([8] * 5 + [12, 19])
    return x, torch.tensor([0.0] * 5 + [1.0, 1.0]), patient, torch.arange(7)


def test_patient_draws_keep_complete_pool_and_change_each_epoch() -> None:
    dataset = PatientDataset(PlannedSignals())
    sampler = PatientBatches(dataset.groups, patients=2, segments=2, seed=42)
    first = list(sampler)
    assert len(dataset) == 7 and len(sampler) == 2
    indices = torch.tensor([i for batch in first for i in batch])
    assert torch.bincount(dataset.patient[indices]).tolist() == [2, 1, 1]
    assert len(indices.unique()) == 4
    assert first == list(sampler)
    exposure = set(indices.tolist())
    for epoch in range(1, 12):
        sampler.epoch = epoch
        exposure.update(i for batch in sampler for i in batch)
    assert exposure == set(range(7))
    assert len(dataset.source.plan) == 7


def test_patient_metadata_validation() -> None:
    source = PlannedSignals()
    source.plan.loc[0, "epilepsy"] = 1
    with pytest.raises(ValueError, match="inconsistent labels"):
        PatientDataset(source)
    with pytest.raises(ValueError, match="aligned"):
        PatientDataset(TensorDataset(torch.zeros(1, 2, 32), torch.zeros(1)))
    source = PlannedSignals()
    dataset = PatientDataset(source)
    source.tensors[1][0] = 1
    with pytest.raises(ValueError, match="disagrees"):
        dataset[0]


def test_patient_mean_repeats_do_not_reweight_patients() -> None:
    losses = torch.tensor([1.0, 1.0, 1.0, 9.0], requires_grad=True)
    pooled = patient_means(losses, torch.tensor([0, 0, 0, 1]), 2).mean()
    pooled.backward()
    assert pooled == 5
    assert torch.allclose(losses.grad, torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2]))
    with pytest.raises(ValueError, match="no usable"):
        patient_means(losses, torch.zeros(4, dtype=torch.long), 2)


@pytest.mark.parametrize("chunk", [1, 2, 7])
def test_reconstruction_gradient_matches_patient_reference(chunk: int) -> None:
    model = _model(chunk)
    x, _, patient, _ = _batch()
    _, inverse = patient.unique(sorted=True, return_inverse=True)
    losses, _, _, usable = segment_values(model, x)
    expected = patient_means(losses[usable], inverse[usable], 3).mean()
    gradient = torch.autograd.grad(expected, model.atoms)[0]
    value, skipped = reconstruction_backward(model, x, patient)
    assert value == pytest.approx(float(expected), rel=1e-6)
    assert skipped == 1
    torch.testing.assert_close(model.atoms.grad, gradient, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("hidden", [(), (4,)])
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_two_pass_joint_gradient_matches_direct_pooled_loss(hidden: tuple[int, ...], chunk: int) -> None:
    model = _model(chunk)
    batch = _batch()
    x, labels, patient, _ = batch
    _, inverse = patient.unique(sorted=True, return_inverse=True)
    losses, features, _, usable = segment_values(model, x)
    q = patient_means(features[usable], inverse[usable], 3)
    head = PatientHead(q.detach(), 1e-3, hidden)
    replay_head = deepcopy(head)
    targets = patient_means(labels, inverse, 3)
    reconstruction = patient_means(losses[usable], inverse[usable], 3).mean()
    classification = F.binary_cross_entropy_with_logits(head(q), targets) + 0.01 * head.penalty()
    expected = 0.7 * reconstruction + 0.3 * classification
    expected.backward()
    reference_atoms = model.atoms.grad.clone()
    model.zero_grad(set_to_none=True)
    r, c = joint_backward(model, replay_head, batch, (0.7, 0.3, 0.01))
    assert r == pytest.approx(float(reconstruction), rel=1e-6)
    assert c == pytest.approx(float(classification), rel=1e-6)
    torch.testing.assert_close(model.atoms.grad, reference_atoms, atol=2e-6, rtol=3e-5)
    for direct, replay in zip(head.parameters(), replay_head.parameters()):
        torch.testing.assert_close(direct.grad, replay.grad, atol=2e-6, rtol=3e-5)
    assert not head.mean.requires_grad and not head.scale.requires_grad


def test_classification_gradient_reaches_atoms_and_is_not_segment_bce() -> None:
    model = _model()
    batch = _batch()
    head = PatientHead(torch.tensor([[0.1, 0.7], [0.7, 0.4], [0.9, 0.2]]), 1e-3)
    with torch.no_grad():
        head.network[0].weight.copy_(torch.tensor([[0.6, -0.8]]))
    before = model.atoms.detach().clone()
    optimizer = torch.optim.SGD([model.atoms], lr=1e-2)
    joint_backward(model, head, batch, (0.0, 1.0, 0.0))
    assert model.atoms.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(before, model.atoms)
    logits = torch.tensor([0.0, 4.0])
    pooled_loss = F.binary_cross_entropy_with_logits(logits.mean(), torch.tensor(1.0))
    segment_loss = F.binary_cross_entropy_with_logits(logits, torch.ones(2))
    assert not torch.isclose(pooled_loss, segment_loss)


def test_full_pool_summaries_and_empty_patient_guard() -> None:
    model, source = _model(), PlannedSignals()
    loader = patient_loader(DataLoader(source), model, False)
    summary = summarize_patients(model, loader)
    r, f, zero, _ = segment_values(model, source.tensors[0])
    assert summary.counts.tolist() == [5, 1, 1]
    torch.testing.assert_close(summary.features[0], f[:5].mean(0))
    torch.testing.assert_close(summary.reconstruction[0], r[:5].mean())
    torch.testing.assert_close(summary.zero[0], zero[:5].mean())
    source.tensors[0][-1] = 0
    with pytest.raises(ValueError, match="no usable"):
        summarize_patients(model, loader)


def test_scaled_floor_and_forward_amplitude_features() -> None:
    q = torch.tensor([[0.2, 0.0], [0.3, 0.0], [0.5, 0.0]])
    head1, head2 = PatientHead(q, 1e-6 / 60), PatientHead(q * 2, 1e-6 / 30)
    torch.testing.assert_close((q - head1.mean) / head1.scale, (2 * q - head2.mean) / head2.scale)
    model = _model()
    x = PlannedSignals().tensors[0]
    _, feature, _, _ = segment_values(model, x)
    torch.testing.assert_close(model(x), feature)
    code, _ = model.encode(x)
    assert bool(((code > 0).sum(-1) <= 1).all())


def test_pretraining_patient_epochs_and_checkpoint(tmp_path: Path) -> None:
    source, validation = PlannedSignals(), PlannedSignals("v", seed=21)
    plan = source.plan.copy(deep=True)
    model = _model()
    model.fit_unsupervised(DataLoader(source), DataLoader(validation), {"train_subjects": ["sa", "sb", "sc"]})
    pd.testing.assert_frame_equal(plan, source.plan)
    assert all(record["patients_seen"] == 3 and record["segments_seen"] == 4 for record in model.history)
    assert all(record["n_steps"] == 2 for record in model.history)
    assert model.fit_meta["manifest_segments"] == 7
    model.save_artifacts(tmp_path)
    restored = WaveformSAE(model.spec, model.train_spec, pretrained=tmp_path / "sae_state.pt", device="cpu")
    torch.testing.assert_close(restored.atoms, model.atoms)
    with pytest.raises(ValueError, match="overlap"):
        _model().fit_unsupervised(DataLoader(source), DataLoader(source), {"train_subjects": ["sa", "sb", "sc"]})


def test_supervised_stages_and_reload(tmp_path: Path) -> None:
    source, validation = PlannedSignals(), PlannedSignals("v", seed=21)
    model = _model()
    # A single central delta reconstructs a selected positive sample, so the
    # reconstruction-gain guard holds independently of random pretraining quality.
    model.spec = replace(model.spec, n_atoms=1)
    model.atoms = nn.Parameter(torch.zeros(1, 2, 5))
    with torch.no_grad():
        model.atoms[0, 0, 2] = 1
    model.fitted = True
    model.provenance = {"train_subjects": ["sa", "sb", "sc"]}
    model.fit_meta = {"train_spec": vars(model.train_spec)}
    spec = SupervisedSpec(warmup_epochs=2, joint_epochs=2, calibration_batches=2, ramp_epochs=1)
    original = model.atoms.detach().clone()
    with patch("src.models.components.patient_waveform._calibrate", return_value=(1.0, [{"ratio": 1.0}], [[0]])):
        head, metrics = fit_patient_supervised(model, (DataLoader(source), DataLoader(validation)), spec, tmp_path)
    assert metrics["val_patient_auroc"] >= metrics["frozen_val_patient_auroc"]
    loaded, loaded_head = load_patient_model(tmp_path / "patient_model.pt")
    torch.testing.assert_close(loaded.atoms, model.atoms)
    q = model(source.tensors[0]).detach()
    torch.testing.assert_close(loaded_head(q), head(q))
    assert loaded.fit_meta["supervised"]["r0"] > 0
    if metrics["selected_joint_epoch"] == 0:
        torch.testing.assert_close(model.atoms, original)


def test_supervised_configuration_and_guards() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_patient_waveform")
    assert cfg.require_window_manifests and cfg.feature.spec.top_r == 1
    assert cfg.feature.spec.atom_len == 256 and cfg.feature.train_spec.sampling_unit == "patient"
    spec = instantiate(cfg.supervised, _convert_="all")
    validate_supervised(_model(), spec)
    with pytest.raises(ValueError, match="lambda_center"):
        validate_supervised(_model(), replace(spec, lambda_center=float("nan")))
    with pytest.raises(ValueError, match="R=1"):
        model = _model()
        model.spec = replace(model.spec, top_r=2)
        validate_supervised(model, spec)


def test_gradient_calibration_changes_no_parameters() -> None:
    model = _model()
    base = DataLoader(PlannedSignals())
    full = summarize_patients(model, patient_loader(base, model, False))
    head = PatientHead(full.features, 1e-3)
    with torch.no_grad():
        head.network[0].weight.copy_(torch.tensor([[0.4, -0.7]]))
    atoms, state = model.atoms.detach().clone(), deepcopy(head.state_dict())
    center, records, draws = _calibrate(
        model,
        head,
        patient_loader(base, model, True),
        SupervisedSpec(calibration_batches=3),
        float(full.reconstruction.mean()),
    )
    assert center > 0 and len(records) == len(draws) == 3
    assert all(record["reconstruction_norm"] > 0 and record["classification_norm"] > 0 for record in records)
    torch.testing.assert_close(atoms, model.atoms)
    for key in state:
        torch.testing.assert_close(state[key], head.state_dict()[key])
