"""Strict CUDA preprocessing and patient updates; run on the HPC, without corpus data."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
import rootutils
import torch
from torch.utils.data import DataLoader, TensorDataset

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.models.components.patient_waveform import (  # noqa: E402
    PatientHead,
    joint_backward,
    patient_loader,
    reconstruction_backward,
    summarize_patients,
)
from src.models.components.shapeconv_sae import ShapeConvSAE  # noqa: E402
from src.models.components.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the HPC CUDA allocation")


@pytest.fixture
def strict_determinism(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Use the campaign's strict settings, restoring process settings after each test."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic


@pytest.mark.usefixtures("strict_determinism")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("length", [7, 8, 30720])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lower_median_and_mad_match_original_values(device: str, length: int, dtype: torch.dtype) -> None:
    """Preserve even/odd medians, ties, clipping, scale floors and non-finite handling."""
    rows = torch.randn(8, length, generator=torch.Generator().manual_seed(17), dtype=dtype)
    rows[1] = torch.arange(length, dtype=dtype)  # Distinguishes lower median from interpolation.
    rows[2] = 3.0
    rows[3] = 0.0
    rows[3, -1] = 1e6  # Zero MAD activates the std floor and clipping.
    rows[4] = rows[4].round()  # Repeated values.
    rows[5, 0], rows[6, 0], rows[7, 0] = float("nan"), float("inf"), -float("inf")
    median = rows.median(dim=-1, keepdim=True).values
    centered = rows - median
    mad = centered.abs().median(dim=-1, keepdim=True).values * 1.4826
    scale = torch.maximum(mad, 0.01 * centered.std(dim=-1, keepdim=True))
    expected = (centered / scale.clamp_min(1e-8)).clamp(-20, 20)

    values = rows.to(device)
    torch.testing.assert_close(ShapeConvSAE._row_median(values).cpu(), median, rtol=0, atol=0, equal_nan=True)
    actual = ShapeConvSAE._robust_scale(values)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6, equal_nan=True)
    torch.testing.assert_close(ShapeConvSAE._robust_scale(values), actual, rtol=0, atol=0, equal_nan=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


class PlannedSignals(TensorDataset):
    """Small patient pool with the sweep geometry and a failed channel/segment."""

    def __init__(self) -> None:
        signals = torch.randn(4, 2, 1024, generator=torch.Generator().manual_seed(13))
        signals[0, 1], signals[3] = float("nan"), 0.0
        labels = torch.tensor([0, 0, 1, 1])
        super().__init__(signals, labels)
        self.plan = pd.DataFrame({"subject": ["a", "a", "b", "b"], "epilepsy": labels})
        self.target_len = 1024
        self.target_channels = ["a", "b"]


@CUDA
@pytest.mark.usefixtures("strict_determinism")
def test_cuda_patch_initialization_and_patient_updates_repeat_exactly() -> None:
    """Exercise patch initialization, reconstruction, full summaries and tied joint backward."""
    loader = DataLoader(PlannedSignals(), batch_size=2)
    results = []
    for _ in range(2):
        torch.manual_seed(7)
        model = WaveformSAE(
            WaveformSpec(n_atoms=4, atom_len=256, pool_len=512, top_r=1, pre_emphasis="none"),
            WaveformTrainSpec(
                sampling_unit="patient",
                patients_per_batch=2,
                segments_per_patient=2,
                segment_batch=2,
                init="patches",
                recenter_atoms=False,
            ),
            device="cuda",
            random_state=7,
        )
        model._init_atoms(loader, torch.Generator().manual_seed(7))
        batch = next(iter(patient_loader(loader, model, training=True)))
        assert model.atoms is not None
        optimizer = torch.optim.Adam([model.atoms], lr=1e-3)
        reconstruction_backward(model, batch[0], batch[2])
        assert model.atoms.grad is not None and bool(torch.isfinite(model.atoms.grad).all())
        assert float(model.atoms.grad.norm()) > 0
        initial_gradient = model.atoms.grad.detach().clone()
        optimizer.step()
        with torch.no_grad():
            model.atoms.copy_(model._constrain(model.atoms))

        summary = summarize_patients(model, patient_loader(loader, model, training=False))
        assert summary.counts.tolist() == [2, 1] and summary.skipped == 1
        head = PatientHead(summary.features, scale_floor=1e-6 / 2)
        optimizer = torch.optim.Adam([model.atoms, *head.parameters()], lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        joint_backward(model, head, batch, (1.0 / float(summary.reconstruction.mean()), 0.1, 1e-3))
        parameters = [model.atoms, *head.parameters()]
        gradients = []
        for parameter in parameters:
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            gradients.append(parameter.grad.detach().clone())
        optimizer.step()
        results.append(
            [
                initial_gradient,
                summary.features,
                summary.reconstruction,
                summary.totals,
                *gradients,
                *(parameter.detach().clone() for parameter in parameters),
            ]
        )
    for first, second in zip(*results, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
