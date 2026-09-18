"""Unit invariants of the ShapeConv SAE (no corpus; CPU; seconds).

Runs under pytest (``uv run pytest tests/test_shapeconv_sae.py``) or as a script
(``uv run python tests/test_shapeconv_sae.py``), which executes every ``test_*``
function and exits 1 on the first failure. Covered: the projection (including the
flat case), the encoder/decoder adjoint identity, the analytic soft-threshold
response, NMS plateaus and ties, checkpoint round trip with a non-default spec,
refusal of a checkpoint without provenance and of a structural mismatch, chunk-size
invariance of the features, cross-split rejection by the provenance check, and the
sign/shift behaviour of the diversity term.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.models.components.shapeconv_sae import CHECKPOINT_NAME, AtomSpec, ShapeConvSAE, TrainSpec  # noqa: E402
from src.utils import check_pretrained_provenance  # noqa: E402

PROVENANCE = {"train_subjects": ["s1", "s2"], "data": {"signal_mode": "bipolar", "target_sfreq": 256}}


def _tiny_model(seed: int = 0, **spec_kwargs) -> ShapeConvSAE:
    spec = AtomSpec(n_atoms=4, atom_len=16, **spec_kwargs)
    train_spec = TrainSpec(epochs=1, crop_len=64, crops_per_row=4, crop_batch=64, n_init_samples=64)
    return ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=seed, device="cpu")


def _tiny_loader(seed: int = 0) -> tuple[DataLoader, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 2, 256, generator=generator)
    return DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4), x


def test_projection_zero_mean_unit_norm_and_flat() -> None:
    atoms = torch.randn(5, 1, 16)
    projected = ShapeConvSAE._project(atoms)
    assert torch.allclose(projected.mean(-1), torch.zeros(5, 1), atol=1e-6)
    assert torch.allclose(projected.norm(dim=-1), torch.ones(5, 1), atol=1e-6)
    assert torch.equal(ShapeConvSAE._project(torch.ones(1, 1, 16)), torch.zeros(1, 1, 16))


def test_adjoint_identity() -> None:
    generator = torch.Generator().manual_seed(1)
    atoms = ShapeConvSAE._project(torch.randn(3, 1, 8, generator=generator))
    x = torch.randn(2, 1, 40, generator=generator)
    code = torch.randn(2, 3, 33, generator=generator)
    lhs = (F.conv1d(x, atoms) * code).sum()
    rhs = (x * F.conv_transpose1d(code, atoms)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_shrink_analytic_response() -> None:
    model = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16, mode="shrink", thresh=3.0), device="cpu")
    atom = torch.zeros(1, 1, 16)
    atom[0, 0, 0], atom[0, 0, 1] = 1.0, -1.0
    with torch.no_grad():
        model.atoms.copy_(ShapeConvSAE._project(atom))
    row = torch.zeros(1, 1, 200)
    row[..., 100:116] = 10.0 * model.atoms.detach()[0, 0]
    amplitude, rho = model._scores(row, model._project(model.atoms))
    code = model._code(amplitude, rho)
    nonzero = torch.nonzero(code)
    assert nonzero.tolist() == [[0, 0, 100]]
    assert abs(float(code[0, 0, 100]) - 7.0) < 1e-4
    assert abs(float(rho[0, 0, 100]) - 1.0) < 1e-5


def test_nms_plateau_and_ties() -> None:
    model = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=8), device="cpu")  # half-width 4
    plateau = torch.zeros(1, 1, 20)
    plateau[..., 5:8] = 5.0
    assert torch.nonzero(model._local_maxima(plateau))[:, 2].tolist() == [5]
    far = torch.zeros(1, 1, 20)
    far[..., 2], far[..., 12] = 3.0, 3.0
    assert torch.nonzero(model._local_maxima(far))[:, 2].tolist() == [2, 12]
    near = torch.zeros(1, 1, 20)
    near[..., 2], near[..., 5] = 3.0, 3.0
    assert torch.nonzero(model._local_maxima(near))[:, 2].tolist() == [2]


def test_checkpoint_round_trip_and_refusals() -> None:
    loader, x = _tiny_loader()
    model = _tiny_model(rho_min=0.3, amp_min=1.5)
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    features = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model.save_artifacts(out)
        reloaded = ShapeConvSAE(spec=AtomSpec(n_atoms=4, atom_len=16), device="cpu", pretrained=out / CHECKPOINT_NAME)
        assert reloaded.spec == model.spec
        assert reloaded.fitted and reloaded.train_subjects == ["s1", "s2"]
        assert torch.equal(reloaded(x), features)
        reloaded.fit_unsupervised(loader, provenance={"train_subjects": ["other"]})
        assert reloaded.train_subjects == ["s1", "s2"]
        overridden = ShapeConvSAE(
            spec=AtomSpec(n_atoms=4, atom_len=16, amp_min=9.0), device="cpu",
            pretrained=out / CHECKPOINT_NAME, override_spec=True,
        )
        assert overridden.spec.amp_min == 9.0 and overridden.spec.rho_min == 0.0
        checkpoint = torch.load(out / CHECKPOINT_NAME, weights_only=True)
        checkpoint["provenance"] = {}
        torch.save(checkpoint, out / "bare.pt")
        for spec, path in ((AtomSpec(n_atoms=4, atom_len=16), out / "bare.pt"),
                           (AtomSpec(n_atoms=5, atom_len=16), out / CHECKPOINT_NAME)):
            try:
                ShapeConvSAE(spec=spec, device="cpu", pretrained=path)
            except ValueError:
                continue
            raise AssertionError(f"{path.name} with {spec} was not refused")


def test_features_invariant_to_chunk_size() -> None:
    loader, x = _tiny_loader(seed=3)
    model = _tiny_model(seed=3)
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    model.chunk_rows = 1
    one = model(x)
    model.chunk_rows = 5
    assert torch.allclose(one, model(x), atol=1e-5)
    assert one.shape == (8, 12)


def test_provenance_check_rejects_overlap_and_mismatch() -> None:
    class _Extractor:
        provenance = PROVENANCE

    class _Datamodule:
        val_df = pd.DataFrame({"subject": ["s9"]})
        test_df = pd.DataFrame({"subject": ["s2", "s5"]})

    data_cfg = OmegaConf.create({"signal_mode": "bipolar", "target_sfreq": 256})
    try:
        check_pretrained_provenance(_Extractor(), _Datamodule(), data_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("test-subject overlap was not refused")
    _Datamodule.test_df = pd.DataFrame({"subject": ["s5"]})
    check_pretrained_provenance(_Extractor(), _Datamodule(), data_cfg, check_val=True)
    try:
        check_pretrained_provenance(_Extractor(), _Datamodule(), OmegaConf.create({"signal_mode": "raw"}))
    except ValueError:
        pass
    else:
        raise AssertionError("data-setting mismatch was not refused")


def test_pre_emphasis_rows_and_signal_domain_atoms() -> None:
    model = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16, pre_emphasis="diff"), device="cpu")
    x = torch.cumsum(torch.randn(2, 3, 50, generator=torch.Generator().manual_seed(4)), dim=-1)
    rows = model._rows(x)
    assert rows.shape == (6, 49)
    assert torch.allclose(rows, ShapeConvSAE._robust_scale(x.flatten(0, 1).diff(dim=-1)))
    # A zero-mean atom in the difference domain integrates to a waveform that ends where it starts.
    waveform = torch.randn(1, 1, 17, generator=torch.Generator().manual_seed(5))
    waveform[..., -1] = waveform[..., 0]
    with torch.no_grad():
        model.atoms.copy_(ShapeConvSAE._project(waveform.diff(dim=-1)))
    recovered = torch.from_numpy(model.atoms_signal_domain)[None]
    target = ShapeConvSAE._project(waveform[..., 1:])
    assert torch.allclose(recovered, target, atol=1e-5)
    plain = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16, pre_emphasis="none"), device="cpu")
    assert plain._rows(x).shape == (6, 50)


def test_diversity_sign_and_shift_aware() -> None:
    model = ShapeConvSAE(spec=AtomSpec(n_atoms=2, atom_len=16), device="cpu")
    # Support on the first 10 samples only, zero-mean there, so a shift by 3 is an exact linear shift.
    base = torch.zeros(1, 1, 16)
    base[..., :10] = torch.randn(10, generator=torch.Generator().manual_seed(2))
    base[..., :10] -= base[..., :10].mean()
    atom = ShapeConvSAE._project(base)
    flipped = torch.cat([atom, -atom])
    assert abs(float(model._diversity(flipped)) - 1.0) < 1e-5
    shifted = torch.cat([atom, torch.roll(atom, 3, dims=-1)])
    assert abs(float(model._diversity(shifted)) - 1.0) < 1e-5
    single = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16), device="cpu")
    assert float(single._diversity(atom)) == 0.0


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
