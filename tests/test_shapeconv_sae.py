"""Unit invariants of the ShapeConv SAE (no corpus; CPU; seconds).

Runs under pytest (``uv run pytest tests/test_shapeconv_sae.py``) or as a script
(``uv run python tests/test_shapeconv_sae.py``), which executes every ``test_*``
function and exits 1 on the first failure. Covered: the projection (including the
flat case), the encoder/decoder adjoint identity, the analytic soft-threshold
response, NMS plateaus and ties, checkpoint round trip with a non-default spec,
refusal of a checkpoint without provenance and of a structural mismatch, migration
of a format-2 checkpoint and of an old pickled instance, save-load-save
preservation of the fit metadata, chunk-size invariance of the features, the
provenance guards (subject overlap, the independent ``bipolar`` switch, a fitted
extractor without provenance, classifier fit subjects), context-crop / full-row code
equality, the residual-pool lifecycle, the pre-emphasis mappings, the sign/shift
behaviour of the diversity term, the bounded calibration against a brute-force order
statistic (ties, no peaks, small allowance, label filter, chunk sizes, unusable rows),
unusable rows in extraction and in the AR fit, the split preflight, and the
maximum-cardinality event matching of the synthetic stage.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.models.components.shapeconv_sae import CHECKPOINT_NAME, AtomSpec, ShapeConvSAE, TrainSpec  # noqa: E402
from src.utils import check_fit_provenance, check_pretrained_provenance  # noqa: E402

PROVENANCE = {"train_subjects": ["s1", "s2"], "data": {"signal_mode": "bipolar", "target_sfreq": 256}}


def _tiny_model(seed: int = 0, **spec_kwargs) -> ShapeConvSAE:
    spec = AtomSpec(n_atoms=4, atom_len=16, **spec_kwargs)
    train_spec = TrainSpec(epochs=1, crop_len=64, crops_per_row=4, crop_batch=64, n_init_samples=64)
    return ShapeConvSAE(spec=spec, train_spec=train_spec, random_state=seed, device="cpu")


def _tiny_loader(seed: int = 0) -> tuple[DataLoader, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 2, 256, generator=generator)
    return DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4), x


def _expect_value_error(fn, message: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(message)


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
    # A zero-mean atom in the difference domain integrates (L + 1 samples, zero baseline) to a
    # waveform that ends where it starts.
    waveform = torch.randn(1, 1, 17, generator=torch.Generator().manual_seed(5))
    waveform[..., -1] = waveform[..., 0]
    with torch.no_grad():
        model.atoms.copy_(ShapeConvSAE._project(waveform.diff(dim=-1)))
    recovered = torch.from_numpy(model.atoms_signal_domain)[None]
    assert recovered.shape == (1, 1, 17)
    assert torch.allclose(recovered, ShapeConvSAE._project(waveform), atol=1e-5)
    # The analysis filter satisfies <s, diff(x)> = <f, x> for any x (up to the display normalization).
    atom = model._project(model.atoms.detach())
    filt = torch.from_numpy(model.atoms_analysis_filter)[None]
    probe = torch.randn(1, 1, 17, generator=torch.Generator().manual_seed(6))
    lhs = (atom * probe.diff(dim=-1)).sum()
    raw_filter = F.pad(atom, (1, 0)) - F.pad(atom, (0, 1))
    assert torch.allclose(lhs, (raw_filter * probe).sum(), atol=1e-5)
    assert torch.allclose(filt, ShapeConvSAE._project(raw_filter), atol=1e-6)
    plain = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16, pre_emphasis="none"), device="cpu")
    assert plain._rows(x).shape == (6, 50)
    assert plain.atoms_signal_domain.shape == (1, 16)


def test_format2_checkpoint_migrates_to_none_and_refuses_diff() -> None:
    loader, x = _tiny_loader()
    model = _tiny_model(pre_emphasis="none")
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model.save_artifacts(out)
        legacy = torch.load(out / CHECKPOINT_NAME, weights_only=True)
        legacy["format"] = 2
        legacy["spec"] = {k: v for k, v in legacy["spec"].items() if k != "pre_emphasis"}
        legacy["train_spec"], legacy["random_state"] = legacy.pop("fit")["train_spec"], 0
        torch.save(legacy, out / "legacy.pt")
        migrated = ShapeConvSAE(spec=AtomSpec(n_atoms=4, atom_len=16, pre_emphasis="none"), device="cpu",
                                pretrained=out / "legacy.pt")
        assert migrated.spec.pre_emphasis == "none"
        assert torch.equal(migrated(x), model(x))
        _expect_value_error(
            lambda: ShapeConvSAE(spec=AtomSpec(n_atoms=4, atom_len=16), device="cpu", pretrained=out / "legacy.pt"),
            "a format-2 checkpoint was applied to differenced rows",
        )
        legacy["format"] = 1
        torch.save(legacy, out / "older.pt")
        _expect_value_error(
            lambda: ShapeConvSAE(spec=AtomSpec(n_atoms=4, atom_len=16, pre_emphasis="none"), device="cpu",
                                 pretrained=out / "older.pt"),
            "a format-1 checkpoint was accepted",
        )


def test_pickled_extractor_guard_and_migration() -> None:
    loader, x = _tiny_loader()
    model = _tiny_model()
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    data_cfg = OmegaConf.create({"signal_mode": "bipolar", "target_sfreq": 256})

    class _Datamodule:
        test_df = pd.DataFrame({"subject": ["s9"]})

    restored = pickle.loads(pickle.dumps(model))
    check_pretrained_provenance(restored, _Datamodule(), data_cfg)
    assert torch.equal(restored(x), model(x))
    restored.provenance = {}
    _expect_value_error(lambda: check_pretrained_provenance(restored, _Datamodule(), data_cfg),
                        "a fitted extractor without provenance passed the guard")
    unfitted = _tiny_model()
    check_pretrained_provenance(unfitted, _Datamodule(), data_cfg)
    # An instance pickled before pre_emphasis existed: no artifact_format, spec without the field.
    old = _tiny_model(pre_emphasis="none")
    old.fit_unsupervised(loader, provenance=PROVENANCE)
    reference = old(x)
    state = old.__dict__.copy()
    state.pop("artifact_format")
    state.pop("fit_meta")
    spec_state = {k: v for k, v in vars(old.spec).items() if k != "pre_emphasis"}
    state["spec"] = AtomSpec.__new__(AtomSpec)
    object.__setattr__(state["spec"], "__dict__", spec_state)
    revived = ShapeConvSAE.__new__(ShapeConvSAE)
    revived.__setstate__(state)
    assert revived.spec.pre_emphasis == "none"
    assert revived.fit_meta.get("migrated") is True
    check_pretrained_provenance(revived, _Datamodule(), data_cfg)
    assert torch.equal(revived(x), reference)


def test_effective_signal_signature_includes_bipolar_switch() -> None:
    class _Extractor:
        provenance = {"train_subjects": ["s1"], "data": {"signal_mode": "raw", "bipolar": False}}

    class _Datamodule:
        test_df = pd.DataFrame({"subject": ["s9"]})

    check_pretrained_provenance(_Extractor(), _Datamodule(), OmegaConf.create({"signal_mode": "raw", "bipolar": False}))
    _expect_value_error(
        lambda: check_pretrained_provenance(
            _Extractor(), _Datamodule(), OmegaConf.create({"signal_mode": "raw", "bipolar": True})),
        "the bipolar switch was not matched",
    )
    check_pretrained_provenance(
        _Extractor(), _Datamodule(), OmegaConf.create({"signal_mode": "raw", "bipolar": True}), allow_data_mismatch=True
    )


def test_save_load_save_preserves_fit_metadata() -> None:
    loader, _ = _tiny_loader()
    fitted = ShapeConvSAE(
        spec=AtomSpec(n_atoms=4, atom_len=16),
        train_spec=TrainSpec(epochs=1, lam=2.0, crop_len=64, crops_per_row=4, crop_batch=64, n_init_samples=64),
        random_state=7, device="cpu",
    )
    fitted.fit_unsupervised(loader, provenance=PROVENANCE)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        fitted.save_artifacts(out)
        reloaded = ShapeConvSAE(
            spec=AtomSpec(n_atoms=4, atom_len=16), train_spec=TrainSpec(lam=0.5), random_state=42, device="cpu",
            pretrained=out / CHECKPOINT_NAME,
        )
        (out / "again").mkdir()
        reloaded.save_artifacts(out / "again")
        saved = torch.load(out / "again" / CHECKPOINT_NAME, weights_only=True)
        assert saved["fit"]["train_spec"]["lam"] == 2.0 and saved["fit"]["random_state"] == 7
        assert saved["provenance"]["train_subjects"] == ["s1", "s2"]


def test_fit_provenance_rejects_old_val_as_new_test() -> None:
    class _Pipeline:
        fit_provenance_ = {"fit_subjects": ["s1", "s2", "v1"], "calibration_subjects": ["v2"]}

    class _Datamodule:
        test_df = pd.DataFrame({"subject": ["t1"]})

    check_fit_provenance(_Pipeline(), _Datamodule())
    for swapped in ("v1", "v2", "s1"):
        _Datamodule.test_df = pd.DataFrame({"subject": [swapped, "t1"]})
        _expect_value_error(lambda: check_fit_provenance(_Pipeline(), _Datamodule()), f"{swapped} in test passed")

    class _Bare:
        pass

    _Datamodule.test_df = pd.DataFrame({"subject": ["t1"]})
    _expect_value_error(lambda: check_fit_provenance(_Bare(), _Datamodule()), "a pipeline without fit subjects passed")


def test_context_crop_code_matches_full_row_code() -> None:
    model = ShapeConvSAE(
        spec=AtomSpec(n_atoms=3, atom_len=16, thresh=0.5, pre_emphasis="none"),
        train_spec=TrainSpec(crop_len=64), device="cpu",
    )
    atoms = model._project(model.atoms)
    row = torch.randn(1, 1, 400, generator=torch.Generator().manual_seed(8))
    full = model._code(*model._scores(row, atoms))
    ctx, crop_len = model.context, model.train_spec.crop_len
    start = 120
    crop = row[..., start : start + crop_len + 2 * ctx]
    crop = crop - model._center(crop).mean(-1, keepdim=True)
    code = model._code(*model._scores(crop, atoms))
    lo, hi = model._scored_positions()
    assert (lo, hi) == (ctx - 15, ctx + crop_len)
    assert torch.allclose(code[..., lo:hi], full[..., start + lo : start + hi], atol=1e-5)
    # The masked decoder is the adjoint of the encoder on the scored part.
    z = torch.randn_like(code)
    lhs = (model._center(F.conv_transpose1d(z, atoms)) * model._center(crop)).sum()
    rhs = (z * F.conv1d(F.pad(model._center(crop), (ctx, ctx)), atoms)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-3)


def test_residual_pool_lifecycle() -> None:
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    residual = torch.randn(4, 1, 64, generator=torch.Generator().manual_seed(9))
    model._collect_reseed_patches(residual)
    assert model._pool_energy.numel() > 0
    dead, reseeded = model._reset_dead_atoms(torch.ones(4, dtype=torch.long), optimizer)
    assert (dead, reseeded) == (0, 0) and model._pool_energy.numel() == 0
    model._collect_reseed_patches(torch.zeros(4, 1, 64))
    model._collect_reseed_patches(torch.full((4, 1, 64), float("nan")))
    assert model._pool_energy.numel() == 0
    before = model.atoms.detach().clone()
    dead, reseeded = model._reset_dead_atoms(torch.zeros(4, dtype=torch.long), optimizer)
    assert (dead, reseeded) == (4, 0) and torch.equal(model.atoms.detach(), before)
    model._collect_reseed_patches(residual)
    dead, reseeded = model._reset_dead_atoms(torch.tensor([0, 1, 1, 1]), optimizer)
    assert (dead, reseeded) == (1, 1) and model._pool_energy.numel() == 0
    assert not torch.equal(model.atoms.detach()[0], before[0])


def test_format3_checkpoint_migrates_new_fields() -> None:
    loader, x = _tiny_loader()
    model = _tiny_model()
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model.save_artifacts(out)
        legacy = torch.load(out / CHECKPOINT_NAME, weights_only=True)
        legacy["format"] = 3
        for key in ("ar_order", "nms_half_width", "amp_min_relative"):
            legacy["spec"].pop(key)
        legacy["state_dict"].pop("ar_coef")
        torch.save(legacy, out / "format3.pt")
        migrated = ShapeConvSAE(spec=AtomSpec(n_atoms=4, atom_len=16), device="cpu", pretrained=out / "format3.pt")
        assert migrated.spec == model.spec
        assert torch.equal(migrated(x), model(x))
        legacy["spec"]["pre_emphasis"] = "diff_ar"
        torch.save(legacy, out / "format3_ar.pt")
        _expect_value_error(
            lambda: ShapeConvSAE(spec=AtomSpec(n_atoms=4, atom_len=16, pre_emphasis="diff_ar"), device="cpu",
                                 pretrained=out / "format3_ar.pt"),
            "a diff_ar checkpoint without ar_coef was accepted",
        )


def test_nms_half_width_sets_neighbourhood_and_context() -> None:
    wide = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16), device="cpu")
    narrow = ShapeConvSAE(spec=AtomSpec(n_atoms=1, atom_len=16, nms_half_width=2), device="cpu")
    pair = torch.zeros(1, 1, 40)
    pair[..., 10], pair[..., 15] = 3.0, 3.0
    assert torch.nonzero(wide._local_maxima(pair))[:, 2].tolist() == [10]
    assert torch.nonzero(narrow._local_maxima(pair))[:, 2].tolist() == [10, 15]
    assert (wide.nms_half, wide.context) == (8, 23) and (narrow.nms_half, narrow.context) == (2, 17)


def test_relative_amp_min_uses_measured_response_scale() -> None:
    from dataclasses import replace

    loader, x = _tiny_loader(seed=11)
    val_loader, _ = _tiny_loader(seed=12)
    model = _tiny_model(seed=11, amp_min=2.0)
    model.fit_unsupervised(loader, val_dataloader=val_loader, provenance=PROVENANCE)
    assert model.atom_response_std is not None and model.atom_response_std.shape == (4,)
    assert torch.allclose(model._event_threshold(), torch.full((4, 1), 2.0))
    model.spec = replace(model.spec, amp_min_relative=True)
    scale = torch.as_tensor(model.atom_response_std).view(-1, 1)
    assert torch.allclose(model._event_threshold(), 2.0 * scale)
    counts = model(x)[:, :4]
    manual = torch.zeros(16, 4)
    for start, code, _ in model._row_chunks(x):
        manual[start : start + code.shape[0]] = (code.abs() > 2.0 * scale).sum(-1).float()
    assert torch.equal(counts, manual.view(8, 2, 4).sum(1))


def test_diff_ar_whitens_a_coloured_background() -> None:
    generator = torch.Generator().manual_seed(13)
    noise = torch.randn(8, 2, 1024, generator=generator)
    x = torch.zeros_like(noise)
    for t in range(2, 1024):  # AR(2) background, strongly coloured
        x[..., t] = 1.5 * x[..., t - 1] - 0.7 * x[..., t - 2] + noise[..., t]
    loader = DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4)
    model = ShapeConvSAE(
        spec=AtomSpec(n_atoms=4, atom_len=16, pre_emphasis="diff_ar", ar_order=4),
        train_spec=TrainSpec(epochs=1, crop_len=64, crops_per_row=4, crop_batch=64, n_init_samples=64),
        device="cpu",
    )
    before = model._rows(x)
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    assert bool(model.ar_coef.abs().sum() > 0)
    after = model._rows(x)
    assert after.shape == before.shape == (16, 1023)

    def lag1(rows: torch.Tensor) -> float:
        a, b = rows[:, 1:], rows[:, :-1]
        return float((a * b).sum() / (a.norm() * b.norm()))

    assert abs(lag1(after)) < 0.15 < abs(lag1(before))
    assert model.atoms_signal_domain.shape == (4, 16 + 4 * 4 + 1)
    assert model.atoms_analysis_filter.shape == (4, 16 + 4 + 1)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_artifacts(Path(tmp))
        reloaded = ShapeConvSAE(spec=model.spec, device="cpu", pretrained=Path(tmp) / CHECKPOINT_NAME)
        assert torch.equal(reloaded.ar_coef, model.ar_coef) and torch.equal(reloaded(x), model(x))


def test_calibrated_thresholds_bound_background_rate() -> None:
    loader, x = _tiny_loader(seed=21)
    model = _tiny_model(seed=21)
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    background, xb = _tiny_loader(seed=22)
    model.calibrate_amp_min(background, false_alarms_per_channel_minute=6.0, sfreq=256.0)
    thresholds = model.calibrated_thresholds
    assert thresholds is not None and thresholds.shape == (4,) and (thresholds > 0).all()
    minutes = 8 * 2 * 256 / 256.0 / 60.0
    counts = model(xb)[:, :4].sum(0)
    assert bool((counts <= int(6.0 * minutes)).all())
    assert torch.allclose(model._event_threshold().flatten(), torch.as_tensor(thresholds))
    with tempfile.TemporaryDirectory() as tmp:
        model.save_artifacts(Path(tmp))
        reloaded = ShapeConvSAE(spec=model.spec, device="cpu", pretrained=Path(tmp) / CHECKPOINT_NAME)
        assert torch.equal(reloaded.amp_min_atom, model.amp_min_atom)
        assert reloaded.fit_meta["amp_min_calibration"]["false_alarms_per_channel_minute"] == 6.0
        assert (Path(tmp) / "sae_amp_min_atom.npy").exists()


def _exact_calibration(model: ShapeConvSAE, x: torch.Tensor, y: torch.Tensor, rate: float, keep_label) -> np.ndarray:
    """Brute-force reference of ``calibrate_amp_min``: every peak of every usable row, then the order statistic."""
    if keep_label is not None:
        x = x[y == keep_label]
    valid = model._valid_rows(x.flatten(0, 1))
    minutes = float(valid.sum()) * x.shape[2] / 256.0 / 60.0
    allowed = int(rate * minutes)
    peaks = [[] for _ in range(model.spec.n_atoms)]
    with torch.no_grad():
        for _, code, _ in model._row_chunks(x):
            magnitude = code.abs()
            for atom in range(model.spec.n_atoms):
                peaks[atom].append(magnitude[:, atom][magnitude[:, atom] > 0])
    out = np.zeros(model.spec.n_atoms)
    for atom in range(model.spec.n_atoms):
        values = torch.cat(peaks[atom])
        if values.numel() == 0:
            out[atom] = 1e-8
        elif values.numel() <= allowed:
            out[atom] = max(float(values.min()) - 1e-8, 1e-8)
        else:
            out[atom] = float(torch.topk(values, allowed + 1).values[-1])
    return out


def test_calibration_matches_exact_order_statistic() -> None:
    """The bounded single-pass calibration equals the brute-force order statistic in every edge case."""
    loader, _ = _tiny_loader(seed=31)
    model = _tiny_model(seed=31)
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    with torch.no_grad():
        model.log_thresh[0] = float(np.log(1e6))  # atom 0 never fires: the no-peak case
    generator = torch.Generator().manual_seed(32)
    base = torch.randn(6, 2, 256, generator=generator)
    x = torch.cat([base, base[:2]])  # duplicated windows: tied peaks
    y = torch.tensor([0, 0, 1, 0, 1, 0, 0, 0])
    for rate, keep_label in ((0.5, None), (6.0, 0), (4000.0, 0)):  # 4000 exceeds every atom's peak count
        expected = _exact_calibration(model, x, y, rate, keep_label)
        for chunk_rows in (1, 5, 16):
            model.chunk_rows = chunk_rows
            model.calibrate_amp_min(DataLoader(TensorDataset(x, y), batch_size=3), rate, 256.0, keep_label=keep_label)
            got = model.calibrated_thresholds
            assert got is not None and np.allclose(got, expected, atol=1e-6), (rate, keep_label, chunk_rows)
    assert model.calibrated_thresholds[0] < 1e-7, "an atom without peaks gets a near-zero threshold"
    # The label filter equals an explicitly negative-only loader; positive windows do not enter.
    model.calibrate_amp_min(DataLoader(TensorDataset(x, y), batch_size=3), 0.5, 256.0, keep_label=0)
    filtered = model.calibrated_thresholds.copy()
    negative_only = DataLoader(TensorDataset(x[y == 0], y[y == 0]), batch_size=3)
    model.calibrate_amp_min(negative_only, 0.5, 256.0)
    assert np.allclose(model.calibrated_thresholds, filtered)
    meta = model.fit_meta["amp_min_calibration"]
    assert meta["channel_minutes"] == meta["channel_minutes_attempted"] and meta["allowance_per_atom"] >= 0
    # Failed loads (all-zero windows), flat channels and non-finite rows change neither the thresholds
    # nor the usable exposure; the excluded exposure is reported.
    broken = x[y == 0].clone()
    broken = torch.cat([broken, torch.zeros(1, 2, 256), broken[:1]])
    broken[-1, 0] = 3.0  # constant nonzero channel
    broken[-1, 1, 10] = float("nan")
    labels = torch.zeros(broken.shape[0], dtype=torch.long)
    model.calibrate_amp_min(DataLoader(TensorDataset(broken, labels), batch_size=4), 0.5, 256.0, keep_label=0)
    assert np.allclose(model.calibrated_thresholds, filtered)
    meta = model.fit_meta["amp_min_calibration"]
    assert abs(meta["channel_minutes_excluded"] - 4 * 256 / 256.0 / 60.0) < 1e-9
    assert abs(meta["channel_minutes"] - 12 * 256 / 256.0 / 60.0) < 1e-9
    _expect_value_error(
        lambda: model.calibrate_amp_min(DataLoader(TensorDataset(torch.zeros(2, 2, 256), labels[:2]), batch_size=2),
                                        0.5, 256.0),
        "calibration on unusable rows only must be refused",
    )


def test_invalid_rows_contribute_nothing() -> None:
    """A failed load, a flat channel or a non-finite row yields zero features and does not touch the AR fit."""
    loader, x = _tiny_loader(seed=41)
    model = _tiny_model(seed=41, pre_emphasis="diff_ar", ar_order=4)
    model.fit_unsupervised(loader, provenance=PROVENANCE)
    reference = model.ar_coef.clone()
    broken = x.clone()
    broken[0] = 0.0  # failed load
    broken[1, 0] = 2.5  # flat channel
    broken[2, 1, 100] = float("inf")  # non-finite sample
    features = model(broken)
    assert torch.isfinite(features).all()
    assert bool((features[0] == 0).all()), "an all-zero window must give zero features"
    assert torch.allclose(model(x)[3:], features[3:]), "other windows are unaffected"
    assert model.events(broken).filter(pl.col("window") == 0).height == 0
    unusable = torch.cat([torch.zeros(1, 2, 256), torch.full((1, 2, 256), float("nan")), torch.ones(1, 2, 256)])
    padded = DataLoader(TensorDataset(torch.cat([x, unusable]), torch.zeros(11, dtype=torch.long)), batch_size=4)
    model._fit_ar(padded)
    assert torch.allclose(model.ar_coef, reference, atol=1e-6), "unusable rows must not enter the autocorrelation"


def test_split_disjoint_preflight() -> None:
    """The manifest preflight refuses empty splits, shared subjects and subjects with two labels."""
    from src.utils import check_split_disjoint

    class _DM:
        def __init__(self, train, val, test) -> None:
            self.train_df, self.val_df, self.test_df = train, val, test

    def frame(subjects, labels):
        return pd.DataFrame({"subject": subjects, "epilepsy": labels, "path": subjects,
                             "start": [0.0] * len(subjects), "end": [1.0] * len(subjects)})

    check_split_disjoint(_DM(frame(["a", "b"], [0, 1]), frame(["c"], [0]), frame(["d"], [1])))
    _expect_value_error(
        lambda: check_split_disjoint(_DM(frame(["a", "b"], [0, 1]), frame(["b"], [1]), frame(["d"], [1]))),
        "a subject in train and val must be refused",
    )
    _expect_value_error(
        lambda: check_split_disjoint(_DM(frame(["a"], [0]), frame([], []), frame(["d"], [1]))),
        "an empty split must be refused",
    )
    _expect_value_error(
        lambda: check_split_disjoint(_DM(frame(["a", "a"], [0, 1]), frame(["c"], [0]), frame(["d"], [1]))),
        "a subject with two labels must be refused",
    )


def test_event_matching_is_maximum_cardinality() -> None:
    """The stage's matcher finds both feasible pairs where a greedy closest-first pass finds one."""
    from tests.stage7_shapeconv_sae import _match_events

    true = np.array([[0, 0, 0], [0, 0, 6]])
    pred = np.array([[0, 0, 4], [0, 0, 10]])
    mp, mt = _match_events(np, pred, true, tol=6)
    assert len(mp) == 2 and sorted(mp.tolist()) == [0, 1] and sorted(mt.tolist()) == [0, 1]
    # Among maximum matchings the timing cost is minimal: 4 -> 6 and 10 -> ... is infeasible, so 4 -> 0, 10 -> 6.
    assert set(zip(mp.tolist(), mt.tolist())) == {(0, 0), (1, 1)}
    mp, mt = _match_events(np, pred[:1], true, tol=6)
    assert (mp.tolist(), mt.tolist()) == ([0], [1]), "a single prediction takes its closest feasible event"


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
