"""Gate-0 unit invariants of the separable atom SAE (no corpus; CPU; seconds).

Runs under pytest (``uv run pytest tests/test_separable_sae.py``) or as a script
(``uv run python tests/test_separable_sae.py``), which executes every ``test_*``
function and exits 1 on the first failure. The tests are section 3 of
``documentation/separable_atom_sae_protocol.md``; equation numbers refer to
``documentation/Separable Atom Encoder Mathematics.md``. Covered: locality of every
window to its extended window (25), the decoder as the adjoint of the encoder,
identity (26) and the R=1 optimum against a brute-force search, the tie rules of (10),
(17) and (18), the index maps of (20) and (22), the separable fast path against the
full-weight convolutions, equality with format 5 at D=1 under central loss support,
the Adam step of each factor with and without step matching, a small fit on planted
rank-1 events with the unit-norm constraint, the checkpoint round trip with its refusals, and the
spec guards.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.models.components.separable_sae import MODEL_TAG, SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.waveform_sae import (  # noqa: E402
    CHECKPOINT_NAME,
    WaveformSAE,
    WaveformSpec,
    WaveformTrainSpec,
)

PROVENANCE = {"train_subjects": ["s1", "s2"], "data": {"signal_mode": "bipolar", "target_sfreq": 256}}
TOLERANCE = 1e-10
N_CHANNELS = 3


def _model(**spec_kwargs: object) -> SeparableSAE:
    """A CPU model with a small geometry (K=3, L=8, P=8, D=2, R=1) and Gaussian factors over three channels."""
    spec = SeparableSpec(**{"n_atoms": 3, "atom_len": 8, "pool_len": 8, "n_projections": 2, **spec_kwargs})
    model = SeparableSAE(spec=spec, device="cpu")
    width = spec.atom_len + N_CHANNELS * spec.n_projections
    model.atoms = nn.Parameter(model._constrain(_randn(spec.n_atoms, width, seed=7)))
    return model


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _expect_error(function: Callable[[], object], error: type[Exception], message: str) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError(message)


def _projection(model: SeparableSAE, rows: torch.Tensor) -> torch.Tensor:
    """The winning projection ``d_k^star`` ``(B, K, T)`` of equation (10)."""
    return model._correlate(F.pad(rows, model.padding), model._dictionary()).max(dim=2).indices


def _placed(model: SeparableSAE, atom: int, projection: int, start: int) -> torch.Tensor:
    """The template ``u_{k,d} w_k^T`` placed at local position ``start`` of one extended window ``(C, P + L - 1)``."""
    time, channel = model._split(model._dictionary().detach())
    length = model.spec.atom_len
    placed = torch.zeros(N_CHANNELS, model.spec.pool_len + length - 1, dtype=time.dtype)
    placed[:, start : start + length] = channel[atom, :, projection, None] * time[atom][None, :]
    return placed


def _full_weight(model: SeparableSAE) -> torch.Tensor:
    """The effective templates as one ``(K * D, C, L)`` convolution weight."""
    return torch.from_numpy(model.atoms_numpy).flatten(0, 1)


def test_window_depends_only_on_its_extended_window() -> None:
    model = _model()
    pool, (left, right) = model.spec.pool_len, model.padding
    rows = _randn(1, N_CHANNELS, 4 * pool, seed=1)
    valid = torch.ones(1, N_CHANNELS, dtype=torch.bool)
    window = 1
    support = torch.zeros(rows.shape[-1], dtype=torch.bool)
    support[window * pool - left : (window + 1) * pool + right] = True
    changed = rows + _randn(*rows.shape, seed=2) * (~support).to(rows.dtype)
    span = slice(window * pool, (window + 1) * pool)
    before = model._encode_reconstruct(rows, valid, model._dictionary())
    after = model._encode_reconstruct(changed, valid, model._dictionary())
    assert torch.allclose(before[0][..., span], after[0][..., span], atol=TOLERANCE), "the window's code changed"
    assert torch.equal(before[1][..., window], after[1][..., window]), "the selection of the window changed"
    assert torch.equal(_projection(model, rows)[..., span], _projection(model, changed)[..., span])
    assert torch.allclose(before[4][:, :, window], after[4][:, :, window], atol=TOLERANCE), "the window changed"
    assert not torch.allclose(before[4], after[4], atol=TOLERANCE), "the perturbation reached no window"


def test_decoder_is_the_adjoint_of_the_encoder() -> None:
    model = _model()
    spec = model.spec
    extended = _randn(1, N_CHANNELS, spec.pool_len + spec.atom_len - 1, seed=3).requires_grad_()
    code = _randn(1, spec.n_atoms, spec.pool_len, seed=4)
    projection = torch.randint(0, spec.n_projections, code.shape, generator=torch.Generator().manual_seed(5))
    lifted = torch.zeros(1, spec.n_atoms, spec.n_projections, spec.pool_len, dtype=code.dtype)
    lifted = lifted.scatter(2, projection.unsqueeze(2), code.unsqueeze(2))
    responses = model._correlate(extended, model._dictionary().detach())
    (gradient,) = torch.autograd.grad((responses * lifted).sum(), extended)
    decoded = model._decode_extended(code, projection, model._dictionary().detach())
    assert decoded.shape == (1, N_CHANNELS, 1, spec.pool_len + spec.atom_len - 1)
    assert torch.allclose(decoded[:, :, 0], gradient, atol=TOLERANCE), "equation (23) is not the encoder's transpose"


def test_window_loss_identity_and_the_one_template_optimum() -> None:
    for top_r in (1, 2):
        model = _model(top_r=top_r)
        rows = _randn(1, N_CHANNELS, model.spec.pool_len, seed=10 + top_r)
        valid = torch.ones(1, N_CHANNELS, dtype=torch.bool)
        code, _, _, target, estimate = model._encode_reconstruct(rows, valid, model._dictionary().detach())
        window, decoded = target[0, :, 0], estimate[0, :, 0]
        projection = _projection(model, rows)[0]
        placements = [(int(k), int(q)) for k, q in torch.nonzero(code[0] > 0)]
        assert 1 <= len(placements) <= top_r, "the window holds more than R activations"
        templates = [_placed(model, k, int(projection[k, q]), q) for k, q in placements]
        amplitudes = [float(code[0, k, q]) for k, q in placements]
        for template, amplitude in zip(templates, amplitudes):
            assert abs(float(template.norm()) - 1.0) < TOLERANCE, "a placed template has no unit norm"
            assert abs(float((template * window).sum()) - amplitude) < TOLERANCE, "an amplitude is not <Phi, X>"
        assert torch.allclose(decoded, sum(a * t for a, t in zip(amplitudes, templates)), atol=TOLERANCE)
        cross = sum(
            a * b * float((s * t).sum())
            for i, (a, s) in enumerate(zip(amplitudes, templates))
            for j, (b, t) in enumerate(zip(amplitudes, templates))
            if i != j
        )
        identity = float(window.pow(2).sum()) - sum(a * a for a in amplitudes) + cross
        loss = float((window - decoded).pow(2).sum())
        assert abs(loss - identity) < 1e-9, "identity (26) does not hold"
        if top_r == 1:
            best = min(
                float((window - max(float((t * window).sum()), 0.0) * t).pow(2).sum())
                for t in (
                    _placed(model, k, d, q)
                    for k in range(model.spec.n_atoms)
                    for d in range(model.spec.n_projections)
                    for q in range(model.spec.pool_len)
                )
            )
            assert abs(loss - best) < 1e-9, "the R=1 code is not the best one-template code"


def test_tie_rules() -> None:
    model = _model()
    time, channel = model._split(model._dictionary().detach())
    channel = channel.clone()
    channel[..., 1] = channel[..., 0]
    model.atoms = nn.Parameter(torch.cat([time, channel.flatten(1)], dim=1))
    rows = _randn(2, N_CHANNELS, 2 * model.spec.pool_len, seed=20)
    assert bool((_projection(model, rows) == 0).all()), "equal projections do not prefer the smaller index"
    competing = _model(n_atoms=5, top_r=2)
    pooled = torch.tensor([[0.7], [0.0], [0.9], [0.7], [0.0]], dtype=torch.float64)
    selected = competing._select_atoms(pooled)[:, 0].tolist()
    assert selected == [True, False, True, False, False], "the rank rule (18) does not select atoms 1 and 3"
    responses = torch.zeros(1, 5, competing.spec.pool_len, dtype=torch.float64)
    responses[0, 0, [2, 5]] = 1.0
    code, _ = competing._selection(responses)
    assert torch.nonzero(code[0, 0]).flatten().tolist() == [2], "equal positions do not prefer the earliest"


def test_index_maps_of_the_target_and_the_windows() -> None:
    model = _model()
    pool, length, (left, _) = model.spec.pool_len, model.spec.atom_len, model.padding
    rows = _randn(1, N_CHANNELS, 3 * pool, seed=30)
    target = F.pad(rows, model.padding).unfold(-1, pool + length - 1, pool)
    assert target.shape == (1, N_CHANNELS, 3, pool + length - 1)
    for m in range(3):
        for j in range(pool + length - 1):
            sample = m * pool - left + j
            expected = rows[0, :, sample] if 0 <= sample < rows.shape[-1] else torch.zeros(N_CHANNELS)
            assert torch.equal(target[0, :, m, j], expected.to(rows.dtype)), "the target is not equation (22)"
    windows = model._pool_batch(rows)
    for m in range(3):
        assert torch.equal(windows[m], rows[0, :, m * pool : (m + 1) * pool]), "the window split is not (20)"


def test_separable_fast_path_equals_the_full_weight_convolutions() -> None:
    model = _model()
    spec = model.spec
    rows = _randn(2, N_CHANNELS, 3 * spec.pool_len, seed=40)
    padded = F.pad(rows, model.padding)
    weight = _full_weight(model)
    responses = model._correlate(padded, model._dictionary().detach()).flatten(1, 2)
    assert torch.allclose(responses, F.conv1d(padded, weight), atol=TOLERANCE), "the encoder fast path differs"
    code = F.relu(_randn(2, spec.n_atoms, rows.shape[-1], seed=41))
    projection = torch.randint(0, spec.n_projections, code.shape, generator=torch.Generator().manual_seed(42))
    lifted = torch.zeros(*code.shape[:2], spec.n_projections, code.shape[-1], dtype=code.dtype)
    lifted = lifted.scatter(2, projection.unsqueeze(2), code.unsqueeze(2)).flatten(1, 2)
    reference = F.conv_transpose1d(model._pool_batch(lifted), weight).unflatten(0, (2, -1)).transpose(1, 2)
    decoded = model._decode_extended(code, projection, model._dictionary().detach())
    assert torch.allclose(decoded, reference, atol=TOLERANCE), "the decoder fast path differs"


def test_one_projection_with_central_support_equals_format_5() -> None:
    separable = _model(n_projections=1, loss_support="central")
    spec = separable.spec
    format5 = WaveformSAE(
        spec=WaveformSpec(n_atoms=spec.n_atoms, atom_len=spec.atom_len, pool_len=spec.pool_len, top_r=spec.top_r),
        device="cpu",
    )
    format5.atoms = nn.Parameter(torch.from_numpy(separable.atoms_numpy)[:, 0])
    rows = _randn(2, N_CHANNELS, 3 * spec.pool_len, seed=50)
    rows[0, 1] = 0.0
    valid = torch.ones(2, N_CHANNELS, dtype=torch.bool)
    valid[0, 1] = False
    ours = separable._encode_reconstruct(rows, valid, separable._dictionary())
    theirs = format5._encode_reconstruct(rows, valid, format5._dictionary())
    assert torch.allclose(ours[0], theirs[0], atol=TOLERANCE) and torch.equal(ours[1], theirs[1])
    assert torch.allclose(ours[2], theirs[2], atol=TOLERANCE), "the reconstructions differ"
    loss_ours, stats_ours, _ = separable._segment_step(rows, valid, separable._dictionary())
    loss_theirs, stats_theirs, _ = format5._segment_step(rows, valid, format5._dictionary())
    assert abs(float(loss_ours) - float(loss_theirs)) < TOLERANCE and torch.allclose(stats_ours, stats_theirs)


def _planted(seed: int, n_windows: int = 24) -> torch.Tensor:
    """Windows ``(n, 3, 512)``: one rank-1 event per 32 samples, two atoms with two topographies each, plus noise."""
    generator = torch.Generator().manual_seed(seed)
    # The template seed must differ from the model's random_state, or a random start could begin at the answer.
    fixed = torch.Generator().manual_seed(123)
    waveforms = F.normalize(torch.randn(2, 16, generator=fixed), dim=-1)
    topographies = F.normalize(torch.randn(2, 2, 3, generator=fixed), dim=-1)
    x = 0.05 * torch.randn(n_windows, 3, 512, generator=generator)
    for window in range(n_windows):
        for block in range(16):
            start = 32 * block + int(torch.randint(0, 16, (1,), generator=generator))
            atom, focus = block % 2, int(torch.randint(0, 2, (1,), generator=generator))
            amplitude = 2.0 + float(torch.rand(1, generator=generator))
            x[window, :, start : start + 16] += amplitude * topographies[atom, focus, :, None] * waveforms[atom]
    return x


def _fitted(epochs: int) -> tuple[SeparableSAE, torch.Tensor]:
    spec = SeparableSpec(n_atoms=2, atom_len=16, pool_len=32, n_projections=2)
    train_spec = WaveformTrainSpec(epochs=epochs, lr=1e-2, segment_batch=4, init="random", recenter_atoms=False)
    model = SeparableSAE(spec=spec, train_spec=train_spec, random_state=0, device="cpu")
    x, x_val = _planted(1), _planted(2, n_windows=8)
    loader = DataLoader(TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), batch_size=8)
    val_loader = DataLoader(TensorDataset(x_val, torch.zeros(len(x_val), dtype=torch.long)), batch_size=8)
    model.fit_unsupervised(loader, val_loader, provenance=PROVENANCE)
    return model, x


def _residual(model: SeparableSAE, x: torch.Tensor) -> float:
    rows, valid = model._full_rows(x)
    totals, _ = model._window_stats(rows, valid, model._dictionary())
    return float(totals[0] / totals[1])


def test_step_matching_scales_the_adam_step_of_each_factor() -> None:
    for matched, expected in ((True, (np.sqrt(N_CHANNELS), np.sqrt(8))), (False, (1.0, 1.0))):
        model = _model(step_matched=matched)
        time, channel = model._split(model.atoms.detach())
        assert torch.allclose(model._split(model._pack(time, channel))[0], time), "_pack is not the inverse of _split"
        before = [t.clone() for t in (time, channel)]
        weights = (_randn(*time.shape, seed=60), _randn(*channel.shape, seed=61))
        optimizer = torch.optim.Adam([model.atoms], lr=1e-3)
        time, channel = model._split(model.atoms)
        ((time * weights[0]).sum() + (channel * weights[1]).sum()).backward()
        optimizer.step()
        after = model._split(model.atoms.detach())
        for factor, (old, new) in enumerate(zip(before, after)):
            step, target = (new - old).abs(), 1e-3 * expected[factor]
            assert torch.allclose(step, torch.full_like(step, target), rtol=1e-4), (
                f"factor {factor} moved by {float(step.mean()):.2e}, not {target:.2e} (step_matched={matched})"
            )


def test_fit_reduces_the_residual_and_keeps_the_constraint() -> None:
    initial, x = _fitted(epochs=0)
    model, _ = _fitted(epochs=30)
    assert model.fitted and len(model.history) == 30 and "n_atoms_never_won" in model.history[-1]
    assert _residual(model, x) < _residual(initial, x) - 1e-6, "training did not lower the residual"
    time, channel = model._split(model.atoms.detach())
    assert torch.allclose(torch.linalg.vector_norm(time, dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(channel, dim=1), torch.ones(2, 2), atol=1e-5)
    assert model.fit_meta["n_channels"] == 3 and model.n_channels == 3


def test_checkpoint_round_trip_and_refusals() -> None:
    model, x = _fitted(epochs=2)
    code, _ = model.encode(x[:2])
    assert model(x[:2]).shape == (2, 2)
    _expect_error(lambda: model.decode(code), NotImplementedError, "decode without the projections ran")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model.save_artifacts(out)
        path = out / CHECKPOINT_NAME
        reloaded = SeparableSAE(spec=model.spec, device="cpu", pretrained=path)
        assert reloaded.fitted and reloaded.train_subjects == ["s1", "s2"] and reloaded.n_channels == 3
        assert reloaded.history == model.history and torch.equal(reloaded.encode(x[:2])[0], code)
        assert np.load(out / "sae_atoms.npy").shape == (2, 2, 3, 16)
        assert np.load(out / "sae_time_atoms.npy").shape == (2, 16)
        assert np.load(out / "sae_channel_atoms.npy").shape == (2, 3, 2)
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["model"] == MODEL_TAG and checkpoint["format"] == 2
        torch.save({**checkpoint, "model": "waveform_sae", "format": 5}, out / "format5.pt")
        for spec, bad in ((model.spec, out / "format5.pt"), (replace(model.spec, n_projections=3), path)):
            _expect_error(
                lambda spec=spec, bad=bad: SeparableSAE(spec=spec, device="cpu", pretrained=bad),
                ValueError,
                f"{bad.name} with {spec} was not refused",
            )
        _expect_error(
            lambda: WaveformSAE(spec=WaveformSpec(n_atoms=2, atom_len=16, pool_len=32), device="cpu", pretrained=path),
            ValueError,
            "format 5 loaded a separable checkpoint",
        )


def test_spec_guards() -> None:
    _expect_error(lambda: SeparableSAE(spec=WaveformSpec(), device="cpu"), TypeError, "a WaveformSpec was accepted")
    for spec in (
        SeparableSpec(n_projections=0),
        SeparableSpec(loss_support="wide"),
        SeparableSpec(atom_norm="free"),
        SeparableSpec(center_atoms=True),
        SeparableSpec(step_matched=1),  # type: ignore[arg-type]
    ):
        _expect_error(lambda spec=spec: SeparableSAE(spec=spec, device="cpu"), ValueError, f"{spec} was accepted")
    for train_spec in (WaveformTrainSpec(), WaveformTrainSpec(recenter_atoms=False, crop_sampling="tiled")):
        _expect_error(
            lambda train_spec=train_spec: SeparableSAE(train_spec=train_spec, device="cpu"),
            ValueError,
            f"{train_spec} was accepted",
        )
    assert SeparableSAE(device="cpu").n_channels is None


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
