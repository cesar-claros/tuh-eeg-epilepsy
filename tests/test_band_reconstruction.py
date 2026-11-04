"""Checks of the band-limited reconstruction driver ``src/band_reconstruction.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the band-passed view, the domain record and its refusals, the geometry check, and ``read`` end to end on
written fold files: each outcome of the two-arm rule, and the gates that block it (the band-pass, the row match).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

import src.band_reconstruction as driver  # noqa: E402
import src.band_reconstruction_core as core  # noqa: E402
import src.reconstruction_baselines_core as base  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402

N_SEGMENTS = 2
SLOW, DELTA = [0], [1]  # band columns below 1 Hz and 1-4 Hz


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


class _Segments:
    """A stand-in for the pooled cache view: two tones per segment, 0.3 Hz and 10 Hz."""

    plan = pd.DataFrame({"subject": ["p1", "p2"], "epilepsy": [1, 0]})
    target_channels = ["a", "b"]

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = np.arange(30_720) / 256.0
        x = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 10.0 * t)
        return torch.from_numpy(np.stack([x, x]).astype(np.float32)), torch.tensor(index % 2)


def test_band_passed_view_removes_the_slow_tone() -> None:
    view = driver.BandPassed(_Segments(), 256.0)
    assert len(view) == 2 and view.plan is _Segments.plan and view.target_channels == ["a", "b"]
    x, y = view[1]
    assert x.dtype == torch.float32 and x.shape == (2, 30_720) and int(y) == 1
    middle = x[0, 1280:-1280].double()
    assert abs(float(middle.pow(2).mean()) - 0.125) < 0.005, "only the 10 Hz tone (power 0.125) must remain"


def test_domain_record_and_refusals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "read"
        driver.check_domain(root)
        assert json.loads((root / "domain.json").read_text()) == driver.DOMAIN
        driver.check_domain(root)
        (root / "domain.json").write_text(json.dumps({**driver.DOMAIN, "band_hz": [1.0, 45.0]}))
        _expect_value_error(lambda: driver.check_domain(root), "a root of another band was accepted")
        other = Path(tmp) / "baselines"
        (other / "scores").mkdir(parents=True)
        _expect_value_error(lambda: driver.check_domain(other), "a baselines root was accepted")
        _expect_value_error(lambda: driver.read(other, other), "a read without a domain record was accepted")


def test_training_refuses_nondeterministic_cublas() -> None:
    saved = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        try:
            driver.require_deterministic_cublas()
        except RuntimeError:
            pass
        else:
            raise AssertionError("training without CUBLAS_WORKSPACE_CONFIG was accepted")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        driver.require_deterministic_cublas()
    finally:
        if saved is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved
    launcher = (Path(__file__).resolve().parents[1] / "run_band_reconstruction.sh").read_text()
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in launcher, "the launcher must export the cuBLAS setting"


def test_geometry_check() -> None:
    for r in core.RS:
        spec = SeparableSpec(n_atoms=64, atom_len=256, pool_len=512, n_projections=4, top_r=r)
        driver.check_geometry(SeparableSAE(spec=spec, device="cpu"), r)
        _expect_value_error(
            lambda spec=spec, r=r: driver.check_geometry(SeparableSAE(spec=spec, device="cpu"), 5 - r), "R"
        )


def _factors(fast: dict[int, float], pca1_delta: float = 0.95) -> np.ndarray:
    """Residual over signal ``(n_arms, n_bands)``: the trained arms explain delta, and ``fast[r]`` is their 4-30 Hz
    residual; every other arm explains nothing outside delta."""
    factors = np.ones((len(core.ARMS), len(base.BANDS)))
    for r in core.RS:
        factors[core.ARMS.index(core.TRAINED[r]), DELTA] = 0.6 if r == 1 else 0.5
        factors[core.ARMS.index(core.TRAINED[r]), list(core.FAST)] = fast[r]
    factors[core.ARMS.index("PCA-1"), DELTA] = pca1_delta
    for k in base.PCA_KS[1:]:
        factors[core.ARMS.index(f"PCA-{k}"), DELTA] = 0.95 - 0.1 * np.log2(k)
    factors[core.ARMS.index("PCA-4"), DELTA] = 0.75
    factors[core.ARMS.index("S4-pass1"), SLOW] = 0.9
    factors[core.ARMS.index("SVD-1")] = 0.5
    factors[core.ARMS.index("SVD-4")] = 0.3
    return factors


def _write(
    tmp: Path, fast: dict[int, float], *, pca1_delta: float = 0.95, slow_energy: float = 0.02, offset: float = 0.0
) -> tuple[Path, Path]:
    """A readout plan, a band root with training curves and random starts, and five scored folds."""
    readout, root = tmp / "readout", tmp / "band"
    labels = {f"p{i:03d}": int(i % 2 == 0) for i in range(20)}
    folds = {s: i % 5 for i, s in enumerate(sorted(labels))}
    readout.mkdir(parents=True)
    (readout / "folds.json").write_text(
        json.dumps({"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds})
    )
    driver.check_domain(root)
    rng = np.random.default_rng(0)
    factors = _factors(fast, pca1_delta)
    start = rng.standard_normal((64, 256))
    for fold in range(5):
        subject = np.repeat([s for s in sorted(labels) if folds[s] == fold], N_SEGMENTS)
        signal = rng.uniform(1.0, 2.0, (len(subject), len(base.BANDS)))
        signal[:, SLOW + [5]] *= slow_energy
        residual = factors[:, None, :] * signal[None]
        for r in core.RS:
            held_out = 1.0 - residual[core.ARMS.index(core.TRAINED[r])].sum() / signal.sum()
            curve = driver.dictionary_dir(root, "trained", r, fold)
            curve.mkdir(parents=True)
            frame = pl.DataFrame({"epoch": [1, 2], "val_residual_frac": [0.99, 1.0 - held_out - offset]})
            frame.write_csv(curve / "sae_training.csv")
            untrained = driver.dictionary_dir(root, "untrained", r, fold)
            untrained.mkdir(parents=True)
            np.save(untrained / "sae_time_atoms.npy", start)
        (readout / "dictionaries" / f"untrained_f{fold}").mkdir(parents=True)
        np.save(readout / "dictionaries" / f"untrained_f{fold}" / "sae_time_atoms.npy", start)
        n_seg = len(subject)
        two = (len(core.BUDGET_ARMS), n_seg)
        path = driver.score_path(root, fold)
        driver._save_npz(
            path,
            subject=subject,
            fold=np.array(fold),
            arms=np.array(core.ARMS),
            model_arms=np.array(core.MODEL_ARMS),
            budget_arms=np.array(core.BUDGET_ARMS),
            bands=np.array(base.BAND_NAMES),
            n_channels=np.array(20),
            windows_per_segment=np.array(60),
            ceiling_violations=np.zeros(len(core.MODEL_ARMS), dtype=np.int64),
            ceiling_excess=np.full(len(core.MODEL_ARMS), -0.1),
            atom_wins=np.ones((len(core.BUDGET_ARMS), 64), dtype=np.int64),
            dominant_hz=np.tile(np.linspace(1.0, 30.0, 64), (len(core.BUDGET_ARMS), 1)),
            signal_bands=signal,
            signal_total=signal.sum(axis=1),
            residual_bands=residual,
            residual_total=residual.sum(axis=2),
            usable_rows=np.full(n_seg, 20),
            edge_energy=np.full(n_seg, 1.1),
            interior_energy=np.full(n_seg, 1.0),
            windows_won=np.full(two, 60),
            windows_slow=np.full(two, 30),
            slots_won=np.stack([np.full(n_seg, 60), np.full(n_seg, 240)]),
            slots_slow=np.stack([np.full(n_seg, 30), np.full(n_seg, 120)]),
            explained_all=np.full(two, 10.0),
            explained_slow=np.full(two, 6.0),
        )
        assert path.is_file()
    return root, readout


def test_read_gives_delta_only_when_fast_content_is_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 1.0, 4: 1.0})
        record = driver.read(root, readout)
        assert all(record["gates"].values()), record["gates"]
        assert record["conditions"]["R1"] == {"T": True, "F1": False, "F2": False, "F3": False}
        assert record["outcome"] == "delta_only"
        descriptive = record["descriptive"]
        assert all(g == {"R1_vs_R4": 0.0, "R1_vs_pass1": 0.0} for g in descriptive["random_start_gaps"].values())
        assert abs(descriptive["edge_over_interior_energy_per_second"] - 1.1) < 1e-12
        assert abs(descriptive["budget"]["B4-trained"]["share_of_winning_atoms_that_are_slow"] - 0.5) < 1e-12
        summary = pl.read_csv(root / "band_summary.csv")
        assert summary.height == len(core.ARMS) * len(driver.COLUMNS)
        assert pl.read_csv(root / "band_pca_curve.csv")["k"].to_list() == list(base.PCA_KS)
        assert json.loads((root / "band_decision.json").read_text())["outcome"] == "delta_only"


def test_read_gives_sweep_through_either_arm_and_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 1.0, 4: 0.8})
        record = driver.read(root, readout)
        assert record["outcome"] == "sweep" and all(record["conditions"]["R4"].values())
        assert not record["conditions"]["R1"]["F1"]
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 0.85, 4: 1.0})
        assert driver.read(root, readout)["outcome"] == "sweep"
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 1.0, 4: 1.0}, pca1_delta=0.3)
        record = driver.read(root, readout)
        assert record["outcome"] == "stop" and not record["conditions"]["R1"]["T"]


def test_read_is_blocked_by_its_gates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 1.0, 4: 0.8}, slow_energy=30.0)
        record = driver.read(root, readout)
        assert record["outcome"] == "blocked" and not record["gates"]["band_pass_ran"], record["gates"]
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 1.0, 4: 0.8}, offset=2 * base.ROW_MATCH_TOLERANCE)
        record = driver.read(root, readout)
        assert record["outcome"] == "blocked" and not record["gates"]["rows_match_training"]
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write(Path(tmp), {1: 1.0, 4: 0.8})
        driver.score_path(root, 2).unlink()
        try:
            driver.read(root, readout)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing fold was accepted")


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
