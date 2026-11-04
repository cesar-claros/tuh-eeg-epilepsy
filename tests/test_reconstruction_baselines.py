"""Checks of the reconstruction baselines driver ``src/reconstruction_baselines.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure.
Covered: the geometry check, the JSON and the atomic file writes, and ``read`` end to end on written fold files:
each outcome of the rule, and the gates that block it (an incomplete read, a ceiling violation, rows that do not
match training).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

import src.reconstruction_baselines as driver  # noqa: E402
import src.reconstruction_baselines_core as core  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402

N_SEGMENTS = 2


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _residual_factors(fast_trained: float) -> np.ndarray:
    """Residual over signal ``(n_arms, n_bands)``: S4-trained explains 25% of the slow bands."""
    factors = np.ones((len(core.ARMS), len(core.BANDS)))
    slow = [i for i in range(len(core.BANDS)) if i not in core.fast_bands()]
    factors[core.ARMS.index("S4-trained"), slow] = 0.70
    factors[core.ARMS.index("S4-trained"), core.fast_bands()] = fast_trained
    factors[core.ARMS.index("PCA-1"), slow] = 0.95
    for k in core.PCA_KS[1:]:
        factors[core.ARMS.index(f"PCA-{k}"), slow] = 0.95 - 0.1 * np.log2(k)
    factors[core.ARMS.index("SVD-1")] = 0.5
    return factors


def _write_read(
    root: Path, fast_trained: float, *, drop: str | None = None, violations: int = 0, validation_offset: float = 0.0
) -> tuple[Path, Path]:
    """A readout root (plan and training curves) and five scored folds whose energies follow ``_residual_factors``.

    The best validation EV of each fold's curve is the fold's held-out EV of S4-trained plus ``validation_offset``.
    """
    readout = root / "readout"
    labels = {f"p{i:03d}": int(i % 2 == 0) for i in range(20)}
    folds = {s: i % 5 for i, s in enumerate(sorted(labels))}
    readout.mkdir(parents=True)
    (readout / "folds.json").write_text(
        json.dumps({"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds})
    )
    rng = np.random.default_rng(0)
    factors = _residual_factors(fast_trained)
    trained = core.ARMS.index("S4-trained")
    for fold in range(5):
        subjects = [s for s in sorted(labels) if folds[s] == fold and s != drop]
        subject = np.repeat(subjects, N_SEGMENTS)
        signal_bands = rng.uniform(1.0, 2.0, (len(subject), len(core.BANDS)))
        residual_bands = factors[:, None, :] * signal_bands[None]
        held_out = 1.0 - residual_bands[trained].sum() / signal_bands.sum()
        curve = readout / "dictionaries" / f"trained_f{fold}"
        curve.mkdir(parents=True)
        best = 1.0 - held_out - validation_offset
        pl.DataFrame({"epoch": [1, 2], "val_residual_frac": [0.99, best]}).write_csv(curve / "sae_training.csv")
        driver._save_npz(
            driver.score_path(root, fold),
            subject=subject,
            fold=np.array(fold),
            arms=np.array(core.ARMS),
            bands=np.array(core.BAND_NAMES),
            n_channels=np.array(20),
            windows_per_segment=np.array(60),
            ceiling_violations=np.array([violations if fold == 0 else 0, 0, 0]),
            ceiling_excess=np.zeros(3),
            atom_wins=np.arange(core.N_ATOMS),
            dominant_hz=np.linspace(0.5, 20.0, core.N_ATOMS),
            signal_bands=signal_bands,
            signal_total=signal_bands.sum(axis=1),
            residual_bands=residual_bands,
            residual_total=residual_bands.sum(axis=2),
            usable_rows=np.full(len(subject), 20),
            windows_won=np.full(len(subject), 60),
            windows_slow=np.full(len(subject), 54),
            explained_all=np.full(len(subject), 10.0),
            explained_slow=np.full(len(subject), 9.0),
        )
    return root, readout


def test_geometry_check_refuses_other_settings() -> None:
    spec = SeparableSpec(n_atoms=core.N_ATOMS, atom_len=core.ATOM_LEN, pool_len=core.POOL_LEN, n_projections=4)
    driver._check_geometry(SeparableSAE(spec=spec, device="cpu"))
    for change in ({"n_atoms": 32}, {"pool_len": 256}, {"n_projections": 2}, {"top_r": 2}, {"pre_emphasis": "diff"}):
        other = SeparableSpec(**{**spec.__dict__, **change})
        _expect_value_error(
            lambda other=other: driver._check_geometry(SeparableSAE(spec=other, device="cpu")), str(change)
        )


def test_json_and_atomic_writes() -> None:
    assert driver._plain({"a": np.float64(float("nan")), "b": [np.int64(3), (1.5,)]}) == {"a": None, "b": [3, [1.5]]}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scores" / "fold0.npz"
        driver._save_npz(path, values=np.arange(3))
        assert path.is_file() and not list(path.parent.glob("*.partial")), "a partial file was left behind"
        with np.load(path) as data:
            assert data["values"].tolist() == [0, 1, 2]
        column = pd.DataFrame({"subject": ["p001", "p002"]})["subject"].astype(str)
        assert column.to_numpy().dtype == object, "the case this guards against: a pandas text column"
        other = Path(tmp) / "scores" / "fold1.npz"
        _expect_value_error(lambda: driver._save_npz(other, subject=column.to_numpy()), "an object array was saved")
        assert not other.exists() and not list(other.parent.glob("*.partial")), "a refused write left a file"
        driver._save_npz(other, subject=driver._text_array(column))
        with np.load(other) as data:
            assert data["subject"].tolist() == ["p001", "p002"]


def test_read_gives_slow_only_when_fast_content_is_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write_read(Path(tmp), fast_trained=1.0)
        record = driver.read(root, readout)
        assert all(record["gates"].values()), record["gates"]
        assert record["conditions"] == {"T": True, "H1": False, "H2": False, "H3": False}
        assert record["outcome"] == "slow_only"
        assert abs(record["deciding"]["H1_above_4_S4_trained_ev"]) < 1e-12
        summary = pl.read_csv(root / "reconstruction_summary.csv")
        assert summary.height == len(core.ARMS) * (len(core.BANDS) + 2)
        curve = pl.read_csv(root / "pca_curve.csv")
        assert curve["k"].to_list() == list(core.PCA_KS) and (curve["total"].diff().drop_nulls() > 0).all()
        budget = record["descriptive"]["budget"]
        assert abs(budget["share_of_won_windows_won_by_slow_atoms"] - 0.9) < 1e-12
        assert record["counts"]["patients"] == 20 and record["counts"]["rejected_rows"] == 0
        assert json.loads((root / "reconstruction_decision.json").read_text())["outcome"] == "slow_only"


def test_read_gives_sweep_and_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write_read(Path(tmp), fast_trained=0.8)
        record = driver.read(root, readout)
        assert record["outcome"] == "sweep", record["conditions"]
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write_read(Path(tmp), fast_trained=1.0)
        for fold in range(5):  # PCA-1 now explains more than S4-trained in total
            path = driver.score_path(root, fold)
            with np.load(path) as stored:
                data = {key: stored[key] for key in stored.files}
            i, j = core.ARMS.index("PCA-1"), core.ARMS.index("S4-trained")
            data["residual_bands"][i] = 0.5 * data["residual_bands"][j]
            data["residual_total"][i] = data["residual_bands"][i].sum(axis=1)
            driver._save_npz(path, **data)
        record = driver.read(root, readout)
        assert record["outcome"] == "stop" and not record["conditions"]["T"]


def test_read_is_blocked_by_its_gates() -> None:
    cases = (
        ({"drop": "p003"}, "complete"),
        ({"violations": 1}, "ceiling"),
        ({"validation_offset": 2 * core.ROW_MATCH_TOLERANCE}, "rows_match_training"),
    )
    for options, gate in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root, readout = _write_read(Path(tmp), fast_trained=0.8, **options)
            record = driver.read(root, readout)
            assert record["outcome"] == "blocked" and not record["gates"][gate], (gate, record["gates"])
    with tempfile.TemporaryDirectory() as tmp:
        root, readout = _write_read(Path(tmp), fast_trained=0.8)
        driver.score_path(root, 4).unlink()
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
