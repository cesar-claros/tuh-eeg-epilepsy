"""Checks of the stages of ``src/band_code_readout.py`` on written feature files (no corpus; CPU; under a minute).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure. Covered:
the root checks, the dictionary folders, the arms and their scalers, the HYDRA scaler, the refusal of the heads before
the gates pass, gates that block on counts that do not add up and on an explained-energy gap, the permutation null, and
the gates, heads and read end to end in a ``code_adds`` case (a code that separates the classes inside each drug group,
untrained atoms and band power without signal) and a ``no_advantage`` case (the same noise in every arm), with the files
the read writes.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import src.band_code_readout as driver  # noqa: E402

N_PATIENTS, N_SEGMENTS = 100, 2
K, C, D, L = 4, 4, 2, 16


def _expect(error: type[Exception], function, message: str) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError(message)


@contextmanager
def _loose_permutation_gate() -> Iterator[None]:
    """A wide permutation margin: with 100 synthetic patients the null of 20 draws is too noisy for 0.03."""
    saved = driver.PERMUTATION_MARGIN
    driver.PERMUTATION_MARGIN = 0.2
    try:
        yield
    finally:
        driver.PERMUTATION_MARGIN = saved


def _cohort(tmp: Path) -> tuple[dict, Path, Path]:
    """A plan of 100 patients in 5 folds, its readout root (plan, patients, saved scores) and its audit root."""
    rng = np.random.default_rng(0)
    subjects = [f"p{i:03d}" for i in range(N_PATIENTS)]
    labels = {s: int(i % 2 == 0) for i, s in enumerate(subjects)}
    folds = {s: (i // 2) % 5 for i, s in enumerate(subjects)}
    plan = {"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds}
    readout, audit = tmp / "readout", tmp / "audit"
    (readout / "scores").mkdir(parents=True)
    audit.mkdir()
    (readout / "folds.json").write_text(json.dumps(plan))
    pl.DataFrame(
        {
            "subject": subjects,
            "label": [labels[s] for s in subjects],
            "fold": [folds[s] for s in subjects],
            "segments": rng.integers(2, 60, N_PATIENTS).tolist(),
        }
    ).write_csv(readout / "patients.csv")
    for name in driver.SAVED:
        pl.DataFrame(
            {
                "arm": name,
                "subject": subjects,
                "fold": [folds[s] for s in subjects],
                "label": [labels[s] for s in subjects],
                "score": rng.standard_normal(N_PATIENTS).tolist(),
            }
        ).write_csv(readout / "scores" / f"{name}.csv")
    on = [rng.random() < (0.8 if labels[s] else 0.1) for s in subjects]
    status = ["unknown" if i % 12 == 5 else ("on" if o else "off") for i, o in enumerate(on)]
    pl.DataFrame({"subject": subjects, "status": status}).write_csv(audit / "drug_exposure.csv")
    pl.DataFrame(
        {
            "subject": subjects,
            "age": rng.uniform(20, 70, N_PATIENTS).round(1).tolist(),
            "sex": ["F" if i % 3 else "M" for i in range(N_PATIENTS)],
        }
    ).write_csv(audit / "covariates.csv")
    return plan, readout, audit


def _sets(rng: np.random.Generator, n: int, signal: np.ndarray | None) -> dict[str, np.ndarray]:
    """The code feature sets of ``n`` segments; ``signal`` (per segment) enters the first hard count."""
    hard, soft = rng.standard_normal((n, K)), rng.standard_normal((n, K))
    if signal is not None:
        hard[:, 0] += signal
    won = rng.integers(0, 4 * 60, n)
    code = np.concatenate([hard, soft], axis=1)
    return {
        "code": code,
        "counts": np.maximum(np.expm1(code), 0.0),  # zero-inflated, as the rare atoms are
        "templates": rng.standard_normal((n, 2 * K * D)),
        "hard": hard,
        "soft": soft,
        "tails": rng.standard_normal((n, 2 * K)),
        "channels": rng.standard_normal((n, K * C)),
        "asymmetry": rng.uniform(0, 1, (n, K)),
        "won_total": won,
        "positive_total": won.copy(),
        "slots": np.full(n, 4 * 60),
    }


def _write_features(root: Path, plan: dict, case: str, ev_gap: float = 0.0, broken_counts: bool = False) -> None:
    """Feature files of every dictionary kind and fold, and the band file, for ``case`` ``code_adds`` or ``same``."""
    rng = np.random.default_rng(1)
    subjects = np.repeat(sorted(plan["labels"]), N_SEGMENTS)
    label = np.array([plan["labels"][s] for s in subjects])
    latent = {s: 2.5 * plan["labels"][s] + 0.5 * rng.standard_normal() for s in sorted(plan["labels"])}
    signal = np.array([latent[s] for s in subjects]) + 0.1 * rng.standard_normal(len(subjects))
    same = _sets(rng, len(subjects), None)
    for kind in driver.DICTIONARIES:
        for fold in range(driver.N_FOLDS):
            if case == "same":
                sets = same
            else:
                sets = _sets(rng, len(subjects), signal if kind in ("trained", "seed2") else None)
            if broken_counts and kind == "trained" and fold == 2:
                sets = dict(sets, positive_total=sets["won_total"] + 1)
            driver._save_npz(
                driver.feature_file(root, kind, fold),
                subject=driver._text_array(subjects),
                label=label,
                fold=np.array(fold),
                kind=np.array(kind),
                setting=np.array(driver.DICTIONARIES[kind].name),
                waveforms=rng.standard_normal((K, L)),
                patterns=np.linalg.qr(rng.standard_normal((K, C, D)))[0],
                dominant_hz=rng.uniform(1, 30, K),
                repeat_gap=np.array(0.0),
                repeat_segments=np.array(24),
                ev_gap=np.array(np.nan if kind == "untrained" else ev_gap),
                load_failures=np.array(0),
                residual=rng.uniform(0.5, 1.0, len(subjects)),
                signal=np.ones(len(subjects)),
                **sets,
            )
    band = same["code"] if case == "same" else rng.standard_normal((len(subjects), C * 8))
    driver._save_npz(
        driver.feature_file(root, "band"),
        subject=driver._text_array(subjects),
        label=label,
        band_scaled=band,
        band_raw=band if case == "same" else rng.standard_normal((len(subjects), C * 8)),
        level=rng.standard_normal((len(subjects), C)),
    )


def test_root_checks_and_folders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, sweep, readout = Path(tmp) / "root", Path(tmp) / "sweep", Path(tmp) / "readout"
        root.mkdir()
        driver.check_root(root, sweep, readout)
        driver.check_root(root, sweep, readout)
        assert {"domain.json", "readout_roots.json"} <= {p.name for p in root.iterdir()}
        _expect(ValueError, lambda: driver.check_root(root, Path(tmp) / "other", readout), "other roots were accepted")
        for name in ("band_decision.json", "sweep.json", "selection.json"):
            other = Path(tmp) / name.split(".")[0]
            other.mkdir()
            (other / name).write_text("{}")
            _expect(ValueError, lambda other=other: driver.check_root(other, sweep, readout), f"a root with {name}")
        assert driver.dictionary_folder(root, sweep, "untrained", 3) == root / "dictionaries" / "untrained_f3"
        trained = driver.dictionary_folder(root, sweep, "trained", 3)
        assert trained == sweep / "dictionaries" / "K64_L128_D8_P512_R4_E30_s1_f3"
        assert driver.dictionary_folder(root, sweep, "seed2", 0).name == "K64_L128_D8_P512_R4_E30_s2_f0"
        assert driver.feature_file(root, "band") == root / "features" / "band.npz"


def test_arms_cover_the_protocol() -> None:
    assert list(driver.ARMS)[:3] == ["Code", "Code-untrained", "Band-scaled"], "the deciding arms come first"
    assert len(driver.ARMS) == 13 and set(driver.MATCHED) <= set(driver.ARMS)
    assert driver.ARMS["Code-hydra"] == ("trained", ("counts",)) and driver.SPARSE_ARMS == ("Code-hydra",)
    assert driver.scaler_for("Code-hydra") is driver.HydraRows
    assert all(driver.scaler_for(arm) is driver.StandardScaler for arm in driver.ARMS if arm != "Code-hydra")


def test_hydra_scaler_on_zero_inflated_rows() -> None:
    rng = np.random.default_rng(3)
    x = rng.exponential(2.0, (400, 3)) * (rng.random((400, 3)) < np.array([0.9, 0.5, 0.05]))
    x[:5, 0] = -1.0  # a negative value is clipped to 0
    scaled = driver.HydraRows().fit(x).transform(x)
    root = np.sqrt(np.clip(x, 0, None))
    zero = (root == 0).mean(axis=0)
    expected = (root - root.mean(axis=0)) * (root != 0) / (root.std(axis=0, ddof=1) + zero**4 + 1e-8)
    assert np.allclose(scaled, expected, atol=1e-6), "the HYDRA scaler: square root, zeros kept, q**4 damping"
    assert (scaled[root == 0] == 0).all(), "zeros stay at 0"
    rare = np.abs(scaled[:, 2][root[:, 2] > 0]).max()
    standard = np.abs(driver.StandardScaler().fit_transform(x)[:, 2]).max()
    assert rare < standard / 2, "a mostly-zero feature is damped below its StandardScaler value"
    kinds = {kind for kind, _ in driver.ARMS.values()}
    assert kinds == {*driver.DICTIONARIES, "band"}


def test_heads_refuse_before_the_gates_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, _, _ = _cohort(Path(tmp))
        root = Path(tmp) / "root"
        root.mkdir()
        _expect(ValueError, lambda: driver.heads(root, plan), "heads ran without gates")
        (root / "gates.json").write_text(json.dumps({"status": "blocked"}))
        _expect(ValueError, lambda: driver.heads(root, plan), "heads ran after blocked gates")
        assert not (root / "scores").exists()


def test_gates_block_on_counts_and_on_the_energy_gap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, _, _ = _cohort(Path(tmp))
        root = Path(tmp) / "root"
        _write_features(root, plan, "code_adds", broken_counts=True)
        with _loose_permutation_gate():
            record = driver.gates(root, plan)
        assert record["status"] == "blocked" and not record["gates"]["counts_add_up"]
        assert record["gates"]["explained_energy_matches_sweep"] and record["gates"]["complete"]
        assert record["permuted_mean"] < 0.65, "the null must not see the labels"
    with tempfile.TemporaryDirectory() as tmp:
        plan, _, _ = _cohort(Path(tmp))
        root = Path(tmp) / "root"
        _write_features(root, plan, "code_adds", ev_gap=1e-4)
        with _loose_permutation_gate():
            record = driver.gates(root, plan)
        assert record["status"] == "blocked" and not record["gates"]["explained_energy_matches_sweep"]


def test_end_to_end_code_adds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, readout, audit = _cohort(Path(tmp))
        root = Path(tmp) / "root"
        _write_features(root, plan, "code_adds")
        with _loose_permutation_gate():
            assert driver.gates(root, plan)["status"] == "pass"
        driver.heads(root, plan)
        assert {f"{arm}.csv" for arm in driver.ARMS} == {p.name for p in (root / "scores").iterdir()}
        record = driver.read(root, readout, audit, n_boot=200)
        assert record["outcome"] == "code_adds", record["conditions"]
        assert record["conditions"] == {"L": True, "B": True, "G": True}
        assert record["patients"] == N_PATIENTS and record["known_drug_status"] == N_PATIENTS - 8
        summary = pl.read_csv(root / "band_code_summary.csv")
        assert summary.height == len(driver.ARMS) + len(driver.SAVED)
        code = summary.filter(pl.col("arm") == "Code").row(0, named=True)
        assert code["auroc"] > 0.9 and code["gain"] > 0
        assert set(record["descriptive"]["matched"]) == set(driver.MATCHED)
        assert record["descriptive"]["seed_rank_correlation"] > 0.5
        assert set(record["descriptive"]["scaler_check_Code_hydra_minus_Code"]) == {"value", "low", "high"}
        top = record["descriptive"]["top_atoms"]
        assert len(top["per_fold"]) == driver.N_FOLDS and set(top["recurring_in_every_fold"]) == set(range(5))
        assert (root / "band_code_decision.json").is_file()
        rerun = driver.read(root, readout, audit, n_boot=200)
        assert rerun["deciding"] == record["deciding"], "the read must repeat"


def test_end_to_end_no_advantage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, readout, audit = _cohort(Path(tmp))
        root = Path(tmp) / "root"
        _write_features(root, plan, "same")
        with _loose_permutation_gate():
            assert driver.gates(root, plan)["status"] == "pass"
        driver.heads(root, plan)
        record = driver.read(root, readout, audit, n_boot=200)
        learning = record["deciding"]["L_Code_minus_Code_untrained"]
        assert learning == {"value": 0.0, "low": 0.0, "high": 0.0}, "identical features must give identical scores"
        assert record["outcome"] == "no_advantage", record["conditions"]


def test_read_is_blocked_by_failed_gates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        plan, readout, audit = _cohort(Path(tmp))
        root = Path(tmp) / "root"
        _write_features(root, plan, "code_adds")
        with _loose_permutation_gate():
            driver.gates(root, plan)
        driver.heads(root, plan)
        record = json.loads((root / "gates.json").read_text())
        (root / "gates.json").write_text(json.dumps(dict(record, status="blocked")))
        assert driver.read(root, readout, audit, n_boot=200)["outcome"] == "blocked"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
