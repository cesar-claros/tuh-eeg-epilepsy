"""Checks of the frozen encoder check driver ``src/encoder_check.py`` (no corpus; no weights; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first
failure. Covered: the cohort files, step 0 on written features, the gates on written features, the refusal
of a read before the gates pass, and the rule end to end: an encoder with its own signal differs from one
that only copies the drug, and noise reads ``no_advantage``.
"""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import src.encoder_check as driver  # noqa: E402
from src.frozen_readout_core import assign_folds  # noqa: E402

N_PATIENTS, N_SEGMENTS, N_BOOT, N_SPLITS = 156, 8, 200, 5
SEED = 11


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _subject(i: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "aaaaa" + letters[i // 676 % 26] + letters[i // 26 % 26] + letters[i % 26]


def _fixture(base: Path) -> dict:
    """Written readout and audit roots, with the drug split of the report."""
    rng = np.random.default_rng(SEED)
    subjects = [_subject(i) for i in range(N_PATIENTS)]
    labels = {s: int(i < 79) for i, s in enumerate(subjects)}
    folds = assign_folds(labels, SEED)
    readout, audit = base / "readout", base / "audit"
    (readout / "scores").mkdir(parents=True)
    (readout / "features").mkdir()
    audit.mkdir()
    (readout / "folds.json").write_text(
        json.dumps({"seed": SEED, "n_segments": N_SEGMENTS, "folds": folds, "labels": labels})
    )
    y = np.array([labels[s] for s in subjects])
    fold = np.array([folds[s] for s in subjects])
    status = np.array(["on"] * 66 + ["off"] * 10 + ["unknown"] * 3 + ["on"] * 4 + ["off"] * 58 + ["unknown"] * 15)
    age = rng.normal(55, 15, N_PATIENTS)
    pl.DataFrame(
        {
            "subject": subjects,
            "label": y.tolist(),
            "fold": fold.tolist(),
            "segments": rng.integers(8, 60, N_PATIENTS).tolist(),
            "recordings": [2] * N_PATIENTS,
            "native_rate": ([250.0, 256.0] * N_PATIENTS)[:N_PATIENTS],
        }
    ).write_csv(readout / "patients.csv")
    pl.DataFrame({"subject": subjects, "status": status.tolist()}).write_csv(audit / "drug_exposure.csv")
    pl.DataFrame({"subject": subjects, "age": age.tolist()}).write_csv(audit / "covariates.csv")
    rows = np.repeat(np.arange(N_PATIENTS), N_SEGMENTS)
    names = np.array(subjects)[rows]
    spectral = rng.standard_normal((len(rows), 200)).astype(np.float32)
    spectral[:, 0] += 0.3 * y[rows] + (age[rows] - 55) / 15
    spectral[:, 1] += 0.5 * y[rows]
    np.savez(readout / "features" / "Spectral.npz", subject=names, label=y[rows], features=spectral)
    np.savez(readout / "features" / "Spectral-all.npz", subject=names, label=y[rows], features=spectral)
    hydra = rng.standard_normal((N_PATIENTS, 300)).astype(np.float32)
    hydra[:, 0] += 0.7 * y
    np.savez(readout / "features" / "HYDRA-all.npz", subject=np.array(subjects), label=y, features=hydra)
    score = 0.7 * y + rng.standard_normal(N_PATIENTS)
    for name in ("Spectral", "Spectral-all", "HYDRA-all"):
        pl.DataFrame(
            {"subject": subjects, "fold": fold.tolist(), "label": y.tolist(), "score": score.tolist()}
        ).write_csv(readout / "scores" / f"{name}.csv")
    return {
        "subjects": subjects,
        "y": y,
        "fold": fold,
        "status": status,
        "age": age,
        "readout": readout,
        "audit": audit,
    }


def _write_encoder(root: Path, fix: dict, per_patient: np.ndarray) -> None:
    """Written encoder features: ``per_patient`` is one signal per patient on the first feature, and noise elsewhere."""
    rng = np.random.default_rng(SEED + 1)
    (root / "features").mkdir(parents=True)
    y, subjects = fix["y"], np.array(fix["subjects"])

    def rows(index: np.ndarray, width: int = 32) -> np.ndarray:
        x = rng.standard_normal((len(index), width)).astype(np.float32)
        x[:, 0] += per_patient[index].astype(np.float32)
        x[:, 1] += ((fix["age"][index] - 55) / 15).astype(np.float32)
        return x

    every = np.repeat(np.arange(N_PATIENTS), N_SEGMENTS + 4)
    fixed = np.tile(np.r_[np.ones(N_SEGMENTS, bool), np.zeros(4, bool)], N_PATIENTS)
    np.savez(
        root / "features" / "REVE-all.npz",
        subject=subjects[every],
        label=y[every],
        features=rows(every),
        fixed=fixed,
        valid=np.ones(len(every), bool),
    )
    probes = np.repeat(np.arange(N_PATIENTS), 6)
    np.savez(root / "features" / "REVE-H.npz", subject=subjects[probes], label=y[probes], features=rows(probes))
    for fold in range(5):
        index = np.repeat(np.arange(N_PATIENTS), N_SEGMENTS)
        np.savez(
            root / "features" / f"REVE-level_f{fold}.npz", subject=subjects[index], label=y[index], features=rows(index)
        )
    extra = np.arange(0, N_PATIENTS, 3)
    flags = np.arange(len(extra)) % 3
    np.savez(
        root / "features" / "REVE-extra.npz",
        subject=subjects[extra],
        label=y[extra],
        features=rows(extra),
        seizure=flags != 1,
        high_rate=flags != 0,
        valid=np.ones(len(extra), bool),
    )
    (root / "repeat.json").write_text(json.dumps({"max_difference": 0.0}))


def _saved_reve(folder: Path, rename: bool = False) -> None:
    """A small REVE saved as the Hub gives it: no ``n_outputs`` in the configuration and no head in the weights."""
    from braindecode.models import REVE
    from safetensors.torch import load_file, save_file

    REVE(n_outputs=2, n_chans=17, n_times=2000, sfreq=200, depth=2).save_pretrained(folder)
    config = json.loads((folder / "config.json").read_text())
    for key in ("n_outputs", "n_chans", "n_times", "sfreq", "chs_info", "input_window_seconds"):
        config.pop(key, None)
    (folder / "config.json").write_text(json.dumps(config))
    weights = load_file(folder / "model.safetensors")
    kept = {
        ("renamed." if rename else "") + k: v.clone() for k, v in weights.items() if not k.startswith("final_layer")
    }
    save_file(kept, folder / "model.safetensors")


def test_the_encoder_loads_on_the_installed_braindecode_and_refuses_empty_weights() -> None:
    """Uses a small model with random weights, not the gated checkpoint; it checks the call and the guard."""
    try:
        import braindecode.models  # noqa: F401
        import safetensors.torch  # noqa: F401
    except ImportError:
        return  # a local run without braindecode
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "good"
        folder.mkdir()
        _saved_reve(folder)
        encode = driver.ReveEncoder("cpu", checkpoint=str(folder))
        windows = np.random.default_rng(0).standard_normal((3, 17, 2000)).astype(np.float32)
        out = encode(windows)
        assert out.shape == (3, 512) and np.isfinite(out).all(), out.shape
        assert np.array_equal(out, encode(windows)), "a repeat must give the same values"
        assert encode(windows[:, :, :920]).shape == (3, 512), "the 5 s windows of read H must run"
        broken = Path(tmp) / "broken"
        broken.mkdir()
        _saved_reve(broken, rename=True)
        _expect_value_error(lambda: driver.ReveEncoder("cpu", checkpoint=str(broken)), "weights that fill nothing")


def test_cohort_reads_the_files_of_both_roots() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fix = _fixture(Path(tmp))
        cohort = driver.load_cohort(fix["readout"], fix["audit"], driver.load_plan(fix["readout"]))
        assert cohort.subject.tolist() == sorted(fix["subjects"]) and (cohort.label == fix["y"]).all()
        assert (cohort.status == "on").sum() == 70 and len(cohort.segments) == N_PATIENTS
        arm = driver.saved_scores(fix["readout"] / "scores" / "Spectral.csv", cohort)
        assert np.array_equal(arm.subject, cohort.subject)
        other = pl.read_csv(fix["readout"] / "scores" / "Spectral.csv", schema_overrides={"subject": pl.Utf8})
        other.head(10).write_csv(fix["readout"] / "scores" / "Spectral.csv")
        _expect_value_error(
            lambda: driver.saved_scores(fix["readout"] / "scores" / "Spectral.csv", cohort),
            "a file with other patients",
        )


def test_step0_writes_a_report_that_decides_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fix = _fixture(Path(tmp))
        report = driver.run_step0(Path(tmp) / "check", fix["readout"], fix["audit"], n_splits=N_SPLITS)
        assert "decides nothing" in report["rule"] and (Path(tmp) / "check" / "step0.json").is_file()
        for name in ("Spectral-all", "HYDRA-all"):
            assert report[name]["linear_splits"]["splits"] == N_SPLITS
            assert len(report[name]["mlp_folds"]) == 3 and len(report[name]["mlp_minus_linear"]) == 3


def test_step0_reads_a_signal_in_correlated_features() -> None:
    """A read of a real signal must not fall to chance; the first run did, through a second scale of rotated rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fix = _fixture(Path(tmp))
        rng = np.random.default_rng(SEED + 5)
        factors = rng.standard_normal((N_PATIENTS, 8))
        factors[:, 0] += 1.5 * fix["y"]
        features = factors @ rng.standard_normal((8, 300)) + 0.3 * rng.standard_normal((N_PATIENTS, 300))
        for name in ("Spectral-all", "HYDRA-all"):
            np.savez(
                fix["readout"] / "features" / f"{name}.npz",
                subject=np.array(fix["subjects"]),
                label=fix["y"],
                features=features.astype(np.float32),
            )
        report = driver.run_step0(Path(tmp) / "check", fix["readout"], fix["audit"], n_splits=40)
        for name in ("Spectral-all", "HYDRA-all"):
            assert report[name]["linear_splits"]["mean"] > 0.7, report[name]
            assert report[name]["mlp_folds"][0] > 0.7, report[name]


def test_gates_pass_or_block_on_written_features() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fix = _fixture(Path(tmp))
        root = Path(tmp) / "check"
        _write_encoder(root, fix, 0.9 * fix["y"])
        report = driver.run_gates(root, fix["readout"], fix["audit"])
        assert report["permuted_ok"] and report["age_ok"] and report["repeat_ok"] and report["load_ok"], report
        assert report["status"] in ("pass", "review"), report
        (root / "repeat.json").write_text(json.dumps({"max_difference": 1.0}))
        assert driver.run_gates(root, fix["readout"], fix["audit"])["status"] == "blocked"


def test_the_read_refuses_to_run_before_the_gates_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fix = _fixture(Path(tmp))
        root = Path(tmp) / "check"
        root.mkdir()
        _expect_value_error(lambda: driver.run_read(root, fix["readout"], fix["audit"]), "no gates at all")
        (root / "gates.json").write_text(json.dumps({"status": "blocked"}))
        _expect_value_error(lambda: driver.run_read(root, fix["readout"], fix["audit"]), "blocked gates")
        (root / "gates.json").write_text(json.dumps({"status": "review"}))
        _expect_value_error(lambda: driver.run_read(root, fix["readout"], fix["audit"]), "a review needs a decision")


def _read(tmp: str, per_patient) -> dict:
    fix = _fixture(Path(tmp))
    root = Path(tmp) / "check"
    _write_encoder(root, fix, per_patient(fix))
    (root / "gates.json").write_text(json.dumps({"status": "pass"}))
    decision = driver.run_read(root, fix["readout"], fix["audit"], n_boot=N_BOOT, n_splits=N_SPLITS)
    summary = pl.read_csv(root / "encoder_summary.csv")
    assert summary["read"].to_list() == ["D", "H", "D-level", "A", "A-mlp", "A-seiz", "A-rates", "C"]
    assert (root / "scores" / "D.csv").is_file() and decision["spread_of_C_over_random_splits"]["splits"] == N_SPLITS
    assert abs(decision["gain_ceiling"] - 0.5 * 844 / 4712) < 1e-12
    return decision


def test_noise_reads_no_advantage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decision = _read(tmp, lambda fix: np.zeros(N_PATIENTS))
        assert decision["outcome"] == "no_advantage", decision


def test_a_copy_of_the_drug_reads_high_and_gains_far_less_than_a_signal_of_its_own() -> None:
    """A copy of the drug beats band power and only breaks ties at random; a real signal nearly reaches the ceiling."""

    def copy_of_drug(fix: dict) -> np.ndarray:
        return 8.0 * (fix["status"] == "on") + 4.0 * fix["y"] * (fix["status"] == "unknown")

    with tempfile.TemporaryDirectory() as tmp:
        copied = _read(tmp, copy_of_drug)
    with tempfile.TemporaryDirectory() as tmp:
        own = _read(tmp, lambda fix: 8.0 * fix["y"])
    assert copied["difference_from_spectral"][1] > 0, copied
    assert copied["outcome"] in ("chart_reader", "encoder_adds"), copied
    assert own["outcome"] == "encoder_adds", own
    assert copied["gain_over_drug"][0] < 0.05 < own["gain_over_drug"][0], (copied, own)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(test):
            test()
    print("all encoder check driver checks passed")
