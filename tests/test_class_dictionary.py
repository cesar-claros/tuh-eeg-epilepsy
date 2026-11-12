"""Checks of the stages of ``src/class_dictionary.py`` on written files (no corpus; CPU; under a minute).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure. Covered:
the root checks, the settings of the dictionaries, the split stage and its refusal of a changed split, gates that
block on a training curve that did not move and on a rows gap, and the gates and read end to end in a ``score_adds``
case (class dictionaries that fit their own class better, halves and band power without signal), a ``no_advantage``
case (no signal anywhere) and an inverted case (the orientation is fixed, so a reversed score does not pass), with the
files the read writes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import src.class_dictionary as driver  # noqa: E402
import src.class_dictionary_core as core  # noqa: E402

N_PATIENTS, N_SEGMENTS = 100, 2
K, L, C, D = 4, 16, 4, 2


def _expect(error: type[Exception], function, message: str) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError(message)


def _roots(tmp: Path) -> dict[str, Path]:
    """A plan of 100 patients in 5 folds with its readout root, audit root and band-code root (saved `Band-scaled`
    scores and the adopted dictionaries' residuals)."""
    rng = np.random.default_rng(0)
    subjects = [f"p{i:03d}" for i in range(N_PATIENTS)]
    labels = {s: int(i % 2 == 0) for i, s in enumerate(subjects)}
    folds = {s: (i // 2) % 5 for i, s in enumerate(subjects)}
    plan = {"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds}
    roots = {name: tmp / name for name in ("readout", "audit", "band_code", "sweep", "root")}
    for name in ("readout", "audit", "sweep"):
        roots[name].mkdir()
    (roots["band_code"] / "scores").mkdir(parents=True)
    (roots["band_code"] / "features").mkdir()
    (roots["readout"] / "folds.json").write_text(json.dumps(plan))
    pl.DataFrame(
        {"subject": subjects, "label": [labels[s] for s in subjects], "fold": [folds[s] for s in subjects]}
        | {"segments": rng.integers(2, 60, N_PATIENTS).tolist()}
    ).write_csv(roots["readout"] / "patients.csv")
    on = [rng.random() < (0.8 if labels[s] else 0.1) for s in subjects]
    status = ["unknown" if i % 12 == 5 else ("on" if o else "off") for i, o in enumerate(on)]
    pl.DataFrame({"subject": subjects, "status": status}).write_csv(roots["audit"] / "drug_exposure.csv")
    pl.DataFrame(
        {
            "subject": subjects,
            "age": rng.uniform(20, 70, N_PATIENTS).round(1).tolist(),
            "sex": ["F" if i % 3 else "M" for i in range(N_PATIENTS)],
        }
    ).write_csv(roots["audit"] / "covariates.csv")
    pl.DataFrame(
        {
            "arm": "Band-scaled",
            "subject": subjects,
            "fold": [folds[s] for s in subjects],
            "label": [labels[s] for s in subjects],
            "score": rng.standard_normal(N_PATIENTS).tolist(),
        }
    ).write_csv(roots["band_code"] / "scores" / "Band-scaled.csv")
    every = np.repeat(subjects, N_SEGMENTS)
    for fold in range(5):
        for kind in ("trained", "seed2"):
            signal = rng.uniform(1.0, 2.0, len(every))
            driver._save_npz(
                roots["band_code"] / "features" / f"{kind}_f{fold}.npz",
                subject=driver._text_array(every),
                residual=signal * rng.uniform(0.75, 0.8, len(every)),
                signal=signal,
            )
    return roots | {"plan": plan}  # type: ignore[operator]


def _write_fold_files(roots: dict, case: str, *, rows_gap: float = 0.0, stalled: str | None = None) -> None:
    """The splits, the dictionary folders with their curves, and an energy file per fold for ``case``: ``adds``
    (class dictionaries fit their own class better), ``null`` (no signal) or ``inverted`` (the reverse)."""
    rng = np.random.default_rng(1)
    plan, root = roots["plan"], roots["root"]
    driver.split(root, roots["sweep"], roots["readout"])
    names = core.dictionary_names()
    shared = (rng.standard_normal((K, L)), np.linalg.qr(rng.standard_normal((K, C, D)))[0])
    for fold in range(5):
        for name in names:
            folder = driver.dictionary_dir(root, name, fold)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sae_state.pt").write_bytes(b"")
            curve = [0.9, 0.9] if name == stalled else [0.9, 0.8]
            pl.DataFrame({"epoch": [1, 2], "val_residual_frac": curve}).write_csv(folder / "sae_training.csv")
        held = sorted(s for s, f in plan["folds"].items() if f == fold)
        subject = np.repeat(held, N_SEGMENTS)
        label = np.array([plan["labels"][s] for s in subject])
        signal = rng.uniform(1.0, 2.0, len(subject))
        bands = np.stack([signal * w for w in (0.0, 0.4, 0.3, 0.2, 0.1, 0.0)], axis=1)
        effect = {"adds": 0.03, "null": 0.0, "inverted": -0.03}[case]
        residual, residual_bands, waveforms, patterns = [], [], [], []
        for name in names:
            arm, _ = core.parse_name(name)
            share = 0.78 + 0.01 * rng.standard_normal(len(subject))
            if arm == "epi":
                share -= effect * (label - 0.5)
            elif arm == "non":
                share += effect * (label - 0.5)
            residual.append(share * signal)
            residual_bands.append(bands * share[:, None])
            fresh = (rng.standard_normal((K, L)), np.linalg.qr(rng.standard_normal((K, C, D)))[0])
            pair = shared if arm in ("epi", "non") else fresh
            waveforms.append(pair[0])
            patterns.append(pair[1])
        driver._save_npz(
            driver.energy_file(root, fold),
            subject=driver._text_array(subject),
            label=label,
            fold=np.array(fold),
            names=np.array(names),
            residual_total=np.stack(residual),
            residual_bands=np.stack(residual_bands),
            atom_wins=np.tile(np.array([3, 2, 1, 0]), (len(names), 1)),
            waveforms=np.stack(waveforms),
            patterns=np.stack(patterns),
            signal_total=signal,
            signal_bands=bands,
            rows_gap=np.array(rows_gap),
            repeat_gap=np.array(0.0),
            repeat_segments=np.array(24),
            load_failures=np.array(0),
        )


def _read(roots: dict) -> dict:
    return driver.read(roots["root"], roots["readout"], roots["audit"], roots["band_code"], n_boot=200)


def test_root_checks_and_settings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, sweep, readout = Path(tmp) / "root", Path(tmp) / "sweep", Path(tmp) / "readout"
        root.mkdir()
        driver.check_root(root, sweep, readout)
        driver.check_root(root, sweep, readout)
        _expect(ValueError, lambda: driver.check_root(root, Path(tmp) / "other", readout), "other roots were accepted")
        other = Path(tmp) / "band_code"
        other.mkdir()
        (other / "band_code_decision.json").write_text("{}")
        _expect(ValueError, lambda: driver.check_root(other, sweep, readout), "a band-code root was accepted")
    trained, start = driver.setting("epi_s1"), driver.setting("non-start_s2")
    assert (trained.K, trained.L, trained.D, trained.P, trained.R) == (64, 128, 8, 512, 4)
    assert trained.epochs == 56 and trained.seed == 1 and start.epochs == 1 and start.seed == 2
    assert driver.dictionary_dir(Path("r"), "A_s2", 3) == Path("r") / "dictionaries" / "A_s2_f3"


def test_split_stage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        records = driver.split(roots["root"], roots["sweep"], roots["readout"])
        assert len(records) == 5 and all(all(r["checks"].values()) for r in records)
        assert driver.split(roots["root"], roots["sweep"], roots["readout"]) == records, "a rerun is the same"
        path = driver.split_file(roots["root"], 2)
        stored = json.loads(path.read_text())
        stored["arms"]["A"]["train"] = stored["arms"]["A"]["train"][1:]
        path.write_text(json.dumps(stored))
        _expect(ValueError, lambda: driver.split(roots["root"], roots["sweep"], roots["readout"]), "a changed split")


def test_gates_block() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "adds", stalled="B_s2")
        record = driver.gates(roots["root"], roots["plan"])
        assert record["status"] == "blocked" and not record["gates"]["training_moved"]
        assert record["not_moved"] == [f"B_s2_f{f}" for f in range(5)] and record["gates"]["splits"]
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "adds", rows_gap=1e-6)
        record = driver.gates(roots["root"], roots["plan"])
        assert record["status"] == "blocked" and not record["gates"]["rows_match_sweep"]
        assert _read(roots)["outcome"] == "blocked"


def test_end_to_end_score_adds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "adds")
        assert driver.gates(roots["root"], roots["plan"])["status"] == "pass"
        record = _read(roots)
        assert record["outcome"] == "score_adds", record["conditions"]
        assert record["deciding"]["S_Score_auroc"]["value"] > 0.9
        assert record["patients"] == N_PATIENTS and record["known_drug_status"] == N_PATIENTS - 8
        summary = pl.read_csv(roots["root"] / "class_dictionary_summary.csv")
        expected = {"Score", "Score-halves", "Score-start", "Score-s1", "Score-s2", "Seed-null", "Band-scaled"}
        assert expected | {f"Score-{b}" for b in core.BAND_COLUMNS} == set(summary["arm"].to_list())
        halves = summary.filter(pl.col("arm") == "Score-halves").row(0, named=True)
        assert 0.3 < halves["auroc"] < 0.7, "the halves carry no signal"
        recurrence = record["descriptive"]["recurrence"]
        assert recurrence["mean_epi_in_non"] == 1.0 and recurrence["mean_A_in_B"] < 0.5
        assert record["descriptive"]["dead_atom_share"]["epi_s1"] == 0.25
        assert record["descriptive"]["seed_rank_correlation_within_fold"] > 0.5
        assert (roots["root"] / "class_dictionary_decision.json").is_file()
        assert _read(roots)["deciding"] == record["deciding"], "the read must repeat"


def test_end_to_end_null_and_inverted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "null")
        assert driver.gates(roots["root"], roots["plan"])["status"] == "pass"
        record = _read(roots)
        assert record["outcome"] == "no_advantage", record["conditions"]
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "inverted")
        driver.gates(roots["root"], roots["plan"])
        record = _read(roots)
        assert record["deciding"]["S_Score_auroc"]["value"] < 0.2, "the orientation is fixed: no sign is chosen"
        assert not record["conditions"]["S"] and record["outcome"] == "no_advantage"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
