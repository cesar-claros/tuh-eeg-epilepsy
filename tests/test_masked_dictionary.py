"""Checks of the stages of ``src/masked_dictionary.py`` on written files (no corpus; CPU; under a minute).

Runs under pytest. Covered: the root checks, the setting of the dictionaries, the split stage and its refusal of a
changed split, gates that block on a broken mask, on a training table without the mask record and on a rows gap, and
the gates and the read end to end: a ``closes_gap`` case (the epilepsy-only atoms win more windows in epilepsy
patients, ``Spectral`` without signal), a case with a strong ``Spectral`` (the gap condition fails), a case without
signal, and an inverted case (the orientation is fixed), with the files the read writes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.masked_dictionary as driver  # noqa: E402
import src.masked_dictionary_core as core  # noqa: E402

N_PATIENTS, N_SEGMENTS, N_WINDOWS, N_BOOT = 100, 2, 50, 200
K = core.N_SHARED + core.N_EPILEPSY_ONLY


def _roots(tmp: Path, spectral: float = 0.0) -> dict:
    """A plan of 100 patients in 5 folds with its readout root (saved ``Spectral`` scores with a class effect of
    ``spectral``), audit root, band-code root (saved ``Band-scaled`` scores) and sweep root."""
    rng = np.random.default_rng(0)
    subjects = [f"p{i:03d}" for i in range(N_PATIENTS)]
    labels = {s: int(i % 2 == 0) for i, s in enumerate(subjects)}
    folds = {s: (i // 2) % 5 for i, s in enumerate(subjects)}
    plan = {"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds}
    roots = {name: tmp / name for name in ("readout", "audit", "band_code", "sweep", "root")}
    for folder in ("readout/scores", "audit", "band_code/scores", "sweep"):
        (tmp / folder).mkdir(parents=True)
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
    for path, arm, effect in (
        (roots["readout"] / "scores" / "Spectral.csv", "Spectral", spectral),
        (roots["band_code"] / "scores" / "Band-scaled.csv", "Band-scaled", 0.0),
    ):
        score = [effect * labels[s] + rng.standard_normal() for s in subjects]
        pl.DataFrame(
            {"arm": arm, "subject": subjects, "fold": [folds[s] for s in subjects]}
            | {"label": [labels[s] for s in subjects], "score": score}
        ).write_csv(path)
    return roots | {"plan": plan}


def _write_fold_files(
    roots: dict,
    case: str,
    *,
    violations: float | None = 0.0,
    rows_gap: float = 0.0,
    names: list[str] | None = None,
    root: Path | None = None,
) -> None:
    """The splits, the dictionary folders with their training tables, and a count file per fold for ``case``:
    ``signal`` (the epilepsy-only atoms of the supervised arms win more in epilepsy patients), ``null`` or
    ``inverted``. ``names`` (default: the 7 dictionaries of this read) and ``root`` serve the tests of the hinge
    redesign."""
    rng = np.random.default_rng(1)
    plan, root = roots["plan"], roots["root"] if root is None else root
    driver.split(root, roots["sweep"], roots["readout"])
    names = core.dictionary_names() if names is None else names
    rate = {"signal": (0.05, 0.25), "null": (0.1, 0.1), "inverted": (0.25, 0.05)}[case]
    slots = 4 * N_WINDOWS
    for fold in range(5):
        for name in names:
            folder = driver.dictionary_dir(root, name, fold)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "sae_state.pt").write_bytes(b"")
            table = {"epoch": [1, 2], "val_residual_frac": [0.9, 0.8]}
            if violations is not None:
                table["mask_violations"] = [0.0, violations if name == names[1] else 0.0]
            pl.DataFrame(table).write_csv(folder / "sae_training.csv")
        held = sorted(s for s, f in plan["folds"].items() if f == fold)
        subject = np.repeat(held, N_SEGMENTS)
        label = np.array([plan["labels"][s] for s in subject])
        arrays: dict[str, list[np.ndarray]] = {key: [] for key in (*driver.PER_SEGMENT, "atom_wins")}
        for name in names:
            informed = name.split("_")[0] in ("sup", "strong", "sup2")
            share = np.where(label == 1, rate[1], rate[0]) if informed else np.full(len(label), 0.1)
            epilepsy = rng.binomial(slots, share)
            arrays["wins_epilepsy"].append(epilepsy)
            arrays["wins_shared"].append(slots - epilepsy)
            arrays["energy_epilepsy"].append(epilepsy * 2.0)
            arrays["energy_shared"].append((slots - epilepsy) * 3.0)
            arrays["leak_windows"].append(np.minimum(epilepsy, N_WINDOWS))
            arrays["residual_total"].append(np.full(len(label), 0.8))
            wins = np.zeros((len(label), K), dtype=np.int64)
            wins[:, : core.N_SHARED] = 1
            wins[:, core.N_SHARED] = epilepsy  # one epilepsy-only atom holds every win of its group
            arrays["atom_wins"].append(wins)
        driver._save_npz(
            driver.count_file(root, fold),
            subject=driver._text_array(subject),
            label=label,
            fold=np.array(fold),
            names=np.array(names),
            **{key: np.stack(value) for key, value in arrays.items()},
            windows=np.full(len(label), N_WINDOWS),
            signal_total=np.ones(len(label)),
            rows_gap=np.array(rows_gap),
            repeat_gap=np.array(0.0),
            repeat_segments=np.array(24),
            load_failures=np.array(0),
        )


def _read(roots: dict) -> dict:
    return driver.read(roots["root"], roots["readout"], roots["audit"], roots["band_code"], n_boot=N_BOOT)


def test_roots_and_settings() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, sweep, readout = Path(tmp) / "root", Path(tmp) / "sweep", Path(tmp) / "readout"
        root.mkdir()
        driver.check_root(root, sweep, readout)
        driver.check_root(root, sweep, readout)  # the same roots again
        with pytest.raises(ValueError, match="other roots"):
            driver.check_root(root, sweep, Path(tmp) / "other")
        other = Path(tmp) / "class"
        other.mkdir()
        (other / "class_dictionary_decision.json").write_text("{}")
        with pytest.raises(ValueError, match="root of its own"):
            driver.check_root(other, sweep, readout)
    chosen = driver.setting("perm_s2")
    assert (chosen.K, chosen.L, chosen.D, chosen.P, chosen.R, chosen.epochs, chosen.seed) == (80, 128, 8, 512, 4, 30, 2)
    assert driver.ADOPTED.K == core.N_SHARED  # the shared group has the size of the adopted dictionary


def test_split_stage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        records = driver.split(roots["root"], roots["sweep"], roots["readout"])
        assert len(records) == 5 and all(all(r["checks"].values()) for r in records)
        assert driver.split(roots["root"], roots["sweep"], roots["readout"]) == records  # a rerun is the same
        path = driver.split_file(roots["root"], 0)
        stored = json.loads(path.read_text())
        stored["dictionary_patients"] = stored["dictionary_patients"][1:]
        path.write_text(json.dumps(stored))
        with pytest.raises(ValueError, match="differs"):
            driver.split(roots["root"], roots["sweep"], roots["readout"])


def test_gates_block() -> None:
    for kwargs, gate in (
        ({"violations": 3.0}, "mask_holds"),  # a code entry of an epilepsy-only atom on a label-0 segment
        ({"violations": None}, "mask_holds"),  # a training table without the mask record
        ({"rows_gap": 1e-3}, "rows_match_sweep"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            roots = _roots(Path(tmp))
            _write_fold_files(roots, "signal", **kwargs)
            record = driver.gates(roots["root"], roots["plan"])
            assert record["status"] == "blocked" and not record["gates"][gate], gate
            assert sum(record["gates"].values()) == len(record["gates"]) - 1
            assert _read(roots)["outcome"] == "blocked"


def test_read_closes_gap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "signal")
        gates = driver.gates(roots["root"], roots["plan"])
        assert gates["status"] == "pass" and all(gates["gates"].values()) and not gates["mask_broken"]
        record = _read(roots)
        assert record["outcome"] == "closes_gap" and record["conditions"]["S"] and record["conditions"]["C"]
        arms = {row["arm"]: row for row in record["descriptive"]["arms"]}
        assert set(arms) == {*core.SCORES, "Score-energy", "Spectral", "Band-scaled", "Score+Spectral"}
        assert arms["Score"]["auroc"] > 0.9 and abs(arms["Score-perm"]["auroc"] - 0.5) < 0.2
        assert arms["Score-strong"]["auroc"] > 0.9 and abs(arms["Score-mask"]["auroc"] - 0.5) < 0.2
        described = record["descriptive"]
        assert described["Score_minus_Score_perm"]["low"] > 0 and described["Score_minus_Score_mask"]["low"] > 0
        assert set(described["per_arm"]) == {"sup", "perm", "mask", "strong"}
        assert described["per_arm"]["sup"]["unused_epilepsy_atoms"] == pytest.approx(15 / 16)
        assert 0 < described["per_arm"]["mask"]["leak_rate"] <= 1
        assert described["per_arm"]["sup"]["residual_share"] == pytest.approx(0.8)
        assert described["per_arm"]["sup"]["last_val_residual_frac"] == pytest.approx(0.8)
        assert set(described["matched"]) == {"Score", "Spectral"}
        table = pl.read_csv(roots["root"] / "masked_dictionary_summary.csv")
        assert table.height == len(arms)
        assert json.loads((roots["root"] / "masked_dictionary_decision.json").read_text())["outcome"] == "closes_gap"


def test_read_with_a_strong_spectral() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp), spectral=8.0)  # Spectral separates the classes almost fully
        _write_fold_files(roots, "null")
        driver.gates(roots["root"], roots["plan"])
        record = _read(roots)
        assert not record["conditions"]["S"] and not record["conditions"]["C"]
        assert record["outcome"] in ("no_signal", "review")
        assert record["deciding"]["C_Score_minus_Spectral"]["high"] < 0


def test_read_inverted_does_not_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        roots = _roots(Path(tmp))
        _write_fold_files(roots, "inverted")
        driver.gates(roots["root"], roots["plan"])
        record = _read(roots)
        assert not record["conditions"]["S"] and record["deciding"]["S_Score_auroc"]["high"] < 0.5
        assert record["outcome"] in ("no_signal", "review")
