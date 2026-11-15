"""Checks of the stages of ``src/spectral_decomposition.py`` on written files (no corpus; CPU; about a minute).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first failure. Covered:
the input checks (patients, segments, column layout, channel order) and the refusal of a locked patient; the fit, the
gates and the read end to end in a ``fusion_gains`` case (two other arms with independent signal) and a ``no_gain``
case (the other arms repeat ``Spectral``), with the files the read writes; a wrong column layout and a changed saved
score, which block the read; and a root without a channel record, which skips the channel arms and still reads.
For the small-arm lead: the stage end to end on a finished read, its files, a rerun that keeps the refits, and a
changed summary, which blocks it.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import src.spectral_decomposition as driver  # noqa: E402
import src.spectral_decomposition_core as core  # noqa: E402
from src.frozen_readout_core import fit_head  # noqa: E402

N_PATIENTS, N_SEGMENTS, N_BOOT = 100, 2, 200
CHANNELS = sorted(c for chain in core.CHAINS.values() for c in chain)
LEAD_TEST_SEEDS = (11, 12)
_CACHE: dict[str, dict] = {}


def _expect(error: type[Exception], function, message: str) -> None:
    try:
        function()
    except error:
        return
    raise AssertionError(message)


def _features(subjects: list[str], labels: dict[str, int], rng: np.random.Generator) -> np.ndarray:
    """Rows with the layout of ``SpectralFeatures``; epilepsy patients have more power in the second band."""
    rows = []
    for s in subjects:
        gain = 1.0 + 0.45 * labels[s] + 0.5 * rng.standard_normal()
        for _ in range(N_SEGMENTS):
            power = rng.uniform(1.0, 3.0, (core.N_CHANNELS, core.N_BANDS))
            power[:, 1] *= np.exp(0.5 * gain)
            rows.append(np.concatenate([np.log10(power), power / power.sum(axis=1, keepdims=True)], axis=1).ravel())
    return np.array(rows, dtype=np.float32)


def _write_scores(path: Path, subjects: list[str], plan: dict, score: np.ndarray, arm: str | None = None) -> None:
    columns = {
        "subject": subjects,
        "fold": [plan["folds"][s] for s in subjects],
        "label": [plan["labels"][s] for s in subjects],
        "score": score.tolist(),
    }
    pl.DataFrame(({"arm": arm} if arm else {}) | columns).write_csv(path)


def _roots(tmp: Path) -> dict:
    """A plan of 100 patients in 5 folds with a readout root (features, saved ``Spectral`` scores from the head of
    pass 1, channel record), an encoder root and an audit root."""
    rng = np.random.default_rng(0)
    subjects = [f"p{i:03d}" for i in range(N_PATIENTS)]
    labels = {s: int(i % 2 == 0) for i, s in enumerate(subjects)}
    folds = {s: (i // 2) % 5 for i, s in enumerate(subjects)}
    plan = {"pass": 1, "seed": 1, "n_segments": N_SEGMENTS, "labels": labels, "folds": folds}
    roots = {name: tmp / name for name in ("readout", "audit", "encoder", "root")}
    for folder in (
        "readout/features",
        "readout/scores",
        "readout/dictionaries/untrained_f0",
        "audit",
        "encoder/scores",
    ):
        (tmp / folder).mkdir(parents=True)
    (roots["readout"] / "folds.json").write_text(json.dumps(plan))
    pl.DataFrame(
        {"subject": subjects, "label": [labels[s] for s in subjects], "fold": [folds[s] for s in subjects]}
        | {"segments": rng.integers(2, 60, N_PATIENTS).tolist()}
    ).write_csv(roots["readout"] / "patients.csv")
    x = _features(subjects, labels, rng)
    every = np.repeat(subjects, N_SEGMENTS)
    np.savez(
        roots["readout"] / "features" / "Spectral.npz",
        subject=every,
        label=np.array([labels[s] for s in every]),
        features=x,
    )
    rows = []
    for fold in range(5):  # the saved scores of pass 1: its head on all columns
        held = np.array([folds[s] == fold for s in every])
        result = fit_head(x, np.array([labels[s] for s in every]), every, held, StandardScaler, seed=1 + fold)
        rows += [
            {"arm": "Spectral", "subject": s, "fold": fold, "label": int(y), "score": float(v), "windows": 2}
            | {"c": result.c}
            for s, y, v in zip(result.subject, result.label, result.score, strict=True)
        ]
    pl.DataFrame(rows).write_csv(roots["readout"] / "scores" / "Spectral.csv")
    (roots["readout"] / "dictionaries" / "untrained_f0" / "sae_channels.txt").write_text("\n".join(CHANNELS) + "\n")
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
    return roots | {"plan": plan, "subjects": subjects}


def _other_arms(roots: dict, case: str) -> None:
    """The saved ``HYDRA`` and REVE scores: ``independent`` (their own strong signal) or ``same`` (``Spectral``)."""
    rng = np.random.default_rng(2)
    plan, subjects = roots["plan"], roots["subjects"]
    label = np.array([plan["labels"][s] for s in subjects])
    spectral = driver._scores(roots["readout"] / "scores" / "Spectral.csv")
    for path, arm in (
        (roots["readout"] / "scores" / "HYDRA.csv", "HYDRA"),
        (roots["encoder"] / "scores" / "D.csv", None),
    ):
        score = spectral.score if case == "same" else 2.0 * label + rng.standard_normal(N_PATIENTS)
        _write_scores(path, subjects, plan, score, arm)


def _base() -> dict:
    """One fitted root with its gates, shared by the tests that only read."""
    if "base" not in _CACHE:
        roots = _roots(Path(tempfile.mkdtemp()))
        _other_arms(roots, "independent")
        driver.fit(roots["root"], roots["readout"])
        _CACHE["base"] = roots
    return _CACHE["base"]


def _copy(roots: dict) -> dict:
    tmp = Path(tempfile.mkdtemp())
    for name in ("readout", "audit", "encoder", "root"):
        shutil.copytree(roots[name], tmp / name)
    return roots | {name: tmp / name for name in ("readout", "audit", "encoder", "root")}


def _read(roots: dict) -> dict:
    return driver.read(roots["root"], roots["readout"], roots["audit"], roots["encoder"], n_boot=N_BOOT)


def _lead(roots: dict) -> dict:
    return driver.lead(
        roots["root"], roots["readout"], roots["audit"], roots["encoder"], n_perm=300, seeds=LEAD_TEST_SEEDS
    )


def test_inputs_and_locked_patient() -> None:
    roots = _base()
    inputs = json.loads((roots["root"] / "inputs.json").read_text())
    assert inputs["patients"] == N_PATIENTS and inputs["patients_ok"] and inputs["segments_ok"] and inputs["labels_ok"]
    assert inputs["columns"] == 200 and inputs["layout_gap"] < core.LAYOUT_TOLERANCE
    assert inputs["channels"] == CHANNELS and inputs["channels_ok"] and not inputs["channel_problems"]
    locked = _copy(roots)
    path = locked["readout"] / "features" / "Spectral.npz"
    data = dict(np.load(path))
    data["subject"] = np.append(data["subject"], "test001")
    data["label"] = np.append(data["label"], 1)
    data["features"] = np.vstack([data["features"], data["features"][:1]])
    np.savez(path, **data)
    _expect(ValueError, lambda: driver.fit(locked["root"], locked["readout"]), "a locked patient must be refused")
    short = _copy(roots)
    path = short["readout"] / "features" / "Spectral.npz"
    data = dict(np.load(path))
    np.savez(path, **{k: v[:-1] for k, v in data.items()})  # one patient with one segment too few
    _expect(ValueError, lambda: driver.fit(short["root"], short["readout"]), "a missing segment must be refused")


def test_fusion_gains() -> None:
    roots = _copy(_base())
    gates = driver.gates(roots["root"], roots["readout"], roots["encoder"])
    assert gates["status"] == "pass" and gates["reproduction"]["ok"] and gates["reproduction"]["score_gap"] < 1e-8
    assert gates["skipped_arms"] == [] and gates["complete_ok"] and all(gates["score_files"].values())
    record = _read(roots)
    assert record["outcome"] == "fusion_gains" and record["conditions"] == {"F": True}
    deciding = record["deciding"]["F_Fusion_minus_Spectral"]
    assert deciding["low"] > 0 and deciding["low"] <= deciding["value"] <= deciding["high"]
    arms = {row["arm"]: row for row in record["descriptive"]["arms"]}
    assert len(arms) == 26 + 7 and arms["Fusion"]["auroc"] > arms["Spectral"]["auroc"]
    assert arms["Spectral"]["difference"] == 0.0 and arms[core.FULL]["reference"] == core.FULL
    assert abs(arms[core.FULL]["auroc"] - arms["Spectral"]["auroc"]) < 1e-9  # gate 3: the same scores
    assert arms["Fusion-HYDRA-REVE"]["reference"] == "Spectral" and arms["Spec-rel"]["reference"] == core.FULL
    # The signal sits in the second band: the arm without it loses, and the arm with it alone keeps most of it.
    assert arms["Spec-drop-4_8"]["difference"] < arms["Spec-drop-30_45"]["difference"]
    assert arms["Spec-band-4_8"]["auroc"] > arms["Spec-band-30_45"]["auroc"]
    null = record["descriptive"]["permutation_null"]
    assert set(null) == set(core.NULL_ARMS) and all(len(v["draws"]) == 20 for v in null.values())
    assert set(record["descriptive"]["matched"]) == {"Fusion", "Spectral"}
    assert len(record["descriptive"]["rank_correlation_within_fold"]) == 6
    table = pl.read_csv(roots["root"] / "decomposition_summary.csv")
    assert table.height == 33 and {"arm", "auroc", "gain", "off_off", "difference", "reference"} <= set(table.columns)
    assert json.loads((roots["root"] / "fusion_decision.json").read_text())["outcome"] == "fusion_gains"


def test_no_gain_when_the_arms_repeat_spectral() -> None:
    roots = _copy(_base())
    _other_arms(roots, "same")
    assert driver.gates(roots["root"], roots["readout"], roots["encoder"])["status"] == "pass"
    record = _read(roots)
    assert record["outcome"] == "no_gain" and record["conditions"] == {"F": False}
    deciding = record["deciding"]["F_Fusion_minus_Spectral"]
    assert deciding["value"] == 0.0 and deciding["low"] == 0.0  # the same ranks: no difference at all
    agreement = record["descriptive"]["rank_correlation_within_fold"]
    assert all(abs(v - 1.0) < 1e-9 for v in agreement.values())


def test_wrong_layout_blocks() -> None:
    roots = _copy(_base())
    path = roots["readout"] / "features" / "Spectral.npz"
    data = dict(np.load(path))
    cube = core.split_columns(data["features"])
    data["features"] = np.concatenate([cube[:, :, core.N_BANDS :], cube[:, :, : core.N_BANDS]], axis=2).reshape(
        len(cube), -1
    )
    np.savez(path, **data)
    inputs = driver.check_inputs(roots["readout"], roots["plan"])
    assert inputs["layout_gap"] > core.LAYOUT_TOLERANCE
    (roots["root"] / "inputs.json").write_text(json.dumps(inputs))
    gates = driver.gates(roots["root"], roots["readout"], roots["encoder"])
    assert gates["status"] == "fail" and not gates["layout_ok"]
    record = _read(roots)
    assert record["outcome"] == "blocked" and "deciding" not in record


def test_changed_saved_scores_block() -> None:
    roots = _copy(_base())
    path = roots["readout"] / "scores" / "Spectral.csv"
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8})
    frame.with_columns(
        pl.when(pl.col("subject") == "p000").then(pl.col("score") + 0.01).otherwise(pl.col("score")).alias("score")
    ).write_csv(path)
    gates = driver.gates(roots["root"], roots["readout"], roots["encoder"])
    assert gates["status"] == "fail" and not gates["reproduction"]["ok"] and gates["reproduction"]["same_c"]
    moved = _copy(_base())
    path = moved["encoder"] / "scores" / "D.csv"
    pl.read_csv(path, schema_overrides={"subject": pl.Utf8}).with_columns((pl.col("fold") + 1) % 5).write_csv(path)
    gates = driver.gates(moved["root"], moved["readout"], moved["encoder"])
    assert gates["status"] == "fail" and not gates["score_files"]["REVE"] and not gates["inputs_ok"]


def test_missing_channel_record_skips_arms() -> None:
    roots = _roots(Path(tempfile.mkdtemp()))
    _other_arms(roots, "independent")
    shutil.rmtree(roots["readout"] / "dictionaries")
    names, problems = driver.channel_order(roots["readout"])
    assert names is None and problems
    driver.fit(roots["root"], roots["readout"])
    written = {p.stem for p in (roots["root"] / "scores").glob("*.csv")}
    assert written == {a for a in core.arm_names() if not core.needs_channels(a)} and len(written) == 15
    gates = driver.gates(roots["root"], roots["readout"], roots["encoder"])
    assert gates["status"] == "pass" and not gates["channels_ok"] and len(gates["skipped_arms"]) == 11
    record = _read(roots)
    assert record["outcome"] in ("fusion_gains", "no_gain") and len(record["descriptive"]["arms"]) == 15 + 7
    # Two records that disagree give no order either.
    for i, order in enumerate((CHANNELS, CHANNELS[::-1])):
        folder = roots["readout"] / "dictionaries" / f"untrained_f{i}"
        folder.mkdir(parents=True)
        (folder / "sae_channels.txt").write_text("\n".join(order) + "\n")
    assert driver.channel_order(roots["readout"])[0] is None


def test_lead_stage() -> None:
    roots = _copy(_base())
    driver.gates(roots["root"], roots["readout"], roots["encoder"])
    _expect(ValueError, lambda: _lead(roots), "a root without a read must be refused")  # no summary yet
    _read(roots)
    record = _lead(roots)
    gates = json.loads((roots["root"] / "lead_gates.json").read_text())
    assert gates["status"] == "pass" and gates["arms"] == 33 and gates["cell_gap"] < 1e-12 and gates["refit_gap"] < 1e-8
    assert record["outcome"] in ("survives", "dropped") and set(record["conditions"]) == {"P", "M"}
    selection, median = record["deciding"]["P_selection"], record["deciding"]["M_median_cell"]
    assert 0 < selection["p"] <= 1 and selection["permutations"] == 300
    assert selection["arm"] in set(core.arm_names()) | {"Spectral", "HYDRA", "REVE"} or selection["arm"].startswith(
        "Fusion"
    )
    assert 0.5 <= selection["largest"] <= 1.0  # the largest of 33 arms is never below one half here
    assert record["conditions"]["P"] == (selection["p"] < core.LEAD_P)
    assert record["conditions"]["M"] == (median["median"] >= core.LEAD_MEDIAN) and median["assignments"] == 2
    cell = record["cell"]
    assert cell["epilepsy"] > 0 and cell["non_epilepsy"] > 0
    described = record["descriptive"]
    assert set(described["lead_arms"]) == set(core.LEAD_ARMS)
    assert described["null_maximum"]["p95"] >= described["null_maximum"]["median"] > 0.5
    assert set(described["assignments"]) == {core.LEAD_ARM, core.FULL}
    for arm in described["assignments"].values():
        assert len(arm["values"]) == 2 and arm["cell"]["min"] <= arm["cell"]["median"] <= arm["cell"]["max"]
    files = sorted(p.name for p in (roots["root"] / "lead").glob("*.csv"))
    assert files == [f"scores_s{s}.csv" for s in LEAD_TEST_SEEDS]
    frame = pl.read_csv(roots["root"] / "lead" / files[0], schema_overrides={"subject": pl.Utf8})
    assert frame.height == 2 * N_PATIENTS and set(frame["arm"].to_list()) == {core.LEAD_ARM, core.FULL}
    # The new assignment differs from the plan's, and it keeps every fold stratified.
    moved = sum(row["fold"] != roots["plan"]["folds"][row["subject"]] for row in frame.to_dicts()) / frame.height
    assert moved > 0.5
    stamp = (roots["root"] / "lead" / files[0]).stat().st_mtime_ns
    again = _lead(roots)  # a rerun keeps the refits and gives the same record
    assert (roots["root"] / "lead" / files[0]).stat().st_mtime_ns == stamp
    assert again["deciding"] == record["deciding"]


def test_lead_blocks_on_a_changed_summary() -> None:
    roots = _copy(_base())
    driver.gates(roots["root"], roots["readout"], roots["encoder"])
    _read(roots)
    path = roots["root"] / "decomposition_summary.csv"
    pl.read_csv(path).with_columns(pl.col("off_off") + 0.01).write_csv(path)
    record = _lead(roots)
    assert record["outcome"] == "blocked" and "deciding" not in record
    gates = json.loads((roots["root"] / "lead_gates.json").read_text())
    assert gates["status"] == "fail" and not gates["cell_ok"] and gates["refit_ok"]
    assert not (roots["root"] / "lead").exists()  # no refit on other assignments after a failed gate


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
