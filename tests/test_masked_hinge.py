"""Checks of the stages of ``src/masked_hinge.py`` on written files (no corpus; CPU; under a minute).

Runs under pytest. Covered: the refusal of the root of the first run; the strength that stage 2 reads from the pilot
record and its refusal without ``go``; the first-fit gate on training tables; the pilot decision on written counts
(``go`` with the largest strength that works, the control flag, ``no_working_hinge``, and ``blocked`` for each stage-1
gate); and stage 2 on written count files: the gates with gate 8 (``pass``, ``collapsed``, ``blocked``, and
``blocked`` before ``collapsed``), and the read (a ``closes_gap`` case with its files, a case without signal, and the
outcomes ``collapsed`` and ``blocked`` that open no score).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.masked_dictionary as first  # noqa: E402
import src.masked_dictionary_core as core  # noqa: E402
import src.masked_hinge as driver  # noqa: E402
from tests.test_masked_dictionary import N_BOOT, N_SEGMENTS, N_WINDOWS, _roots, _write_fold_files  # noqa: E402

K = core.N_SHARED + core.N_EPILEPSY_ONLY
USE = {1.0: 16, 10.0: 12, 100.0: 9, 1000.0: 3}  # epilepsy-only atoms in use at each strength
LEAK = {1.0: 0.16, 10.0: 0.08, 100.0: 0.02, 1000.0: 0.0}  # against a reference leak of 0.2


def _monitor_arrays(subject: np.ndarray, use: list[int], leak: list[float], selected: list[int]) -> dict:
    """Monitoring counts in which dictionary ``i`` has ``use[i]`` epilepsy-only atoms with wins in every segment and
    a leak of ``leak[i]`` in every segment, so the reads do not depend on the groups."""
    wins = np.zeros((len(use), len(subject), K), dtype=np.int64)
    wins[:, :, : core.N_SHARED] = 1
    for i, n in enumerate(use):
        wins[i, :, core.N_SHARED : core.N_SHARED + n] = 1
    leaks = np.stack([np.full(len(subject), round(value * N_WINDOWS)) for value in leak])
    return {
        "subject": driver._text_array(subject),
        "atom_wins": wins,
        "leak_windows": leaks,
        "windows": np.full(len(subject), N_WINDOWS),
        "selected_epoch": np.array(selected),
    }


def _write_tables(folder: Path, *, leak_rate: float = 0.1, in_use: int = 16, violations: float = 0.0) -> None:
    """A finished pilot dictionary: an empty checkpoint and the two training tables around the last warm epoch."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sae_state.pt").write_bytes(b"")
    epochs = [9.0, 10.0, 11.0]
    pl.DataFrame(
        {"epoch": epochs, "leak_rate": [0.3, leak_rate, 0.05], "mask_violations": [0.0, violations, 0.0]}
    ).write_csv(folder / "sae_training.csv")
    rows = [
        {"epoch": epoch, "atom": atom, "won_windows": int(atom < core.N_SHARED + in_use), "scored_windows": 100}
        for epoch in (9, 10, 11)
        for atom in range(K)
    ]
    pl.DataFrame(rows).write_csv(folder / "sae_won_windows.csv")


def _write_pilot(
    roots: dict,
    *,
    use: dict[float, int] = USE,
    leak: dict[float, float] = LEAK,
    perm_use: dict[float, int] | None = None,
    reference: tuple[int, float] = (16, 0.2),
    selected: int = 30,
    violations: float = 0.0,
    first_in_use: int = 16,
) -> Path:
    """The splits of both roots, the 8 pilot dictionaries with their tables, and the pilot counts."""
    root, first_root = roots["root"], roots["first"]
    driver.split(root, roots["sweep"], roots["readout"])
    first.split(first_root, roots["sweep"], roots["readout"])
    names = core.pilot_names()
    for name in names:
        folder = first.dictionary_dir(root, name, core.PILOT_FOLD)
        _write_tables(
            folder,
            in_use=first_in_use if name == names[0] else 16,
            violations=violations if name == names[3] else 0.0,
        )
    record = json.loads(first.split_file(root, core.PILOT_FOLD).read_text())
    subject = np.repeat(sorted(record["monitor_patients"]), N_SEGMENTS)
    perm_use = use if perm_use is None else perm_use
    uses = [(perm_use if arm == "perm" else use)[eta] for eta in core.ETA_GRID for arm in core.PILOT_ARMS]
    leaks = [leak[eta] for eta in core.ETA_GRID for _ in core.PILOT_ARMS]
    arrays = _monitor_arrays(subject, [*uses, reference[0]], [*leaks, reference[1]], [selected] * len(names) + [30])
    driver._save_npz(driver.pilot_counts_file(root), names=np.array([*names, core.REFERENCE]), **arrays)
    return first_root


def _decide(roots: dict) -> dict:
    return driver.pilot_decide(roots["root"], roots["first"], roots["plan"])


def _write_stage2(
    roots: dict,
    case: str,
    *,
    use: dict[str, int] | None = None,
    selected: int = 30,
    pilot: str = "go",
    **kwargs: float | None,
) -> None:
    """A pilot record, the stage-2 dictionaries with their count and monitoring files, and the counts of the first
    run. ``use`` gives the atoms in use of single dictionaries by ``{name}_f{fold}``; the others keep all 16."""
    root, names = roots["root"], core.stage2_names()
    _write_fold_files(roots, case, names=names, **kwargs)
    _write_fold_files(roots, "null", root=roots["first"])
    driver._save_json(root / "pilot.json", {"outcome": pilot, "eta_star": 10.0 if pilot == "go" else None})
    for fold in range(5):
        record = json.loads(first.split_file(root, fold).read_text())
        subject = np.repeat(sorted(record["monitor_patients"]), N_SEGMENTS)
        uses = [(use or {}).get(f"{name}_f{fold}", 16) for name in names]
        arrays = _monitor_arrays(subject, uses, [0.05] * len(names), [selected] * len(names))
        driver._save_npz(driver.monitor_file(root, fold), names=np.array(names), **arrays)


def _read(roots: dict) -> dict:
    return driver.read(
        roots["root"], roots["readout"], roots["audit"], roots["band_code"], roots["first"], n_boot=N_BOOT
    )


def _all_roots(tmp_path: Path, spectral: float = 0.0) -> dict:
    return _roots(tmp_path, spectral) | {"first": tmp_path / "first"}


def test_roots_and_strength(tmp_path: Path) -> None:
    sweep, readout = tmp_path / "sweep", tmp_path / "readout"
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "masked_dictionary_decision.json").write_text("{}")
    with pytest.raises(ValueError, match="first run"):
        driver.check_root(earlier, sweep, readout)
    root = tmp_path / "root"
    root.mkdir()
    driver.check_root(root, sweep, readout)
    with pytest.raises(ValueError, match="other roots"):
        driver.check_root(root, sweep, tmp_path / "other")
    with pytest.raises(FileNotFoundError, match="pilot"):
        driver.chosen_strength(root)
    for outcome in ("no_working_hinge", "blocked"):
        driver._save_json(root / "pilot.json", {"outcome": outcome, "eta_star": None})
        with pytest.raises(ValueError, match="stage 2 does not run"):
            driver.chosen_strength(root)
    driver._save_json(root / "pilot.json", {"outcome": "go", "eta_star": 100.0})
    assert driver.chosen_strength(root) == 100.0
    assert driver.FULL_EPOCH == 15 and driver.HINGE == {"hinge_start_epoch": 11, "hinge_ramp_epochs": 5}


def test_pilot_check(tmp_path: Path) -> None:
    folder = first.dictionary_dir(tmp_path, core.pilot_names()[0], core.PILOT_FOLD)
    with pytest.raises(FileNotFoundError, match="not finished"):
        driver.pilot_check(tmp_path)
    _write_tables(folder, leak_rate=0.1, in_use=8)
    record = driver.pilot_check(tmp_path)
    assert record["status"] == "pass" and record["leak_rate"] == 0.1  # the row of epoch 10, not 9 or 11
    assert sum(w > 0 for w in record["epilepsy_only_won_windows"]) == 8
    assert json.loads((tmp_path / "pilot_first_fit.json").read_text())["status"] == "pass"
    _write_tables(folder, leak_rate=0.1, in_use=7)
    assert driver.pilot_check(tmp_path)["checks"] == {"leaks": True, "atoms_in_use": False}
    _write_tables(folder, leak_rate=0.0, in_use=16)
    stopped = driver.pilot_check(tmp_path)
    assert stopped["status"] == "stop" and stopped["checks"] == {"leaks": False, "atoms_in_use": True}
    pl.DataFrame({"epoch": [1.0], "leak_rate": [0.2], "mask_violations": [0.0]}).write_csv(folder / "sae_training.csv")
    with pytest.raises(ValueError, match="epoch 10"):
        driver.pilot_check(tmp_path)


def test_pilot_go(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path)
    _write_pilot(roots)
    record = _decide(roots)
    assert record["outcome"] == "go" and record["eta_star"] == 100.0 and all(record["gates"].values())
    works = {eta: read["works"] for eta, read in record["reads"]["sup"].items()}
    assert works == {"1": False, "10": True, "100": True, "1000": False}
    assert record["reads"]["sup"]["100"]["use"] == 9 and record["reads"]["sup"]["100"]["leak"] == pytest.approx(0.02)
    assert record["reference"]["sup"] == {"use": 16, "leak": pytest.approx(0.2)}
    assert record["control_works_at_eta_star"] is True  # the same reads: the hinge works without label information
    assert record["monitor_segments"] == record["monitor_patients"] * N_SEGMENTS
    stored = json.loads((roots["root"] / "pilot.json").read_text())
    assert stored["outcome"] == "go" and driver.chosen_strength(roots["root"]) == 100.0


def test_pilot_control_and_no_working_hinge(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path / "control")
    _write_pilot(roots, perm_use={1.0: 16, 10.0: 12, 100.0: 2, 1000.0: 0})
    record = _decide(roots)
    assert record["outcome"] == "go" and record["eta_star"] == 100.0
    assert record["control_works_at_eta_star"] is False and record["reads"]["perm"]["10"]["works"]

    roots = _all_roots(tmp_path / "none")
    _write_pilot(roots, leak=dict.fromkeys(core.ETA_GRID, 0.16))  # in use, but no strength halves the leak
    record = _decide(roots)
    assert record["outcome"] == "no_working_hinge" and record["eta_star"] is None
    assert all(record["gates"].values()) and record["control_works_at_eta_star"] is None
    with pytest.raises(ValueError, match="no_working_hinge"):
        driver.chosen_strength(roots["root"])


@pytest.mark.parametrize(
    ("kwargs", "gate"),
    [
        ({"reference": (5, 0.2)}, "reference_in_use"),
        ({"reference": (16, 0.0)}, "reference_leaks"),
        ({"selected": 12}, "full_strength_selected"),  # a dictionary from before the full hinge
        ({"violations": 2.0}, "mask_holds"),
        ({"first_in_use": 3}, "first_fit"),
    ],
)
def test_pilot_blocked(tmp_path: Path, kwargs: dict, gate: str) -> None:
    roots = _all_roots(tmp_path)
    _write_pilot(roots, **kwargs)
    record = _decide(roots)
    assert record["outcome"] == "blocked" and record["eta_star"] is None
    assert [name for name, ok in record["gates"].items() if not ok] == [gate]


def test_pilot_blocked_on_files(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path / "split")
    _write_pilot(roots)
    path = first.split_file(roots["first"], core.PILOT_FOLD)
    stored = json.loads(path.read_text())
    stored["monitor_patients"] = stored["monitor_patients"][1:]
    path.write_text(json.dumps(stored))  # the reference trained on another split
    assert [n for n, ok in _decide(roots)["gates"].items() if not ok] == ["split_of_first_run"]

    roots = _all_roots(tmp_path / "names")
    _write_pilot(roots)
    data = driver._load_npz(driver.pilot_counts_file(roots["root"]))
    data["names"] = data["names"][::-1].copy()
    driver._save_npz(driver.pilot_counts_file(roots["root"]), **data)
    with pytest.raises(ValueError, match="other dictionaries"):
        _decide(roots)


def test_stage2_closes_gap(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path)
    _write_stage2(roots, "signal", use={"perm2_s1_f2": 0})  # the control arm does not count for gate 8
    gates = driver.gates(roots["root"], roots["plan"])
    assert gates["status"] == "pass" and all(gates["gates"].values()) and gates["collapsed"] == []
    assert gates["monitor_reads"]["perm2_s1_f2"]["use"] == 0 and gates["monitor_reads"]["sup2_s1_f0"]["use"] == 16
    assert gates["pilot"] == {"outcome": "go", "eta_star": 10.0}
    record = _read(roots)
    assert record["outcome"] == "closes_gap" and record["conditions"]["S"] and record["conditions"]["C"]
    arms = {row["arm"]: row for row in record["descriptive"]["arms"]}
    assert set(arms) == {*core.SCORES2, "Score-energy", "Score-mask", "Spectral", "Band-scaled", "Score+Spectral"}
    assert arms["Score"]["auroc"] > 0.9 and abs(arms["Score-perm"]["auroc"] - 0.5) < 0.2
    assert abs(arms["Score-mask"]["auroc"] - 0.5) < 0.2  # the arm of the first run without a hinge
    described = record["descriptive"]
    assert set(described["per_arm"]) == {"sup2", "perm2"}
    assert len(described["Score_auroc_by_fold"]) == 5
    assert described["Score_auroc_pilot_fold"] == described["Score_auroc_by_fold"][core.PILOT_FOLD]
    assert described["Score_minus_Score_perm"]["low"] > 0 and described["Score_minus_Score_mask"]["low"] > 0
    assert pl.read_csv(roots["root"] / "masked_hinge_summary.csv").height == len(arms)
    stored = json.loads((roots["root"] / "masked_hinge_decision.json").read_text())
    assert stored["outcome"] == "closes_gap" and stored["pilot"]["eta_star"] == 10.0


def test_stage2_without_signal(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path, spectral=8.0)
    _write_stage2(roots, "null")
    assert driver.gates(roots["root"], roots["plan"])["status"] == "pass"
    record = _read(roots)
    assert not record["conditions"]["S"] and not record["conditions"]["C"]
    assert record["outcome"] in ("no_signal", "review")


def test_stage2_collapsed_and_blocked(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path / "collapsed")
    _write_stage2(roots, "signal", use={"sup2_s2_f3": 7})
    gates = driver.gates(roots["root"], roots["plan"])
    assert gates["status"] == "collapsed" and gates["collapsed"] == ["sup2_s2_f3"]
    assert [name for name, ok in gates["gates"].items() if not ok] == ["atoms_in_use"]
    record = _read(roots)
    assert record["outcome"] == "collapsed" and "deciding" not in record and "descriptive" not in record
    assert not (roots["root"] / "masked_hinge_summary.csv").exists()  # no score is read

    for name, kwargs, gate in (
        ("mask", {"violations": 3.0}, "mask_holds"),
        ("rows", {"rows_gap": 1e-3}, "rows_match_sweep"),
        ("epoch", {"selected": 14}, "full_strength_selected"),
        ("pilot", {"pilot": "no_working_hinge"}, "pilot_go"),
    ):
        roots = _all_roots(tmp_path / name)
        _write_stage2(roots, "signal", use={"sup2_s1_f0": 0}, **kwargs)  # gate 8 fails too: blocked comes first
        gates = driver.gates(roots["root"], roots["plan"])
        assert gates["status"] == "blocked", name
        assert [n for n, ok in gates["gates"].items() if not ok] == [gate, "atoms_in_use"], name
        record = _read(roots)
        assert record["outcome"] == "blocked" and "deciding" not in record
