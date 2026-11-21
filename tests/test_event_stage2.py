"""Checks of stage 2a of the Events corpus protocol (no corpus; CPU; about two minutes).

Runs under pytest. Covered: the scan helpers (the score of a second from two channels, alarms as runs, the threshold
for an alarm rate, the marks found); the model on scaled rows does not scale them again; the passes of a crop fit;
and stage 2a on a small written corpus with planted events of three classes: the crops with their shifted copies, the
crop dictionaries of every fold (short fits of the real model), the supervised reference, the features, the read
with its rule, and the scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rootutils
import torch

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.event_dictionary as inventory  # noqa: E402
import src.event_dictionary_core as units_core  # noqa: E402
import src.event_pilot as pilot  # noqa: E402
import src.event_response_core as core  # noqa: E402
import src.event_stage2 as stage2  # noqa: E402

SFREQ = core.SFREQ
CHANNELS = sorted(units_core.MONTAGE[i] for i in range(22) if i not in (8, 13))
SPEC = {
    "n_atoms": 8,
    "atom_len": 128,
    "pool_len": 512,
    "top_r": 4,
    "pre_emphasis": "none",
    "atom_norm": "unit",
    "center_atoms": False,
    "n_projections": 2,
    "loss_support": "extended",
    "step_matched": True,
}
TRAIN = {
    "epochs": 30,
    "lr": 0.001,
    "crop_len": 1024,
    "crop_sampling": "full",
    "crops_per_window": 16,
    "crop_batch": 64,
    "max_crop_energy_ratio": None,
    "init": "random",
    "recenter_atoms": False,
    "full_window_val": True,
    "segment_batch": 2,
    "sampling_unit": "patient",
    "patients_per_batch": 8,
    "segments_per_patient": 4,
    "reseed_dead": False,
}


def test_scan_helpers() -> None:
    grid = torch.zeros(4, 30)
    grid += torch.linspace(0, 0.1, 30)  # a small trend, so that the spread of a channel is not zero
    grid[0, 10] = grid[1, 10] = 5.0  # two channels agree in second 10
    grid[2, 20] = 9.0  # one channel alone in second 20
    score = core.second_scores(grid).numpy()
    assert score.shape == (30,) and int(score.argmax()) == 10 and score[20] < score[10] / 10
    values = np.array([0.0, 3.0, 3.0, 0.0, 2.0, 0.0, 5.0, 5.0, 5.0, 0.0])
    assert core.alarm_starts(values, 1.0).tolist() == [1, 4, 6] and core.alarm_starts(values, 4.0).tolist() == [6]
    assert core.alarm_starts(values, 9.0).tolist() == []
    assert core.alarms_per_record([values, values], 1.0, record_s=10.0) == 3.0
    assert core.threshold_for_rate([values, values], 1.0, record_s=10.0) == 3.0  # above 3: one alarm per record
    assert core.threshold_for_rate([values, values], 3.0, record_s=10.0) == 0.0
    assert np.isnan(core.threshold_for_rate([], 1.0))
    assert core.found_marks(values, [0, 3, 8], 4.0) == 1 and core.found_marks(values, [5, 9], 4.0) == 2


def test_prescaled_rows_and_epochs() -> None:
    model = stage2.PreScaledSAE(
        spec=stage2.SeparableSpec(**SPEC), train_spec=stage2.WaveformTrainSpec(**TRAIN), random_state=0, device="cpu"
    )
    x = 7.0 * torch.randn(3, 20, 2560, generator=torch.Generator().manual_seed(0))
    x[1, 4] = 0.0  # a constant row is not usable
    rows, valid = model._rows(x)
    assert rows.shape == (3, 20, 2560) and torch.equal(rows[0], x[0])  # the rows keep the scale of their record
    assert not bool(valid[1, 4]) and int(valid.sum()) == 59
    assert model._rows(torch.randn(1, 20, 2700))[0].shape[-1] == 2560  # cut to whole windows of 2 s
    assert stage2.epochs_for(145, 8) == 30 and stage2.epochs_for(22, 8) == 130 and stage2.epochs_for(8, 8) == 390


def _corpus(tmp: Path) -> dict[str, Path]:
    """24 patients with one file of 60 s: 6 with spikes, 4 and 4 with two kinds of periodic discharge, 4 with
    artifacts and 6 with background marks only. Each class has its own planted waveform."""
    rng = np.random.default_rng(0)
    data, root = tmp / "corpus", tmp / "root"
    t = (np.arange(128) - 64) / 256
    shapes = {
        1: np.exp(-((t / 0.02) ** 2)) - 0.5 * np.exp(-(((t - 0.08) / 0.06) ** 2)),
        2: np.sin(2 * np.pi * 6 * t) * np.exp(-((t / 0.1) ** 2)),
        3: np.sign(t) * np.exp(-((t / 0.05) ** 2)),
    }
    kinds = [1] * 6 + [2] * 4 + [3] * 4 + [5] * 4 + [6] * 6
    position = {name: i for i, name in enumerate(CHANNELS)}
    signals, durations = {}, {}
    for index, kind in enumerate(kinds):
        patient = f"{index:08d}"
        file = f"{patient}_00000001.edf"
        folder = data / "edf" / "train" / patient
        folder.mkdir(parents=True)
        (folder / file).write_bytes(b"")
        signal = rng.normal(0.0, 1.0, (20, 60 * SFREQ)).astype(np.float32)
        lines = [f"{c},{s}.0,{s + 1}.0,6" for c in (1, 2, 15) for s in (40, 45, 50)]
        if kind in shapes:
            wave = shapes[kind] / np.abs(shapes[kind]).max()
            for second in (10, 20, 30):
                for channel in (0, 14):
                    centre = second * SFREQ + 128
                    signal[position[units_core.MONTAGE[channel]], centre - 64 : centre + 64] += 2.0 * wave
                    lines.append(f"{channel},{second}.0,{second + 1}.0,{kind}")
        elif kind == 5:
            for second in (12, 22):
                signal[position[units_core.MONTAGE[4]], second * SFREQ : (second + 1) * SFREQ] *= 2
                lines.append(f"4,{second}.0,{second + 1}.0,5")
        (folder / file).with_suffix(".rec").write_text("\n".join(lines) + "\n")
        signals[(patient, file)], durations[file] = signal, 60.0
    (data / "edf" / "eval").mkdir()
    reader = lambda path: {"duration": durations[path.name], "sfreq": 250.0, "channels": ["EEG FP1-REF"]}  # noqa: E731
    inventory.inventory(data, root, reader)
    for (patient, file), signal in signals.items():
        path = pilot.cache_path(root, patient, file)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)
    (root / "cache" / "channels.json").write_text(json.dumps(CHANNELS))
    pilot.features(root, data, None, device="cpu", n_random=1)
    pilot.read(root, n_boot=100)
    return {"data": data, "root": root}


def test_stage2a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _corpus(tmp_path)["root"]
    monkeypatch.setattr(stage2, "BANDS", {"30": core.BANDS["30"]})  # one band keeps the test short
    monkeypatch.setattr(
        stage2,
        "SETS",
        {name: entry | {"atoms": (8, 4) if len(entry["atoms"]) == 2 else (8,)} for name, entry in stage2.SETS.items()},
    )
    monkeypatch.setattr(stage2, "MIN_ATOMS_IN_USE", 1)
    table = stage2.load_units(root)
    x, lines = stage2.build_crops(root, table, "30")
    assert x.shape[1:] == (20, stage2.CROP_N) and x.dtype == torch.float16 and len(x) == lines.height
    spikes = lines.filter(pl.col("label") == "spsw")
    assert spikes["patient"].n_unique() == 6 and spikes.height == 6 * 3 * 4  # 3 marks, each with 4 crop positions
    assert set(lines["label"].unique()) == {"spsw", "gped", "pled", "artf"}
    assert float(x.float().abs().max()) <= 20.0  # the rows of the record, clipped; the crop is not scaled again

    for fold in range(5):
        stage2.fit(root, fold, "30", (SPEC, TRAIN), device="cpu", steps=1)
    fits = sorted((root / "stage2" / "fits").glob("f*/30/*/fit.json"))
    assert len(fits) == 5 * 8  # Crops, Crops-balanced and three class sets at two sizes, in five folds
    record = json.loads((root / "stage2" / "fits" / "f0" / "30" / "Crops-spsw-K4" / "fit.json").read_text())
    assert record["atoms"] == 4 and record["patients"] == 4 and record["crops"] == 4 * 12 and record["steps"] >= 1
    assert 1 <= record["atoms_in_use"] <= 4
    balanced = json.loads((root / "stage2" / "fits" / "f0" / "30" / "Crops-balanced-K8" / "fit.json").read_text())
    assert balanced["subjects"] == balanced["patients"]  # here a patient has one class: a unit is a patient and class
    waves = np.load(root / "stage2" / "fits" / "f0" / "30" / "Crops-spsw-K4" / "sae_time_atoms.npy")
    assert waves.shape == (4, 128)
    stage2.fit(root, 0, "30", (SPEC, TRAIN), device="cpu", steps=1)  # finished dictionaries are kept

    for fold in range(5):
        stage2.reference(root, fold, device="cpu", steps=5)
        stage2.features(root, fold, device="cpu")
    ranks = np.load(root / "stage2" / "rank_f0.npy")
    columns = json.loads((root / "stage2" / "columns_f0.json").read_text())
    assert ranks.shape == (table.height, 8 + 8 + 3 * (8 + 4)) and len(columns) == ranks.shape[1]
    assert np.isfinite(ranks).all() and ranks.min() >= 0 and ranks.max() <= 1
    assert stage2.trained_candidates(columns, "spsw") == list(range(8 + 8 + 12))
    assert len(stage2.trained_candidates(columns, "artf")) == 16  # no class crops for a class that does not decide
    reference = np.load(root / "stage2" / "reference_f0.npy")
    assert reference.shape == (table.height, 5) and np.isfinite(reference).all()

    # The small corpus has too few records for the gates of the pilot. A pilot that did not pass blocks the read.
    pilot_record = json.loads((root / "pilot.json").read_text())
    (root / "pilot.json").write_text(json.dumps(pilot_record | {"status": "blocked"}))
    assert stage2.read(root, n_boot=50)["outcome"] == "blocked"
    (root / "pilot.json").write_text(json.dumps(pilot_record | {"status": "pass"}))
    result = stage2.read(root, n_boot=200)
    assert result["outcome"] in ("epileptiform_identified", "none") and result["pilot_status"] == "pass"
    assert result["dictionaries"] == 40 and result["dictionaries_below_the_use_floor"] == 0
    spsw = result["classes"]["spsw"]
    assert set(spsw) >= {"trained", "untrained", "simple", "reference", "gains", "outcome"}
    assert all(c["set"] in ("Crops", "Crops-balanced", "Crops-spsw") for c in spsw["trained"]["chosen"])
    assert 0.0 <= spsw["trained"]["score"] <= 1.0 and spsw["trained"]["interval"][0] <= spsw["trained"]["score"]
    assert set(spsw["gains"]) == {"untrained", "simple", "reference"}
    assert spsw["outcome"] in ("identified (strong)", "identified (weak)", "detected_not_specific", "not_identified")
    assert 0.0 <= spsw["trained"]["mean_rank_class"] <= 1.0
    assert "outcome" not in result["classes"]["eyem"] and result["classes"]["eyem"]["trained"]["score"] is None
    assert pl.read_csv(root / "stage2_summary.csv").height == 5 * 4

    scanned = stage2.scan(root, device="cpu")
    assert scanned["background_only_patients"] == 6 and set(scanned["classes"]) == set(stage2.DECIDING)
    entry = scanned["classes"]["spsw"]["simple"]
    assert set(entry) == {str(rate) for rate in stage2.SCAN_RATES}
    assert entry["1.0"]["marked_seconds"] == 18 and 0 <= entry["1.0"]["found"] <= 18
    low, high = entry["0.2"]["share_found"], entry["5.0"]["share_found"]
    assert low <= high  # more alarms allowed: at least as many marks found
    assert entry["5.0"]["alarms_per_5_min_on_held_out_background"] is not None
    assert json.loads((root / "scan.json").read_text())["rates"] == list(stage2.SCAN_RATES)
