"""Checks of stage 1 of the Events corpus protocol: responses, ranks, scores and the pilot (no corpus; CPU).

Runs under pytest. Covered: the response of a waveform is centered on the patch; the simple measures; the rank inside
a record leaves out the seconds that overlap the window; the AUROC of columns and the pair counts agree, and the
bootstrap of patients with every patient once gives the point value; the choice of a feature uses the training folds
only; the outcome of a class from S, L and B; the displaced seconds of the null; and the pilot on a small written
corpus in which one waveform is planted at the marked spike seconds: the unit table and its exclusions, the three
reads, the time-alignment check, and the read with its gates.
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

SFREQ = core.SFREQ


def _spike() -> np.ndarray:
    """A sharp waveform of 128 samples and unit norm."""
    t = (np.arange(128) - 64) / 256
    wave = np.exp(-((t / 0.02) ** 2)) - 0.5 * np.exp(-(((t - 0.08) / 0.06) ** 2))
    return (wave / np.linalg.norm(wave)).astype(np.float32)


def test_constants() -> None:
    assert list(core.BANDS) == ["30", "70", "70-diff"] and core.BANDS["70-diff"] == (1.0, 70.0, True)
    assert len(core.simple_names("30")) == 7 and len(core.simple_names("70")) == 8
    assert "power_30_70" not in core.simple_names("30") and core.simple_names("70-diff") == core.SIMPLE
    assert (core.S_FLOOR, core.STRONG, core.GAIN) == (0.65, 0.80, 0.03) and core.LEVEL == pytest.approx(1 - 0.05 / 3)
    first, second = core.random_atoms("30", 0), core.random_atoms("70", 0)
    assert first.shape == (64, 128) and np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(first, core.random_atoms("30", 0)) and not np.array_equal(first, second)


def test_response_is_centered() -> None:
    wave = torch.from_numpy(_spike())
    rows = torch.zeros(2, 4 * SFREQ)
    rows[0, 600 - 64 : 600 + 64] = 3 * wave  # the patch is centered on sample 600
    response = core.responses(rows, wave.unsqueeze(0))
    assert response.shape == (2, 1, 4 * SFREQ)
    assert int(response[0, 0].argmax()) == 600 and float(response[0, 0, 600]) == pytest.approx(3.0, abs=1e-4)
    assert float(response[1].abs().max()) == 0.0
    flipped = core.responses(-rows, wave.unsqueeze(0))
    assert torch.allclose(flipped, response)  # the absolute value: the sign of the discharge does not matter


def test_measures() -> None:
    t = torch.arange(SFREQ) / SFREQ
    windows = torch.stack([torch.sin(2 * torch.pi * 10 * t), 2 * torch.sin(2 * torch.pi * 40 * t)])
    found = core.measures(windows, "70")
    names = core.simple_names("70")
    assert found.shape == (2, 8)
    assert int(found[0, :5].argmax()) == names.index("power_8_13") and int(found[1, :5].argmax()) == 4
    assert float(found[0, names.index("peak_to_peak")]) == pytest.approx(2.0, abs=0.01)
    assert float(found[1, names.index("max_abs")]) == pytest.approx(2.0, abs=0.01)
    assert float(found[1, names.index("line_length")]) > float(found[0, names.index("line_length")])
    assert core.measures(windows, "30").shape == (2, 7)


def test_record_features_and_rank() -> None:
    wave = torch.from_numpy(_spike())
    generator = torch.Generator().manual_seed(0)
    rows = 0.1 * torch.randn(3, 20 * SFREQ, generator=generator)
    start = 7 * SFREQ + 100  # a window that is not on the grid of whole seconds
    rows[1, start + 64 : start + 192] += 5 * wave
    atoms = torch.stack([wave, torch.from_numpy(core.random_atoms("30", 0)[0])])
    grid, at = core.record_features(rows, atoms, "30", [start, 2 * SFREQ])
    assert grid.shape == (3, 2 + 7, 20) and at.shape == (2, 3, 2 + 7)
    rank = core.rank_in_record(at[0], grid, start)
    assert rank.shape == (3, 9) and float(rank[1, 0]) == 1.0  # the planted channel, its own waveform
    assert 0.0 <= float(rank[0, 0]) < 1.0
    # The seconds 7 and 8 hold the planted event. They overlap the window and are not in its reference, so a
    # window without the event is compared with 19 seconds, and the planted seconds stand above it.
    quiet = core.rank_in_record(at[1], grid, 2 * SFREQ)
    assert float(quiet[1, 0]) <= 17 / 19
    keep_all = core.rank_in_record(torch.full((3, 9), -1.0), grid, start)
    assert float(keep_all.max()) == 0.0


def test_auroc_and_pair_counts() -> None:
    rng = np.random.default_rng(0)
    values = np.round(rng.normal(size=300), 1)  # with ties
    positive = rng.random(300) < 0.3
    values[positive] += 0.8
    patients = rng.choice(np.array(["a", "b", "c", "d", "e"]), size=300)
    direct = core.auroc_columns(values[:, None], positive)[0]
    lower = (values[positive][:, None] > values[~positive][None]).sum()
    ties = (values[positive][:, None] == values[~positive][None]).sum()
    assert direct == pytest.approx((lower + 0.5 * ties) / (positive.sum() * (~positive).sum()))
    counts = core.pair_counts(values, positive, patients)
    assert core.auroc_of(counts) == pytest.approx(direct)
    assert core.auroc_of(counts, np.ones((1, 5)))[0] == pytest.approx(direct)  # every patient once
    twice = np.array([[2.0, 1, 1, 1, 0]])
    repeated = np.concatenate([np.flatnonzero(patients == "a"), np.flatnonzero(patients != "e")])
    again = core.auroc_columns(values[repeated][:, None], positive[repeated])[0]
    assert core.auroc_of(counts, twice)[0] == pytest.approx(again)  # a draw equals the units repeated
    draws = core.patient_draws(patients, 500, seed=0)
    assert draws.shape == (500, 5) and np.all(draws.sum(axis=1) == 5)
    values[:3] = np.nan
    assert np.isfinite(core.auroc_columns(values[:, None], positive)[0])
    assert core.pair_counts(values, positive, patients)["n_pos"].sum() == positive[3:].sum()
    assert np.isnan(core.auroc_columns(values[:, None], np.zeros(300, dtype=bool))[0])


def test_choice_uses_training_folds_only() -> None:
    rng = np.random.default_rng(1)
    n = 600
    fold = rng.integers(0, 3, size=n)
    patient = np.array([f"p{f}{i % 7}" for i, f in enumerate(fold)])
    label = np.where(rng.random(n) < 0.3, "spsw", "bckg")
    signal = (label == "spsw") * 1.5 + rng.normal(size=n)
    trap = rng.normal(size=n)
    trap[fold != 1] = (label[fold != 1] == "spsw") * 9.0  # perfect in folds 0 and 2, noise in fold 1
    features = np.stack([signal, trap, rng.normal(size=n)], axis=1)
    chosen = core.choose_and_score(features, [[0, 1, 2]] * 3, label, patient, fold, "spsw", 3)
    # With fold 0 held out, the trap is noise on half of the training units and is not chosen.
    assert chosen[0]["column"] == 0 and core.auroc_of(chosen[0]["counts"]) > 0.75
    assert chosen[1]["column"] == 1  # with fold 1 held out, the trap is perfect on the training folds
    assert core.auroc_of(chosen[1]["counts"]) < 0.65  # and it is noise on the held-out fold: no held-out unit chose it
    assert core.choose_and_score(features, [[0, 1, 2]] * 3, label, patient, fold, "gped", 3) == [None] * 3
    draws = [
        {"patients": np.unique(patient[fold == f]), "counts": core.patient_draws(patient[fold == f], 200, seed=f)}
        for f in range(3)
    ]
    summary = core.summarize(chosen, draws)
    assert summary["folds"] == 3 and summary["boot"].shape == (200,)
    low, high = core.interval(summary["boot"])
    assert low < summary["score"] < high


def test_class_outcome() -> None:
    good, none = {"gain": 0.05, "low": 0.01}, {"gain": 0.05, "low": -0.01}
    small = {"gain": 0.02, "low": 0.01}
    assert core.class_outcome(0.85, good, good) == "identified (strong)"
    assert core.class_outcome(0.70, good, good) == "identified (weak)"
    assert core.class_outcome(0.70, good, none) == "detected_not_specific"
    assert core.class_outcome(0.70, small, good) == "detected_not_specific"
    assert core.class_outcome(0.64, good, good) == "not_identified"


def test_null_starts() -> None:
    rng = np.random.default_rng(0)
    starts = np.array([10 * SFREQ + 100, 10 * SFREQ + 100, 30 * SFREQ])
    found = pilot.null_starts(starts, 60, rng)
    assert found.shape == (3,) and np.all(found % SFREQ == 0)
    seconds = found // SFREQ
    assert np.all((seconds >= 1) & (seconds <= 58))
    assert not np.isin(seconds, [9, 10, 11, 12, 29, 30, 31, 32]).any()
    assert np.all(pilot.null_starts(np.array([SFREQ]), 4, rng) == -1)  # no free second


# ---------------------------------------------------------------------- the pilot on a written corpus

DURATION = 60.0
CHANNELS = [
    name.title() if False else name for name in sorted(units_core.MONTAGE[i] for i in range(22) if i not in (8, 13))
]


def _corpus(tmp: Path) -> dict[str, Path]:
    """30 patients with one file of 60 s: 10 with spikes on channels 0 and 14 at 3 seconds, 10 with artifacts, and
    10 with background marks only. The spike waveform is planted at the marked spike seconds, at an amplitude that
    its own waveform finds and that a random waveform or the amplitude of the second does not. An artifact second
    has twice the amplitude."""
    rng = np.random.default_rng(0)
    data, root, sweep = tmp / "corpus", tmp / "root", tmp / "sweep"
    wave = _spike()
    position = {name: i for i, name in enumerate(CHANNELS)}
    signals, durations = {}, {}
    for index in range(30):
        patient = f"{index:08d}"
        file = f"{patient}_00000001.edf"
        folder = data / "edf" / "train" / patient
        folder.mkdir(parents=True)
        (folder / file).write_bytes(b"")
        signal = rng.normal(0.0, 1.0, (20, int(DURATION * SFREQ))).astype(np.float32)
        lines = [f"{c},{s}.0,{s + 1}.0,6" for c in (1, 2, 15) for s in (40, 45, 50)]
        if index < 10:
            for second in (10, 20, 30):
                for channel in (0, 14):
                    centre = second * SFREQ + 128
                    signal[position[units_core.MONTAGE[channel]], centre - 64 : centre + 64] += 1.5 * wave / wave.max()
                    lines.append(f"{channel},{second}.0,{second + 1}.0,1")
        elif index < 20:
            for second in (12, 22):
                signal[position[units_core.MONTAGE[4]], second * SFREQ : (second + 1) * SFREQ] *= 2
                lines.append(f"4,{second}.0,{second + 1}.0,5")
        (folder / file).with_suffix(".rec").write_text("\n".join(lines) + "\n")
        signals[(patient, file)], durations[file] = signal, DURATION
    (data / "edf" / "eval").mkdir()
    reader = lambda path: {"duration": durations[path.name], "sfreq": 250.0, "channels": ["EEG FP1-REF"]}  # noqa: E731
    inventory.inventory(data, root, reader)
    for (patient, file), signal in signals.items():
        path = pilot.cache_path(root, patient, file)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, signal)
    (root / "cache" / "channels.json").write_text(json.dumps(CHANNELS))
    for fold in range(5):
        folder = sweep / "dictionaries" / f"{pilot.ADOPTED}_f{fold}"
        folder.mkdir(parents=True)
        atoms = core.random_atoms("30", 10 + fold, n_atoms=6)
        atoms[2] = wave  # every dictionary of the other corpus holds the spike waveform
        np.save(folder / "sae_time_atoms.npy", atoms)
    return {"data": data, "root": root, "sweep": sweep}


def test_unit_table_and_exclusions(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    table, left = pilot.unit_table(paths["root"], paths["data"], CHANNELS)
    assert table.height == 10 * 15 + 10 * 11 + 10 * 9 and left == {
        "repaired_channel": 0,
        "bad_recording": 0,
        "no_cache": 0,
    }
    assert set(table["label"].unique()) == {"spsw", "artf", "bckg"} and table["fold"].n_unique() == 5
    spike = table.filter(pl.col("label") == "spsw")
    assert sorted(spike["channel"].unique().to_list()) == sorted(CHANNELS.index(n) for n in ("FP1-F7", "FP1-F3"))
    assert spike["start"].min() == 10.0 and table["piece_stop"].unique().to_list() == [60.0]
    # A bad electrode: the units of its channels leave. Too many bad channels: every unit of the recording leaves.
    first = paths["data"] / "edf" / "train" / "00000000" / "00000000_00000001-bads.json"
    first.write_text(json.dumps({"bad_channels": ["Fp1"], "too_many_bad_channels": False}))
    second = paths["data"] / "edf" / "train" / "00000001" / "00000001_00000001-bads.json"
    second.write_text(json.dumps({"bad_channels": ["T3", "O2"], "too_many_bad_channels": True}))
    pilot.cache_path(paths["root"], "00000002", "00000002_00000001.edf").unlink()
    _, left = pilot.unit_table(paths["root"], paths["data"], CHANNELS)
    assert left == {"repaired_channel": 6, "bad_recording": 15, "no_cache": 15}
    assert pilot.read_bads(first) == (("Fp1",), False) and pilot.read_bads(Path("none.edf")) == ((), False)


def test_dictionaries_and_columns(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    atoms, columns = pilot.dictionaries(paths["sweep"], n_random=1)
    assert atoms["30"].shape == (5 * 6 + 64, 128) and atoms["70"].shape == (64, 128)
    assert len(columns) == (30 + 64 + 7) + (64 + 8) + (64 + 8)
    assert pilot.family_columns(columns, "transfer", 3) == list(range(18, 24))
    assert len(pilot.family_columns(columns, "untrained", 0)) == 3 * 64
    assert len(pilot.family_columns(columns, "simple", 0)) == 7 + 8 + 8
    assert columns[30 + 64]["dictionary"] == "power_1_4" and columns[-1]["dictionary"] == "max_abs"
    without, described = pilot.dictionaries(None, n_random=1)
    assert without["30"].shape == (64, 128) and not any(c["family"] == "transfer" for c in described)


def test_pilot_features_and_read(tmp_path: Path) -> None:
    paths = _corpus(tmp_path)
    pilot.features(paths["root"], paths["data"], paths["sweep"], device="cpu", n_random=1)
    folder = paths["root"] / "pilot"
    rank, plain, null = (np.load(folder / f"{name}.npy") for name in pilot.READS)
    table = pl.read_csv(folder / "units.csv", schema_overrides={"patient": pl.Utf8})
    columns = json.loads((folder / "columns.json").read_text())
    assert rank.shape == plain.shape == null.shape == (table.height, len(columns)) and table.height == 350
    assert np.nanmin(rank) >= 0.0 and np.nanmax(rank) <= 1.0 and np.isfinite(rank).all() and np.isfinite(null).all()
    spikes = (table["label"] == "spsw").to_numpy()
    own = [i for i, c in enumerate(columns) if c["family"] == "transfer" and c["index"] == 2]
    # The planted waveform stands near the top of its record. The two other planted seconds of its channel are in
    # the reference, so the rank is a little below 1.
    assert rank[spikes][:, own].min() >= 0.9 and rank[spikes][:, own].mean() > 0.96
    assert abs(float(null[spikes][:, own].mean()) - 0.5) < 0.15  # at a displaced second it does not
    alignment = json.loads((folder / "alignment.json").read_text())
    assert alignment["pass"] and alignment["auroc"] >= 0.95 and alignment["auroc_2s_later"] < 0.60
    assert alignment["records"] == 6  # the records of fold 0, one planted unit in each: not the number of units

    record = pilot.read(paths["root"], n_boot=300)
    spsw = record["classes"]["spsw"]
    # The artifact units are loud for a whole second, so the spike waveform also responds to them: the score of
    # spikes against all other units is high and not perfect.
    assert spsw["rank"]["transfer"]["score"] > 0.9
    assert all(c["family"] == "transfer" and c["index"] == 2 for c in spsw["rank"]["transfer"]["chosen"])
    assert [c["dictionary"] for c in spsw["rank"]["transfer"]["chosen"]] == [f"f{f}" for f in range(5)]
    low, high = spsw["rank"]["transfer"]["interval"]
    assert low <= spsw["rank"]["transfer"]["score"] <= high
    # On this corpus the waveform beats the random waveforms (L holds). The largest absolute value of the second
    # finds the planted spike as well (B fails), so the rule gives "detected_not_specific".
    assert spsw["rank"]["gain_over_untrained"]["gain"] > 0.03 and spsw["rank"]["gain_over_untrained"]["low"] > 0
    assert spsw["rank"]["gain_over_simple"]["gain"] < 0.03
    assert spsw["rank"]["outcome_if_it_decided"] == "detected_not_specific"
    assert record["classes"]["artf"]["rank"]["simple"]["score"] > 0.9  # an amplitude change: a simple measure finds it
    means = spsw["rank"]["transfer"]["mean_rank"]
    assert means["class"] > 0.95 and 0.3 < means["background"] < 0.7  # the event stands out; the background does not
    assert set(spsw["rank"]["by_band"]) == {"30", "70", "70-diff"}
    assert set(spsw["rank"]["by_band"]["70"]) == {"untrained", "simple"} and "by_band" not in spsw["plain"]
    assert record["patients_per_class"] == {"spsw": 10, "gped": 0, "pled": 0, "eyem": 0, "artf": 10, "bckg": 30}
    assert sum(record["patients_per_class_and_fold"]["spsw"]) == 10
    assert 0.4 < spsw["null"]["transfer"]["score"] < 0.6
    assert record["gates"]["time_alignment"] and record["gates"]["reader_agrees"] is None
    assert record["gates"]["displaced_units_at_chance"] and record["status"] == "pass"
    assert 0.48 <= record["displaced_units"]["mean_rank"] <= 0.52 and record["displaced_units"]["intervals"] == 3
    assert record["units"] == 350 and record["units_per_class"] == {
        "spsw": 60,
        "gped": 0,
        "pled": 0,
        "eyem": 0,
        "artf": 20,
        "bckg": 270,
    }
    assert set(record["confound"]) == set(pilot.EVENT_CLASSES)
    assert record["classes"]["gped"]["rank"]["transfer"]["score"] is None  # no unit of this class
    summary = pl.read_csv(paths["root"] / "pilot_summary.csv")
    assert summary.height == 3 * 5 * 3 and set(summary["read"].unique()) == set(pilot.READS)
    (paths["root"] / "reader_check.json").write_text(json.dumps({"pass": False, "largest_gap": 1.0}))
    assert pilot.read(paths["root"], n_boot=50)["status"] == "blocked"
