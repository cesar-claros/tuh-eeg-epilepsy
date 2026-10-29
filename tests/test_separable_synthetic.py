"""Unit checks of the gate-1 synthetic driver ``src/separable_synthetic.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the
first failure. Covered: the montage topography and its phase reversal, the event schedule and
its draws, the planted amplitude in robust units after the row scaling, the waveform match and
its offset, the rank-1 factors of a format-5 atom, the detection counts, the readout of a
dictionary that holds the planted template, the recovery rules and stage outcomes of protocol
section 4.3 under both matching rules, the input figure, the threshold re-read of section 10.6 (its
helpers and an exact template), the oracle matched filter of section 10.9 (its template, model,
detection and rule), the readouts on differenced rows of section 10.11, the initialization rule of
section 10.12, and small stage-1a runs with their summary, also on differenced rows from a random
start.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import src.separable_synthetic as gate  # noqa: E402
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.shapeconv_sae import ShapeConvSAE  # noqa: E402
from src.models.components.waveform_sae import WaveformSAE, WaveformSpec  # noqa: E402

CHANNELS = [f"{a}-{c}" for a, c in TUHEEGEpilepsy._TCP_BIPOLAR]
SFREQ = 256.0


def _expect_value_error(function: Any, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def _background(n: int = 2, n_times: int = 30720) -> gate.ARBackground:
    return gate.ARBackground(n, CHANNELS, n_times, SFREQ, seed=0, split="val")


def test_focal_topography_has_the_phase_reversal() -> None:
    pattern = gate.focal_topography(CHANNELS, "T3")
    value = dict(zip(CHANNELS, pattern))
    assert abs(float(np.linalg.norm(pattern)) - 1.0) < 1e-12
    assert value["F7-T3"] > 0 > value["T3-T5"] and value["A1-T3"] > 0 > value["T3-C3"], "no phase reversal at T3"
    assert value["Fp2-F8"] == 0.0 and value["C4-P4"] == 0.0, "a distant derivation moved"
    without_ears = [c for c in CHANNELS if "A1" not in c and "A2" not in c]
    assert len(gate.focal_topography(without_ears, "C4")) == 20
    _expect_value_error(lambda: gate.focal_topography(CHANNELS, "Oz"), "an absent focus was accepted")


def test_event_schedule_rate_gap_foci_and_draws() -> None:
    length = round(gate.TEMPLATE_SECONDS * SFREQ)
    onsets, foci = gate.event_schedule(np.random.default_rng(0), 30720, SFREQ, length)
    assert len(onsets) == 4 and onsets[-1] + length <= 30720, "two events per minute over 120 s"
    assert bool((np.diff(onsets) >= gate.MIN_GAP_SECONDS * SFREQ).all()), "events closer than the gap"
    assert sorted(foci.tolist()) == [0, 0, 1, 1], "the foci are not balanced"
    view = gate.PlantedEvents(_background(3), np.arange(3), 6, seed=0, split="val", sfreq=SFREQ)
    first, again, other = view.events(0, 30720), view.events(0, 30720), view.events(1, 30720)
    assert np.array_equal(first[0], again[0]) and not np.array_equal(first[0], other[0])


def test_planted_amplitude_in_robust_units() -> None:
    view = gate.PlantedEvents(_background(1), np.arange(1), 6, seed=0, split="val", sfreq=SFREQ)
    background, _ = view.source[0]
    constant = CHANNELS.index("C3-Cz")  # a derivation that both foci move
    background[constant] = 0.0
    onsets, foci = view.events(0, background.shape[-1])
    planted = gate.plant(background, onsets, foci, 6.0, view.shape, view.topographies)
    assert torch.equal(planted[constant], background[constant]), "a constant row received an event"
    assert torch.equal(gate.plant(background, onsets, foci, 0.0, view.shape, view.topographies), background)
    difference = ShapeConvSAE._robust_scale(planted) - ShapeConvSAE._robust_scale(background)
    strongest = int(np.abs(view.topographies[foci[0]]).argmax())
    peak = float(difference[strongest, onsets[0] : onsets[0] + len(view.shape)].abs().max())
    assert abs(peak - 6.0) < 0.5, f"the encoded peak is {peak:.2f} robust units, not 6"


def test_waveform_match_and_offset() -> None:
    unit, _ = gate.event_template(SFREQ)
    waveform = np.zeros((2, 256))
    waveform[0, 50 : 50 + len(unit)] = -unit
    waveform[1] = np.random.default_rng(1).standard_normal(256)
    rho, offset = gate.waveform_match(waveform / np.linalg.norm(waveform, axis=1, keepdims=True), unit)
    assert abs(rho[0] - 1.0) < 1e-9 and offset[0] == 50, "an embedded template is not found"
    assert rho[1] < 0.5


def test_factors_of_a_rank_one_format5_atom() -> None:
    model = WaveformSAE(spec=WaveformSpec(n_atoms=1, atom_len=16, pool_len=16), device="cpu")
    topography = torch.tensor([3.0, -4.0, 0.0], dtype=torch.float64) / 5.0
    waveform = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64).sin()
    model.atoms = nn.Parameter((topography[:, None] * waveform[None, :])[None])
    time, channel, share = gate.factors(model)
    assert abs(abs(float(time[0] @ waveform.numpy())) / float(waveform.norm()) - 1.0) < 1e-9
    assert abs(abs(float(channel[0, :, 0] @ topography.numpy())) - 1.0) < 1e-9
    assert share is not None and abs(float(share[0]) - 1.0) < 1e-9


def test_detection_counts() -> None:
    recall, precision, events, activations = gate.detection([np.array([100, 500, 900])], [np.array([102, 700])], 13)
    assert (recall, events, activations) == (0.5, 2, 3) and abs(precision - 1 / 3) < 1e-12
    recall, precision, _, _ = gate.detection([np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)], 13)
    assert np.isnan(recall) and np.isnan(precision)


def _template_model(view: gate.PlantedEvents, offset: int = 89) -> SeparableSAE:
    """One separable atom that holds the planted waveform at ``offset`` and both topographies."""
    model = SeparableSAE(spec=SeparableSpec(n_atoms=1, atom_len=256, pool_len=512, n_projections=2), device="cpu")
    time = torch.zeros(1, 256)
    time[0, offset : offset + len(view.unit)] = torch.from_numpy(view.unit)
    channel = torch.from_numpy(view.topographies.T.copy()).float()[None]
    model.atoms = nn.Parameter(model._constrain(model._pack(time, channel)))
    return model


def test_readout_of_the_planted_template() -> None:
    view = gate.PlantedEvents(_background(4), np.arange(4), 6, seed=0, split="val", sfreq=SFREQ)
    offset = 89
    model = _template_model(view, offset)
    record = gate.readout(model, view, batch_size=2, num_workers=0)
    assert abs(record["rho_w"] - 1.0) < 1e-6 and record["best_offset"] == offset
    assert record["rho_u_T3"] > 0.999 and record["rho_u_C4"] > 0.999
    assert record["n_events"] == 16 and record["recall"] >= 0.9, f"recall {record['recall']} of the exact template"


def test_threshold_helpers() -> None:
    responses = torch.tensor([[0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 0.0, 5.0]])
    assert torch.nonzero(gate.local_maxima(responses, 1)[0]).flatten().tolist() == [1, 3, 7]
    heights = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    assert gate.calibrated_threshold(heights, minutes=2.0, rate=1.0) == 3.0, "two peaks must stay above"
    assert gate.calibrated_threshold(heights, minutes=2.0, rate=10.0) == 0.0
    assert gate.patient_halves(np.array(["b", "a", "b", "c"])).tolist() == [False, True, False, True]
    assert gate.json_safe({"a": float("nan"), "b": {"c": [1.0, float("nan")]}}) == {"a": None, "b": {"c": [1.0, None]}}


def test_threshold_readout_of_the_planted_template() -> None:
    background = _background(4)
    calibration = gate.PlantedEvents(background, np.arange(2), 0, seed=0, split="val", sfreq=SFREQ)
    evaluation = gate.PlantedEvents(background, np.arange(2, 4), 6, seed=0, split="val", sfreq=SFREQ)
    loader = {"batch_size": 2, "num_workers": 0}
    record = gate.threshold_readout(_template_model(evaluation), calibration, evaluation, {"T3": 0, "C4": 0}, loader)
    assert abs(record["calibration_minutes"] - 4.0) < 1e-9 and record["evaluation_segments"] == 2
    assert record["T3_recall"] >= 0.9 and record["C4_recall"] >= 0.9, "the exact template missed its events"
    thresholds = [record["T3_curve"][str(rate)]["threshold"] for rate in gate.THRESHOLD_FA_CURVE]
    assert thresholds == sorted(thresholds, reverse=True), "a higher false-alarm rate raised the threshold"
    point = record["C4_curve"][str(gate.THRESHOLD_FA_PER_MINUTE)]
    assert abs(record["evaluation_minutes"] - 4.0) < 1e-9 and point["n_events"] == 4
    hits = round(point["precision"] * point["n_detections"])
    assert abs(point["evaluation_fa_per_minute"] - (point["n_detections"] - hits) / 4.0) < 1e-12


def test_oracle_template_and_model() -> None:
    unit, _ = gate.event_template(SFREQ)
    differenced = gate.row_template(SFREQ, "diff")
    assert np.allclose(gate.row_template(SFREQ, "none"), unit) and len(differenced) == len(unit) + 1
    signal = np.zeros(400)
    signal[100 : 100 + len(unit)] = unit
    rows = np.diff(signal, prepend=signal[0])  # the row pipeline's first difference
    assert np.allclose(rows[100 : 100 + len(differenced)] / np.linalg.norm(rows), differenced)
    assert not rows[:100].any() and not rows[100 + len(differenced) :].any(), "the differenced event moved"
    spec = WaveformSpec(n_atoms=64, atom_len=256, pool_len=512, top_r=1)
    for domain in gate.ORACLE_DOMAINS:
        model, offset = gate.oracle_model(spec, CHANNELS, domain, SFREQ, "cpu")
        template = gate.row_template(SFREQ, domain)
        time, channel, _ = gate.factors(model)
        assert offset == 89 and model.spec.n_atoms == 2 and model.spec.pre_emphasis == domain
        assert np.allclose(time[:, offset : offset + len(template)], template, atol=1e-6)
        for f, focus in enumerate(gate.FOCI):
            assert abs(float(channel[f, :, 0] @ gate.focal_topography(CHANNELS, focus)) - 1.0) < 1e-5


def test_readout_in_the_difference_domain() -> None:
    view = gate.PlantedEvents(_background(4), np.arange(4), 6, seed=0, split="val", sfreq=SFREQ)
    spec = WaveformSpec(n_atoms=64, atom_len=256, pool_len=512, top_r=1)
    model, offset = gate.oracle_model(spec, CHANNELS, "diff", SFREQ, "cpu")
    record = gate.readout(model, view, batch_size=2, num_workers=0)
    assert abs(record["T3_rho_w"] - 1.0) < 1e-5 and abs(record["C4_rho_w"] - 1.0) < 1e-5, "no row-domain match"
    assert record["best_offset"] == offset and (record["T3_atom"], record["C4_atom"]) == (0, 1)
    plain, _ = gate.waveform_match(gate.factors(model)[0], view.unit)
    assert plain.max() < 0.9, "the plain template also matches a differenced atom"


def test_arm_pre_emphasis_reads_the_saved_config() -> None:
    from omegaconf import OmegaConf

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        OmegaConf.save(OmegaConf.create({"feature": {"spec": {"pre_emphasis": "diff"}}}), folder / "config.yaml")
        assert gate.arm_pre_emphasis(folder) == "diff"


def test_oracle_detects_on_the_ar_background() -> None:
    background = _background(4)
    calibration = gate.PlantedEvents(background, np.arange(2), 0, seed=0, split="val", sfreq=SFREQ)
    evaluation = gate.PlantedEvents(background, np.arange(2, 4), 6, seed=0, split="val", sfreq=SFREQ)
    spec = WaveformSpec(n_atoms=64, atom_len=256, pool_len=512, top_r=1)
    loader = {"batch_size": 2, "num_workers": 0}
    for domain in gate.ORACLE_DOMAINS:
        model, offset = gate.oracle_model(spec, CHANNELS, domain, SFREQ, "cpu")
        offsets = dict.fromkeys((0, 1), offset)
        record = gate.threshold_readout(model, calibration, evaluation, {"T3": 0, "C4": 1}, loader, offsets)
        for focus in gate.FOCI:
            assert record[f"{focus}_recall"] >= 0.9, f"the {domain} oracle missed its {focus} events"


def _oracle_record(domain: str, amplitude: int, seed: int, precision: float | None) -> dict[str, Any]:
    point = {"threshold": 1.0, "recall": 0.95, "precision": precision, "n_events": 4, "n_detections": 4}
    curve = {str(rate): {**point, "evaluation_fa_per_minute": 0.5} for rate in gate.THRESHOLD_FA_CURVE}
    focus = {
        f"{f}_{k}": v for f in gate.FOCI for k, v in (("recall", 0.95), ("precision", precision), ("curve", curve))
    }
    return {"domain": domain, "amplitude": amplitude, "seed": seed, **focus}


def test_oracle_rule_and_summary() -> None:
    cases = (
        ({"none": set(), "diff": set()}, "task_limit"),
        ({"none": set(), "diff": {6}}, "input_limit"),
        ({"none": {6}, "diff": set()}, "atoms_limit"),
        ({"none": {3}, "diff": {3}}, "task_limit"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for passing, expected in cases:
            for domain in gate.ORACLE_DOMAINS:
                for amplitude in gate.AMPLITUDES[1:]:
                    for seed in gate.SEEDS:
                        precision = 0.6 if amplitude in passing[domain] else 0.3
                        record = _oracle_record(domain, amplitude, seed, precision)
                        (root / f"oracle_{domain}_a{amplitude}_s{seed}.json").write_text(json.dumps(record))
            assert gate.summarize_oracle(root)["outcome"] == expected, f"{passing} did not give {expected}"
        assert (root / "oracle_summary.csv").is_file() and (root / "oracle_decision.json").is_file()
        (root / "oracle_diff_a3_s1.json").unlink()
        assert gate.summarize_oracle(root)["outcome"] == "incomplete"
    one_seed = [_oracle_record("none", 6, 0, 0.6), _oracle_record("none", 6, 1, None)]
    assert gate.oracle_passes(one_seed, "none")[6] is False, "a seed without detections passed"


def _row(stage: str, arm: str, amplitude: int, seed: int, passed: bool) -> dict[str, Any]:
    good = {"rho_w": 0.95, "recall": 0.9, "precision": 0.6, "rho_u_T3": 0.9, "rho_u_C4": 0.2}
    bad = {"rho_w": 0.4, "recall": 0.1, "precision": 0.1, "rho_u_T3": 0.3, "rho_u_C4": 0.2}
    metrics = good if passed else bad
    if amplitude == 0:
        metrics = {**bad, "rho_w": 0.5, "recall": float("nan"), "precision": float("nan")}
    focus = {
        f"{f}_{k}": metrics[k if k != "rho_u" else f"rho_u_{f}"]
        for f in gate.FOCI
        for k in ("rho_w", "recall", "precision", "rho_u")
    }
    return {"stage": stage, "arm": arm, "amplitude": amplitude, "seed": seed, **metrics, **focus}


def _rows(stage: str, recovered: dict[str, set[int]]) -> list[dict[str, Any]]:
    return [
        _row(stage, arm, a, s, a in recovered[arm])
        for arm in ("f5", "s4ext")
        for a in gate.AMPLITUDES
        for s in gate.SEEDS
    ]


def _outcome(stage: str, rows: list[dict[str, Any]]) -> str:
    return gate.stage_outcome(stage, {arm: gate.recovery(rows, stage, arm) for arm in ("f5", "s4ext")})


def test_recovery_rules_and_stage_outcomes() -> None:
    cases = (
        ({"f5": set(), "s4ext": set()}, "objective_mismatch"),
        ({"f5": set(), "s4ext": {3, 6}}, "separable_helps"),
        ({"f5": {6}, "s4ext": {6}}, "no_separation"),
        ({"f5": {6}, "s4ext": set()}, "f5_only"),
        ({"f5": {3}, "s4ext": {6}}, "mixed"),
    )
    for recovered, expected in cases:
        assert _outcome("1b", _rows("1b", recovered)) == expected, f"{recovered} did not give {expected}"
    assert _outcome("1a", _rows("1a", {"f5": set(), "s4ext": set()})) == "fix_setup"
    assert _outcome("1a", _rows("1a", {"f5": {3}, "s4ext": set()})) == "proceed_to_1b"
    assert _outcome("1b", _rows("1b", {"f5": {6}, "s4ext": {6}})[1:]) == "incomplete"
    rows = _rows("1b", {"f5": {6}, "s4ext": {6}})
    for row in rows:
        if row["amplitude"] == 0:
            row["rho_w"] = 0.9  # the trained arms then gain only 0.05 over the event-free control
    assert _outcome("1b", rows) == "objective_mismatch", "the gain over the event-free control was ignored"


def test_per_focus_matching_judges_each_focus_on_its_own_events() -> None:
    rows = _rows("1b", {"f5": {6}, "s4ext": {6}})
    for row in rows:
        if row["arm"] == "f5" and row["amplitude"] == 6:
            row["recall"] = 0.5  # one atom per focus: the best atom detects half of all events
    assert gate.PER_FOCUS_MATCHING is True, "the amendment of 2026-10-28 is not in force"
    assert gate.recovery(rows, "1b", "f5")[6] is True, "the focus atoms were not judged on their own events"
    try:
        gate.PER_FOCUS_MATCHING = False
        assert gate.recovery(rows, "1b", "f5")[6] is False, "the single-atom rule ignored the low recall"
    finally:
        gate.PER_FOCUS_MATCHING = True


def test_plot_input_writes_a_figure() -> None:
    settings = gate.Stage1aSettings(n_times=7680)
    with tempfile.TemporaryDirectory() as tmp:
        path = gate.plot_input(Path(tmp) / "input.png", seed=0, index=1, settings=settings)
        assert path.is_file() and path.stat().st_size > 10_000, "no figure was written"


def _write_root(
    root: Path, recovered: dict[str, set[int]], provenance: dict[str, Any], seeds: tuple[int, ...] = gate.SEEDS
) -> None:
    for row in _rows("1a", recovered):
        row = {**row, "seed": seeds[gate.SEEDS.index(row["seed"])]}
        folder = root / f"1a_{row['arm']}_a{row['amplitude']}_s{row['seed']}"
        folder.mkdir(parents=True)
        (folder / "synthetic_readout.json").write_text(json.dumps(row))
        (folder / "synthetic_provenance.json").write_text(json.dumps(provenance))
        (folder / "sae_training.csv").write_text("epoch,n_atoms_never_won,median_coherence\n1,5,0.9\n2,3,0.5\n")


def test_initialization_rule_of_sections_10_12_and_10_13() -> None:
    cases = (
        ({"f5": {6}, "s4ext": {3, 6}}, {"f5": {6}, "s4ext": {6}}, ["s4ext_a3"], [], True),
        ({"f5": {6}, "s4ext": set()}, {"f5": {6}, "s4ext": set()}, [], [], False),
        ({"f5": set(), "s4ext": set()}, {"f5": {6}, "s4ext": set()}, [], ["f5_a6"], False),
    )
    for random_recovered, patch_recovered, gains, losses, run_1b in cases:
        with tempfile.TemporaryDirectory() as tmp:
            random_root, patch_root = Path(tmp) / "random", Path(tmp) / "patches"
            _write_root(random_root, random_recovered, {"init": "random", "pre_emphasis": "none"})
            _write_root(patch_root, patch_recovered, {})  # arms written before section 10.12
            decision = gate.compare_initializations(random_root, patch_root)
            assert (decision["gains"], decision["losses"], decision["gain"]) == (gains, losses, run_1b)
            assert decision["median_atoms_never_won"] == {"random": 3.0, "patches": 3.0}
            assert decision["median_coherence"] == {"random": 0.5, "patches": 0.5}
            expected_cells = 2 * sum(len(v) for v in random_recovered.values())
            assert decision["event_atom_cells"]["random"] == expected_cells
            assert (random_root / "init_decision.json").is_file()
    with tempfile.TemporaryDirectory() as tmp:
        random_root, patch_root = Path(tmp) / "random", Path(tmp) / "patches"
        _write_root(random_root, {"f5": {6}, "s4ext": set()}, {"init": "random", "pre_emphasis": "diff"})
        _write_root(patch_root, {"f5": {6}, "s4ext": set()}, {"init": "patches", "pre_emphasis": "none"})
        _expect_value_error(lambda: gate.compare_initializations(random_root, patch_root), "domains were mixed")
        _expect_value_error(lambda: gate.compare_initializations(random_root, random_root), "two random roots passed")
        short = Path(tmp) / "patches_5_epochs"
        _write_root(short, {"f5": {6}, "s4ext": set()}, {"init": "patches", "pre_emphasis": "diff"})
        for path in short.glob("*/synthetic_readout.json"):
            path.write_text(json.dumps({**json.loads(path.read_text()), "epochs": 5}))
        _expect_value_error(lambda: gate.compare_initializations(random_root, short), "epoch counts were mixed")
        for folder in random_root.glob("1a_s4ext_a3_s1"):
            (folder / "synthetic_readout.json").unlink()
        _write_root(
            Path(tmp) / "patches_diff", {"f5": {6}, "s4ext": set()}, {"init": "patches", "pre_emphasis": "diff"}
        )
        decision = gate.compare_initializations(random_root, Path(tmp) / "patches_diff")
        assert decision["gain"] is None, "a missing arm did not leave the rule open"
    with tempfile.TemporaryDirectory() as tmp:
        random_root, patch_root = Path(tmp) / "random", Path(tmp) / "patches"
        settings = {"init": "random", "pre_emphasis": "none", "reseed": True}
        _write_root(random_root, {"f5": {6}, "s4ext": {3}}, settings)
        _write_root(random_root, {"f5": {6}, "s4ext": {3}}, settings, seeds=(2, 3))
        _write_root(patch_root, {"f5": {6}, "s4ext": set()}, {})
        _write_root(patch_root, {"f5": {6}, "s4ext": set()}, {}, seeds=(2, 3))
        decision = gate.compare_initializations(random_root, patch_root, seeds=(2, 3))
        assert decision["gain"] is True and decision["gains"] == ["s4ext_a3"] and decision["random_reseeds"]
        assert (random_root / "init_decision_s2_s3.json").is_file(), "the replication overwrote the first decision"
        assert decision["event_atom_cells"]["random"] == 4, "cells of other seeds were counted"
        reseeding = Path(tmp) / "patches_reseed"
        _write_root(reseeding, {"f5": {6}, "s4ext": set()}, {"init": "patches", "reseed": True})
        _expect_value_error(
            lambda: gate.compare_initializations(random_root, reseeding), "a reseeding patch root passed"
        )


def test_stage_1a_run_and_summary() -> None:
    settings = gate.Stage1aSettings(n_train=4, n_val=2, n_atoms=2, atom_len=32, pool_len=64, n_times=7680)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for arm in ("f5", "s4ext"):
            output = root / f"1a_{arm}_a6_s0"
            record = gate.fit_stage_1a(output, epochs=1, device="cpu", settings=settings)
            assert record["n_events"] == 2 and record["epochs"] == 1 and 0.0 <= record["rho_w"] <= 1.0
            for name in ("synthetic_readout.json", "synthetic_provenance.json", "sae_state.pt"):
                assert (output / name).is_file(), f"{arm} wrote no {name}"
        decision = gate.summarize(root)
        assert decision["1a"]["outcome"] == "incomplete" and (root / "synthetic_summary.csv").is_file()
        assert json.loads((root / "synthetic_decision.json").read_text())["per_focus_matching"] is True
        _expect_value_error(lambda: gate.parse_arm("1a_f5_a4_s0"), "an amplitude outside the gate was accepted")
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "1a_s4ext_a6_s0"
        record = gate.fit_stage_1a(
            output, epochs=2, device="cpu", settings=settings, pre_emphasis="diff", init="random", reseed=True
        )
        assert record["n_events"] == 2 and 0.0 <= record["rho_w"] <= 1.0
        provenance = json.loads((output / "synthetic_provenance.json").read_text())
        assert (provenance["pre_emphasis"], provenance["init"], provenance["reseed"]) == ("diff", "random", True)
        assert "n_reseeded" in (output / "sae_training.csv").read_text().splitlines()[0]
        saved = torch.load(output / "sae_state.pt", map_location="cpu", weights_only=False)
        assert saved["spec"]["pre_emphasis"] == "diff", "the checkpoint does not record the differencing"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
