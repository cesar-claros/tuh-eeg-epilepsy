"""Unit checks of the gate-1 synthetic driver ``src/separable_synthetic.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the
first failure. Covered: the montage topography and its phase reversal, the event schedule and
its draws, the planted amplitude in robust units after the row scaling, the waveform match and
its offset, the rank-1 factors of a format-5 atom, the detection counts, the readout of a
dictionary that holds the planted template, the recovery rules and stage outcomes of protocol
section 4.3 under both matching rules, and a small stage-1a run with its summary.
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


def test_readout_of_the_planted_template() -> None:
    view = gate.PlantedEvents(_background(4), np.arange(4), 6, seed=0, split="val", sfreq=SFREQ)
    model = SeparableSAE(spec=SeparableSpec(n_atoms=1, atom_len=256, pool_len=512, n_projections=2), device="cpu")
    offset = 89
    time = torch.zeros(1, 256)
    time[0, offset : offset + len(view.unit)] = torch.from_numpy(view.unit)
    channel = torch.from_numpy(view.topographies.T.copy()).float()[None]
    model.atoms = nn.Parameter(torch.cat([time, channel.flatten(1)], dim=1))
    record = gate.readout(model, view, batch_size=2, num_workers=0)
    assert abs(record["rho_w"] - 1.0) < 1e-6 and record["best_offset"] == offset
    assert record["rho_u_T3"] > 0.999 and record["rho_u_C4"] > 0.999
    assert record["n_events"] == 16 and record["recall"] >= 0.9, f"recall {record['recall']} of the exact template"


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


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
