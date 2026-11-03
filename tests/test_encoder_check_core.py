"""Checks of the pure logic of the frozen encoder check, ``src/encoder_check_core.py`` (numpy, scikit-learn; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first
failure. Covered: patch tiling and windows, average reference and both scales, the window draw, the choice
of added recordings and the refusal of locked patients, the heads, the split spread, the row rotation,
the age read, the gain with its ceiling, the gates and the outcomes of the rule.
"""

from __future__ import annotations

import inspect
import math
import re
import tempfile
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import src.encoder_check_core as core  # noqa: E402
from src.frozen_readout_core import ArmScores, auroc  # noqa: E402


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def test_electrodes_are_the_tcp_montage_without_the_ears() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "data" / "components" / "tuh_eeg_epilepsy.py"
    text = source.read_text()
    block = text[text.index("_TCP_BIPOLAR = (") : text.index("_TCP_BIPOLAR = (") + 700]
    block = block[: block.index("\n    )")]
    named = {name for pair in re.findall(r'\("(\w+)", "(\w+)"\)', block) for name in pair}
    assert named - {"A1", "A2"} == set(core.ELECTRODES), named ^ set(core.ELECTRODES)


def test_tiled_length_fits_the_patches() -> None:
    assert core.tiled_length(2000) == 2000, "10 s tile exactly: 11 patches"
    assert core.tiled_length(1000) == 920, "5 s crop to 5 patches"
    assert core.tiled_length(200) == 200
    _expect_value_error(lambda: core.tiled_length(199), "no patch fits")


def test_common_average_and_segment_scale() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 400)) * np.array([[1.0], [10.0], [100.0]]) + 5.0
    assert abs(core.common_average(x).mean(axis=0)).max() < 1e-12
    z = core.zscore_channels(x)
    assert abs(z.mean(axis=1)).max() < 1e-9 and abs(z.std(axis=1) - 1).max() < 1e-9, "the scale removes level"
    flat = core.zscore_channels(np.ones((2, 50)))
    assert (flat == 0).all(), "a flat channel must stay finite"
    spike = np.zeros((1, 1000))
    spike[0, 0] = 1e6
    assert core.zscore_channels(spike).max() == core.CLIP


def test_level_scale_keeps_the_level_of_a_patient() -> None:
    sums = np.array([[10.0, 0.0], [30.0, 0.0], [1000.0, 0.0]])
    squares = np.array([[60.0, 10.0], [100.0, 10.0], [1.0e6, 10.0]])
    counts = np.array([10.0, 10.0, 10.0])
    train = np.array([True, True, False])
    mean, sd = core.level_scale(sums, squares, counts, train)
    assert abs(mean[0] - 2.0) < 1e-12 and abs(sd[0] ** 2 - (160.0 / 20 - 4.0)) < 1e-12, "held-out patients stay out"
    x = np.array([[2.0, 4.0], [0.0, 0.0]])
    scaled = core.scale_level(x, mean, sd)
    assert scaled[0, 1] > scaled[0, 0], "a louder signal must stay louder"


def test_segment_windows_and_embedding() -> None:
    segment = np.zeros((17, core.SEGMENT_SAMPLES), dtype=np.float32)
    segment[0, core.WINDOW_SAMPLES] = 7.0
    windows = core.segment_windows(segment)
    assert windows.shape == (12, 17, 2000), windows.shape
    assert windows[1, 0, 0] == 7.0, "the second window must start at sample 2000"
    assert core.segment_windows(segment, core.PROBE_WINDOW_SAMPLES).shape == (24, 17, 920)
    seen = []

    def encode(batch: np.ndarray) -> np.ndarray:
        seen.append(batch.shape)
        return batch.mean(axis=(1, 2), keepdims=False)[:, None] * np.ones((len(batch), core.EMBED_DIM))

    embedding = core.embed_segment(segment, encode)
    assert seen == [(12, 17, 2000)] and embedding.shape == (core.EMBED_DIM,)
    assert abs(embedding[0] - 7.0 / (12 * 17 * 2000)) < 1e-9, "the embedding is a mean over windows"


def test_random_windows_are_stable_and_inside_the_segments() -> None:
    first = core.random_windows("aaaaaanr", 8)
    assert first == core.random_windows("aaaaaanr", 8) and first != core.random_windows("aaaaaebo", 8)
    assert len(first) == core.N_PROBE_WINDOWS and all(0 <= a < 8 for a, _ in first)
    assert all(0 <= b <= core.SEGMENT_SAMPLES - core.PROBE_WINDOW_SAMPLES for _, b in first)
    assert first == sorted(first), "a patient's segments are read once, in order"


def test_extra_recordings_choose_and_refuse() -> None:
    base = {"subject": "aaaaaaaa", "path": "/a", "n_seizure": 0, "sfreq": 250.0, "duration": 600.0}
    rows = [
        base,
        {**base, "path": "/b", "n_seizure": 2},
        {**base, "path": "/c", "sfreq": 512.0},
        {**base, "path": "/d", "n_seizure": 1, "sfreq": 1000.0},
        {**base, "path": "/e", "n_seizure": 1, "duration": 100.0},
        {**base, "path": "/f", "subject": "lockedpt", "n_seizure": 1},
        {**base, "path": "/g", "n_seizure": 1},
    ]
    chosen = core.extra_recordings(rows, {"aaaaaaaa"}, {"/g"}, {"lockedpt"})
    flags = {c["row"]["path"]: (c["seizure"], c["high_rate"]) for c in chosen}
    assert flags == {"/b": (True, False), "/c": (False, True), "/d": (True, True)}, flags
    _expect_value_error(lambda: core.extra_recordings(rows, {"lockedpt"}, set(), {"lockedpt"}), "a locked patient")


def test_unloaded_tensors_finds_what_the_file_did_not_fill() -> None:
    same = np.arange(4.0)
    first = {"encoder.a": same, "encoder.b": np.zeros(3), "final_layer.2.weight": np.zeros(2)}
    second = {"encoder.a": same.copy(), "encoder.b": np.ones(3), "final_layer.2.weight": np.ones(2)}
    assert core.unloaded_tensors(first, second) == ["encoder.b"], "the head is not used, so it does not count"
    assert core.unloaded_tensors(first, first) == []


def test_text_array_saves_and_loads_without_pickle() -> None:
    """The first extra read saved a pandas column as an object array, and the read could not load it."""
    column = pd.Series(["aaaaaaaa", "aaaaaaab"], dtype=object)
    assert column.astype(str).to_numpy().dtype == object, "this is the form that np.load refuses"
    values = core.text_array(column)
    assert values.dtype.kind == "U"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rows.npz"
        np.savez(path, subject=values)
        assert np.load(path)["subject"].tolist() == ["aaaaaaaa", "aaaaaaab"]


def test_segment_starts_follow_the_campaign_grid() -> None:
    assert core.segment_starts(40 + 2 * 120 + 5) == [40.0, 160.0]
    assert core.segment_starts(100.0) == []


def test_patient_rows_are_means() -> None:
    x = np.array([[1.0], [3.0], [10.0]])
    assert core.patient_rows(x, np.array(["a", "a", "b"]), np.array(["b", "a"])).tolist() == [[10.0], [2.0]]
    _expect_value_error(lambda: core.patient_rows(x, np.array(["a", "a", "a"]), np.array(["a", "b"])), "a lost patient")


def _signal(seed: int = 0, n: int = 156, strength: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (np.arange(n) < n // 2).astype(int)
    x = rng.standard_normal((n, 12))
    x[:, 0] += strength * y
    return x, y, np.arange(n) % 5


def test_the_heads_recover_a_planted_signal_and_ignore_noise() -> None:
    x, y, fold = _signal()
    assert auroc(y, core.fold_scores(x, y, fold, core.linear_scores, 0)) > 0.75
    mlp = core.fold_scores(x, y, fold, lambda *a, seed: core.mlp_scores(*a, seed=seed, max_epochs=60), 0)
    assert auroc(y, mlp) > 0.7, auroc(y, mlp)
    noise = np.random.default_rng(9).standard_normal(x.shape)
    assert abs(auroc(y, core.fold_scores(noise, y, fold, core.linear_scores, 0)) - 0.5) < 0.15


def test_the_helwan_probe_scores_each_held_out_patient_once() -> None:
    x, y, fold = _signal()
    rows = np.repeat(np.arange(len(y)), 3)
    subject = np.array([f"p{i:03d}" for i in rows])
    held = fold[rows] == 0
    patients, score = core.helwan_scores(x[rows], y[rows], subject, held)
    assert len(patients) == int((fold == 0).sum()) and ((score >= 0) & (score <= 1)).all()
    truth = np.array([y[int(s[1:])] for s in patients])
    assert auroc(truth, score) > 0.7, auroc(truth, score)


def test_split_spread_reports_a_distribution() -> None:
    x, y, _ = _signal()
    spread = core.split_spread(x, y, core.linear_scores, n_splits=30)
    assert spread["splits"] == 30 and spread["low"] < spread["mean"] < spread["high"] <= spread["highest"]
    assert 0.02 < spread["sd"] < 0.15, spread


def test_rotation_keeps_the_l2_head_on_wide_rows() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 400))
    y = (np.arange(60) < 30).astype(int)
    a, b = x[:45], x[45:]
    direct = LogisticRegression(C=0.1, max_iter=5000).fit(a, y[:45]).decision_function(b)
    rotated = core.rotate_rows(x)
    assert rotated.shape == (60, 60)
    other = LogisticRegression(C=0.1, max_iter=5000).fit(rotated[:45], y[:45]).decision_function(rotated[45:])
    assert np.corrcoef(direct, other)[0, 1] > 0.9999 and abs(direct - other).max() < 1e-2, "the same scores"


def _correlated_signal(seed: int = 0, n: int = 156, p: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Wide rows whose class signal sits in a shared factor, as in correlated band power."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) < n // 2).astype(int)
    factors = rng.standard_normal((n, 8))
    factors[:, 0] += y
    x = factors @ rng.standard_normal((8, p)) + 0.3 * rng.standard_normal((n, p))
    return (x - x.mean(axis=0)) / x.std(axis=0), y


def test_rotated_rows_read_as_the_rows_do_and_a_second_scale_would_break_them() -> None:
    """The fault of the first step 0: a second scale whitened the rotated rows and read near chance."""
    x, y = _correlated_signal()
    rotated = core.rotate_rows(x)
    direct = core.split_spread(x, y, core.linear_scores, n_splits=60)["mean"]
    fixed = core.split_spread(rotated, y, core.rotated_linear_scores, n_splits=60)["mean"]
    broken = core.split_spread(rotated, y, core.linear_scores, n_splits=60)["mean"]
    assert direct > 0.65 and abs(direct - fixed) < 0.04, (direct, fixed)
    assert broken < fixed - 0.1, (broken, fixed)
    assert core.spread_is_consistent(direct, fixed) and not core.spread_is_consistent(direct, broken)


def test_age_correlation_reads_age_and_rejects_noise() -> None:
    rng = np.random.default_rng(2)
    subjects = np.array([f"p{i:03d}" for i in range(156)])
    age = {s: float(a) for s, a in zip(subjects, rng.normal(50, 18, 156))}
    fold = {s: i % 5 for i, s in enumerate(subjects)}
    rows = np.repeat(subjects, 4)
    x = rng.standard_normal((len(rows), 20))
    assert abs(core.age_correlation(x, rows, age, fold)) < 0.25, "noise must not read age"
    x[:, 0] += np.array([age[s] for s in rows]) / 18.0
    assert core.age_correlation(x, rows, age, fold) > 0.7, "a feature with age must read it"


def _cohort() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = np.r_[np.ones(76, int), np.zeros(62, int)]
    fold = np.r_[np.arange(76) % 5, np.arange(62) % 5]
    status = np.array(["on"] * 66 + ["off"] * 10 + ["on"] * 4 + ["off"] * 58)
    return label, fold, status


def test_the_gain_has_the_ceiling_of_the_report() -> None:
    label, fold, status = _cohort()
    on = status == "on"
    assert abs(core.gain_ceiling(label, on, np.ones(len(label), bool)) - 0.5 * 844 / 4712) < 1e-12
    perfect = label.astype(float)
    value, low, high = core.drug_gain(perfect, label, fold, status, n_boot=200)
    assert abs(value - 0.5 * 844 / 4712) < 1e-6 and low <= value <= high, "a perfect score reaches the ceiling"


def test_the_gain_is_near_zero_for_noise_and_skips_unknown_status() -> None:
    label, fold, status = _cohort()
    rng = np.random.default_rng(4)
    value, low, high = core.drug_gain(rng.standard_normal(len(label)), label, fold, status, n_boot=500)
    assert abs(value) < 0.04 and low < 0 < high, (value, low, high)
    label2 = np.r_[label, [1, 0]]
    fold2 = np.r_[fold, [0, 0]]
    status2 = np.r_[status, ["unknown", "unknown"]]
    score = np.r_[label.astype(float), [-100.0, 100.0]]
    value2, _, _ = core.drug_gain(score, label2, fold2, status2, n_boot=50)
    assert value2 > 0.05, "patients of unknown status must not enter"


def test_alone_interval_reads_the_mean_fold_auroc() -> None:
    label, fold, _ = _cohort()
    subject = np.array([f"p{i:03d}" for i in range(len(label))])
    score = label + np.random.default_rng(5).standard_normal(len(label))
    arm = ArmScores.from_rows(subject, fold, label, score)
    value, low, high = core.alone_interval(arm, n_boot=300)
    assert low < value < high and 0.65 < value < 0.9, (value, low, high)


def test_gates_block_review_and_pass() -> None:
    good = {"permuted_mean": 0.5, "age_encoder": 0.6, "age_spectral": 0.6, "repeat_max": 0.0, "load_failures": 0.0}
    assert core.gate_report(**good, helwan_value=0.67)["status"] == "pass"
    assert core.gate_report(**good, helwan_value=0.80)["status"] == "review", "a miss of read H is not a fault"
    for key, value in (
        ("permuted_mean", 0.56),
        ("age_encoder", 0.49),
        ("repeat_max", 1e-3),
        ("load_failures", 0.02),
    ):
        report = core.gate_report(**{**good, key: value}, helwan_value=0.67)
        assert report["status"] == "blocked", (key, report)
    assert core.gate_report(**{**good, "age_encoder": 0.50}, helwan_value=0.67)["age_ok"], "the margin is inclusive"
    assert math.isclose(core.HELWAN_INTERVAL[0], 0.596)


def test_rule_gives_each_outcome() -> None:
    above, holds, below = (0.2, 0.1, 0.3), (0.02, -0.05, 0.09), (-0.1, -0.2, -0.02)
    gain_up, gain_flat = (0.05, 0.02, 0.08), (0.01, -0.02, 0.04)
    assert core.encoder_outcome(above, gain_up) == "encoder_adds"
    assert core.encoder_outcome(above, gain_flat) == "chart_reader"
    assert core.encoder_outcome(holds, gain_flat) == "no_advantage"
    assert core.encoder_outcome(below, gain_flat) == "no_advantage"
    assert core.encoder_outcome(holds, gain_up) == "review"
    assert core.encoder_outcome(below, gain_up) == "review"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(test):
            test()
    print("all encoder check core checks passed")
