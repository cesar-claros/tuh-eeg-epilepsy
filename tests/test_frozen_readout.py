"""Checks of the frozen readout driver ``src/frozen_readout.py`` (no corpus; CPU; seconds).

Runs under pytest or as a script, which executes every ``test_*`` function and exits 1 on the first
failure. Covered: the cohort and its refusal of locked patients, the fixed and complete segment rows,
the pooled and band-limited views, peak tables that detect an exact template on AR(1) background,
the composed configuration, and heads, summary and review end to end on written features.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import src.frozen_readout as driver  # noqa: E402
import src.frozen_readout_core as core  # noqa: E402
import src.separable_synthetic as gate  # noqa: E402
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.models.components.waveform_sae import WaveformSpec  # noqa: E402

CHANNELS = [f"{a}-{c}" for a, c in TUHEEGEpilepsy._TCP_BIPOLAR]
SFREQ = 256.0


def _expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


class _Source(gate.ARBackground):
    """AR(1) segments with the plan columns that the driver reads."""

    def __init__(self, subjects: list[str], labels: list[int], rates: list[float], n_times: int = 30720) -> None:
        super().__init__(len(subjects), CHANNELS, n_times, SFREQ, seed=0, split="val")
        self.plan = pd.DataFrame(
            {
                "subject": subjects,
                "epilepsy": labels,
                "start": [float(i) for i in range(len(subjects))],
                "description_row": [
                    {"path": f"{s}_{i % 2}.edf", "sfreq": r} for i, (s, r) in enumerate(zip(subjects, rates))
                ],
            }
        )


def _sources() -> dict[str, _Source]:
    train = ["a"] * 12 + ["b"] * 10
    val = ["c"] * 11
    return {
        "train": _Source(train, [1] * 12 + [0] * 10, [256.0] * 9 + [250.0] * 3 + [250.0] * 10, n_times=1024),
        "val": _Source(val, [0] * 11, [250.0] * 11, n_times=1024),
    }


def test_cohort_reads_the_plans_and_refuses_locked_patients() -> None:
    sources = _sources()
    cohort = driver.build_cohort(sources, locked={"z"})
    assert cohort.labels == {"a": 1, "b": 0, "c": 0}
    assert [len(cohort.rows[s]) for s in "abc"] == [12, 10, 11] and cohort.recordings["a"] == 2
    assert cohort.rates["a"] == 256.0, "the native rate must be the patient's majority"
    assert all(name == "val" for name, _ in cohort.rows["c"]) and cohort.keys["a"] == sorted(cohort.keys["a"])
    _expect_value_error(lambda: driver.build_cohort(sources, locked={"a"}), "a locked patient was accepted")
    twice = {"train": sources["train"], "val": _Source(["a"], [1], [256.0], n_times=1024)}
    _expect_value_error(lambda: driver.build_cohort(twice, set()), "a patient in two sources was accepted")
    mixed = {"train": _Source(["a", "a"], [1, 0], [256.0, 256.0], n_times=1024)}
    _expect_value_error(lambda: driver.build_cohort(mixed, set()), "mixed labels were accepted")


def test_fixed_rows_are_stable_and_contiguous() -> None:
    cohort = driver.build_cohort(_sources(), set())
    rows = driver.fixed_rows(cohort, ["c", "a"], 9)
    assert rows == driver.fixed_rows(cohort, ["a", "c"], 9), "the row order must not depend on the input order"
    assert len(rows) == 18 and {name for name, _ in rows[:9]} == {"train"} and {name for name, _ in rows[9:]} == {"val"}
    assert len(driver.every_row(cohort, ["a", "b", "c"])) == 33


def test_views_align_the_plan_and_limit_the_band() -> None:
    sources = _sources()
    cohort = driver.build_cohort(sources, set())
    rows = driver.fixed_rows(cohort, ["a", "c"], 9)
    pooled = driver.PooledSegments(sources, rows)
    assert len(pooled) == 18 and pooled.plan["subject"].tolist() == ["a"] * 9 + ["c"] * 9
    assert pooled.plan["epilepsy"].tolist() == [1] * 9 + [0] * 9 and pooled.target_channels == CHANNELS
    view = driver.BandLimited(pooled, SFREQ)
    x, y = view[0]
    raw, y_raw = pooled[0]
    assert x.shape == raw.shape and x.dtype == raw.dtype and float(y) == float(y_raw)
    spectrum = lambda t: np.abs(np.fft.rfft(t.numpy()[0]))  # noqa: E731
    freqs = np.fft.rfftfreq(raw.shape[-1], 1.0 / SFREQ)
    assert spectrum(x)[freqs > 115].max() < 0.05 * spectrum(raw)[freqs > 115].max(), "the high band was not removed"
    assert view.plan is pooled.plan


class _Patients(torch.utils.data.Dataset):
    """Two segments per patient, with or without planted events."""

    def __init__(self, amplitude: float, n_patients: int = 3) -> None:
        source = gate.ARBackground(2 * n_patients, CHANNELS, 30720, SFREQ, seed=0, split="val")
        self.view = gate.PlantedEvents(source, np.arange(2 * n_patients), amplitude, seed=0, split="val", sfreq=SFREQ)
        self.target_channels = CHANNELS
        subjects = [f"p{i}" for i in range(n_patients) for _ in range(2)]
        self.plan = pd.DataFrame({"subject": subjects, "epilepsy": [0] * len(subjects)})

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, index: int):
        return self.view[index]


def test_peak_tables_detect_an_exact_template() -> None:
    spec = WaveformSpec(n_atoms=64, atom_len=256, pool_len=512, top_r=1)
    model, _ = gate.joint_oracle_model(spec, CHANNELS, "none", SFREQ, "cpu")
    plain = list(driver.stream_patients(model, _Patients(0), SFREQ, 0, details=True))
    planted = list(driver.stream_patients(model, _Patients(6), SFREQ, 0, details=True))
    assert [p.subject for p in plain] == ["p0", "p1", "p2"], "one record per patient, in plan order"
    for record in plain:
        assert record.projection.n_segments == 2 and record.projection.n_templates == 2
        assert record.atom.n_templates == 1 and record.r1.shape == (2, 1) and abs(record.minutes - 2.0) < 1e-12
        assert set(record.projection.segment.tolist()) <= {0, 1}, "segments must count within the patient"
    for template in range(2):
        counts = [
            sum(float(core.peak_counts(p.projection, 6.0)[:, template].sum()) for p in records)
            for records in (plain, planted)
        ]
        assert counts[1] > counts[0], f"template {template} did not see the planted events: {counts}"
    z, _ = core.choose_z([p.projection for p in plain], plain[0].minutes)
    assert z in core.Z_GRID


def test_composed_configuration_builds_both_dictionary_arms() -> None:
    import hydra

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = gate.stage_1b_config(root, root, 0, 40.0, seed=3, pre_emphasis="none", init="random")
        spec = hydra.utils.instantiate(cfg.feature.spec)
        train_spec = hydra.utils.instantiate(cfg.feature.train_spec)
        assert train_spec.init == "random" and train_spec.sampling_unit == "patient" and spec.top_r == 1
        for schedule in (train_spec, replace(train_spec, epochs=1, lr=0.0)):
            model = gate.build_model("s4ext", spec, schedule, 3, "cpu")
            assert model.spec.n_projections == driver.N_PROJECTIONS and model.spec.n_atoms == 64


def _write_root(root: Path, pass_index: int = 1) -> dict:
    """A root with features for every arm: signal in ``S4-proj``, a weak version in HYDRA, noise elsewhere."""
    rng = np.random.default_rng(0)
    subjects = [f"s{i:02d}" for i in range(50)]
    labels = {s: i % 2 for i, s in enumerate(subjects)}
    seed = core.PASS_SEEDS[pass_index]
    folds = core.assign_folds(labels, seed)
    plan = {"pass": pass_index, "seed": seed, "n_segments": 9, "folds": folds, "labels": labels}
    root.mkdir(parents=True, exist_ok=True)
    (root / "folds.json").write_text(json.dumps(plan))
    pd.DataFrame(
        {
            "subject": subjects,
            "label": [labels[s] for s in subjects],
            "fold": [folds[s] for s in subjects],
            "segments": [9 + 20 * labels[s] for s in subjects],
            "recordings": [1 + labels[s] for s in subjects],
            "native_rate": [256.0 if i % 4 < 2 else 250.0 for i in range(50)],
        }
    ).to_csv(root / "patients.csv", index=False)
    (root / "baselines.json").write_text(json.dumps({"segments_per_patient": 1.0}))
    nine = [s for s in subjects for _ in range(9)]
    many = [s for s in subjects for _ in range(9 + 20 * labels[s])]
    y = np.array([labels[s] for s in nine], dtype=float)
    y_many = np.array([labels[s] for s in many], dtype=float)
    in_nine = np.concatenate([np.arange(9 + 20 * labels[s]) < 9 for s in subjects])
    (root / "features").mkdir()

    def block(n: int, dim: int, signal: np.ndarray | None, strength: float) -> np.ndarray:
        x = np.abs(rng.normal(size=(n, dim))).astype(np.float32)
        if signal is not None:
            x[:, 0] += strength * signal.astype(np.float32)
        return x

    for fold in range(core.N_FOLDS):
        keys = np.array([f"train|{i}" for i in range(len(nine))])
        driver._save_features(
            driver.feature_path(root, "S4-proj", fold), nine, labels, block(len(nine), 256, y, 3.0), key=keys
        )
        for arm, dim in (("S4-atom", 64), ("S4-R1", 64)):
            driver._save_features(driver.feature_path(root, arm, fold), nine, labels, block(len(nine), dim, y, 2.0))
        driver._save_features(
            driver.feature_path(root, "S4-untrained", fold), nine, labels, block(len(nine), 256, None, 0)
        )
        driver._save_features(
            driver.feature_path(root, "S4-proj-all", fold),
            many,
            labels,
            block(len(many), 256, y_many, 3.0),
            in_nine=in_nine,
        )
        folder = driver.dictionary_dir(root, "trained", fold)
        folder.mkdir(parents=True)
        np.save(folder / "sae_time_atoms.npy", rng.normal(size=(64, 256)))
        np.save(folder / "sae_channel_atoms.npy", rng.normal(size=(64, 20, 4)))
        (folder / "sae_channels.txt").write_text("\n".join(f"c{i}" for i in range(20)))
        pd.DataFrame({"val_residual_frac": [0.9, 0.8]}).to_csv(folder / "sae_training.csv", index=False)
        untrained = driver.dictionary_dir(root, "untrained", fold)
        untrained.mkdir(parents=True)
        pd.DataFrame({"val_residual_frac": [0.99]}).to_csv(untrained / "sae_training.csv", index=False)
    driver._save_features(driver.feature_path(root, "Spectral", None), nine, labels, block(len(nine), 200, None, 0))
    driver._save_features(driver.feature_path(root, "HYDRA", None), nine, labels, block(len(nine), 300, y, 3.0))
    driver._save_features(
        driver.feature_path(root, "Spectral-all", None), many, labels, block(len(many), 200, None, 0), in_nine=in_nine
    )
    means = np.stack([block(1, 300, None, 0)[0] for _ in subjects])
    driver._save_features(driver.feature_path(root, "HYDRA-all", None), subjects, labels, means)
    (root / "positive_control.json").write_text(json.dumps({"passed": True}))
    return plan


def test_heads_summary_and_review_on_written_features() -> None:
    draws = driver.N_PERMUTATIONS
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pass1"
        plan = _write_root(root)
        driver.N_PERMUTATIONS = 2  # the null only needs to run here
        try:
            driver.run_heads(argparse.Namespace(root=root))
        finally:
            driver.N_PERMUTATIONS = draws
        for arm in (*driver.ARMS, "S4-sparse"):
            frame = pd.read_csv(root / "scores" / f"{arm}.csv")
            assert len(frame) == 50 and set(frame["fold"]) == set(range(core.N_FOLDS)), f"{arm} misses patients"
            assert all(frame["fold"] == frame["subject"].map(plan["folds"])), f"{arm} scored a patient out of fold"
        assert (root / "permutation_null.json").is_file() and len(
            json.loads((root / "coef/S4-sparse_templates.json").read_text())
        ) == 5
        decision = driver.summarize(root)
        assert decision["outcome"] == "confirm", decision
        assert decision["conditions"] == {"Spectral": True, "S4-untrained": True, "HYDRA": True}
        assert decision["gates"] == {"positive_control": True, "training": True}
        assert (root / "readout_summary.csv").is_file()
        table = pd.read_csv(root / "readout_summary.csv").set_index("arm")
        assert table.loc["S4-proj", "mean_fold_auroc"] > 0.9 and table.loc["Spectral", "mean_fold_auroc"] < 0.8
        assert "bag_link_epilepsy" in table.columns and "auroc_rate_250" in table.columns

        driver.review(root)
        templates = pd.read_csv(root / "review" / "templates.csv")
        assert len(templates) == 5 * driver.TOP_TEMPLATES and (templates["template"] == 0).any(), (
            "signal template absent"
        )
        assert set(json.loads((root / "review" / "recurrence.json").read_text())["recurring_templates"]) == set("01234")

        (root / "positive_control.json").write_text(json.dumps({"passed": False}))
        assert driver.summarize(root)["outcome"] == "blocked", "a failed control must block"
        (root / "positive_control.json").unlink()
        assert driver.summarize(root)["outcome"] == "incomplete", "a missing control must not pass"


def _write_scores(root: Path, plan: dict, arm: str, strength: float) -> None:
    rng = np.random.default_rng(len(arm))
    subjects = sorted(plan["labels"])
    labels = np.array([plan["labels"][s] for s in subjects])
    frame = pd.DataFrame(
        {
            "subject": subjects,
            "fold": [plan["folds"][s] for s in subjects],
            "label": labels,
            "score": strength * labels + rng.normal(size=len(subjects)),
        }
    )
    (root / "scores").mkdir(exist_ok=True)
    frame.to_csv(root / "scores" / f"{arm}.csv", index=False)


def test_second_pass_reads_review_or_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pass2"
        plan = _write_root(root, pass_index=2)
        for arm, strength in (("S4-proj", 6.0), ("Spectral", 0.0), ("S4-untrained", 0.0), ("HYDRA", 6.0)):
            _write_scores(root, plan, arm, strength)
        assert driver.summarize(root)["outcome"] == "review"
        primary = pd.read_csv(root / "scores" / "S4-proj.csv")
        primary["score"] += 1.0
        primary.to_csv(root / "scores" / "Spectral.csv", index=False)
        assert driver.summarize(root)["outcome"] == "stop", "a tie with Spectral must stop"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("all checks passed")
