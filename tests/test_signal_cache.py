"""Signal-cache identity, worker safety and patient-training equivalence; run on HPC."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rootutils
import torch
from torch import nn
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components import signal_cache  # noqa: E402
from src.data.components.window_dataset import WindowDataset  # noqa: E402
from src.models.components.patient_waveform import (  # noqa: E402
    PatientHead,
    SupervisedSpec,
    fit_patient_supervised,
    joint_backward,
)
from src.models.components.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec  # noqa: E402
from src.run_patient_waveform_cache_smoke import ValidationAudit, count_preprocessing  # noqa: E402


def _source(folder: Path, prefix: str = "s") -> WindowDataset:
    """Build real window metadata backed by tiny files; signal processing is mocked separately."""
    folder.mkdir(exist_ok=True)
    recording = folder / f"{prefix}.edf"
    if not recording.exists():
        recording.write_bytes(b"synthetic recording identity")
    plan = pd.DataFrame(
        {
            "subject": [prefix + "a"] * 5 + [prefix + "b", prefix + "c"],
            "epilepsy": [0] * 5 + [1, 1],
            "start": np.arange(7, dtype=float),
            "end": np.arange(7, dtype=float) + 1,
            "path": str(recording),
            "description_row": [{"path": str(recording), "montage": "test"}] * 7,
        }
    )
    return WindowDataset(
        plan,
        "bipolar",
        ["a", "b"],
        256.0,
        32,
        {"test": {"channels": ["a", "b"]}},
        [0.1, None],
        True,
        False,
        None,
        (),
        0.0,
        False,
        ["epilepsy"],
        notch_freqs=[60.0],
        interpolate_bad_channels=True,
    )


def _processed(source: WindowDataset, row: pd.Series) -> np.ndarray:
    """Deterministic stand-in for external EDF/filtering work, preserving the loader contract."""
    seed = int(row["start"]) + (13 if str(row["subject"]).startswith("s") else 29)
    return np.random.default_rng(seed).normal(size=(len(source.target_channels), source.target_len)).astype(np.float32)


def test_cache_reuses_exact_values_and_returns_independent_tensors(tmp_path: Path) -> None:
    """A new dataset instance reuses persisted values without aliasing returned tensors."""
    source = _source(tmp_path)
    with patch.object(WindowDataset, "_process", autospec=True, side_effect=_processed) as process:
        expected, label = source[0]
        source.enable_signal_cache(tmp_path / "cache")
        actual, cached_label = source[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.zero_()
        restored = _source(tmp_path)
        restored.enable_signal_cache(tmp_path / "cache")
        torch.testing.assert_close(restored[0][0], expected, rtol=0, atol=0)
        assert process.call_count == 2  # Uncached reference and the first cache miss only.
        assert torch.equal(label, cached_label)
        assert source.signal_cache is not None and restored.signal_cache is not None
        assert source.signal_cache.identity == restored.signal_cache.identity


def test_failures_are_retried_and_corrupt_entries_are_rebuilt(tmp_path: Path) -> None:
    """Transient EDF failures are not frozen into zero entries; damaged cache files recover."""
    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    assert source.signal_cache is not None
    expected = _processed(source, source.plan.iloc[0])
    with patch.object(WindowDataset, "_process", side_effect=[None, expected, expected]) as process:
        assert torch.count_nonzero(source[0][0]) == 0
        assert not list(source.signal_cache.directory.glob("*.npz"))
        np.testing.assert_array_equal(source[0][0].numpy(), expected)
        (source.signal_cache.directory / "00000000.npz").write_bytes(b"interrupted/corrupt archive")
        np.testing.assert_array_equal(source[0][0].numpy(), expected)
        np.testing.assert_array_equal(source[0][0].numpy(), expected)
        assert process.call_count == 3


def test_nonfinite_and_constant_processed_rows_are_preserved(tmp_path: Path) -> None:
    """Caching does not alter the model's invalid-channel/segment masks."""
    source = _source(tmp_path)
    data = np.zeros((2, 32), dtype=np.float32)
    data[0, 1] = np.nan
    source.enable_signal_cache(tmp_path / "cache")
    with patch.object(WindowDataset, "_process", return_value=data) as process:
        first, second = source[0][0], source[0][0]
        torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=True)
        assert process.call_count == 1


@pytest.mark.parametrize("change", ["filter", "notch", "channels", "length", "plan", "edf", "sidecar", "runtime"])
def test_cache_identity_changes_with_preprocessing_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Changes affecting the result select a separate cache namespace."""
    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    assert source.signal_cache is not None
    original = source.signal_cache.identity
    if change == "filter":
        source.filter_freq = [1.0, 40.0]
    elif change == "notch":
        source.notch_freqs = None
    elif change == "channels":
        source.target_channels.reverse()
    elif change == "length":
        source.target_len *= 2
    elif change == "plan":
        source.plan = source.plan.iloc[::-1].reset_index(drop=True)
    elif change == "edf":
        (tmp_path / "s.edf").write_bytes(b"changed recording")
    elif change == "sidecar":
        (tmp_path / "s-bads.json").write_text('{"bad_channels": ["a"]}')
    else:
        monkeypatch.setattr(signal_cache, "version", lambda name: "different-runtime")
    source.enable_signal_cache(tmp_path / "cache")
    assert source.signal_cache.identity != original


def test_source_changes_mid_run_are_refused(tmp_path: Path) -> None:
    """A new repair sidecar cannot silently leave a stale signal in use."""
    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    with patch.object(WindowDataset, "_process", autospec=True, side_effect=_processed) as process:
        source[0]
        (tmp_path / "s-bads.json").write_text("{}")
        with pytest.raises(RuntimeError, match="source changed"):
            source[0]
        assert process.call_count == 1


def test_write_failure_publishes_no_partial_cache(tmp_path: Path) -> None:
    """Failed writes fail the run and leave no apparently complete cache entry."""
    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    with (
        patch.object(WindowDataset, "_process", autospec=True, side_effect=_processed),
        patch.object(signal_cache.os, "fsync", side_effect=OSError("storage full")),
        pytest.raises(OSError, match="storage full"),
    ):
        source[0]
    assert source.signal_cache is not None
    assert not list(source.signal_cache.directory.glob("*.npz"))
    assert not list(source.signal_cache.directory.glob("*.tmp"))


def test_workers_share_cache_and_compute_each_window_once(tmp_path: Path) -> None:
    """Duplicate draws and replacement DataLoader workers share process-safe disk entries."""
    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    marker = tmp_path / "processed.txt"

    def process(dataset: WindowDataset, row: pd.Series) -> np.ndarray:
        with marker.open("a") as handle:
            handle.write(f"{int(row['start'])}\n")
        return _processed(dataset, row)

    with patch.object(WindowDataset, "_process", process):
        loader = DataLoader(source, batch_size=1, sampler=[0, 0, 1, 1], num_workers=2, multiprocessing_context="fork")
        first = [x for x, _ in loader]
        second = [x for x, _ in loader]  # New worker processes must still hit the disk cache.
    assert sorted(marker.read_text().splitlines()) == ["0", "1"]
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cached_training_preserves_draws_gradients_curves_and_selection(tmp_path: Path) -> None:
    """Cache use leaves the supervised experiment's numerical trajectory unchanged."""
    initial = WaveformSAE(
        WaveformSpec(n_atoms=1, atom_len=5, pool_len=4, top_r=1),
        WaveformTrainSpec(sampling_unit="patient", patients_per_batch=2, segments_per_patient=2, segment_batch=2),
        device="cpu",
    )
    initial.atoms = nn.Parameter(torch.zeros(1, 2, 5))
    with torch.no_grad():
        initial.atoms[0, 0, 2] = 1.0
    initial.fitted = True
    initial.provenance = {"train_subjects": ["sa", "sb", "sc"]}
    initial.fit_meta = {"train_spec": vars(initial.train_spec)}
    sources = _source(tmp_path, "s"), _source(tmp_path, "v")
    spec = SupervisedSpec(warmup_epochs=2, joint_epochs=2, calibration_batches=2, ramp_epochs=1)
    captures: list[list[torch.Tensor]] = []

    def capture(
        model: WaveformSAE, head: PatientHead, batch: tuple[torch.Tensor, ...], scales: tuple[float, float, float]
    ) -> tuple[float, float]:
        result = joint_backward(model, head, batch, scales)
        assert model.atoms is not None and model.atoms.grad is not None
        captures.append([batch[3].clone(), model.atoms.grad.clone(), *[p.grad.clone() for p in head.parameters()]])
        return result

    outcomes = []
    with patch.object(WindowDataset, "_process", autospec=True, side_effect=_processed) as process:
        for cached in (False, True):
            if cached:
                for source in sources:
                    source.enable_signal_cache(tmp_path / "cache")
            captures = []
            model = deepcopy(initial)
            output = tmp_path / str(cached)
            with patch("src.models.components.patient_waveform.joint_backward", side_effect=capture):
                head, metrics = fit_patient_supervised(model, tuple(DataLoader(s) for s in sources), spec, output)
            outcomes.append((model, head, metrics, captures))
            if cached:
                calls = process.call_count
                for source in sources:
                    for _ in DataLoader(source):
                        pass
                assert process.call_count == calls
    uncached_run, cached_run = outcomes
    assert uncached_run[2] == cached_run[2]
    torch.testing.assert_close(uncached_run[0].atoms, cached_run[0].atoms, rtol=0, atol=0)
    for key, value in uncached_run[1].state_dict().items():
        torch.testing.assert_close(value, cached_run[1].state_dict()[key], rtol=0, atol=0)
    for first, second in zip(uncached_run[3], cached_run[3], strict=True):
        for actual, expected in zip(first, second, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in ("patient_joint_training.csv", "patient_gradient_calibration.csv", "patient_warmup.csv"):
        assert (tmp_path / "False" / name).read_bytes() == (tmp_path / "True" / name).read_bytes()
    states = [torch.load(tmp_path / str(cached) / "patient_model.pt", weights_only=True) for cached in (False, True)]
    assert states[0]["joint_draws"] == states[1]["joint_draws"]


def test_cached_pretraining_keeps_validation_frequency_and_selected_dictionary(tmp_path: Path) -> None:
    """Full validation still runs every epoch; caching does not affect reconstruction selection."""
    sources = _source(tmp_path, "s"), _source(tmp_path, "v")
    outcomes = []
    with patch.object(WindowDataset, "_process", autospec=True, side_effect=_processed):
        for cached in (False, True):
            if cached:
                for source in sources:
                    source.enable_signal_cache(tmp_path / "cache")
            model = WaveformSAE(
                WaveformSpec(n_atoms=2, atom_len=5, pool_len=4),
                WaveformTrainSpec(
                    epochs=2,
                    sampling_unit="patient",
                    patients_per_batch=2,
                    segments_per_patient=2,
                    segment_batch=2,
                    init="patches",
                    recenter_atoms=False,
                ),
                device="cpu",
                random_state=7,
            )
            with patch("src.models.components.patient_waveform.log.info") as messages:
                model.fit_unsupervised(
                    *(DataLoader(s) for s in sources), provenance={"train_subjects": ["sa", "sb", "sc"]}
                )
            starts = [call for call in messages.call_args_list if call.args[0].startswith("{}: {} segments")]
            assert [call.args[1] for call in starts] == [
                "Reconstruction validation 1/2",
                "Reconstruction validation 2/2",
            ]
            assert all(call.args[2] == 7 for call in starts)
            finishes = [call for call in messages.call_args_list if "loader wait" in call.args[0]]
            assert len(finishes) == 2
            assert all(call.args[2] >= 0 and call.args[3] >= 0 and call.args[4] >= 0 for call in finishes)
            outcomes.append(model)
    first, second = outcomes
    torch.testing.assert_close(first.atoms, second.atoms, rtol=0, atol=0)
    assert first.history == second.history
    assert first.fit_meta["selected_epoch"] == second.fit_meta["selected_epoch"]
    assert first.fit_meta["pretraining_draws"] == second.fit_meta["pretraining_draws"]


@pytest.mark.parametrize("change", ["settings", "window"])
def test_attached_cache_refuses_mutated_dataset(tmp_path: Path, change: str) -> None:
    """An in-place configuration or plan edit must not serve stale values."""
    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    if change == "settings":
        source.filter_freq = [1.0, None]
    else:
        source.plan.loc[0, "start"] += 1.0
    with pytest.raises(RuntimeError, match="settings/window changed"):
        source[0]


def test_unsupported_signal_mode_is_refused(tmp_path: Path) -> None:
    """ICA needs additional dependency fingerprints before cache support can be enabled."""
    source = _source(tmp_path)
    source.mode = "ica_clean"
    with pytest.raises(ValueError, match="sensor preprocessing"):
        source.enable_signal_cache(tmp_path / "cache")


@pytest.mark.parametrize("workers", [0, 2])
def test_cache_smoke_counts_worker_misses_without_changing_training(tmp_path: Path, workers: int) -> None:
    """Diagnostic hooks preserve draws, objectives and selected atoms across worker counts."""
    from src.models.components import patient_waveform

    outcomes = []
    with patch.object(WindowDataset, "_process", _processed):
        for instrumented in (False, True):
            sources = _source(tmp_path, "s"), _source(tmp_path, "v")
            for source in sources:
                source.enable_signal_cache(tmp_path / str(instrumented))
            model = WaveformSAE(
                WaveformSpec(n_atoms=2, atom_len=5, pool_len=4),
                WaveformTrainSpec(
                    epochs=2,
                    sampling_unit="patient",
                    patients_per_batch=2,
                    segments_per_patient=2,
                    segment_batch=2,
                    init="patches",
                    recenter_atoms=False,
                ),
                device="cpu",
                random_state=7,
            )
            loaders = tuple(DataLoader(source, num_workers=workers) for source in sources)
            provenance = {"train_subjects": ["sa", "sb", "sc"]}
            if instrumented:
                cache = sources[1].signal_cache
                assert cache is not None
                with count_preprocessing(cache.identity, workers) as counter:
                    observer = ValidationAudit(sources[1], counter, tmp_path)
                    with patch.object(patient_waveform, "summarize_patients", observer.summarize):
                        model.fit_unsupervised(*loaders, provenance=provenance)
                    observer.finish()
                assert [row["preprocessing_calls"] for row in observer.records] == [7, 0]
                assert [row["cache_entries_before"] for row in observer.records] == [0, 7]
                assert all(row["cache_entries_after"] == 7 and row["verified"] for row in observer.records)
                assert all(row["validation_seconds"] > 0 for row in observer.records)
                assert (tmp_path / "cache_smoke_summary.json").is_file()
                assert (tmp_path / "validation_cache_comparison.csv").is_file()
            else:
                model.fit_unsupervised(*loaders, provenance=provenance)
            outcomes.append(model)
    first, second = outcomes
    torch.testing.assert_close(first.atoms, second.atoms, rtol=0, atol=0)
    assert first.history == second.history
    assert first.fit_meta["selected_epoch"] == second.fit_meta["selected_epoch"]
    assert first.fit_meta["pretraining_draws"] == second.fit_meta["pretraining_draws"]


@pytest.mark.parametrize("failure", ["prepopulated", "failed_load", "warm_rebuild"])
def test_cache_smoke_refuses_incomplete_or_rebuilt_cache(tmp_path: Path, failure: str) -> None:
    """A faster second pass alone cannot certify complete cache reuse."""
    from src.models.components.patient_waveform import PatientSummary

    source = _source(tmp_path)
    source.enable_signal_cache(tmp_path / "cache")
    assert source.signal_cache is not None
    loader = DataLoader(source)

    def process(dataset: WindowDataset, row: pd.Series) -> np.ndarray | None:
        return None if failure == "failed_load" and row["start"] == 0 else _processed(dataset, row)

    def evaluate(model: WaveformSAE, batches: DataLoader, description: str) -> PatientSummary:
        for _ in batches:
            pass
        return cast(PatientSummary, SimpleNamespace(counts=torch.tensor([len(source)]), skipped=0))

    with (
        patch.object(WindowDataset, "_process", process),
        count_preprocessing(source.signal_cache.identity, 0) as counter,
    ):
        observer = ValidationAudit(source, counter, tmp_path)
        observer.evaluate = evaluate
        epoch = 1
        if failure == "prepopulated":
            source[0]
        elif failure == "warm_rebuild":
            observer.summarize(cast(WaveformSAE, None), loader, "Reconstruction validation 1/2")
            (source.signal_cache.directory / "00000000.npz").write_bytes(b"corrupt archive")
            epoch = 2
        with pytest.raises(RuntimeError, match="cache audit failed"):
            observer.summarize(cast(WaveformSAE, None), loader, f"Reconstruction validation {epoch}/2")
        assert observer.records[-1]["verified"] is False
        with pytest.raises(RuntimeError, match="two verified"):
            observer.finish()
        assert not (tmp_path / "cache_smoke_summary.json").exists()


def test_cache_smoke_refuses_uninstrumented_worker_context() -> None:
    """Do not report zero preprocessing merely because spawn workers missed the hook."""
    with patch("src.run_patient_waveform_cache_smoke.mp.get_start_method", return_value="spawn"):
        with pytest.raises(RuntimeError, match="NUM_WORKERS=0"), count_preprocessing("unused", 2):
            pass
