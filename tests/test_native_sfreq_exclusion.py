"""Native-rate recording exclusions, including saved window plans, without EEG reads."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pandas as pd
import pytest
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.data.components.window_dataset as window_dataset  # noqa: E402
import src.data.tuh_eeg_epilepsy_datamodule as data_module  # noqa: E402
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.utils import split_provenance  # noqa: E402

EXCLUDED_RATES = [400.0, 512.0, 1000.0]


def _engine() -> TUHEEGEpilepsy:
    """Provide metadata for both classes at every rate, without scanning the corpus."""
    engine = object.__new__(TUHEEGEpilepsy)
    engine.descriptions = pd.DataFrame(
        [
            {
                "subject": subject,
                "path": Path(f"/corpus/{subject}/session/montage/{rate}.edf"),
                "sfreq": float(rate),
                "epilepsy": label,
            }
            for subject, label in (("s0", False), ("s1", True))
            for rate in (250, 256, 400, 512, 1000)
        ]
    )
    engine.exclude_paths = None
    return engine


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_excluded_rate_fails_before_setup(rate: float) -> None:
    """A native-rate policy cannot contain invalid frequencies."""
    with pytest.raises(ValueError, match="exclude_native_sfreqs"):
        data_module.TUHEEGDataModule(exclude_native_sfreqs=[rate])


@pytest.mark.parametrize("rates", [None, []])
def test_disabled_policy_does_not_require_rate_metadata(rates: list[float] | None) -> None:
    """Disabling the rate filter preserves explicit exclusions and older data paths."""
    engine = _engine()
    engine.descriptions = engine.descriptions.drop(columns="sfreq")
    engine.exclude_paths = {"manual.edf"}

    data_module.TUHEEGDataModule(exclude_native_sfreqs=rates)._apply_native_sfreq_exclusions(engine)

    assert engine.exclude_paths == {"manual.edf"}


def test_exclusions_use_native_rate_and_preserve_other_recordings() -> None:
    """The same native rates are removed in both classes, independently of the resampling target."""
    engine = _engine()
    original = engine.descriptions.copy(deep=True)
    manual_path = "/corpus/s0/session/montage/250.edf"
    engine.exclude_paths = {manual_path}
    module = data_module.TUHEEGDataModule(target_sfreq=512, exclude_native_sfreqs=[1000.0, 512.0, 400.0, 400.0])

    module._apply_native_sfreq_exclusions(engine)
    kept = TUHEEGEpilepsy._filter_excluded_paths(engine.descriptions, engine.exclude_paths)

    assert module.exclude_native_sfreqs == EXCLUDED_RATES
    assert engine.exclude_paths is not None and manual_path in engine.exclude_paths
    assert len(engine.exclude_paths) == 7
    assert set(kept["sfreq"]) == {250.0, 256.0}
    assert set(kept["subject"]) == {"s0", "s1"}
    assert kept.groupby("epilepsy").size().to_dict() == {False: 1, True: 2}
    pd.testing.assert_frame_equal(engine.descriptions, original)


@pytest.mark.parametrize("rate", [None, "unknown", 0.0, -1.0, float("inf")])
def test_unknown_native_rate_cannot_silently_bypass_policy(rate: object) -> None:
    """Invalid header metadata fails before any window selection."""
    engine = _engine()
    engine.descriptions["sfreq"] = engine.descriptions["sfreq"].astype(object)
    engine.descriptions.loc[0, "sfreq"] = rate
    module = data_module.TUHEEGDataModule(exclude_native_sfreqs=EXCLUDED_RATES)

    with pytest.raises(ValueError, match="native sfreq"):
        module._apply_native_sfreq_exclusions(engine)


def test_missing_native_rate_column_fails_clearly() -> None:
    """An active filter requires native metadata rather than assuming a target rate."""
    engine = _engine()
    engine.descriptions = engine.descriptions.drop(columns="sfreq")

    with pytest.raises(ValueError, match="native sfreq"):
        data_module.TUHEEGDataModule(exclude_native_sfreqs=EXCLUDED_RATES)._apply_native_sfreq_exclusions(engine)


@pytest.mark.parametrize("lazy_loading", [False, True])
def test_setup_applies_rate_policy_before_loading(monkeypatch: pytest.MonkeyPatch, lazy_loading: bool) -> None:
    """Both generated-window paths see the exclusions before planning or reading signals."""
    engine = _engine()
    monkeypatch.setattr(data_module, "TUHEEGEpilepsy", Mock(return_value=engine))
    labels = Mock(spec=["squeeze"])
    labels.squeeze.return_value = sentinel.labels
    eager_result: dict[str, tuple[object, ...]] = {
        split: (sentinel.samples, labels, sentinel.metadata) for split in ("train", "val", "test")
    }
    lazy_result: dict[str, tuple[object, ...]] = {
        split: (sentinel.dataset, sentinel.metadata) for split in ("train", "val", "test")
    }

    def check_exclusions(*_args: object, **_kwargs: object) -> dict[str, tuple[object, ...]]:
        assert engine.exclude_paths is not None and len(engine.exclude_paths) == 6
        return lazy_result if lazy_loading else eager_result

    eager_loader = Mock(side_effect=check_exclusions)
    lazy_loader = Mock(side_effect=check_exclusions)
    monkeypatch.setattr(engine, "load_data", eager_loader)
    monkeypatch.setattr(window_dataset, "build_lazy_datasets", lazy_loader)
    monkeypatch.setattr(data_module, "TensorDataset", Mock(return_value=sentinel.dataset))
    module = data_module.TUHEEGDataModule(lazy_loading=lazy_loading, exclude_native_sfreqs=EXCLUDED_RATES)

    module.setup()

    (lazy_loader if lazy_loading else eager_loader).assert_called_once()
    (eager_loader if lazy_loading else lazy_loader).assert_not_called()
    assert module.data_train is sentinel.dataset


def test_fixed_csvs_use_header_rates_and_keep_original_splits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Old path prefixes and resampled CSV rates do not bypass native-rate exclusions."""
    engine = _engine()
    engine.montages = {}
    engine.ica_keep_labels = ("brain", "other")
    engine.brain_ic_min_gof = 0.0
    engine.brain_ic_use_dipoles = True
    engine.interpolate_bad_channels = False
    engine.drop_bad_segments = False
    engine.drop_seizure_segments = False
    engine.apply_aas = False
    engine.aas_fmax = 2.5
    engine.skip_start_s = 0.0
    # Three disjoint patients, each with retained and excluded recordings.
    engine.descriptions["subject"] = ["train"] * 5 + ["val"] * 3 + ["test"] * 2
    engine.descriptions["sfreq"] = [250, 400, 512, 1000, 400, 256, 512, 1000, 256, 400]
    window_csvs = {}
    expected = {}
    for split, rows in engine.descriptions.groupby("subject"):
        manifest = rows[["path", "sfreq"]].copy()
        manifest["path"] = manifest["path"].map(lambda p: str(p).replace("/corpus/", "/old/root/"))
        manifest["sfreq"] = 256  # Deliberately a resampled rate; headers are authoritative.
        manifest["start"], manifest["end"] = 0.0, 120.0
        path = tmp_path / f"windows_{split}.csv"
        manifest.to_csv(path, index=False)
        window_csvs[split] = str(path)
        expected[split] = rows.loc[~rows["sfreq"].isin(EXCLUDED_RATES), "path"].tolist()
    data_module.TUHEEGDataModule(exclude_native_sfreqs=EXCLUDED_RATES)._apply_native_sfreq_exclusions(engine)
    monkeypatch.setattr(window_dataset, "_scan_common_channels", Mock(return_value=["C3", "C4"]))
    monkeypatch.setattr(window_dataset, "WindowDataset", Mock(return_value=sentinel.dataset))

    datasets = window_dataset._build_lazy_from_csv(
        engine,
        window_csvs,
        window_len_s=120.0,
        target_sfreq=256.0,
        mode="raw",
        filter_freq=None,
        target_name="epilepsy",
        pick_channels=None,
        rename_channels=True,
        set_montage=False,
        ic_bag_max_k=20,
        ic_bag_sign_normalize=True,
        ic_bag_rank_by="variance",
    )

    for split, (dataset, metadata) in datasets.items():
        assert dataset is sentinel.dataset
        assert metadata["path"].tolist() == expected[split]
        assert set(metadata["subject"]) == {split}
        assert metadata[["start", "end"]].values.tolist() == [[0.0, 120.0]]


def test_rate_policy_is_recorded_in_provenance() -> None:
    """Checkpoint lineage records the native-rate restriction alongside the effective plan."""
    frame = pd.DataFrame({"subject": ["s0"], "path": ["r.edf"], "start": [0.0], "end": [120.0]})
    module = SimpleNamespace(train_df=frame, val_df=frame, test_df=frame)
    config = OmegaConf.create({"exclude_native_sfreqs": EXCLUDED_RATES, "target_sfreq": 256})

    provenance = split_provenance(module, config)

    assert provenance["data"]["exclude_native_sfreqs"] == EXCLUDED_RATES
    assert provenance["data"]["target_sfreq"] == 256
