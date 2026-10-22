"""Temporal filter configuration and loader forwarding without EEG reads."""

from __future__ import annotations

from unittest.mock import Mock, sentinel

import pytest
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.data.components.window_dataset as window_dataset  # noqa: E402
import src.data.tuh_eeg_epilepsy_datamodule as data_module  # noqa: E402


@pytest.mark.parametrize("cutoffs", [[0.1, None], [None, 45], [1, 45]])
def test_temporal_filter_preserves_one_sided_and_bandpass_cutoffs(cutoffs: list[float | None]) -> None:
    """Nullable endpoints remain nullable instead of failing float conversion."""
    module = data_module.TUHEEGDataModule(filter_freq=cutoffs, notch_freqs=[60])

    assert module.filter_freq == cutoffs
    assert module.filter_freq is not cutoffs
    assert module.notch_freqs == [60.0]
    assert all(f is None or isinstance(f, float) for f in module.filter_freq)


@pytest.mark.parametrize("cutoffs", [None, [], [None, None]])
def test_disabled_temporal_filter_does_not_disable_notch(cutoffs: list[float | None] | None) -> None:
    """An omitted temporal filter retains the independent line-noise setting."""
    module = data_module.TUHEEGDataModule(filter_freq=cutoffs, notch_freqs=[60])

    assert module.filter_freq is None
    assert module.notch_freqs == [60.0]


def test_hydra_list_config_preserves_null_highpass_endpoint() -> None:
    """Hydra's list representation is converted to a plain MNE-compatible list."""
    config = OmegaConf.create({"filter_freq": [0.1, None], "notch_freqs": [60]})

    module = data_module.TUHEEGDataModule(filter_freq=config.filter_freq, notch_freqs=config.notch_freqs)

    assert module.filter_freq == [0.1, None]
    assert type(module.filter_freq) is list
    assert module.notch_freqs == [60.0]


@pytest.mark.parametrize(
    "cutoffs",
    [
        [0.1],
        [0.1, 45, 60],
        [-0.1, None],
        [0, None],
        [None, 0],
        [None, -45],
        [float("nan"), None],
        [None, float("inf")],
        [45, 1],
        [45, 45],
    ],
)
def test_invalid_filter_geometry_fails_before_setup(cutoffs: list[float | None]) -> None:
    """Malformed filters fail at configuration time rather than during EEG loading."""
    with pytest.raises(ValueError, match="filter_freq"):
        data_module.TUHEEGDataModule(filter_freq=cutoffs)


@pytest.mark.parametrize("cutoffs", [[0.1, None], [None, 45], [1, 45], None])
@pytest.mark.parametrize("lazy_loading", [False, True])
def test_filter_and_notch_reach_eager_and_lazy_loaders(
    monkeypatch: pytest.MonkeyPatch,
    cutoffs: list[float | None] | None,
    lazy_loading: bool,
) -> None:
    """Both data paths receive the same endpoints and independently configured notch."""
    engine = Mock(spec=["load_data"])
    monkeypatch.setattr(data_module, "TUHEEGEpilepsy", Mock(return_value=engine))
    lazy_loader = Mock(
        return_value={split: (sentinel.dataset, sentinel.metadata) for split in ("train", "val", "test")}
    )
    monkeypatch.setattr(window_dataset, "build_lazy_datasets", lazy_loader)
    labels = Mock(spec=["squeeze"])
    labels.squeeze.return_value = sentinel.labels
    engine.load_data.return_value = {
        split: (sentinel.samples, labels, sentinel.metadata) for split in ("train", "val", "test")
    }
    monkeypatch.setattr(data_module, "TensorDataset", Mock(return_value=sentinel.dataset))
    module = data_module.TUHEEGDataModule(filter_freq=cutoffs, notch_freqs=[60], lazy_loading=lazy_loading)

    module.setup()

    loader = lazy_loader if lazy_loading else engine.load_data
    unused_loader = engine.load_data if lazy_loading else lazy_loader
    loader.assert_called_once()
    unused_loader.assert_not_called()
    assert loader.call_args.kwargs["filter_freq"] == cutoffs
    assert loader.call_args.kwargs["notch_freqs"] == [60.0]
    assert module.data_train is sentinel.dataset
    assert module.data_val is sentinel.dataset
    assert module.data_test is sentinel.dataset
