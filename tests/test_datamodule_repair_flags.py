"""Repair and exclusion flags need the lazy loader; checked without EEG reads."""

from __future__ import annotations

from unittest.mock import Mock, sentinel

import pytest
import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.data.components.window_dataset as window_dataset  # noqa: E402
import src.data.tuh_eeg_epilepsy_datamodule as data_module  # noqa: E402

FLAGS = ("interpolate_bad_channels", "drop_bad_segments", "drop_seizure_segments", "apply_aas")


@pytest.mark.parametrize("flag", FLAGS)
def test_repair_flag_without_lazy_loading_fails_at_configuration(flag: str) -> None:
    """The eager loader ignores the flag, so the datamodule refuses it before any setup."""
    with pytest.raises(ValueError, match=rf"data\.lazy_loading=true to use data\.{flag};"):
        data_module.TUHEEGDataModule(lazy_loading=False, **{flag: True})


def test_every_refused_flag_is_named() -> None:
    """One error lists all the flags that the eager loader would ignore."""
    with pytest.raises(ValueError) as error:
        data_module.TUHEEGDataModule(**dict.fromkeys(FLAGS, True))
    assert all(f"data.{flag}" in str(error.value) for flag in FLAGS)


@pytest.mark.parametrize("flag", FLAGS)
def test_repair_flag_with_lazy_loading_reaches_the_engine(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    """With lazy loading the flag is accepted and passed to the engine that builds the lazy datasets."""
    engine_class = Mock(return_value=sentinel.engine)
    monkeypatch.setattr(data_module, "TUHEEGEpilepsy", engine_class)
    lazy_loader = Mock(
        return_value={split: (sentinel.dataset, sentinel.metadata) for split in ("train", "val", "test")}
    )
    monkeypatch.setattr(window_dataset, "build_lazy_datasets", lazy_loader)
    module = data_module.TUHEEGDataModule(lazy_loading=True, **{flag: True})

    module.setup()

    assert getattr(module, flag) is True
    assert engine_class.call_args.kwargs[flag] is True
    assert lazy_loader.call_args.args[0] is sentinel.engine


def test_eager_loading_without_repair_flags_is_unchanged() -> None:
    """The default configuration, eager and without repair, still builds."""
    module = data_module.TUHEEGDataModule()
    assert not module.lazy_loading
    assert not any(getattr(module, flag) for flag in FLAGS)
