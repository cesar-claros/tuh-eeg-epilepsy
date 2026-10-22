"""Recording-start skip: window planning, supplied plans, datamodule and provenance, without EEG reads."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import numpy as np
import pandas as pd
import pytest
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.data.components.window_dataset as window_dataset  # noqa: E402
import src.data.tuh_eeg_epilepsy_datamodule as data_module  # noqa: E402
from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.utils import split_provenance  # noqa: E402

WINDOW_S = 120.0


def _planner(skip_start_s: float) -> TUHEEGEpilepsy:
    """An engine without a corpus scan; the plan helpers read only ``skip_start_s`` from it."""
    engine = object.__new__(TUHEEGEpilepsy)
    engine.skip_start_s = skip_start_s
    return engine


def _recordings() -> pd.DataFrame:
    """Two subjects with recordings around the one-window and the skip thresholds."""
    return pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "path": ["r0.edf", "r1.edf", "r2.edf", "r3.edf"],
            "duration": [100.0, 1214.0, 149.0, 150.0],
            "epilepsy": [True, True, False, False],
        }
    )


@pytest.mark.parametrize(("skip", "expected"), [(0.0, [0, 10, 1, 1]), (30.0, [0, 9, 0, 1])])
def test_window_counts_leave_out_the_skipped_head(skip: float, expected: list[int]) -> None:
    """A recording holds floor((duration - skip - window) / stride) + 1 windows, and never fewer than 0."""
    counts, limit = _planner(skip)._calculate_limit_per_subject(_recordings(), WINDOW_S, WINDOW_S, False)

    assert counts.sort_values("path")["n_windows"].tolist() == expected
    assert limit is None


def test_generated_windows_start_at_the_skip_and_stay_inside_the_recording() -> None:
    """Windows lie on the grid skip + k * stride, and each one ends within its recording."""
    planner = _planner(30.0)
    counts, limit = planner._calculate_limit_per_subject(_recordings(), WINDOW_S, WINDOW_S, False)

    plan = planner._generate_windows_list(counts, WINDOW_S, WINDOW_S, False, limit, np.random.RandomState(0))

    durations = dict(zip(counts["path"], counts["duration"], strict=True))
    assert len(plan) == int(counts["n_windows"].sum())
    assert (plan["start"] >= 30.0).all()
    assert all(end <= durations[path] for path, end in zip(plan["path"], plan["end"], strict=True))
    assert plan.loc[plan["path"] == "r1.edf", "start"].tolist() == [30.0 + WINDOW_S * k for k in range(9)]
    assert plan.loc[plan["path"] == "r3.edf", ["start", "end"]].values.tolist() == [[30.0, 150.0]]


def test_supplied_plan_drops_windows_that_start_inside_the_skip() -> None:
    """A windows_*.csv dumped with a smaller skip loses only its earlier windows."""
    plan = pd.DataFrame({"path": ["a.edf"] * 3, "start": [0.0, 30.0, 120.0], "end": [120.0, 150.0, 240.0]})

    kept = window_dataset._drop_skipped_starts(plan, 30.0, "train")

    assert kept["start"].tolist() == [30.0, 120.0]
    assert window_dataset._drop_skipped_starts(plan, 0.0, "train").equals(plan)


@pytest.mark.parametrize("skip", [-1.0, float("nan"), float("inf")])
def test_invalid_skip_fails_at_configuration(skip: float) -> None:
    """A negative or non-finite skip fails before any setup."""
    with pytest.raises(ValueError, match="skip_start_s"):
        data_module.TUHEEGDataModule(skip_start_s=skip)


def test_skip_reaches_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """The datamodule passes the skip to the engine that plans the windows."""
    engine_class = Mock(return_value=sentinel.engine)
    monkeypatch.setattr(data_module, "TUHEEGEpilepsy", engine_class)
    lazy_loader = Mock(
        return_value={split: (sentinel.dataset, sentinel.metadata) for split in ("train", "val", "test")}
    )
    monkeypatch.setattr(window_dataset, "build_lazy_datasets", lazy_loader)
    module = data_module.TUHEEGDataModule(lazy_loading=True, skip_start_s=30)

    module.setup()

    assert module.skip_start_s == 30.0
    assert engine_class.call_args.kwargs["skip_start_s"] == 30.0


def test_skip_is_recorded_in_the_provenance() -> None:
    """Checkpoints record the skip with the other window-plan settings."""
    frame = pd.DataFrame({"subject": ["s1"], "path": ["a.edf"], "start": [30.0], "end": [150.0]})
    datamodule = SimpleNamespace(train_df=frame, val_df=frame, test_df=frame)

    provenance = split_provenance(datamodule, OmegaConf.create({"skip_start_s": 30.0, "seed": 42}))

    assert provenance["data"]["skip_start_s"] == 30.0
