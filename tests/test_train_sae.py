"""Shared SAE entry-point dispatch and configuration checks without EEG reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest
import rootutils
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.train_sae as entrypoint  # noqa: E402
import src.utils.utils as shared_utils  # noqa: E402


@dataclass
class _RunnerMocks:
    cfg: DictConfig
    datamodule: Mock
    feature: Mock
    instantiate: Mock
    instantiate_feature: Mock
    check_splits: Mock
    dump_metadata: Mock
    provenance: dict[str, list[str]]


@pytest.fixture
def runner_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _RunnerMocks:
    """Replace dataset, fit, and save operations while keeping runner control flow."""
    cfg = OmegaConf.create(
        {
            "seed": None,
            "data": {"_target_": "example.DataModule", "seed": 42, "target_sfreq": 256},
            "feature": {"_target_": "example.LocalCompetitionSAE"},
            "paths": {"output_dir": str(tmp_path)},
        }
    )
    datamodule = Mock(spec=["setup", "train_dataloader", "val_dataloader"])
    datamodule.train_dataloader.return_value = sentinel.train_loader
    datamodule.val_dataloader.return_value = sentinel.val_loader
    feature = Mock(spec=["fit_unsupervised", "save_artifacts", "fitted", "history"])
    feature.fitted = False
    feature.history = [{"epoch": 1, "objective": 0.4}, {"epoch": 2, "objective": 0.2}]
    instantiate = Mock(return_value=datamodule)
    instantiate_feature = Mock(return_value=feature)
    check_splits, dump_metadata = Mock(), Mock()
    provenance = {"train_subjects": ["training-subject"]}
    monkeypatch.setattr(entrypoint.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(entrypoint, "instantiate_feature", instantiate_feature)
    monkeypatch.setattr(entrypoint, "check_split_disjoint", check_splits)
    monkeypatch.setattr(entrypoint, "dump_window_metadata", dump_metadata)
    monkeypatch.setattr(entrypoint, "split_provenance", Mock(return_value=provenance))
    monkeypatch.setattr(entrypoint, "log", Mock())
    monkeypatch.setattr(shared_utils, "log", Mock())
    monkeypatch.setattr(
        entrypoint.HydraConfig,
        "get",
        Mock(return_value=SimpleNamespace(runtime=SimpleNamespace(choices={"feature": "local_competition_sae"}))),
    )
    return _RunnerMocks(
        cfg, datamodule, feature, instantiate, instantiate_feature, check_splits, dump_metadata, provenance
    )


@pytest.mark.parametrize("feature_class", ["ShapeConvSAE", "WaveformSAE", "LocalCompetitionSAE"])
def test_shared_runner_dispatches_through_generic_hooks(runner_mocks: _RunnerMocks, feature_class: str) -> None:
    """Every SAE follows the same split checks, fit, provenance, and artifact hooks."""
    run = runner_mocks
    run.cfg.feature._target_ = f"example.{feature_class}"

    metrics, objects = entrypoint.train_sae(run.cfg)

    run.instantiate.assert_called_once_with(run.cfg.data)
    run.datamodule.setup.assert_called_once_with()
    run.check_splits.assert_called_once_with(run.datamodule)
    run.dump_metadata.assert_called_once_with(Path(run.cfg.paths.output_dir), run.datamodule)
    run.instantiate_feature.assert_called_once_with(run.cfg.feature)
    run.feature.fit_unsupervised.assert_called_once_with(
        sentinel.train_loader, sentinel.val_loader, provenance=run.provenance
    )
    run.feature.save_artifacts.assert_called_once_with(Path(run.cfg.paths.output_dir))
    assert metrics == {"epoch": 2, "objective": 0.2}
    assert objects == {"cfg": run.cfg, "datamodule": run.datamodule, "feature_extractor": run.feature}


def test_unsupported_calibration_is_rejected_before_fit(runner_mocks: _RunnerMocks) -> None:
    """A calibration-only option cannot trigger an expensive unsupported fit first."""
    run = runner_mocks
    run.cfg.feature.calibrate_fa = 0.1

    with pytest.raises(TypeError, match="does not support threshold calibration"):
        entrypoint.train_sae(run.cfg)

    run.feature.fit_unsupervised.assert_not_called()
    run.feature.save_artifacts.assert_not_called()
    run.datamodule.train_dataloader.assert_not_called()
    run.datamodule.val_dataloader.assert_not_called()


def test_hydra_instantiates_local_competition_with_arbitrary_center_length() -> None:
    """The new feature configuration composes through the shared training entry point."""
    from src.models.components.local_competition_sae import LocalCompetitionSAE

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train_sae",
            overrides=[
                "feature=local_competition_sae",
                "feature.device=cpu",
                "feature.spec.n_atoms=3",
                "feature.spec.atom_len=7",
                "feature.spec.competition_scale=5",
                "feature.train_spec.center_len=17",
            ],
        )
        feature = shared_utils.instantiate_feature(cfg.feature)

    assert isinstance(feature, LocalCompetitionSAE)
    assert feature.spec.competition_scale == 5
    assert feature.train_spec.center_len == 17
    assert feature.spec.pre_emphasis == "none"
    assert not feature.fitted


@pytest.mark.parametrize(
    ("config_name", "feature_class", "skip_start_s"),
    [
        ("train_sae_local_competition", "LocalCompetitionSAE", 30.0),
        ("train_sae_waveform", "WaveformSAE", 0.0),
        ("waveform_diagnostics", "WaveformSAE", 0.0),
    ],
)
def test_sae_baselines_compose_highpass_and_notch_without_changing_shared_defaults(
    config_name: str, feature_class: str, skip_start_s: float
) -> None:
    """Training and diagnostic presets preserve waveforms with matching temporal filters.

    The upper cutoff stays null, and the shared defaults stay unchanged.
    """
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        baseline = compose(config_name=config_name)
        shared = compose(config_name="train_sae")

    assert baseline.feature._target_.endswith(feature_class)
    assert baseline.feature.spec.pre_emphasis == "none"
    assert baseline.data.signal_mode == "bipolar"
    assert baseline.data.interpolate_bad_channels and baseline.data.lazy_loading
    assert baseline.data.skip_start_s == skip_start_s
    assert list(baseline.data.filter_freq) == [0.1, None]
    assert list(baseline.data.notch_freqs) == [60.0]
    if feature_class == "WaveformSAE":
        assert list(baseline.data.exclude_native_sfreqs) == [400.0, 512.0, 1000.0]
        assert baseline.feature.train_spec.crop_sampling == "full"
        assert baseline.feature.train_spec.max_crop_energy_ratio is None
        assert baseline.feature.train_spec.segment_batch == 2
        assert baseline.feature.spec.top_r == 1
    else:
        assert baseline.data.exclude_native_sfreqs is None
    assert shared.feature._target_.endswith("ShapeConvSAE")
    assert shared.data.signal_mode == "raw"
    assert not shared.data.interpolate_bad_channels and not shared.data.lazy_loading
    assert shared.data.skip_start_s == 0.0
    assert shared.data.filter_freq is None and shared.data.notch_freqs is None
    assert shared.data.exclude_native_sfreqs is None
