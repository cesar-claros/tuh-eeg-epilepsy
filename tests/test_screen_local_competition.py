"""Local-competition screen: helpers, configuration, and a small end-to-end screen on CPU.

Run in the HPC container from ``code/`` with
``python -m pytest -q tests/test_screen_local_competition.py``. No EEG file is read:
the end-to-end test uses a synthetic datamodule.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import rootutils

rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Dataset, TensorDataset  # noqa: E402

import src.screen_local_competition as screen  # noqa: E402
from src.models.components.local_competition_sae import (  # noqa: E402
    CHECKPOINT_NAME,
    LocalCompetitionSAE,
    LocalCompetitionSpec,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose(overrides: list[str]) -> DictConfig:
    """Compose the screen configuration with ``overrides``."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="screen_local_competition", overrides=overrides)


def _tiny_cfg(root: Path) -> DictConfig:
    """A CPU screen of three tiny arms in stage T: two seeds of one model and another P."""
    cfg = _compose(
        [
            "feature.device=cpu",
            "feature.spec.n_atoms=3",
            "feature.spec.atom_len=5",
            "feature.spec.competition_scale=4",
            "feature.train_spec.epochs=2",
            "feature.train_spec.center_len=16",
            "feature.train_spec.crop_batch=4",
            "screen.stages=[T]",
            "screen.eval_windows=2",
            "screen.gallery_atoms=2",
            "+screen.arms.T={T_a:{},T_b:{random_state:7},T_c:{spec:{competition_scale:2}}}",
        ]
    )
    cfg.paths.output_dir = str(root)
    return cfg


class _Windows(Dataset):
    """A map-style dataset like the lazy loader: tensors per item, channel names and rate."""

    def __init__(self, windows: torch.Tensor, labels: torch.Tensor) -> None:
        self.windows, self.labels = windows, labels
        self.target_channels = ["c0", "c1"]
        self.target_sfreq = 256.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.labels[index]


def _frame(prefix: str, n_windows: int, offset: float = 0.0) -> pd.DataFrame:
    """Window metadata with two windows per subject and one label per subject."""
    return pd.DataFrame(
        {
            "subject": [f"{prefix}{i // 2}" for i in range(n_windows)],
            "path": [f"{prefix}{i}.edf" for i in range(n_windows)],
            "start": [offset + i for i in range(n_windows)],
            "end": [offset + i + 0.25 for i in range(n_windows)],
            "epilepsy": [(i // 2) % 2 for i in range(n_windows)],
        }
    )


def _datamodule(val_offset: float = 0.0) -> SimpleNamespace:
    """A set-up synthetic datamodule: lazy-style train and val loaders, a test frame never read."""
    generator = torch.Generator().manual_seed(3)
    train = torch.randn(6, 2, 64, generator=generator)
    val = torch.randn(4, 2, 64, generator=generator)
    val[1, 1] = 0.0
    train_set = _Windows(train, torch.tensor([0, 0, 1, 1, 0, 0]))
    val_set = _Windows(val, torch.tensor([0, 0, 1, 1]))
    return SimpleNamespace(
        train_df=_frame("tr", 6),
        val_df=_frame("va", 4, val_offset),
        test_df=_frame("te", 2),
        train_dataloader=lambda: DataLoader(train_set, batch_size=4),
        val_dataloader=lambda: DataLoader(val_set, batch_size=4),
    )


def _atoms(seed: int, n_atoms: int = 4, n_channels: int = 3, length: int = 9) -> np.ndarray:
    """Random dictionary ``(K, C, L)``."""
    return np.random.default_rng(seed).standard_normal((n_atoms, n_channels, length))


def test_best_lag_similarity_matches_a_direct_sum_over_lags() -> None:
    """The FFT path equals the normalized channel-summed correlation at the best lag, floored at 0."""
    first, second = _atoms(0), _atoms(1, n_atoms=2)
    result = screen.best_lag_similarity(first, second)
    length = first.shape[-1]
    for i, a in enumerate(first / np.linalg.norm(first.reshape(len(first), -1), axis=1)[:, None, None]):
        for j, b in enumerate(second / np.linalg.norm(second.reshape(len(second), -1), axis=1)[:, None, None]):
            per_lag = [
                sum(
                    float(a[c, t + lag] * b[c, t])
                    for c in range(a.shape[0])
                    for t in range(length)
                    if 0 <= t + lag < length
                )
                for lag in range(-length + 1, length)
            ]
            assert result[i, j] == pytest.approx(max(0.0, *per_lag), abs=1e-9)


def test_shifted_and_permuted_copies_are_recognized() -> None:
    """A shifted copy inside the support is a duplicate; a permutation matches one to one."""
    atoms = np.zeros((3, 2, 12))
    atoms[0, 0, 3:6] = [1.0, -2.0, 1.0]
    atoms[1, 1, 4:8] = [1.0, 1.0, -1.0, -1.0]
    atoms[2, :, 5] = [1.0, -1.0]
    shifted = np.roll(atoms, 2, axis=-1)
    assert np.diag(screen.best_lag_similarity(atoms, shifted)) == pytest.approx(np.ones(3))
    matched = screen.matched_similarity(atoms, shifted[[2, 0, 1]])
    assert matched == pytest.approx(np.ones(3))
    with pytest.raises(ValueError, match="shapes"):
        screen.best_lag_similarity(atoms, atoms[:, :1])


def test_temporal_extent_flags_edge_waveforms_and_measures_the_span() -> None:
    """Energy at one edge gives edge share 1 and the shortest span; even energy fills the support."""
    atoms = np.zeros((2, 2, 10))
    atoms[0, 1, 0] = 3.0
    atoms[1] = 1.0
    edge, span = screen.temporal_extent(atoms)
    assert edge == pytest.approx([1.0, 0.1])
    assert span == pytest.approx([0.1, 1.0])


def test_plateau_epoch_and_density_bound() -> None:
    """The plateau is the first epoch near the minimum; the bound follows equation 17."""
    assert screen.plateau_epoch(np.array([1.0, 0.5, 0.401, 0.4, 0.4]), 0.005) == 3
    assert screen.plateau_epoch(np.array([0.3]), 0.005) == 1
    assert screen.density_bound(100, 32, 1) == pytest.approx(0.04)
    assert screen.density_bound(100, 32, 2) == pytest.approx(0.08)
    assert screen.density_bound(64, 32, 1) == pytest.approx(1 / 32)


def test_atom_usage_counts_silent_atoms_and_energy_entropy() -> None:
    """Two atoms with equal energy and one silent atom give two effective atoms."""
    per_atom = torch.tensor([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 5.0, 0.0]], dtype=torch.float64)
    usage = LocalCompetitionSAE.atom_usage(per_atom, "val_")
    assert usage == pytest.approx({"val_silent_atoms": 1.0, "val_effective_atoms": 2.0})
    assert LocalCompetitionSAE.atom_usage(torch.zeros(3, 2))["effective_atoms"] == 0.0


def test_cache_loader_keeps_order_batches_and_channel_metadata() -> None:
    """A streamed dataset is read once in order; an in-memory one is shared without a copy."""
    windows, labels = torch.randn(5, 2, 7), torch.arange(5)
    streamed = screen.cache_loader(DataLoader(_Windows(windows, labels), batch_size=2), "test")
    assert streamed.batch_size == 2 and not streamed.drop_last
    assert streamed.dataset.target_channels == ["c0", "c1"] and streamed.dataset.target_sfreq == 256.0
    batches = list(streamed)
    assert [len(x) for x, _ in batches] == [2, 2, 1]
    assert torch.equal(torch.cat([x for x, _ in batches]), windows)
    assert torch.equal(torch.cat([y for _, y in batches]), labels)
    eager = TensorDataset(windows, labels)
    shared = screen.cache_loader(DataLoader(eager, batch_size=3), "test")
    assert shared.dataset.tensors[0].data_ptr() == windows.data_ptr()
    assert shared.dataset.target_channels is None


def test_shipped_configuration_composes_and_every_arm_builds() -> None:
    """The nested baseline composes with the preset's data settings, and every arm of every stage builds its model."""
    cfg = _compose(["feature.device=cpu"])
    assert cfg.task_name == "screen_local_competition"
    assert cfg.data.signal_mode == "bipolar"
    assert cfg.data.interpolate_bad_channels and cfg.data.lazy_loading
    assert cfg.data.skip_start_s == 30.0
    assert list(cfg.data.filter_freq) == [0.1, None] and list(cfg.data.notch_freqs) == [60.0]
    arms = screen.resolve_arms(cfg)
    assert {"A", "B", "C"} <= {arm.stage for arm in arms.values()}
    assert len([name for name, arm in arms.items() if arm.stage == "B"]) == 14
    for name, arm in arms.items():
        model = screen.instantiate_feature(arm.feature)
        assert isinstance(model, LocalCompetitionSAE), name
    assert screen.instantiate_feature(arms["B_P256_L160"].feature).spec.competition_scale == 256
    assert screen.instantiate_feature(arms["C_untied"].feature).spec.weight_tying == "untied"
    assert screen.instantiate_feature(arms["A_s43"].feature).random_state == 43


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("+screen.arms.A.A_bad={spec:{pool_len:8}}", "no keys"),
        ("+screen.arms.B.A_s42={}", "repeated"),
        ("screen.stages=[Z]", "not defined"),
    ],
)
def test_invalid_arms_fail_before_any_data_is_read(override: str, message: str) -> None:
    """Unknown keys, repeated names and unknown stages are rejected by ``resolve_arms``."""
    with pytest.raises(ValueError, match=message):
        screen.resolve_arms(_compose(["feature.device=cpu", override]))


def test_evaluate_interior_matches_a_direct_computation() -> None:
    """The common evaluation equals the training forward pass restricted to the interior."""
    model = LocalCompetitionSAE(
        spec=LocalCompetitionSpec(n_atoms=2, atom_len=4, competition_scale=3, pre_emphasis="none"), device="cpu"
    )
    model.initialize_dictionary(2)
    x = torch.randn(3, 2, 30, generator=torch.Generator().manual_seed(5))
    x[2, 0] = 1.0
    loader = DataLoader(TensorDataset(x, torch.zeros(3)), batch_size=2)
    margin = model.margin + 1
    with pytest.raises(ValueError, match="context"):
        screen.evaluate_interior(model, loader, model.margin - 1, 2)
    totals, per_atom, per_window = screen.evaluate_interior(model, loader, margin, 1)
    with torch.no_grad():
        rows, valid = model._rows(x)
        code = model._select(model._responses(rows, model._dictionary("enc")))
        reconstruction = model._decode(code, model._dictionary("dec"))
    inner = slice(margin, 30 - margin)
    mask = valid.unsqueeze(-1).double()
    error = ((reconstruction - rows)[..., inner].double() * mask).square().sum()
    assert totals[0] == pytest.approx(float(error), rel=1e-5)
    assert totals[1] == pytest.approx(float(rows[..., inner].double().square().sum()), rel=1e-5)
    assert totals[4] == int(valid.sum()) * (30 - 2 * margin) and totals[5] == 3 * (30 - 2 * margin)
    assert per_atom[2] == pytest.approx((code[..., inner] > 0).sum((0, 2)).double().numpy())
    assert per_window.shape == (3, 4) and per_window[:, 0].sum() == pytest.approx(totals[0])


def test_screen_fits_keeps_and_scores_arms_on_common_samples(tmp_path: Path) -> None:
    """Three tiny arms are fitted, scored and plotted; a second run keeps them; changes are refused."""
    cfg = _tiny_cfg(tmp_path)
    arms = screen.resolve_arms(cfg)
    summary = screen.run_screen(cfg, _datamodule(), tmp_path, arms)

    assert summary["arm"].to_list() == ["T_a", "T_b", "T_c"]
    assert summary["eval_margin"].unique().to_list() == [7]
    assert summary["stage"].unique().to_list() == ["T"]
    for column in ("residual_common", "gain_common", "density_ratio", "effective_atoms", "seconds_per_epoch"):
        assert summary[column].is_finite().all(), column
    assert (summary["density_ratio"] <= 1.0 + 1e-9).all()
    seeds = summary.filter(pl.col("arm").is_in(["T_a", "T_b"]))
    assert seeds["seed_group"].n_unique() == 1 and seeds["seed_match_median"].is_finite().all()
    assert summary.filter(pl.col("arm") == "T_c")["seed_match_median"].is_nan().all()
    for name in (
        "screen_summary.csv",
        "screen_curves.csv",
        "screen_subjects.csv",
        "screen_seed_groups.csv",
        "screen_curves.png",
        "screen_tradeoff.png",
        "screen_geometry.png",
        "windows_train.csv",
    ):
        assert (tmp_path / name).exists(), name
    for arm in ("T_a", "T_b", "T_c"):
        for name in (CHECKPOINT_NAME, screen.FEATURE_FILE, screen.FIT_FILE, "atoms_top.png", "sae_training.csv"):
            assert (tmp_path / arm / name).exists(), (arm, name)
    curves = pl.read_csv(tmp_path / "screen_curves.csv")
    assert {"atom_drift_median", "val_effective_atoms", "epoch_seconds"} <= set(curves.columns)
    assert curves.height == 6

    stamp = (tmp_path / "T_a" / CHECKPOINT_NAME).stat().st_mtime_ns
    again = screen.run_screen(cfg, _datamodule(), tmp_path, arms)
    assert (tmp_path / "T_a" / CHECKPOINT_NAME).stat().st_mtime_ns == stamp
    assert again["residual_common"].to_list() == pytest.approx(summary["residual_common"].to_list())

    with pytest.raises(ValueError, match="fitted with other"):
        screen.run_screen(cfg, _datamodule(val_offset=100.0), tmp_path, arms)
    changed = OmegaConf.merge(cfg, {"screen": {"arms": {"T": {"T_b": {"random_state": 8}}}}})
    with pytest.raises(ValueError, match="another feature node"):
        screen.run_screen(changed, _datamodule(), tmp_path, screen.resolve_arms(changed))


def test_record_invocation_writes_overrides_and_code_state(tmp_path: Path) -> None:
    """Every invocation leaves its overrides and the commit it ran from."""
    screen.record_invocation(tmp_path, ["screen.stages=[A]"])
    (folder,) = (tmp_path / "invocations").iterdir()
    assert (folder / "overrides.txt").read_text(encoding="utf-8") == "screen.stages=[A]\n"
    for name in ("sha.txt", "status.txt", "diff.patch", "untracked.txt"):
        assert (folder / name).exists(), name
