"""Hyperparameter screen of the local-competition SAE on one read of the windows.

Composes ``configs/screen_local_competition.yaml``: the ``train_sae_local_competition``
baseline plus a ``screen`` node whose ``arms`` are grouped in stages. An arm is a set of
overrides of the ``feature`` node (``spec``, ``train_spec``, ``random_state``). The run
checks every arm before it reads data, sets up the datamodule once, checks that the
splits are subject-disjoint, reads the train and validation windows into memory once
(the preset loads lazily, so the test windows are never read), and fits every arm of
``screen.stages`` in turn. Each
arm writes the ``train_sae.py`` artifacts to ``<output_dir>/<arm>/`` together with
``feature.yaml`` (its resolved feature node) and ``screen_fit.csv`` (stage, wall time,
peak GPU memory). An arm whose checkpoint exists is kept, so later stages reuse the root
of earlier ones. A kept arm must come from the same training and validation window plan
and data settings, and a selected kept arm must have the same feature node.

Then every arm in the root is scored on the same validation samples: the interior
``[h, T - h)`` of every window, with ``h`` the largest context ``L + P - 2`` among the
arms. No label is read. Outputs in ``<output_dir>``:

- ``screen_summary.csv``: one row per arm: hyperparameters, fit cost, learning-curve
  readouts (best and plateau epoch, last-epoch drift), common-sample reconstruction
  (residual, gain, residual at the best gain), code density and its bound, atom use
  (silent, effective, share of the top three subjects), dictionary structure
  (duplicates, edge-heavy atoms, energy span, peak frequency, spatial spread) and the
  similarity of matched atoms across seeds;
- ``screen_curves.csv``: the per-epoch histories of all arms;
- ``screen_subjects.csv``: per arm and validation subject, the residual and code share;
- ``screen_seed_groups.csv``: arms that differ only in ``random_state``, with the spread
  of their residual and the similarity of their matched atoms;
- ``screen_curves.png``, ``screen_tradeoff.png``, ``screen_geometry.png`` and
  ``<arm>/atoms_top.png``;
- ``invocations/<time>/``: the overrides, commit, status, diff and untracked sources of
  each invocation.

Run on the HPC from ``code/``::

    python src/screen_local_competition.py 'screen.stages=[A]' data.num_workers=8 seed=42 \\
        data.seed=42 output_dir=logs/screen_local_competition/<name>
"""

from __future__ import annotations

import hashlib
import itertools
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import lightning
import numpy as np
import polars as pl
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

rootutils.setup_root(__file__, pythonpath=True)

from src.models.components.local_competition_sae import CHECKPOINT_NAME, LocalCompetitionSAE  # noqa: E402
from src.utils import (  # noqa: E402
    RankedLogger,
    check_split_disjoint,
    dump_window_metadata,
    extras,
    instantiate_feature,
    split_provenance,
    task_wrapper,
)
from src.waveform_diagnostics import atom_descriptors  # noqa: E402

if TYPE_CHECKING:
    from lightning import LightningDataModule

log = RankedLogger(__name__, rank_zero_only=True)

FEATURE_FILE = "feature.yaml"
FIT_FILE = "screen_fit.csv"
_PLAN_KEYS = ("train_windows_sha256", "val_windows_sha256", "data")
_RUN_KEYS = ("pretrained", "device")
_GEOMETRY_KEYS = ("competition_scale", "atom_len")
_ARM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
_ABSENT = object()
_CURVES = (
    ("val_residual_frac", "validation residual fraction", "linear"),
    ("atom_drift_median", "median atom drift per epoch (1 - cosine)", "log"),
    ("val_effective_atoms", "effective atoms (validation)", "linear"),
)
_GEOMETRY_PANELS = (
    ("residual_common", "validation residual (common samples)"),
    ("gain_common", "reconstruction gain"),
    ("density_ratio", "code density / its bound"),
    ("effective_atoms", "effective atoms"),
    ("edge_heavy_atoms", "edge-heavy atoms"),
    ("energy_span_median", "median energy span / L"),
)


class CachedWindows(TensorDataset):
    """Windows and labels held in memory, with the channel names and rate of their source dataset."""

    def __init__(self, windows: torch.Tensor, labels: torch.Tensor, source: Any) -> None:
        super().__init__(windows, labels)
        self.target_channels = getattr(source, "target_channels", None)
        self.target_sfreq = getattr(source, "target_sfreq", None)


@dataclass(frozen=True)
class Arm:
    """One arm of the screen: its stage, its overrides and its resolved feature node."""

    stage: str
    overrides: DictConfig
    feature: DictConfig


@dataclass(frozen=True)
class EvalContext:
    """The validation windows and settings shared by the evaluation of every arm."""

    loader: DataLoader
    subjects: list[str]
    channels: list[str]
    n_times: int
    margin: int
    sfreq: float
    device: str
    screen: DictConfig


@dataclass(frozen=True)
class ArmResult:
    """The evaluation of one fitted arm."""

    row: dict[str, Any]
    curves: pl.DataFrame
    subjects: pl.DataFrame
    atoms: np.ndarray


def cache_loader(loader: DataLoader, desc: str) -> DataLoader:
    """Read every window of ``loader`` into memory once, in loader order.

    The returned loader keeps the batch size and the order, with no shuffling and no
    dropped batch, so a model sees the same parent batches as from ``loader``. An
    in-memory ``TensorDataset`` (the eager datamodule) is reused without a copy.

    Args:
        loader: The datamodule loader of one split.
        desc: Progress-bar label.

    Returns:
        A loader over a ``CachedWindows`` dataset.

    Raises:
        ValueError: If the loader yields another number of windows than its dataset holds.
    """
    dataset = loader.dataset
    if isinstance(dataset, TensorDataset):
        windows, labels = dataset.tensors[0], dataset.tensors[1]
    else:
        n_windows = len(dataset)  # type: ignore[arg-type]
        buffers: list[torch.Tensor] = []
        offset = 0
        for x, y in tqdm(loader, desc=desc):
            if not buffers:
                buffers = [x.new_empty((n_windows, *x.shape[1:])), y.new_empty((n_windows, *y.shape[1:]))]
            buffers[0][offset : offset + len(x)] = x
            buffers[1][offset : offset + len(x)] = y
            offset += len(x)
        if not buffers or offset != n_windows:
            raise ValueError(f"{desc}: the loader yielded {offset} of {n_windows} windows")
        windows, labels = buffers
    return DataLoader(CachedWindows(windows, labels, dataset), batch_size=loader.batch_size, shuffle=False)


def _leaves(node: Any, prefix: str = "") -> list[str]:
    """Dotted paths of the leaves of a nested config."""
    if isinstance(node, DictConfig) and len(node):
        return [path for key, value in node.items() for path in _leaves(value, f"{prefix}{key}.")]
    return [prefix.rstrip(".")] if prefix else []


def resolve_arms(cfg: DictConfig) -> dict[str, Arm]:
    """Every arm of ``cfg.screen.arms`` in config order, with its feature node checked and resolved.

    Raises:
        ValueError: On a selected stage that is not defined, an arm name that is repeated
            or is not a plain directory name, or an override of a key that the feature
            node does not define.
    """
    unknown_stages = [str(stage) for stage in cfg.screen.stages if stage not in cfg.screen.arms]
    if unknown_stages:
        raise ValueError(f"stages {unknown_stages} are not defined in screen.arms ({list(cfg.screen.arms)})")
    arms: dict[str, Arm] = {}
    for stage, group in cfg.screen.arms.items():
        for name, overrides in (group or {}).items():
            if not _ARM_NAME.fullmatch(str(name)) or str(name) in arms:
                raise ValueError(f"arm name {name!r} is repeated or is not a plain directory name")
            overrides = overrides if overrides is not None else OmegaConf.create()
            paths = _leaves(overrides)
            unknown = [key for key in paths if OmegaConf.select(cfg.feature, key, default=_ABSENT) is _ABSENT]
            if unknown:
                raise ValueError(f"arm {name}: the feature node has no keys {unknown}")
            arms[str(name)] = Arm(str(stage), overrides, cast(DictConfig, OmegaConf.merge(cfg.feature, overrides)))
    return arms


def _feature_values(feature: DictConfig) -> dict[str, Any]:
    """The plain feature node without the keys that steer loading rather than the fit."""
    plain = cast(dict[str, Any], OmegaConf.to_container(feature, resolve=True))
    return {key: value for key, value in plain.items() if key not in _RUN_KEYS}


def fitted_arms(root: Path) -> list[Path]:
    """Arm directories of ``root`` that hold a checkpoint and a feature node, sorted by name."""
    return sorted(path.parent for path in root.glob(f"*/{FEATURE_FILE}") if (path.parent / CHECKPOINT_NAME).exists())


def check_same_plan(arm_dir: Path, provenance: dict[str, Any]) -> None:
    """Refuse a fitted arm whose training or validation windows or data settings differ from this run.

    Raises:
        ValueError: On a difference in the window-plan fingerprints or the data settings.
    """
    checkpoint = torch.load(arm_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
    saved = checkpoint.get("provenance") or {}
    differing = [key for key in _PLAN_KEYS if saved.get(key) != provenance.get(key)]
    if differing:
        raise ValueError(f"{arm_dir} was fitted with other {differing}; choose a new output_dir")


def record_invocation(root: Path, overrides: list[str]) -> None:
    """Archive the overrides and the code state of this invocation in ``root/invocations/<time>``.

    Writes the commit, ``git status``, the diff against the commit and copies of the
    untracked files under ``src``, ``configs`` and ``tests``, so uncommitted code is kept.
    A failed ``git`` call is logged and leaves its file empty.
    """
    target = root / "invocations" / time.strftime("%Y%m%d_%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    (target / "overrides.txt").write_text("\n".join(overrides) + "\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[1]
    commands = {
        "sha.txt": ("rev-parse", "HEAD"),
        "status.txt": ("status", "--short"),
        "diff.patch": ("diff", "HEAD"),
        "untracked.txt": ("ls-files", "--others", "--exclude-standard", "--", "src", "configs", "tests"),
    }
    outputs: dict[str, str] = {}
    for name, args in commands.items():
        try:
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
            outputs[name] = result.stdout
        except (OSError, subprocess.CalledProcessError) as error:
            log.warning(f"git {' '.join(args)} failed ({error}); the code record of this invocation is incomplete")  # noqa: G004
            outputs[name] = ""
        (target / name).write_text(outputs[name], encoding="utf-8")
    for relative in outputs["untracked.txt"].splitlines():
        destination = target / "untracked" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)


def fit_arm(arm: Arm, loaders: tuple[DataLoader, DataLoader], provenance: dict[str, Any], arm_dir: Path) -> None:
    """Fit one arm and write its artifacts, fit cost and feature node to ``arm_dir``.

    The feature node is written last: an arm directory with a checkpoint and a feature
    node is complete.

    Raises:
        TypeError: If the feature node does not build an unfitted ``LocalCompetitionSAE``.
    """
    model = instantiate_feature(arm.feature)
    if not isinstance(model, LocalCompetitionSAE) or model.fitted:
        raise TypeError("the screen fits new LocalCompetitionSAE dictionaries; check feature._target_ and pretrained")
    cuda = model.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    start = time.perf_counter()
    model.fit_unsupervised(*loaders, provenance=provenance)
    seconds = time.perf_counter() - start
    model.save_artifacts(arm_dir)
    peak = torch.cuda.max_memory_allocated(model.device) / 2**30 if cuda else float("nan")
    pl.DataFrame({"stage": [arm.stage], "fit_seconds": [seconds], "peak_gpu_gib": [peak]}).write_csv(arm_dir / FIT_FILE)
    OmegaConf.save(arm.feature, arm_dir / FEATURE_FILE, resolve=True)


@torch.no_grad()
def evaluate_interior(
    model: LocalCompetitionSAE, loader: DataLoader, margin: int, eval_windows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruction and code statistics of ``model`` on samples ``[margin, T - margin)`` of every window.

    Whole windows go through the public ``encode`` and ``decode``. With ``margin`` at
    least the model context ``L + P - 2``, the interior equals the contextual chunks of
    training. ``encode`` zeroes unusable channels, so a channel is scored when its
    processed row is not all zero, and a window without a usable channel adds no code.

    Args:
        model: A fitted dictionary.
        loader: Windows ``(B, C, T)`` with labels; the labels are not read.
        margin: Samples left out at each window edge.
        eval_windows: Windows encoded together.

    Returns:
        Totals ``(6,)``: squared error, signal energy, reconstruction energy, signal and
        reconstruction product, scored channel samples and scored window positions. Per
        atom ``(3, K)``: code energy, code sum and positive count (the rows that
        ``LocalCompetitionSAE.atom_usage`` reads). Per window ``(n, 4)``: squared error,
        signal energy, code energy and positive count.

    Raises:
        ValueError: If ``margin`` is below the model context or leaves no interior.
    """
    if margin < model.margin:
        raise ValueError(f"margin {margin} is below the model context {model.margin}")
    totals = torch.zeros(6, dtype=torch.float64, device=model.device)
    per_atom = torch.zeros(3, model.spec.n_atoms, dtype=torch.float64, device=model.device)
    per_window: list[torch.Tensor] = []
    for x, _ in loader:
        for part in x.split(eval_windows):
            code, rows = model.encode(part)
            n_interior = rows.shape[-1] - 2 * margin
            if n_interior < 1:
                raise ValueError(f"windows of {rows.shape[-1]} samples have no interior at margin {margin}")
            interior = slice(margin, margin + n_interior)
            valid = rows.ne(0).any(-1, keepdim=True)
            target = rows[..., interior]
            reconstruction = model.decode(code)[..., interior] * valid
            code = code[..., interior] * valid.any(1, keepdim=True)
            error = (reconstruction - target).square().sum((1, 2))
            signal = target.square().sum((1, 2))
            energy = code.square()
            positive = code > 0
            values = (
                error.sum(),
                signal.sum(),
                reconstruction.square().sum(),
                (target * reconstruction).sum(),
                valid.sum() * n_interior,
                valid.any(1).sum() * n_interior,
            )
            totals += torch.stack([value.double() for value in values])
            atom_values = (energy.sum((0, 2)), code.sum((0, 2)), positive.sum((0, 2)).to(code.dtype))
            per_atom += torch.stack(atom_values).double()
            window_values = (error, signal, energy.sum((1, 2)), positive.sum((1, 2)).to(code.dtype))
            per_window.append(torch.stack(window_values, dim=1).double().cpu())
    return totals.cpu().numpy(), per_atom.cpu().numpy(), torch.cat(per_window).numpy()


def best_lag_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Largest channel-summed cross-correlation over integer lags between two sets of atoms.

    Each atom is scaled to unit Frobenius norm and lags run over every overlap, with
    zeros outside the support. An atom and a shifted copy that keeps its whole support
    score 1, so the value flags redundancy up to a shift.

    Args:
        first: Atoms ``(K1, C, L)``.
        second: Atoms ``(K2, C, L)`` on the same channels and length.

    Returns:
        Similarities ``(K1, K2)``, at most 1. For ``L > 1`` the transform also holds lags
        without overlap, which count as 0, so a pair that correlates negatively at every
        lag scores 0.

    Raises:
        ValueError: If the channel counts or lengths differ.

    Example:
        >>> atom = np.zeros((1, 1, 8))
        >>> atom[0, 0, 2:4] = 1.0
        >>> round(float(best_lag_similarity(atom, np.roll(atom, 3, axis=-1))[0, 0]), 6)
        1.0
    """
    if first.shape[1:] != second.shape[1:]:
        raise ValueError(f"atom shapes differ: {first.shape[1:]} and {second.shape[1:]}")
    n_fft = 1 << (2 * first.shape[-1] - 2).bit_length()

    def spectrum(atoms: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(atoms.reshape(len(atoms), -1), axis=1)
        return np.fft.rfft(atoms / np.maximum(norm, 1e-12)[:, None, None], n=n_fft, axis=-1)

    cross = np.einsum("icf,jcf->ijf", spectrum(first), spectrum(second).conj(), optimize=True)
    return np.fft.irfft(cross, n=n_fft, axis=-1).max(-1)


def matched_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Best-lag similarities of the one-to-one atom matching of two dictionaries with the largest sum."""
    similarity = best_lag_similarity(first, second)
    rows, cols = linear_sum_assignment(similarity, maximize=True)
    return similarity[rows, cols]


def temporal_extent(atoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where the energy of each atom ``(K, C, L)`` lies along its support.

    Returns:
        The larger energy share of the first and the last tenth of the support (about
        0.1 for energy spread evenly, near 1 for a waveform cut by an edge), and the
        span from 5 to 95 percent of the cumulative energy as a fraction of ``L``.
    """
    energy = np.square(atoms).sum(1)
    energy = energy / np.maximum(energy.sum(-1, keepdims=True), 1e-12)
    length = energy.shape[-1]
    tenth = max(1, round(length / 10))
    edge = np.maximum(energy[:, :tenth].sum(-1), energy[:, -tenth:].sum(-1))
    cumulative = np.cumsum(energy, axis=-1)
    span = ((cumulative < 0.95).sum(-1) - (cumulative < 0.05).sum(-1) + 1) / length
    return edge, span


def plateau_epoch(values: np.ndarray, tolerance: float) -> int:
    """First epoch, counted from 1, whose value is within ``tolerance`` (relative) of the minimum."""
    return int(np.argmax(values <= values.min() * (1.0 + tolerance))) + 1


def density_bound(n_samples: int, scale: int, rank: int) -> float:
    """Largest share of ``n_samples`` consecutive positions that one atom can hold (equation 17)."""
    return (rank * (n_samples // scale) + min(rank, n_samples % scale)) / n_samples


def seed_group(model: LocalCompetitionSAE) -> str:
    """Key shared by the arms whose model and schedule differ at most in ``random_state``."""
    fields = {**asdict(model.spec), **{f"train_{key}": value for key, value in asdict(model.train_spec).items()}}
    return hashlib.sha256(repr(sorted(fields.items())).encode()).hexdigest()[:8]


def _column_median(frame: pl.DataFrame, column: str) -> float:
    """Median of a numeric column, NaN when the column is absent or empty."""
    if column not in frame.columns or not frame.height:
        return float("nan")
    return float(np.median(frame[column].to_numpy()))


def _pyplot() -> Any:
    """Matplotlib's pyplot on the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(index: int) -> dict[str, str]:
    """A distinct color and line style for the ``index``-th line of a panel."""
    return {"color": f"C{index % 10}", "linestyle": ("-", "--", ":")[index // 10 % 3]}


def plot_atoms(atoms: np.ndarray, order: np.ndarray, titles: list[str], sample_ms: float, path: Path) -> None:
    """Grid of the atoms in ``order``, all channels stacked with the dominant channel in blue."""
    plt = _pyplot()
    n_cols = 4
    n_rows = max(1, -(-len(order) // n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.2 * n_rows), squeeze=False)
    for ax, atom in zip(axes.flat, order, strict=False):
        shape = atoms[atom] - atoms[atom].mean(-1, keepdims=True)
        step = 2.2 * float(np.abs(shape).max()) + 1e-12
        dominant = int(np.square(atoms[atom]).sum(-1).argmax())
        t_ms = np.arange(shape.shape[-1]) * sample_ms
        for channel, trace in enumerate(shape):
            style = {"lw": 1.0, "color": "#1f6fb2"} if channel == dominant else {"lw": 0.5, "color": "0.45"}
            ax.plot(t_ms, trace - channel * step, **style)
        ax.set_yticks([])
        ax.set_title(titles[atom], fontsize=7)
        ax.set_xlabel("ms", fontsize=6)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(order) :]:
        ax.axis("off")
    fig.suptitle(
        f"{path.parent.name}: decoder atoms by validation code-energy share, mean removed per channel", fontsize=8
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(curves: pl.DataFrame, stage_of: dict[str, str], path: Path) -> None:
    """Per-epoch monitors, one row per monitor and one column per stage."""
    plt = _pyplot()
    stages = sorted(set(stage_of.values()))
    fig, axes = plt.subplots(len(_CURVES), len(stages), figsize=(4.8 * len(stages), 3.0 * len(_CURVES)), squeeze=False)
    for col, stage in enumerate(stages):
        arms = [arm for arm, value in stage_of.items() if value == stage]
        for row, (column, label, scale) in enumerate(_CURVES):
            ax = axes[row, col]
            for index, arm in enumerate(arms):
                part = curves.filter(pl.col("arm") == arm)
                if column in part.columns and part.height:
                    ax.plot(part["epoch"].to_numpy(), part[column].to_numpy(), lw=1.0, label=arm, **_style(index))
            ax.set_yscale(scale)
            ax.set_xlabel("epoch", fontsize=7)
            ax.set_ylabel(label, fontsize=7)
            ax.tick_params(labelsize=6)
        axes[0, col].set_title(f"stage {stage}", fontsize=8)
        axes[0, col].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tradeoff(summary: pl.DataFrame, path: Path) -> None:
    """Common-sample residual against code density, one point per arm; color = stage, marker = L."""
    plt = _pyplot()
    stages = sorted(set(summary["stage"].to_list()))
    lengths = sorted(set(summary["atom_len"].to_list()))
    markers = "osD^v<>ph"
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for item in summary.iter_rows(named=True):
        point = (item["density_common"], item["residual_common"])
        color = f"C{stages.index(item['stage']) % 10}"
        ax.scatter(*point, color=color, marker=markers[lengths.index(item["atom_len"]) % len(markers)], s=28)
        ax.annotate(item["arm"], point, fontsize=5, xytext=(3, 2), textcoords="offset points")
    for index, stage in enumerate(stages):
        ax.scatter([], [], color=f"C{index % 10}", marker="o", label=f"stage {stage}")
    for index, length in enumerate(lengths):
        ax.scatter([], [], color="0.4", marker=markers[index % len(markers)], label=f"L = {length}")
    ax.set_xscale("log")
    ax.set_xlabel("validation code density (positive coefficients per atom and sample)", fontsize=8)
    ax.set_ylabel("validation residual fraction (common samples)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_geometry(summary: pl.DataFrame, base: dict[str, Any], path: Path) -> None:
    """Metrics against P, one line per L, over the arms equal to ``base`` outside P and L."""
    same = [pl.col(key) == value for key, value in base.items() if key not in _GEOMETRY_KEYS]
    grid = summary.filter(pl.all_horizontal(same))
    if grid.height < 2:
        log.info("Geometry figure skipped: fewer than two arms equal the base configuration outside P and L")
        return
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5))
    for ax, (column, label) in zip(axes.flat, _GEOMETRY_PANELS, strict=True):
        for index, length in enumerate(sorted(set(grid["atom_len"].to_list()))):
            part = grid.filter(pl.col("atom_len") == length).sort("competition_scale")
            x, y = part["competition_scale"].to_numpy(), part[column].to_numpy()
            ax.plot(x, y, marker="o", label=f"L = {length}", **_style(index))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("competition scale P (samples)", fontsize=7)
        ax.set_ylabel(label, fontsize=7)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("arms equal to the base configuration except P and L", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_arm(arm_dir: Path, feature: DictConfig, context: EvalContext) -> ArmResult:
    """Score one fitted arm on the common samples, read its history and measure its dictionary.

    Also writes ``<arm>/atoms_top.png``.
    """
    feature = cast(DictConfig, OmegaConf.merge(feature, {"pretrained": str(arm_dir / CHECKPOINT_NAME)}))
    feature.device = context.device
    model = instantiate_feature(feature)
    screen = context.screen
    totals, per_atom, per_window = evaluate_interior(model, context.loader, context.margin, int(screen.eval_windows))
    error, signal, reconstruction, product, _, positions = totals.tolist()
    spec = model.spec
    usage = LocalCompetitionSAE.atom_usage(torch.from_numpy(per_atom))
    bound = density_bound(context.n_times - 2 * context.margin, spec.competition_scale, spec.top_r)
    density = per_atom[2].sum() / max(spec.n_atoms * positions, 1.0)

    history = pl.DataFrame(model.history)
    last: dict[str, Any] = history.row(-1, named=True) if history.height else {}
    val = history["val_residual_frac"].to_numpy() if "val_residual_frac" in history.columns else np.empty(0)
    fit = pl.read_csv(arm_dir / FIT_FILE).row(0, named=True) if (arm_dir / FIT_FILE).exists() else {}

    windows = pl.DataFrame(
        {
            "subject": context.subjects,
            "error": per_window[:, 0],
            "signal": per_window[:, 1],
            "code_energy": per_window[:, 2],
            "positives": per_window[:, 3],
        }
    )
    subjects = (
        windows.group_by("subject")
        .agg(pl.len().alias("windows"), pl.col("error", "signal", "code_energy", "positives").sum())
        .with_columns(
            residual_frac=pl.col("error") / pl.col("signal"),
            code_share=pl.col("code_energy") / pl.col("code_energy").sum(),
        )
        .sort("code_share", descending=True)
        .select(pl.lit(arm_dir.name).alias("arm"), "subject", "windows", "residual_frac", "code_share", "positives")
    )

    atoms = np.load(arm_dir / "sae_atoms.npy")
    descriptors = atom_descriptors(atoms, context.channels, context.sfreq)
    peak = np.asarray(descriptors["peak_hz_encoded"], dtype=float)
    nearest = np.full(len(atoms), np.nan)
    if len(atoms) > 1:
        similarity = best_lag_similarity(atoms, atoms)
        np.fill_diagonal(similarity, -np.inf)
        nearest = similarity.max(1)
    edge, span = temporal_extent(atoms)
    share = per_atom[0] / max(float(per_atom[0].sum()), 1e-12)
    order = np.argsort(-share)[: int(screen.gallery_atoms)]
    titles = [f"atom {k}: {100 * share[k]:.1f} % energy, {peak[k]:.0f} Hz" for k in range(len(atoms))]
    signal_atoms = np.load(arm_dir / "sae_atoms_signal.npy")
    plot_atoms(signal_atoms, order, titles, 1000.0 / context.sfreq, arm_dir / "atoms_top.png")

    row: dict[str, Any] = {
        "arm": arm_dir.name,
        "stage": str(fit.get("stage", "")),
        **asdict(spec),
        **asdict(model.train_spec),
        "random_state": model.random_state,
        "atom_ms": 1000.0 * spec.atom_len / context.sfreq,
        "competition_ms": 1000.0 * spec.competition_scale / context.sfreq,
        "fit_seconds": float(fit.get("fit_seconds", np.nan)),
        "peak_gpu_gib": float(fit.get("peak_gpu_gib", np.nan)),
        "seconds_per_epoch": _column_median(history, "epoch_seconds"),
        "steps": float(last.get("cumulative_steps", np.nan)),
        "train_residual_last": float(last.get("residual_frac", np.nan)),
        "val_residual_last": float(last.get("val_residual_frac", np.nan)),
        "val_residual_best": float(val.min()) if val.size else np.nan,
        "best_epoch": int(val.argmin()) + 1 if val.size else None,
        "plateau_epoch": plateau_epoch(val, float(screen.plateau_tol)) if val.size else None,
        "val_change_last5": float((val[-6] - val[-1]) / val[-1]) if val.size > 5 else np.nan,
        "drift_median_last": float(last.get("atom_drift_median", np.nan)),
        "eval_margin": context.margin,
        "residual_common": error / max(signal, 1e-12),
        "gain_common": product / max(reconstruction, 1e-12),
        "residual_at_gain_common": 1.0 - product**2 / max(signal * reconstruction, 1e-12),
        "density_common": float(density),
        "density_bound": bound,
        "density_ratio": float(density) / bound,
        "silent_atoms": int(usage["silent_atoms"]),
        "effective_atoms": usage["effective_atoms"],
        "top3_subject_code_share": float(subjects["code_share"].head(3).sum()),
        "subject_residual_max": float(np.nanmax(subjects["residual_frac"].to_numpy())),
        "duplicate_atoms": int((nearest >= float(screen.duplicate_threshold)).sum()),
        "nearest_similarity_median": float(np.median(nearest)),
        "edge_heavy_atoms": int((edge > float(screen.edge_share)).sum()),
        "energy_span_median": float(np.median(span)),
        "peak_hz_median": float(np.median(peak)),
        "peak_hz_q10": float(np.quantile(peak, 0.1)),
        "peak_hz_q90": float(np.quantile(peak, 0.9)),
        "channels_90_median": float(np.median(descriptors["n_channels_90"])),
        "rank_one_median": float(np.median(descriptors["rank_one_frac"])),
        "seed_group": seed_group(model),
    }
    curves = history.select(pl.lit(arm_dir.name).alias("arm"), pl.all()) if history.height else pl.DataFrame()
    return ArmResult(row, curves, subjects, atoms)


def add_seed_stability(results: list[ArmResult]) -> pl.DataFrame:
    """Match the atoms of arms that differ only in ``random_state`` and fill ``seed_match_median``.

    For each pair of arms in a group, the median best-lag similarity of the matched atoms;
    an arm's ``seed_match_median`` is the mean of its pair medians (NaN without a partner).

    Returns:
        One row per group of two or more arms: its arms, the mean, standard deviation
        (ddof 1) and range of their common-sample residual, and the mean pair median.
    """
    groups: dict[str, list[ArmResult]] = {}
    for result in results:
        result.row["seed_match_median"] = float("nan")
        groups.setdefault(result.row["seed_group"], []).append(result)
    rows = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        medians = np.full((len(members), len(members)), np.nan)
        for i, j in itertools.combinations(range(len(members)), 2):
            medians[i, j] = medians[j, i] = np.median(matched_similarity(members[i].atoms, members[j].atoms))
        for i, member in enumerate(members):
            member.row["seed_match_median"] = float(np.nanmean(medians[i]))
        residuals = np.array([member.row["residual_common"] for member in members])
        rows.append(
            {
                "seed_group": key,
                "arms": " ".join(member.row["arm"] for member in members),
                "n_seeds": len(members),
                "residual_mean": float(residuals.mean()),
                "residual_sd": float(residuals.std(ddof=1)),
                "residual_range": float(np.ptp(residuals)),
                "seed_match_median": float(np.nanmean(medians)),
            }
        )
    return pl.DataFrame(rows)


def evaluate_root(root: Path, val_loader: DataLoader, val_subjects: list[str], cfg: DictConfig) -> pl.DataFrame:
    """Score every fitted arm of ``root`` on the same validation samples and write the screen outputs.

    Returns:
        The summary table, one row per arm, or an empty table when ``root`` has no arm.
    """
    arm_dirs = fitted_arms(root)
    if not arm_dirs:
        log.warning(f"No fitted arm in {root}")  # noqa: G004
        return pl.DataFrame()
    features = [cast(DictConfig, OmegaConf.load(arm_dir / FEATURE_FILE)) for arm_dir in arm_dirs]
    margin = cfg.screen.eval_margin
    if margin is None:
        margin = max(int(feature.spec.atom_len) + int(feature.spec.competition_scale) - 2 for feature in features)
    dataset = val_loader.dataset
    context = EvalContext(
        loader=val_loader,
        subjects=val_subjects,
        channels=[str(channel) for channel in (getattr(dataset, "target_channels", None) or [])],
        n_times=int(dataset[0][0].shape[-1]),
        margin=int(margin),
        sfreq=float(cfg.data.target_sfreq),
        device=str(cfg.feature.device),
        screen=cfg.screen,
    )
    results = [
        evaluate_arm(arm_dir, feature, context)
        for arm_dir, feature in tqdm(list(zip(arm_dirs, features, strict=True)), desc="Scoring arms on common samples")
    ]
    seed_groups = add_seed_stability(results)
    summary = pl.DataFrame([result.row for result in results], infer_schema_length=None)
    summary.write_csv(root / "screen_summary.csv")
    pl.concat([result.subjects for result in results]).write_csv(root / "screen_subjects.csv")
    seed_groups.write_csv(root / "screen_seed_groups.csv")
    histories = [result.curves for result in results if result.curves.height]
    if histories:
        curves = pl.concat(histories, how="diagonal_relaxed")
        curves.write_csv(root / "screen_curves.csv")
        stage_of = dict(zip(summary["arm"].to_list(), summary["stage"].to_list(), strict=True))
        plot_curves(curves, stage_of, root / "screen_curves.png")

    base = instantiate_feature(cfg.feature)
    base_values = {**asdict(base.spec), **asdict(base.train_spec), "random_state": base.random_state}
    plot_tradeoff(summary, root / "screen_tradeoff.png")
    plot_geometry(summary, base_values, root / "screen_geometry.png")

    shown = (
        "arm",
        "residual_common",
        "gain_common",
        "density_ratio",
        "effective_atoms",
        "silent_atoms",
        "duplicate_atoms",
        "edge_heavy_atoms",
        "plateau_epoch",
        "seconds_per_epoch",
        "seed_match_median",
    )
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=3
    ):
        log.info(f"Screen summary, {context.margin} samples left out at each window edge:\n{summary.select(shown)}")  # noqa: G004
    return summary


def run_screen(cfg: DictConfig, datamodule: Any, root: Path, arms: dict[str, Arm]) -> pl.DataFrame:
    """Fit the arms of ``cfg.screen.stages`` on one read of the windows, then score every arm in ``root``.

    Args:
        cfg: The composed screen configuration.
        datamodule: A set-up datamodule exposing the split frames and loaders.
        root: The output directory shared by the stages of one screen.
        arms: The arms from ``resolve_arms(cfg)``.

    Returns:
        The summary table, one row per fitted arm in ``root``.

    Raises:
        ValueError: If ``data.target_sfreq`` is unset, a kept arm comes from another window
            plan or holds another feature node, or the validation frame and loader disagree.
    """
    if cfg.data.get("target_sfreq") is None:
        raise ValueError("the screen reports durations and frequencies; set data.target_sfreq")
    check_split_disjoint(datamodule)
    provenance = split_provenance(datamodule, cfg.data)
    kept = fitted_arms(root)
    for arm_dir in kept:
        check_same_plan(arm_dir, provenance)
    selected = {name: arm for name, arm in arms.items() if arm.stage in cfg.screen.stages}
    for name, arm in selected.items():
        if root / name in kept:
            saved = cast(DictConfig, OmegaConf.load(root / name / FEATURE_FILE))
            if _feature_values(saved) != _feature_values(arm.feature):
                raise ValueError(f"{root / name} holds another feature node; rename the arm or choose a new output_dir")
    dump_window_metadata(root, datamodule)
    train_loader = cache_loader(datamodule.train_dataloader(), "Reading training windows")
    val_loader = cache_loader(datamodule.val_dataloader(), "Reading validation windows")
    val_subjects = [str(subject) for subject in datamodule.val_df["subject"]]
    if len(val_subjects) != len(val_loader.dataset):  # type: ignore[arg-type]
        raise ValueError(f"val_df has {len(val_subjects)} rows but the validation loader has other windows")
    for name, arm in selected.items():
        if root / name in kept:
            log.info(f"{name}: fitted earlier, kept")  # noqa: G004
            continue
        log.info(f"{name} (stage {arm.stage}): fitting with {OmegaConf.to_container(arm.overrides)}")  # noqa: G004
        fit_arm(arm, (train_loader, val_loader), provenance, root / name)
    return evaluate_root(root, val_loader, val_subjects, cfg)


@task_wrapper
def screen(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the arms, set up the datamodule once and run the screen into ``paths.output_dir``.

    Returns:
        The number of arms in the summary, and the instantiated objects.
    """
    arms = resolve_arms(cfg)
    root = Path(cfg.paths.output_dir)
    record_invocation(root, list(HydraConfig.get().overrides.task))
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # noqa: G004
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    summary = run_screen(cfg, datamodule, root, arms)
    return {"n_arms": summary.height}, {"cfg": cfg, "datamodule": datamodule, "summary": summary}


@hydra.main(version_base="1.3", config_path="../configs", config_name="screen_local_competition.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point of the local-competition SAE hyperparameter screen.

    Parameters
    ----------
    cfg : DictConfig
        A configuration composed by Hydra.
    """
    extras(cfg)
    screen(cfg)


if __name__ == "__main__":
    main()
