"""Gate 1 of the separable atom SAE probe: recovery of planted rank-1 events without labels.

Protocol: ``documentation/separable_atom_sae_protocol.md``, section 4 (equation numbers refer to
``documentation/Separable Atom Encoder Mathematics.md``). Spike-and-wave events ``a u* s^T`` from
two montage foci are planted into background segments; each arm learns a dictionary by
reconstruction alone, and the readout asks whether it learned the event. Three subcommands:

- ``run`` fits one arm, named ``{stage}_{arm}_a{amplitude}_s{seed}`` by its output directory:
  stage ``1a`` (AR(1) background with artifacts, CPU or GPU, no corpus) or ``1b`` (real segments
  of non-epilepsy patients from the campaign cache, GPU); arm ``f5`` (format 5) or ``s4ext``
  (separable, D=4, extended loss support). Writes the dictionary artifacts, ``synthetic_readout.json``
  and ``synthetic_provenance.json``.
- ``summarize`` reads the arms of a root, applies the rules fixed before any result (constants
  below) and writes ``synthetic_summary.csv`` and ``synthetic_decision.json``.
- ``plot`` draws one stage-1a validation segment as the model sees it, at every amplitude, with the
  waveform and the topographies of the events (CPU, seconds, no corpus).

Events are planted in the loader's signal domain, scaled by the robust scale of each channel, so
that after the model's row scaling every event is ``a (u*/max|u*|) (s/max|s|)``: peak amplitude
``a`` in robust units on the strongest derivation, topography ``u*``. Validation events are fresh
draws on held-out segments (1b: held-out non-epilepsy validation patients). The test split is
never read.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger
from scipy.signal import lfilter
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.shapeconv_sae import ShapeConvSAE  # noqa: E402
from src.models.components.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec  # noqa: E402
from tests.stage7_shapeconv_sae import _spike_wave  # noqa: E402

# Planted events (protocol section 4.1), fixed before any result.
TEMPLATE_SECONDS = 0.3  # spike-and-wave length: 77 samples at 256 Hz
FOCI = ("T3", "C4")  # focus A (left temporal) and focus B (right central)
FOCUS_POTENTIAL, NEIGHBOR_POTENTIAL = -1.0, -0.5  # electrode potentials behind u* = B g / ||B g||
AMPLITUDES = (0, 3, 6)  # robust units on the strongest derivation; 0 is the event-free control
EVENTS_PER_MINUTE = 2.0
MIN_GAP_SECONDS = 2.0
SEEDS = (0, 1)
EPOCHS = 5
N_PROJECTIONS = 4  # D of the separable arm
# Recovery rules (protocol section 4.3), confirmed on 2026-10-28.
RHO_W_MIN = 0.9  # waveform recovery
RHO_W_GAIN_MIN = 0.2  # over the event-free arm of the same seed
RECALL_MIN = 0.8
PRECISION_MIN = 0.5
RHO_U_MIN = 0.8  # topography recovery, for at least one focus
DETECTION_TOLERANCE = 13  # samples, about 50 ms at 256 Hz
# True (amendment of 2026-10-28, before any result): one atom per focus (largest template match),
# each judged on the events of its focus. False: one best-matching atom (largest waveform match)
# for all events, as first written in section 4.2; it scores format 5, which needs one atom per
# focus, at about half recall.
PER_FOCUS_MATCHING = True
# Stage 1a background (stage-7 AR(1) rows with artifacts) and geometry.
AR_COEFFICIENT = 0.95
ARTIFACTS_PER_MINUTE, ARTIFACT_AMPLITUDE, ARTIFACT_SAMPLES = 3.0, 4.0, 100
SPLIT_CODES = {"train": 0, "val": 1}

ARM_PATTERN = re.compile(r"^(?P<stage>1a|1b)_(?P<arm>f5|s4ext)_a(?P<amplitude>\d+)_s(?P<seed>\d+)$")
NEXT_STEPS = {
    "objective_mismatch": "Stop the architecture comparison; decide on the objective or the sampling, not gate 2",
    "separable_helps": "The separable prior helps with rare events: go to gate 2",
    "no_separation": "The gate does not separate the arms: go to gate 2",
    "f5_only": "Go to gate 2 and expect the separable arm to fail it",
    "mixed": "Each arm recovers an amplitude the other misses: go to gate 2",
    "fix_setup": "Both arms fail on the easy background: fix the code or the setup before stage 1b",
    "proceed_to_1b": "At least one arm recovers on the easy background: run stage 1b",
    "incomplete": "Some arms are missing: finish them before reading the gate",
}


@dataclass(frozen=True)
class Stage1aSettings:
    """Stage-1a sizes: complete synthetic segments of the campaign geometry with fewer atoms."""

    n_train: int = 64
    n_val: int = 16
    n_atoms: int = 16
    atom_len: int = 256
    pool_len: int = 512
    n_times: int = 30720
    sfreq: float = 256.0


def parse_arm(name: str) -> dict[str, Any]:
    """Stage, arm, amplitude and seed of an arm directory name.

    Raises
    ------
    ValueError
        If the name does not follow ``{1a|1b}_{f5|s4ext}_a{amplitude}_s{seed}`` or uses an amplitude
        outside ``AMPLITUDES``.
    """
    match = ARM_PATTERN.match(name)
    if match is None or int(match["amplitude"]) not in AMPLITUDES:
        raise ValueError(
            f"name the arm {{1a|1b}}_{{f5|s4ext}}_a{{{'|'.join(map(str, AMPLITUDES))}}}_s{{seed}}, got {name}"
        )
    return {
        "name": name,
        "stage": match["stage"],
        "arm": match["arm"],
        "amplitude": int(match["amplitude"]),
        "seed": int(match["seed"]),
    }


# ---------------------------------------------------------------------- events


def event_template(sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """The stage-7 spike-and-wave template: unit norm for matching, unit peak for planting."""
    unit = _spike_wave(np, round(TEMPLATE_SECONDS * sfreq))
    return unit, unit / np.abs(unit).max()


def focal_topography(channels: list[str], focus: str) -> np.ndarray:
    """Unit bipolar pattern ``B g`` of a negative potential at ``focus`` and half of it at its montage neighbors.

    ``channels`` are ``anode-cathode`` names; an electrode's neighbors are the electrodes that share
    a derivation with it. The phase reversal around the focus comes from the montage.

    Raises
    ------
    ValueError
        If no derivation touches ``focus``.
    """
    pairs = [tuple(name.split("-")) for name in channels]
    if not any(focus in pair for pair in pairs):
        raise ValueError(f"no derivation of {channels} touches {focus}")
    neighbors = {e for pair in pairs if focus in pair for e in pair} - {focus}
    potential = {focus: FOCUS_POTENTIAL, **dict.fromkeys(neighbors, NEIGHBOR_POTENTIAL)}
    pattern = np.array([potential.get(a, 0.0) - potential.get(c, 0.0) for a, c in pairs])
    return pattern / np.linalg.norm(pattern)


def event_schedule(rng: np.random.Generator, n_times: int, sfreq: float, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Onsets and foci of one segment: ``EVENTS_PER_MINUTE`` events, uniform, at least ``MIN_GAP_SECONDS`` apart.

    The onsets are uniform over all placements with the gap; half of the events come from each
    focus, in random order (focus A takes the extra event of an odd count).
    """
    count = round(EVENTS_PER_MINUTE * n_times / sfreq / 60.0)
    gap = round(MIN_GAP_SECONDS * sfreq)
    slack = n_times - length - (count - 1) * gap
    if count == 0 or slack < 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    onsets = np.sort(rng.integers(0, slack + 1, size=count)) + gap * np.arange(count)
    foci = rng.permutation(np.arange(count) % len(FOCI))
    return onsets.astype(np.int64), foci.astype(np.int64)


def row_scale(x: torch.Tensor) -> torch.Tensor:
    """The robust scale ``(C, 1)`` of the shared row pipeline (``ShapeConvSAE._robust_scale``)."""
    centered = x - ShapeConvSAE._row_median(x)
    mad = ShapeConvSAE._row_median(centered.abs()) * 1.4826
    return torch.maximum(mad, 0.01 * centered.std(dim=-1, keepdim=True))


def plant(
    x: torch.Tensor, onsets: np.ndarray, foci: np.ndarray, amplitude: float, shape: np.ndarray, topographies: np.ndarray
) -> torch.Tensor:
    """Add the events to one segment ``(C, T)``; channel ``c`` of an event is ``a * scale_c * u*_c / max|u*| * s``.

    A constant (unusable) row has zero scale and stays constant.
    """
    if amplitude == 0 or len(onsets) == 0:
        return x
    scale = row_scale(x)
    patterns = torch.from_numpy(topographies / np.abs(topographies).max(axis=1, keepdims=True)).to(x.dtype)
    out = x.clone()
    waveform = torch.from_numpy(shape).to(x.dtype)
    for onset, focus in zip(onsets.tolist(), foci.tolist()):
        out[:, onset : onset + len(shape)] += amplitude * scale * patterns[focus, :, None] * waveform
    return out


class ARBackground(Dataset):
    """Stage-1a background: AR(1) rows of unit standard deviation with Hanning artifacts, seeded per segment."""

    def __init__(self, n_segments: int, channels: list[str], n_times: int, sfreq: float, seed: int, split: str) -> None:
        self.n_segments, self.target_channels, self.n_times, self.sfreq = n_segments, list(channels), n_times, sfreq
        self.seed, self.split_code = seed, SPLIT_CODES[split]

    def __len__(self) -> int:
        """Number of segments."""
        return self.n_segments

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment ``index`` ``(C, T)`` and a zero label."""
        rng = np.random.default_rng([self.seed, self.split_code, index, 1])
        rows = lfilter([1.0], [1.0, -AR_COEFFICIENT], rng.standard_normal((len(self.target_channels), self.n_times)))
        rows /= rows.std(axis=-1, keepdims=True)
        n_artifacts = rng.poisson(ARTIFACTS_PER_MINUTE * self.n_times / self.sfreq / 60.0)
        for _ in range(n_artifacts):
            onset = int(rng.integers(0, self.n_times - ARTIFACT_SAMPLES))
            rows[rng.integers(len(self.target_channels)), onset : onset + ARTIFACT_SAMPLES] += (
                ARTIFACT_AMPLITUDE * np.hanning(ARTIFACT_SAMPLES)
            )
        return torch.from_numpy(rows.astype(np.float32)), torch.tensor(0)


class PlantedEvents(Dataset):
    """A view of selected segments of a source dataset with planted events, drawn from (seed, split, index).

    Exposes ``target_channels`` and, when the source has one, the selected rows of its ``plan``, so
    the patient loop can wrap it. Signal loading stays with the source (and its cache).
    """

    def __init__(
        self, source: Dataset, indices: np.ndarray, amplitude: float, seed: int, split: str, sfreq: float
    ) -> None:
        self.source, self.indices, self.amplitude, self.sfreq = source, np.asarray(indices), float(amplitude), sfreq
        self.seed, self.split_code = seed, SPLIT_CODES[split]
        self.target_channels = list(source.target_channels)
        self.unit, self.shape = event_template(sfreq)
        self.topographies = np.stack([focal_topography(self.target_channels, focus) for focus in FOCI])
        plan = getattr(source, "plan", None)
        if plan is not None:
            self.plan = plan.iloc[self.indices].reset_index(drop=True)

    def __len__(self) -> int:
        """Number of selected segments."""
        return len(self.indices)

    def events(self, index: int, n_times: int) -> tuple[np.ndarray, np.ndarray]:
        """Onsets and foci of segment ``index`` (the same draw as ``__getitem__``)."""
        rng = np.random.default_rng([self.seed, self.split_code, int(self.indices[index]), 0])
        return event_schedule(rng, n_times, self.sfreq, len(self.shape))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The source segment with its events and the source label."""
        x, y = self.source[int(self.indices[index])]
        onsets, foci = self.events(index, x.shape[-1])
        return plant(x, onsets, foci, self.amplitude, self.shape, self.topographies), y


# ---------------------------------------------------------------------- readout


def factors(model: WaveformSAE) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Unit waveforms ``(K, L)``, unit topographies ``(K, C, D')`` and, for format 5, the rank-1 energy share.

    Format-5 atoms are reduced to their best rank-1 approximation (top singular vectors, ``D' = 1``).
    """
    atoms = model._dictionary().detach().double().cpu()
    if isinstance(model, SeparableSAE):
        time, channel = model._split(atoms)
        return time.numpy(), channel.numpy(), None
    left, singular, right = torch.linalg.svd(atoms, full_matrices=False)
    share = singular[:, 0].pow(2) / singular.pow(2).sum(-1).clamp_min(1e-30)
    return right[:, 0].numpy(), left[:, :, :1].numpy(), share.numpy()


def waveform_match(waveforms: np.ndarray, unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest absolute normalized cross-correlation over lags of every waveform with the template, and its offset.

    The offset ``o`` places the template's first sample at sample ``o`` of the atom; both norms are
    full norms, so energy outside the matched part lowers the match.
    """
    rho, offset = np.zeros(len(waveforms)), np.zeros(len(waveforms), dtype=np.int64)
    for k, waveform in enumerate(waveforms):
        full = np.correlate(waveform, unit, mode="full")
        best = int(np.abs(full).argmax())
        rho[k] = abs(full[best]) / max(np.linalg.norm(waveform) * np.linalg.norm(unit), 1e-30)
        offset[k] = best - (len(unit) - 1)
    return rho, offset


def detection(positions: list[np.ndarray], expected: list[np.ndarray], tolerance: int) -> tuple[float, float, int, int]:
    """Recall and precision of one atom's activations against expected positions, segment by segment.

    ``positions[j]`` holds the atom's positive-code positions in segment ``j`` and ``expected[j]`` the
    aligned positions of its events. Returns recall, precision, events and activations (NaN when
    there are no events or no activations).
    """
    found = hits = n_events = n_activations = 0
    for active, target in zip(positions, expected):
        n_events, n_activations = n_events + len(target), n_activations + len(active)
        if len(active) and len(target):
            distance = np.abs(active[:, None] - target[None, :])
            found += int((distance.min(axis=0) <= tolerance).sum())
            hits += int((distance.min(axis=1) <= tolerance).sum())
    recall = found / n_events if n_events else math.nan
    precision = hits / n_activations if n_activations else math.nan
    return recall, precision, n_events, n_activations


@torch.no_grad()
def readout(model: WaveformSAE, dataset: PlantedEvents, batch_size: int, num_workers: int) -> dict[str, Any]:
    """Waveform and topography recovery and detection of the planted validation events (protocol section 4.2)."""
    waveforms, topographies, rank_one = factors(model)
    rho_w, offset = waveform_match(waveforms, dataset.unit)
    rho_u = np.abs(np.einsum("kcd,fc->kfd", topographies, dataset.topographies)).max(axis=2)
    best = int(rho_w.argmax())
    per_focus = [int((rho_w * rho_u[:, f]).argmax()) for f in range(len(FOCI))]
    atoms = sorted({best, *per_focus})
    left = model.padding[0]
    positions: dict[int, list[np.ndarray]] = {k: [] for k in atoms}
    expected: dict[tuple[int, int | None], list[np.ndarray]] = {}
    index = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for x, _ in loader:
        code, _ = model.encode(x)
        for j in range(len(x)):
            onsets, foci = dataset.events(index, code.shape[-1])
            for k in atoms:
                positions[k].append(torch.nonzero(code[j, k] > 0).flatten().cpu().numpy())
                aligned = onsets + left - offset[k]
                expected.setdefault((k, None), []).append(aligned)
                for f in range(len(FOCI)):
                    expected.setdefault((k, f), []).append(aligned[foci == f])
            index += 1
    recall, precision, n_events, n_activations = detection(positions[best], expected[(best, None)], DETECTION_TOLERANCE)
    record: dict[str, Any] = {
        "rho_w": float(rho_w[best]),
        "best_atom": best,
        "best_offset": int(offset[best]),
        **{f"rho_u_{focus}": float(rho_u[best, f]) for f, focus in enumerate(FOCI)},
        "recall": recall,
        "precision": precision,
        "n_events": n_events,
        "n_activations": n_activations,
        "rank_one_share": None if rank_one is None else float(rank_one[best]),
        "rho_w_per_atom": rho_w.tolist(),
    }
    for f, (focus, k) in enumerate(zip(FOCI, per_focus)):
        focus_recall, focus_precision, _, _ = detection(positions[k], expected[(k, f)], DETECTION_TOLERANCE)
        record.update(
            {
                f"{focus}_atom": k,
                f"{focus}_rho_w": float(rho_w[k]),
                f"{focus}_rho_u": float(rho_u[k, f]),
                f"{focus}_recall": focus_recall,
                f"{focus}_precision": focus_precision,
            }
        )
    return record


# ---------------------------------------------------------------------- fitting


def build_model(arm: str, spec: WaveformSpec, train_spec: WaveformTrainSpec, seed: int, device: str) -> WaveformSAE:
    """Format 5 (``f5``) or the separable arm with ``N_PROJECTIONS`` projections and extended support (``s4ext``)."""
    if arm == "f5":
        return WaveformSAE(spec=spec, train_spec=train_spec, random_state=seed, device=device)
    fields = {name: getattr(spec, name) for name in WaveformSpec.__dataclass_fields__}
    separable = SeparableSpec(**fields, n_projections=N_PROJECTIONS, loss_support="extended")
    return SeparableSAE(spec=separable, train_spec=train_spec, random_state=seed, device=device)


def _finish(
    model: WaveformSAE, val: PlantedEvents, output: Path, arm: dict[str, Any], batch_size: int, num_workers: int
) -> dict[str, Any]:
    """Save the dictionary and the readout of one fitted arm."""
    from src.run_patient_waveform_sweep import save_json

    model.save_artifacts(output)
    record = {
        **arm,
        "epochs": len(model.history),
        "n_steps": int(sum(row.get("n_steps", 0) for row in model.history)),
        "val_residual_frac": model.history[-1]["val_residual_frac"] if model.history else math.nan,
        **readout(model, val, batch_size, num_workers),
    }
    logger.info(
        f"Arm {arm['name']}: rho_w {record['rho_w']:.3f}, recall {record['recall']:.3f}, "
        f"precision {record['precision']:.3f}, rho_u {record['rho_u_T3']:.3f}/{record['rho_u_C4']:.3f}"
    )
    # Recall and precision are undefined without events (the event-free arms); JSON stores them as null.
    record = {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in record.items()}
    save_json(output / "synthetic_readout.json", record)
    return record


def fit_stage_1a(
    output: Path, epochs: int, device: str, settings: Stage1aSettings = Stage1aSettings()
) -> dict[str, Any]:
    """Fit one stage-1a arm on AR(1) background with planted events (segment sampling, patch initialization)."""
    from src.run_patient_waveform_sweep import save_json

    arm = parse_arm(output.name)
    if arm["stage"] != "1a":
        raise ValueError(f"{output.name} is not a stage-1a arm")
    output.mkdir(parents=True, exist_ok=False)
    channels = [f"{a}-{c}" for a, c in TUHEEGEpilepsy._TCP_BIPOLAR]
    common = {"channels": channels, "n_times": settings.n_times, "sfreq": settings.sfreq, "seed": arm["seed"]}
    train = PlantedEvents(
        ARBackground(settings.n_train, split="train", **common),
        np.arange(settings.n_train),
        arm["amplitude"],
        arm["seed"],
        "train",
        settings.sfreq,
    )
    val = PlantedEvents(
        ARBackground(settings.n_val, split="val", **common),
        np.arange(settings.n_val),
        arm["amplitude"],
        arm["seed"],
        "val",
        settings.sfreq,
    )
    spec = WaveformSpec(n_atoms=settings.n_atoms, atom_len=settings.atom_len, pool_len=settings.pool_len, top_r=1)
    train_spec = WaveformTrainSpec(epochs=epochs, lr=1e-3, segment_batch=2, init="patches", recenter_atoms=False)
    save_json(
        output / "synthetic_provenance.json",
        {
            **arm,
            "settings": asdict(settings),
            "epochs": epochs,
            "commit": _git("rev-parse", "HEAD").strip(),
            "uncommitted": _git("status", "--porcelain").splitlines(),
        },
    )
    torch.manual_seed(arm["seed"])
    model = build_model(arm["arm"], spec, train_spec, arm["seed"], device)
    model.fit_unsupervised(
        DataLoader(train, batch_size=8),
        DataLoader(val, batch_size=8),
        provenance={"train_subjects": ["synthetic"], "synthetic": {**arm, "stage": "1a"}},
    )
    return _finish(model, val, output, arm, batch_size=8, num_workers=0)


def fit_stage_1b(args: argparse.Namespace) -> dict[str, Any]:
    """Fit one stage-1b arm on real non-epilepsy segments of the campaign configuration (patient sampling, GPU).

    Raises
    ------
    RuntimeError
        Without a CUDA GPU.
    ValueError
        If the arm is not a stage-1b arm, the patient counts changed, or a signal cache is cold
        without ``--allow-cold-cache``.
    """
    import hydra
    import lightning
    from omegaconf import OmegaConf

    from src.run_patient_waveform_sweep import check_audit, digest, make_config, save_json
    from src.utils import check_split_disjoint, split_provenance

    output = args.output.resolve()
    arm = parse_arm(output.name)
    if arm["stage"] != "1b":
        raise ValueError(f"{output.name} is not a stage-1b arm")
    if not torch.cuda.is_available():
        raise RuntimeError("run stage 1b inside the HPC container on an allocated CUDA GPU")
    manifests, cache_root = args.manifests.resolve(), args.signal_cache_dir.expanduser().resolve()
    audit = check_audit(manifests)
    existing = {p.name for p in cache_root.iterdir() if p.is_dir()}
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    cfg = make_config(manifests, args.data_dir.resolve(), args.num_workers, skip_start_s=audit["skip_start_s"])
    cfg.seed, cfg.feature.random_state, cfg.supervised = arm["seed"], arm["seed"], None
    OmegaConf.save(cfg, output / "config.yaml", resolve=False)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    for split, patients in (("train", 117), ("val", 39), ("test", 39)):
        if getattr(datamodule, f"{split}_df").subject.nunique() != patients:
            raise ValueError(f"{split} patient count changed during loading")
    views, identities = {}, {}
    for split in ("train", "val"):
        source = getattr(datamodule, f"data_{split}")
        source.enable_signal_cache(cache_root)
        identities[split] = source.signal_cache.identity
        if identities[split] not in existing and not args.allow_cold_cache:
            raise ValueError(f"{split} signals have no entries under {cache_root}; pass the campaign cache")
        negative = np.flatnonzero(source.plan["epilepsy"].to_numpy() == 0)
        views[split] = PlantedEvents(
            source, negative, arm["amplitude"], arm["seed"], split, float(cfg.data.target_sfreq)
        )
    provenance = split_provenance(datamodule, cfg.data)
    provenance["train_subjects"] = sorted(str(s) for s in views["train"].plan["subject"].unique())
    provenance["synthetic"] = {**arm, "stage": "1b", "classes": "non-epilepsy"}
    save_json(
        output / "synthetic_provenance.json",
        {
            **arm,
            "epochs": args.epochs,
            "commit": _git("rev-parse", "HEAD").strip(),
            "uncommitted": _git("status", "--porcelain").splitlines(),
            "audit": audit,
            "manifests": {path.name: digest(path) for path in manifests.glob("windows_*.csv")},
            "signal_cache": {"root": str(cache_root), **identities},
            "train_patients": len(provenance["train_subjects"]),
            "val_patients": int(views["val"].plan["subject"].nunique()),
            "train_segments": len(views["train"]),
            "val_segments": len(views["val"]),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
        },
    )
    spec = hydra.utils.instantiate(cfg.feature.spec)
    train_spec = replace(hydra.utils.instantiate(cfg.feature.train_spec), epochs=args.epochs)
    lightning.seed_everything(arm["seed"], workers=True)
    model = build_model(arm["arm"], spec, train_spec, arm["seed"], "cuda")
    base = datamodule.train_dataloader()
    loaders = [
        DataLoader(views[split], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for split in ("train", "val")
    ]
    model.fit_unsupervised(*loaders, provenance=provenance)
    return _finish(model, views["val"], output, arm, batch_size=model.chunk_windows, num_workers=args.num_workers)


def _git(*args: str) -> str:
    from src.waveform_probe import _git as git

    return git(*args)


# ---------------------------------------------------------------------- figure

# Categorical slots 1 and 2 of the dataviz reference palette (validated: CVD Delta E 24.7, contrast >= 3:1).
FOCUS_COLORS = ("#2a78d6", "#eb6834")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ZOOM_SECONDS = 1.0  # half-width of the event zooms
TRACE_SPACING = 8.0  # robust units between stacked channels


def plot_input(path: Path, seed: int = 0, index: int = 0, settings: Stage1aSettings = Stage1aSettings()) -> Path:
    """Draw stage-1a validation segment ``index`` of ``seed`` as the model sees it, after the row scaling.

    Rows: the whole segment at the largest amplitude, with every event shaded by focus; the first
    event of each focus at each event amplitude, over the event-free background in gray; the
    waveform ``s`` and the topographies ``u*`` (unit peak). Units are robust units of each channel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    channels = [f"{a}-{c}" for a, c in TUHEEGEpilepsy._TCP_BIPOLAR]
    sfreq, n_times = settings.sfreq, settings.n_times
    background = ARBackground(index + 1, channels, n_times, sfreq, seed, "val")
    views = {a: PlantedEvents(background, np.array([index]), a, seed, "val", sfreq) for a in AMPLITUDES}
    rows = {a: ShapeConvSAE._robust_scale(view[0][0]).numpy() for a, view in views.items()}
    view, top = views[AMPLITUDES[-1]], AMPLITUDES[-1]
    onsets, foci = view.events(0, n_times)
    length, seconds = len(view.shape), np.arange(n_times) / sfreq
    offsets = TRACE_SPACING * np.arange(len(channels))[::-1]
    plt.rcParams.update(
        {
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(16, 20), facecolor=SURFACE)
    grid = figure.add_gridspec(3, 2 * len(FOCI), height_ratios=[1.2, 1.0, 0.45], hspace=0.3, wspace=0.15)

    axis = figure.add_subplot(grid[0, :])
    axis.plot(seconds, (rows[top] + offsets[:, None]).T, color=INK, linewidth=0.4)
    for onset, focus in zip(onsets.tolist(), foci.tolist()):
        axis.axvspan(onset / sfreq, (onset + length) / sfreq, color=FOCUS_COLORS[focus], alpha=0.3, linewidth=0)
    axis.set(yticks=offsets, yticklabels=channels, xlim=(0, n_times / sfreq), xlabel="Time (s)")
    axis.set_title(f"Validation segment {index}, seed {seed}, amplitude {top}: encoded rows (robust units)", loc="left")
    axis.legend(
        handles=[Patch(color=c, alpha=0.3, label=f"{f} event") for f, c in zip(FOCI, FOCUS_COLORS)],
        loc="upper right",
        frameon=False,
        ncols=len(FOCI),
    )

    half = round(ZOOM_SECONDS * sfreq)
    for f, focus in enumerate(FOCI):
        if not (foci == f).any():
            continue
        onset = int(onsets[foci == f][0])
        span = slice(max(onset + length // 2 - half, 0), min(onset + length // 2 + half, n_times))
        for column, amplitude in enumerate(AMPLITUDES[1:]):
            axis = figure.add_subplot(grid[1, len(AMPLITUDES[1:]) * f + column])
            axis.plot(seconds[span], (rows[0][:, span] + offsets[:, None]).T, color=MUTED, alpha=0.45, linewidth=1.2)
            axis.plot(seconds[span], (rows[amplitude][:, span] + offsets[:, None]).T, color=INK, linewidth=0.6)
            axis.axvspan(onset / sfreq, (onset + length) / sfreq, color=FOCUS_COLORS[f], alpha=0.25, linewidth=0)
            axis.set(yticks=offsets, yticklabels=channels if f == column == 0 else [], xlabel="Time (s)")
            axis.set_title(f"{focus} event, amplitude {amplitude} (gray: without the event)", loc="left")

    axis = figure.add_subplot(grid[2, 0])
    axis.plot(1000 * np.arange(length) / sfreq, view.shape, color=INK, linewidth=2)
    axis.set(xlabel="Time (ms)", title="Waveform s (unit peak)")
    axis = figure.add_subplot(grid[2, 1:])
    patterns = view.topographies / np.abs(view.topographies).max(axis=1, keepdims=True)
    width, position = 0.4, np.arange(len(channels))
    for f, (focus, color) in enumerate(zip(FOCI, FOCUS_COLORS)):
        axis.bar(position + (f - 0.5) * width, patterns[f], width=0.9 * width, color=color, label=f"u* {focus}")
    axis.axhline(0.0, color=MUTED, linewidth=0.8)
    axis.set_xticks(position, channels, rotation=90)
    axis.set(ylabel="Weight (unit peak)", title="Topographies u*: event amplitude on each derivation")
    axis.legend(frameon=False, ncols=len(FOCI))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------- decision


def _passes(row: dict[str, Any], baseline: float) -> bool:
    """The four criteria of section 4.3 for one seed, under the configured matching rule."""
    if not PER_FOCUS_MATCHING:
        rho_u = max(row[f"rho_u_{focus}"] for focus in FOCI)
        candidates = [(row["rho_w"], row["recall"], row["precision"], rho_u)]
    else:
        candidates = [(row[f"{f}_rho_w"], row[f"{f}_recall"], row[f"{f}_precision"], row[f"{f}_rho_u"]) for f in FOCI]

    def meets(value: float | None, threshold: float) -> bool:
        return value is not None and not math.isnan(value) and value >= threshold

    return any(
        meets(rho_w, RHO_W_MIN)
        and meets(rho_w - baseline, RHO_W_GAIN_MIN)
        and meets(recall, RECALL_MIN)
        and meets(precision, PRECISION_MIN)
        and meets(rho_u, RHO_U_MIN)
        for rho_w, recall, precision, rho_u in candidates
    )


def recovery(rows: list[dict[str, Any]], stage: str, arm: str) -> dict[int, bool | None]:
    """Whether ``arm`` recovers at each event amplitude of ``stage`` in every seed (``None`` if incomplete)."""
    table = {(r["amplitude"], r["seed"]): r for r in rows if r["stage"] == stage and r["arm"] == arm}
    verdict: dict[int, bool | None] = {}
    for amplitude in AMPLITUDES[1:]:
        if any((a, s) not in table for a in (0, amplitude) for s in SEEDS):
            verdict[amplitude] = None
            continue
        verdict[amplitude] = all(_passes(table[(amplitude, s)], table[(0, s)]["rho_w"]) for s in SEEDS)
    return verdict


def stage_outcome(stage: str, verdicts: dict[str, dict[int, bool | None]]) -> str:
    """The outcome label of a stage from the recovery verdicts of both arms (protocol section 4.3)."""
    if any(v is None for arm in verdicts.values() for v in arm.values()):
        return "incomplete"
    recovered = {arm: {a for a, ok in verdict.items() if ok} for arm, verdict in verdicts.items()}
    separable, format5 = recovered["s4ext"], recovered["f5"]
    if stage == "1a":
        return "proceed_to_1b" if separable or format5 else "fix_setup"
    top = AMPLITUDES[-1]
    if top not in separable and top not in format5:
        return "objective_mismatch"
    if separable == format5:
        return "no_separation"
    if separable - format5 and not format5 - separable:
        return "separable_helps"
    if format5 - separable and not separable - format5:
        return "f5_only"
    return "mixed"


def summarize(root: Path) -> dict[str, Any]:
    """Apply the gate-1 rules to the arms of ``root``; write the summary CSV and the decision JSON."""
    import json

    from src.run_patient_waveform_sweep import save_json

    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and ARM_PATTERN.match(p.name)):
        path = folder / "synthetic_readout.json"
        if path.is_file():
            rows.append(json.loads(path.read_text()))
    if not rows:
        raise ValueError(f"no finished arms under {root}")
    flat = [{k: v for k, v in row.items() if k != "rho_w_per_atom"} for row in rows]
    pl.DataFrame(flat).sort("stage", "arm", "amplitude", "seed").write_csv(root / "synthetic_summary.csv")
    decision: dict[str, Any] = {"per_focus_matching": PER_FOCUS_MATCHING}
    for stage in sorted({row["stage"] for row in rows}):
        verdicts = {arm: recovery(rows, stage, arm) for arm in ("f5", "s4ext")}
        outcome = stage_outcome(stage, verdicts)
        decision[stage] = {
            "recovers": {arm: {str(a): v for a, v in verdict.items()} for arm, verdict in verdicts.items()},
            "outcome": outcome,
            "next_step": NEXT_STEPS[outcome],
        }
        logger.info(f"Stage {stage}: {outcome}. {NEXT_STEPS[outcome]}")
    save_json(root / "synthetic_decision.json", decision)
    return decision


def main() -> None:
    """Command-line entry point: ``run`` one arm, ``summarize`` a root or ``plot`` an input segment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fit one arm; its directory name encodes stage, arm, amplitude and seed")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--device", default="auto", help="stage 1a only; stage 1b always uses CUDA")
    run.add_argument("--manifests", type=Path, help="stage 1b: the 40 s offset audit")
    run.add_argument("--data-dir", type=Path, help="stage 1b: the corpus root")
    run.add_argument("--signal-cache-dir", type=Path, help="stage 1b: the campaign signal cache")
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--allow-cold-cache", action="store_true")
    summary = commands.add_parser("summarize", help="apply the gate-1 rules to the arms of a root")
    summary.add_argument("--root", type=Path, required=True)
    figure = commands.add_parser("plot", help="draw one stage-1a validation segment as the model sees it")
    figure.add_argument("--seed", type=int, default=0)
    figure.add_argument("--index", type=int, default=0, help="validation segment")
    figure.add_argument("--output", type=Path, help="default logs/separable_synthetic/input_1a_s{seed}_i{index}.png")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.root.resolve())
    elif args.command == "plot":
        default = ROOT / "logs" / "separable_synthetic" / f"input_1a_s{args.seed}_i{args.index}.png"
        plot_input((args.output or default).resolve(), args.seed, args.index)
    elif parse_arm(args.output.name)["stage"] == "1a":
        fit_stage_1a(args.output.resolve(), args.epochs, args.device)
    else:
        missing = [flag for flag in ("manifests", "data_dir", "signal_cache_dir") if getattr(args, flag) is None]
        if missing:
            parser.error(f"stage 1b needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
        fit_stage_1b(args)


if __name__ == "__main__":
    main()
