"""Gate 1 of the separable atom SAE probe: recovery of planted rank-1 events without labels.

Protocol: ``documentation/separable_atom_sae_protocol.md``, section 4 (equation numbers refer to
``documentation/Separable Atom Encoder Mathematics.md``). Spike-and-wave events ``a u* s^T`` from
two montage foci are planted into background segments; each arm learns a dictionary by
reconstruction alone, and the readout asks whether it learned the event. Six subcommands:

- ``run`` fits one arm, named ``{stage}_{arm}_a{amplitude}_s{seed}`` by its output directory:
  stage ``1a`` (AR(1) background with artifacts, CPU or GPU, no corpus) or ``1b`` (real segments
  of non-epilepsy patients from the campaign cache, GPU); arm ``f5`` (format 5) or ``s4ext``
  (separable, D=4, extended loss support). Writes the dictionary artifacts, ``synthetic_readout.json``
  and ``synthetic_provenance.json``. ``--pre-emphasis diff`` differences every row before the robust
  scaling (protocol section 10.11, post hoc); ``--init random`` starts from Gaussian atoms in place
  of data patches (section 10.12, post hoc); ``--reseed 1`` replaces atoms that won no window from
  the residual after every epoch (stage 1a only; section 10.13, post hoc).
- ``summarize`` reads the arms of a root, applies the rules fixed before any result (constants
  below) and writes ``synthetic_summary.csv`` and ``synthetic_decision.json``.
- ``reread`` re-reads the stage-1b dictionaries of a root with threshold detection, calibrated on
  event-free validation patients (protocol section 10.6; post hoc, no retraining).
- ``oracle`` applies the same threshold readout to oracle atoms that hold the planted template and
  topography, on plain and first-differenced rows (protocol section 10.9; post hoc, no training).
- ``compare`` applies the initialization rule of sections 10.12 and 10.13 to a root of
  random-initialized arms and a root of patch-initialized arms with the same pre-emphasis, at the
  given seeds.
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
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.signal import lfilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402
from src.models.components.separable_sae import SeparableSAE, SeparableSpec  # noqa: E402
from src.models.components.shapeconv_sae import ShapeConvSAE  # noqa: E402
from src.models.components.waveform_sae import (  # noqa: E402
    CHECKPOINT_NAME,
    WaveformSAE,
    WaveformSpec,
    WaveformTrainSpec,
)
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


def json_safe(value: Any) -> Any:
    """``value`` with every NaN float, at any depth of dicts and lists, replaced by ``None`` (JSON ``null``)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


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


def row_template(sfreq: float, pre_emphasis: str = "none") -> np.ndarray:
    """The unit planted template as the encoder sees it: the template itself or its first difference.

    Under ``diff``, an event that starts at ``onset`` becomes ``diff([0, s, 0])`` (one sample
    longer), which also starts at ``onset``, so the alignment of section 4.2 does not change
    (protocol sections 10.9 and 10.11).
    """
    unit, _ = event_template(sfreq)
    if pre_emphasis == "diff":
        unit = np.diff(np.r_[0.0, unit, 0.0])
    return unit / np.linalg.norm(unit)


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
    """Waveform and topography recovery and detection of the planted validation events (protocol section 4.2).

    ``rho_w`` compares the waveforms with the template in the model's row domain (``row_template``).
    """
    waveforms, topographies, rank_one = factors(model)
    rho_w, offset = waveform_match(waveforms, row_template(dataset.sfreq, model.spec.pre_emphasis))
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
    record = json_safe(record)
    save_json(output / "synthetic_readout.json", record)
    return record


def fit_stage_1a(
    output: Path,
    epochs: int,
    device: str,
    settings: Stage1aSettings = Stage1aSettings(),
    pre_emphasis: str = "none",
    init: str = "patches",
    reseed: bool = False,
) -> dict[str, Any]:
    """Fit one stage-1a arm on AR(1) background with planted events (segment sampling, patch or random start)."""
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
    spec = WaveformSpec(
        n_atoms=settings.n_atoms,
        atom_len=settings.atom_len,
        pool_len=settings.pool_len,
        top_r=1,
        pre_emphasis=pre_emphasis,
    )
    train_spec = WaveformTrainSpec(
        epochs=epochs, lr=1e-3, segment_batch=2, init=init, recenter_atoms=False, reseed_dead=reseed
    )
    save_json(
        output / "synthetic_provenance.json",
        {
            **arm,
            "settings": asdict(settings),
            "epochs": epochs,
            "pre_emphasis": pre_emphasis,
            "init": init,
            "reseed": reseed,
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


def stage_1b_config(
    manifests: Path, data_dir: Path, num_workers: int, skip_start_s: float, seed: int, pre_emphasis: str, init: str
) -> Any:
    """The campaign configuration of one stage-1b arm: ``make_config`` with the arm's seed, row domain and start.

    Only keys that the composed configuration holds are set: it is in struct mode, so a new key
    raises (``reseed_dead`` is not one of them; stage 1b refuses reseeding instead).
    """
    from src.run_patient_waveform_sweep import make_config

    cfg = make_config(manifests, data_dir, num_workers, skip_start_s=skip_start_s)
    cfg.seed, cfg.feature.random_state, cfg.supervised = seed, seed, None
    cfg.feature.spec.pre_emphasis, cfg.feature.train_spec.init = pre_emphasis, init
    return cfg


def fit_stage_1b(args: argparse.Namespace) -> dict[str, Any]:
    """Fit one stage-1b arm on real non-epilepsy segments of the campaign configuration (patient sampling, GPU).

    Raises
    ------
    RuntimeError
        Without a CUDA GPU.
    ValueError
        If the arm is not a stage-1b arm, reseeding is asked for (stage 1a only, protocol section
        10.13), the patient counts changed, or a signal cache is cold without ``--allow-cold-cache``.
    """
    import hydra
    import lightning
    from omegaconf import OmegaConf

    from src.run_patient_waveform_sweep import check_audit, digest, save_json
    from src.utils import check_split_disjoint, split_provenance

    output = args.output.resolve()
    arm = parse_arm(output.name)
    if arm["stage"] != "1b":
        raise ValueError(f"{output.name} is not a stage-1b arm")
    if args.reseed:  # checked before the arm directory exists, so a refused arm leaves nothing behind
        raise ValueError("dead-atom reseeding is implemented for stage 1a only (protocol section 10.13)")
    if not torch.cuda.is_available():
        raise RuntimeError("run stage 1b inside the HPC container on an allocated CUDA GPU")
    manifests, cache_root = args.manifests.resolve(), args.signal_cache_dir.expanduser().resolve()
    audit = check_audit(manifests)
    existing = {p.name for p in cache_root.iterdir() if p.is_dir()}
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    cfg = stage_1b_config(
        manifests,
        args.data_dir.resolve(),
        args.num_workers,
        audit["skip_start_s"],
        arm["seed"],
        args.pre_emphasis,
        args.init,
    )
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
            "pre_emphasis": args.pre_emphasis,
            "init": args.init,
            "reseed": bool(args.reseed),
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


# ---------------------------------------------------------------------- post-hoc threshold re-read

# Protocol section 10.6, fixed on 2026-10-29 before any result. Post hoc: it does not change the gate decision.
THRESHOLD_FA_PER_MINUTE = 0.5  # primary false-alarm rate on the event-free calibration half
THRESHOLD_FA_CURVE = (0.25, 0.5, 1.0, 2.0)  # descriptive rates
PEAK_RADIUS = 64  # non-maximum suppression of response peaks, samples (0.25 s at 256 Hz)


def local_maxima(responses: torch.Tensor, radius: int) -> torch.Tensor:
    """Boolean mask of the positive local maxima of ``responses`` ``(..., T)`` within ``radius`` samples."""
    flat = responses.reshape(-1, 1, responses.shape[-1])
    pooled = F.max_pool1d(flat, 2 * radius + 1, stride=1, padding=radius).reshape(responses.shape)
    return (responses == pooled) & (responses > 0)


def calibrated_threshold(heights: np.ndarray, minutes: float, rate: float) -> float:
    """Threshold that leaves ``rate * minutes`` calibration peaks above it (0 when fewer peaks exist)."""
    allowed = int(rate * minutes)
    ordered = np.sort(heights)[::-1]
    return float(ordered[allowed]) if allowed < len(ordered) else 0.0


def patient_halves(subjects: np.ndarray) -> np.ndarray:
    """Calibration mask over segments: the patients at even positions of the sorted patient IDs."""
    calibration = np.unique(subjects)[::2]
    return np.isin(subjects, calibration)


def _atom_responses(model: WaveformSAE, x: torch.Tensor, atoms: list[int]) -> torch.Tensor:
    """Responses ``(B, A, T)`` of the given atoms before the selection: equation (9), on the encoded rows."""
    rows, _ = model._rows(x)
    return F.relu(model._responses(rows, model._dictionary()))[:, atoms]


@torch.no_grad()
def threshold_readout(
    model: WaveformSAE,
    calibration: PlantedEvents,
    evaluation: PlantedEvents,
    focus_atoms: dict[str, int],
    loader_kwargs: dict[str, int],
    offsets: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Recall and precision of thresholded response peaks of each focus atom (protocol section 10.6).

    ``calibration`` holds event-free segments and sets one threshold per atom and false-alarm rate;
    ``evaluation`` holds the planted segments of other patients. ``offsets`` gives the template's
    start in each atom; by default it is the best-lag offset of the planted template in the model's
    row domain. Returns, per focus, the recall and precision at ``THRESHOLD_FA_PER_MINUTE``
    (``{focus}_recall``, ``{focus}_precision``) and the curve over ``THRESHOLD_FA_CURVE``, with the
    thresholds and the durations of both halves. Each curve point also holds the events, the
    detections and the false-alarm rate per minute of the evaluation half (section 10.9).
    """
    atoms = sorted(set(focus_atoms.values()))
    if offsets is None:
        waveforms, _, _ = factors(model)
        _, best = waveform_match(waveforms, row_template(evaluation.sfreq, model.spec.pre_emphasis))
        offsets = {k: int(best[k]) for k in atoms}
    heights: dict[int, list[np.ndarray]] = {k: [] for k in atoms}
    minutes = 0.0
    for x, _ in DataLoader(calibration, shuffle=False, **loader_kwargs):
        responses = _atom_responses(model, x, atoms)
        peaks = local_maxima(responses, PEAK_RADIUS)
        for a, k in enumerate(atoms):
            heights[k].append(responses[:, a][peaks[:, a]].cpu().numpy())
        minutes += x.shape[0] * x.shape[-1] / calibration.sfreq / 60.0
    thresholds = {
        k: {rate: calibrated_threshold(np.concatenate(heights[k]), minutes, rate) for rate in THRESHOLD_FA_CURVE}
        for k in atoms
    }
    left, index, evaluation_minutes = model.padding[0], 0, 0.0
    positions: dict[tuple[int, float], list[np.ndarray]] = {(k, r): [] for k in atoms for r in THRESHOLD_FA_CURVE}
    expected: dict[str, list[np.ndarray]] = {focus: [] for focus in FOCI}
    for x, _ in DataLoader(evaluation, shuffle=False, **loader_kwargs):
        responses = _atom_responses(model, x, atoms)
        peaks = local_maxima(responses, PEAK_RADIUS)
        for j in range(len(x)):
            onsets, foci = evaluation.events(index, x.shape[-1])
            for a, k in enumerate(atoms):
                for rate in THRESHOLD_FA_CURVE:
                    above = peaks[j, a] & (responses[j, a] > thresholds[k][rate])
                    positions[(k, rate)].append(torch.nonzero(above).flatten().cpu().numpy())
            for f, focus in enumerate(FOCI):
                expected[focus].append((onsets + left - offsets[focus_atoms[focus]])[foci == f])
            index += 1
        evaluation_minutes += x.shape[0] * x.shape[-1] / evaluation.sfreq / 60.0
    record: dict[str, Any] = {
        "calibration_minutes": minutes,
        "evaluation_minutes": evaluation_minutes,
        "evaluation_segments": index,
    }
    for focus in FOCI:
        k = focus_atoms[focus]
        curve = {}
        for rate in THRESHOLD_FA_CURVE:
            recall, precision, n_events, n_detections = detection(
                positions[(k, rate)], expected[focus], DETECTION_TOLERANCE
            )
            hits = round(precision * n_detections) if n_detections else 0
            curve[str(rate)] = {
                "threshold": thresholds[k][rate],
                "recall": recall,
                "precision": precision,
                "n_events": n_events,
                "n_detections": n_detections,
                "evaluation_fa_per_minute": (n_detections - hits) / max(evaluation_minutes, 1e-30),
            }
        primary = curve[str(THRESHOLD_FA_PER_MINUTE)]
        record.update(
            {
                f"{focus}_atom": k,
                f"{focus}_recall": primary["recall"],
                f"{focus}_precision": primary["precision"],
                f"{focus}_curve": curve,
            }
        )
    return record


def _stage_1b_validation(args: argparse.Namespace) -> tuple[Any, Any]:
    """The campaign configuration and its cached validation dataset, as stage 1b built them (no training)."""
    import hydra

    from src.run_patient_waveform_sweep import check_audit, make_config
    from src.utils import check_split_disjoint

    manifests, cache_root = args.manifests.resolve(), args.signal_cache_dir.expanduser().resolve()
    audit = check_audit(manifests)
    existing = {p.name for p in cache_root.iterdir() if p.is_dir()}
    cfg = make_config(manifests, args.data_dir.resolve(), args.num_workers, skip_start_s=audit["skip_start_s"])
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    source = datamodule.data_val
    source.enable_signal_cache(cache_root)
    if source.signal_cache.identity not in existing:
        raise ValueError(f"validation signals have no entries under {cache_root}; pass the campaign cache")
    return cfg, source


def arm_pre_emphasis(folder: Path) -> str:
    """The pre-emphasis a stage-1b arm trained with, from its saved ``config.yaml``."""
    from omegaconf import OmegaConf

    return str(OmegaConf.load(folder / "config.yaml").feature.spec.pre_emphasis)


def reread(args: argparse.Namespace) -> dict[str, Any]:
    """Re-read every stage-1b dictionary of ``args.root`` with threshold detection, then apply the gate rules.

    Each arm is rebuilt with the pre-emphasis it trained with (``arm_pre_emphasis``). Arms that
    already hold ``threshold_readout.json`` are skipped. Writes ``threshold_summary.csv`` and
    ``threshold_decision.json`` to the root.
    """
    import hydra

    from src.run_patient_waveform_sweep import save_json

    root = args.root.resolve()
    arms = [parse_arm(p.name) for p in sorted(root.iterdir()) if p.is_dir() and ARM_PATTERN.match(p.name)]
    targets = [a for a in arms if a["stage"] == "1b" and a["amplitude"] > 0]
    cfg, source = _stage_1b_validation(args)
    sfreq = float(cfg.data.target_sfreq)
    negative = np.flatnonzero(source.plan["epilepsy"].to_numpy() == 0)
    calibrate = patient_halves(source.plan["subject"].astype(str).to_numpy()[negative])
    spec = hydra.utils.instantiate(cfg.feature.spec)
    train_spec = hydra.utils.instantiate(cfg.feature.train_spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_kwargs = {"batch_size": 8, "num_workers": args.num_workers}
    for arm in targets:
        folder = root / arm["name"]
        if (folder / "threshold_readout.json").is_file():
            continue
        original = json.loads((folder / "synthetic_readout.json").read_text())
        pre_emphasis = arm_pre_emphasis(folder)
        model = build_model(arm["arm"], replace(spec, pre_emphasis=pre_emphasis), train_spec, arm["seed"], device)
        model.load_checkpoint(folder / CHECKPOINT_NAME)
        views = [
            PlantedEvents(source, negative[mask], amplitude, arm["seed"], "val", sfreq)
            for mask, amplitude in ((calibrate, 0), (~calibrate, arm["amplitude"]))
        ]
        focus_atoms = {focus: int(original[f"{focus}_atom"]) for focus in FOCI}
        record = threshold_readout(model, *views, focus_atoms, loader_kwargs)
        record = json_safe(record)
        commit = _git("rev-parse", "HEAD").strip()
        save_json(folder / "threshold_readout.json", {**arm, "pre_emphasis": pre_emphasis, "commit": commit, **record})
        logger.info(
            f"Re-read {arm['name']}: "
            + ", ".join(f"{f} recall {record[f'{f}_recall']}, precision {record[f'{f}_precision']}" for f in FOCI)
        )
    return summarize_threshold(root)


def summarize_threshold(root: Path) -> dict[str, Any]:
    """Apply the gate rules to the stage-1b arms with the re-read recall and precision (protocol section 10.6)."""
    from src.run_patient_waveform_sweep import save_json

    rows, curves = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and ARM_PATTERN.match(p.name)):
        row = (
            json.loads((folder / "synthetic_readout.json").read_text())
            if (folder / "synthetic_readout.json").is_file()
            else None
        )
        if row is None or row["stage"] != "1b":
            continue
        reread_path = folder / "threshold_readout.json"
        if row["amplitude"] > 0 and reread_path.is_file():
            threshold = json.loads(reread_path.read_text())
            for focus in FOCI:
                row[f"{focus}_recall"], row[f"{focus}_precision"] = (
                    threshold[f"{focus}_recall"],
                    threshold[f"{focus}_precision"],
                )
                for rate, point in threshold[f"{focus}_curve"].items():
                    curves.append(
                        {
                            "arm": row["arm"],
                            "amplitude": row["amplitude"],
                            "seed": row["seed"],
                            "focus": focus,
                            "atom": threshold[f"{focus}_atom"],
                            "fa_per_minute": float(rate),
                            **point,
                        }
                    )
        elif row["amplitude"] > 0:
            continue  # not re-read yet: recovery() then reports the amplitude as incomplete
        rows.append(row)
    if curves:
        # Records written before section 10.9 lack the evaluation counts; their cells stay empty.
        pl.DataFrame(curves, infer_schema_length=None).sort(
            "arm", "amplitude", "seed", "focus", "fa_per_minute"
        ).write_csv(root / "threshold_summary.csv")
    verdicts = {arm: recovery(rows, "1b", arm) for arm in ("f5", "s4ext")}
    outcome = stage_outcome("1b", verdicts)
    decision = {
        "rule": "protocol section 10.6 (post hoc)",
        "fa_per_minute": THRESHOLD_FA_PER_MINUTE,
        "recovers": {arm: {str(a): v for a, v in verdict.items()} for arm, verdict in verdicts.items()},
        "outcome": outcome,
    }
    save_json(root / "threshold_decision.json", decision)
    logger.info(f"Threshold re-read of stage 1b: {outcome}")
    return decision


# ---------------------------------------------------------------------- post-hoc oracle matched filter

# Protocol section 10.9, fixed on 2026-10-29 before any result. Post hoc: it does not change the gate decision.
ORACLE_DOMAINS = ("none", "diff")  # pre-emphasis of the rows; "none" is the encoder of every gate-1 arm
ORACLE_PATTERN = re.compile(r"^oracle_(?P<domain>none|diff)_a(?P<amplitude>\d+)_s(?P<seed>\d+)\.json$")
ORACLE_OUTCOMES = {
    "atoms_limit": "A perfect atom passes where the learned atoms failed: a changed objective is worth a test",
    "input_limit": "Only differenced rows let a perfect atom pass: the next test is a retrain on differenced rows",
    "task_limit": "No perfect atom passes at amplitude 6: the objective is not the limit; this supports closing",
    "incomplete": "Some oracle runs are missing: finish them before reading the rule",
}


def oracle_model(
    spec: WaveformSpec, channels: list[str], domain: str, sfreq: float, device: str
) -> tuple[SeparableSAE, int]:
    """The oracle dictionary of section 10.9: atom ``f`` holds the planted template and ``u*`` of focus ``f``.

    Every other field of ``spec`` (atom length, pooling, clip) is the campaign's, so the rows and
    the responses are those of the gate-1 arms. Returns the model and the template's offset in
    every atom.
    """
    template = row_template(sfreq, domain)
    offset = (spec.atom_len - len(template)) // 2
    fields = {name: getattr(spec, name) for name in WaveformSpec.__dataclass_fields__}
    fields.update(n_atoms=len(FOCI), pre_emphasis=domain)
    model = SeparableSAE(spec=SeparableSpec(**fields, n_projections=1, loss_support="extended"), device=device)
    time = torch.zeros(len(FOCI), spec.atom_len, dtype=torch.float64)
    time[:, offset : offset + len(template)] = torch.from_numpy(template)
    channel = torch.from_numpy(np.stack([focal_topography(channels, focus) for focus in FOCI]))[:, :, None]
    model.atoms = nn.Parameter(model._constrain(model._pack(time, channel)).float().to(model.device))
    return model, offset


def oracle_passes(records: list[dict[str, Any]], domain: str) -> dict[int, bool | None]:
    """Whether the oracle of ``domain`` passes at each event amplitude (section 10.9; ``None`` if incomplete).

    It passes when, in every seed, at least one focus meets the recall and precision criteria of
    section 4.3 at ``THRESHOLD_FA_PER_MINUTE``.
    """
    table = {(r["amplitude"], r["seed"]): r for r in records if r["domain"] == domain}

    def meets(value: float | None, threshold: float) -> bool:
        return value is not None and not math.isnan(value) and value >= threshold

    verdict: dict[int, bool | None] = {}
    for amplitude in AMPLITUDES[1:]:
        if any((amplitude, seed) not in table for seed in SEEDS):
            verdict[amplitude] = None
            continue
        verdict[amplitude] = all(
            any(
                meets(table[(amplitude, seed)][f"{f}_recall"], RECALL_MIN)
                and meets(table[(amplitude, seed)][f"{f}_precision"], PRECISION_MIN)
                for f in FOCI
            )
            for seed in SEEDS
        )
    return verdict


def oracle_outcome(verdicts: dict[str, dict[int, bool | None]]) -> str:
    """The outcome label of section 10.9 from the verdicts of both domains, read at the top amplitude."""
    if any(v is None for verdict in verdicts.values() for v in verdict.values()):
        return "incomplete"
    top = AMPLITUDES[-1]
    if verdicts["none"][top]:
        return "atoms_limit"
    return "input_limit" if verdicts["diff"][top] else "task_limit"


def oracle(args: argparse.Namespace) -> dict[str, Any]:
    """Threshold readout of the oracle atoms in both domains, at every event amplitude and seed (section 10.9).

    Uses the validation patients, halves and event draws of the re-read (section 10.6). Runs that
    already hold their ``oracle_{domain}_a{amplitude}_s{seed}.json`` are skipped. Writes
    ``oracle_summary.csv`` and ``oracle_decision.json`` to the root.

    Raises
    ------
    ValueError
        If the root is not a directory.
    """
    import hydra

    from src.run_patient_waveform_sweep import save_json

    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory; pass the gate-1 root")
    cfg, source = _stage_1b_validation(args)
    sfreq = float(cfg.data.target_sfreq)
    negative = np.flatnonzero(source.plan["epilepsy"].to_numpy() == 0)
    calibrate = patient_halves(source.plan["subject"].astype(str).to_numpy()[negative])
    spec = hydra.utils.instantiate(cfg.feature.spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_kwargs = {"batch_size": 8, "num_workers": args.num_workers}
    focus_atoms = {focus: f for f, focus in enumerate(FOCI)}
    for domain in ORACLE_DOMAINS:
        model, offset = oracle_model(spec, list(source.target_channels), domain, sfreq, device)
        for amplitude in AMPLITUDES[1:]:
            for seed in SEEDS:
                path = root / f"oracle_{domain}_a{amplitude}_s{seed}.json"
                if path.is_file():
                    continue
                views = [
                    PlantedEvents(source, negative[mask], planted, seed, "val", sfreq)
                    for mask, planted in ((calibrate, 0), (~calibrate, amplitude))
                ]
                offsets = dict.fromkeys(focus_atoms.values(), offset)
                record = threshold_readout(model, *views, focus_atoms, loader_kwargs, offsets)
                commit = _git("rev-parse", "HEAD").strip()
                header = {"domain": domain, "amplitude": amplitude, "seed": seed, "offset": offset, "commit": commit}
                record = json_safe({**header, **record})
                save_json(path, record)
                logger.info(
                    f"Oracle {domain} a{amplitude} s{seed}: "
                    + ", ".join(
                        f"{f} recall {record[f'{f}_recall']}, precision {record[f'{f}_precision']}" for f in FOCI
                    )
                )
    return summarize_oracle(root)


def summarize_oracle(root: Path) -> dict[str, Any]:
    """Apply the rule of section 10.9 to the oracle runs of ``root``; write the summary CSV and the decision JSON.

    Raises
    ------
    ValueError
        If the root holds no oracle run.
    """
    from src.run_patient_waveform_sweep import save_json

    records = [json.loads(p.read_text()) for p in sorted(root.iterdir()) if ORACLE_PATTERN.match(p.name)]
    if not records:
        raise ValueError(f"no oracle runs under {root}")
    points = [
        {
            "domain": r["domain"],
            "amplitude": r["amplitude"],
            "seed": r["seed"],
            "focus": focus,
            "fa_per_minute": float(rate),
            **point,
        }
        for r in records
        for focus in FOCI
        for rate, point in r[f"{focus}_curve"].items()
    ]
    pl.DataFrame(points, infer_schema_length=None).sort(
        "domain", "amplitude", "seed", "focus", "fa_per_minute"
    ).write_csv(root / "oracle_summary.csv")
    verdicts = {domain: oracle_passes(records, domain) for domain in ORACLE_DOMAINS}
    outcome = oracle_outcome(verdicts)
    decision = {
        "rule": "protocol section 10.9 (post hoc)",
        "fa_per_minute": THRESHOLD_FA_PER_MINUTE,
        "passes": {domain: {str(a): v for a, v in verdict.items()} for domain, verdict in verdicts.items()},
        "outcome": outcome,
        "next_step": ORACLE_OUTCOMES[outcome],
    }
    save_json(root / "oracle_decision.json", decision)
    logger.info(f"Oracle matched filter: {outcome}. {ORACLE_OUTCOMES[outcome]}")
    return decision


# ---------------------------------------------------------------------- post-hoc initialization screen


def root_arms(root: Path, stage: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Readouts of the finished ``stage`` arms of ``root`` and the settings they share.

    The shared settings are the pre-emphasis, the start, the reseeding and the epochs. Each
    readout gains ``pre_emphasis``, ``init`` and ``reseed`` from its provenance (arms written
    before sections 10.11 to 10.13 ran with ``none``, ``patches`` and no reseeding), and
    ``n_atoms_never_won`` and ``median_coherence`` from the last epoch of its training record.

    Raises
    ------
    ValueError
        If the root holds no finished arm of the stage, or arms with different settings.
    """
    rows, settings = [], set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and ARM_PATTERN.match(p.name)):
        path = folder / "synthetic_readout.json"
        if parse_arm(folder.name)["stage"] != stage or not path.is_file():
            continue
        row = json.loads(path.read_text())
        provenance = folder / "synthetic_provenance.json"
        saved = json.loads(provenance.read_text()) if provenance.is_file() else {}
        row["pre_emphasis"], row["init"] = saved.get("pre_emphasis", "none"), saved.get("init", "patches")
        row["reseed"] = bool(saved.get("reseed", False))
        training = folder / "sae_training.csv"
        history = pl.read_csv(training) if training.is_file() else None
        never_won = history is not None and "n_atoms_never_won" in history.columns and len(history) > 0
        row["n_atoms_never_won"] = float(history["n_atoms_never_won"][-1]) if never_won else None
        coherent = history is not None and "median_coherence" in history.columns and len(history) > 0
        row["median_coherence"] = float(history["median_coherence"][-1]) if coherent else None
        rows.append(row)
        settings.add((row["pre_emphasis"], row["init"], row["reseed"], row.get("epochs")))
    if len(settings) != 1:
        raise ValueError(f"{root}: expected stage-{stage} arms with one setting, got {sorted(settings)}")
    pre_emphasis, init, reseed, epochs = settings.pop()
    return rows, {"pre_emphasis": pre_emphasis, "init": init, "reseed": reseed, "epochs": epochs}


def event_atom_cells(rows: list[dict[str, Any]]) -> int:
    """(arm, amplitude, seed, focus) cells with an event atom: ``rho_w`` and ``rho_u`` of section 4.3 for that focus."""
    return sum(
        row[f"{f}_rho_w"] >= RHO_W_MIN and row[f"{f}_rho_u"] >= RHO_U_MIN
        for row in rows
        if row["amplitude"] > 0
        for f in FOCI
    )


def _median(rows: list[dict[str, Any]], key: str, seeds: tuple[int, ...]) -> float | None:
    """Median of ``key`` over the rows at ``seeds`` that hold it (``None`` when none does)."""
    values = [r[key] for r in rows if r["seed"] in seeds and r.get(key) is not None]
    return float(np.median(values)) if values else None


def compare_initializations(
    random_root: Path, patch_root: Path, stage: str = "1a", seeds: tuple[int, ...] = SEEDS
) -> dict[str, Any]:
    """The rule of protocol sections 10.12 and 10.13: does a random start recover where the patch start does not?

    The random root may reseed dead atoms; the patch root may not. Writes
    ``init_decision.json`` (seeds 0 and 1) or ``init_decision_s{seed}_...json`` (other seeds, the
    replication of section 10.13) to ``random_root``: the recovery verdicts of both roots at
    ``seeds``, the gains and losses of the random start, the gain flag (``None`` while an arm is
    missing), and the descriptive event-atom cells, starved atoms and coherence.

    Raises
    ------
    ValueError
        If the roots do not hold random and patch starts with the same pre-emphasis and epochs, or
        the patch root reseeds.
    """
    from src.run_patient_waveform_sweep import save_json

    rows, settings = {}, {}
    for name, root in (("random", random_root), ("patches", patch_root)):
        rows[name], settings[name] = root_arms(root, stage)
        if settings[name]["init"] != name:
            raise ValueError(f"{root} holds init={settings[name]['init']} arms; expected {name}")
    if settings["patches"]["reseed"]:
        raise ValueError(f"{patch_root} reseeds dead atoms; section 10.13 compares against patch starts without it")
    for key in ("pre_emphasis", "epochs"):
        if settings["random"][key] != settings["patches"][key]:
            raise ValueError(f"the roots differ in {key}: {settings}")
    arms = ("f5", "s4ext")
    seeds = tuple(seeds)
    verdicts = {name: {arm: recovery(rows[name], stage, arm, seeds) for arm in arms} for name in rows}
    complete = all(v is not None for verdict in verdicts.values() for arm in verdict.values() for v in arm.values())
    pairs = [(arm, a) for arm in arms for a in AMPLITUDES[1:]]
    gains = [f"{arm}_a{a}" for arm, a in pairs if verdicts["random"][arm][a] and not verdicts["patches"][arm][a]]
    losses = [f"{arm}_a{a}" for arm, a in pairs if verdicts["patches"][arm][a] and not verdicts["random"][arm][a]]
    decision = {
        "rule": "protocol sections 10.12 and 10.13 (post hoc)",
        "stage": stage,
        "seeds": list(seeds),
        "pre_emphasis": settings["random"]["pre_emphasis"],
        "random_reseeds": settings["random"]["reseed"],
        "random_root": str(random_root),
        "patch_root": str(patch_root),
        "recovers": {
            name: {arm: {str(a): v for a, v in verdict[arm].items()} for arm in arms}
            for name, verdict in verdicts.items()
        },
        "gains": gains,
        "losses": losses,
        "gain": bool(gains) if complete else None,
        "event_atom_cells": {name: event_atom_cells([r for r in rows[name] if r["seed"] in seeds]) for name in rows},
        "median_atoms_never_won": {name: _median(rows[name], "n_atoms_never_won", seeds) for name in rows},
        "median_coherence": {name: _median(rows[name], "median_coherence", seeds) for name in rows},
    }
    name = "init_decision.json" if seeds == SEEDS else "init_decision_" + "_".join(f"s{s}" for s in seeds) + ".json"
    save_json(random_root / name, decision)
    logger.info(
        f"Initialization screen ({decision['pre_emphasis']}): gains {gains}, losses {losses}, "
        f"gain at seeds {list(seeds)}: {decision['gain']}"
    )
    return decision


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
            axis.set_title(f"{focus} event, amplitude {amplitude}\ngray: without the event", loc="left")

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


def recovery(
    rows: list[dict[str, Any]], stage: str, arm: str, seeds: tuple[int, ...] = SEEDS
) -> dict[int, bool | None]:
    """Whether ``arm`` recovers at each event amplitude of ``stage`` in every seed (``None`` if incomplete)."""
    table = {(r["amplitude"], r["seed"]): r for r in rows if r["stage"] == stage and r["arm"] == arm}
    verdict: dict[int, bool | None] = {}
    for amplitude in AMPLITUDES[1:]:
        if any((a, s) not in table for a in (0, amplitude) for s in seeds):
            verdict[amplitude] = None
            continue
        verdict[amplitude] = all(_passes(table[(amplitude, s)], table[(0, s)]["rho_w"]) for s in seeds)
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
    """Command-line entry point for the six subcommands of the module docstring."""
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
    run.add_argument(
        "--pre-emphasis", choices=("none", "diff"), default="none", help="row pre-emphasis (section 10.11)"
    )
    run.add_argument("--init", choices=("patches", "random"), default="patches", help="atom start (section 10.12)")
    run.add_argument("--reseed", type=int, choices=(0, 1), default=0, help="reseed dead atoms (section 10.13)")
    summary = commands.add_parser("summarize", help="apply the gate-1 rules to the arms of a root")
    summary.add_argument("--root", type=Path, required=True)
    again = commands.add_parser("reread", help="post-hoc threshold re-read of the stage-1b dictionaries of a root")
    again.add_argument("--root", type=Path, required=True)
    again.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
    again.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
    again.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
    again.add_argument("--num-workers", type=int, default=4)
    best = commands.add_parser("oracle", help="post-hoc threshold readout of oracle atoms that hold the planted event")
    best.add_argument("--root", type=Path, required=True, help="the gate-1 root that holds the re-read")
    best.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
    best.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
    best.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
    best.add_argument("--num-workers", type=int, default=4)
    screen = commands.add_parser("compare", help="post-hoc initialization rule: random against patch starts")
    screen.add_argument("--random-root", type=Path, required=True, help="a root of random-initialized arms")
    screen.add_argument("--patch-root", type=Path, required=True, help="a root of patch-initialized arms")
    screen.add_argument("--stage", choices=("1a", "1b"), default="1a")
    screen.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="2 3 for the replication")
    figure = commands.add_parser("plot", help="draw one stage-1a validation segment as the model sees it")
    figure.add_argument("--seed", type=int, default=0)
    figure.add_argument("--index", type=int, default=0, help="validation segment")
    figure.add_argument("--output", type=Path, help="default logs/separable_synthetic/input_1a_s{seed}_i{index}.png")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.root.resolve())
    elif args.command == "reread":
        reread(args)
    elif args.command == "oracle":
        oracle(args)
    elif args.command == "compare":
        compare_initializations(args.random_root.resolve(), args.patch_root.resolve(), args.stage, tuple(args.seeds))
    elif args.command == "plot":
        default = ROOT / "logs" / "separable_synthetic" / f"input_1a_s{args.seed}_i{args.index}.png"
        plot_input((args.output or default).resolve(), args.seed, args.index)
    elif parse_arm(args.output.name)["stage"] == "1a":
        fit_stage_1a(
            args.output.resolve(),
            args.epochs,
            args.device,
            pre_emphasis=args.pre_emphasis,
            init=args.init,
            reseed=bool(args.reseed),
        )
    else:
        missing = [flag for flag in ("manifests", "data_dir", "signal_cache_dir") if getattr(args, flag) is None]
        if missing:
            parser.error(f"stage 1b needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
        fit_stage_1b(args)


if __name__ == "__main__":
    main()
