"""Band-limited reconstruction of the separable dictionary: once the rows are band-limited to 1-30 Hz, does learning
capture content in 4-30 Hz, at R=1 or at R=4?

Protocol: ``documentation/separable_band_limited_reconstruction_protocol.md``. Run on the HPC from ``code/``; a root
holds one read, and ``domain.json`` records its band. ``src/band_reconstruction_core.py`` holds the pure logic and the
rule; the helpers of ``src/reconstruction_baselines.py`` are reused. Subcommands, in order:

- ``fit-dictionary`` trains (or, with ``--arm untrained``, only starts) the separable dictionary of one outer fold at
  ``R=1`` or ``R=4`` on band-limited rows, with the settings and seeds of readout pass 1.
- ``fit`` computes the spatial patterns and the window PCA of one fold on band-limited rows.
- ``score`` reconstructs the fixed segments of the fold's own patients with every arm.
- ``read`` applies the gates and the rule, and writes ``band_decision.json``, ``band_summary.csv`` and
  ``band_pca_curve.csv``.

The 39 test patients stay locked: the readout's cohort refuses them. No label enters a fit or a score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.band_reconstruction_core import (  # noqa: E402
    ARMS,
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    BAND_ORDER,
    BAR,
    BUDGET_ARMS,
    FAST,
    FAST_FLOOR,
    FIXED,
    IN_BAND,
    IN_BAND_MIN,
    MODEL_ARMS,
    PASS1,
    RS,
    SLOW_ATOM_HZ,
    TRAINED,
    UNTRAINED,
    arm_conditions,
    band_bank,
    band_outcome,
    band_pass,
    score_arms,
)
from src.frozen_readout_core import N_FOLDS, dictionary_split  # noqa: E402
from src.reconstruction_baselines import (  # noqa: E402
    _by_patient,
    _git,
    _interval_record,
    _save_json,
    _save_npz,
    _text_array,
    fit_path,
    score_path,
)
from src.reconstruction_baselines_core import (  # noqa: E402
    ATOM_LEN,
    BAND_NAMES,
    CONFIDENCE,
    N_ATOMS,
    N_BOOTSTRAP,
    N_SPATIAL,
    PARSEVAL_TOLERANCE,
    PCA_KS,
    POOL_LEN,
    ROW_MATCH_TOLERANCE,
    SFREQ,
    Interval,
    accumulate_moments,
    band_weights,
    bootstrap_counts,
    ev_interval,
    paired_interval,
    parseval_error,
    pooled_ev,
    set_fixed_atoms,
    smallest_k,
    top_components,
)
from src.separable_factor_analysis import dominant_frequency  # noqa: E402

DOMAIN = {"band_hz": [BAND_LOW_HZ, BAND_HIGH_HZ], "order": BAND_ORDER, "protocol": "band-limited reconstruction"}
PREDICTION = "delta_only about 55%, sweep about 35% (mostly through R=4), stop about 10% (protocol section 7)"
CHECKPOINT = "sae_state.pt"  # waveform_sae.CHECKPOINT_NAME
EXTRACT_BATCH = 8  # frozen_readout.EXTRACT_BATCH
# Fields of a score file whose segment axis is the second; the other per-segment fields have it first.
SECOND_AXIS = (
    "residual_bands",
    "residual_total",
    "windows_won",
    "windows_slow",
    "slots_won",
    "slots_slow",
    "explained_all",
    "explained_slow",
)
SEGMENT_FIELDS = (*SECOND_AXIS, "signal_bands", "signal_total", "usable_rows", "edge_energy", "interior_energy")
COLUMNS = ("total", *BAND_NAMES, "in_band", "4_30")


class BandPassed(Dataset):
    """A dataset with the 1-30 Hz band-pass of protocol section 3 on every segment, before the row scaling."""

    def __init__(self, base: Any, sfreq: float) -> None:
        self.base, self.sfreq = base, sfreq
        self.plan, self.target_channels = base.plan, base.target_channels

    def __len__(self) -> int:
        """Number of segments."""
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The band-passed segment and its label."""
        x, y = self.base[index]
        return torch.from_numpy(band_pass(x.numpy(), self.sfreq).copy()), y


# ---------------------------------------------------------------------- root and models


def dictionary_dir(root: Path, arm: str, r: int, fold: int) -> Path:
    """The band-limited dictionary of one arm (``trained`` or ``untrained``), ``R`` and fold."""
    return root / "dictionaries" / f"B{r}-{arm}_f{fold}"


def check_domain(root: Path) -> None:
    """Record the domain of a new root, or refuse a root of another read.

    Raises
    ------
    ValueError
        If the root records another domain, or holds a read of the baselines protocol.
    """
    path = root / "domain.json"
    if path.is_file():
        if json.loads(path.read_text()) != DOMAIN:
            raise ValueError(f"{root} holds a read of another domain: {path.read_text()}")
        return
    if (root / "reconstruction_decision.json").is_file() or (root / "scores").is_dir():
        raise ValueError(f"{root} holds a read of the baselines protocol; use a new root")
    _save_json(path, DOMAIN)


def check_geometry(model: Any, top_r: int) -> None:
    """Refuse a dictionary whose geometry differs from the protocol's.

    Raises
    ------
    ValueError
        If ``K``, ``L``, ``P``, ``R``, ``D`` or the row domain differ.
    """
    spec = model.spec
    found = (spec.n_atoms, spec.atom_len, spec.pool_len, spec.top_r, spec.n_projections, spec.pre_emphasis)
    wanted = (N_ATOMS, ATOM_LEN, POOL_LEN, top_r, N_SPATIAL, "none")
    if found != wanted:
        raise ValueError(f"dictionary geometry {found} differs from the protocol's {wanted}")


def _context(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    """The readout's plan, configuration, sources and cohort for one fold, with the fold's seed.

    Raises
    ------
    RuntimeError
        Without a CUDA GPU.
    ValueError
        If the rate is not 256 Hz or the cohort differs from the readout's patients.
    """
    from src.frozen_readout import build_cohort, load_plan, load_sources

    if not torch.cuda.is_available():
        raise RuntimeError("run inside the HPC container on an allocated CUDA GPU")
    readout = args.readout_root.resolve()
    plan = load_plan(readout)
    seed = plan["seed"] + fold
    cfg, datamodule, sources, locked, _ = load_sources(args, seed)
    sfreq = float(cfg.data.target_sfreq)
    if sfreq != SFREQ:
        raise ValueError(f"the protocol reads {SFREQ} Hz rows, the configuration gives {sfreq} Hz")
    cohort = build_cohort(sources, locked)
    if set(cohort.labels) != set(plan["labels"]):
        raise ValueError("the cohort differs from the patients of the readout plan")
    return {
        "readout": readout,
        "plan": plan,
        "seed": seed,
        "cfg": cfg,
        "datamodule": datamodule,
        "sources": sources,
        "cohort": cohort,
        "sfreq": sfreq,
    }


def _specs(context: dict[str, Any], top_r: int) -> tuple[Any, Any]:
    import hydra

    spec = replace(hydra.utils.instantiate(context["cfg"].feature.spec), top_r=top_r)
    return spec, hydra.utils.instantiate(context["cfg"].feature.train_spec)


def _model(context: dict[str, Any], top_r: int, folder: Path | None = None) -> Any:
    """A separable model at ``top_r`` on the GPU, loaded from ``folder`` when given (without atoms otherwise)."""
    from src.separable_synthetic import build_model

    model = build_model("s4ext", *_specs(context, top_r), context["seed"], "cuda")
    if folder is not None:
        model.load_checkpoint(folder / CHECKPOINT)
    check_geometry(model, top_r)
    return model


def _fixed_model(context: dict[str, Any], top_r: int, n_channels: int, patterns: np.ndarray) -> Any:
    """A separable model at ``top_r`` whose atoms are the 1-30 Hz bank with the fold's spatial patterns."""
    model = _model(context, top_r)
    model.atoms = torch.nn.Parameter(torch.zeros(N_ATOMS, ATOM_LEN + n_channels * N_SPATIAL, device="cuda"))
    set_fixed_atoms(model, band_bank(), patterns)
    return model


def _view(context: dict[str, Any], rows: list[tuple[str, int]]) -> BandPassed:
    from src.frozen_readout import PooledSegments

    return BandPassed(PooledSegments(context["sources"], rows), context["sfreq"])


def _fixed_view(context: dict[str, Any], patients: list[str]) -> BandPassed:
    """The band-passed fixed segments of ``patients``, patient by patient."""
    from src.frozen_readout import fixed_rows

    return _view(context, fixed_rows(context["cohort"], sorted(patients), context["plan"]["n_segments"]))


# ---------------------------------------------------------------------- stages


def fit_dictionary(args: argparse.Namespace) -> None:
    """Train the band-limited dictionary of one fold at ``--top-r``, or with ``--arm untrained`` keep its random start.

    It mirrors ``frozen_readout.fit_dictionary``: the same dictionary split, seed, settings and loaders, with the
    band-passed view and ``top_r``. A finished arm is skipped, and the arm directory is created only after the data are
    ready.
    """
    import lightning

    from src.frozen_readout import every_row
    from src.utils import split_provenance

    root, fold, r = args.root.resolve(), args.fold, args.top_r
    check_domain(root)
    output = dictionary_dir(root, args.arm, r, fold)
    if (output / CHECKPOINT).is_file():
        logger.info(f"{output.name}: finished")
        return
    context = _context(args, fold)
    plan, cohort, seed = context["plan"], context["cohort"], context["seed"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    train_patients, val_patients = dictionary_split(plan["labels"], plan["folds"], fold, plan["seed"])
    views = {
        "train": _view(context, every_row(cohort, train_patients)),
        "val": _fixed_view(context, val_patients),
    }
    spec, train_spec = _specs(context, r)
    if args.arm == "untrained":
        train_spec = replace(train_spec, epochs=1, lr=0.0)
    provenance = split_provenance(context["datamodule"], context["cfg"].data)
    provenance["train_subjects"] = train_patients
    provenance["band_reconstruction"] = {"fold": fold, "arm": args.arm, "top_r": r, **DOMAIN}
    base = context["datamodule"].train_dataloader()
    loaders = [
        DataLoader(views[split], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for split in ("train", "val")
    ]
    from src.separable_synthetic import build_model

    lightning.seed_everything(seed, workers=True)
    model = build_model("s4ext", spec, train_spec, seed, "cuda")
    check_geometry(model, r)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(
        output / "fit_provenance.json",
        {
            "fold": fold,
            "arm": args.arm,
            "top_r": r,
            "seed": seed,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted": _git("status", "--porcelain").splitlines(),
            **DOMAIN,
        },
    )
    model.fit_unsupervised(*loaders, provenance=provenance)
    model.save_artifacts(output)


def fit_fold(args: argparse.Namespace) -> None:
    """The spatial patterns and the uncentered window PCA of one fold, on band-limited rows of its fit patients."""
    root, fold = args.root.resolve(), args.fold
    check_domain(root)
    path = fit_path(root, fold)
    if path.is_file():
        logger.info(f"fit fold {fold}: finished")
        return
    context = _context(args, fold)
    plan = context["plan"]
    patients, _ = dictionary_split(plan["labels"], plan["folds"], fold, plan["seed"])
    if any(plan["folds"][s] == fold for s in patients):
        raise ValueError(f"a fit patient belongs to fold {fold}")
    model = _model(context, 1)
    window_moment = channel_moment = None
    n_windows = n_samples = 0
    with torch.no_grad():
        for x, _ in DataLoader(_fixed_view(context, patients), batch_size=EXTRACT_BATCH, num_workers=args.num_workers):
            rows, _ = model._full_rows(x)
            if window_moment is None:
                size = rows.shape[1] * POOL_LEN
                window_moment = torch.zeros(size, size, dtype=torch.float64, device=rows.device)
                channel_moment = torch.zeros(rows.shape[1], rows.shape[1], dtype=torch.float64, device=rows.device)
            added = accumulate_moments(rows, window_moment, channel_moment)
            n_windows, n_samples = n_windows + added[0], n_samples + added[1]
    window_moment /= n_windows
    basis, values = top_components(window_moment, max(PCA_KS))
    del window_moment
    patterns, channel_values = top_components(channel_moment / n_samples, N_SPATIAL)
    share = (values.cumsum(0) / values.sum()).cpu().numpy()
    _save_npz(
        path,
        basis=basis.float().cpu().numpy(),
        values=values.cpu().numpy(),
        patterns=patterns.cpu().numpy(),
        channel_values=channel_values.cpu().numpy(),
        patients=_text_array(sorted(patients)),
        n_windows=np.array(n_windows),
    )
    _save_json(
        path.with_suffix(".json"),
        {
            "fold": fold,
            "fit_patients": len(patients),
            "windows": n_windows,
            "training_energy_share": {f"PCA-{k}": float(share[k - 1]) for k in PCA_KS},
            "spatial_energy_share": float((channel_values[:N_SPATIAL].sum() / channel_values.sum()).item()),
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted": _git("status", "--porcelain").splitlines(),
        },
    )
    logger.info(f"fit fold {fold}: {len(patients)} patients, {n_windows} windows, PCA-1 share {share[0]:.3f}")


def score_fold(args: argparse.Namespace) -> None:
    """Reconstruct the fixed segments of one fold's patients with every arm; save energies per segment and band.

    Raises
    ------
    FileNotFoundError
        If the fold's fit or one of its dictionaries is missing.
    ValueError
        If a fit patient is among the scored patients.
    """
    root, fold = args.root.resolve(), args.fold
    check_domain(root)
    path = score_path(root, fold)
    if path.is_file():
        logger.info(f"score fold {fold}: finished")
        return
    needed = [fit_path(root, fold)] + [
        dictionary_dir(root, arm, r, fold) / CHECKPOINT for arm in ("trained", "untrained") for r in RS
    ]
    for item in needed:
        if not item.is_file():
            raise FileNotFoundError(f"{item} is missing: run its stage for fold {fold} first")
    with np.load(fit_path(root, fold)) as fit:
        basis, patterns, fit_patients = fit["basis"], fit["patterns"], set(fit["patients"].tolist())
    context = _context(args, fold)
    plan, sfreq = context["plan"], context["sfreq"]
    scored = sorted(s for s, f in plan["folds"].items() if f == fold)
    if fit_patients & set(scored):
        raise ValueError(f"a fit patient is among the scored patients of fold {fold}")
    models: dict[str, Any] = {}
    for r in RS:
        models[TRAINED[r]] = _model(context, r, dictionary_dir(root, "trained", r, fold))
        models[UNTRAINED[r]] = _model(context, r, dictionary_dir(root, "untrained", r, fold))
    n_channels = models[TRAINED[1]].n_channels
    for r in RS:
        models[FIXED[r]] = _fixed_model(context, r, n_channels, patterns)
    models[PASS1] = _model(context, 1, context["readout"] / "dictionaries" / f"trained_f{fold}")
    dominant = {}
    slow = {}
    for name in BUDGET_ARMS:
        model = models[name]
        dominant[name] = dominant_frequency(model._split(model._dictionary().detach())[0].cpu().numpy(), sfreq)
        slow[name] = torch.as_tensor(dominant[name] < SLOW_ATOM_HZ, device="cuda")
    basis_t = torch.as_tensor(basis, device="cuda")
    view = _fixed_view(context, scored)
    subjects = _text_array(view.plan["subject"])
    parts: dict[str, list[np.ndarray]] = {name: [] for name in SEGMENT_FIELDS}
    violations = np.zeros(len(MODEL_ARMS), dtype=np.int64)
    excess = np.full(len(MODEL_ARMS), -np.inf)
    atom_wins = np.zeros((len(BUDGET_ARMS), N_ATOMS), dtype=np.int64)
    weights, n_times = None, 0
    reader = models[TRAINED[1]]
    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
        with torch.no_grad():
            rows, valid = reader._full_rows(x)
        if weights is None:
            n_times = rows.shape[-1]
            weights = band_weights(n_times, sfreq).to(rows.device)
        result = {
            key: value.cpu().numpy() for key, value in score_arms(rows, valid, models, basis_t, weights, slow).items()
        }
        for name in SEGMENT_FIELDS:
            parts[name].append(result[name])
        violations += result["ceiling_violations"]
        excess = np.maximum(excess, result["ceiling_excess"])
        atom_wins += result["atom_wins"]
    arrays = {name: np.concatenate(chunks, axis=1 if name in SECOND_AXIS else 0) for name, chunks in parts.items()}
    _save_npz(
        path,
        subject=subjects,
        fold=np.array(fold),
        arms=np.array(ARMS),
        model_arms=np.array(MODEL_ARMS),
        budget_arms=np.array(BUDGET_ARMS),
        bands=np.array(BAND_NAMES),
        n_channels=np.array(n_channels),
        windows_per_segment=np.array(n_times // POOL_LEN),
        ceiling_violations=violations,
        ceiling_excess=excess,
        atom_wins=atom_wins,
        dominant_hz=np.stack([dominant[name] for name in BUDGET_ARMS]),
        **arrays,
    )
    evs = {name: pooled_ev(arrays["residual_total"][ARMS.index(name)], arrays["signal_total"]) for name in BUDGET_ARMS}
    logger.info(f"score fold {fold}: {len(scored)} patients, {len(subjects)} segments, EV {evs}")


# ---------------------------------------------------------------------- read


def _validation_ev(root: Path, r: int, fold: int) -> float:
    """The best validation EV of a fold's trained band dictionary at ``R=r``, from its ``sae_training.csv``."""
    path = dictionary_dir(root, "trained", r, fold) / "sae_training.csv"
    return 1.0 - float(pl.read_csv(path)["val_residual_frac"].min())


def _start_gap(root: Path, readout_root: Path, fold: int) -> dict[str, float | None]:
    """Largest waveform difference of this read's random starts from each other and from the pass-1 start."""

    def waveforms(folder: Path) -> np.ndarray | None:
        path = folder / "sae_time_atoms.npy"
        return np.load(path) if path.is_file() else None

    one = waveforms(dictionary_dir(root, "untrained", 1, fold))
    four = waveforms(dictionary_dir(root, "untrained", 4, fold))
    pass1 = waveforms(readout_root / "dictionaries" / f"untrained_f{fold}")

    def gap(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        return None if a is None or b is None or a.shape != b.shape else float(np.abs(a - b).max())

    return {"R1_vs_R4": gap(one, four), "R1_vs_pass1": gap(one, pass1)}


def read(root: Path, readout_root: Path) -> dict[str, Any]:
    """Apply the gates and the rule of protocol sections 6 and 7 to a root with five scored folds.

    Raises
    ------
    FileNotFoundError
        If a fold is not scored.
    ValueError
        If the root records another domain, or a fold file holds other arms or bands.
    """
    path = root / "domain.json"
    if not path.is_file() or json.loads(path.read_text()) != DOMAIN:
        raise ValueError(f"{root} records no band-limited domain")
    plan = json.loads((readout_root / "folds.json").read_text())
    folds_of, labels = plan["folds"], plan["labels"]
    data = []
    for fold in range(N_FOLDS):
        file = score_path(root, fold)
        if not file.is_file():
            raise FileNotFoundError(f"fold {fold} is not scored: run score --fold {fold}")
        with np.load(file) as stored:
            item = {key: stored[key] for key in stored.files}
        layout = (item["arms"].tolist(), item["bands"].tolist(), item["budget_arms"].tolist())
        if layout != (list(ARMS), list(BAND_NAMES), list(BUDGET_ARMS)):
            raise ValueError(f"{file.name} holds other arms or bands")
        data.append(item)

    def stack(name: str) -> np.ndarray:
        return np.concatenate([d[name] for d in data], axis=1 if name in SECOND_AXIS else 0)

    segment_subject = np.concatenate([d["subject"] for d in data]).astype(str)
    segment_fold = np.concatenate([np.full(len(d["subject"]), int(d["fold"])) for d in data])
    patients = np.unique(segment_subject)
    index = np.searchsorted(patients, segment_subject)
    n = len(patients)
    signal_bands, signal_total = stack("signal_bands"), stack("signal_total")
    residual_bands, residual_total = stack("residual_bands"), stack("residual_total")
    e_bands, e_total = _by_patient(signal_bands, index, n), _by_patient(signal_total, index, n)
    r_bands = _by_patient(residual_bands.transpose(1, 0, 2), index, n).transpose(1, 0, 2)
    r_total = _by_patient(residual_total.T, index, n).T
    fold = np.array([folds_of[s] for s in patients])
    label = np.array([labels[s] for s in patients])
    segments = np.bincount(index, minlength=n)
    columns = {"total": (e_total, r_total)}
    columns |= {name: (e_bands[:, b], r_bands[:, :, b]) for b, name in enumerate(BAND_NAMES)}
    columns["in_band"] = (e_bands[:, list(IN_BAND)].sum(axis=1), r_bands[:, :, list(IN_BAND)].sum(axis=2))
    columns["4_30"] = (e_bands[:, list(FAST)].sum(axis=1), r_bands[:, :, list(FAST)].sum(axis=2))
    arm = {name: i for i, name in enumerate(ARMS)}
    counts = bootstrap_counts(fold)

    def ev(name: str, column: str, where: np.ndarray | None = None) -> float:
        signal, residual = columns[column]
        keep = np.ones(n, dtype=bool) if where is None else where
        return pooled_ev(residual[arm[name]][keep], signal[keep])

    def diff(a: str, b: str, column: str) -> Interval:
        signal, residual = columns[column]
        return paired_interval(residual[arm[a]], residual[arm[b]], signal, counts)

    summary = []
    for name in ARMS:
        for column, (signal, residual) in columns.items():
            summary.append(
                {"arm": name, "column": column, **_interval_record(ev_interval(residual[arm[name]], signal, counts))}
            )
    pl.DataFrame(summary).write_csv(root / "band_summary.csv")

    # Gates (section 6); the unit tests run in the launcher before any stage.
    in_band_share = float(columns["in_band"][0].sum() / e_total.sum())
    held_out = {r: {f: ev(TRAINED[r], "total", fold == f) for f in range(N_FOLDS)} for r in RS}
    validation = {r: {f: _validation_ev(root, r, f) for f in range(N_FOLDS)} for r in RS}
    parseval = max(
        parseval_error(signal_bands, signal_total),
        *(parseval_error(residual_bands[i], residual_total[i]) for i in range(len(ARMS))),
    )
    violations = sum(int(d["ceiling_violations"].sum()) for d in data)
    complete = (
        set(patients.tolist()) == set(labels)
        and all(folds_of[s] == f for s, f in zip(segment_subject, segment_fold))
        and bool((segments == plan["n_segments"]).all())
    )
    gates = {
        "complete": complete,
        "band_pass_ran": in_band_share >= IN_BAND_MIN,
        "ceilings": violations == 0,
        "rows_match_training": all(
            abs(held_out[r][f] - validation[r][f]) <= ROW_MATCH_TOLERANCE for r in RS for f in range(N_FOLDS)
        ),
        "training": all(ev(TRAINED[r], "total") > ev(UNTRAINED[r], "total") for r in RS),
        "parseval": parseval <= PARSEVAL_TOLERANCE,
    }

    # Rule (section 7).
    deciding: dict[str, Any] = {}
    met: dict[int, dict[str, bool]] = {}
    for r in RS:
        total_vs_bar = diff(TRAINED[r], BAR[r], "total")
        fast_ev = ev(TRAINED[r], "4_30")
        fast_vs_bar = diff(TRAINED[r], BAR[r], "4_30")
        fast_vs_fixed = diff(TRAINED[r], FIXED[r], "4_30")
        met[r] = arm_conditions(total_vs_bar, fast_ev, fast_vs_bar, fast_vs_fixed)
        deciding[f"R{r}"] = {
            f"T_total_{TRAINED[r]}_minus_{BAR[r]}": _interval_record(total_vs_bar),
            f"F1_4_30_{TRAINED[r]}_ev": _interval_record(
                ev_interval(columns["4_30"][1][arm[TRAINED[r]]], columns["4_30"][0], counts)
            ),
            "F1_floor": FAST_FLOOR,
            f"F2_4_30_{TRAINED[r]}_minus_{BAR[r]}": _interval_record(fast_vs_bar),
            f"F3_4_30_{TRAINED[r]}_minus_{FIXED[r]}": _interval_record(fast_vs_fixed),
        }
    outcome = band_outcome(gates, met)

    # Descriptive reads (section 8).
    curve = [{"k": k, **{column: ev(f"PCA-{k}", column) for column in columns}} for k in PCA_KS]
    pl.DataFrame(curve).write_csv(root / "band_pca_curve.csv")
    windows = int(data[0]["windows_per_segment"])
    budget = {}
    for i, name in enumerate(BUDGET_ARMS):
        won, slow = stack("windows_won")[i].sum(), stack("windows_slow")[i].sum()
        slots, slow_slots = stack("slots_won")[i].sum(), stack("slots_slow")[i].sum()
        explained_all, explained_slow = stack("explained_all")[i].sum(), stack("explained_slow")[i].sum()
        budget[name] = {
            "share_of_windows_with_a_winner": float(won / (len(segment_subject) * windows)),
            "share_of_winning_atoms_that_are_slow": float(slow_slots / max(slots, 1)),
            "share_of_won_windows_won_only_by_slow_atoms": float(slow / max(won, 1)),
            "share_of_explained_energy_in_those_windows": float(explained_slow / explained_all)
            if explained_all
            else None,
            "atoms_without_held_out_wins": [int((d["atom_wins"][i] == 0).sum()) for d in data],
            "slow_atoms": [int((d["dominant_hz"][i] < SLOW_ATOM_HZ).sum()) for d in data],
        }
    ceilings = {r: {c: ev(f"SVD-{r}", c) for c in columns} for r in RS}
    record = {
        "rule": "protocol section 7",
        "outcome": outcome,
        "prediction": PREDICTION,
        "domain": DOMAIN,
        "gates": gates,
        "gate_details": {
            "in_band_energy_share": in_band_share,
            "in_band_minimum": IN_BAND_MIN,
            "held_out_ev": held_out,
            "validation_ev": validation,
            "row_match_tolerance": ROW_MATCH_TOLERANCE,
            "ceiling_violations": violations,
            "ceiling_largest_excess": float(max(float(d["ceiling_excess"].max()) for d in data)),
            "parseval_largest_error": parseval,
        },
        "conditions": {f"R{r}": met[r] for r in RS},
        "deciding": deciding,
        "bootstrap": {"draws": N_BOOTSTRAP, "confidence": CONFIDENCE, "resampled": "patients within fold"},
        "descriptive": {
            "band_energy_share": {c: float(columns[c][0].sum() / e_total.sum()) for c in (*BAND_NAMES, "4_30")},
            "training_total_minus_untrained": {
                f"R{r}": _interval_record(diff(TRAINED[r], UNTRAINED[r], "total")) for r in RS
            },
            "smallest_pca_k_reaching": {
                TRAINED[r]: {
                    c: smallest_k({k: ev(f"PCA-{k}", c) for k in PCA_KS}, ev(TRAINED[r], c)) for c in ("total", "4_30")
                }
                for r in RS
            },
            "share_of_ceiling": {
                TRAINED[r]: {c: ev(TRAINED[r], c) / ceilings[r][c] if ceilings[r][c] > 0 else None for c in columns}
                for r in RS
            },
            "budget": budget,
            "training_in_this_domain_B1_minus_pass1": {
                c: _interval_record(diff(TRAINED[1], PASS1, c)) for c in ("total", "4_30")
            },
            "R4_minus_R1": {c: _interval_record(diff(TRAINED[4], TRAINED[1], c)) for c in ("total", "4_30")},
            "random_start_gaps": {f: _start_gap(root, readout_root, f) for f in range(N_FOLDS)},
            "edge_over_interior_energy_per_second": float(stack("edge_energy").sum() / stack("interior_energy").sum()),
            "selection_optimism": {f"R{r}": {f: validation[r][f] - held_out[r][f] for f in range(N_FOLDS)} for r in RS},
            "ev_per_class": {
                group: {name: {c: ev(name, c, label == value) for c in ("total", "4_30")} for name in ARMS}
                for group, value in (("epilepsy", 1), ("non_epilepsy", 0))
            },
        },
        "counts": {
            "patients": n,
            "segments": len(segment_subject),
            "per_fold_and_class": {
                f: {
                    "epilepsy": int(((fold == f) & (label == 1)).sum()),
                    "non_epilepsy": int(((fold == f) & (label == 0)).sum()),
                }
                for f in range(N_FOLDS)
            },
            "rejected_rows": int(sum((d["n_channels"] - d["usable_rows"]).sum() for d in data)),
        },
    }
    _save_json(root / "band_decision.json", record)
    logger.info(f"Band-limited reconstruction: {outcome} (gates {gates}, conditions {met})")
    return record


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("fit-dictionary", "train or start the band-limited dictionary of one fold"),
        ("fit", "the spatial patterns and the window PCA of one fold"),
        ("score", "the energies of one fold's patients under every arm"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
        if name == "fit-dictionary":
            command.add_argument("--top-r", type=int, required=True, choices=RS)
            command.add_argument("--arm", choices=("trained", "untrained"), required=True)
    reading = commands.add_parser("read", help="apply the gates and the rule")
    reading.add_argument("--root", type=Path, required=True)
    reading.add_argument("--readout-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit-dictionary":
        fit_dictionary(args)
    elif args.command == "fit":
        fit_fold(args)
    elif args.command == "score":
        score_fold(args)
    else:
        read(args.root.resolve(), args.readout_root.resolve())


if __name__ == "__main__":
    main()
