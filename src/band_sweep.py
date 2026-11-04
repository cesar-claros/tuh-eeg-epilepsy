"""Hyperparameter sweep of the band-limited separable dictionary: at a fixed budget, which settings of K, L, D and P
give the best held-out reconstruction in 4-30 Hz, how do they interact, and does the best beat the base?

Protocol: ``documentation/separable_band_sweep_protocol.md``. Run on the HPC from ``code/``. A sweep root records the
band domain and the band-limited root that it reuses (``sweep.json``). ``src/band_sweep_core.py`` holds the pure logic
and the rules; the data path of ``src/band_reconstruction.py`` is reused. Subcommands:

- ``list`` prints the settings that a launcher stage fits or scores.
- ``fit`` trains the dictionary of one setting and fold (``--smoke``: one epoch, for the memory check).
- ``score`` reconstructs the fixed segments of the fold's own patients with one setting.
- ``select`` applies the screen gates and the selection rule on folds 0 and 1 (``selection.json``).
- ``read`` applies the confirmation rule on folds 2 to 4 (``sweep_decision.json``).

The 39 test patients stay locked: the readout's cohort refuses them. No label enters a fit, a score or a selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger
from torch.utils.data import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.band_reconstruction import (  # noqa: E402
    CHECKPOINT,
    DOMAIN,
    EXTRACT_BATCH,
    _context,
    _fixed_view,
    _view,
    check_domain,
    require_deterministic_cublas,
)
from src.band_reconstruction_core import FAST, SLOW_ATOM_HZ  # noqa: E402
from src.band_sweep_core import (  # noqa: E402
    BASE,
    CONFIRM_FOLDS,
    DESCRIPTIVE,
    FACTORS,
    MARGIN,
    SCREEN_FOLDS,
    SECOND_SEED_OFFSET,
    SMOKE,
    Setting,
    check_setting,
    confirmation_outcome,
    effect_intervals,
    factorial,
    healthy,
    largest_near_copy_group,
    parse_setting,
    score_setting,
    screen_settings,
    select,
    separable_spec,
)
from src.frozen_readout_core import dictionary_split  # noqa: E402
from src.reconstruction_baselines import _by_patient, _git, _interval_record, _save_json, _save_npz, _text_array  # noqa: E402
from src.reconstruction_baselines_core import (  # noqa: E402
    BAND_NAMES,
    PARSEVAL_TOLERANCE,
    band_weights,
    bootstrap_counts,
    ev_interval,
    paired_interval,
    parseval_error,
    pooled_ev,
)
from src.separable_factor_analysis import dominant_frequency  # noqa: E402

PREDICTION = "adopt about 50% (protocol section 7)"
ROW_IDENTITY = 1e-9  # gate 3: relative gap of the signal band energies from the band-limited run
PER_SEGMENT = ("signal_bands", "signal_total", "residual_bands", "residual_total", "ceiling_bands", "ceiling_total")
COLUMNS = ("total", *BAND_NAMES, "4_30")


# ---------------------------------------------------------------------- roots and paths


def check_sweep_root(root: Path, band_root: Path) -> None:
    """Record the band domain and the band-limited root of a new sweep root, or refuse a root of another read.

    Raises
    ------
    ValueError
        If the root is a band-limited root, or records another band-limited root.
    """
    if (root / "band_decision.json").is_file():
        raise ValueError(f"{root} is a band-limited root; the sweep needs a root of its own")
    check_domain(root)
    record = {"band_root": str(band_root.resolve())}
    path = root / "sweep.json"
    if path.is_file():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"{root} belongs to another band-limited root: {path.read_text()}")
        return
    _save_json(path, record)


def dictionary_path(root: Path, band_root: Path, setting: Setting, fold: int, smoke: bool = False) -> Path:
    """The dictionary of a setting and fold; the base at seed 1 is the R=4 arm of the band-limited run."""
    if smoke:
        return root / "smoke" / f"{setting.name}_f{fold}"
    if setting == BASE:
        return band_root / "dictionaries" / f"B4-trained_f{fold}"
    return root / "dictionaries" / f"{setting.name}_f{fold}"


def score_file(root: Path, setting: Setting, fold: int) -> Path:
    """The energies of one setting on one fold's patients."""
    return root / "scores" / f"{setting.name}_f{fold}.npz"


def stage_settings(stage: str, root: Path | None = None) -> list[Setting]:
    """The settings of a launcher stage: ``screen-fit``, ``screen-score``, ``confirm-fit`` or ``confirm-score``.

    The confirmation stages read ``selection.json``; with no selected setting they hold no setting.
    """
    if stage == "screen-fit":
        return screen_settings()
    if stage == "screen-score":
        return [BASE, *screen_settings()]
    if root is None:
        raise ValueError("the confirmation stages need the sweep root")
    selected = json.loads((root / "selection.json").read_text())["selected"]
    if selected is None:
        return []
    chosen = parse_setting(selected)
    fits = [chosen, chosen.with_seed(2), BASE.with_seed(2)]
    if stage == "confirm-fit":
        return fits
    if stage == "confirm-score":
        return [BASE, *fits]
    raise ValueError(f"unknown stage {stage!r}")


# ---------------------------------------------------------------------- models


def _seeded(args: argparse.Namespace, setting: Setting, fold: int) -> dict[str, Any]:
    context = _context(args, fold)
    if setting.seed == 2:
        context["seed"] += SECOND_SEED_OFFSET
    return context


def _build(context: dict[str, Any], setting: Setting, epochs: int | None = None) -> Any:
    """An untrained ``SeparableSAE`` of a setting on the GPU, with the campaign's training settings."""
    import hydra

    from src.models.components.separable_sae import SeparableSAE

    spec = separable_spec(setting, hydra.utils.instantiate(context["cfg"].feature.spec))
    train_spec = replace(hydra.utils.instantiate(context["cfg"].feature.train_spec), epochs=epochs or setting.epochs)
    model = SeparableSAE(spec=spec, train_spec=train_spec, random_state=context["seed"], device="cuda")
    check_setting(model, setting)
    return model


def _load(context: dict[str, Any], setting: Setting, folder: Path) -> Any:
    model = _build(context, setting)
    model.load_checkpoint(folder / CHECKPOINT)
    return model


# ---------------------------------------------------------------------- stages


def fit(args: argparse.Namespace) -> None:
    """Train the dictionary of one setting and fold. The base at seed 1 exists and is skipped, as is a finished fit.

    It mirrors ``band_reconstruction.fit_dictionary`` with the setting's geometry, epochs and seed.
    """
    import lightning

    from src.frozen_readout import every_row
    from src.utils import split_provenance

    root, band_root, fold = args.root.resolve(), args.band_root.resolve(), args.fold
    check_sweep_root(root, band_root)
    setting = parse_setting(args.setting)
    output = dictionary_path(root, band_root, setting, fold, args.smoke)
    if (output / CHECKPOINT).is_file():
        logger.info(f"{output.name}: finished")
        return
    require_deterministic_cublas()
    context = _seeded(args, setting, fold)
    plan, cohort, seed = context["plan"], context["cohort"], context["seed"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    train_patients, val_patients = dictionary_split(plan["labels"], plan["folds"], fold, plan["seed"])
    views = {"train": _view(context, every_row(cohort, train_patients)), "val": _fixed_view(context, val_patients)}
    provenance = split_provenance(context["datamodule"], context["cfg"].data)
    provenance["train_subjects"] = train_patients
    provenance["band_sweep"] = {"fold": fold, "setting": setting.name, "smoke": args.smoke, **DOMAIN}
    base = context["datamodule"].train_dataloader()
    loaders = [
        DataLoader(views[split], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for split in ("train", "val")
    ]
    lightning.seed_everything(seed, workers=True)
    model = _build(context, setting, epochs=1 if args.smoke else None)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(
        output / "fit_provenance.json",
        {
            "fold": fold,
            "setting": setting.name,
            "smoke": args.smoke,
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
    if args.smoke:
        logger.info(f"smoke {setting.name}: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def score(args: argparse.Namespace) -> None:
    """Reconstruct the fixed segments of one fold's patients with one setting; save energies per segment and band.

    Raises
    ------
    FileNotFoundError
        If the setting's dictionary is missing.
    """
    root, band_root, fold = args.root.resolve(), args.band_root.resolve(), args.fold
    check_sweep_root(root, band_root)
    setting = parse_setting(args.setting)
    path = score_file(root, setting, fold)
    if path.is_file():
        logger.info(f"score {setting.name} fold {fold}: finished")
        return
    folder = dictionary_path(root, band_root, setting, fold)
    if not (folder / CHECKPOINT).is_file():
        raise FileNotFoundError(f"{folder} holds no dictionary: fit {setting.name} on fold {fold} first")
    context = _seeded(args, setting, fold)
    model = _load(context, setting, folder)
    scored = sorted(s for s, f in context["plan"]["folds"].items() if f == fold)
    view = _fixed_view(context, scored)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in PER_SEGMENT}
    violations, excess = 0, -np.inf
    wins = np.zeros(setting.K, dtype=np.int64)
    weights = None
    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
        with torch.no_grad():
            rows, valid = model._full_rows(x)
        if weights is None:
            weights = band_weights(rows.shape[-1], context["sfreq"]).to(rows.device)
        out = {key: value.cpu().numpy() for key, value in score_setting(rows, valid, model, weights).items()}
        for name in PER_SEGMENT:
            parts[name].append(out[name])
        violations += int(out["ceiling_violations"])
        excess = max(excess, float(out["ceiling_excess"]))
        wins += out["atom_wins"]
    waveforms = model._split(model._dictionary().detach())[0].cpu().numpy()
    arrays = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    _save_npz(
        path,
        subject=_text_array(view.plan["subject"]),
        fold=np.array(fold),
        setting=np.array(setting.name),
        bands=np.array(BAND_NAMES),
        ceiling_violations=np.array(violations),
        ceiling_excess=np.array(excess),
        atom_wins=wins,
        waveforms=waveforms,
        dominant_hz=dominant_frequency(waveforms, context["sfreq"]),
        **arrays,
    )
    ev = pooled_ev(arrays["residual_total"], arrays["signal_total"])
    logger.info(f"score {setting.name} fold {fold}: {len(scored)} patients, total EV {ev:.3f}")


# ---------------------------------------------------------------------- reading


def _fold_file(root: Path, setting: Setting, fold: int) -> dict[str, np.ndarray]:
    path = score_file(root, setting, fold)
    if not path.is_file():
        raise FileNotFoundError(f"{setting.name} is not scored on fold {fold}")
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


class Scores:
    """The per-patient energies of one setting on a set of folds, pooled over each patient's segments."""

    def __init__(self, root: Path, setting: Setting, folds: tuple[int, ...]) -> None:
        self.setting = setting
        self.parts = [_fold_file(root, setting, f) for f in folds]
        subject = np.concatenate([p["subject"] for p in self.parts]).astype(str)
        self.segment_subject = subject
        self.segment_fold = np.concatenate([np.full(len(p["subject"]), int(p["fold"])) for p in self.parts])
        self.patients = np.unique(subject)
        index = np.searchsorted(self.patients, subject)
        self.segments = np.bincount(index, minlength=len(self.patients))

        def pool(name: str) -> np.ndarray:
            return _by_patient(np.concatenate([p[name] for p in self.parts]), index, len(self.patients))

        self.signal = {"total": pool("signal_total"), **self._bands(pool("signal_bands"))}
        self.residual = {"total": pool("residual_total"), **self._bands(pool("residual_bands"))}
        self.ceiling = {"total": pool("ceiling_total"), **self._bands(pool("ceiling_bands"))}

    @staticmethod
    def _bands(values: np.ndarray) -> dict[str, np.ndarray]:
        out = {name: values[:, b] for b, name in enumerate(BAND_NAMES)}
        out["4_30"] = values[:, list(FAST)].sum(axis=1)
        return out

    def ev(self, column: str, where: np.ndarray | None = None) -> float:
        keep = np.ones(len(self.patients), dtype=bool) if where is None else where
        return pooled_ev(self.residual[column][keep], self.signal[column][keep])

    def wins(self) -> list[np.ndarray]:
        return [p["atom_wins"] for p in self.parts]

    def health(self) -> dict[str, Any]:
        """Dead atoms and the largest near-copy group among living atoms, per fold, and the slow share of wins."""
        dead, groups, slow = [], [], []
        for p in self.parts:
            wins = p["atom_wins"]
            living = wins > 0
            dead.append(int((~living).sum()))
            groups.append(largest_near_copy_group(p["waveforms"][living]) if living.any() else 0)
            slow.append(float(wins[p["dominant_hz"] < SLOW_ATOM_HZ].sum() / max(wins.sum(), 1)))
        return {"dead_atoms": dead, "largest_near_copy_group": groups, "slow_share_of_wins": slow}


def _band_arm(band_root: Path, arm: str, folds: tuple[int, ...], column: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient residual and signal of one arm of the band-limited run."""
    parts = []
    for f in folds:
        with np.load(band_root / "scores" / f"fold{f}.npz") as stored:
            parts.append({key: stored[key] for key in stored.files})
    subject = np.concatenate([p["subject"] for p in parts]).astype(str)
    patients = np.unique(subject)
    index = np.searchsorted(patients, subject)
    i = parts[0]["arms"].tolist().index(arm)
    signal_bands = np.concatenate([p["signal_bands"] for p in parts])
    residual_bands = np.concatenate([p["residual_bands"][i] for p in parts])
    if column == "total":
        signal = np.concatenate([p["signal_total"] for p in parts])
        residual = np.concatenate([p["residual_total"][i] for p in parts])
    else:
        cols = list(FAST) if column == "4_30" else [BAND_NAMES.index(column)]
        signal, residual = signal_bands[:, cols].sum(axis=1), residual_bands[:, cols].sum(axis=1)
    return _by_patient(residual, index, len(patients)), _by_patient(signal, index, len(patients))


def _rows_identical(band_root: Path, scores: Scores, folds: tuple[int, ...]) -> float:
    """Largest relative gap of a setting's segment signal band energies from those of the band-limited run."""
    gap = 0.0
    for part, f in zip(scores.parts, folds):
        with np.load(band_root / "scores" / f"fold{f}.npz") as stored:
            if stored["subject"].astype(str).tolist() != part["subject"].astype(str).tolist():
                return float("inf")
            reference = stored["signal_bands"]
        gap = max(gap, float((np.abs(part["signal_bands"] - reference) / np.maximum(reference, 1e-12)).max()))
    return gap


def _training_improved(root: Path, band_root: Path, setting: Setting, folds: tuple[int, ...]) -> bool:
    for f in folds:
        curve = pl.read_csv(dictionary_path(root, band_root, setting, f) / "sae_training.csv")["val_residual_frac"]
        if not float(curve.min()) < float(curve[0]):
            return False
    return True


def _gates(
    root: Path, band_root: Path, scores: dict[str, Scores], folds: tuple[int, ...], plan: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, float]]:
    """The gates of protocol section 6 on one group of folds, and their measured values."""
    expected = {s for s, f in plan["folds"].items() if f in folds}
    first = next(iter(scores.values()))
    identity = max(_rows_identical(band_root, s, folds) for s in scores.values())
    parseval = max(
        parseval_error(
            np.concatenate([p[f"{kind}_bands"] for p in s.parts]), np.concatenate([p[f"{kind}_total"] for p in s.parts])
        )
        for s in scores.values()
        for kind in ("signal", "residual")
    )
    return {
        "complete": all(
            set(s.patients.tolist()) == expected
            and bool((s.segments == plan["n_segments"]).all())
            and all(plan["folds"][p] == f for p, f in zip(s.segment_subject, s.segment_fold))
            for s in scores.values()
        )
        and all(np.array_equal(s.patients, first.patients) for s in scores.values()),
        "rows_identical": identity <= ROW_IDENTITY,
        "ceilings": all(int(p["ceiling_violations"]) == 0 for s in scores.values() for p in s.parts),
        "training_improved": all(_training_improved(root, band_root, s.setting, folds) for s in scores.values()),
        "parseval": parseval <= PARSEVAL_TOLERANCE,
    }, {"row_identity_largest_gap": identity, "parseval_largest_error": parseval}


def _table(scores: dict[str, Scores], base: Scores, counts: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for name, s in scores.items():
        for column in COLUMNS:
            interval = ev_interval(s.residual[column], s.signal[column], counts)
            record = {"setting": name, "column": column, **_interval_record(interval)}
            if s is not base:
                versus = paired_interval(s.residual[column], base.residual[column], s.signal[column], counts)
                record |= {f"minus_base_{k}": v for k, v in _interval_record(versus).items()}
            rows.append(record)
    return rows


def select_stage(root: Path, band_root: Path, readout_root: Path) -> dict[str, Any]:
    """Screen gates, effects and the selection rule of protocol section 7, on folds 0 and 1."""
    plan = json.loads((readout_root / "folds.json").read_text())
    cells = [s for _, s in factorial()]
    scores = {s.name: Scores(root, s, SCREEN_FOLDS) for s in [*cells, *DESCRIPTIVE.values()]}
    base = scores[BASE.name]
    gates, details = _gates(root, band_root, scores, SCREEN_FOLDS, plan)
    counts = bootstrap_counts(np.array([plan["folds"][p] for p in base.patients]))
    signal = base.signal["4_30"]
    versus = {
        s.name: paired_interval(scores[s.name].residual["4_30"], base.residual["4_30"], signal, counts)
        for s in cells[1:]
    }
    health = {s.name: healthy(scores[s.name].wins()) for s in cells[1:]}
    ranked = [(s.name, scores[s.name].ev("4_30")) for s in cells[1:]]
    selected = select(ranked, versus, health) if all(gates.values()) else None
    effects = {
        column: {
            e: _interval_record(i)
            for e, i in effect_intervals(
                np.stack([scores[s.name].residual[column] for s in cells]), base.signal[column], counts
            ).items()
        }
        for column in ("4_30", "13_30", "total")
    }
    pl.DataFrame(_table(scores, base, counts)).write_csv(root / "screen_summary.csv")
    pl.DataFrame([{"column": c, "effect": e, **v} for c, items in effects.items() for e, v in items.items()]).write_csv(
        root / "screen_effects.csv"
    )
    record = {
        "rule": "protocol section 7",
        "gates": gates,
        "gate_details": details,
        "selected": selected,
        "screen_patients": len(base.patients),
        "ranking": sorted(
            (
                {
                    "setting": n,
                    "ev_4_30": ev,
                    "minus_base": _interval_record(versus[n]),
                    "healthy": health[n],
                    "amplitudes_per_second": parse_setting(n).amplitudes_per_second,
                    "bits_per_second": parse_setting(n).bits_per_second,
                }
                for n, ev in ranked
            ),
            key=lambda r: -r["ev_4_30"],
        ),
        "base_ev_4_30": base.ev("4_30"),
        "effects": effects,
        "factors": {k: {"low": v[0], "high": v[1]} for k, v in FACTORS.items()},
        "health": {n: s.health() for n, s in scores.items()},
        "descriptive": _screen_descriptive(root, band_root, scores, base, counts),
    }
    _save_json(root / "selection.json", record)
    logger.info(f"Sweep screen: gates {gates}, selected {selected}")
    return record


def _screen_descriptive(
    root: Path, band_root: Path, scores: dict[str, Scores], base: Scores, counts: np.ndarray
) -> dict[str, Any]:
    """The budget curve, the convergence check and the fixed reference, on the screen patients."""
    curve = []
    for r, name in ((1, None), (4, BASE.name), (8, DESCRIPTIVE["R8"].name), (16, DESCRIPTIVE["R16"].name)):
        point: dict[str, Any] = {"R": r, "amplitudes_per_second": r * 256.0 / 512}
        for column in ("total", "4_30"):
            if name is None:
                residual, signal = _band_arm(band_root, "B1-trained", SCREEN_FOLDS, column)
                point[f"code_{column}"] = pooled_ev(residual, signal)
                ceiling, _ = _band_arm(band_root, "SVD-1", SCREEN_FOLDS, column)
            else:
                point[f"code_{column}"] = scores[name].ev(column)
                ceiling, signal = scores[name].ceiling[column], scores[name].signal[column]
            point[f"ceiling_{column}"] = pooled_ev(ceiling, signal)
            pca, pca_signal = _band_arm(band_root, f"PCA-{r}", SCREEN_FOLDS, column)
            point[f"pca_{column}"] = pooled_ev(pca, pca_signal)
        curve.append(point)
    pl.DataFrame(curve).write_csv(root / "screen_budget_curve.csv")
    e60 = scores[DESCRIPTIVE["E60"].name]
    fixed, fixed_signal = _band_arm(band_root, "B1-fixed", SCREEN_FOLDS, "4_30")
    return {
        "budget_curve": curve,
        "convergence_E60_minus_E30": {
            c: _interval_record(paired_interval(e60.residual[c], base.residual[c], base.signal[c], counts))
            for c in ("total", "4_30")
        },
        "fixed_reference_B1_fixed_ev_4_30": pooled_ev(fixed, fixed_signal),
    }


def read(root: Path, band_root: Path, readout_root: Path) -> dict[str, Any]:
    """The confirmation rule of protocol section 7 on folds 2 to 4, after ``select``; writes ``sweep_decision.json``.

    Raises
    ------
    FileNotFoundError
        If ``selection.json`` or a confirmation score is missing.
    """
    selection = json.loads((root / "selection.json").read_text())
    plan = json.loads((readout_root / "folds.json").read_text())
    record: dict[str, Any] = {"rule": "protocol section 7", "prediction": PREDICTION, "screen": selection}
    if not all(selection["gates"].values()):
        record["outcome"] = "blocked"
    elif selection["selected"] is None:
        record["outcome"] = "base_holds"
        record["reason"] = "no screen setting beat the base with a healthy dictionary"
    else:
        chosen = parse_setting(selection["selected"])
        arms = {
            "base_s1": BASE,
            "base_s2": BASE.with_seed(2),
            "selected_s1": chosen,
            "selected_s2": chosen.with_seed(2),
        }
        scores = {key: Scores(root, s, CONFIRM_FOLDS) for key, s in arms.items()}
        gates, details = _gates(root, band_root, {s.setting.name: s for s in scores.values()}, CONFIRM_FOLDS, plan)
        counts = bootstrap_counts(np.array([plan["folds"][p] for p in scores["base_s1"].patients]))
        signal = scores["base_s1"].signal["4_30"]
        per_seed = {
            seed: paired_interval(
                scores[f"selected_s{seed}"].residual["4_30"], scores[f"base_s{seed}"].residual["4_30"], signal, counts
            )
            for seed in (1, 2)
        }
        health_ok = all(healthy(scores[f"selected_s{seed}"].wins()) for seed in (1, 2))
        outcome = confirmation_outcome(per_seed, health_ok) if all(gates.values()) else "blocked"
        seed_noise = paired_interval(
            scores["base_s2"].residual["4_30"], scores["base_s1"].residual["4_30"], signal, counts
        )
        record |= {
            "outcome": outcome,
            "selected": chosen.name,
            "confirmation_gates": gates,
            "confirmation_gate_details": details,
            "confirmation_patients": len(scores["base_s1"].patients),
            "per_seed_selected_minus_base_4_30": {s: _interval_record(i) for s, i in per_seed.items()},
            "mean_gain_4_30": float(np.mean([i.value for i in per_seed.values()])),
            "margin": MARGIN,
            "selected_healthy": health_ok,
            "descriptive": {
                "seed_noise_base_s2_minus_s1_4_30": _interval_record(seed_noise),
                "ev": {key: {c: s.ev(c) for c in COLUMNS} for key, s in scores.items()},
                "health": {key: s.health() for key, s in scores.items()},
                "per_seed_selected_minus_base_13_30": {
                    seed: _interval_record(
                        paired_interval(
                            scores[f"selected_s{seed}"].residual["13_30"],
                            scores[f"base_s{seed}"].residual["13_30"],
                            scores["base_s1"].signal["13_30"],
                            counts,
                        )
                    )
                    for seed in (1, 2)
                },
            },
        }
    _save_json(root / "sweep_decision.json", record)
    logger.info(f"Sweep: {record['outcome']}")
    return record


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="print the settings of a launcher stage")
    listing.add_argument(
        "--stage", required=True, choices=("smoke", "screen-fit", "screen-score", "confirm-fit", "confirm-score")
    )
    listing.add_argument("--root", type=Path)
    for name, description in (("fit", "train the dictionary of one setting and fold"), ("score", "score one setting")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--band-root", type=Path, required=True, help="the band-limited root")
        command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--fold", type=int, required=True, choices=range(5))
        command.add_argument("--setting", required=True)
        if name == "fit":
            command.add_argument("--smoke", action="store_true", help="one epoch, under smoke/, for the memory check")
    for name, description in (("select", "screen gates and selection"), ("read", "confirmation rule")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--band-root", type=Path, required=True)
        command.add_argument("--readout-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "list":
        settings = SMOKE if args.stage == "smoke" else stage_settings(args.stage, args.root)
        sys.stdout.write("".join(f"{s.name}\n" for s in settings))
    elif args.command == "fit":
        fit(args)
    elif args.command == "score":
        score(args)
    elif args.command == "select":
        select_stage(args.root.resolve(), args.band_root.resolve(), args.readout_root.resolve())
    else:
        read(args.root.resolve(), args.band_root.resolve(), args.readout_root.resolve())


if __name__ == "__main__":
    main()
