"""Hinge redesign of the class-masked separable dictionary: a pilot of the slow hinge step, then the score of the
first protocol with a gate on the use of the epilepsy-only atoms.

Protocol: ``documentation/masked_hinge_redesign_protocol.md``. Run on the HPC from ``code/``; a root holds one pilot
and, after ``go``, one full run. The root of the first run (``documentation/masked_separable_sae_protocol.md``) gives
the reference dictionary without a hinge and the descriptive arm ``Score-mask``. ``src/masked_dictionary_core.py``
holds the pilot rule and the outcome rule; ``src/masked_dictionary.py`` holds the stages that this driver reuses;
``src/models/components/masked_separable_sae.py`` holds the model (``SlowHingeSAE``). Subcommands, in order:

Stage 1, the pilot (fold 0, seed 1):

- ``split`` writes the splits of the first protocol; no corpus.
- ``pilot-fit`` trains one pilot dictionary (arm ``sup`` or ``perm``, one strength).
- ``pilot-check`` applies stage-1 gate 2 to the first pilot dictionary (``pilot_first_fit.json``); it exits with 1
  if the gate fails.
- ``pilot-read`` codes the monitoring patients of fold 0 with the 8 pilot dictionaries and the reference, with no
  mask (``pilot_counts.npz``), then applies the gates and the rule of section 4 (``pilot.json``: ``go`` with the
  strength for stage 2, ``no_working_hinge`` or ``blocked``).

Stage 2, the full run (refused unless the pilot gave ``go``):

- ``fit`` trains one dictionary (arm ``sup2`` or ``perm2``, seed, fold) at the pilot's strength; the dictionaries of
  fold 0 at seed 1 are copies of the pilot's.
- ``score`` writes the counts of one fold's held-out segments (``counts/fold{f}.npz``) and of its monitoring
  patients (``monitor/fold{f}.npz``), both coded with no mask.
- ``gates`` checks gates 2 to 7 of the first protocol and gate 8 (``gates.json``: ``pass``, ``blocked`` or
  ``collapsed``).
- ``read`` applies the rule of section 7 (``masked_hinge_decision.json``). If a gate failed, it writes the outcome
  and opens no score.

The 39 test patients stay locked: the readout's cohort refuses them. The pilot reads only monitoring patients of
fold 0, which are training-side patients of that fold.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rootutils
import torch
from loguru import logger
from torch.utils.data import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

import src.masked_dictionary as first  # noqa: E402
from src.band_code_readout import _aligned, _arm_summary, _scores, load_cohort  # noqa: E402
from src.band_reconstruction import CHECKPOINT, EXTRACT_BATCH, _fixed_view  # noqa: E402
from src.band_sweep import _seeded  # noqa: E402
from src.band_sweep_core import Setting, check_setting, separable_spec  # noqa: E402
from src.class_dictionary import _interval, _load_json, _load_npz, _within_fold_spearman  # noqa: E402
from src.covariate_audit_core import matched_read, pair_mask  # noqa: E402
from src.encoder_check_core import alone_interval, drug_gain  # noqa: E402
from src.frozen_readout_core import N_FOLDS, ArmScores, auroc, paired_interval  # noqa: E402
from src.masked_dictionary_core import (  # noqa: E402
    ETA_GRID,
    GAP_MARGIN,
    MIN_USED,
    N_EPILEPSY_ONLY,
    N_SHARED,
    PILOT_ARMS,
    PILOT_FOLD,
    PILOT_SEED,
    RAMP_EPOCHS,
    REFERENCE,
    SCORES2,
    SEEDS2,
    WARM_EPOCHS,
    eta_label,
    first_fit_check,
    parse_stage2,
    pilot_name,
    pilot_names,
    pilot_reads,
    pilot_rule,
    redesign_outcome,
    share,
    stage2_names,
)
from src.reconstruction_baselines import _save_json, _save_npz, _text_array  # noqa: E402
from src.spectral_decomposition_core import rank_mean  # noqa: E402

PROTOCOL = "documentation/masked_hinge_redesign_protocol.md"
PREDICTION = (
    "no_working_hinge about 65%; if stage 2 runs: no_signal about 22% of the total, separates_only 7%, "
    "collapsed 3%, the other outcomes 3% together (protocol section 7)"
)
FULL_EPOCH = WARM_EPOCHS + RAMP_EPOCHS  # the first epoch at full hinge strength
HINGE = {"hinge_start_epoch": WARM_EPOCHS + 1, "hinge_ramp_epochs": RAMP_EPOCHS}
MONITOR_KEYS = ("atom_wins", "leak_windows")


# ---------------------------------------------------------------------- roots and files


def check_root(root: Path, sweep_root: Path, readout_root: Path) -> None:
    """Record the roots this run uses, or refuse the root of the first run, a root of another kind or other roots.

    Raises
    ------
    ValueError
        If the root holds the first run or a read of another protocol, or records other roots.
    """
    if (root / "masked_dictionary_decision.json").is_file():
        raise ValueError(f"{root} holds the first run; the hinge redesign needs a root of its own")
    first.check_root(root, sweep_root, readout_root)


def pilot_counts_file(root: Path) -> Path:
    """The counts of fold 0's monitoring patients under the pilot dictionaries and the reference."""
    return root / "pilot_counts.npz"


def monitor_file(root: Path, fold: int) -> Path:
    """The counts of one fold's monitoring patients under its stage-2 dictionaries (gate 8)."""
    return root / "monitor" / f"fold{fold}.npz"


def chosen_strength(root: Path) -> float:
    """The strength that the pilot of ``root`` chose for stage 2.

    Raises
    ------
    FileNotFoundError
        If the root has no pilot record.
    ValueError
        If the pilot did not give ``go``.
    """
    path = root / "pilot.json"
    if not path.is_file():
        raise FileNotFoundError(f"{root} has no pilot.json: run the pilot first")
    record = _load_json(path)
    if record["outcome"] != "go":
        raise ValueError(f"the pilot of {root} gave {record['outcome']}: stage 2 does not run")
    return float(record["eta_star"])


# ---------------------------------------------------------------------- model


def _slow(context: dict[str, Any], chosen: Setting, eta: float) -> Any:
    """An untrained ``SlowHingeSAE`` at a hinge strength on the GPU, with the campaign's training settings."""
    import hydra

    from src.models.components.masked_separable_sae import SlowHingeSAE, SlowHingeSpec

    base = separable_spec(chosen, hydra.utils.instantiate(context["cfg"].feature.spec))
    spec = SlowHingeSpec(**asdict(base), n_shared=N_SHARED, hinge_weight=0.0, hinge_strength=float(eta), **HINGE)
    train_spec = replace(hydra.utils.instantiate(context["cfg"].feature.train_spec), epochs=chosen.epochs)
    model = SlowHingeSAE(spec=spec, train_spec=train_spec, random_state=context["seed"], device="cuda")
    check_setting(model, chosen)
    return model


def _loaded(context: dict[str, Any], folder: Path, seed: int, eta: float) -> Any:
    model = _slow(context, replace(first.MASKED, seed=seed), eta)
    model.load_checkpoint(folder / CHECKPOINT)
    return model


@torch.no_grad()
def monitor_counts(
    context: dict[str, Any], models: dict[str, Any], patients: list[str], num_workers: int
) -> dict[str, np.ndarray]:
    """What the epilepsy-only atoms win on the fixed segments of ``patients`` under every model, with no mask."""
    view = _fixed_view(context, list(patients))
    names = list(models)
    reader = models[names[0]]
    parts: dict[tuple[str, str], list[np.ndarray]] = {}
    windows: list[np.ndarray] = []
    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=num_workers):
        rows, valid = reader._full_rows(x)  # the same rows for every dictionary: one row pipeline
        for name in names:
            out = first._segment_counts(models[name], rows, valid)
            for key in MONITOR_KEYS:
                parts.setdefault((key, name), []).append(out[key])
            if name == names[0]:
                windows.append(out["windows"])
    return {
        "subject": _text_array(view.plan["subject"]),
        "names": np.array(names),
        **{key: np.stack([np.concatenate(parts[(key, n)]) for n in names]) for key in MONITOR_KEYS},
        "windows": np.concatenate(windows),
        "selected_epoch": np.array([int(models[n].fit_meta.get("selected_epoch", 0)) for n in names]),
    }


# ---------------------------------------------------------------------- stage 1: the pilot


def split(root: Path, sweep_root: Path, readout_root: Path) -> list[dict[str, Any]]:
    """Write the splits of the first protocol to this root; no corpus."""
    check_root(root, sweep_root, readout_root)
    return first.split(root, sweep_root, readout_root)


def pilot_fit(args: argparse.Namespace) -> None:
    """Train one pilot dictionary: fold 0, seed 1, the arm's labels and one hinge strength."""
    check_root(args.root.resolve(), args.sweep_root, args.readout_root)
    args.fold = PILOT_FOLD
    eta = float(args.eta)
    first.train_dictionary(
        args,
        pilot_name(args.arm, eta),
        replace(first.MASKED, seed=PILOT_SEED),
        lambda context, chosen: _slow(context, chosen, eta),
        args.arm == "perm",
        {"stage": "pilot", "hinge_strength": eta, **HINGE},
    )


def pilot_check(root: Path) -> dict[str, Any]:
    """Stage-1 gate 2 on the training tables of the first pilot dictionary; writes ``pilot_first_fit.json``.

    Raises
    ------
    FileNotFoundError
        If the first pilot dictionary is not finished.
    ValueError
        If its tables do not hold the last epoch with the mask alone.
    """
    folder = first.dictionary_dir(root, pilot_names()[0], PILOT_FOLD)
    if not (folder / CHECKPOINT).is_file():
        raise FileNotFoundError(f"{folder} is not finished: fit the first pilot dictionary")
    table = pl.read_csv(folder / "sae_training.csv").filter(pl.col("epoch") == WARM_EPOCHS)
    won = pl.read_csv(folder / "sae_won_windows.csv").filter(
        (pl.col("epoch") == WARM_EPOCHS) & (pl.col("atom") >= N_SHARED)
    )
    if table.height != 1 or won.height != N_EPILEPSY_ONLY:
        raise ValueError(f"{folder}: the training tables do not hold epoch {WARM_EPOCHS} of every atom")
    leak = float(table["leak_rate"][0])
    wins = [float(v) for v in won.sort("atom")["won_windows"].to_list()]
    checks = first_fit_check(leak, wins)
    record = {
        "dictionary": folder.name,
        "epoch": WARM_EPOCHS,
        "leak_rate": leak,
        "epilepsy_only_won_windows": wins,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "stop",
    }
    _save_json(root / "pilot_first_fit.json", record)
    logger.info(f"first pilot fit: {record['status']} {checks}")
    return record


def pilot_counts(args: argparse.Namespace) -> None:
    """Code fold 0's monitoring patients with the pilot dictionaries and the reference; a finished file is kept.

    Raises
    ------
    FileNotFoundError
        If a pilot dictionary or the reference is missing.
    """
    root, first_root = args.root.resolve(), args.first_root.resolve()
    check_root(root, args.sweep_root, args.readout_root)
    path = pilot_counts_file(root)
    if path.is_file():
        logger.info(f"{path.name}: finished")
        return
    folders = {name: first.dictionary_dir(root, name, PILOT_FOLD) for name in pilot_names()}
    folders[REFERENCE] = first.dictionary_dir(first_root, REFERENCE, PILOT_FOLD)
    missing = [str(f) for f in folders.values() if not (f / CHECKPOINT).is_file()]
    if missing:
        raise FileNotFoundError(f"missing dictionaries: {missing}")
    record = _load_json(first.split_file(root, PILOT_FOLD))
    context = _seeded(args, first.ADOPTED, PILOT_FOLD)
    models = {
        pilot_name(arm, eta): _loaded(context, folders[pilot_name(arm, eta)], PILOT_SEED, eta)
        for eta in ETA_GRID
        for arm in PILOT_ARMS
    }
    models[REFERENCE] = first._loaded(context, first_root, REFERENCE, PILOT_FOLD)
    data = monitor_counts(context, models, record["monitor_patients"], args.num_workers)
    _save_npz(path, **data)
    logger.info(f"{path.name}: {len(data['subject'])} segments, {len(models)} dictionaries")


def pilot_decide(root: Path, first_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """The gates and the rule of section 4 on the pilot counts and the training tables; writes ``pilot.json``.

    Raises
    ------
    ValueError
        If the count file does not hold the pilot dictionaries and the reference.
    """
    data = _load_npz(pilot_counts_file(root))
    names = data["names"].astype(str).tolist()
    if names != [*pilot_names(), REFERENCE]:
        raise ValueError(f"{pilot_counts_file(root)} holds other dictionaries: {names}")
    index = {name: i for i, name in enumerate(names)}
    record = _load_json(first.split_file(root, PILOT_FOLD))
    subject = data["subject"].astype(str)
    monitor = sorted(record["monitor_patients"])
    group = {
        "sup": np.array([plan["labels"][s] for s in subject]),
        "perm": np.array([record["permuted_labels"][s] for s in subject]),
    }

    def reads(name: str, arm: str) -> dict[str, float]:
        i = index[name]
        return pilot_reads(data["atom_wins"][i], data["leak_windows"][i], data["windows"], group[arm])

    reference = {arm: reads(REFERENCE, arm) for arm in PILOT_ARMS}
    by_arm = {arm: {eta: reads(pilot_name(arm, eta), arm) for eta in ETA_GRID} for arm in PILOT_ARMS}
    rules = {arm: pilot_rule(by_arm[arm], reference[arm]["leak"]) for arm in PILOT_ARMS}
    first_fit = pilot_check(root)
    folders = {name: first.dictionary_dir(root, name, PILOT_FOLD) for name in pilot_names()}
    violations = {name: first._violations(folder) for name, folder in folders.items()}
    selected = {name: int(data["selected_epoch"][index[name]]) for name in pilot_names()}
    first_split = first.split_file(first_root, PILOT_FOLD)
    passed = {
        "first_fit": first_fit["status"] == "pass",
        "mask_holds": all(v == 0 for v in violations.values()),
        "reference_in_use": reference["sup"]["use"] >= MIN_USED,
        "reference_leaks": reference["sup"]["leak"] > 0,
        "split_of_first_run": first_split.is_file() and _load_json(first_split) == record,
        "monitor_complete": sorted(set(subject.tolist())) == monitor
        and all(int((subject == s).sum()) == plan["n_segments"] for s in monitor),
        "full_strength_selected": all(epoch >= FULL_EPOCH for epoch in selected.values()),
    }
    rule = rules["sup"]
    outcome = rule["outcome"] if all(passed.values()) else "blocked"
    eta_star = rule["eta_star"] if outcome == "go" else None
    decision = {
        "protocol": f"{PROTOCOL}, section 4",
        "outcome": outcome,
        "eta_star": eta_star,
        "prediction": PREDICTION,
        "gates": passed,
        "first_fit": first_fit,
        "reference": {"dictionary": f"{REFERENCE}_f{PILOT_FOLD}", "root": str(first_root), **reference},
        "reads": {
            arm: {eta_label(eta): {**by_arm[arm][eta], "works": rules[arm]["works"][eta]} for eta in ETA_GRID}
            for arm in PILOT_ARMS
        },
        "control_works_at_eta_star": None if eta_star is None else rules["perm"]["works"][eta_star],
        "monitor_patients": len(monitor),
        "monitor_segments": len(subject),
        "group_1_patients": {arm: int(len(set(subject[group[arm] == 1].tolist()))) for arm in PILOT_ARMS},
        "selected_epoch": selected,
        "mask_violations": violations,
    }
    _save_json(root / "pilot.json", decision)
    logger.info(f"pilot: {outcome}, strength for stage 2 {eta_star} (gates {passed})")
    return decision


# ---------------------------------------------------------------------- stage 2: the full run


def fit(args: argparse.Namespace) -> None:
    """Train one stage-2 dictionary at the pilot's strength. The dictionaries of the pilot's fold and seed are
    copies of the pilot's: the same patients, seed, labels and strength."""
    root, fold = args.root.resolve(), args.fold
    check_root(root, args.sweep_root, args.readout_root)
    eta = chosen_strength(root)
    name = f"{args.arm}_s{args.seed}"
    arm, seed = parse_stage2(name)
    output = first.dictionary_dir(root, name, fold)
    if fold == PILOT_FOLD and seed == PILOT_SEED and not (output / CHECKPOINT).is_file():
        source = first.dictionary_dir(root, pilot_name(arm.removesuffix("2"), eta), PILOT_FOLD)
        shutil.copytree(source, output, dirs_exist_ok=True)
        _save_json(output / "copied_from.json", {"source": source.name, "hinge_strength": eta})
        logger.info(f"{output.name}: copied from {source.name}")
        return
    first.train_dictionary(
        args,
        name,
        replace(first.MASKED, seed=seed),
        lambda context, chosen: _slow(context, chosen, eta),
        arm == "perm2",
        {"stage": "full", "hinge_strength": eta, **HINGE},
    )


def score(args: argparse.Namespace) -> None:
    """The counts of one fold's held-out segments and of its monitoring patients under its 4 dictionaries."""
    root, fold = args.root.resolve(), args.fold
    check_root(root, args.sweep_root, args.readout_root)
    eta = chosen_strength(root)
    names = stage2_names()

    def load(context: dict[str, Any], name: str) -> Any:
        return _loaded(context, first.dictionary_dir(root, name, fold), parse_stage2(name)[1], eta)

    first.score_fold(args, names, load)
    path = monitor_file(root, fold)
    if path.is_file():
        logger.info(f"monitor {path.name}: finished")
        return
    record = _load_json(first.split_file(root, fold))
    context = _seeded(args, first.ADOPTED, fold)
    models = {name: load(context, name) for name in names}
    data = monitor_counts(context, models, record["monitor_patients"], args.num_workers)
    _save_npz(path, **data)
    logger.info(f"monitor {path.name}: {len(data['subject'])} segments")


def gates(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Gates 2 to 7 of the first protocol on the stage-2 dictionaries, and gate 8 on the monitoring counts.

    The status is ``pass``, ``blocked`` (a gate of the first protocol fails, the pilot did not give ``go``, or a
    selected epoch is below full hinge strength) or ``collapsed`` (only gate 8 fails).
    """
    names = stage2_names()
    record = first.gates(root, plan, names)
    pilot = _load_json(root / "pilot.json")
    monitor: dict[str, dict[str, float]] = {}
    selected: dict[str, int] = {}
    complete = True
    for fold in range(N_FOLDS):
        split_record = _load_json(first.split_file(root, fold))
        data = _load_npz(monitor_file(root, fold))
        subject = data["subject"].astype(str)
        patients = sorted(split_record["monitor_patients"])
        complete = complete and data["names"].astype(str).tolist() == names
        complete = complete and sorted(set(subject.tolist())) == patients
        complete = complete and all(int((subject == s).sum()) == plan["n_segments"] for s in patients)
        group = {
            "sup2": np.array([plan["labels"][s] for s in subject]),
            "perm2": np.array([split_record["permuted_labels"][s] for s in subject]),
        }
        for i, name in enumerate(names):
            arm = parse_stage2(name)[0]
            key = f"{name}_f{fold}"
            monitor[key] = pilot_reads(data["atom_wins"][i], data["leak_windows"][i], data["windows"], group[arm])
            selected[key] = int(data["selected_epoch"][i])
    deciding = [key for key in monitor if key.startswith("sup2_")]
    collapsed = sorted(key for key in deciding if monitor[key]["use"] < MIN_USED)
    record["gates"] |= {
        "pilot_go": pilot["outcome"] == "go",
        "monitor_complete": complete,
        "full_strength_selected": all(epoch >= FULL_EPOCH for epoch in selected.values()),
    }
    blocked = not all(record["gates"].values())
    record["gates"]["atoms_in_use"] = not collapsed
    record["status"] = "blocked" if blocked else ("collapsed" if collapsed else "pass")
    record |= {
        "collapsed": collapsed,
        "monitor_reads": monitor,
        "selected_epoch": selected,
        "pilot": {"outcome": pilot["outcome"], "eta_star": pilot["eta_star"]},
    }
    _save_json(root / "gates.json", record)
    logger.info(f"gates: {record['status']} {record['gates']}")
    return record


def read(
    root: Path, readout_root: Path, audit_root: Path, band_code_root: Path, first_root: Path, n_boot: int = 10_000
) -> dict[str, Any]:
    """The rule of section 7 and the descriptive reads; writes ``masked_hinge_decision.json`` and its table. If a
    gate failed, the record holds the outcome (``blocked`` or ``collapsed``) and no score."""
    plan = _load_json(readout_root / "folds.json")
    gate_record = _load_json(root / "gates.json")
    header = {
        "protocol": f"{PROTOCOL}, section 7",
        "prediction": PREDICTION,
        "pilot": gate_record["pilot"],
        "gates": gate_record,
    }
    if gate_record["status"] != "pass":
        outcome = redesign_outcome(gate_record["status"] != "blocked", False, False, False, False)
        record = {"outcome": outcome, **header}
        _save_json(root / "masked_hinge_decision.json", record)
        logger.info(f"masked hinge read: {outcome}; no score is read")
        return record
    cohort = load_cohort(readout_root, audit_root, plan)
    counts = first._counts(root)
    names = counts["names"]
    wins = {n: share(counts["wins_epilepsy"][n], counts["wins_shared"][n]) for n in names}
    energy = {n: share(counts["energy_epilepsy"][n], counts["energy_shared"][n]) for n in names}
    arms = {label: first._arm(counts, wins, arm, seeds, cohort) for label, (arm, seeds) in SCORES2.items()}
    arms["Score-energy"] = first._arm(counts, energy, *SCORES2["Score"], cohort)
    earlier = first._counts(first_root)  # the first run: its arm without a hinge, descriptive
    earlier_wins = {
        n: share(earlier["wins_epilepsy"][n], earlier["wins_shared"][n]) for n in earlier["names"] if "mask" in n
    }
    arms["Score-mask"] = first._arm(earlier, earlier_wins, "mask", (1, 2), cohort)
    roots = {"readout": readout_root, "band_code": band_code_root}
    saved = {
        name: _aligned(_scores(roots[where] / "scores" / f"{name}.csv"), cohort) for name, where in first.SAVED.items()
    }
    spectral = saved["Spectral"]
    separates = alone_interval(arms["Score"], n_boot)
    drug = drug_gain(arms["Score"].score, cohort["label"], cohort["fold"], cohort["status"], n_boot)
    gap = paired_interval(arms["Score"], spectral, n_boot=n_boot)
    met = {"S": separates[1] > 0.5, "G": drug[1] > 0, "C": gap[1] > -GAP_MARGIN}
    outcome = redesign_outcome(True, True, met["S"], met["G"], met["C"])
    fused = ArmScores(
        cohort["subject"],
        cohort["fold"],
        cohort["label"],
        rank_mean([arms["Score"].score, spectral.score], cohort["fold"]),
    )
    summary = [_arm_summary(n, a, cohort, n_boot) for n, a in {**arms, **saved, "Score+Spectral": fused}.items()]
    pl.DataFrame(summary).write_csv(root / "masked_hinge_summary.csv")
    mask = pair_mask(cohort["label"], cohort["age"], cohort["sex"])
    matched = {}
    for name, arm in (("Score", arms["Score"]), ("Spectral", spectral)):
        m = matched_read(arm.score, cohort["label"], cohort["fold"], mask, n_boot=n_boot)
        matched[name] = {"value": m.value, "low": m.low, "high": m.high, "pairs": m.pairs}
    by_fold = [
        auroc(cohort["label"][cohort["fold"] == f], arms["Score"].score[cohort["fold"] == f]) for f in range(N_FOLDS)
    ]
    record = {
        "outcome": outcome,
        **header,
        "conditions": met,
        "deciding": {
            "S_Score_auroc": _interval(separates),
            "G_gain_over_drug": _interval(drug),
            "C_Score_minus_Spectral": _interval(gap),
        },
        "patients": len(cohort["subject"]),
        "known_drug_status": int((cohort["status"] != "unknown").sum()),
        "descriptive": {
            "arms": summary,
            "matched": matched,
            "per_arm": first._per_arm(counts, root, lambda name: parse_stage2(name)[0]),
            # The pilot chose the strength with labels of fold 0's training side; fold 0 is free of that choice.
            "Score_auroc_by_fold": by_fold,
            "Score_auroc_pilot_fold": by_fold[PILOT_FOLD],
            "Score_auroc_other_folds": float(np.mean([v for f, v in enumerate(by_fold) if f != PILOT_FOLD])),
            "Score_minus_Score_perm": _interval(paired_interval(arms["Score"], arms["Score-perm"], n_boot=n_boot)),
            "Score_minus_Score_mask": _interval(paired_interval(arms["Score"], arms["Score-mask"], n_boot=n_boot)),
            "fusion_minus_Spectral": _interval(paired_interval(fused, spectral, n_boot=n_boot)),
            "seed_rank_correlation_within_fold": _within_fold_spearman(arms["Score-s1"], arms["Score-s2"]),
            "rank_correlation_within_fold": {
                name: _within_fold_spearman(arms["Score"], arm) for name, arm in saved.items()
            },
        },
    }
    _save_json(root / "masked_hinge_decision.json", record)
    logger.info(f"masked hinge read: {outcome} (conditions {met})")
    return record


# ---------------------------------------------------------------------- entry point


def _corpus(command: argparse.ArgumentParser) -> None:
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--sweep-root", type=Path, required=True, help="the sweep root (rows reference)")
    command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
    command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
    command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
    command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
    command.add_argument("--num-workers", type=int, default=4)


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("split", help="the splits of the first protocol")
    for flag, text in (("--root", "the root"), ("--sweep-root", "the sweep root"), ("--readout-root", "pass 1 root")):
        command.add_argument(flag, type=Path, required=True, help=text)
    command = commands.add_parser("pilot-fit", help="one pilot dictionary")
    _corpus(command)
    command.add_argument("--arm", choices=PILOT_ARMS, required=True)
    command.add_argument("--eta", type=float, choices=ETA_GRID, required=True, help="the hinge strength")
    command = commands.add_parser("pilot-check", help="stage-1 gate 2 on the first pilot dictionary")
    command.add_argument("--root", type=Path, required=True)
    command = commands.add_parser("pilot-read", help="the reads, the gates and the rule of the pilot")
    _corpus(command)
    command.add_argument("--first-root", type=Path, required=True, help="the root of the first run")
    command = commands.add_parser("fit", help="one stage-2 dictionary of one fold")
    _corpus(command)
    command.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    command.add_argument("--arm", choices=tuple(SEEDS2), required=True)
    command.add_argument("--seed", type=int, choices=(1, 2), required=True)
    command = commands.add_parser("score", help="the counts of one fold")
    _corpus(command)
    command.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
    command = commands.add_parser("gates", help="gates 2 to 8")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True)
    command = commands.add_parser("read", help="the rule and the descriptive reads")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True)
    command.add_argument("--audit-root", type=Path, required=True, help="the covariate audit root")
    command.add_argument("--band-code-root", type=Path, required=True, help="the band-code readout root")
    command.add_argument("--first-root", type=Path, required=True, help="the root of the first run")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "split":
        split(root, args.sweep_root.resolve(), args.readout_root.resolve())
    elif args.command == "pilot-fit":
        pilot_fit(args)
    elif args.command == "pilot-check":
        if pilot_check(root)["status"] != "pass":
            sys.exit(1)
    elif args.command == "pilot-read":
        pilot_counts(args)
        pilot_decide(root, args.first_root.resolve(), _load_json(args.readout_root.resolve() / "folds.json"))
    elif args.command == "fit":
        fit(args)
    elif args.command == "score":
        score(args)
    elif args.command == "gates":
        gates(root, _load_json(args.readout_root.resolve() / "folds.json"))
    else:
        read(
            root,
            args.readout_root.resolve(),
            args.audit_root.resolve(),
            args.band_code_root.resolve(),
            args.first_root.resolve(),
        )


if __name__ == "__main__":
    main()
