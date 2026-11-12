"""Class-conditional dictionaries: does the class whose separable dictionary reconstructs a held-out patient better
predict that patient's class, beyond band power on the same rows and beyond the maintenance drug?

Protocol: ``documentation/class_dictionary_score_protocol.md``. Run on the HPC from ``code/``; a root holds one read and
records the sweep root and the readout root that it reads (``readout_roots.json``). ``src/class_dictionary_core.py``
holds the splits, the energies, the scores and the rule. Subcommands, in order:

- ``split`` writes the patients of every arm of every fold (``splits/fold{f}.json``); no corpus.
- ``fit`` trains one dictionary (arm, seed, fold), or keeps a start (one epoch at learning rate 0).
- ``score`` writes the residual energies of one fold's held-out segments under its 12 dictionaries, with the data
  of gates 3, 5 and 6.
- ``gates`` checks gates 2 to 6 of section 6 (the launcher runs the tests, gate 1) and writes ``gates.json``.
- ``read`` applies the rule of section 7 and the reads of section 8 (``class_dictionary_decision.json``).

The 39 test patients stay locked: the readout's cohort refuses them. Labels enter only the choice of each class
dictionary's patients; no fitted step sees a held-out patient.
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.band_code_readout import _aligned, _arm_summary, _scores, load_cohort  # noqa: E402
from src.band_reconstruction import (  # noqa: E402
    CHECKPOINT,
    DOMAIN,
    EXTRACT_BATCH,
    _fixed_view,
    _view,
    check_domain,
    require_deterministic_cublas,
)
from src.band_sweep import _build, _load, _seeded  # noqa: E402
from src.band_sweep import score_file as sweep_score_file  # noqa: E402
from src.band_sweep_core import Setting, check_setting, separable_spec  # noqa: E402
from src.class_dictionary_core import (  # noqa: E402
    BAND_COLUMNS,
    EPOCHS,
    N_REPEAT,
    PAIRS,
    REPEAT_TOLERANCE,
    ROW_TOLERANCE,
    SEEDS,
    SEGMENT_CAP,
    STARTS,
    TRAINED,
    capped_positions,
    check_split,
    class_split,
    dictionary_names,
    pair_difference,
    parse_name,
    patient_scores,
    recurrence_share,
    residual_shares,
    score_outcome,
    segment_energies,
    training_moved,
)
from src.covariate_audit_core import matched_read, pair_mask  # noqa: E402
from src.encoder_check_core import alone_interval, drug_gain  # noqa: E402
from src.frozen_readout_core import N_FOLDS, ArmScores, paired_interval  # noqa: E402
from src.reconstruction_baselines import _git, _save_json, _save_npz, _text_array  # noqa: E402
from src.reconstruction_baselines_core import band_weights  # noqa: E402

ADOPTED = Setting(K=64, L=128, D=8, P=512, R=4)  # the rows reference: the sweep's adopted dictionary at seed 1
PREDICTION = (
    "no_advantage about 55%, separates_only 25%, review 10%, beats_band_power 6%, score_adds 4% (protocol section 7)"
)
SAVED_BAND = "Band-scaled"  # the saved arm of the band-code readout that B compares with


def setting(name: str) -> Setting:
    """The setting of a dictionary: the adopted geometry, 56 epochs (1 for a start) and the dictionary's seed."""
    arm, seed = parse_name(name)
    return replace(ADOPTED, epochs=1 if arm in STARTS else EPOCHS, seed=seed)


# ---------------------------------------------------------------------- roots and files


def check_root(root: Path, sweep_root: Path, readout_root: Path) -> None:
    """Record the band domain and the roots this read uses, or refuse a root of another kind or other roots.

    Raises
    ------
    ValueError
        If the root holds a read of another protocol, or records other roots.
    """
    for name in ("band_decision.json", "sweep.json", "selection.json", "band_code_decision.json"):
        if (root / name).is_file():
            raise ValueError(f"{root} holds {name}; the class-dictionary read needs a root of its own")
    check_domain(root)
    record = {"sweep_root": str(sweep_root.resolve()), "readout_root": str(readout_root.resolve())}
    path = root / "readout_roots.json"
    if path.is_file():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"{root} reads other roots: {path.read_text()}")
        return
    _save_json(path, record)


def split_file(root: Path, fold: int) -> Path:
    """The patients of every arm of one fold."""
    return root / "splits" / f"fold{fold}.json"


def dictionary_dir(root: Path, name: str, fold: int) -> Path:
    """The folder of one dictionary of one fold."""
    return root / "dictionaries" / f"{name}_f{fold}"


def energy_file(root: Path, fold: int) -> Path:
    """The residual energies of one fold's held-out segments (not ``scores/``: ``check_domain`` reads a ``scores``
    folder as a root of the baselines protocol)."""
    return root / "energies" / f"fold{fold}.npz"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run its stage first")
    return json.loads(path.read_text())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run its stage first")
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


# ---------------------------------------------------------------------- stages


def split(root: Path, sweep_root: Path, readout_root: Path) -> list[dict[str, Any]]:
    """Write the patients of every arm of every fold (section 3); no corpus. A written split must not change.

    Raises
    ------
    ValueError
        If a stored split differs from the plan's.
    """
    check_root(root, sweep_root, readout_root)
    plan = _load_json(readout_root / "folds.json")
    records = []
    for fold in range(N_FOLDS):
        record = class_split(plan["labels"], plan["folds"], fold, plan["seed"])
        record["checks"] = check_split(record, plan["labels"], plan["folds"])
        path = split_file(root, fold)
        if path.is_file() and json.loads(path.read_text()) != json.loads(json.dumps(record)):
            raise ValueError(f"{path} differs from the plan's split")
        _save_json(path, record)
        sizes = {arm: len(part["train"]) for arm, part in record["arms"].items()}
        logger.info(f"split fold {fold}: {sizes}, checks {all(record['checks'].values())}")
        records.append(record)
    return records


def fit(args: argparse.Namespace) -> None:
    """Train one dictionary of one fold, or keep a start; a finished dictionary is skipped.

    It mirrors ``band_sweep.fit``: the seeds, loaders and training settings of the sweep, with the arm's patients, at
    most ``SEGMENT_CAP`` distinct segments per patient, and ``EPOCHS`` epochs (one at learning rate 0 for a start).
    """
    import hydra
    import lightning

    from src.models.components.separable_sae import SeparableSAE
    from src.utils import split_provenance

    root, fold, name = args.root.resolve(), args.fold, f"{args.arm}_s{args.seed}"
    check_root(root, args.sweep_root, args.readout_root)
    arm, _ = parse_name(name)
    output = dictionary_dir(root, name, fold)
    if (output / CHECKPOINT).is_file():
        logger.info(f"{output.name}: finished")
        return
    patients = _load_json(split_file(root, fold))["arms"][arm]
    require_deterministic_cublas()
    chosen = setting(name)
    context = _seeded(args, chosen, fold)
    cohort, seed = context["cohort"], context["seed"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    rows = [cohort.rows[s][i] for s in sorted(patients["train"]) for i in capped_positions(s, cohort.keys[s])]
    views = {"train": _view(context, rows), "val": _fixed_view(context, patients["val"])}
    provenance = split_provenance(context["datamodule"], context["cfg"].data)
    provenance["train_subjects"] = sorted(patients["train"])
    provenance["class_dictionary"] = {"fold": fold, "dictionary": name, "cap": SEGMENT_CAP, **DOMAIN}
    base = context["datamodule"].train_dataloader()
    loaders = [
        DataLoader(views[part], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for part in ("train", "val")
    ]
    lightning.seed_everything(seed, workers=True)
    if arm in STARTS:
        spec = separable_spec(chosen, hydra.utils.instantiate(context["cfg"].feature.spec))
        train_spec = replace(hydra.utils.instantiate(context["cfg"].feature.train_spec), epochs=1, lr=0.0)
        model = SeparableSAE(spec=spec, train_spec=train_spec, random_state=seed, device="cuda")
        check_setting(model, chosen)
    else:
        model = _build(context, chosen)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(
        output / "fit_provenance.json",
        {
            "fold": fold,
            "dictionary": name,
            "setting": chosen.name,
            "seed": seed,
            "train_patients": len(patients["train"]),
            "train_segments": len(rows),
            "val_patients": len(patients["val"]),
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted": _git("status", "--porcelain").splitlines(),
            **DOMAIN,
        },
    )
    model.fit_unsupervised(*loaders, provenance=provenance)
    model.save_artifacts(output)


def score(args: argparse.Namespace) -> None:
    """The residual energies of one fold's held-out segments under its 12 dictionaries, with the rows gap against the
    sweep (gate 3), a repeat of the first batches (gate 5) and the failed loads (gate 6).

    Raises
    ------
    FileNotFoundError
        If a dictionary of the fold is missing.
    ValueError
        If the held-out segments differ from the sweep's score file.
    """
    root, fold = args.root.resolve(), args.fold
    sweep_root = args.sweep_root.resolve()
    check_root(root, sweep_root, args.readout_root)
    path = energy_file(root, fold)
    if path.is_file():
        logger.info(f"{path.name}: finished")
        return
    names = dictionary_names()
    missing = [n for n in names if not (dictionary_dir(root, n, fold) / CHECKPOINT).is_file()]
    if missing:
        raise FileNotFoundError(f"fold {fold} misses the dictionaries {missing}: fit them first")
    context = _seeded(args, ADOPTED, fold)
    models = {n: _load(context, setting(n), dictionary_dir(root, n, fold)) for n in names}
    scored = sorted(s for s, f in context["plan"]["folds"].items() if f == fold)
    view = _fixed_view(context, scored)
    subjects = _text_array(view.plan["subject"])
    reader = models[names[0]]
    weights = None
    parts: dict[str, list[np.ndarray]] = {}
    first: list[torch.Tensor] = []
    failures = 0

    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
        failures += int((x.flatten(1).abs().amax(dim=1) == 0).sum())
        rows, valid = reader._full_rows(x)  # the same rows for every dictionary: one spec
        if weights is None:
            weights = band_weights(rows.shape[-1], context["sfreq"]).to(rows.device)
        if sum(len(t) for t in first) < N_REPEAT:
            first.append(x)
        for name in names:
            out = segment_energies(rows, valid, models[name], weights)
            for key in ("residual_total", "residual_bands", "atom_wins"):
                parts.setdefault(f"{key}:{name}", []).append(out[key])
            if name == names[0]:
                parts.setdefault("signal_total", []).append(out["signal_total"])
                parts.setdefault("signal_bands", []).append(out["signal_bands"])
    arrays = {k: np.concatenate(v) if not k.startswith("atom_wins") else np.sum(v, axis=0) for k, v in parts.items()}
    repeat = torch.cat(first)  # the first whole batches, rescored with the same batch shapes
    again = []
    for start in range(0, len(repeat), EXTRACT_BATCH):
        rows, valid = reader._full_rows(repeat[start : start + EXTRACT_BATCH])
        again.append(segment_energies(rows, valid, models[names[0]], weights)["residual_total"])
    shares = residual_shares(arrays[f"residual_total:{names[0]}"][: len(repeat)], arrays["signal_total"][: len(repeat)])
    repeat_gap = float(
        np.abs(residual_shares(np.concatenate(again), arrays["signal_total"][: len(repeat)]) - shares).max()
    )
    stored = _load_npz(sweep_score_file(sweep_root, ADOPTED, fold))
    if stored["subject"].astype(str).tolist() != subjects.tolist():
        raise ValueError(f"the held-out segments of fold {fold} differ from the sweep's score file")
    rows_gap = float(
        np.max(np.abs(arrays["signal_total"] - stored["signal_total"]) / np.maximum(stored["signal_total"], 1e-12))
    )
    atoms = {n: [t.detach().cpu().numpy() for t in m._split(m._dictionary().detach())] for n, m in models.items()}
    _save_npz(
        path,
        subject=subjects,
        label=np.array([context["plan"]["labels"][s] for s in subjects]),
        fold=np.array(fold),
        names=np.array(names),
        residual_total=np.stack([arrays[f"residual_total:{n}"] for n in names]),
        residual_bands=np.stack([arrays[f"residual_bands:{n}"] for n in names]),
        atom_wins=np.stack([arrays[f"atom_wins:{n}"] for n in names]),
        waveforms=np.stack([atoms[n][0] for n in names]),
        patterns=np.stack([atoms[n][1] for n in names]),
        signal_total=arrays["signal_total"],
        signal_bands=arrays["signal_bands"],
        rows_gap=np.array(rows_gap),
        repeat_gap=np.array(repeat_gap),
        repeat_segments=np.array(len(repeat)),
        load_failures=np.array(failures),
    )
    logger.info(f"{path.name}: {len(subjects)} segments, rows gap {rows_gap:.2e}, repeat gap {repeat_gap:.2e}")


def _curve(folder: Path) -> list[float]:
    path = folder / "sae_training.csv"
    if not path.is_file():
        return []
    return pl.read_csv(path)["val_residual_frac"].to_list()


def gates(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Gates 2 to 6 of section 6 on the split and energy files and the training curves."""
    names = dictionary_names()
    splits = [_load_json(split_file(root, f)) for f in range(N_FOLDS)]
    split_checks = [check_split(s, plan["labels"], plan["folds"]) for s in splits]
    files = [_load_npz(energy_file(root, f)) for f in range(N_FOLDS)]
    subjects = np.concatenate([d["subject"].astype(str) for d in files])
    expected = sorted(plan["labels"])
    complete = (
        sorted(set(subjects.tolist())) == expected
        and all(int((subjects == s).sum()) == plan["n_segments"] for s in expected)
        and all(all(plan["folds"][s] == f for s in d["subject"].astype(str)) for f, d in enumerate(files))
        and all(d["names"].astype(str).tolist() == names for d in files)
        and all((dictionary_dir(root, n, f) / CHECKPOINT).is_file() for n in names for f in range(N_FOLDS))
    )
    trained = [(n, f) for n in names if parse_name(n)[0] in TRAINED for f in range(N_FOLDS)]
    moved = {f"{n}_f{f}": training_moved(_curve(dictionary_dir(root, n, f))) for n, f in trained}
    rows_gap = max(float(d["rows_gap"]) for d in files)
    repeat_gap = max(float(d["repeat_gap"]) for d in files)
    repeat_segments = min(int(d["repeat_segments"]) for d in files)
    failures = sum(int(d["load_failures"]) for d in files)
    passed = {
        "splits": all(all(c.values()) for c in split_checks),
        "rows_match_sweep": rows_gap <= ROW_TOLERANCE,
        "training_moved": all(moved.values()),
        "repeat_run": repeat_gap <= REPEAT_TOLERANCE and repeat_segments >= N_REPEAT,
        "complete": complete and failures == 0,
    }
    record = {
        "status": "pass" if all(passed.values()) else "blocked",
        "gates": passed,
        "split_checks": split_checks,
        "not_moved": sorted(k for k, v in moved.items() if not v),
        "rows_gap": rows_gap,
        "repeat_gap": repeat_gap,
        "repeat_segments": repeat_segments,
        "load_failures": failures,
    }
    _save_json(root / "gates.json", record)
    logger.info(f"gates: {record['status']} {passed}")
    return record


# ---------------------------------------------------------------------- read


def _energies(root: Path) -> dict[str, Any]:
    """The held-out segments of all folds with their residual shares per dictionary (total and per band)."""
    files = [_load_npz(energy_file(root, f)) for f in range(N_FOLDS)]
    names = files[0]["names"].astype(str).tolist()
    total = {
        n: np.concatenate([residual_shares(d["residual_total"][i], d["signal_total"]) for d in files])
        for i, n in enumerate(names)
    }
    bands = {
        band: {
            n: np.concatenate(
                [residual_shares(d["residual_bands"][i][:, column], d["signal_bands"][:, column]) for d in files]
            )
            for i, n in enumerate(names)
        }
        for band, column in BAND_COLUMNS.items()
    }
    return {
        "files": files,
        "subject": np.concatenate([d["subject"].astype(str) for d in files]),
        "fold": np.concatenate([np.full(len(d["subject"]), int(d["fold"])) for d in files]),
        "total": total,
        "bands": bands,
    }


def _arm(
    energies: dict[str, Any],
    shares: dict[str, np.ndarray],
    pair: tuple[str, str],
    seeds: tuple[int, ...],
    cohort: dict[str, Any],
) -> ArmScores:
    patients, values = patient_scores(pair_difference(shares, *pair, seeds), energies["subject"])
    if patients.tolist() != cohort["subject"].tolist():
        raise ValueError("the scored patients differ from the plan's")
    return ArmScores(cohort["subject"], cohort["fold"], cohort["label"], values)


def _within_fold_spearman(first: ArmScores, second: ArmScores) -> float:
    values = [
        float(spearmanr(first.score[first.fold == f], second.score[second.fold == f]).statistic)
        for f in np.unique(first.fold)
    ]
    return float(np.mean(values))


def _seed_null(band_code_root: Path, cohort: dict[str, Any]) -> ArmScores:
    """Section 2: the residual difference of the adopted seed-2 and seed-1 dictionaries, which saw the same patients."""
    values: dict[str, float] = {}
    for fold in range(N_FOLDS):
        one = _load_npz(band_code_root / "features" / f"trained_f{fold}.npz")
        two = _load_npz(band_code_root / "features" / f"seed2_f{fold}.npz")
        subject = one["subject"].astype(str)
        held = np.array([cohort["fold_of"][s] == fold for s in subject])
        difference = residual_shares(two["residual"], two["signal"]) - residual_shares(one["residual"], one["signal"])
        patients, means = patient_scores(difference[held], subject[held])
        values |= dict(zip(patients.tolist(), means.tolist(), strict=True))
    return ArmScores(
        cohort["subject"], cohort["fold"], cohort["label"], np.array([values[s] for s in cohort["subject"]])
    )


def _interval(values: tuple[float, float, float]) -> dict[str, float]:
    return dict(zip(("value", "low", "high"), values, strict=True))


def read(
    root: Path, readout_root: Path, audit_root: Path, band_code_root: Path, n_boot: int = 10_000
) -> dict[str, Any]:
    """The rule of section 7 and the reads of section 8; writes ``class_dictionary_decision.json`` and its table."""
    plan = _load_json(readout_root / "folds.json")
    gate_record = _load_json(root / "gates.json")
    cohort = load_cohort(readout_root, audit_root, plan)
    cohort["fold_of"] = plan["folds"]
    energies = _energies(root)
    total = energies["total"]
    arms = {name: _arm(energies, total, pair, SEEDS, cohort) for name, pair in PAIRS.items()}
    for seed in SEEDS:
        arms[f"Score-s{seed}"] = _arm(energies, total, PAIRS["Score"], (seed,), cohort)
    for band, shares in energies["bands"].items():
        arms[f"Score-{band}"] = _arm(energies, shares, PAIRS["Score"], SEEDS, cohort)
    arms["Seed-null"] = _seed_null(band_code_root, cohort)
    band = _aligned(_scores(band_code_root / "scores" / f"{SAVED_BAND}.csv"), cohort)
    separates = alone_interval(arms["Score"], n_boot)
    beyond_band = paired_interval(arms["Score"], band, n_boot=n_boot)
    drug = drug_gain(arms["Score"].score, cohort["label"], cohort["fold"], cohort["status"], n_boot)
    met = {"S": separates[1] > 0.5, "B": beyond_band[1] > 0, "G": drug[1] > 0}
    outcome = score_outcome(gate_record["status"] == "pass", met["S"], met["B"], met["G"])
    summary = [_arm_summary(n, a, cohort, n_boot) for n, a in {**arms, SAVED_BAND: band}.items()]
    pl.DataFrame(summary).write_csv(root / "class_dictionary_summary.csv")
    mask = pair_mask(cohort["label"], cohort["age"], cohort["sex"])
    matched = {}
    for name, arm in (("Score", arms["Score"]), (SAVED_BAND, band)):
        m = matched_read(arm.score, cohort["label"], cohort["fold"], mask, n_boot=n_boot)
        matched[name] = {"value": m.value, "low": m.low, "high": m.high, "pairs": m.pairs}
    record = {
        "protocol": "documentation/class_dictionary_score_protocol.md, section 7",
        "outcome": outcome,
        "prediction": PREDICTION,
        "gates": gate_record,
        "conditions": met,
        "deciding": {
            "S_Score_auroc": _interval(separates),
            "B_Score_minus_Band_scaled": _interval(beyond_band),
            "G_gain_over_drug": _interval(drug),
        },
        "patients": len(cohort["subject"]),
        "known_drug_status": int((cohort["status"] != "unknown").sum()),
        "descriptive": {
            "arms": summary,
            "matched": matched,
            "Score_minus_Score_start": _interval(paired_interval(arms["Score"], arms["Score-start"], n_boot=n_boot)),
            "seed_rank_correlation_within_fold": _within_fold_spearman(arms["Score-s1"], arms["Score-s2"]),
            "band_scaled_rank_correlation_within_fold": _within_fold_spearman(arms["Score"], band),
            "recurrence": _recurrence(energies["files"]),
            "dead_atom_share": _dead_atoms(energies["files"]),
        },
    }
    _save_json(root / "class_dictionary_decision.json", record)
    logger.info(f"class-dictionary read: {outcome} (conditions {met})")
    return record


def _recurrence(files: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Section 8: the share of the atoms of one dictionary that recur in its pair, per fold and seed."""
    rows = []
    for d in files:
        names = d["names"].astype(str).tolist()
        atoms = {n: (d["waveforms"][i], d["patterns"][i]) for i, n in enumerate(names)}
        for seed in SEEDS:
            rows.append(
                {
                    "fold": int(d["fold"]),
                    "seed": seed,
                    "epi_in_non": recurrence_share(atoms[f"epi_s{seed}"], atoms[f"non_s{seed}"]),
                    "A_in_B": recurrence_share(atoms[f"A_s{seed}"], atoms[f"B_s{seed}"]),
                }
            )
    return {
        "rows": rows,
        "mean_epi_in_non": float(np.mean([r["epi_in_non"] for r in rows])),
        "mean_A_in_B": float(np.mean([r["A_in_B"] for r in rows])),
    }


def _dead_atoms(files: list[dict[str, np.ndarray]]) -> dict[str, float]:
    """Section 8: the largest share, over the folds, of atoms that win no held-out window, per dictionary."""
    names = files[0]["names"].astype(str).tolist()
    return {n: max(float((d["atom_wins"][i] == 0).mean()) for d in files) for i, n in enumerate(names)}


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("split", help="the patients of every arm of every fold")
    for flag, text in (("--root", "the root"), ("--sweep-root", "the sweep root"), ("--readout-root", "pass 1 root")):
        command.add_argument(flag, type=Path, required=True, help=text)
    for name, description in (("fit", "one dictionary of one fold"), ("score", "the energies of one fold")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--sweep-root", type=Path, required=True, help="the sweep root (rows reference)")
        command.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
        command.add_argument("--manifests", type=Path, required=True, help="the 40 s offset audit")
        command.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
        command.add_argument("--signal-cache-dir", type=Path, required=True, help="the campaign signal cache")
        command.add_argument("--num-workers", type=int, default=4)
        command.add_argument("--fold", type=int, required=True, choices=range(N_FOLDS))
        if name == "fit":
            command.add_argument("--arm", choices=(*TRAINED, *STARTS), required=True)
            command.add_argument("--seed", type=int, choices=SEEDS, required=True)
    command = commands.add_parser("gates", help="gates 2 to 6")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True)
    command = commands.add_parser("read", help="the rule and the descriptive reads")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--readout-root", type=Path, required=True)
    command.add_argument("--audit-root", type=Path, required=True, help="the covariate audit root")
    command.add_argument("--band-code-root", type=Path, required=True, help="the band-code readout root")
    args = parser.parse_args()
    if args.command == "split":
        split(args.root.resolve(), args.sweep_root.resolve(), args.readout_root.resolve())
    elif args.command == "fit":
        fit(args)
    elif args.command == "score":
        score(args)
    elif args.command == "gates":
        gates(args.root.resolve(), _load_json(args.readout_root.resolve() / "folds.json"))
    else:
        read(args.root.resolve(), args.readout_root.resolve(), args.audit_root.resolve(), args.band_code_root.resolve())


if __name__ == "__main__":
    main()
