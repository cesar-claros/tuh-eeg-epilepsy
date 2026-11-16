"""Class-masked separable dictionary: does the share of windows that the epilepsy-only atoms win on a held-out
patient predict that patient's class?

Protocol: ``documentation/masked_separable_sae_protocol.md``. Run on the HPC from ``code/``; a root holds one read and
records the sweep root and the readout root that it reads (``readout_roots.json``). ``src/masked_dictionary_core.py``
holds the arms, the permuted labels, the counts and the rule; ``src/models/components/masked_separable_sae.py`` holds
the model. Subcommands, in order:

- ``split`` writes the dictionary and monitoring patients of every fold with the permuted labels of the control
  (``splits/fold{f}.json``); no corpus.
- ``fit`` trains one dictionary (arm, seed, fold).
- ``score`` codes one fold's held-out segments with every dictionary of the fold and no mask, and writes what each
  atom group wins (``counts/fold{f}.npz``), with the data of gates 3, 6 and 7.
- ``gates`` checks gates 2 to 7 of section 6 (the launcher runs the tests, gate 1) and writes ``gates.json``.
- ``read`` applies the rule of section 7 and the reads of section 8 (``masked_dictionary_decision.json``).

The 39 test patients stay locked: the readout's cohort refuses them. Labels enter the training of a dictionary through
the mask and the hinge; no fitted step sees a held-out patient, and a held-out segment is coded without a mask.
"""

from __future__ import annotations

import argparse
import json
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
from src.band_sweep import _seeded  # noqa: E402
from src.band_sweep import score_file as sweep_score_file  # noqa: E402
from src.band_sweep_core import Setting, check_setting, separable_spec  # noqa: E402
from src.class_dictionary import _curve, _interval, _load_json, _load_npz, _within_fold_spearman  # noqa: E402
from src.class_dictionary_core import capped_positions  # noqa: E402
from src.covariate_audit_core import matched_read, pair_mask  # noqa: E402
from src.encoder_check_core import alone_interval, drug_gain  # noqa: E402
from src.frozen_readout_core import N_FOLDS, ArmScores, paired_interval  # noqa: E402
from src.masked_dictionary_core import (  # noqa: E402
    EPOCHS,
    GAP_MARGIN,
    HINGE,
    N_EPILEPSY_ONLY,
    N_REPEAT,
    N_SHARED,
    REPEAT_TOLERANCE,
    ROW_TOLERANCE,
    SCORES,
    SEEDS,
    SEGMENT_CAP,
    check_split,
    dictionary_names,
    group_counts,
    masked_outcome,
    masked_split,
    parse_name,
    patient_scores,
    seed_mean,
    share,
    training_moved,
)
from src.reconstruction_baselines import _git, _save_json, _save_npz, _text_array  # noqa: E402
from src.spectral_decomposition_core import rank_mean  # noqa: E402

ADOPTED = Setting(K=64, L=128, D=8, P=512, R=4)  # the rows reference: the sweep's adopted dictionary at seed 1
MASKED = Setting(K=N_SHARED + N_EPILEPSY_ONLY, L=128, D=8, P=512, R=4, epochs=EPOCHS)
PREDICTION = "no_signal about 70%, separates_only 17%, adds_to_drug 6%, review 4%, closes_gap 3% (protocol section 7)"
SAVED = {"Spectral": "readout", "Band-scaled": "band_code"}  # the saved arms and the root that holds each
MATCHED = ("Score", "Spectral")
PER_SEGMENT = ("wins_shared", "wins_epilepsy", "energy_shared", "energy_epilepsy", "leak_windows", "residual_total")


def setting(name: str) -> Setting:
    """The setting of a dictionary: the masked geometry with the dictionary's seed."""
    return replace(MASKED, seed=parse_name(name)[1])


# ---------------------------------------------------------------------- roots and files


def check_root(root: Path, sweep_root: Path, readout_root: Path) -> None:
    """Record the band domain and the roots this read uses, or refuse a root of another kind or other roots.

    Raises
    ------
    ValueError
        If the root holds a read of another protocol, or records other roots.
    """
    others = ("band_decision.json", "sweep.json", "selection.json", "band_code_decision.json")
    for name in (*others, "class_dictionary_decision.json", "fusion_decision.json"):
        if (root / name).is_file():
            raise ValueError(f"{root} holds {name}; the masked-dictionary read needs a root of its own")
    check_domain(root)
    record = {"sweep_root": str(sweep_root.resolve()), "readout_root": str(readout_root.resolve())}
    path = root / "readout_roots.json"
    if path.is_file():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"{root} reads other roots: {path.read_text()}")
        return
    _save_json(path, record)


def split_file(root: Path, fold: int) -> Path:
    """The patients and the permuted labels of one fold."""
    return root / "splits" / f"fold{fold}.json"


def dictionary_dir(root: Path, name: str, fold: int) -> Path:
    """The folder of one dictionary of one fold."""
    return root / "dictionaries" / f"{name}_f{fold}"


def count_file(root: Path, fold: int) -> Path:
    """What each atom group wins on one fold's held-out segments (not ``scores/``: ``check_domain`` reads a
    ``scores`` folder as a root of the baselines protocol)."""
    return root / "counts" / f"fold{fold}.npz"


# ---------------------------------------------------------------------- stages


def split(root: Path, sweep_root: Path, readout_root: Path) -> list[dict[str, Any]]:
    """Write the patients and the permuted labels of every fold (section 3); no corpus. A written split must not
    change.

    Raises
    ------
    ValueError
        If a stored split differs from the plan's.
    """
    check_root(root, sweep_root, readout_root)
    plan = _load_json(readout_root / "folds.json")
    records = []
    for fold in range(N_FOLDS):
        record = masked_split(plan["labels"], plan["folds"], fold, plan["seed"])
        record["checks"] = check_split(record, plan["labels"], plan["folds"])
        path = split_file(root, fold)
        if path.is_file() and json.loads(path.read_text()) != json.loads(json.dumps(record)):
            raise ValueError(f"{path} differs from the plan's split")
        _save_json(path, record)
        sizes = (len(record["dictionary_patients"]), len(record["monitor_patients"]))
        logger.info(f"split fold {fold}: {sizes}, checks {all(record['checks'].values())}")
        records.append(record)
    return records


def _model(context: dict[str, Any], chosen: Setting, arm: str) -> Any:
    """An untrained ``MaskedSeparableSAE`` of an arm on the GPU, with the campaign's training settings."""
    import hydra

    from src.models.components.masked_separable_sae import MaskedSeparableSAE, MaskedSeparableSpec

    base = separable_spec(chosen, hydra.utils.instantiate(context["cfg"].feature.spec))
    spec = MaskedSeparableSpec(**asdict(base), n_shared=N_SHARED, hinge_weight=HINGE[arm])
    train_spec = replace(hydra.utils.instantiate(context["cfg"].feature.train_spec), epochs=chosen.epochs)
    model = MaskedSeparableSAE(spec=spec, train_spec=train_spec, random_state=context["seed"], device="cuda")
    check_setting(model, chosen)
    return model


def fit(args: argparse.Namespace) -> None:
    """Train one dictionary of one fold; a finished dictionary is skipped.

    It mirrors ``class_dictionary.fit``: the seeds, loaders and training settings of the sweep, with all dictionary
    patients, at most ``SEGMENT_CAP`` distinct segments per patient, and the arm's hinge weight. The control arm
    trains on the permuted labels of the split.
    """
    import lightning

    from src.utils import split_provenance

    root, fold, name = args.root.resolve(), args.fold, f"{args.arm}_s{args.seed}"
    check_root(root, args.sweep_root, args.readout_root)
    arm, _ = parse_name(name)
    output = dictionary_dir(root, name, fold)
    if (output / CHECKPOINT).is_file():
        logger.info(f"{output.name}: finished")
        return
    record = _load_json(split_file(root, fold))
    require_deterministic_cublas()
    chosen = setting(name)
    context = _seeded(args, chosen, fold)
    cohort, seed = context["cohort"], context["seed"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    train = sorted(record["dictionary_patients"])
    rows = [cohort.rows[s][i] for s in train for i in capped_positions(s, cohort.keys[s], SEGMENT_CAP)]
    views = {"train": _view(context, rows), "val": _fixed_view(context, record["monitor_patients"])}
    provenance = split_provenance(context["datamodule"], context["cfg"].data)
    provenance["train_subjects"] = train
    provenance["masked_dictionary"] = {"fold": fold, "dictionary": name, "cap": SEGMENT_CAP, **DOMAIN}
    base = context["datamodule"].train_dataloader()
    loaders = [
        DataLoader(views[part], batch_size=base.batch_size, num_workers=base.num_workers, pin_memory=base.pin_memory)
        for part in ("train", "val")
    ]
    lightning.seed_everything(seed, workers=True)
    model = _model(context, chosen, arm)
    if arm == "perm":
        model.label_override = dict(record["permuted_labels"])
    output.mkdir(parents=True, exist_ok=True)
    _save_json(
        output / "fit_provenance.json",
        {
            "fold": fold,
            "dictionary": name,
            "setting": chosen.name,
            "seed": seed,
            "n_shared": N_SHARED,
            "hinge_weight": HINGE[arm],
            "labels": "permuted" if arm == "perm" else "true",
            "train_patients": len(train),
            "train_segments": len(rows),
            "val_patients": len(record["monitor_patients"]),
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted": _git("status", "--porcelain").splitlines(),
            **DOMAIN,
        },
    )
    model.fit_unsupervised(*loaders, provenance=provenance)
    model.save_artifacts(output)


@torch.no_grad()
def _segment_counts(model: Any, rows: torch.Tensor, valid: torch.Tensor) -> dict[str, np.ndarray]:
    """What each atom group wins on one batch, coded without a mask, with the residual energy per segment."""
    code, _, reconstruction, _, _ = model._encode_reconstruct(rows, valid, model._dictionary())
    residual = rows - reconstruction * valid.unsqueeze(-1).to(rows.dtype)
    counts = group_counts(code, N_SHARED, model.spec.pool_len)
    return counts | {"residual_total": residual.double().pow(2).sum(dim=(1, 2)).cpu().numpy()}


def score(args: argparse.Namespace) -> None:
    """The counts of one fold's held-out segments under its 7 dictionaries, with the rows gap against the sweep
    (gate 3), a repeat of the first batches (gate 6) and the failed loads (gate 7).

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
    path = count_file(root, fold)
    if path.is_file():
        logger.info(f"{path.name}: finished")
        return
    names = dictionary_names()
    missing = [n for n in names if not (dictionary_dir(root, n, fold) / CHECKPOINT).is_file()]
    if missing:
        raise FileNotFoundError(f"fold {fold} misses the dictionaries {missing}: fit them first")
    context = _seeded(args, ADOPTED, fold)
    models = {}
    for name in names:
        models[name] = _model(context, setting(name), parse_name(name)[0])
        models[name].load_checkpoint(dictionary_dir(root, name, fold) / CHECKPOINT)
    scored = sorted(s for s, f in context["plan"]["folds"].items() if f == fold)
    view = _fixed_view(context, scored)
    subjects = _text_array(view.plan["subject"])
    reader = models[names[0]]
    parts: dict[str, list[np.ndarray]] = {}
    first: list[torch.Tensor] = []
    failures = 0
    for x, _ in DataLoader(view, batch_size=EXTRACT_BATCH, shuffle=False, num_workers=args.num_workers):
        failures += int((x.flatten(1).abs().amax(dim=1) == 0).sum())
        rows, valid = reader._full_rows(x)  # the same rows for every dictionary: one row pipeline
        if sum(len(t) for t in first) < N_REPEAT:
            first.append(x)
        parts.setdefault("signal_total", []).append(rows.double().pow(2).sum(dim=(1, 2)).cpu().numpy())
        for name in names:
            out = _segment_counts(models[name], rows, valid)
            for key in (*PER_SEGMENT, "atom_wins"):
                parts.setdefault(f"{key}:{name}", []).append(out[key])
            if name == names[0]:
                parts.setdefault("windows", []).append(out["windows"])
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    repeat = torch.cat(first)  # the first whole batches, coded again with the same batch shapes
    again: dict[str, list[np.ndarray]] = {"wins_epilepsy": [], "wins_shared": []}
    for start in range(0, len(repeat), EXTRACT_BATCH):
        rows, valid = reader._full_rows(repeat[start : start + EXTRACT_BATCH])
        out = _segment_counts(models[names[0]], rows, valid)
        for key, values in again.items():
            values.append(out[key])
    stored_share = share(
        arrays[f"wins_epilepsy:{names[0]}"][: len(repeat)], arrays[f"wins_shared:{names[0]}"][: len(repeat)]
    )
    repeat_share = share(np.concatenate(again["wins_epilepsy"]), np.concatenate(again["wins_shared"]))
    repeat_gap = float(np.abs(repeat_share - stored_share).max())
    stored = _load_npz(sweep_score_file(sweep_root, ADOPTED, fold))
    if stored["subject"].astype(str).tolist() != subjects.tolist():
        raise ValueError(f"the held-out segments of fold {fold} differ from the sweep's score file")
    rows_gap = float(
        np.max(np.abs(arrays["signal_total"] - stored["signal_total"]) / np.maximum(stored["signal_total"], 1e-12))
    )
    _save_npz(
        path,
        subject=subjects,
        label=np.array([context["plan"]["labels"][s] for s in subjects]),
        fold=np.array(fold),
        names=np.array(names),
        **{key: np.stack([arrays[f"{key}:{n}"] for n in names]) for key in (*PER_SEGMENT, "atom_wins")},
        windows=arrays["windows"],
        signal_total=arrays["signal_total"],
        rows_gap=np.array(rows_gap),
        repeat_gap=np.array(repeat_gap),
        repeat_segments=np.array(len(repeat)),
        load_failures=np.array(failures),
    )
    logger.info(f"{path.name}: {len(subjects)} segments, rows gap {rows_gap:.2e}, repeat gap {repeat_gap:.2e}")


def _violations(folder: Path) -> float | None:
    """Gate 4: the code entries of epilepsy-only atoms on label-0 training segments, summed over the epochs."""
    path = folder / "sae_training.csv"
    if not path.is_file():
        return None
    table = pl.read_csv(path)
    return float(table["mask_violations"].sum()) if "mask_violations" in table.columns else None


def gates(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Gates 2 to 7 of section 6 on the split and count files and the training tables."""
    names = dictionary_names()
    splits = [_load_json(split_file(root, f)) for f in range(N_FOLDS)]
    split_checks = [check_split(s, plan["labels"], plan["folds"]) for s in splits]
    files = [_load_npz(count_file(root, f)) for f in range(N_FOLDS)]
    subjects = np.concatenate([d["subject"].astype(str) for d in files])
    expected = sorted(plan["labels"])
    complete = (
        sorted(set(subjects.tolist())) == expected
        and all(int((subjects == s).sum()) == plan["n_segments"] for s in expected)
        and all(all(plan["folds"][s] == f for s in d["subject"].astype(str)) for f, d in enumerate(files))
        and all(d["names"].astype(str).tolist() == names for d in files)
        and all((dictionary_dir(root, n, f) / CHECKPOINT).is_file() for n in names for f in range(N_FOLDS))
    )
    every = [(n, f) for n in names for f in range(N_FOLDS)]
    moved = {f"{n}_f{f}": training_moved(_curve(dictionary_dir(root, n, f))) for n, f in every}
    violations = {f"{n}_f{f}": _violations(dictionary_dir(root, n, f)) for n, f in every}
    rows_gap = max(float(d["rows_gap"]) for d in files)
    repeat_gap = max(float(d["repeat_gap"]) for d in files)
    repeat_segments = min(int(d["repeat_segments"]) for d in files)
    failures = sum(int(d["load_failures"]) for d in files)
    passed = {
        "splits": all(all(c.values()) for c in split_checks),
        "rows_match_sweep": rows_gap <= ROW_TOLERANCE,
        "mask_holds": all(v == 0 for v in violations.values()),
        "training_moved": all(moved.values()),
        "repeat_run": repeat_gap <= REPEAT_TOLERANCE and repeat_segments >= N_REPEAT,
        "complete": complete and failures == 0,
    }
    record = {
        "status": "pass" if all(passed.values()) else "blocked",
        "gates": passed,
        "split_checks": split_checks,
        "not_moved": sorted(k for k, v in moved.items() if not v),
        "mask_broken": sorted(k for k, v in violations.items() if v != 0),
        "rows_gap": rows_gap,
        "repeat_gap": repeat_gap,
        "repeat_segments": repeat_segments,
        "load_failures": failures,
    }
    _save_json(root / "gates.json", record)
    logger.info(f"gates: {record['status']} {passed}")
    return record


# ---------------------------------------------------------------------- read


def _counts(root: Path) -> dict[str, Any]:
    """The held-out segments of all folds with the counts of every dictionary."""
    files = [_load_npz(count_file(root, f)) for f in range(N_FOLDS)]
    names = files[0]["names"].astype(str).tolist()
    stacked = {
        key: {n: np.concatenate([d[key][i] for d in files]) for i, n in enumerate(names)}
        for key in (*PER_SEGMENT, "atom_wins")
    }
    return stacked | {
        "names": names,
        "subject": np.concatenate([d["subject"].astype(str) for d in files]),
        "label": np.concatenate([d["label"] for d in files]),
        "fold": np.concatenate([np.full(len(d["subject"]), int(d["fold"])) for d in files]),
        "windows": np.concatenate([d["windows"] for d in files]),
        "signal_total": np.concatenate([d["signal_total"] for d in files]),
    }


def _arm(counts: dict[str, Any], shares: dict[str, np.ndarray], arm: str, seeds: tuple[int, ...], cohort: dict) -> Any:
    patients, values = patient_scores(seed_mean(shares, arm, seeds), counts["subject"])
    if patients.tolist() != cohort["subject"].tolist():
        raise ValueError("the scored patients differ from the plan's")
    return ArmScores(cohort["subject"], cohort["fold"], cohort["label"], values)


def _per_arm(counts: dict[str, Any], root: Path) -> dict[str, dict[str, float]]:
    """Section 8 per arm, as means over its dictionaries: the leak on the held-out non-epilepsy segments, the share
    of the epilepsy-only atoms that win no window on the held-out epilepsy segments, the residual share on the
    held-out segments, and the last validation residual of training."""
    negative, positive = counts["label"] == 0, counts["label"] == 1
    out: dict[str, dict[str, list[float]]] = {}
    for name in counts["names"]:
        arm = parse_name(name)[0]
        record = out.setdefault(arm, {"leak_rate": [], "unused_epilepsy_atoms": [], "residual_share": [], "val": []})
        record["leak_rate"].append(
            float(counts["leak_windows"][name][negative].sum() / counts["windows"][negative].sum())
        )
        for fold in range(N_FOLDS):
            where = positive & (counts["fold"] == fold)
            wins = counts["atom_wins"][name][where][:, N_SHARED:].sum(axis=0)
            record["unused_epilepsy_atoms"].append(float((wins == 0).mean()))
            curve = _curve(dictionary_dir(root, name, fold))
            if curve:
                record["val"].append(curve[-1])
        record["residual_share"].append(float(counts["residual_total"][name].sum() / counts["signal_total"].sum()))
    return {
        arm: {
            "leak_rate": float(np.mean(v["leak_rate"])),
            "unused_epilepsy_atoms": float(np.mean(v["unused_epilepsy_atoms"])),
            "residual_share": float(np.mean(v["residual_share"])),
            "last_val_residual_frac": float(np.mean(v["val"])) if v["val"] else float("nan"),
        }
        for arm, v in out.items()
    }


def read(
    root: Path, readout_root: Path, audit_root: Path, band_code_root: Path, n_boot: int = 10_000
) -> dict[str, Any]:
    """The rule of section 7 and the reads of section 8; writes ``masked_dictionary_decision.json`` and its table."""
    plan = _load_json(readout_root / "folds.json")
    gate_record = _load_json(root / "gates.json")
    cohort = load_cohort(readout_root, audit_root, plan)
    counts = _counts(root)
    names = counts["names"]
    wins = {n: share(counts["wins_epilepsy"][n], counts["wins_shared"][n]) for n in names}
    energy = {n: share(counts["energy_epilepsy"][n], counts["energy_shared"][n]) for n in names}
    arms = {label: _arm(counts, wins, arm, seeds, cohort) for label, (arm, seeds) in SCORES.items()}
    arms["Score-energy"] = _arm(counts, energy, *SCORES["Score"], cohort)
    roots = {"readout": readout_root, "band_code": band_code_root}
    saved = {name: _aligned(_scores(roots[where] / "scores" / f"{name}.csv"), cohort) for name, where in SAVED.items()}
    spectral = saved["Spectral"]
    separates = alone_interval(arms["Score"], n_boot)
    drug = drug_gain(arms["Score"].score, cohort["label"], cohort["fold"], cohort["status"], n_boot)
    gap = paired_interval(arms["Score"], spectral, n_boot=n_boot)
    met = {"S": separates[1] > 0.5, "G": drug[1] > 0, "C": gap[1] > -GAP_MARGIN}
    outcome = masked_outcome(gate_record["status"] == "pass", met["S"], met["G"], met["C"])
    fused = ArmScores(
        cohort["subject"],
        cohort["fold"],
        cohort["label"],
        rank_mean([arms["Score"].score, spectral.score], cohort["fold"]),
    )
    summary = [_arm_summary(n, a, cohort, n_boot) for n, a in {**arms, **saved, "Score+Spectral": fused}.items()]
    pl.DataFrame(summary).write_csv(root / "masked_dictionary_summary.csv")
    mask = pair_mask(cohort["label"], cohort["age"], cohort["sex"])
    matched = {}
    for name, arm in (("Score", arms["Score"]), ("Spectral", spectral)):
        m = matched_read(arm.score, cohort["label"], cohort["fold"], mask, n_boot=n_boot)
        matched[name] = {"value": m.value, "low": m.low, "high": m.high, "pairs": m.pairs}
    record = {
        "protocol": "documentation/masked_separable_sae_protocol.md, section 7",
        "outcome": outcome,
        "prediction": PREDICTION,
        "gates": gate_record,
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
            "per_arm": _per_arm(counts, root),
            "Score_minus_Score_perm": _interval(paired_interval(arms["Score"], arms["Score-perm"], n_boot=n_boot)),
            "Score_minus_Score_mask": _interval(paired_interval(arms["Score"], arms["Score-mask"], n_boot=n_boot)),
            "fusion_minus_Spectral": _interval(paired_interval(fused, spectral, n_boot=n_boot)),
            "seed_rank_correlation_within_fold": _within_fold_spearman(arms["Score-s1"], arms["Score-s2"]),
            "rank_correlation_within_fold": {
                name: _within_fold_spearman(arms["Score"], arm) for name, arm in saved.items()
            },
        },
    }
    _save_json(root / "masked_dictionary_decision.json", record)
    logger.info(f"masked dictionary read: {outcome} (conditions {met})")
    return record


# ---------------------------------------------------------------------- entry point


def main() -> None:
    """Command-line entry point for the subcommands of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("split", help="the patients and the permuted labels of every fold")
    for flag, text in (("--root", "the root"), ("--sweep-root", "the sweep root"), ("--readout-root", "pass 1 root")):
        command.add_argument(flag, type=Path, required=True, help=text)
    for name, description in (("fit", "one dictionary of one fold"), ("score", "the counts of one fold")):
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
            command.add_argument("--arm", choices=tuple(SEEDS), required=True)
            command.add_argument("--seed", type=int, choices=(1, 2), required=True)
    command = commands.add_parser("gates", help="gates 2 to 7")
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
