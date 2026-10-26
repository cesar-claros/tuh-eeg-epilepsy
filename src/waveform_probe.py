"""First format-5 probe: atom competition per window against the format-4 rule at a matched budget.

Protocol: ``documentation/waveform_format5_probe_protocol.md``. Two subcommands:

- ``run`` fits one reconstruction arm with the campaign configuration of
  ``src/run_patient_waveform_sweep.py`` (L=256, P=512, K=64, patient sampling, seed-42
  split), changing only the seed, ``top_r``, the epochs and the learning rate. It reuses
  an existing signal cache and refuses to build a new one unless asked. The same file
  runs in a worktree of the format-4 commit, so both rules share the code path.
- ``summarize`` reads the arms of a probe root, applies the decision rules fixed in the
  protocol before any result, and writes ``probe_summary.csv`` and ``probe_decision.json``.

Arm directories are named ``{f4|f5}_r{R}[_frozen]_s{seed}``. A frozen arm uses ``lr=0``
for one epoch: the validation statistics of the initial dictionary. Run on the HPC
inside the container; ``summarize`` needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

# Decision thresholds, fixed before the first probe result (protocol section 4).
RELATIVE_EV_GAIN = 0.10  # paired explained-energy gain needed to call one rule better
MIN_EV_SCALE = 0.01  # floor of the reference explained energy in the relative test
HEALTHY_NEVER_WON = 8  # at most this many atoms without a won training window: healthy
STARVING_NEVER_WON = 16  # at least this many: starving
HEALTHY_MAX_SHARE = 0.10  # busiest atom's share of won windows: healthy at or below
STARVING_MAX_SHARE = 0.25  # starving at or above
RISING_EPOCHS = 3  # never-won count strictly rising over this many final epochs: starving
ARM_PATTERN = re.compile(r"^(?P<rule>f4|f5)_r(?P<r>\d+)(?P<frozen>_frozen)?_s(?P<seed>\d+)$")

NEXT_STEPS: dict[str | tuple[str, str], str] = {
    "incomplete": "Some arms are missing or failed; rerun them before reading the probe.",
    "undertrained": (
        "Training did not improve on the frozen dictionary for at least one rule; repeat the trained arms "
        "with more epochs before comparing the rules."
    ),
    ("f5_better", "healthy"): "Run the 30-epoch campaign on format 5.",
    ("f5_better", "watch"): "Fit format 5 for 10 epochs and read n_atoms_never_won again before the campaign.",
    ("f5_better", "starving"): (
        "Add a dead-atom remedy (re-seeding first), repeat the format-5 arms, then run the campaign."
    ),
    ("no_difference", "healthy"): (
        "Reconstruction does not separate the rules at a matched budget. Keep format 5 for its local "
        "selection and full window coverage, and run block B (R=2, 4) to test a larger budget."
    ),
    ("no_difference", "watch"): (
        "Format 5 shows no reconstruction gain and uneven atom use: add a dead-atom remedy, or run the "
        "campaign on format 4 (revert 8ae22c3)."
    ),
    ("no_difference", "starving"): (
        "Format 5 adds atom starvation without a reconstruction gain: run the campaign on format 4 "
        "(revert 8ae22c3), or add a dead-atom remedy and repeat the probe."
    ),
    "f4_better": (
        "The matched-budget comparison favours format 4, so the motivation for format 5 fails at R=1. "
        "Run the campaign on format 4 (revert 8ae22c3) and record the negative result."
    ),
}


@dataclass(frozen=True)
class ArmResult:
    """Readout of one arm at its selected epoch (minimum ``val_objective``)."""

    arm: str
    rule: str
    top_r: int
    frozen: bool
    seed: int
    epochs: int
    selected_epoch: int
    val_objective: float
    val_explained: float  # 1 - val_full_residual_frac
    val_gain: float
    n_effective_atoms: float  # from the validation energy shares of the selected dictionary
    val_atoms_never_won: float  # format 5 only, NaN otherwise
    never_won_last: float  # format 5 training, last epoch; NaN otherwise
    max_won_share_last: float
    never_won_rising: bool


def read_arm(folder: Path) -> ArmResult:
    """Read ``sae_training.csv`` and ``sae_atom_stats.csv`` of one finished arm.

    Raises
    ------
    ValueError
        If the folder name does not follow ``{f4|f5}_r{R}[_frozen]_s{seed}`` or a file is missing.
    """
    match = ARM_PATTERN.match(folder.name)
    if match is None:
        raise ValueError(f"not a probe arm directory: {folder}")
    history = pl.read_csv(folder / "sae_training.csv")
    stats = pl.read_csv(folder / "sae_atom_stats.csv")
    selected = history.row(int(history["val_objective"].arg_min()), named=True)
    share = stats["energy_share"].to_numpy()
    total = float((share**2).sum())
    never_won = history["n_atoms_never_won"].to_list() if "n_atoms_never_won" in history.columns else []
    shares = history["max_won_share"].to_list() if "max_won_share" in history.columns else []
    tail = never_won[-RISING_EPOCHS:]
    return ArmResult(
        arm=folder.name,
        rule=match["rule"],
        top_r=int(match["r"]),
        frozen=match["frozen"] is not None,
        seed=int(match["seed"]),
        epochs=history.height,
        selected_epoch=int(selected["epoch"]),
        val_objective=float(selected["val_objective"]),
        val_explained=1.0 - float(selected["val_full_residual_frac"]),
        val_gain=float(selected["val_full_gain"]),
        n_effective_atoms=1.0 / total if total > 0 else 0.0,
        val_atoms_never_won=float((stats["won_windows"] == 0).sum()) if "won_windows" in stats.columns else math.nan,
        never_won_last=float(never_won[-1]) if never_won else math.nan,
        max_won_share_last=float(shares[-1]) if shares else math.nan,
        never_won_rising=len(tail) == RISING_EPOCHS and all(a < b for a, b in zip(tail, tail[1:])),
    )


def reconstruction_verdict(pairs: list[tuple[ArmResult, ArmResult]]) -> str:
    """``f5_better``, ``f4_better`` or ``no_difference`` from seed-paired ``(format 5, format 4)`` arms.

    One rule is better only when it explains more validation energy and has the lower
    patient objective in every seed, and its mean paired gain reaches ``RELATIVE_EV_GAIN``
    of the other rule's mean explained energy.
    """
    if not pairs:
        raise ValueError("need at least one seed with both rules")
    gains = [f5.val_explained - f4.val_explained for f5, f4 in pairs]
    mean_gain = sum(gains) / len(gains)
    f4_scale = max(sum(f4.val_explained for _, f4 in pairs) / len(pairs), MIN_EV_SCALE)
    f5_scale = max(sum(f5.val_explained for f5, _ in pairs) / len(pairs), MIN_EV_SCALE)
    if (
        all(g > 0 for g in gains)
        and all(f5.val_objective < f4.val_objective for f5, f4 in pairs)
        and mean_gain >= RELATIVE_EV_GAIN * f4_scale
    ):
        return "f5_better"
    if (
        all(g < 0 for g in gains)
        and all(f4.val_objective < f5.val_objective for f5, f4 in pairs)
        and -mean_gain >= RELATIVE_EV_GAIN * f5_scale
    ):
        return "f4_better"
    return "no_difference"


def usage_verdict(arms: list[ArmResult]) -> str:
    """``healthy``, ``watch`` or ``starving`` from the training won-window record of trained format-5 arms."""
    if not arms:
        raise ValueError("need at least one trained format-5 arm")
    if any(
        a.never_won_last >= STARVING_NEVER_WON or a.max_won_share_last >= STARVING_MAX_SHARE or a.never_won_rising
        for a in arms
    ):
        return "starving"
    if all(a.never_won_last <= HEALTHY_NEVER_WON and a.max_won_share_last <= HEALTHY_MAX_SHARE for a in arms):
        return "healthy"
    return "watch"


def next_step(reconstruction: str, usage: str, trained: bool) -> str:
    """The action that the protocol assigns to one combination of verdicts."""
    if not trained:
        return NEXT_STEPS["undertrained"]
    if reconstruction == "f4_better":
        return NEXT_STEPS["f4_better"]
    return NEXT_STEPS[(reconstruction, usage)]


def summarize(root: Path) -> dict[str, Any]:
    """Read every arm under ``root``, write the summary table and the decision, and return the decision."""
    arms = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and ARM_PATTERN.match(p.name)):
        if (folder / "sae_training.csv").exists() and (folder / "sae_atom_stats.csv").exists():
            arms[folder.name] = read_arm(folder)
        else:
            logger.warning(f"{folder.name}: no finished training files; treated as missing")
    if arms:
        pl.DataFrame([asdict(a) for a in arms.values()]).write_csv(root / "probe_summary.csv")
    trained = {(a.rule, a.seed): a for a in arms.values() if a.top_r == 1 and not a.frozen}
    frozen = {(a.rule, a.seed): a for a in arms.values() if a.top_r == 1 and a.frozen}
    seeds = sorted({seed for rule, seed in trained if rule == "f5"} & {seed for rule, seed in trained if rule == "f4"})
    decision: dict[str, Any] = {"arms": sorted(arms), "seeds": seeds}
    complete = bool(seeds) and ("f5", 0) in frozen and ("f4", 0) in frozen and 0 in seeds
    if not complete:
        decision.update(reconstruction=None, usage=None, trained=None, next_step=NEXT_STEPS["incomplete"])
    else:
        pairs = [(trained[("f5", s)], trained[("f4", s)]) for s in seeds]
        training_gain = {
            rule: trained[(rule, 0)].val_explained - frozen[(rule, 0)].val_explained for rule in ("f5", "f4")
        }
        is_trained = all(gain > 0 for gain in training_gain.values())
        reconstruction = reconstruction_verdict(pairs)
        usage = usage_verdict([f5 for f5, _ in pairs])
        decision.update(
            paired_explained_gain={str(f5.seed): f5.val_explained - f4.val_explained for f5, f4 in pairs},
            training_gain_seed0=training_gain,
            reconstruction=reconstruction,
            usage=usage,
            trained=is_trained,
            next_step=next_step(reconstruction, usage, is_trained),
        )
    (root / "probe_decision.json").write_text(json.dumps(decision, indent=2, allow_nan=True) + "\n")
    for arm in arms.values():
        logger.info(
            f"{arm.arm}: epoch {arm.selected_epoch}/{arm.epochs}, explained {arm.val_explained:.4f}, "
            f"objective {arm.val_objective:.6g}, gain {arm.val_gain:.3f}, effective atoms "
            f"{arm.n_effective_atoms:.1f}, never won (train last / val) {arm.never_won_last} / "
            f"{arm.val_atoms_never_won}, max won share {arm.max_won_share_last}"
        )
    logger.info(f"Decision: {json.dumps(decision, allow_nan=True)}")
    return decision


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False).stdout


def run_arm(args: argparse.Namespace) -> None:
    """Fit one reconstruction arm with the campaign configuration and save its artifacts to ``args.output``."""
    import hydra
    import lightning
    import torch
    from omegaconf import OmegaConf

    from src.models.components import waveform_sae
    from src.run_patient_waveform_sweep import check_audit, digest, make_config, save_json
    from src.utils import check_split_disjoint, dump_window_metadata, split_provenance

    manifests, output = args.manifests.resolve(), args.output.resolve()
    audit = check_audit(manifests)
    if not torch.cuda.is_available():
        raise RuntimeError("run inside the HPC container on an allocated CUDA GPU")
    cache_root = args.signal_cache_dir.expanduser().resolve()
    if not cache_root.is_dir():
        raise ValueError(f"signal cache root not found: {cache_root}")
    existing = {p.name for p in cache_root.iterdir() if p.is_dir()}
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    cfg = make_config(manifests, args.data_dir.resolve(), args.num_workers)
    cfg.seed, cfg.feature.random_state = args.seed, args.seed
    cfg.feature.spec.top_r = args.top_r
    cfg.feature.train_spec.epochs = args.epochs
    if args.lr is not None:
        cfg.feature.train_spec.lr = args.lr
    cfg.supervised = None
    cfg.signal_cache_dir = str(cache_root)
    OmegaConf.save(cfg, output / "config.yaml", resolve=False)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    check_split_disjoint(datamodule)
    for split, patients in (("train", 117), ("val", 39), ("test", 39)):
        if getattr(datamodule, f"{split}_df").subject.nunique() != patients:
            raise ValueError(f"{split} patient count changed during loading")
    dump_window_metadata(output, datamodule)
    identities = {}
    for split in ("train", "val"):
        dataset = getattr(datamodule, f"data_{split}")
        dataset.enable_signal_cache(cache_root)
        identity = dataset.signal_cache.identity
        if identity not in existing and not args.allow_cold_cache:
            folder = cache_root / identity
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
            raise ValueError(
                f"{split} signals have no entries under {cache_root} (identity {identity}); the probe would "
                "rebuild the cache. Pass the campaign cache, or --allow-cold-cache to build one."
            )
        identities[split] = identity
    save_json(
        output / "probe_provenance.json",
        {
            "arm": output.name,
            "code_root": str(ROOT),
            "commit": _git("rev-parse", "HEAD").strip(),
            "uncommitted": _git("status", "--porcelain").splitlines(),
            "checkpoint_format": waveform_sae.CHECKPOINT_FORMAT,
            "seed": args.seed,
            "top_r": args.top_r,
            "epochs": args.epochs,
            "lr": float(cfg.feature.train_spec.lr),
            "audit": audit,
            "manifests": {path.name: digest(path) for path in manifests.glob("windows_*.csv")},
            "signal_cache": {"root": str(cache_root), **identities},
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
    )
    lightning.seed_everything(args.seed, workers=True)
    model = hydra.utils.instantiate(cfg.feature)
    logger.info(
        f"Probe arm {output.name}: format {waveform_sae.CHECKPOINT_FORMAT}, R={args.top_r}, seed {args.seed}, "
        f"{args.epochs} epoch(s), lr {cfg.feature.train_spec.lr}. Test signals are not evaluated."
    )
    model.fit_unsupervised(
        datamodule.train_dataloader(), datamodule.val_dataloader(), provenance=split_provenance(datamodule, cfg.data)
    )
    model.save_artifacts(output)


def main() -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fit one arm")
    run.add_argument("--manifests", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--signal-cache-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="new arm directory, named as in the protocol")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--top-r", type=int, default=1)
    run.add_argument("--epochs", type=int, required=True)
    run.add_argument("--lr", type=float, default=None, help="default: the campaign rate; 0 for a frozen arm")
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--allow-cold-cache", action="store_true")
    report = commands.add_parser("summarize", help="read the arms and apply the decision rules")
    report.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if not ARM_PATTERN.match(args.output.name):
            raise ValueError(f"name the arm directory as {{f4|f5}}_r{{R}}[_frozen]_s{{seed}}, got {args.output.name}")
        run_arm(args)
    else:
        summarize(args.root.resolve())


if __name__ == "__main__":
    main()
