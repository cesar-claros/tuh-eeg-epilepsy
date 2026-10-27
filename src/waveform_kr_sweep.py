"""Reconstruction-only sweep of the atom count K, the atoms per window R and the initialization.

Protocol: ``documentation/waveform_kr_init_sweep_protocol.md``. Two subcommands:

- ``run`` fits one arm with ``src/waveform_probe.py``'s ``run_arm``: the campaign configuration of
  ``make_config`` on an audit that ``check_audit`` accepts (the 40 s offset audit), the campaign
  signal cache, and the K, R, initialization and seed encoded in the arm directory name
  ``K{K}_R{R}_{patches|random}_s{seed}[_frozen]``. A frozen arm runs one epoch with ``lr=0``: the
  validation statistics of its initial dictionary.
- ``summarize`` reads the arms of a sweep root, applies the decision rules fixed before any result
  (constants below), and writes ``kr_sweep_summary.csv``, ``kr_sweep_decision.json`` and
  ``kr_sweep.png``.

At P=512 a segment holds at most ``60 R`` coefficients for every K: R sets the coefficient budget,
K the choice of waveforms per window. Run on the HPC inside the container.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import rootutils
from loguru import logger

ROOT = rootutils.setup_root(__file__, indicator=[".git", "pyproject.toml"], pythonpath=True)

from src.waveform_probe import run_arm, selected_readout  # noqa: E402

# Decision rules, fixed before the first result (protocol section 4).
EV_TOLERANCE = 0.01  # minimum explained-energy tolerance for the knee and the initialization verdict
SEED_FACTOR = 3.0  # the tolerance grows to this multiple of the measured seed spread when larger
HEALTHY_NEVER_WON = 1 / 8  # healthy: at most this fraction of the K atoms without a won training window
STARVING_NEVER_WON = 1 / 4  # starving: at least this fraction
HEALTHY_SHARE = 6.4  # healthy: busiest atom's won-window share at most HEALTHY_SHARE / K (0.10 at K=64)
STARVING_SHARE = 16.0  # starving: at least STARVING_SHARE / K (0.25 at K=64)
GAIN_FLOOR = 0.9  # val_full_gain below this flags overshoot from stacked, correlated winners
REFERENCE = (64, 1)  # the campaign's K and R
ARM_PATTERN = re.compile(r"^K(?P<k>\d+)_R(?P<r>\d+)_(?P<init>patches|random)_s(?P<seed>\d+)(?P<frozen>_frozen)?$")
HEALTH_ORDER = {"healthy": 0, "watch": 1, "starving": 2}


def parse_arm(name: str) -> dict[str, Any]:
    """K, R, initialization, seed and frozen flag of an arm directory name.

    Raises
    ------
    ValueError
        If the name does not follow ``K{K}_R{R}_{patches|random}_s{seed}[_frozen]`` or ``R > K``.
    """
    match = ARM_PATTERN.match(name)
    if match is None:
        raise ValueError(f"name the arm K{{K}}_R{{R}}_{{patches|random}}_s{{seed}}[_frozen], got {name}")
    arm = {
        "arm": name,
        "k": int(match["k"]),
        "r": int(match["r"]),
        "init": match["init"],
        "seed": int(match["seed"]),
        "frozen": match["frozen"] is not None,
    }
    if not 1 <= arm["r"] <= arm["k"]:
        raise ValueError(f"{name}: need 1 <= R <= K")
    return arm


def health(k: int, never_won: float, max_share: float, rising: bool) -> str:
    """``healthy``, ``watch`` or ``starving`` from the last training epoch, with thresholds scaled by K."""
    if never_won >= STARVING_NEVER_WON * k or max_share >= STARVING_SHARE / k or rising:
        return "starving"
    if never_won <= HEALTHY_NEVER_WON * k and max_share <= HEALTHY_SHARE / k:
        return "healthy"
    return "watch"


def tolerance(arms: list[dict[str, Any]]) -> tuple[float, float | None]:
    """The explained-energy tolerance and the largest seed spread among trained arms run with several seeds."""
    groups: dict[tuple[int, int, str], list[float]] = {}
    for arm in arms:
        if not arm["frozen"]:
            groups.setdefault((arm["k"], arm["r"], arm["init"]), []).append(arm["val_explained"])
    spreads = [max(values) - min(values) for values in groups.values() if len(values) > 1]
    spread = max(spreads) if spreads else None
    return max(EV_TOLERANCE, SEED_FACTOR * spread) if spread is not None else EV_TOLERANCE, spread


def knee(arms: list[dict[str, Any]], tol: float) -> dict[int, dict[str, Any]]:
    """Per R: the smallest non-starving K whose explained energy is within ``tol`` of the best non-starving K.

    Uses trained, seed-0, patch-initialized arms. ``saturated`` is false when the largest K tested is
    the best and leads the next smaller K by more than ``tol``: the knee may lie above the grid.
    """
    out: dict[int, dict[str, Any]] = {}
    grid = [a for a in arms if not a["frozen"] and a["seed"] == 0 and a["init"] == "patches"]
    for r in sorted({a["r"] for a in grid}):
        row = sorted((a for a in grid if a["r"] == r), key=lambda a: a["k"])
        usable = [a for a in row if a["health"] != "starving"]
        explained = {a["k"]: a["val_explained"] for a in row}
        if not usable:
            out[r] = {"knee_k": None, "best_k": None, "explained_by_k": explained, "saturated": None}
            continue
        best = max(usable, key=lambda a: a["val_explained"])
        chosen = min((a for a in usable if a["val_explained"] >= best["val_explained"] - tol), key=lambda a: a["k"])
        ks = [a["k"] for a in usable]
        saturated = True
        if best["k"] == max(ks) and len(ks) > 1:
            below = max(a["val_explained"] for a in usable if a["k"] < best["k"])
            saturated = best["val_explained"] - below <= tol
        out[r] = {
            "knee_k": chosen["k"],
            "knee_explained": chosen["val_explained"],
            "best_k": best["k"],
            "best_explained": best["val_explained"],
            "explained_by_k": explained,
            "starving_k": [a["k"] for a in row if a["health"] == "starving"],
            "saturated": saturated,
        }
    return out


def init_verdict(arms: list[dict[str, Any]], tol: float) -> dict[str, Any]:
    """``random_better``, ``patches_better`` or ``no_difference`` over matched trained seed-0 (K, R) pairs.

    Random initialization is better only if it explains at least ``tol`` more in every pair and is never
    less healthy; patches are better if they explain at least ``tol`` more in every pair.
    """
    cells = {(a["k"], a["r"], a["init"]): a for a in arms if not a["frozen"] and a["seed"] == 0}
    pairs = [
        (cells[(k, r, "random")], cells[(k, r, "patches")])
        for (k, r, init) in cells
        if init == "random" and (k, r, "patches") in cells
    ]
    if not pairs:
        return {"verdict": None, "pairs": {}}
    gains = {f"K{rnd['k']}_R{rnd['r']}": rnd["val_explained"] - pat["val_explained"] for rnd, pat in pairs}
    no_less_healthy = all(HEALTH_ORDER[rnd["health"]] <= HEALTH_ORDER[pat["health"]] for rnd, pat in pairs)
    if all(g >= tol for g in gains.values()) and no_less_healthy:
        verdict = "random_better"
    elif all(g <= -tol for g in gains.values()):
        verdict = "patches_better"
    else:
        verdict = "no_difference"
    return {"verdict": verdict, "pairs": gains}


def campaign_note(knees: dict[int, dict[str, Any]], init: dict[str, Any], reference: dict[str, Any] | None) -> str:
    """The protocol's reading of the sweep for the campaign (K and initialization at R=1)."""
    notes = []
    row = knees.get(REFERENCE[1])
    if reference is None or row is None or row["knee_k"] is None:
        notes.append("The R=1 row or the K=64, R=1 reference arm is missing or starving everywhere; rerun it.")
    else:
        if reference["health"] == "starving":
            notes.append("The campaign setting K=64, R=1 starves atoms: add a dead-atom remedy before the campaign.")
        k = row["knee_k"]
        if k < REFERENCE[0]:
            notes.append(f"K={k} reconstructs within the tolerance of the best at R=1: a smaller dictionary is enough.")
        elif k > REFERENCE[0]:
            notes.append(f"K=64 is more than the tolerance below the best at R=1: consider K={k} (more compute).")
        else:
            notes.append("K=64 is the knee at R=1: keep it.")
        if not row["saturated"]:
            notes.append("The largest K is still the best by more than the tolerance: the knee may lie above K=128.")
    if init["verdict"] == "random_better":
        notes.append("Random initialization reconstructs better in every matched pair: consider it.")
    elif init["verdict"] is not None:
        notes.append("Keep patch initialization.")
    return " ".join(notes)


def summarize(root: Path) -> dict[str, Any]:
    """Read every finished arm under ``root``, write the summary, the decision and the figure; return the decision."""
    arms = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and ARM_PATTERN.match(p.name)):
        if not ((folder / "sae_training.csv").exists() and (folder / "sae_atom_stats.csv").exists()):
            logger.warning(f"{folder.name}: no finished training files; treated as missing")
            continue
        arm = {**parse_arm(folder.name), **selected_readout(folder)}
        arm["health"] = (
            "frozen"
            if arm["frozen"]
            else health(arm["k"], arm["never_won_last"], arm["max_won_share_last"], arm["never_won_rising"])
        )
        arm["overshoot"] = arm["val_gain"] < GAIN_FLOOR
        arm["converged"] = arm["frozen"] or arm["selected_epoch"] < arm["epochs"]
        arms.append(arm)
    if not arms:
        raise ValueError(f"no finished arms under {root}")
    pl.DataFrame(arms).write_csv(root / "kr_sweep_summary.csv")
    tol, spread = tolerance(arms)
    knees = knee(arms, tol)
    init = init_verdict(arms, tol)
    reference = next(
        (
            a
            for a in arms
            if (a["k"], a["r"]) == REFERENCE and a["init"] == "patches" and a["seed"] == 0 and not a["frozen"]
        ),
        None,
    )
    decision = {
        "arms": [a["arm"] for a in arms],
        "tolerance": tol,
        "seed_spread": spread,
        "knee_by_r": {str(r): v for r, v in knees.items()},
        "init": init,
        "frozen_explained": {a["arm"]: a["val_explained"] for a in arms if a["frozen"]},
        "starving": [a["arm"] for a in arms if a["health"] == "starving"],
        "overshoot": [a["arm"] for a in arms if a["overshoot"]],
        "not_converged": [a["arm"] for a in arms if not a["converged"]],
        "campaign": campaign_note(knees, init, reference),
    }
    (root / "kr_sweep_decision.json").write_text(json.dumps(decision, indent=2, allow_nan=True) + "\n")
    plot(arms, decision, root / "kr_sweep.png")
    for arm in arms:
        logger.info(
            f"{arm['arm']}: explained {arm['val_explained']:.4f} (epoch {arm['selected_epoch']}/{arm['epochs']}), "
            f"gain {arm['val_gain']:.3f}, never won {arm['never_won_last']}, max share {arm['max_won_share_last']}, "
            f"{arm['health']}"
        )
    logger.info(f"Decision: {json.dumps(decision, allow_nan=True)}")
    return decision


def plot(arms: list[dict[str, Any]], decision: dict[str, Any], path: Path) -> None:
    """Explained energy and the fraction of atoms that never won, against K, per R and initialization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: "#0072B2", 2: "#D55E00", 4: "#009E73", 8: "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    trained = [a for a in arms if not a["frozen"] and a["seed"] == 0]
    for r in sorted({a["r"] for a in trained}):
        for init, style in (("patches", "-"), ("random", ":")):
            row = sorted((a for a in trained if a["r"] == r and a["init"] == init), key=lambda a: a["k"])
            if not row:
                continue
            ks = [a["k"] for a in row]
            color = colors.get(r, "#555555")
            label = f"R={r}, {init}"
            axes[0].plot(ks, [a["val_explained"] for a in row], style, marker="o", color=color, label=label)
            axes[1].plot(ks, [a["never_won_last"] / a["k"] for a in row], style, marker="o", color=color, label=label)
            for a in row:
                if a["health"] == "starving":
                    axes[0].plot(a["k"], a["val_explained"], "x", color="#000000", ms=9)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Atoms K")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Validation explained energy (1 - residual fraction)")
    axes[0].set_title(f"Reconstruction (x = starving); tolerance {decision['tolerance']:.3f}", fontsize=9)
    axes[1].axhline(HEALTHY_NEVER_WON, color="#999999", ls="--", lw=0.8)
    axes[1].axhline(STARVING_NEVER_WON, color="#999999", ls=":", lw=0.8)
    axes[1].set_ylabel("Atoms without a won training window / K")
    axes[1].set_title("Atom starvation (dashed: healthy limit, dotted: starving limit)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fit one arm; K, R, init, seed and frozen come from the --output name")
    run.add_argument("--manifests", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--signal-cache-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--epochs", type=int, required=True, help="trained arms; frozen arms always run one epoch")
    run.add_argument("--num-workers", type=int, default=4)
    report = commands.add_parser("summarize", help="read the arms and apply the decision rules")
    report.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.root.resolve())
        return
    arm = parse_arm(args.output.name)
    run_arm(
        argparse.Namespace(
            manifests=args.manifests,
            data_dir=args.data_dir,
            signal_cache_dir=args.signal_cache_dir,
            output=args.output,
            seed=arm["seed"],
            top_r=arm["r"],
            n_atoms=arm["k"],
            init=arm["init"],
            epochs=1 if arm["frozen"] else args.epochs,
            lr=0.0 if arm["frozen"] else None,
            num_workers=args.num_workers,
            allow_cold_cache=False,
        )
    )


if __name__ == "__main__":
    main()
