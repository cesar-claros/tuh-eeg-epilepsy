"""Post-hoc reads of stage 2 of the hinge redesign (``documentation/masked_hinge_redesign_protocol.md``, section 14).

Stage 2 gave ``collapsed``, so no score is read: this script does not open the count files of the held-out patients.
It reads the monitoring counts (the data of gate 8) and the training tables, and it asks how often the epilepsy-only
atoms win in group 1 against group 0, on the monitoring patients (new to the dictionary) and in training. Each arm
is read with its own groups: the true labels for ``sup2``, the permuted labels for ``perm2``. The reads were chosen
after the pilot, so they describe and do not decide. CPU, saved files only, seconds:

    python analysis/masked_hinge/stage2_posthoc.py --root <the root> --readout-root <pass 1 root>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

N_SHARED, WINDOWS, LAST, N_FOLDS = 64, 60, 26, 5  # shared atoms, windows per segment, first of the last five epochs


def dictionary_table(root: Path, readout_root: Path) -> pl.DataFrame:
    """One row per stage-2 dictionary: the monitoring reads by group and the training rates of the last epochs."""
    plan = json.loads((readout_root / "folds.json").read_text())
    rows = []
    for fold in range(N_FOLDS):
        data = np.load(root / "monitor" / f"fold{fold}.npz")
        split = json.loads((root / "splits" / f"fold{fold}.json").read_text())
        subject = data["subject"].astype(str)
        groups = {
            "sup2": np.array([plan["labels"][s] for s in subject]),
            "perm2": np.array([split["permuted_labels"][s] for s in subject]),
        }
        for i, name in enumerate(data["names"].astype(str).tolist()):
            arm = name.split("_")[0]
            one, zero = groups[arm] == 1, groups[arm] == 0
            only, leak, windows = data["atom_wins"][i][:, N_SHARED:], data["leak_windows"][i], data["windows"]
            folder = root / "dictionaries" / f"{name}_f{fold}"
            table = pl.read_csv(folder / "sae_training.csv")
            last = table.filter(pl.col("epoch") >= LAST)
            won = pl.read_csv(folder / "sae_won_windows.csv").filter(pl.col("atom") >= N_SHARED)
            late = won.filter(pl.col("epoch") >= LAST)
            wins = late.group_by("epoch").agg(pl.col("won_windows").sum()).sort("epoch")["won_windows"].to_numpy()
            positive = (last["segments_seen"] - last["masked_negative_segments"]).to_numpy() * WINDOWS
            alive = won.filter(pl.col("epoch") == won["epoch"].max()).filter(pl.col("won_windows") > 0).height
            rows.append(
                {
                    "dictionary": f"{name}_f{fold}",
                    "arm": arm,
                    "atoms_1": int((only[one].sum(axis=0) > 0).sum()),
                    "atoms_0": int((only[zero].sum(axis=0) > 0).sum()),
                    "windows_won_1": int(leak[one].sum()),
                    "windows_1": int(windows[one].sum()),
                    "windows_won_0": int(leak[zero].sum()),
                    "windows_0": int(windows[zero].sum()),
                    "train_rate_1": float((wins / positive).mean()),
                    "train_leak_0": float(last["leak_rate"].mean()),
                    "train_atoms_last_epoch": alive,
                    "val_residual_frac": float(table["val_residual_frac"][-1]),
                }
            )
    return pl.DataFrame(rows)


def arm_table(table: pl.DataFrame) -> pl.DataFrame:
    """Per arm, over its 10 dictionaries: the mean atoms with a win, the pooled window shares on the monitoring
    patients, and the mean training rates."""
    out = table.group_by("arm").agg(
        pl.col("atoms_1").mean(),
        pl.col("atoms_0").mean(),
        (pl.col("windows_won_1").sum() / pl.col("windows_1").sum()).alias("monitor_share_1"),
        (pl.col("windows_won_0").sum() / pl.col("windows_0").sum()).alias("monitor_share_0"),
        pl.col("windows_won_1").sum().alias("monitor_wins_1"),
        pl.col("windows_won_0").sum().alias("monitor_wins_0"),
        pl.col("train_rate_1").mean(),
        pl.col("train_leak_0").mean(),
        pl.col("train_atoms_last_epoch").mean(),
        pl.col("val_residual_frac").mean(),
    )
    return out.with_columns(
        (pl.col("monitor_share_1") / pl.col("monitor_share_0")).alias("monitor_ratio"),
        (pl.col("train_rate_1") / pl.col("train_leak_0")).alias("train_ratio"),
    ).sort("arm", descending=True)


def main() -> None:
    """Print the table of the dictionaries and the table of the arms."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="the root of the pilot and of stage 2")
    parser.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
    args = parser.parse_args()
    table = dictionary_table(args.root, args.readout_root)
    ratios = table.with_columns(
        (pl.col("windows_won_1") / pl.col("windows_1") / (pl.col("windows_won_0") / pl.col("windows_0"))).alias(
            "monitor_ratio"
        ),
        (pl.col("train_rate_1") / pl.col("train_leak_0")).alias("train_ratio"),
    )
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=260, float_precision=4):
        print(ratios.drop("windows_1", "windows_0"))
        print(arm_table(table))
        print(
            ratios.group_by("arm").agg(
                pl.col("monitor_ratio").median(), (pl.col("monitor_ratio") > 1).sum().alias("n_above_1")
            )
        )


if __name__ == "__main__":
    main()
