"""Post-hoc reads of the hinge-redesign pilot (``documentation/masked_hinge_redesign_protocol.md``, section 13).

The pilot rule counts an epilepsy-only atom as in use if it wins one window. These reads ask what the rule does not:
how often the epilepsy-only atoms win in group 1 against group 0, on the monitoring patients (new to the dictionary)
and in training. They were chosen after the pilot record was seen, so they describe and do not decide. CPU, saved
files only, seconds:

    python analysis/masked_hinge/pilot_posthoc.py --root <pilot root> --first-root <root of the first masked run> \
        --readout-root <pass 1 root>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

N_SHARED, WINDOWS, LAST = 64, 60, 26  # shared atoms, windows per segment, first of the last five epochs


def monitor_table(root: Path, readout_root: Path) -> pl.DataFrame:
    """Per dictionary and grouping of the 25 monitoring patients, coded with no mask: the atoms with a win in each
    group, the share of windows with a win of an epilepsy-only atom in each group, their placements, and the AUROC
    of the patient's win share for group 1 against group 0."""
    data = np.load(root / "pilot_counts.npz")
    subject = data["subject"].astype(str)
    plan = json.loads((readout_root / "folds.json").read_text())
    split = json.loads((root / "splits" / "fold0.json").read_text())
    groups = {
        "true": np.array([plan["labels"][s] for s in subject]),
        "permuted": np.array([split["permuted_labels"][s] for s in subject]),
    }
    patients = np.array(sorted(set(subject.tolist())))
    rows = []
    for i, name in enumerate(data["names"].astype(str).tolist()):
        wins, leak = data["atom_wins"][i], data["leak_windows"][i]
        only = wins[:, N_SHARED:]
        total = wins.sum(axis=1)
        share = np.divide(only.sum(axis=1), total, out=np.zeros(len(total)), where=total > 0)
        score = np.array([share[subject == p].mean() for p in patients])
        for kind, group in groups.items():
            one, zero = group == 1, group == 0
            label = np.array([group[subject == p][0] for p in patients])
            rows.append(
                {
                    "dictionary": name,
                    "groups": kind,
                    "atoms_with_win_1": int((only[one].sum(axis=0) > 0).sum()),
                    "atoms_with_win_0": int((only[zero].sum(axis=0) > 0).sum()),
                    "window_share_1": float(leak[one].sum() / data["windows"][one].sum()),
                    "window_share_0": float(leak[zero].sum() / data["windows"][zero].sum()),
                    "placements_1": int(only[one].sum()),
                    "placements_0": int(only[zero].sum()),
                    "patient_auroc": float(roc_auc_score(label, score)),
                    "patients_with_score_0": int((score == 0).sum()),
                }
            )
    table = pl.DataFrame(rows)
    return table.with_columns((pl.col("window_share_1") / pl.col("window_share_0")).alias("ratio"))


def training_table(folders: dict[str, Path]) -> pl.DataFrame:
    """Per dictionary, the mean of the last five training epochs: the placements of the epilepsy-only atoms per window
    of the group-1 segments, the leak in the group-0 segments, and the last validation residual."""
    rows = []
    for name, folder in folders.items():
        table = pl.read_csv(folder / "sae_training.csv")
        last = table.filter(pl.col("epoch") >= LAST)
        won = pl.read_csv(folder / "sae_won_windows.csv").filter(
            (pl.col("epoch") >= LAST) & (pl.col("atom") >= N_SHARED)
        )
        wins = won.group_by("epoch").agg(pl.col("won_windows").sum()).sort("epoch")["won_windows"].to_numpy()
        windows = (last["segments_seen"] - last["masked_negative_segments"]).to_numpy() * WINDOWS
        hit, leak = float((wins / windows).mean()), float(last["leak_rate"].mean())
        rows.append(
            {
                "dictionary": name,
                "placements_per_window_1": hit,
                "leak_0": leak,
                "ratio": hit / leak if leak > 0 else float("inf"),
                "val_residual_frac": float(table["val_residual_frac"][-1]),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Print the two tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, required=True, help="the pilot root")
    parser.add_argument("--first-root", type=Path, required=True, help="the root of the first masked run")
    parser.add_argument("--readout-root", type=Path, required=True, help="the root of frozen readout pass 1")
    args = parser.parse_args()
    folders = {path.name.removesuffix("_f0"): path for path in sorted((args.root / "dictionaries").iterdir())}
    folders["mask_s1 (first run)"] = args.first_root / "dictionaries" / "mask_s1_f0"
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(monitor_table(args.root, args.readout_root))
        print(training_table(folders))


if __name__ == "__main__":
    main()
