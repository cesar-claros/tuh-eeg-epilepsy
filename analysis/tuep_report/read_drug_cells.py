# ruff: noqa: E501
"""Post-hoc read of the saved scores by class and drug status. Reads result files only; fits nothing."""

import csv
import sys
from pathlib import Path

import numpy as np

AUDIT, READOUT = Path(sys.argv[1]), Path(sys.argv[2])
patients = {r["subject"]: r for r in csv.DictReader(open(READOUT / "patients.csv"))}
status = {r["subject"]: r["status"] for r in csv.DictReader(open(AUDIT / "medication.csv"))}
cov = {r["subject"]: r for r in csv.DictReader(open(AUDIT / "covariates.csv"))}
subjects = sorted(patients)
fold = np.array([int(patients[s]["fold"]) for s in subjects])
label = np.array([int(patients[s]["label"]) for s in subjects])
state = np.array([status[s] for s in subjects])
age = np.array([float(cov[s]["age"]) for s in subjects])
segments = np.array([float(patients[s]["segments"]) for s in subjects])


def standardized(arm):
    rows = {r["subject"]: float(r["score"]) for r in csv.DictReader(open(READOUT / "scores" / f"{arm}.csv"))}
    s = np.array([rows[x] for x in subjects])
    out = s.copy()
    for f in np.unique(fold):
        out[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    return out


def wins(a, b):
    return (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])


def compare(a, b, n_boot=10000, seed=0):
    """AUROC of a above b, with a 95% interval from a bootstrap of the patients of each group."""
    w = wins(a, b).astype(float)
    rng = np.random.default_rng(seed)
    pa = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=n_boot)
    pb = rng.multinomial(len(b), np.full(len(b), 1 / len(b)), size=n_boot)
    draws = ((pa @ w) * pb).sum(axis=1) / (pa.sum(axis=1) * pb.sum(axis=1))
    return w.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


cells = {
    "epilepsy on": (label == 1) & (state == "on"),
    "epilepsy off": (label == 1) & (state == "off"),
    "non-epilepsy on": (label == 0) & (state == "on"),
    "non-epilepsy off": (label == 0) & (state == "off"),
    "epilepsy unknown": (label == 1) & (state == "unknown"),
    "non-epilepsy unknown": (label == 0) & (state == "unknown"),
}
print("cells (deciding list): n, median age, median segments")
for name, where in cells.items():
    print(
        f"  {name:22s} {int(where.sum()):3d}  age {np.median(age[where]):5.1f}  segments {np.median(segments[where]):6.1f}"
    )

for arm in ("Spectral", "HYDRA", "S4-R1", "S4-proj"):
    z = standardized(arm)
    print(f"\n== {arm}: mean standardized score per cell")
    for name, where in cells.items():
        print(f"  {name:22s} {z[where].mean():+.2f}")
    print("  comparisons, AUROC of the first group above the second:")
    for first, second in (
        ("non-epilepsy on", "non-epilepsy off"),
        ("epilepsy on", "epilepsy off"),
        ("epilepsy off", "non-epilepsy off"),
        ("epilepsy on", "non-epilepsy on"),
        ("epilepsy on", "non-epilepsy off"),
    ):
        v, lo, hi = compare(z[cells[first]], z[cells[second]])
        print(f"    {first:18s} vs {second:18s} {v:.3f} [{lo:.3f}, {hi:.3f}]")

print("\n== by hand: status alone from the tables of audit_decision.json, among known status")
for name, tp, fn, fp, tn in (("deciding list", 70, 7, 25, 40), ("narrow list", 66, 10, 4, 58)):
    tpr, fpr = tp / (tp + fn), fp / (fp + tn)
    print(f"  {name}: on in epilepsy {tpr:.3f}, on in non-epilepsy {fpr:.3f}, AUROC {0.5 + 0.5 * (tpr - fpr):.3f}")
print("\n== bounds on the on group of the rule: 65 epilepsy and 23 non-epilepsy in matched pairs")
print("  narrow is a subset of the deciding list. Deciding on: 70 and 25. Narrow on: 66 and 4.")
print("  epilepsy in the group on a maintenance drug: at least", 65 - (70 - 66), "of 65")
print("  non-epilepsy in the group on a maintenance drug: at most", min(4, 23), "of 23")
