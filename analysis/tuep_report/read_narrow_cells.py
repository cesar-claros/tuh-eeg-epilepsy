# ruff: noqa: E501
"""Post-hoc read of the saved scores by class and maintenance-drug status. Reads result files only; fits nothing."""

import csv
import sys
from pathlib import Path

import numpy as np

AUDIT, READOUT = Path(sys.argv[1]), Path(sys.argv[2])
patients = {r["subject"]: r for r in csv.DictReader(open(READOUT / "patients.csv"))}
exposure = {r["subject"]: r for r in csv.DictReader(open(AUDIT / "drug_exposure.csv"))}
cov = {r["subject"]: r for r in csv.DictReader(open(AUDIT / "covariates.csv"))}
subjects = sorted(patients)
fold = np.array([int(patients[s]["fold"]) for s in subjects])
label = np.array([int(patients[s]["label"]) for s in subjects])
state = np.array([exposure[s]["status"] for s in subjects])
segments = np.array([float(patients[s]["segments"]) for s in subjects])
age = np.array([float(cov[s]["age"]) for s in subjects])
sex = np.array([cov[s]["sex"] for s in subjects])


def standardized(arm):
    rows = {r["subject"]: float(r["score"]) for r in csv.DictReader(open(READOUT / "scores" / f"{arm}.csv"))}
    s = np.array([rows[x] for x in subjects])
    out = s.copy()
    for f in np.unique(fold):
        out[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    return out


def wins(a, b):
    return (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])


def compare(a, b, mask=None, n_boot=10000, seed=0):
    w = wins(a, b).astype(float)
    m = np.ones_like(w) if mask is None else mask.astype(float)
    rng = np.random.default_rng(seed)
    pa = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=n_boot)
    pb = rng.multinomial(len(b), np.full(len(b), 1 / len(b)), size=n_boot)
    total = ((pa @ m) * pb).sum(axis=1)
    draws = ((pa @ (w * m)) * pb).sum(axis=1)[total > 0] / total[total > 0]
    return w[m > 0].mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), int(m.sum())


cells = {
    "epilepsy on": (label == 1) & (state == "on"),
    "epilepsy off": (label == 1) & (state == "off"),
    "non-epilepsy on": (label == 0) & (state == "on"),
    "non-epilepsy off": (label == 0) & (state == "off"),
}
print("cells (maintenance drugs only): n, median age, median segments")
for name, where in cells.items():
    print(
        f"  {name:18s} {int(where.sum()):3d}  age {np.median(age[where]):5.1f}  segments {np.median(segments[where]):6.1f}"
    )
for arm in ("Spectral", "HYDRA"):
    z = standardized(arm)
    print(f"\n== {arm}: mean standardized score")
    for name, where in cells.items():
        print(
            f"  {name:18s} {z[where].mean():+.2f}  values {np.round(np.sort(z[where]), 2).tolist() if where.sum() <= 10 else ''}"
        )
    for first, second in (
        ("epilepsy on", "epilepsy off"),
        ("epilepsy on", "non-epilepsy off"),
        ("epilepsy off", "non-epilepsy off"),
        ("non-epilepsy on", "non-epilepsy off"),
        ("epilepsy on", "non-epilepsy on"),
    ):
        v, lo, hi, _ = compare(z[cells[first]], z[cells[second]])
        print(f"  {first:18s} above {second:18s} {v:.3f} [{lo:.3f}, {hi:.3f}]")
    on, off = cells["epilepsy on"], cells["epilepsy off"]
    for ratio in (1.5, 2.0):
        near = np.abs(np.log(segments[on])[:, None] - np.log(segments[off])[None, :]) <= np.log(ratio)
        v, lo, hi, pairs = compare(z[on], z[off], near)
        print(
            f"  on above off, bag ratio within {ratio}: {v:.3f} [{lo:.3f}, {hi:.3f}] pairs {pairs}; on patients {int(near.any(axis=1).sum())}, off patients {int(near.any(axis=0).sum())}"
        )
    both = segments[on][:, None] <= segments[off].max()
    print(f"  on patients with a bag no larger than the largest off bag ({segments[off].max():.0f}): {int(both.sum())}")
    small = on & (segments <= segments[off].max())
    v, lo, hi, _ = compare(z[small], z[off])
    print(f"  on (those bags) above off: {v:.3f} [{lo:.3f}, {hi:.3f}]")
    large = on & (segments > segments[off].max())
    v, lo, hi, _ = compare(z[large], z[off])
    print(f"  on (larger bags) above off: {v:.3f} [{lo:.3f}, {hi:.3f}], n {int(large.sum())}")
print("\noff epilepsy bags:", np.sort(segments[cells["epilepsy off"]]).tolist())
print("on non-epilepsy bags:", np.sort(segments[cells["non-epilepsy on"]]).tolist())
print(
    "segments alone, on above off inside epilepsy:",
    compare(segments[cells["epilepsy on"]], segments[cells["epilepsy off"]])[:3],
)
