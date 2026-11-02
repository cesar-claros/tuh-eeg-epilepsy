"""Numbers for the measurement report: the drug baseline and what the EEG adds. Result files only; fits nothing."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

AUDIT, READOUT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
patients = {r["subject"]: r for r in csv.DictReader(open(READOUT / "patients.csv"))}
exposure = {r["subject"]: r for r in csv.DictReader(open(AUDIT / "drug_exposure.csv"))}
subjects = sorted(patients)
fold = np.array([int(patients[s]["fold"]) for s in subjects])
label = np.array([int(patients[s]["label"]) for s in subjects])
state = np.array([exposure[s]["status"] for s in subjects])
segments = np.array([float(patients[s]["segments"]) for s in subjects])
recordings = np.array([float(patients[s]["recordings"]) for s in subjects])


def scores(arm):
    rows = {r["subject"]: float(r["score"]) for r in csv.DictReader(open(READOUT / "scores" / f"{arm}.csv"))}
    s = np.array([rows[x] for x in subjects])
    out = s.copy()
    for f in np.unique(fold):
        out[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    return out


def wins(a, b):
    return (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])


def counts(groups, n_boot, rng):
    c = np.zeros((n_boot, len(groups)))
    for g in np.unique(groups):
        w = np.flatnonzero(groups == g)
        c[:, w] = rng.multinomial(len(w), np.full(len(w), 1 / len(w)), size=n_boot)
    return c


def interval(values, keep, n_boot=10000, seed=0):
    """Mean over pairs of ``values`` (epilepsy by non-epilepsy) of kept patients, resampled within fold and class."""
    p, n = (label == 1) & keep, (label == 0) & keep
    v = values(p, n)
    rng = np.random.default_rng(seed)
    a, b = counts(fold[p], n_boot, rng), counts(fold[n], n_boot, rng)
    total = a.sum(axis=1) * b.sum(axis=1)
    draws = ((a @ v) * b).sum(axis=1) / total
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mean_fold(v, keep):
    return float(
        np.mean(
            [
                wins(v[keep & (fold == f) & (label == 1)], v[keep & (fold == f) & (label == 0)]).mean()
                for f in np.unique(fold)
            ]
        )
    )


known = state != "unknown"
on = (state == "on").astype(float)
every = np.ones(len(label), bool)
result = {}
result["known"] = {
    "patients": int(known.sum()),
    "epilepsy": int((known & (label == 1)).sum()),
    "non_epilepsy": int((known & (label == 0)).sum()),
}
result["drug_alone_known"] = interval(lambda p, n: wins(on[p], on[n]).astype(float), known)
result["drug_alone_known_mean_fold"] = mean_fold(on, known)
result["drug_alone_unknown_as_off"] = interval(lambda p, n: wins(on[p], on[n]).astype(float), every)
for name, x in (("segments", segments), ("recordings", recordings)):
    result[f"{name}_known"] = interval(lambda p, n: wins(x[p], x[n]).astype(float), known)
for arm in ("Spectral", "HYDRA"):
    z = scores(arm)
    result[f"{arm}_known"] = interval(lambda p, n: wins(z[p], z[n]).astype(float), known)
    # Rank by drug status first, and let the EEG score break the ties. No model is fitted.
    combined = on * 1000.0 + z
    result[f"{arm}_drug_then_score"] = interval(lambda p, n: wins(combined[p], combined[n]).astype(float), known)
    result[f"{arm}_added_to_drug"] = interval(
        lambda p, n: wins(combined[p], combined[n]).astype(float) - wins(on[p], on[n]).astype(float), known
    )
    result[f"{arm}_minus_drug"] = interval(
        lambda p, n: wins(z[p], z[n]).astype(float) - wins(on[p], on[n]).astype(float), known
    )
    cells = {}
    for name, where in (
        ("Epilepsy, on a maintenance drug", (label == 1) & (state == "on")),
        ("Epilepsy, off", (label == 1) & (state == "off")),
        ("Non-epilepsy, on a maintenance drug", (label == 0) & (state == "on")),
        ("Non-epilepsy, off", (label == 0) & (state == "off")),
    ):
        rng = np.random.default_rng(1)
        draws = counts(fold[where], 10000, rng)
        means = (draws @ z[where]) / draws.sum(axis=1)
        cells[name] = {
            "n": int(where.sum()),
            "mean": float(z[where].mean()),
            "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5)),
            "values": np.round(z[where], 3).tolist(),
        }
    result[f"{arm}_cells"] = cells
OUT.write_text(json.dumps(result, indent=1))
for key, value in result.items():
    if not key.endswith("_cells"):
        print(key, value if not isinstance(value, tuple) else tuple(round(x, 3) for x in value))
for arm in ("Spectral", "HYDRA"):
    print(
        arm,
        {
            k: (v["n"], round(v["mean"], 2), round(v["low"], 2), round(v["high"], 2))
            for k, v in result[f"{arm}_cells"].items()
        },
    )
