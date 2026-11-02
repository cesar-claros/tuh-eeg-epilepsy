# ruff: noqa: E501
"""Session year and montage type of the 156 readout patients against their labels and held-out scores.

Reads the corpus directory listings and the result files only. Patients outside patients.csv (the locked test
patients among them) are dropped while the listing is parsed, before any statistic.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
LISTING = Path(sys.argv[2])
ARMS = ["Spectral", "HYDRA", "S4-R1", "S4-proj", "S4-untrained"]

patients = {r["subject"]: r for r in csv.DictReader(open(ROOT / "patients.csv"))}

files = defaultdict(list)  # subject -> [(session, year, montage)] per edf
for name in ("00_epilepsy.txt", "01_no_epilepsy.txt"):
    subject = session = year = montage = None
    for line in open(LISTING / name, encoding="utf-8"):
        token = re.sub(r"^[^a-zA-Z0-9]*", "", line.rstrip()).strip()
        if re.fullmatch(r"[a-z]{8}", token):
            subject = token if token in patients else None
        elif m := re.fullmatch(r"s(\d{3})_(\d{4})", token):
            session, year = int(m.group(1)), int(m.group(2))
        elif re.fullmatch(r"\d{2}_tcp_[a-z_]+", token):
            montage = token
        elif token.endswith(".edf") and subject is not None:
            files[subject].append((session, year, montage))

missing = [s for s in patients if s not in files]
print("patients", len(patients), "parsed", len(files), "missing", missing[:5])

rows = sorted(csv.DictReader(open(ROOT / "scores" / "S4-proj.csv")), key=lambda r: r["subject"])
subject = np.array([r["subject"] for r in rows])
fold = np.array([int(r["fold"]) for r in rows])
label = np.array([int(r["label"]) for r in rows])
segments = np.array([float(patients[s]["segments"]) for s in subject])
rate = np.array([float(patients[s]["native_rate"]) for s in subject])

first_year = np.array([min(f[1] for f in files[s]) for s in subject], dtype=float)
last_year = np.array([max(f[1] for f in files[s]) for s in subject], dtype=float)
median_year = np.array([np.median([f[1] for f in files[s]]) for s in subject])
span = last_year - first_year
sessions = np.array([len({f[0] for f in files[s]}) for s in subject], dtype=float)
montages = sorted({f[2] for s in subject for f in files[s]})
share = {m: np.array([np.mean([f[2] == m for f in files[s]]) for s in subject]) for m in montages}
majority = np.array([max(montages, key=lambda m: sum(f[2] == m for f in files[s])) for s in subject])
le_share = sum(share[m] for m in montages if "_le" in m)


def wins(p, n):
    return (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])


def auroc(y, s):
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(wins(s[y == 1], s[y == 0]).mean())


def mean_fold(s):
    return float(np.mean([auroc(label[fold == f], s[fold == f]) for f in np.unique(fold)]))


def interval(s, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.unique(fold)
    draws = np.zeros((n_boot, len(folds)))
    for i, f in enumerate(folds):
        w = fold == f
        d = wins(s[w & (label == 1)], s[w & (label == 0)]).astype(float)
        pd = rng.integers(0, d.shape[0], size=(n_boot, d.shape[0]))
        nd = rng.integers(0, d.shape[1], size=(n_boot, d.shape[1]))
        for j in range(n_boot):
            draws[j, i] = d[np.ix_(pd[j], nd[j])].mean()
    m = draws.mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def spearman(a, b):
    def ranks(v):
        return np.argsort(np.argsort(v + 1e-9 * np.random.default_rng(0).standard_normal(len(v)))).astype(float)

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


print("\n== class counts by year band (first session year)")
for lo, hi in ((2000, 2006), (2007, 2009), (2010, 2012), (2013, 2015), (2016, 2030)):
    w = (first_year >= lo) & (first_year <= hi)
    print(f"{lo}-{hi}: epilepsy {int((w & (label == 1)).sum())}, non-epilepsy {int((w & (label == 0)).sum())}")

print("\n== class counts by majority montage")
for m in montages:
    w = majority == m
    print(f"{m}: epilepsy {int((w & (label == 1)).sum())}, non-epilepsy {int((w & (label == 0)).sum())}")

print("\n== rate by first year and class")
for r in (250.0, 256.0):
    w = rate == r
    print(r, "first-year quartiles", np.quantile(first_year[w], [0, 0.25, 0.5, 0.75, 1]))

print("\n== metadata alone, mean-fold AUROC (above 0.5 = higher in epilepsy)")
for name, v in (
    ("segments", segments),
    ("sessions", sessions),
    ("first year", first_year),
    ("median year", median_year),
    ("last year", last_year),
    ("year span", span),
    ("share of linked-ear files", le_share),
    ("native rate", rate),
):
    lo, hi = interval(v)
    print(f"{name:28s} {mean_fold(v):.3f} [{lo:.3f}, {hi:.3f}]")

z = {}
for arm in ARMS:
    r = sorted(csv.DictReader(open(ROOT / "scores" / f"{arm}.csv")), key=lambda x: x["subject"])
    assert [x["subject"] for x in r] == list(subject)
    s = np.array([float(x["score"]) for x in r])
    q = s.copy()
    for f in np.unique(fold):
        q[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    z[arm] = q

print("\n== Spearman of score with metadata, inside each class (epilepsy, non-epilepsy)")
for arm in ARMS:
    parts = []
    for name, v in (
        ("median year", median_year),
        ("LE share", le_share),
        ("sessions", sessions),
        ("log segments", np.log(segments)),
    ):
        parts.append(
            f"{name} {spearman(z[arm][label == 1], v[label == 1]):+.2f} {spearman(z[arm][label == 0], v[label == 0]):+.2f}"
        )
    print(f"{arm:13s} " + " | ".join(parts))

print("\n== AUROC on matched pairs; scores standardized in each fold; pair count of 6083")
p, n = label == 1, label == 0
year_close = np.abs(median_year[p][:, None] - median_year[n][None, :]) <= 2
same_montage = majority[p][:, None] == majority[n][None, :]
same_rate = rate[p][:, None] == rate[n][None, :]
bag_close = np.abs(np.log(segments[p])[:, None] - np.log(segments[n])[None, :]) <= np.log(1.5)
for name, mask in (
    ("all pairs", np.ones_like(year_close)),
    ("year within 2", year_close),
    ("same montage", same_montage),
    ("same rate", same_rate),
    ("bag within 1.5x", bag_close),
    ("year + montage + rate", year_close & same_montage & same_rate),
    ("year + montage + rate + bag", year_close & same_montage & same_rate & bag_close),
):
    values = " ".join(f"{arm} {wins(z[arm][p], z[arm][n])[mask].mean():.3f}" for arm in ARMS)
    print(f"{name:30s} n={int(mask.sum()):5d}  {values}")
    if name.startswith("year + montage + rate + bag"):
        print(
            "   patients in matched pairs:",
            int(mask.any(axis=1).sum()),
            "epilepsy,",
            int(mask.any(axis=0).sum()),
            "non-epilepsy",
        )
    if name == "year + montage + rate":
        print(
            "   patients in matched pairs:",
            int(mask.any(axis=1).sum()),
            "epilepsy,",
            int(mask.any(axis=0).sum()),
            "non-epilepsy",
        )

print("\n== matched baselines: segments on the same matched pairs")
for name, mask in (("year + montage + rate", year_close & same_montage & same_rate),):
    print(name, "segments", round(float(wins(segments[p], segments[n])[mask].mean()), 3))

print("\n== band power against year and montage, patient mean over channels")
d = np.load(ROOT / "features" / "Spectral.npz")
sub, x = d["subject"], d["features"]
m = np.stack([x[sub == s].mean(axis=0) for s in subject]).reshape(len(subject), 20, 10)
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
for b, name in enumerate(BANDS):
    v = m[:, :, b].mean(axis=1)
    print(
        f"abs {name:6s} Spearman with median year: epilepsy {spearman(v[label == 1], median_year[label == 1]):+.2f},"
        f" non-epilepsy {spearman(v[label == 0], median_year[label == 0]):+.2f};"
        f" AUROC on year+montage+rate pairs {wins(v[p], v[n])[year_close & same_montage & same_rate].mean():.3f}"
    )
