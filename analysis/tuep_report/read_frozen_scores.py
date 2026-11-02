# ruff: noqa: E501
"""Descriptive read of the frozen readout result files. Reads CSV, NPZ and JSON only; runs no project code."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
ARMS = [
    "S4-proj",
    "S4-untrained",
    "S4-atom",
    "S4-R1",
    "S4-sparse",
    "Spectral",
    "HYDRA",
    "S4-proj-all",
    "Spectral-all",
    "HYDRA-all",
]


def table(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


patients = {r["subject"]: r for r in table(ROOT / "patients.csv")}
scores = {}
for arm in ARMS:
    rows = sorted(table(ROOT / "scores" / f"{arm}.csv"), key=lambda r: r["subject"])
    scores[arm] = {
        "subject": np.array([r["subject"] for r in rows]),
        "fold": np.array([int(r["fold"]) for r in rows]),
        "label": np.array([int(r["label"]) for r in rows]),
        "score": np.array([float(r["score"]) for r in rows]),
    }
subject = scores["S4-proj"]["subject"]
fold = scores["S4-proj"]["fold"]
label = scores["S4-proj"]["label"]
segments = np.array([float(patients[s]["segments"]) for s in subject])
recordings = np.array([float(patients[s]["recordings"]) for s in subject])
rate = np.array([float(patients[s]["native_rate"]) for s in subject])


def wins(p, n):
    return (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])


def auroc(y, s):
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(wins(s[y == 1], s[y == 0]).mean())


def mean_fold(s):
    return float(np.mean([auroc(label[fold == f], s[fold == f]) for f in np.unique(fold)]))


def paired(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.unique(fold)
    draws = np.zeros((n_boot, len(folds)))
    point = np.zeros(len(folds))
    for i, f in enumerate(folds):
        w = fold == f
        pos, neg = w & (label == 1), w & (label == 0)
        d = wins(a[pos], a[neg]) - wins(b[pos], b[neg])
        point[i] = d.mean()
        n_p, n_n = d.shape
        pd = rng.integers(0, n_p, size=(n_boot, n_p))
        nd = rng.integers(0, n_n, size=(n_boot, n_n))
        for j in range(n_boot):
            draws[j, i] = d[np.ix_(pd[j], nd[j])].mean()
    m = draws.mean(axis=1)
    return float(point.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def single(a, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.unique(fold)
    draws = np.zeros((n_boot, len(folds)))
    for i, f in enumerate(folds):
        w = fold == f
        d = wins(a[w & (label == 1)], a[w & (label == 0)]).astype(float)
        n_p, n_n = d.shape
        pd = rng.integers(0, n_p, size=(n_boot, n_p))
        nd = rng.integers(0, n_n, size=(n_boot, n_n))
        for j in range(n_boot):
            draws[j, i] = d[np.ix_(pd[j], nd[j])].mean()
    m = draws.mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


print("== mean-fold AUROC with 95% interval (patient bootstrap)")
for arm in ARMS:
    s = scores[arm]["score"]
    assert np.array_equal(scores[arm]["subject"], subject) and np.array_equal(scores[arm]["fold"], fold)
    lo, hi = single(s)
    print(f"{arm:14s} {mean_fold(s):.3f} [{lo:.3f}, {hi:.3f}]")

print("\n== baselines, mean fold, same patients")
log_segments = np.log(segments)
print(f"segments {mean_fold(segments):.3f}; recordings {mean_fold(recordings):.3f}; rate {mean_fold(rate):.3f}")

print("\n== paired differences, descriptive")
pairs = [
    ("Spectral", "HYDRA"),
    ("Spectral", "S4-R1"),
    ("HYDRA", "S4-R1"),
    ("S4-R1", "S4-untrained"),
    ("S4-proj-all", "S4-proj"),
    ("Spectral-all", "Spectral"),
    ("HYDRA-all", "HYDRA"),
    ("HYDRA-all", "Spectral-all"),
    ("S4-proj-all", "Spectral-all"),
]
for a, b in pairs:
    print(f"{a} - {b}: %.3f [%.3f, %.3f]" % paired(scores[a]["score"], scores[b]["score"]))
for arm in ("Spectral", "HYDRA", "S4-R1"):
    print(f"segments - {arm}: %.3f [%.3f, %.3f]" % paired(segments, scores[arm]["score"]))


print("\n== AUROC within bag-size strata (tertiles of segment count over all patients), pooled scores")
edges = np.quantile(segments, [1 / 3, 2 / 3])
stratum = np.digitize(segments, edges)
print("edges", edges, "patients per stratum and class:")
for k in range(3):
    print(k, int(((stratum == k) & (label == 1)).sum()), int(((stratum == k) & (label == 0)).sum()))
for arm in ("Spectral", "HYDRA", "S4-R1", "S4-proj", "S4-untrained"):
    s = scores[arm]["score"]
    # scores differ in scale between folds only mildly; report pooled within stratum, plus the fold-standardized form
    z = s.copy()
    for f in np.unique(fold):
        z[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    within = [auroc(label[stratum == k], z[stratum == k]) for k in range(3)]
    weights = [((stratum == k) & (label == 1)).sum() * ((stratum == k) & (label == 0)).sum() for k in range(3)]
    print(
        arm,
        "per stratum",
        [round(w, 3) for w in within],
        f"pair-weighted {np.average(within, weights=weights):.3f}",
    )


print("\n== bag-size matched read: AUROC only on pairs (epilepsy, non-epilepsy) with segment ratio within 1.5")
for arm in ("Spectral", "HYDRA", "S4-R1", "S4-proj", "S4-untrained"):
    s = scores[arm]["score"]
    z = s.copy()
    for f in np.unique(fold):
        z[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    p, n = label == 1, label == 0
    w = wins(z[p], z[n])
    close = np.abs(np.log(segments[p])[:, None] - np.log(segments[n])[None, :]) <= np.log(1.5)
    print(f"{arm:14s} {w[close].mean():.3f} on {int(close.sum())} of {close.size} pairs")


print("\n== rate within 250 Hz only, mean-free pooled AUROC on fold-standardized scores, and group size")
g = rate == 250
print("250 Hz: epilepsy", int((g & (label == 1)).sum()), "non-epilepsy", int((g & (label == 0)).sum()))
for arm in ("Spectral", "HYDRA", "S4-R1", "S4-proj", "S4-untrained"):
    s = scores[arm]["score"]
    z = s.copy()
    for f in np.unique(fold):
        z[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    print(f"{arm:14s} 250Hz {auroc(label[g], z[g]):.3f}  segments within 250Hz {auroc(label[g], segments[g]):.3f}")


print("\n== correlation between arm scores, Spearman within class")
names = ["Spectral", "HYDRA", "S4-R1", "S4-proj", "S4-untrained", "S4-atom"]
z = {}
for arm in names:
    s = scores[arm]["score"]
    q = s.copy()
    for f in np.unique(fold):
        q[fold == f] = (s[fold == f] - s[fold == f].mean()) / s[fold == f].std()
    z[arm] = q
for i, a in enumerate(names):
    for b in names[i + 1 :]:
        print(
            f"{a} vs {b}: epilepsy {spearman(z[a][label == 1], z[b][label == 1]):+.2f}, non-epilepsy {spearman(z[a][label == 0], z[b][label == 0]):+.2f}"
        )


print("\n== S4 feature files: per fold, held-out versus dictionary-side patients")
for name in ("S4-proj", "S4-atom", "S4-untrained", "S4-R1"):
    print(name)
    for f in range(5):
        data = np.load(ROOT / "features" / f"{name}_f{f}.npz")
        sub, y, x = data["subject"], data["label"], data["features"]
        held = np.isin(sub, subject[fold == f])
        uniq = np.unique(sub)
        mean_row = x.mean(axis=1)
        zero = float((x == 0).mean())
        per = {s: mean_row[sub == s].mean() for s in uniq}
        v = np.array([per[s] for s in uniq])
        yy = np.array([y[sub == s][0] for s in uniq])
        h = np.isin(uniq, subject[fold == f])
        print(
            f" f{f} rows {x.shape} zero {zero:.3f} mean feature train {x[~held].mean():.4f} held {x[held].mean():.4f}"
            f" | mean-rate AUROC train {auroc(yy[~h], v[~h]):.3f} held {auroc(yy[h], v[h]):.3f}"
        )


print("\n== S4-proj: single-template AUROC on held-out patients (sign free of the head)")
for name in ("S4-proj", "S4-untrained"):
    print(name)
    for f in range(5):
        data = np.load(ROOT / "features" / f"{name}_f{f}.npz")
        sub, y, x = data["subject"], data["label"], data["features"]
        held_subjects = subject[fold == f]
        rows, ys = [], []
        for s in held_subjects:
            rows.append(x[sub == s].mean(axis=0))
            ys.append(y[sub == s][0])
        rows, ys = np.array(rows), np.array(ys)
        a = np.array([auroc(ys, rows[:, t]) for t in range(rows.shape[1])])
        print(
            f" f{f} median {np.median(a):.3f} q05 {np.quantile(a, 0.05):.3f} q95 {np.quantile(a, 0.95):.3f} max {a.max():.3f} min {a.min():.3f}"
        )


print("\n== dictionary health")
for kind in ("trained", "untrained"):
    for f in range(5):
        base = ROOT / "dictionaries" / f"{kind}_f{f}"
        rows = table(base / "sae_atom_stats.csv")
        won = np.array([float(r["won_windows"]) for r in rows])
        share = np.array([float(r["energy_share"]) for r in rows])
        order = np.argsort(share)[::-1]
        time_atoms = np.load(base / "sae_time_atoms.npy")
        spectrum = np.abs(np.fft.rfft(time_atoms, axis=-1)) ** 2
        freq = np.fft.rfftfreq(time_atoms.shape[-1], d=1 / 256.0)
        dom = freq[spectrum[:, 1:].argmax(axis=1) + 1]
        norm = time_atoms / np.linalg.norm(time_atoms, axis=1, keepdims=True)
        gram = np.abs(norm @ norm.T)
        np.fill_diagonal(gram, 0)
        alive = won > 0
        low = spectrum[:, freq <= 4].sum(1) / spectrum.sum(1)
        e = share / max(share.sum(), 1e-12)
        print(
            f"{kind} f{f}: alive {int(alive.sum())}/64, top-5 energy {e[order[:5]].sum():.3f},"
            f" dominant Hz over alive median {np.median(dom[alive]):.2f} max {dom[alive].max():.2f},"
            f" energy below 4 Hz {float((e * low).sum()):.3f},"
            f" coherence alive median {np.median(gram[np.ix_(alive, alive)].max(axis=1)):.2f}"
        )

print("\n== z")
for kind in ("trained", "untrained"):
    print(kind, [json.load(open(ROOT / "features" / f"z_{kind}_f{f}.json"))["z"] for f in range(5)])
