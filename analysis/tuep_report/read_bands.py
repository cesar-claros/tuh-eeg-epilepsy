# ruff: noqa: E501
"""Descriptive read of band power per class and atom copies. Reads result files only."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
BANDS = ["delta 1-4", "theta 4-8", "alpha 8-13", "beta 13-30", "gamma 30-45"]


def wins(p, n):
    return (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])


pat = {r["subject"]: r for r in csv.DictReader(open(ROOT / "patients.csv"))}
d = np.load(ROOT / "features" / "Spectral.npz")
sub, y, x = d["subject"], d["label"], d["features"]
print("spectral rows", x.shape)
uniq = np.unique(sub)
m = np.stack([x[sub == s].mean(axis=0) for s in uniq]).reshape(len(uniq), 20, 10)
lab = np.array([int(pat[s]["label"]) for s in uniq])
seg = np.array([float(pat[s]["segments"]) for s in uniq])
rate = np.array([float(pat[s]["native_rate"]) for s in uniq])
rng = np.random.default_rng(0)


def boot(v, mask=None, n=4000):
    mask = np.ones(len(v), bool) if mask is None else mask
    w = wins(v[mask & (lab == 1)], v[mask & (lab == 0)]).astype(float)
    a, b = w.shape
    pd = rng.integers(0, a, size=(n, a))
    nd = rng.integers(0, b, size=(n, b))
    draws = np.array([w[np.ix_(pd[j], nd[j])].mean() for j in range(n)])
    return w.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


print("AUROC of the mean over channels, epilepsy high = above 0.5; all patients and 250 Hz only")
g = rate == 250
p, n = lab == 1, lab == 0
close = np.abs(np.log(seg[p])[:, None] - np.log(seg[n])[None, :]) <= np.log(1.5)
for kind, off in (("abs", 0), ("rel", 5)):
    for b, name in enumerate(BANDS):
        v = m[:, :, off + b].mean(axis=1)
        a = boot(v)
        print(
            f"{kind} {name:12s} {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]  250Hz {boot(v, g, 1000)[0]:.3f}  bag-matched {wins(v[p], v[n])[close].mean():.3f}  class medians {np.median(v[p]):.3f} {np.median(v[n]):.3f}"
        )
for kind, off in (("abs", 0), ("rel", 5)):
    tot = m[:, :, off : off + 5].mean(axis=(1, 2))
ch = open(ROOT / "dictionaries" / "trained_f0" / "sae_channels.txt").read().split()
print("per-channel AUROC, beta abs and rel, delta abs")
for c, name in enumerate(ch):
    print(name, " ".join(f"{wins(m[p, c, i], m[n, c, i]).mean():.2f}" for i in (3, 8, 0, 5)))
print("\natom copies among alive atoms")
for f in range(5):
    base = ROOT / "dictionaries" / f"trained_f{f}"
    rows = list(csv.DictReader(open(base / "sae_atom_stats.csv")))
    won = np.array([float(r["won_windows"]) for r in rows])
    t = np.load(base / "sae_time_atoms.npy")
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    alive = won > 0
    gram = np.abs(t @ t.T)
    np.fill_diagonal(gram, 0)
    # greedy groups at |cos| above 0.9 among alive atoms
    left = list(np.flatnonzero(alive))
    groups = []
    while left:
        i = left.pop(0)
        g2 = [i] + [j for j in left if gram[i, j] > 0.9]
        left = [j for j in left if j not in g2]
        groups.append(g2)
    sizes = sorted((len(z) for z in groups), reverse=True)
    print(
        f"f{f}: alive {alive.sum()}, groups at 0.9: {len(groups)}, sizes {sizes[:6]}, coherence dead-atom median {np.median(gram[~alive][:, ~alive].max(axis=1)):.2f}, all-atom median {np.median(gram.max(axis=1)):.2f}"
    )
