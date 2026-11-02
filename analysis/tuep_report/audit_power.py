# ruff: noqa: E501
"""Rough power of the proposed covariate audit rule, by simulation. Synthetic patients only; reads no project data."""

from statistics import NormalDist

import numpy as np

PHI = NormalDist()
N_POS, N_NEG = 79, 77
CALIPER = 0.25  # 5 years with an age sd of about 20 years


def wins(p, n):
    return (p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])


def logistic(x, y, steps=12, ridge=1e-3):
    x1 = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(x1.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x1 @ b))
        w = np.clip(p * (1 - p), 1e-6, None)
        h = (x1 * w[:, None]).T @ x1 + ridge * np.eye(x1.shape[1])
        b += np.linalg.solve(h, x1.T @ (y - p) - ridge * b)
    return b


def predict(b, x):
    return np.column_stack([np.ones(len(x)), x]) @ b


def one(rng, d, delta, b_age):
    y = np.r_[np.ones(N_POS, int), np.zeros(N_NEG, int)]
    age = rng.standard_normal(len(y)) - d * y
    sex = rng.integers(0, 2, len(y)).astype(float)
    eeg = -b_age * age + delta * y + rng.standard_normal(len(y))
    fold = np.zeros(len(y), int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % 5
    m0, m1 = np.zeros(5), np.zeros(5)
    x0 = np.column_stack([age, age**2, sex])
    x1 = np.column_stack([x0, eeg])
    for f in range(5):
        test = fold == f
        s0 = predict(logistic(x0[~test], y[~test]), x0[test])
        s1 = predict(logistic(x1[~test], y[~test]), x1[test])
        yt = y[test]
        m0[f] = wins(s0[yt == 1], s0[yt == 0]).mean()
        m1[f] = wins(s1[yt == 1], s1[yt == 0]).mean()
    p, n = y == 1, y == 0
    close = (np.abs(age[p][:, None] - age[n][None, :]) <= CALIPER) & (sex[p][:, None] == sex[n][None, :])
    matched = wins(eeg[p], eeg[n])[close].mean()
    marginal = wins(eeg[p], eeg[n]).mean()
    return m0.mean(), m1.mean() - m0.mean(), matched, marginal, close.any(axis=1).sum(), close.any(axis=0).sum()


def scenario(a0, m, r, draws=1500, seed=0):
    rng = np.random.default_rng(seed)
    d = 2**0.5 * PHI.inv_cdf(a0)
    delta = 2**0.5 * PHI.inv_cdf(m)
    b_age = r / (1 - r**2) ** 0.5
    out = np.array([one(rng, d, delta, b_age) for _ in range(draws)])
    m0, diff, matched, marginal = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    s1 = diff - 1.96 * diff.std() > 0
    s2 = matched - 1.96 * matched.std() > 0.5
    return {
        "marginal": marginal.mean(),
        "m0": m0.mean(),
        "diff": diff.mean(),
        "sd_diff": diff.std(),
        "matched": matched.mean(),
        "sd_matched": matched.std(),
        "s1": s1.mean(),
        "s2": s2.mean(),
        "holds": (s1 & s2).mean(),
        "gone": (~s1 & ~s2).mean(),
        "mixed": (s1 ^ s2).mean(),
        "kept": (out[:, 4].mean(), out[:, 5].mean()),
    }


print(
    "a0 = AUROC of age alone; m = true AUROC of the score at fixed age and sex; r = within-class correlation of score and age"
)
print("a0   m    r   | marginal  M0    M1-M0  sd    | matched sd    | passS1 passS2 | holds gone mixed | patients kept")
for r in (0.3, 0.5):
    for a0 in (0.55, 0.65, 0.75):
        for m in (0.50, 0.55, 0.60, 0.65, 0.70):
            s = scenario(a0, m, r)
            print(
                f"{a0:.2f} {m:.2f} {r:.1f} | {s['marginal']:.3f}  {s['m0']:.3f} {s['diff']:+.3f} {s['sd_diff']:.3f} |"
                f" {s['matched']:.3f}  {s['sd_matched']:.3f} | {s['s1']:.2f}   {s['s2']:.2f}   |"
                f" {s['holds']:.2f}  {s['gone']:.2f} {s['mixed']:.2f} | {s['kept'][0]:.0f} {s['kept'][1]:.0f}"
            )
