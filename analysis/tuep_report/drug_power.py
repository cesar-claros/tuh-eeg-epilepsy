"""Rough power of the drug read at 98.33% intervals. Synthetic scores only; reads no project data."""

from statistics import NormalDist

import numpy as np

PHI = NormalDist()
Z = PHI.inv_cdf(1 - (1 - 0.9833) / 2)
rng = np.random.default_rng(0)


def auroc(a, b):
    return ((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).mean()


def group_power(n1, n2, true, draws=4000):
    shift = 2**0.5 * PHI.inv_cdf(true)
    v = np.array([auroc(rng.standard_normal(n1) + shift, rng.standard_normal(n2)) for _ in range(draws)])
    return v.mean(), v.std(), np.mean(v - Z * v.std() > 0.5)


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def count_power(n, rho, draws=4000):
    out = []
    for _ in range(draws):
        x = rng.standard_normal(n)
        y = rho * x + (1 - rho**2) ** 0.5 * rng.standard_normal(n)
        k = np.clip(np.round(x * 0.9 + 1.4), 0, 4)
        out.append(np.corrcoef(rank(k + 1e-6 * rng.standard_normal(n)), rank(y))[0, 1])
    v = np.array(out)
    return v.mean(), v.std(), np.mean(v - Z * v.std() > 0)


print("Z", round(Z, 3))
for sizes in ((66, 10), (35, 30), (50, 15)):
    for true in (0.60, 0.70, 0.80):
        m, s, p = group_power(*sizes, true)
        print(f"groups {sizes}, true AUROC {true:.2f}: sd {s:.3f}, half width {Z * s:.3f}, power {p:.2f}")
for rho in (0.2, 0.3, 0.4):
    m, s, p = count_power(76, rho)
    print(f"count n 76, latent correlation {rho:.1f}: read {m:.2f}, sd {s:.3f}, power {p:.2f}")
