# ruff: noqa: E501
"""Why did the age validity gate fail? Reads session_ages.csv and covariates.csv only: no label, class or score."""

import csv
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
sessions = defaultdict(list)
for row in csv.DictReader(open(ROOT / "session_ages.csv")):
    age = float(row["age"]) if row["age"] not in ("", "NaN") else float("nan")
    sessions[row["subject"]].append((row["session"], int(row["year"]), age))

print("patients", len(sessions), "sessions", sum(len(v) for v in sessions.values()))
print("session count per patient:", sorted(Counter(len(v) for v in sessions.values()).items()))
fractional = sum(1 for v in sessions.values() for s in v if not float(s[2]).is_integer())
print("sessions whose recordings disagree on age (non-integer mean):", fractional)

all_d, per_patient, top = [], {}, []
for subject, items in sessions.items():
    d = [(b[2] - a[2]) - (b[1] - a[1]) for a, b in itertools.combinations(items, 2)]
    if d:
        d = np.array(d)
        all_d += d.tolist()
        per_patient[subject] = float(np.mean(np.abs(d) <= 1))
        top.append((int((np.abs(d) > 1).sum()), len(d), subject))
all_d = np.array(all_d)
print("\nsession pairs", len(all_d), "share within 1 year", float(np.mean(np.abs(all_d) <= 1)))
print("|d| histogram over pairs:")
for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 200)):
    n = int(((np.abs(all_d) >= lo) & (np.abs(all_d) <= hi)).sum())
    print(f"  {lo:3d} to {hi:3d}: {n:5d}  {n / len(all_d):.3f}")
print("share within 2, 3, 5, 10 years:", [round(float(np.mean(np.abs(all_d) <= k)), 3) for k in (2, 3, 5, 10)])
print("sign of d: negative", int((all_d < -1).sum()), "positive", int((all_d > 1).sum()))

share = np.array(list(per_patient.values()))
print("\npatients with two or more sessions:", len(share))
print("patient-weighted share within 1 year (mean of each patient's share):", float(share.mean()))
print("patients with every pair within 1 year:", int((share == 1).sum()), "of", len(share))
top.sort(reverse=True)
print("failing pairs, top 10 patients (failing, total):", [(a, b) for a, b, _ in top[:10]])
failing = np.array([a for a, _, _ in top], dtype=float)
print(
    "share of all failing pairs held by top 1, 3, 5, 10 patients:",
    [round(float(failing[:k].sum() / failing.sum()), 3) for k in (1, 3, 5, 10)],
)

print("\nimplied birth year (year minus age), spread within each patient:")
spread = {s: max(y - a for _, y, a in v) - min(y - a for _, y, a in v) for s, v in sessions.items() if len(v) >= 2}
values = np.array(list(spread.values()))
for k in (1, 2, 3, 5, 10, 20):
    print(
        f"  spread at most {k:2d} years: {int((values <= k).sum()):3d} of {len(values)}  {float(np.mean(values <= k)):.3f}"
    )
print("  largest spreads:", sorted(values.tolist(), reverse=True)[:12])

print("\nform of the failure, sessions at least 2 years apart:")
apart = [
    (b[1] - a[1], b[2] - a[2])
    for v in sessions.values()
    for a, b in itertools.combinations(v, 2)
    if b[1] - a[1] >= 2 or a[1] - b[1] >= 2
]
apart = np.array(apart, dtype=float)
print(
    "  pairs",
    len(apart),
    "age unchanged",
    int((apart[:, 1] == 0).sum()),
    "age moves the opposite way",
    int((np.sign(apart[:, 1]) * np.sign(apart[:, 0]) < 0).sum()),
)
same_year = [(b[2] - a[2]) for v in sessions.values() for a, b in itertools.combinations(v, 2) if a[1] == b[1]]
same_year = np.array(same_year, dtype=float)
print(
    "  pairs in one calendar year:",
    len(same_year),
    "with an age difference above 1:",
    int((np.abs(same_year) > 1).sum()),
)

print("\nerror of each session against the patient's median birth year:")
errors = []
outlier_sessions = 0
for v in sessions.values():
    if len(v) < 2:
        continue
    births = np.array([y - a for _, y, a in v])
    e = births - np.median(births)
    errors += e.tolist()
errors = np.array(errors)
for k in (1, 2, 5, 10):
    print(f"  within {k:2d} years: {float(np.mean(np.abs(errors) <= k)):.3f}")
print(
    "  sessions",
    len(errors),
    "beyond 5 years:",
    int((np.abs(errors) > 5).sum()),
    "beyond 10 years:",
    int((np.abs(errors) > 10).sum()),
)

print("\nread age (covariates.csv) against the patient's median-birth-year age at the read year:")
diffs = []
for row in csv.DictReader(open(ROOT / "covariates.csv")):
    v = sessions[row["subject"]]
    if len(v) < 2:
        continue
    birth = np.median([y - a for _, y, a in v])
    diffs.append(float(row["age"]) - (float(row["read_year"]) - birth))
diffs = np.array(diffs)
print("  patients", len(diffs), "median |difference|", float(np.median(np.abs(diffs))))
for k in (1, 2, 5, 10):
    print(f"  within {k:2d} years: {float(np.mean(np.abs(diffs) <= k)):.3f}")
print(
    "  beyond 5 years:",
    int((np.abs(diffs) > 5).sum()),
    "largest:",
    sorted(np.abs(diffs).round(1).tolist(), reverse=True)[:8],
)
print(
    "ages outside 18 to 90 in session_ages (count only):",
    int(sum(1 for v in sessions.values() for s in v if s[2] < 18 or s[2] > 90)),
)
