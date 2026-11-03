"""The age-matched read of the encoder check (protocol section 7, fixed before read D was opened). Result files only."""

import csv
import sys
import types
from pathlib import Path

code, check, audit, readout = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(code))
package = types.ModuleType("src")
package.__path__ = [str(code / "src")]
sys.modules["src"] = package

import numpy as np  # noqa: E402

import src.covariate_audit_core as core  # noqa: E402

cov = {r["subject"]: r for r in csv.DictReader(open(audit / "covariates.csv"))}


def scores(path):
    rows = sorted(csv.DictReader(open(path)), key=lambda r: r["subject"])
    return (
        np.array([r["subject"] for r in rows]),
        np.array([int(r["fold"]) for r in rows]),
        np.array([int(r["label"]) for r in rows]),
        np.array([float(r["score"]) for r in rows]),
    )


subject, fold, label, _ = scores(readout / "scores" / "Spectral.csv")
age = np.array([float(cov[s]["age"]) for s in subject])
sex = np.array([cov[s]["sex"] or core.MISSING_SEX for s in subject])
mask = core.pair_mask(label, age, sex)
print("pairs", int(mask.sum()), "patients", int(mask.any(axis=1).sum()), int(mask.any(axis=0).sum()))
arms = {"Spectral (saved)": readout / "scores" / "Spectral.csv"}
arms |= {f"REVE {name}": check / "scores" / f"{name}.csv" for name in ("D", "C", "H", "A-seiz")}
for name, path in arms.items():
    s, f, y, v = scores(path)
    assert (s == subject).all() and (f == fold).all() and (y == label).all(), name
    read = core.matched_read(v, label, fold, mask)
    print(f"{name:18s} matched {read.value:.3f} ({read.low:.3f} to {read.high:.3f})")
_, _, _, d = scores(check / "scores" / "D.csv")
_, _, _, spectral = scores(readout / "scores" / "Spectral.csv")
diff = core.matched_difference(d, spectral, label, fold, mask)
print(f"REVE D minus Spectral, matched {diff.value:.3f} ({diff.low:.3f} to {diff.high:.3f})")
z_d = core.standardize_within_fold(d, fold)
z_s = core.standardize_within_fold(spectral, fold)
for value, group in ((1, "epilepsy"), (0, "non-epilepsy")):
    where = label == value
    rho = np.corrcoef(np.argsort(np.argsort(z_d[where])), np.argsort(np.argsort(z_s[where])))[0, 1]
    rho_age = np.corrcoef(np.argsort(np.argsort(z_d[where])), np.argsort(np.argsort(age[where])))[0, 1]
    print(f"inside {group}: rank correlation of REVE D with Spectral {rho:+.2f}, with age {rho_age:+.2f}")
