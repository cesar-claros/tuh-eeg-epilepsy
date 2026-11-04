"""Post-hoc read of the band-limited reconstruction (protocol section 13.4): B4-trained against the better fixed
reference and against PCA-64 in 4-30 Hz, with the protocol's own paired bootstrap. Result files only.

Usage: python -I read_fixed_reference.py <code folder> <band root> <readout root>
"""

import json
import sys
import types
from pathlib import Path

code, root, readout = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(code))
package = types.ModuleType("src")
package.__path__ = [str(code / "src")]
sys.modules["src"] = package

import numpy as np  # noqa: E402

import src.reconstruction_baselines_core as base  # noqa: E402

plan = json.loads((readout / "folds.json").read_text())
data = [dict(np.load(root / "scores" / f"fold{f}.npz")) for f in range(5)]
arms = data[0]["arms"].tolist()
subject = np.concatenate([d["subject"] for d in data]).astype(str)
patients = np.unique(subject)
index = np.searchsorted(patients, subject)
fast = [2, 3, 4]  # 4-8, 8-13, 13-30 Hz
signal = np.zeros(len(patients))
np.add.at(signal, index, np.concatenate([d["signal_bands"] for d in data])[:, fast].sum(axis=1))
residual = np.zeros((len(arms), len(patients)))
bands = np.concatenate([d["residual_bands"] for d in data], axis=1)[:, :, fast].sum(axis=2)
for i in range(len(arms)):
    np.add.at(residual[i], index, bands[i])
counts = base.bootstrap_counts(np.array([plan["folds"][s] for s in patients]))
for a, b in (("B4-trained", "B4-fixed"), ("B4-trained", "B1-fixed"), ("B4-trained", "PCA-64"), ("B1-trained", "B1-fixed")):
    d = base.paired_interval(residual[arms.index(a)], residual[arms.index(b)], signal, counts)
    print(f"{a} minus {b}, 4-30 Hz: {d.value:+.4f} ({d.low:+.4f} to {d.high:+.4f})")
for name in ("B4-fixed", "B1-fixed"):
    print(f"{name} EV in 4-30 Hz: {base.pooled_ev(residual[arms.index(name)], signal):+.4f}")
