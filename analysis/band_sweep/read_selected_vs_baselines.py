"""Post-hoc read of the band sweep (protocol section 13.9): the adopted setting against the baselines of the
band-limited run, on the same rows, with the protocols' paired patient bootstrap. Result files only.

The confirmation patients (folds 2 to 4) did not enter the selection, so they give the clean read. All 156 patients
add the screen patients, on which the setting was selected, so that read is optimistic.

Usage: python -I read_selected_vs_baselines.py <code folder> <sweep root> <band root> <readout root>
"""

import json
import sys
import types
from pathlib import Path

code, sweep, band, readout = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(code))
package = types.ModuleType("src")
package.__path__ = [str(code / "src")]
sys.modules["src"] = package

import numpy as np  # noqa: E402

import src.reconstruction_baselines_core as base  # noqa: E402

SELECTED = "K64_L128_D8_P512_R4_E30"
COLUMNS = {"total": None, "4_30": [2, 3, 4], "13_30": [4], "1_4": [1], "4_8": [2], "8_13": [3]}
BASELINES = ("B4-trained", "B4-untrained", "B1-fixed", "B4-fixed", "S4-pass1", "PCA-4", "PCA-16", "PCA-64", "SVD-4")
plan = json.loads((readout / "folds.json").read_text())


def per_patient(subject, signal_bands, signal_total, residual_bands, residual_total):
    patients = np.unique(subject)
    index = np.searchsorted(patients, subject)
    out = {}
    for name, cols in COLUMNS.items():
        s = signal_total if cols is None else signal_bands[:, cols].sum(axis=1)
        r = residual_total if cols is None else residual_bands[:, cols].sum(axis=1)
        out[name] = (np.bincount(index, s, len(patients)), np.bincount(index, r, len(patients)))
    return patients, out


def load(folds, seed):
    parts, bands = [], []
    for f in folds:
        with np.load(sweep / "scores" / f"{SELECTED}_s{seed}_f{f}.npz") as d:
            parts.append({k: d[k] for k in ("subject", "signal_bands", "signal_total", "residual_bands", "residual_total")})
        with np.load(band / "scores" / f"fold{f}.npz") as d:
            bands.append({k: d[k] for k in d.files})
    subject = np.concatenate([p["subject"] for p in parts]).astype(str)
    assert subject.tolist() == np.concatenate([b["subject"] for b in bands]).astype(str).tolist(), "segments differ"
    sb = np.concatenate([p["signal_bands"] for p in parts])
    assert np.allclose(sb, np.concatenate([b["signal_bands"] for b in bands]), rtol=1e-9), "rows differ"
    st = np.concatenate([p["signal_total"] for p in parts])
    patients, selected = per_patient(subject, sb, st, np.concatenate([p["residual_bands"] for p in parts]),
                                     np.concatenate([p["residual_total"] for p in parts]))
    arms = bands[0]["arms"].tolist()
    others = {}
    for name in BASELINES:
        i = arms.index(name)
        _, others[name] = per_patient(subject, sb, st, np.concatenate([b["residual_bands"][i] for b in bands]),
                                      np.concatenate([b["residual_total"][i] for b in bands]))
    counts = base.bootstrap_counts(np.array([plan["folds"][p] for p in patients]))
    return patients, selected, others, counts


def report(title, folds, seed):
    patients, selected, others, counts = load(folds, seed)
    print(f"\n== {title}: {len(patients)} patients, adopted setting at seed {seed}")
    header = "".join(f"{c:>9s}" for c in COLUMNS)
    print(f"{'EV':28s}{header}")
    print(f"{'adopted ' + SELECTED[:12]:28s}" + "".join(f"{base.pooled_ev(selected[c][1], selected[c][0]):9.4f}" for c in COLUMNS))
    for name in BASELINES:
        print(f"{name:28s}" + "".join(f"{base.pooled_ev(others[name][c][1], others[name][c][0]):9.4f}" for c in COLUMNS))
    print("adopted minus baseline (95% interval):")
    for name in BASELINES:
        cells = []
        for c in ("total", "4_30", "13_30"):
            d = base.paired_interval(selected[c][1], others[name][c][1], selected[c][0], counts)
            cells.append(f"{c} {d.value:+.4f} ({d.low:+.4f} to {d.high:+.4f})")
        print(f"  {name:12s} " + "   ".join(cells))
    share = {c: base.pooled_ev(selected[c][1], selected[c][0]) / base.pooled_ev(others["SVD-4"][c][1], others["SVD-4"][c][0])
             for c in ("total", "4_30", "13_30")}
    print("share of the SVD-4 ceiling:", {c: round(v, 3) for c, v in share.items()})


report("confirmation folds 2-4 (clean)", (2, 3, 4), 1)
report("confirmation folds 2-4 (clean)", (2, 3, 4), 2)
report("all folds (screen folds 0-1 selected on, so optimistic)", (0, 1, 2, 3, 4), 1)
