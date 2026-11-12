"""Post-hoc reads of the band-code readout (protocol section 13.5): the shape of the count features and paired
differences between saved arm scores, with the protocol's patient bootstrap. Result files only; no head is fitted.

These eight differences were chosen after the outcome was known, and no correction for their number applies, so they
suggest explanations and decide nothing.

Usage: python -I read_posthoc.py <code folder> <readout root> <pass 1 root>
"""

import sys
import types
from pathlib import Path

code, root, pass1 = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(code))
stub = types.ModuleType("rootutils")
stub.setup_root = lambda *a, **k: code
sys.modules["rootutils"] = stub
package = types.ModuleType("src")
package.__path__ = [str(code / "src")]
sys.modules["src"] = package

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.frozen_readout_core import ArmScores, paired_interval  # noqa: E402

PAIRS = (
    ("Code-templates", "Code"),
    ("Code-templates", "Band-scaled"),
    ("Band-raw", "Band-scaled"),
    ("Code+level", "Code"),
    ("Band-raw", "Code+level"),
    ("Spectral", "Band-raw"),
    ("Code-channels", "Code"),
    ("Code-tails", "Code"),
)


def _arm(path: Path) -> ArmScores:
    frame = pl.read_csv(path, schema_overrides={"subject": pl.Utf8})
    columns = (frame[name].to_numpy() for name in ("fold", "label", "score"))
    return ArmScores.from_rows(frame["subject"].to_numpy().astype(str), *columns)


print("count features of fold 0 (all 156 patients)")
for kind in ("trained", "untrained"):
    with np.load(root / "features" / f"{kind}_f0.npz") as data:
        hard = data["counts"][:, :64] * 2.0  # wins per 2-minute segment
        subject, label = data["subject"], data["label"]
    zero = (hard == 0).mean(axis=0)
    share = np.sort(hard.mean(axis=0) / 240)[::-1]
    patients = np.unique(subject)
    means = np.array([np.log1p(hard[subject == p] / 2.0).mean(axis=0) for p in patients])
    labels = np.array([label[subject == p][0] for p in patients])
    gap = (means[labels == 1].mean(axis=0) - means[labels == 0].mean(axis=0)) / (means.std(axis=0) + 1e-12)
    print(
        f"  {kind}: wins per segment {hard.sum(axis=1).min():.0f}-{hard.sum(axis=1).max():.0f} of 240; atoms 0 in "
        f">50% / >90% / all segments: {(zero > 0.5).sum()} / {(zero > 0.9).sum()} / {(zero == 1).sum()}; "
        f"atoms with 90% of wins: {np.searchsorted(np.cumsum(share), 0.9) + 1}; "
        f"largest standardized class gap of an atom: {np.abs(gap).max():.2f}"
    )

arms = {path.stem: _arm(path) for path in (root / "scores").glob("*.csv")}
arms["Spectral"] = _arm(pass1 / "scores" / "Spectral.csv")
print("paired differences of mean-fold AUROC, 10,000 patient draws")
for first, second in PAIRS:
    value, low, high = paired_interval(arms[first], arms[second], n_boot=10_000)
    print(f"  {first:15s} minus {second:12s} {value:+.3f} ({low:+.3f} to {high:+.3f})")
