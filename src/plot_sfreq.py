"""Per-class native sampling-rate histogram (diagnostic).

Reads the corpus descriptions (the native EDF sampling rate of every recording) and
plots / prints the distribution of native ``sfreq`` split by epilepsy class. Use it
to check whether the two cohorts differ in native sampling rate, which the fixed
``data.target_sfreq`` resample then exposes at the high-frequency end of the PSD
(e.g. the epilepsy traces not decaying toward Nyquist). Optionally restrict to the
recordings named in a run's ``windows_*.csv`` so it matches a specific PSD figure.

Run from ``code/`` (needs the corpus, like ``precompute_psd.py``)::

    python src/plot_sfreq.py
    python src/plot_sfreq.py --windows_csv logs/train/runs/<ts>/windows_train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(root / "data"), help="Parent of the version folder.")
    parser.add_argument("--version", default="v3.0.0", help="Corpus version subfolder.")
    parser.add_argument(
        "--windows_csv", default=None,
        help="Restrict to the recordings named in this run's windows CSV (match a PSD figure).",
    )
    parser.add_argument("--out", default=None, help="Output PNG (default: <cwd>/sfreq_by_class.png).")
    args = parser.parse_args()

    tuh = TUHEEGEpilepsy(data_dir=args.data_dir, version=args.version)
    df = tuh.descriptions[["path", "subject", "epilepsy", "sfreq"]].copy()
    df["epilepsy"] = df["epilepsy"].astype(bool).astype(int)
    if args.windows_csv:
        used = set(pd.read_csv(args.windows_csv)["path"].astype(str))
        df = df[df["path"].astype(str).isin(used)]
        if df.empty:
            raise SystemExit("No descriptions matched the windows CSV paths.")

    # Recording-level and subject-level crosstabs (a subject is counted once per
    # distinct native rate it has).
    rec_ct = pd.crosstab(df["sfreq"], df["epilepsy"]).rename(columns=CLASS_NAMES)
    subj = df.drop_duplicates(["subject", "sfreq", "epilepsy"])
    subj_ct = pd.crosstab(subj["sfreq"], subj["epilepsy"]).rename(columns=CLASS_NAMES)
    print("Recordings per native sfreq x class:\n", rec_ct, "\n")
    print("Subjects per native sfreq x class:\n", subj_ct)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = sorted(df["sfreq"].unique())
    x = np.arange(len(rates))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(rates)), 4.2))
    for k, cls in enumerate((0, 1)):
        counts = [int(((df["sfreq"] == r) & (df["epilepsy"] == cls)).sum()) for r in rates]
        bars = ax.bar(x + (k - 0.5) * w, counts, w, color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
        ax.bar_label(bars, fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r)}" for r in rates])
    ax.set_xlabel("native sampling rate (Hz)")
    ax.set_ylabel("recordings")
    ax.set_title("Native sampling rate per class" + (" (windows subset)" if args.windows_csv else ""))
    ax.legend()
    fig.tight_layout()
    out = Path(args.out) if args.out else Path.cwd() / "sfreq_by_class.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
