"""Rank recordings by PSD anomaly (energy + spectral roughness).

Finds recordings whose cached PSD is unusual: very high or low total power, or a
"wiggly" (rough) spectrum, a comb of peaks / ripple that signals periodic artifacts,
clipping, short-record variance, or a degenerate / bad-channel signal. Per recording
it loads the channel-averaged PSD and scores:

  - power_db   : 10 log10 of total power (the "norm"); extremes are loud / silent.
  - roughness_db : RMS of (log-PSD minus a smoothed log-PSD), the ripple amplitude in
                   dB; high = wiggly (peaky comb or jittery estimate).
  - flatness   : spectral flatness in [0, 1]; low = tonal / peaky.

Operates on the same sidecars as the other PSD tools, so pass the matching
``--bipolar`` / ``--notch_freqs`` / ``--native`` you precomputed. Source: ``--sfreq``
/ ``--all-recordings`` (whole corpus, like plot_sfreq) or ``--windows_csv`` (one run).
Writes a ranked CSV and prints the top / bottom recordings; feed the worst paths to
the segment viewer.

    python src/rank_psd_anomaly.py --sfreq 250 --native --bipolar --notch_freqs 60 120 --top 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
_EPS = 1e-30


def _psd_suffix(bipolar: bool = False, notch_freqs=None, native: bool = False) -> str:
    """PSD sidecar suffix. MUST match TUHEEGEpilepsy._psd_suffix (the writer)."""
    s = "-psd"
    if bipolar:
        s += "-bipolar"
    if native:
        s += "-native"
    if notch_freqs:
        s += "-notch-" + "-".join(str(int(round(f))) for f in notch_freqs)
    return s + ".npz"


def _recording_metrics(edf: str, suffix: str, smooth_hz: float = 5.0):
    """Anomaly metrics for one recording's channel-averaged PSD, or None if missing."""
    npz = Path(str(edf).replace(".edf", suffix))
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=False)
    freqs = d["freqs"]
    cm = d["psd"].mean(axis=0)  # channel-average (matches the per-subject figure)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    power = float(cm.sum() * df)  # ~ total power, the norm
    lp = 10.0 * np.log10(cm + _EPS)  # log-PSD (dB)
    w = max(5, int(round(smooth_hz / df)))
    w += 1 - (w % 2)  # force odd so the moving average is centered
    smooth = np.convolve(lp, np.ones(w) / w, mode="same")
    roughness = float(np.sqrt(np.mean((lp - smooth) ** 2)))  # ripple RMS (dB)
    flatness = float(np.exp(np.mean(np.log(cm + _EPS))) / (cm.mean() + _EPS))
    return {
        "native_sfreq": round(float(d["sfreq"]), 1),
        "n_channels": int(d["psd"].shape[0]),
        "dur_s": round(int(d["n_times"]) / float(d["sfreq"]), 1),
        "power_db": round(10.0 * np.log10(power + _EPS), 2),
        "roughness_db": round(roughness, 3),
        "flatness": round(flatness, 4),
    }


def _records_from_windows(windows_csv: str):
    df = pd.read_csv(windows_csv).drop_duplicates("path")
    return [(str(r["path"]), str(r["subject"]), int(bool(r["epilepsy"]))) for _, r in df.iterrows()]


def _records_from_corpus(data_dir, version, sfreq, exclude_seizures=False, min_duration_s=None):
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: PLC0415

    # add_annotations=True so descriptions carries n_seizure / duration (needed below).
    tuh = TUHEEGEpilepsy(data_dir=data_dir or str(root / "data"), version=version, add_annotations=True)
    df = tuh.descriptions[["path", "subject", "epilepsy", "sfreq", "n_seizure", "duration"]].copy()
    df["epilepsy"] = df["epilepsy"].astype(bool).astype(int)
    if sfreq is not None:
        df = df[np.isclose(df["sfreq"].astype(float), float(sfreq))]
    if exclude_seizures:
        df = df[df["n_seizure"] == 0]
    if min_duration_s is not None:
        df = df[df["duration"].astype(float) >= float(min_duration_s)]
    if df.empty:
        raise SystemExit("No recordings left after the sfreq/seizure/duration filters.")
    return [(str(r["path"]), str(r["subject"]), int(r["epilepsy"])) for _, r in df.iterrows()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows_csv", default=None, help="windows_train.csv from a run.")
    parser.add_argument("--all-recordings", action="store_true", dest="all_recordings",
                        help="Use the whole corpus.")
    parser.add_argument("--sfreq", type=float, default=None, help="Corpus mode: keep only this native rate.")
    parser.add_argument("--data_dir", default=None, help="Corpus mode: parent of the version folder.")
    parser.add_argument("--version", default="v3.0.0", help="Corpus mode: version subfolder.")
    parser.add_argument("--exclude_seizures", action="store_true",
                        help="Corpus mode: drop recordings with a seizure annotation.")
    parser.add_argument("--min_duration_min", type=float, default=None,
                        help="Corpus mode: drop recordings shorter than this many minutes.")
    parser.add_argument("--bipolar", action="store_true", help="Read the bipolar sidecars.")
    parser.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Read the notched sidecars.")
    parser.add_argument("--native", action="store_true", help="Read the native-rate sidecars.")
    parser.add_argument("--rank", default="roughness_db", choices=["roughness_db", "power_db", "flatness"],
                        help="Metric to sort by (default roughness_db).")
    parser.add_argument("--top", type=int, default=15, help="Rows to print at each end.")
    parser.add_argument("--out", default=None, help="Output CSV (default: ./psd_anomaly<scope>.csv).")
    args = parser.parse_args()
    suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)

    if args.all_recordings or args.sfreq is not None:
        records = _records_from_corpus(
            args.data_dir, args.version, args.sfreq,
            exclude_seizures=args.exclude_seizures,
            min_duration_s=None if args.min_duration_min is None else args.min_duration_min * 60.0,
        )
    elif args.windows_csv:
        records = _records_from_windows(args.windows_csv)
    else:
        raise SystemExit("Provide --windows_csv, or --all-recordings / --sfreq for the whole corpus.")

    rows = []
    for path, subj, cls in records:
        m = _recording_metrics(path, suffix)
        if m is None:
            continue
        rows.append({"subject": subj, "class": CLASS_NAMES[cls], **m, "path": path})
    if not rows:
        raise SystemExit(f"No {suffix} sidecars found for those recordings.")

    out = pd.DataFrame(rows)
    ascending = args.rank == "flatness"  # low flatness = tonal/abnormal
    out = out.sort_values(args.rank, ascending=ascending).reset_index(drop=True)

    scope = f"sfreq{args.sfreq:g}" if args.sfreq is not None else ("allrates" if args.all_recordings else "windows")
    csv_path = Path(args.out) if args.out else Path.cwd() / f"psd_anomaly-{scope}{suffix.replace('.npz','')}.csv"
    out.to_csv(csv_path, index=False)

    cols = ["subject", "class", "native_sfreq", "dur_s", "power_db", "roughness_db", "flatness", "path"]
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(f"\n=== most anomalous by {args.rank} ({len(out)} recordings) ===")
        print(out[cols].head(args.top).to_string(index=False))
        print(f"\n=== least anomalous by {args.rank} ===")
        print(out[cols].tail(args.top).to_string(index=False))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
