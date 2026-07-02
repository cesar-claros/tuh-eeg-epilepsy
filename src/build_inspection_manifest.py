"""Build a CSV manifest of the plotted anomalous recordings, for human labeling.

Pairs the segment plots (execute_psd_segments.sh -> diagnostics/psd_segments/) and the
per-channel spectrograms (execute_spectrograms_channels.sh -> diagnostics/spectrograms/)
into one row per recording, so a labeler can open each plot and mark the corrupted channels
and time windows. Each row carries the subject / session / recording (parsed from the plot
filename), enriched with the native rate and anomaly scores from the rank_psd_anomaly.py
CSVs, the two plot filenames, and empty columns for the labeler to fill in.

Run from code/ (paths are relative), after the plot scripts have produced the PNGs::

    python src/build_inspection_manifest.py --csv diagnostics/psd/psd_anomaly-sfreq250-*.csv \
        diagnostics/psd/psd_anomaly-sfreq256-*.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# TUH EDF stem: <subject>_s<session digits>_t<recording digits> (e.g. aaaaapsj_s001_t002).
STEM_RE = re.compile(r"^(?P<subject>.+)_s(?P<session>\d+)_t(?P<recording>\d+)$")
LABELER_COLS = ["corrupted", "bad_channels", "bad_time_windows", "notes"]
COLS = [
    "native_sfreq", "rank", "subject", "session", "recording", "recording_id",
    "year", "montage", "class", "anomaly", "power", "roughness", "flatness", "dur_s",
    "segments_png", "spectrograms_png", *LABELER_COLS,
]


def _parse_stem(stem: str) -> dict:
    m = STEM_RE.match(stem)
    if not m:
        return {"subject": "", "session": "", "recording": ""}
    return {"subject": m.group("subject"), "session": "s" + m.group("session"), "recording": "t" + m.group("recording")}


def _load_anomaly_csvs(csv_paths) -> dict:
    """stem -> row dict from the rank_psd_anomaly.py CSVs (for rate / score enrichment)."""
    by_stem = {}
    for cp in csv_paths or []:
        with open(cp, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                print(f"!! {cp} has no 'path' column; skipping.")
                continue
            for r in reader:
                stem = Path(r["path"]).stem if r.get("path") else ""
                if stem:
                    by_stem[stem] = r
    return by_stem


def _scan(dir_path: str, suffix: str) -> dict:
    """{stem: filename} for files named <stem><suffix> under dir_path."""
    out = {}
    d = Path(dir_path)
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name.endswith(suffix):
                out[f.name[: -len(suffix)]] = f.name
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments_dir", default="diagnostics/psd_segments", help="psd_segments plot dir.")
    parser.add_argument("--spectrograms_dir", default="diagnostics/spectrograms", help="spectrogram_channels plot dir.")
    parser.add_argument("--csv", nargs="*", default=None, help="rank_psd_anomaly.py CSV(s) to enrich rate / scores.")
    parser.add_argument("--out", default=None,
                        help="Output manifest CSV (default: <segments_dir parent>/inspection_manifest.csv).")
    args = parser.parse_args()

    seg = _scan(args.segments_dir, "_segments.png")
    spec = _scan(args.spectrograms_dir, "_spectrograms.png")
    stems = sorted(set(seg) | set(spec))
    if not stems:
        raise SystemExit(f"No plots found in {args.segments_dir} or {args.spectrograms_dir}. Run the plot scripts first.")
    by_stem = _load_anomaly_csvs(args.csv)

    rows = []
    for stem in stems:
        p = _parse_stem(stem)
        a = by_stem.get(stem, {})
        # session folder / year / montage from the EDF path in the anomaly CSV, if present.
        parts = Path(a["path"]).parts if a.get("path") else ()
        montage = parts[-2] if len(parts) >= 2 else ""
        session_folder = parts[-3] if len(parts) >= 3 else ""
        ym = re.search(r"_(\d{4})$", session_folder) if session_folder else None
        rows.append({
            "native_sfreq": a.get("native_sfreq", ""),
            "subject": a.get("subject", "") or p["subject"],
            "session": p["session"],
            "recording": p["recording"],
            "recording_id": stem,
            "year": ym.group(1) if ym else "",
            "montage": montage,
            "class": a.get("class", ""),
            "anomaly": a.get("anomaly", ""),
            "power": a.get("power", ""),
            "roughness": a.get("roughness", ""),
            "flatness": a.get("flatness", ""),
            "dur_s": a.get("dur_s", ""),
            "segments_png": seg.get(stem, ""),
            "spectrograms_png": spec.get(stem, ""),
            **{c: "" for c in LABELER_COLS},
        })

    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    # Worst first within each native rate.
    rows.sort(key=lambda r: (_num(r["native_sfreq"]), -_num(r["anomaly"]), r["subject"], r["recording"]))
    rank = {}
    for r in rows:
        rate = r["native_sfreq"]
        rank[rate] = rank.get(rate, 0) + 1
        r["rank"] = rank[rate]

    out = Path(args.out) if args.out else Path(args.segments_dir).parent / "inspection_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    missing = sum(1 for r in rows if not r["segments_png"] or not r["spectrograms_png"])
    print(f"wrote {out}  ({len(rows)} recordings; {len(seg)} segment plots, {len(spec)} spectrograms"
          + (f"; {missing} missing one of the two plots)" if missing else ")"))


if __name__ == "__main__":
    main()
