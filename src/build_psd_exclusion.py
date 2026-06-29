"""Collect per-sampling-rate top-anomalous recordings into one training-exclusion file.

Reads one or more ``rank_psd_anomaly.py`` CSVs (each row a recording with an
``anomaly`` score and a ``native_sfreq``), selects the top-N most anomalous recordings
*per native sampling rate*, and writes the union of their EDF paths to a plain-text
exclusion file (one path per line; ``#`` comment lines and blanks are ignored by the
reader). Feed the file to training via ``data.exclude_recordings_file=<path>`` to drop
those recordings from the pool before windowing.

Typical flow (rank each rate, then pick how many of each to drop)::

    python src/rank_psd_anomaly.py --sfreq 250 --bipolar --notch_freqs 60 120 --out diagnostics/psd/anom-250.csv
    python src/rank_psd_anomaly.py --sfreq 256 --bipolar --notch_freqs 60 120 --out diagnostics/psd/anom-256.csv
    python src/rank_psd_anomaly.py --sfreq 400 --bipolar --notch_freqs 60 120 --out diagnostics/psd/anom-400.csv
    python src/rank_psd_anomaly.py --sfreq 1000 --bipolar --notch_freqs 60 120 --out diagnostics/psd/anom-1000.csv
    python src/build_psd_exclusion.py \
        --csv diagnostics/psd/anom-250.csv diagnostics/psd/anom-256.csv \
              diagnostics/psd/anom-400.csv diagnostics/psd/anom-1000.csv \
        --top 250:30 256:25 400:10 1000:5 \
        --out diagnostics/psd/exclude_recordings.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_NEEDED = {"anomaly", "path", "native_sfreq"}


def _parse_top(specs: list) -> dict:
    """['250:30', '256:25'] -> {250.0: 30, 256.0: 25}."""
    out: dict = {}
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"--top entry '{s}' must be RATE:N (e.g. 250:30).")
        rate, n = s.split(":", 1)
        try:
            out[float(rate)] = int(n)
        except ValueError:
            raise SystemExit(f"--top entry '{s}' must be RATE:N with numeric values (got '{s}').")
        if out[float(rate)] < 0:
            raise SystemExit(f"--top entry '{s}' must have N >= 0.")
    return out


def _read_rows(csv_paths: list) -> dict:
    """Pool rows from the anomaly CSVs, deduped by normalized path (keep max anomaly).

    Returns {normalized_path: (anomaly, native_sfreq, raw_path, subject)}.
    """
    best: dict = {}
    for cp in csv_paths:
        with open(cp, newline="") as f:
            reader = csv.DictReader(f)
            missing = _NEEDED - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"{cp} is missing columns {sorted(missing)}; is it a rank_psd_anomaly.py CSV?")
            for r in reader:
                p = str(Path(r["path"]))
                a = float(r["anomaly"])
                if p not in best or a > best[p][0]:
                    best[p] = (a, float(r["native_sfreq"]), r["path"], r.get("subject", ""))
    return best


def _default_out() -> Path:
    import rootutils

    return rootutils.setup_root(__file__, pythonpath=True) / "diagnostics" / "psd" / "exclude_recordings.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", nargs="+", required=True, help="rank_psd_anomaly.py CSV(s).")
    parser.add_argument(
        "--top", nargs="+", required=True, metavar="RATE:N",
        help="Per native-rate count of most-anomalous recordings to exclude, "
        "e.g. 250:30 256:25 400:10 1000:5.",
    )
    parser.add_argument("--out", default=None,
                        help="Output exclusion file (default: <root>/diagnostics/psd/exclude_recordings.txt).")
    parser.add_argument("--tol", type=float, default=0.5,
                        help="Hz tolerance matching a recording's native_sfreq to a --top rate (default 0.5).")
    args = parser.parse_args()

    top = _parse_top(args.top)
    rows = _read_rows(args.csv)

    summary = []  # (rate, requested, available, selected)
    seen: set = set()
    uniq: list = []  # normalized paths, in selection order
    for rate, n in sorted(top.items()):
        cand = sorted(
            ((a, sf, raw) for _p, (a, sf, raw, _s) in rows.items() if abs(sf - rate) <= args.tol),
            key=lambda t: t[0], reverse=True,
        )
        picked = cand[:n]
        for _a, _sf, raw in picked:
            key = str(Path(raw))
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        summary.append((rate, n, len(cand), len(picked)))

    out = Path(args.out) if args.out else _default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# PSD-anomaly training exclusion list (one EDF path per line; '#' lines ignored).\n")
        f.write(f"# sources: {', '.join(str(c) for c in args.csv)}\n")
        for rate, req, avail, got in summary:
            f.write(f"# rate {rate:g} Hz: requested top {req}, available {avail}, selected {got}\n")
        f.write(f"# total excluded: {len(uniq)}\n")
        for p in uniq:
            f.write(f"{p}\n")

    print(f"wrote {out}  ({len(uniq)} recordings)")
    for rate, req, avail, got in summary:
        note = "" if got == req else f"  (only {avail} available)"
        print(f"  {rate:g} Hz: top {req} -> {got} selected{note}")


if __name__ == "__main__":
    main()
