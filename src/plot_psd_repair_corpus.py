"""Overlay the corpus-mean PSD before vs after repair (interpolation and/or AAS).

Compares the raw per-recording PSDs (``-psd.npz``) against the repaired ones
(``-psd-interp.npz`` / ``-psd-aas.npz`` / ``-psd-interp-aas.npz`` from
``precompute_psd.py --interpolate --aas``) to show the aggregate improvement across the
corpus. For each recording it channel-averages both PSDs on the shared grid, then means
them across recordings. Columns are the two population groups: ALL recordings, and the
AFFECTED subset (recordings a repair actually touches, i.e. with flagged bad channels
when ``--interpolate`` and/or a flagged cardiac-band comb when ``--aas``), where the
change is largest and the harmonic comb should flatten out.

Options:
  --by_class     split each curve into epilepsy vs no-epilepsy (dashed = raw, solid =
                 repaired; color = class), to see whether the repair changes the
                 class-discriminative PSD differently for the two groups.
  --bands        segment the spectrum into one row per canonical EEG band
                 (delta/theta/alpha/beta/gamma), each on its own y-scale; --band_edges
                 F0 F1 ... sets custom segment edges instead.
  --native --sfreq R
                 read the native-rate sidecars and restrict to recordings sampled at R
                 Hz (native-rate PSDs share a grid only within one sampling rate).

Run on the machine that holds the corpus and both sets of ``-psd*.npz`` sidecars, after
``precompute_badchannels.py`` and both ``precompute_psd.py`` passes (raw and repaired).

Examples
--------
::

    # after: precompute_psd.py --n_jobs 8   AND   precompute_psd.py --n_jobs 8 --interpolate --aas
    python src/plot_psd_repair_corpus.py --interpolate --aas
    python src/plot_psd_repair_corpus.py --aas --by_class --bands --exclude_seizures --min_duration_min 2
    python src/plot_psd_repair_corpus.py --interpolate --aas --native --sfreq 250 --by_class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)

CLASSES = (0, 1)
CLASS_NAMES = {0: "no-epilepsy", 1: "epilepsy"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}
# Canonical EEG bands (name, lo, hi); the top band is clipped to the plotted maximum.
CANONICAL_BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
                   ("beta", 13.0, 30.0), ("gamma", 30.0, None)]
_EPS = 1e-20


def _psd_suffix(bipolar: bool = False, notch_freqs=None, native: bool = False,
                interpolate: bool = False, aas: bool = False) -> str:
    """PSD sidecar suffix. MUST match TUHEEGEpilepsy._psd_suffix (the writer)."""
    s = "-psd"
    if bipolar:
        s += "-bipolar"
    if native:
        s += "-native"
    if interpolate:
        s += "-interp"
    if aas:
        s += "-aas"
    if notch_freqs:
        s += "-notch-" + "-".join(str(int(round(f))) for f in notch_freqs)
    return s + ".npz"


def _channel_mean_psd(edf_path: str, suffix: str):
    """(freqs, channel-averaged PSD) for one recording's sidecar, or None if missing."""
    npz = Path(str(edf_path).replace(".edf", suffix))
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=False)
    return d["freqs"], d["psd"].mean(axis=0)


def _affected(edf_path: str, interpolate: bool, aas: bool, aas_fmax: float) -> bool:
    """True if a repair actually touches this recording (per its ``-bads.json`` sidecar).

    Interpolation touches a recording with >=1 flagged bad channel that is not
    ``too_many_bad_channels`` (those are not interpolated). AAS touches a recording whose
    periodic artifact is flagged with a fundamental in the cardiac band (<= aas_fmax).
    """
    sidecar = Path(str(edf_path).replace(".edf", "-bads.json"))
    if not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text()) or {}
    except Exception:  # noqa: BLE001
        return False
    if interpolate and meta.get("bad_channels") and not meta.get("too_many_bad_channels", False):
        return True
    if aas:
        pa = meta.get("periodic_artifact") or {}
        f0 = pa.get("fundamental_hz")
        if pa.get("flagged") and f0 and 0 < f0 <= aas_fmax:
            return True
    return False


def _records(args):
    """(path, subject, epilepsy) records from a run's windows_csv or the whole corpus."""
    if args.windows_csv:
        import pandas as pd

        df = pd.read_csv(args.windows_csv).drop_duplicates("path")
        return [(str(r["path"]), str(r["subject"]), int(bool(r["epilepsy"]))) for _, r in df.iterrows()]
    from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: PLC0415

    tuh = TUHEEGEpilepsy(data_dir=args.data_dir or str(root / "data"), version=args.version,
                         add_annotations=True)
    df = tuh.descriptions[["path", "subject", "epilepsy", "sfreq", "n_seizure", "duration"]].copy()
    if args.sfreq is not None:
        df = df[np.isclose(df["sfreq"].astype(float), float(args.sfreq))]
    if args.exclude_seizures:
        df = df[df["n_seizure"] == 0]
    if args.min_duration_min is not None:
        df = df[df["duration"].astype(float) >= float(args.min_duration_min) * 60.0]
    if df.empty:
        raise SystemExit("No recordings left after the sfreq/seizure/duration filters.")
    return [(str(r["path"]), str(r["subject"]), int(bool(r["epilepsy"]))) for _, r in df.iterrows()]


def _build_bands(use_bands: bool, band_edges, top: float):
    """List of (label, lo, hi) frequency segments to draw as rows.

    ``band_edges`` (if given) defines consecutive segments; else ``use_bands`` selects the
    canonical EEG bands (clipped to ``top``); else a single full-range segment.
    """
    if band_edges:
        edges = sorted({float(e) for e in band_edges})
        return [(f"{lo:g}-{hi:g} Hz", lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    if use_bands:
        out = []
        for name, lo, hi in CANONICAL_BANDS:
            hi = top if hi is None else min(hi, top)
            if lo < top:
                out.append((f"{name}\n{lo:g}-{hi:g} Hz", lo, hi))
        return out
    return [(f"0-{top:g} Hz", 0.0, top)]


def _group_total(acc_g):
    """Sum the per-class (raw, rep, n) of one population group into one (raw, rep, n)."""
    raw = rep = None
    n = 0
    for cls in CLASSES:
        b = acc_g[cls]
        if b["n"] == 0:
            continue
        raw = b["raw"] if raw is None else raw + b["raw"]
        rep = b["rep"] if rep is None else rep + b["rep"]
        n += b["n"]
    return raw, rep, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows_csv", default=None,
                   help="Restrict to a run's training recordings (windows_train.csv). Default: whole corpus.")
    p.add_argument("--data_dir", default=None, help="Corpus mode: parent of the version folder.")
    p.add_argument("--version", default="v3.0.0", help="Corpus mode: version subfolder (default v3.0.0).")
    p.add_argument("--sfreq", type=float, default=None,
                   help="Corpus mode: keep only recordings at this native rate (needed with --native).")
    p.add_argument("--exclude_seizures", action="store_true",
                   help="Corpus mode: drop recordings with a seizure annotation (matches training).")
    p.add_argument("--min_duration_min", type=float, default=None,
                   help="Corpus mode: drop recordings shorter than this many minutes.")
    p.add_argument("--interpolate", action="store_true",
                   help="Compare against the interpolated-repair sidecars (-psd-interp).")
    p.add_argument("--aas", action="store_true",
                   help="Compare against the AAS-repair sidecars (-psd-aas).")
    p.add_argument("--aas_fmax", type=float, default=2.5,
                   help="Cardiac-band cutoff (Hz) used to classify AAS-affected recordings (default 2.5).")
    p.add_argument("--by_class", action="store_true",
                   help="Split each curve into epilepsy vs no-epilepsy (color = class; dashed = raw, "
                   "solid = repaired).")
    p.add_argument("--bands", action="store_true",
                   help="Segment the spectrum into one row per canonical EEG band (delta/theta/alpha/beta/"
                   "gamma), each on its own y-scale.")
    p.add_argument("--band_edges", type=float, nargs="+", default=None,
                   help="Custom frequency segment edges (Hz), e.g. --band_edges 0.5 4 8 13 30 80. Implies "
                   "--bands and overrides the canonical bands.")
    p.add_argument("--bipolar", action="store_true", help="Read the bipolar sidecars for both raw and repaired.")
    p.add_argument("--notch_freqs", type=float, nargs="+", default=None, help="Read the notched sidecars.")
    p.add_argument("--native", action="store_true", help="Read the native-rate sidecars (requires --sfreq).")
    p.add_argument("--fmax", type=float, default=None, help="Max frequency to plot (Hz); default the full grid.")
    p.add_argument("--out", default=None, help="Output PNG (default: diagnostics/psd/psd_repair_corpus<tag>.png).")
    args = p.parse_args()

    if not (args.interpolate or args.aas):
        raise SystemExit("Nothing to compare: pass --interpolate and/or --aas (the repaired side).")
    if args.native and args.sfreq is None:
        raise SystemExit("--native requires --sfreq <rate>: native-rate PSDs share a grid only within one rate.")

    raw_suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native)
    rep_suffix = _psd_suffix(args.bipolar, args.notch_freqs, args.native, args.interpolate, args.aas)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = _records(args)
    ref_freqs = None
    # Per-group (all / affected), per-class running sums of the channel-mean PSD (linear).
    acc = {g: {cls: {"raw": None, "rep": None, "n": 0} for cls in CLASSES} for g in ("all", "aff")}
    n_missing = 0
    for edf, _subj, cls in records:
        raw = _channel_mean_psd(edf, raw_suffix)
        rep = _channel_mean_psd(edf, rep_suffix)
        if raw is None or rep is None:
            n_missing += 1
            continue
        f_raw, cm_raw = raw
        _f_rep, cm_rep = rep
        if ref_freqs is None:
            ref_freqs = f_raw
        # Skip recordings whose grid differs (e.g. a stray native rate) so the mean stays coherent.
        if cm_raw.shape != ref_freqs.shape or cm_rep.shape != ref_freqs.shape:
            n_missing += 1
            continue
        groups = ["all"]
        if _affected(edf, args.interpolate, args.aas, args.aas_fmax):
            groups.append("aff")
        for g in groups:
            b = acc[g][cls]
            b["raw"] = cm_raw if b["raw"] is None else b["raw"] + cm_raw
            b["rep"] = cm_rep if b["rep"] is None else b["rep"] + cm_rep
            b["n"] += 1
    if ref_freqs is None:
        raise SystemExit(f"No matching sidecars found (raw {raw_suffix} / repaired {rep_suffix}); "
                         f"run precompute_psd.py with and without the repair flags first.")

    top = float(args.fmax) if args.fmax is not None else float(ref_freqs.max())
    bands = _build_bands(args.bands or bool(args.band_edges), args.band_edges, top)
    repairs = "+".join(r for r, on in (("interp", args.interpolate), ("aas", args.aas)) if on)
    n_all = sum(acc["all"][c]["n"] for c in CLASSES)
    n_aff = sum(acc["aff"][c]["n"] for c in CLASSES)
    groups_meta = [("all", f"ALL (n={n_all})"), ("aff", f"AFFECTED by {repairs} (n={n_aff})")]

    nrows, ncols = len(bands), len(groups_meta)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 2.9 * nrows + 0.6), squeeze=False)
    for r, (blabel, lo, hi) in enumerate(bands):
        bmask = (ref_freqs >= lo) & (ref_freqs <= hi)
        fx = ref_freqs[bmask]
        for c, (g, gtitle) in enumerate(groups_meta):
            ax = axes[r][c]
            first_cell = (r == 0 and c == 0)
            if sum(acc[g][cls]["n"] for cls in CLASSES) == 0:
                if r == 0:
                    ax.text(0.5, 0.5, "no recordings in this group", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="0.5")
            elif args.by_class:
                for cls in CLASSES:
                    b = acc[g][cls]
                    if b["n"] == 0:
                        continue
                    raw_db = 10.0 * np.log10(b["raw"] / b["n"] + _EPS)[bmask]
                    rep_db = 10.0 * np.log10(b["rep"] / b["n"] + _EPS)[bmask]
                    ax.plot(fx, raw_db, color=CLASS_COLORS[cls], ls="--", lw=1.1, alpha=0.75,
                            label=f"{CLASS_NAMES[cls]} raw (n={b['n']})" if first_cell else None)
                    ax.plot(fx, rep_db, color=CLASS_COLORS[cls], ls="-", lw=1.5,
                            label=f"{CLASS_NAMES[cls]} repaired" if first_cell else None)
            else:
                raw, rep, n = _group_total(acc[g])
                ax.plot(fx, 10.0 * np.log10(raw / n + _EPS)[bmask], color="0.45", lw=1.4,
                        label="raw" if first_cell else None)
                ax.plot(fx, 10.0 * np.log10(rep / n + _EPS)[bmask], color="tab:blue", lw=1.5,
                        label=f"repaired ({repairs})" if first_cell else None)
            ax.set_xlim(lo, hi)
            ax.grid(True, alpha=0.25)
            if r == 0:
                ax.set_title(gtitle, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{blabel}\nPSD (dB)", fontsize=8)
    axes[0][0].legend(fontsize=7, loc="best")

    montage = "bipolar" if args.bipolar else "referential"
    notch = f", notch {[int(round(f)) for f in args.notch_freqs]}" if args.notch_freqs else ""
    rate = f", native {args.sfreq:g} Hz" if args.native else ""
    fig.suptitle(f"Corpus-mean PSD before vs after repair ({montage}{notch}{rate})  |  "
                 f"channel-averaged, mean over recordings  |  {n_missing} skipped (missing/grid)",
                 fontsize=11)
    fig.supxlabel("frequency (Hz)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    tag = rep_suffix.replace("-psd", "").replace(".npz", "")
    if args.by_class:
        tag += "-byclass"
    if bands and len(bands) > 1:
        tag += "-bands"
    out = Path(args.out) if args.out else root / "diagnostics" / "psd" / f"psd_repair_corpus{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}  (all n={n_all}, affected n={n_aff}, skipped {n_missing})")


if __name__ == "__main__":
    main()
