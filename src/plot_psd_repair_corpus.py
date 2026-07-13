"""Overlay the corpus-mean PSD before vs after repair (interpolation and/or AAS).

Compares the raw per-recording PSDs (``-psd.npz``) against the repaired ones
(``-psd-interp.npz`` / ``-psd-aas.npz`` / ``-psd-interp-aas.npz`` from
``precompute_psd.py --interpolate --aas``) to show the aggregate improvement across the
whole corpus. For each recording it channel-averages both PSDs on the shared grid, then
means them across recordings. Two panels are drawn: ALL recordings, and the AFFECTED
subset (recordings a repair actually touches, i.e. with flagged bad channels when
``--interpolate`` and/or a flagged cardiac-band comb when ``--aas``), where the change is
largest and the harmonic comb should flatten out.

Run on the machine that holds the corpus and both sets of ``-psd*.npz`` sidecars, after
``precompute_badchannels.py`` and both ``precompute_psd.py`` passes (raw and repaired).

Examples
--------
::

    # after: precompute_psd.py --n_jobs 8   AND   precompute_psd.py --n_jobs 8 --interpolate --aas
    python src/plot_psd_repair_corpus.py --interpolate --aas
    python src/plot_psd_repair_corpus.py --aas --bipolar --exclude_seizures --min_duration_min 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)


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
    # Running sums of the channel-mean PSD (linear), for all recordings and the affected subset.
    sums = {"all": {"raw": None, "rep": None, "n": 0}, "aff": {"raw": None, "rep": None, "n": 0}}
    n_missing = 0
    for edf, _subj, _cls in records:
        raw = _channel_mean_psd(edf, raw_suffix)
        rep = _channel_mean_psd(edf, rep_suffix)
        if raw is None or rep is None:
            n_missing += 1
            continue
        f_raw, cm_raw = raw
        f_rep, cm_rep = rep
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
            sums[g]["raw"] = cm_raw if sums[g]["raw"] is None else sums[g]["raw"] + cm_raw
            sums[g]["rep"] = cm_rep if sums[g]["rep"] is None else sums[g]["rep"] + cm_rep
            sums[g]["n"] += 1
    if ref_freqs is None:
        raise SystemExit(f"No matching sidecars found (raw {raw_suffix} / repaired {rep_suffix}); "
                         f"run precompute_psd.py with and without the repair flags first.")

    fmask = np.ones_like(ref_freqs, dtype=bool) if args.fmax is None else (ref_freqs <= args.fmax)
    fx = ref_freqs[fmask]
    repairs = "+".join(r for r, on in (("interp", args.interpolate), ("aas", args.aas)) if on)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True, sharey=True)
    panels = [("all", f"ALL recordings (n={sums['all']['n']})"),
              ("aff", f"AFFECTED by {repairs} (n={sums['aff']['n']})")]
    for ax, (g, title) in zip(axes, panels):
        n = sums[g]["n"]
        if n == 0:
            ax.set_title(title + "  (none)", fontsize=10)
            ax.text(0.5, 0.5, "no recordings in this group", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.5")
            continue
        raw_db = 10.0 * np.log10(sums[g]["raw"] / n + 1e-20)[fmask]
        rep_db = 10.0 * np.log10(sums[g]["rep"] / n + 1e-20)[fmask]
        ax.plot(fx, raw_db, color="0.45", lw=1.4, label="raw")
        ax.plot(fx, rep_db, color="tab:blue", lw=1.4, label=f"repaired ({repairs})")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    montage = "bipolar" if args.bipolar else "referential"
    notch = f", notch {[int(round(f)) for f in args.notch_freqs]}" if args.notch_freqs else ""
    fig.suptitle(f"Corpus-mean PSD before vs after repair ({montage}{notch})  |  "
                 f"channel-averaged, mean over recordings  |  {n_missing} skipped (missing/grid)",
                 fontsize=11)
    fig.supxlabel("frequency (Hz)")
    fig.supylabel("PSD (dB)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    tag = rep_suffix.replace("-psd", "").replace(".npz", "")
    out = Path(args.out) if args.out else root / "diagnostics" / "psd" / f"psd_repair_corpus{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}  (all n={sums['all']['n']}, affected n={sums['aff']['n']}, skipped {n_missing})")


if __name__ == "__main__":
    main()
