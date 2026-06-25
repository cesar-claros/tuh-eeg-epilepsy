"""Offline precomputation of per-channel power spectral densities (PSD).

Writes one ``-psd.npz`` next to each recording's ``.edf`` (per-channel Welch PSD on
a common, resampled frequency grid; the 60 Hz line is kept). The PSD is
split-independent, so it is computed once and reused for every train/test split; the
training flow does NOT trigger it. Idempotent: recordings whose ``.npz`` already
exists are skipped, so the script is safe to re-run. Far cheaper than the ICA
precompute (FFT only, no model fitting). Plot the saved PSDs with ``src/plot_psd.py``.

Examples
--------
From the ``code/`` directory inside the container::

    python src/precompute_psd.py --n_jobs 8
    python src/precompute_psd.py --n_jobs 8 --bipolar              # TCP bipolar montage
    python src/precompute_psd.py --n_jobs 8 --notch_freqs 60 120   # notch before the PSD
    python src/precompute_psd.py --n_jobs 8 --target_sfreq 256 --win_sec 4 \
        --data_dir /scratch/.../data

The sidecar name encodes the montage / filtering so variants coexist: ``-psd.npz``
(referential, no notch), ``-psd-bipolar.npz``, ``-psd-notch-60-120.npz``,
``-psd-bipolar-notch-60-120.npz``. Plot with ``src/plot_psd.py`` passing the matching
``--bipolar`` / ``--notch_freqs`` flags.
"""

from __future__ import annotations

import argparse

import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402


def main() -> None:
    """Parse CLI arguments and run the PSD precomputation pass."""
    parser = argparse.ArgumentParser(
        description="Precompute per-channel PSD for the TUH EEG corpus.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(root / "data"),
        help="Parent directory of the version folder (default: <PROJECT_ROOT>/data).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v3.0.0",
        help="Corpus version subfolder (default: v3.0.0).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers (joblib) for the pass (default: 1).",
    )
    parser.add_argument(
        "--target_sfreq",
        type=float,
        default=256.0,
        help="Resample rate (Hz) so every recording shares one frequency grid.",
    )
    parser.add_argument(
        "--win_sec",
        type=float,
        default=4.0,
        help="Welch segment length in seconds (nperseg = win_sec * target_sfreq).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N recordings (for debugging a single file).",
    )
    parser.add_argument(
        "--bipolar",
        action="store_true",
        help="Re-reference to the TCP bipolar montage; writes -psd-bipolar.npz "
        "sidecars (independent of the referential -psd.npz). Plot with "
        "plot_psd.py --bipolar.",
    )
    parser.add_argument(
        "--notch_freqs",
        type=float,
        nargs="+",
        default=None,
        help="Notch frequencies (Hz) applied before the PSD, e.g. --notch_freqs 60 120. "
        "Writes a -notch-... sidecar; plot with plot_psd.py --notch_freqs 60 120. "
        "Default: no notch (line noise kept).",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Do NOT resample: compute each PSD at the recording's native rate (shows "
        "its true bandwidth up to its own Nyquist). Writes -psd-native.npz sidecars; "
        "grids match only WITHIN a native rate, so plot per-rate (plot_psd_subjects.py "
        "--native --sfreq <rate>). Overrides --target_sfreq.",
    )
    args = parser.parse_args()

    recording_ids = list(range(args.limit)) if args.limit is not None else None
    tuh = TUHEEGEpilepsy(
        data_dir=args.data_dir, version=args.version, recording_ids=recording_ids
    )
    tuh.compute_psd(
        n_jobs=args.n_jobs,
        target_sfreq=None if args.native else args.target_sfreq,
        win_sec=args.win_sec,
        bipolar=args.bipolar, notch_freqs=args.notch_freqs,
    )


if __name__ == "__main__":
    main()
