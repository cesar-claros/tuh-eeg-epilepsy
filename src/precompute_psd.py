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
    python src/precompute_psd.py --n_jobs 8 --target_sfreq 250 --win_sec 4 \
        --data_dir /scratch/.../data
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
        default=250.0,
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
    args = parser.parse_args()

    recording_ids = list(range(args.limit)) if args.limit is not None else None
    tuh = TUHEEGEpilepsy(
        data_dir=args.data_dir, version=args.version, recording_ids=recording_ids
    )
    tuh.compute_psd(
        n_jobs=args.n_jobs, target_sfreq=args.target_sfreq, win_sec=args.win_sec
    )


if __name__ == "__main__":
    main()
