"""Offline precomputation of ICA solutions, IC labels, and IC dipoles.

These passes write sibling files next to each recording's ``.edf`` and are a
prerequisite for the ICA-based loading modes (``data.signal_mode=ica_clean`` and
``brain_ic``). The training flow does NOT trigger them; run this script once.

Passes:

- ``labels``  -> ``compute_ica_labels``: writes ``-ica.fif``, ``-ica_labels.csv``
  and ``_ica.edf``. Required for ``ica_clean`` and ``brain_ic`` to work at all.
- ``dipoles`` -> ``compute_ic_dipoles``: writes ``-ica_dipoles.csv`` (one dipole
  per IC). Requires the ``.fif`` first; enables dipole-based region assignment in
  ``brain_ic``. Optional (``brain_ic`` falls back to the electrode heuristic).

Both passes run on CPU and are idempotent: a file whose outputs already exist is
skipped, so the script is safe to re-run and only fills in what is missing.

Examples
--------
From the ``code/`` directory inside the container::

    # fit and cache one dipole per IC (8 workers)
    python src/precompute_ica.py --steps dipoles --n_jobs 8

    # run both passes, pointing at the corpus on scratch
    python src/precompute_ica.py --steps both --n_jobs 8 --data_dir /scratch/.../data
"""

from __future__ import annotations

import argparse

import rootutils

root = rootutils.setup_root(__file__, pythonpath=True)

from src.data.components.tuh_eeg_epilepsy import TUHEEGEpilepsy  # noqa: E402


def main() -> None:
    """Parse CLI arguments and run the requested precomputation passes."""
    parser = argparse.ArgumentParser(
        description="Precompute ICA labels and/or IC dipoles for the TUH EEG corpus.",
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
        "--steps",
        choices=["labels", "dipoles", "both"],
        default="both",
        help="Which passes to run: ICA labels, IC dipoles, or both (default: both).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel workers (joblib) for each pass (default: 1).",
    )
    args = parser.parse_args()

    tuh = TUHEEGEpilepsy(data_dir=args.data_dir, version=args.version)

    if args.steps in ("labels", "both"):
        tuh.compute_ica_labels(n_jobs=args.n_jobs)
    if args.steps in ("dipoles", "both"):
        tuh.compute_ic_dipoles(n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
