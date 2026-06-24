#!/bin/bash
set -uf -o pipefail            # -f: keep Hydra list args like [brain] literal (no globbing)
for s in 0; do
  python src/train.py data.signal_mode=raw feature.track_counts=true feature.random_state=$s \
    output_dir=logs/train/top_kernels/runs/$s model=logistic_regression_l1 data.notch_freqs=[60,120] \
    data.window_len_min=2 data.max_windows_per_subject=20
done
