#!/bin/bash
set -uf -o pipefail            # -f: keep Hydra list args like [brain] literal (no globbing)
for s in 0 1 2 3 4; do
  python src/train.py data.signal_mode=raw feature.track_counts=true feature.random_state=$s \
    output_dir=logs/train/runs_clf_kernels_seed/$s
done