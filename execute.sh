#!/bin/bash
for cap in 10 20 40; do for s in 0 1 2 3 4; do
  python src/train.py data.signal_mode=raw data.max_windows_per_subject=$cap data.seed=$s data.lazy_loading=true
done; done
python src/aggregate_performance.py --runs_root logs/train/runs --split test --level subject

# python src/train.py data.signal_mode=raw data.seed=0
# python src/train.py data.signal_mode=raw data.seed=0 data.lazy_loading=true
