#!/bin/bash
# Window-length ablation on RAW EEG: how does the input window length affect
# subject-level epilepsy classification? Run from inside the container, from code/.
#
# Design (fixed window budget): max_windows_per_subject is held CONSTANT across all
# lengths, so every config gets the same NUMBER of windows per subject; longer
# windows therefore carry more total signal. This answers "for a fixed window
# budget, what length works best?". Only the window length changes between configs.
#
# Every run uses:
#   - signal_mode=raw             (this study is raw-only; no ICA prerequisites)
#   - lazy loading                (O(batch) RAM, so the larger window counts fit)
#   - model.class_weight=balanced (window-level class imbalance fix)
#   - the default 60/20/20 subject split (test ~40 subjects, less noisy)
# No require_keep_labels and no notch here, to isolate the window-length effect on
# the plain raw signal (add either via EXTRA if you want, see below).
#
# Each run writes its artifacts under benchmark_winlen/runs/<name>/ (via the
# output_dir override), so the aggregation reads ONLY this study's runs and is not
# polluted by stale runs in logs/train/runs/.
#
# Grid is LENGTHS x SEEDS runs. For a faster pass, lower the HYDRA size uniformly:
#   EXTRA='feature.n_groups=32' ./execute_window_length.sh

set -uf -o pipefail            # -f: keep Hydra list args literal (no globbing)
cd "$(dirname "$0")"           # run from code/

SEEDS=(0 1 2 3 4)
WINLENS=(0.5 1 2 5)            # window length in MINUTES (fractional ok: 0.5 = 30 s)
CAP=20                         # fixed per-subject window budget (held constant)
OUT_DIR="logs/train/window_length_notch"     # comparison CSVs + per-run artifacts land here
EXTRA="${EXTRA:-}"             # optional uniform overrides, e.g. EXTRA='feature.n_groups=32'

# Overrides applied to every run (raw, balanced, lazy, fixed cap).
COMMON="data.lazy_loading=true model.class_weight=balanced data.signal_mode=raw"

total=$(( ${#WINLENS[@]} * ${#SEEDS[@]} ))
i=0
for wl in "${WINLENS[@]}"; do
  for s in "${SEEDS[@]}"; do
    i=$((i + 1))
    name="winlen${wl}min_seed${s}"
    echo "=== [$i/$total] window=${wl}min  cap=${CAP}  seed=${s} ==="
    # $EXTRA last so a feature.* value override follows any feature= group select.
    python src/train.py $COMMON \
      data.window_len_min="$wl" data.max_windows_per_subject="$CAP" data.seed="$s" \
      output_dir="${OUT_DIR}/runs/${name}" $EXTRA \
      || echo "!!! FAILED: window=${wl}min seed=${s} (continuing)"
  done
done

# Seed-aggregated comparison (per-config mean +/- std over seeds). Reads ONLY this
# study's runs. Each window length is a separate config row (distinguished by the
# data.window_len_min override in overrides_noseed).
python src/aggregate_performance.py --runs_root "${OUT_DIR}/runs" --split test --level subject \
  --out "${OUT_DIR}/performance_comparison.csv"
echo "Comparison tables written to ${OUT_DIR}/"

# python src/aggregate_performance.py --runs_root logs/train/window_length_notch/runs --split test --level subject --out logs/train/window_length_notch/performance_comparison.csv