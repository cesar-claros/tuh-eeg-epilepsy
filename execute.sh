#!/bin/bash
# Benchmark: compare EEG representations for subject-level epilepsy classification
# across per-subject window caps and seeds. Every run uses lazy loading (O(batch)
# RAM, so the larger caps fit) and class_weight=balanced (the window-level class
# imbalance fix). Run from inside the container, from the code/ directory.
#
# Prerequisites (offline, once): brain_ic / ic_bag need -ica.fif + -ica_labels.csv;
# brain_ic with dipoles also needs -ica_dipoles.csv. Generate with:
#   python src/precompute_ica.py --steps both --n_jobs 8
#
# Grid below is ARMS x CAPS x SEEDS runs; trim any array to shrink it.

set -uf -o pipefail            # -f: keep Hydra list args like [brain] literal (no globbing)
cd "$(dirname "$0")"           # run from code/

SEEDS=(0 1 2 3 4)
CAPS=(10 20 40)
OUT_DIR="benchmark"            # comparison CSVs land here (created if missing)

# Overrides applied to every run.
COMMON="data.lazy_loading=true model.class_weight=balanced"

# Benchmark arms: "name|extra overrides" (one distinct representation each).
ARMS=(
  "raw|data.signal_mode=raw"
  "brain_ic_dipole|data.signal_mode=brain_ic data.ica_keep_labels=[brain]"
  "brain_ic_electrode|data.signal_mode=brain_ic data.ica_keep_labels=[brain] data.brain_ic_use_dipoles=false"
  "ic_bag|data.signal_mode=ic_bag data.ica_keep_labels=[brain] feature=ic_bag_transformer"
)

total=$(( ${#ARMS[@]} * ${#CAPS[@]} * ${#SEEDS[@]} ))
i=0
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; ov="${arm#*|}"
  for cap in "${CAPS[@]}"; do
    for s in "${SEEDS[@]}"; do
      i=$((i + 1))
      echo "=== [$i/$total] ${name}  cap=${cap}  seed=${s} ==="
      python src/train.py $COMMON $ov data.max_windows_per_subject="$cap" data.seed="$s" \
        || echo "!!! FAILED: ${name} cap=${cap} seed=${s} (continuing)"
    done
  done
done

# Seed-aggregated comparison (per-config mean +/- std over seeds). Writes
# $OUT_DIR/performance_comparison.csv plus the _test_subject and _by_config tables.
python src/aggregate_performance.py --runs_root logs/train/runs --split test --level subject \
  --out "$OUT_DIR/performance_comparison.csv"
echo "Comparison tables written to ${OUT_DIR}/"

# --- Optional ablations (uncomment to run) -----------------------------------
# ic_bag pooling operator, fixed cap=20:
# for pool in "[mean]" "[max]" "[mean,max,std]"; do for s in "${SEEDS[@]}"; do
#   python src/train.py $COMMON data.signal_mode=ic_bag data.ica_keep_labels=[brain] \
#     feature=ic_bag_transformer feature.pool="$pool" \
#     data.max_windows_per_subject=20 data.seed="$s"
# done; done
#
# class_weight A/B on raw (balanced vs unweighted), fixed cap=20:
# for cw in balanced null; do for s in "${SEEDS[@]}"; do
#   python src/train.py data.lazy_loading=true data.signal_mode=raw \
#     model.class_weight="$cw" data.max_windows_per_subject=20 data.seed="$s"
# done; done
