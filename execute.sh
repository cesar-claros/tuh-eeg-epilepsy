#!/bin/bash
# Benchmark: evaluate all EEG representations for subject-level epilepsy
# classification on an IDENTICAL window set, across per-subject window caps and
# seeds. Run from inside the container, from the code/ directory.
#
# Every run uses:
#   - lazy loading                (O(batch) RAM, so the larger caps fit)
#   - model.class_weight=balanced (window-level class imbalance fix)
#   - require_keep_labels=[brain] (keep only recordings with >=1 brain IC, applied
#                                  before windowing) -- this makes raw / brain_ic /
#                                  ic_bag train and evaluate on the SAME windows for
#                                  a given seed, so the comparison is fair.
#
# Prerequisites (offline, once): the IC modes need -ica.fif + -ica_labels.csv;
# brain_ic with dipoles also needs -ica_dipoles.csv. Generate with:
#   python src/precompute_ica.py --steps both --n_jobs 8
#
# Grid is ARMS x CAPS x SEEDS runs; trim any array to shrink it. For a fast ranking
# pass, lower the HYDRA size uniformly across all arms:
#   EXTRA='feature.n_groups=32' ./execute.sh

set -uf -o pipefail            # -f: keep Hydra list args like [brain] literal (no globbing)
cd "$(dirname "$0")"           # run from code/

SEEDS=(0 1 2 3 4)
CAPS=(10 20 40)
OUT_DIR="benchmark"            # comparison CSVs land here (created if missing)
EXTRA="${EXTRA:-}"             # optional uniform overrides, e.g. EXTRA='feature.n_groups=32'

# Overrides applied to every run (identical windows + balanced + lazy).
COMMON="data.lazy_loading=true model.class_weight=balanced data.require_keep_labels=[brain]"

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
      # $EXTRA after $ov so a feature.* value override follows the feature= group select.
      python src/train.py $COMMON $ov $EXTRA data.max_windows_per_subject="$cap" data.seed="$s" \
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
# ic_bag pooling operator, fixed cap=20 (does max-pool alone catch focal epilepsy?):
# for pool in "[mean]" "[max]" "[mean,max,std]"; do for s in "${SEEDS[@]}"; do
#   python src/train.py $COMMON data.signal_mode=ic_bag data.ica_keep_labels=[brain] \
#     feature=ic_bag_transformer feature.pool="$pool" \
#     data.max_windows_per_subject=20 data.seed="$s"
# done; done
#
# ic_bag IC set: brain-only vs brain+other, fixed cap=20:
# for keep in "[brain]" "[brain,other]"; do for s in "${SEEDS[@]}"; do
#   python src/train.py $COMMON data.signal_mode=ic_bag data.ica_keep_labels="$keep" \
#     feature=ic_bag_transformer data.max_windows_per_subject=20 data.seed="$s"
# done; done
