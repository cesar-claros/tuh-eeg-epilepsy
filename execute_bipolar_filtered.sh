#!/bin/bash
# Bipolar + band-pass + notch experiment. Pipeline per window (filters act in
# referential space BEFORE the bipolar re-reference, the EEG convention):
#   raw -> band-pass [1,100] Hz -> notch 60 Hz -> TCP bipolar montage -> HYDRA -> classifier.
# Run from inside the container, from code/.
#
# The band-pass [1,100] equalizes bandwidth across recordings of different native
# sfreq (the >~100 Hz content is a class confound; see plot_sfreq.py), and matches
# the IC-labels pipeline's band. The 120 Hz line harmonic is already removed by the
# 100 Hz low-pass, so notching only 60 Hz is sufficient.
#
# Every run: lazy loading (O(batch) RAM), balanced class weight, the default
# 256 Hz resample and 60/20/20 subject split. Each run writes under
# logs/train/bipolar_filtered/runs/<name>/ (output_dir override) so the aggregation
# reads ONLY this study's runs.

set -uf -o pipefail            # -f: keep Hydra list args like [1,100] literal (no globbing)
cd "$(dirname "$0")"           # run from code/

SEEDS=(0 1 2 3 4)
CAPS=(10 20 40)                # per-subject window budgets (comparable to the prior benchmark)
OUT_DIR="logs/train/bipolar_filtered"
EXTRA="${EXTRA:-}"             # optional uniform overrides, e.g. EXTRA='feature.n_groups=32'

# Overrides applied to every run.
COMMON="data.lazy_loading=true model.class_weight=balanced"

# Arms: "name|extra overrides". The first is the requested config; the others are
# comparisons for the sfreq-confound question (enable by uncommenting):
#   raw_filt        : same filtering, referential montage (isolates bipolar vs raw).
#   bipolar_nofilt  : bipolar without the band-pass (does dropping 1-100 Hz change AUC?
#                     a drop means the >100 Hz acquisition band was carrying signal).
#   icaclean_bipolar: ICA reconstruction keeping only brain+other ICs, THEN bipolar
#                     (same filter/notch as bipolar_filt, so it isolates ICA cleaning).
#                     Requires the ICA precompute (precompute_ica.py); recordings
#                     without an -ica.fif are skipped.
ARMS=(
  "bipolar_filt|data.signal_mode=bipolar data.filter_freq=[1,100] data.notch_freqs=[60]"
  "bipolar_nofilt|data.signal_mode=bipolar data.notch_freqs=[60]"
  "icaclean_bipolar|data.signal_mode=ica_clean data.ica_keep_labels=[brain,other] data.bipolar=true data.filter_freq=[1,100] data.notch_freqs=[60]"
  # "raw_filt|data.signal_mode=raw data.filter_freq=[1,100] data.notch_freqs=[60]"
)

total=$(( ${#ARMS[@]} * ${#CAPS[@]} * ${#SEEDS[@]} ))
i=0
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; ov="${arm#*|}"
  for cap in "${CAPS[@]}"; do
    for s in "${SEEDS[@]}"; do
      i=$((i + 1))
      echo "=== [$i/$total] ${name}  cap=${cap}  seed=${s} ==="
      # $EXTRA last so a feature.* value override follows any feature= group select.
      python src/train.py $COMMON $ov \
        data.max_windows_per_subject="$cap" data.seed="$s" \
        output_dir="${OUT_DIR}/runs/${name}_cap${cap}_seed${s}" $EXTRA \
        || echo "!!! FAILED: ${name} cap=${cap} seed=${s} (continuing)"
    done
  done
done

# Seed-aggregated comparison (per-config mean +/- std over seeds/caps). Reads ONLY
# this study's runs; each arm is a separate config row (distinguished by its overrides).
python src/aggregate_performance.py --runs_root "${OUT_DIR}/runs" --split test --level subject \
  --out "${OUT_DIR}/performance_comparison.csv"
echo "Comparison tables written to ${OUT_DIR}/"
