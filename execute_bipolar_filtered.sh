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
# 256 Hz resample and 60/20/20 subject split. Each ARM writes its runs under its own
# directory logs/train/<arm>/runs/ (via the output_dir override); the final step
# aggregates across all arms into one comparison table.

set -uf -o pipefail            # -f: keep Hydra list args like [1,100] literal (no globbing)
cd "$(dirname "$0")"           # run from code/

SEEDS=(0 1 2 3 4)
CAPS=(10 20 40)                # per-subject window budgets (comparable to the prior benchmark)
BASE="logs/train"             # each arm -> $BASE/<arm_name>/runs/...
EXTRA="${EXTRA:-}"             # optional uniform overrides, e.g. EXTRA='feature.n_groups=32'

# Overrides applied to every run. require_keep_labels restricts ALL arms to the same
# recording pool (those with an ICA solution and a kept brain/other IC), so every arm
# evaluates the SAME subjects: the seed-based split is over one shared pool, and the
# ica_clean arm no longer drops recordings at load time.
COMMON="data.lazy_loading=true model.class_weight=balanced data.require_keep_labels=[brain,other]"

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
  # "bipolar_filt|data.signal_mode=bipolar data.filter_freq=[1,100] data.notch_freqs=[60]"
  # "bipolar_nofilt|data.signal_mode=bipolar data.notch_freqs=[60]"
  "icaclean_bipolar|data.signal_mode=ica_clean data.ica_keep_labels=[brain,other] data.bipolar=true data.filter_freq=[1,100] data.notch_freqs=[60]"
  # "raw_filt|data.signal_mode=raw data.filter_freq=[1,100] data.notch_freqs=[60]"
)

total=$(( ${#ARMS[@]} * ${#CAPS[@]} * ${#SEEDS[@]} ))
i=0
for arm in "${ARMS[@]}"; do
  name="${arm%%|*}"; ov="${arm#*|}"
  out="${BASE}/${name}"        # this arm's output directory, e.g. logs/train/icaclean_bipolar
  for cap in "${CAPS[@]}"; do
    for s in "${SEEDS[@]}"; do
      i=$((i + 1))
      echo "=== [$i/$total] ${name}  cap=${cap}  seed=${s} ==="
      # $EXTRA last so a feature.* value override follows any feature= group select.
      python src/train.py $COMMON $ov \
        data.max_windows_per_subject="$cap" data.seed="$s" \
        output_dir="${out}/runs/cap${cap}_seed${s}" $EXTRA \
        || echo "!!! FAILED: ${name} cap=${cap} seed=${s} (continuing)"
    done
  done
  # Per-arm seed/cap aggregation -> this arm's own comparison table (reads only its runs).
  python src/aggregate_performance.py --runs_root "${out}/runs" --split test --level subject \
    --out "${out}/performance_comparison.csv"
  echo "--> ${out}/performance_comparison.csv"
done
echo "Done. Per-arm comparison tables: ${BASE}/<arm>/performance_comparison*.csv"
