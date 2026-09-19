#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* ShapeConv SAE, Phase 1 stress tests (documentation/shapeconv_sae_experimental_proposal.md):
#* a small sequence of targeted synthetic conditions at one fresh seed, not a grid.
#* Each run is strict on the event gate (precision and recall > 0.9) except the
#* event-free condition, which only reports false alarms per channel-minute. Lower
#* SNR and artifact conditions characterize failure; they are not required to pass.
#*
#* Run inside the HPC container, from the code/ repo root (about ten minutes):
#*   nohup bash run_sae_stress.sh > sae_stress.log 2>&1 &
#* Override inline, e.g.:  SEED=7000 OUT_ROOT=tests/outputs/sae_stress_x bash run_sae_stress.sh
#*----------------------------------------------------------------------------*
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-tests/outputs/sae_stress_$(date +%Y%m%d)}"
SEED="${SEED:-4000}"
PRE="${PRE:-diff}"

cd "$(dirname "$0")"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "$OUT_ROOT/sha.txt"
git diff > "$OUT_ROOT/dirty.diff"
failures="$OUT_ROOT/failures.txt"
: > "$failures"

# name | extra flags
conditions=(
  "amp4|--amplitude 4 --strict"
  "amp2|--amplitude 2 --strict"
  "event_free|--event-free"
  "band_pass|--band-pass --strict"
  "second_morphology|--second-morphology --strict"
  "pair_gap40|--pair-gap 40 --strict"
  "artifacts|--artifacts --strict"
  "synchronous|--synchronous --strict"
  "amp4_band_pass_artifacts|--amplitude 4 --band-pass --artifacts --strict"
)

for entry in "${conditions[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  run="$OUT_ROOT/${PRE}_${SEED}_${name}"
  mkdir -p "$run"
  # shellcheck disable=SC2086
  if ! python tests/stage7_shapeconv_sae.py --seed "$SEED" --pre-emphasis "$PRE" --amp-min 4 \
       --out-dir "$run" $flags > "$run/stdout.log" 2>&1; then
    echo "STRICT FAILURE $name seed=$SEED" | tee -a "$failures"
  fi
done

summary="$OUT_ROOT/summary.txt"
: > "$summary"
for entry in "${conditions[@]}"; do
  name="${entry%%|*}"
  log="$OUT_ROOT/${PRE}_${SEED}_${name}/stdout.log"
  {
    echo "== $name"
    grep -E "stress|deterministic failures|gate below tolerance|template \|xcorr\||count AUROC|peak \|a\| AUROC" "$log" || true
    grep -E "atom / offset / sign|event precision|event recall|recall_isolated|recall_close|recall_by_template|timing_error|duplicates|unmatched_per|false_alarms|event reconstruction|gate  " "$log" || true
    grep "thresh/response" "$log" | tail -1 || true
  } >> "$summary"
done
cat "$summary"
echo
if [ -s "$failures" ]; then
  echo "Strict failures (characterize, do not lower tolerances):"; cat "$failures"
else
  echo "No strict failures. Outputs under $OUT_ROOT"
fi
