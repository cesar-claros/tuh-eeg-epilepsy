#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* ShapeConv SAE, Phase 1 follow-up defined after the lever comparison of 2026-09-18:
#* a fixed extraction factor did not transfer across SNR (amp_min_relative: FA 1.66 ->
#* 0.09 per channel-minute at 8 sigma, but recall 0.706 at 4 sigma), so thresholds are
#* now calibrated to a false-alarm rate on an event-free draw. This script runs the
#* band-pass and hard conditions with diff_ar (AR order 8 and 16) at two target rates,
#* same seed, strict on the event gate. Read as characterization.
#*
#* Run inside the HPC container, from the code/ repo root (about ten minutes):
#*   nohup bash run_sae_calibration.sh > sae_calibration.log 2>&1 &
#*----------------------------------------------------------------------------*
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-tests/outputs/sae_calibration_$(date +%Y%m%d)}"
SEED="${SEED:-4000}"

cd "$(dirname "$0")"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "$OUT_ROOT/sha.txt"
git diff > "$OUT_ROOT/dirty.diff"
failures="$OUT_ROOT/failures.txt"
: > "$failures"
python tests/test_shapeconv_sae.py > "$OUT_ROOT/unit.log" 2>&1

conditions=(
  "band_pass__diff_ar8_fa0.1|--band-pass --pre-emphasis diff_ar --ar-order 8 --calibrate-fa 0.1"
  "band_pass__diff_ar8_fa1|--band-pass --pre-emphasis diff_ar --ar-order 8 --calibrate-fa 1.0"
  "hard__diff_ar8_fa0.1|--amplitude 4 --band-pass --artifacts --pre-emphasis diff_ar --ar-order 8 --calibrate-fa 0.1"
  "hard__diff_ar8_fa1|--amplitude 4 --band-pass --artifacts --pre-emphasis diff_ar --ar-order 8 --calibrate-fa 1.0"
  "hard__diff_ar16_fa0.1|--amplitude 4 --band-pass --artifacts --pre-emphasis diff_ar --ar-order 16 --calibrate-fa 0.1"
  "hard__diff_ar16|--amplitude 4 --band-pass --artifacts --pre-emphasis diff_ar --ar-order 16"
  "hard__diff_fa0.1|--amplitude 4 --band-pass --artifacts --pre-emphasis diff --calibrate-fa 0.1"
)

for entry in "${conditions[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  run="$OUT_ROOT/${name}_${SEED}"
  mkdir -p "$run"
  # shellcheck disable=SC2086
  if ! python tests/stage7_shapeconv_sae.py --seed "$SEED" --amp-min 4 --strict \
       --out-dir "$run" $flags > "$run/stdout.log" 2>&1; then
    echo "STRICT FAILURE $name seed=$SEED" | tee -a "$failures"
  fi
done

summary="$OUT_ROOT/summary.txt"
: > "$summary"
for entry in "${conditions[@]}"; do
  name="${entry%%|*}"
  log="$OUT_ROOT/${name}_${SEED}/stdout.log"
  {
    echo "== $name"
    grep -E "stress|deterministic failures|gate below tolerance|calibrated thresholds|AR whitening coefficients" "$log" || true
    grep -E "atom / offset / sign|event precision|event recall|recall_isolated|recall_close|false_alarms|val response std" "$log" || true
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
