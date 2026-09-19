#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* ShapeConv SAE, Phase 1 follow-up defined after the stress sequence of 2026-09-18:
#* the corpus-matched band-pass broke the event gate (precision 0.842 alone; 0.404 /
#* 0.281 with 4-sigma events and artifacts) and a 40-sample pair gap halved recall.
#* This script compares the three levers on exactly those conditions, same seed,
#* event gate strict. Arms: diff (baseline, re-run), diff_ar (AR whitening after the
#* difference), diff + amp_min_relative, diff_ar + amp_min_relative; and for the pair
#* gap, diff with an NMS half-width of 16 samples. Read as characterization.
#*
#* Run inside the HPC container, from the code/ repo root (about fifteen minutes):
#*   nohup bash run_sae_levers.sh > sae_levers.log 2>&1 &
#*----------------------------------------------------------------------------*
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-tests/outputs/sae_levers_$(date +%Y%m%d)}"
SEED="${SEED:-4000}"

cd "$(dirname "$0")"
mkdir -p "$OUT_ROOT"
git rev-parse HEAD > "$OUT_ROOT/sha.txt"
git diff > "$OUT_ROOT/dirty.diff"
failures="$OUT_ROOT/failures.txt"
: > "$failures"
python tests/test_shapeconv_sae.py > "$OUT_ROOT/unit.log" 2>&1

# name | flags (all strict on the event gate)
conditions=(
  "band_pass__diff|--band-pass --pre-emphasis diff"
  "band_pass__diff_ar|--band-pass --pre-emphasis diff_ar"
  "band_pass__diff_rel|--band-pass --pre-emphasis diff --amp-min-relative"
  "band_pass__diff_ar_rel|--band-pass --pre-emphasis diff_ar --amp-min-relative"
  "hard__diff|--amplitude 4 --band-pass --artifacts --pre-emphasis diff"
  "hard__diff_ar|--amplitude 4 --band-pass --artifacts --pre-emphasis diff_ar"
  "hard__diff_rel|--amplitude 4 --band-pass --artifacts --pre-emphasis diff --amp-min-relative"
  "hard__diff_ar_rel|--amplitude 4 --band-pass --artifacts --pre-emphasis diff_ar --amp-min-relative"
  "pair40__diff|--pair-gap 40 --pre-emphasis diff"
  "pair40__diff_nms16|--pair-gap 40 --pre-emphasis diff --nms-half 16"
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
    grep -E "stress|deterministic failures|gate below tolerance|template \|xcorr\||AR whitening" "$log" || true
    grep -E "atom / offset / sign|event precision|event recall|recall_isolated|recall_close|duplicates|unmatched_per|false_alarms|val response std" "$log" || true
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
