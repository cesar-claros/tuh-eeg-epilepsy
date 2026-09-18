#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* ShapeConv SAE, Phase 1 of documentation/shapeconv_sae_experimental_proposal.md:
#* fresh-seed synthetic replication (no corpus, CPU).
#*
#* Archives the code state (commit SHA + uncommitted diff), the library versions and
#* the thread count, runs the unit invariants once, then runs the synthetic stage 7
#* on three UNEXAMINED seed triplets (--seed s uses s, s+1, s+2 for train / val /
#* test) with pre-emphasis 'diff' under --strict and 'none' on the same draws as a
#* control, and once with the deployment geometry (K=64, L=80). A strict failure is
#* recorded in $OUT_ROOT/failures.txt and the loop continues; it is evidence about
#* reliability, not permission to lower a tolerance.
#*
#* Run inside the HPC container, from the code/ repo root:
#*   bash run_sae_phase1.sh
#* Override inline, e.g.:  SEEDS="1000 2000" OUT_ROOT=tests/outputs/sae_try bash run_sae_phase1.sh
#*----------------------------------------------------------------------------*
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-tests/outputs/sae_replication_$(date +%Y%m%d)}"
SEEDS="${SEEDS:-1000 2000 3000}"
MODES="${MODES:-diff none}"
AMP_MIN="${AMP_MIN:-4}"
GEOMETRY_SEED="${GEOMETRY_SEED:-1000}"   # one run with K=64, L=80 (the deployment defaults)

cd "$(dirname "$0")"
mkdir -p "$OUT_ROOT"
failures="$OUT_ROOT/failures.txt"
: > "$failures"

# ---- archive the exact code state and environment -------------------------------
git rev-parse HEAD > "$OUT_ROOT/sha.txt"
git diff > "$OUT_ROOT/dirty.diff"            # empty file = clean working tree
git status --short > "$OUT_ROOT/status.txt"
python - > "$OUT_ROOT/env.txt" <<'EOF'
import platform, sklearn, torch, numpy
print(f"python {platform.python_version()}")
print(f"torch {torch.__version__} threads {torch.get_num_threads()} cuda {torch.cuda.is_available()}")
print(f"numpy {numpy.__version__} sklearn {sklearn.__version__}")
print(platform.platform())
EOF
env | grep -E "^(OMP|MKL|TORCH|CUDA)_" | sort > "$OUT_ROOT/threads.txt" || true
echo "OUT_ROOT=$OUT_ROOT SEEDS=$SEEDS MODES=$MODES AMP_MIN=$AMP_MIN" | tee "$OUT_ROOT/settings.txt"

# ---- Phase 0 invariants, once per campaign ---------------------------------------
python tests/test_shapeconv_sae.py > "$OUT_ROOT/unit.log" 2>&1

# ---- fresh seed triplets: diff (strict) and none (control) -----------------------
for seed in $SEEDS; do
  for mode in $MODES; do
    run="$OUT_ROOT/${mode}_${seed}"
    mkdir -p "$run"
    extra=()
    [ "$mode" = diff ] && extra=(--strict)
    if ! python tests/stage7_shapeconv_sae.py --seed "$seed" --pre-emphasis "$mode" --amp-min "$AMP_MIN" \
         --out-dir "$run" ${extra[@]+"${extra[@]}"} > "$run/stdout.log" 2>&1; then
      echo "STRICT FAILURE mode=$mode seed=$seed" | tee -a "$failures"
    fi
  done
done

# ---- the deployment geometry, once -----------------------------------------------
run="$OUT_ROOT/diff_${GEOMETRY_SEED}_k64_l80"
mkdir -p "$run"
if ! python tests/stage7_shapeconv_sae.py --seed "$GEOMETRY_SEED" --pre-emphasis diff --amp-min "$AMP_MIN" \
     --n-atoms 64 --atom-len 80 --strict --out-dir "$run" > "$run/stdout.log" 2>&1; then
  echo "STRICT FAILURE geometry k64_l80 seed=$GEOMETRY_SEED" | tee -a "$failures"
fi

# ---- summary: one line per run from the RESULT and event blocks --------------------
summary="$OUT_ROOT/summary.txt"
: > "$summary"
for log in "$OUT_ROOT"/*/stdout.log; do
  name=$(basename "$(dirname "$log")")
  {
    echo "== $name"
    grep -E "deterministic failures|recovery below tolerance|template recovered|count AUROC|peak \|a\| AUROC" "$log" || true
    grep -E "atom / offset / sign|event precision|event recall|timing_error|false_alarms" "$log" || true
    grep "thresh/response" "$log" | tail -1 || true
  } >> "$summary"
done
cat "$summary"
echo
if [ -s "$failures" ]; then
  echo "Strict failures recorded in $failures:"; cat "$failures"
else
  echo "No strict failures. Outputs under $OUT_ROOT"
fi
