#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Frozen dictionary feature readout (documentation/frozen_feature_readout_protocol.md): do separable peak
#* features beat spectral features and untrained atoms while staying close to HYDRA? One root holds one pass.
#* The order is fixed: tests, folds (no signal read), positive control (a failure stops the run), 5 trained and
#* 5 untrained dictionaries, features, heads, rule. Finished steps are skipped, so a rerun resumes.
#*
#* Run inside the HPC container from code/, on a GPU node:
#*   M=<40 s audit> SMOKE_DIR=<its cache_smoke_P512_* dir> nohup bash run_frozen_readout.sh > frozen_readout.log 2>&1 &
#* Environment: ROOT, PASS=1|2, M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4, ALL_SEGMENTS=1
#* (0 skips the descriptive all-segment reads), CONTROL_ROOT.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,11p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PASS="${PASS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ALL_SEGMENTS="${ALL_SEGMENTS:-1}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/frozen_readout/pass${PASS}_$(date +%Y%m%d_%H%M%S)}"
CONTROL_ROOT="${CONTROL_ROOT:-$SCRIPT_DIR/logs/separable_synthetic/gate1b_random_none_20261030_124956}"
[[ "$PASS" =~ ^[12]$ ]] || { echo "ERROR: PASS must be 1 or 2" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ "$ALL_SEGMENTS" =~ ^[01]$ ]] || { echo "ERROR: ALL_SEGMENTS must be 0 or 1" >&2; exit 2; }
M="${M:?Set M to the 40 s offset audit directory}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
  [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
  SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
fi
for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR" "$CONTROL_ROOT"; do
  [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
done
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP PASS=$PASS NUM_WORKERS=$NUM_WORKERS ALL_SEGMENTS=$ALL_SEGMENTS M=$M SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR DATA_DIR=$DATA_DIR" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_frozen_readout_core.py tests/test_frozen_readout.py 2>&1 | tee -a "$ROOT/tests.log"

DATA_ARGS=(--root "$ROOT" --manifests "$(cd -- "$M" && pwd)" --data-dir "$(cd -- "$DATA_DIR" && pwd)"
  --signal-cache-dir "$(cd -- "$SIGNAL_CACHE_DIR" && pwd)" --num-workers "$NUM_WORKERS")
step() {
  echo "$(date '+%F %T') start $*"
  python -u src/frozen_readout.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

step folds "${DATA_ARGS[@]}" --pass-index "$PASS"
if [[ ! -f "$ROOT/positive_control.json" ]]; then
  step control "${DATA_ARGS[@]}" --control-root "$CONTROL_ROOT"
fi
passed=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['passed'])" "$ROOT/positive_control.json")
if [[ "$passed" != True ]]; then
  echo "Positive control failed: fix the feature code before reading any classification result." >&2
  exit 1
fi
for fold in 0 1 2 3 4; do
  for arm in trained untrained; do
    step fit "${DATA_ARGS[@]}" --fold "$fold" --arm "$arm"
    if [[ ! -f "$ROOT/features/z_${arm}_f${fold}.json" ]]; then
      step extract "${DATA_ARGS[@]}" --fold "$fold" --dictionary "$arm" --read nine
    fi
  done
  if [[ "$ALL_SEGMENTS" == 1 && ! -f "$ROOT/features/S4-proj-all_f${fold}.npz" ]]; then
    step extract "${DATA_ARGS[@]}" --fold "$fold" --dictionary trained --read all
  fi
done
fixed=(Spectral HYDRA)
[[ "$ALL_SEGMENTS" == 1 ]] && fixed+=(Spectral-all HYDRA-all)
for arm in "${fixed[@]}"; do
  [[ -f "$ROOT/features/$arm.npz" ]] || step extract-fixed "${DATA_ARGS[@]}" --arm "$arm"
done
step heads --root "$ROOT"
step summarize --root "$ROOT"
cat "$ROOT/readout_decision.json"
