#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Reconstruction baselines (documentation/separable_reconstruction_baselines_protocol.md): does the R=1 code of the
#* five pass-1 separable dictionaries beat simple codes on held-out patients, in total and above 4 Hz? One root holds
#* one read. Run inside the HPC container from code/, on a GPU node:
#*   READOUT_ROOT=<pass 1 root> M=<40 s audit> SMOKE_DIR=<its cache dir> nohup bash run_reconstruction_baselines.sh > reconstruction_baselines.log 2>&1 &
#* It runs the tests, then fit and score for every fold, then read (reconstruction_decision.json). A rerun with the
#* same ROOT skips finished folds. STAGE=read repeats only the read, on CPU.
#* Environment: ROOT, READOUT_ROOT, STAGE=all|read, M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,9p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/reconstruction_baselines/read_$(date +%Y%m%d_%H%M%S)}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
[[ "$STAGE" =~ ^(all|read)$ ]] || { echo "ERROR: STAGE must be all or read" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ -d "$READOUT_ROOT/dictionaries" && -f "$READOUT_ROOT/folds.json" ]] || {
  echo "ERROR: $READOUT_ROOT holds no readout dictionaries and folds.json" >&2; exit 2; }
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
CORPUS_ARGS=()
if [[ "$STAGE" == all ]]; then
  M="${M:?Set M to the 40 s offset audit directory}"
  DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
  if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
    [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
    SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
  fi
  for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
  done
  CORPUS_ARGS=(--manifests "$(cd -- "$M" && pwd)" --data-dir "$(cd -- "$DATA_DIR" && pwd)"
    --signal-cache-dir "$(cd -- "$SIGNAL_CACHE_DIR" && pwd)" --num-workers "$NUM_WORKERS")
fi
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE READOUT_ROOT=$READOUT_ROOT NUM_WORKERS=$NUM_WORKERS ${M:+M=$M} ${SIGNAL_CACHE_DIR:+SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_reconstruction_baselines_core.py tests/test_reconstruction_baselines.py 2>&1 | tee -a "$ROOT/tests.log"

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/reconstruction_baselines.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

ROOTS=(--root "$ROOT" --readout-root "$READOUT_ROOT")
if [[ "$STAGE" == all ]]; then
  for fold in 0 1 2 3 4; do
    step fit "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold"
    step score "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold"
  done
fi
step read "${ROOTS[@]}"
cat "$ROOT/reconstruction_decision.json"
