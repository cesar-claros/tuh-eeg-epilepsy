#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Band-limited reconstruction (documentation/separable_band_limited_reconstruction_protocol.md): once the rows are
#* band-limited to 1-30 Hz, does a trained separable dictionary at R=1 or R=4 beat simple codes on held-out patients
#* in 4-30 Hz? One root holds one read. Run inside the HPC container from code/, on a GPU node:
#*   READOUT_ROOT=<pass 1 root> M=<40 s audit> SMOKE_DIR=<its cache dir> nohup bash run_band_reconstruction.sh > band_reconstruction.log 2>&1 &
#* It runs the tests, then for every fold the four dictionaries (trained and untrained, R=1 and R=4), the PCA fit and
#* the score, then the read (band_decision.json). A rerun with the same ROOT skips finished steps. STAGE=read repeats
#* only the read, on CPU.
#* Environment: ROOT, READOUT_ROOT, STAGE=all|read, M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,10p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/band_reconstruction/read_$(date +%Y%m%d_%H%M%S)}"
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic training on CUDA needs it, as in run_frozen_readout.sh
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_band_reconstruction_core.py tests/test_band_reconstruction.py \
  tests/test_reconstruction_baselines_core.py tests/test_reconstruction_baselines.py 2>&1 | tee -a "$ROOT/tests.log"

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/band_reconstruction.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

ROOTS=(--root "$ROOT" --readout-root "$READOUT_ROOT")
if [[ "$STAGE" == all ]]; then
  for fold in 0 1 2 3 4; do
    for r in 1 4; do
      step fit-dictionary "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --top-r "$r" --arm trained
      step fit-dictionary "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --top-r "$r" --arm untrained
    done
    step fit "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold"
    step score "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold"
  done
fi
step read "${ROOTS[@]}"
cat "$ROOT/band_decision.json"
