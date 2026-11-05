#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Band-limited hyperparameter sweep (documentation/separable_band_sweep_protocol.md): at 2 amplitudes per second, which
#* settings of K, L, D and P give the best held-out EV in 4-30 Hz, how do they interact, and does the best beat the base?
#* Run inside the HPC container from code/, on a GPU node, in two stages on one root:
#*   STAGE=screen BAND_ROOT=<band-limited root> READOUT_ROOT=<pass 1 root> M=<40 s audit> SMOKE_DIR=<its cache dir> nohup bash run_band_sweep.sh > band_sweep_screen.log 2>&1 &
#*   STAGE=confirm ROOT=<the same root> BAND_ROOT=... READOUT_ROOT=... M=... SMOKE_DIR=... nohup bash run_band_sweep.sh > band_sweep_confirm.log 2>&1 &
#* screen: tests, the memory smoke, fits and scores on folds 0 and 1, then the selection (selection.json).
#* confirm: fits and scores of the selected setting and the base on folds 2 to 4, then the read (sweep_decision.json).
#* STAGE=select or STAGE=read repeats only that step, on CPU. A rerun with the same ROOT skips finished steps.
#* STAGE=map (documentation/separable_band_rkd_map_protocol.md) runs on the sweep root (ROOT=<the screen root>): tests,
#* the missing cells of the K, R, D and L grid on folds 0 and 1, then the map read (map.json).
#* Environment: ROOT, BAND_ROOT, READOUT_ROOT, STAGE, M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,13p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:?Set STAGE to screen, confirm, select, read or map}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/band_sweep/sweep_$(date +%Y%m%d_%H%M%S)}"
BAND_ROOT="${BAND_ROOT:?Set BAND_ROOT to the root of the band-limited run}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
[[ "$STAGE" =~ ^(screen|confirm|select|read|map)$ ]] || { echo "ERROR: STAGE must be screen, confirm, select, read or map" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ -f "$BAND_ROOT/band_decision.json" && -d "$BAND_ROOT/scores" ]] || { echo "ERROR: $BAND_ROOT is not a finished band-limited root" >&2; exit 2; }
[[ -f "$READOUT_ROOT/folds.json" ]] || { echo "ERROR: $READOUT_ROOT holds no folds.json" >&2; exit 2; }
BAND_ROOT=$(cd -- "$BAND_ROOT" && pwd)
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
CORPUS_ARGS=()
if [[ "$STAGE" == screen || "$STAGE" == confirm || "$STAGE" == map ]]; then
  M="${M:?Set M to the 40 s offset audit directory}"
  DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
  if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
    [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
    SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
  fi
  for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
  done
  CORPUS_ARGS=(--readout-root "$READOUT_ROOT" --manifests "$(cd -- "$M" && pwd)" --data-dir "$(cd -- "$DATA_DIR" && pwd)"
    --signal-cache-dir "$(cd -- "$SIGNAL_CACHE_DIR" && pwd)" --num-workers "$NUM_WORKERS")
fi
if [[ "$STAGE" == confirm || "$STAGE" == map ]]; then
  [[ -f "$ROOT/selection.json" ]] || { echo "ERROR: $ROOT has no selection.json: run STAGE=screen on it first" >&2; exit 2; }
fi
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE BAND_ROOT=$BAND_ROOT READOUT_ROOT=$READOUT_ROOT NUM_WORKERS=$NUM_WORKERS ${M:+M=$M} ${SIGNAL_CACHE_DIR:+SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic training on CUDA needs it, as in run_frozen_readout.sh
export HYDRA_FULL_ERROR=1
if [[ "$STAGE" == screen || "$STAGE" == confirm || "$STAGE" == map ]]; then
  python -m pytest -q tests/test_band_map_core.py tests/test_band_map.py \
    tests/test_band_sweep_core.py tests/test_band_sweep.py tests/test_band_reconstruction_core.py \
    tests/test_band_reconstruction.py tests/test_reconstruction_baselines_core.py tests/test_reconstruction_baselines.py \
    2>&1 | tee -a "$ROOT/tests.log"
fi

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/band_sweep.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}
settings() {
  python src/band_sweep.py list --stage "$1" --root "$ROOT"
}

ROOTS=(--root "$ROOT" --band-root "$BAND_ROOT")
case "$STAGE" in
  screen)
    mapfile -t SMOKE < <(settings smoke)
    for name in "${SMOKE[@]}"; do
      step fit "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold 0 --setting "$name" --smoke
    done
    mapfile -t FITS < <(settings screen-fit)
    mapfile -t SCORES < <(settings screen-score)
    for fold in 0 1; do
      for name in "${FITS[@]}"; do
        step fit "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --setting "$name"
      done
      for name in "${SCORES[@]}"; do
        step score "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --setting "$name"
      done
    done
    step select "${ROOTS[@]}" --readout-root "$READOUT_ROOT"
    cat "$ROOT/selection.json"
    ;;
  confirm)
    mapfile -t FITS < <(settings confirm-fit)
    mapfile -t SCORES < <(settings confirm-score)
    for fold in 2 3 4; do
      for name in ${FITS[@]+"${FITS[@]}"}; do
        step fit "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --setting "$name"
      done
      for name in ${SCORES[@]+"${SCORES[@]}"}; do
        step score "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --setting "$name"
      done
    done
    step read "${ROOTS[@]}" --readout-root "$READOUT_ROOT"
    cat "$ROOT/sweep_decision.json"
    ;;
  map)
    mapfile -t CELLS < <(settings map)
    for fold in 0 1; do
      for name in "${CELLS[@]}"; do
        step fit "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --setting "$name"
      done
      for name in "${CELLS[@]}"; do
        step score "${ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --setting "$name"
      done
    done
    step map "${ROOTS[@]}" --readout-root "$READOUT_ROOT"
    cat "$ROOT/map.json"
    ;;
  select)
    step select "${ROOTS[@]}" --readout-root "$READOUT_ROOT"
    ;;
  read)
    step read "${ROOTS[@]}" --readout-root "$READOUT_ROOT"
    ;;
esac
