#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Class-conditional dictionaries (documentation/class_dictionary_score_protocol.md): in each outer fold, one separable
#* dictionary (K64_L128_D8_P512_R4) per class and two on random halves; does the class whose dictionary reconstructs a
#* held-out patient better predict its class, beyond band power on the same rows and beyond the drug? One root holds
#* one read. Run inside the HPC container from code/, on a GPU node:
#*   SWEEP_ROOT=<sweep root> READOUT_ROOT=<pass 1 root> AUDIT_ROOT=<covariate audit root> \
#*     BAND_CODE_ROOT=<band-code readout root> M=<40 s audit> SMOKE_DIR=<its cache dir> \
#*     nohup bash run_class_dictionary.sh > class_dictionary.log 2>&1 &
#* It runs the tests (gate 1), writes the splits, trains 40 dictionaries and keeps 20 starts (5 folds x 6 arms x 2
#* seeds), scores every fold, checks gates 2-6 (gates.json) and stops if one fails, then reads the rule
#* (class_dictionary_decision.json). A rerun with the same ROOT skips finished steps. STAGE=read repeats only the read.
#* Environment: ROOT, SWEEP_ROOT, READOUT_ROOT, AUDIT_ROOT, BAND_CODE_ROOT, STAGE=all|read, M, SMOKE_DIR or
#* SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,15p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/class_dictionary/read_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:?Set SWEEP_ROOT to the sweep root (the rows reference)}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
AUDIT_ROOT="${AUDIT_ROOT:?Set AUDIT_ROOT to the covariate audit root}"
BAND_CODE_ROOT="${BAND_CODE_ROOT:?Set BAND_CODE_ROOT to the band-code readout root}"
[[ "$STAGE" =~ ^(all|read)$ ]] || { echo "ERROR: STAGE must be all or read" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ -d "$SWEEP_ROOT/scores" ]] || { echo "ERROR: $SWEEP_ROOT holds no sweep scores" >&2; exit 2; }
[[ -f "$READOUT_ROOT/folds.json" && -f "$READOUT_ROOT/patients.csv" ]] || {
  echo "ERROR: $READOUT_ROOT holds no folds.json and patients.csv" >&2; exit 2; }
[[ -f "$AUDIT_ROOT/drug_exposure.csv" && -f "$AUDIT_ROOT/covariates.csv" ]] || {
  echo "ERROR: $AUDIT_ROOT holds no drug_exposure.csv and covariates.csv" >&2; exit 2; }
[[ -f "$BAND_CODE_ROOT/scores/Band-scaled.csv" && -d "$BAND_CODE_ROOT/features" ]] || {
  echo "ERROR: $BAND_CODE_ROOT holds no Band-scaled scores and features" >&2; exit 2; }
SWEEP_ROOT=$(cd -- "$SWEEP_ROOT" && pwd)
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
AUDIT_ROOT=$(cd -- "$AUDIT_ROOT" && pwd)
BAND_CODE_ROOT=$(cd -- "$BAND_CODE_ROOT" && pwd)
CORPUS_ARGS=()
if [[ "$STAGE" == all ]]; then
  for fold in 0 1 2 3 4; do
    file="$SWEEP_ROOT/scores/K64_L128_D8_P512_R4_E30_s1_f${fold}.npz"
    [[ -f "$file" ]] || { echo "ERROR: missing sweep score file $file" >&2; exit 2; }
  done
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
echo "$STAMP STAGE=$STAGE SWEEP_ROOT=$SWEEP_ROOT READOUT_ROOT=$READOUT_ROOT AUDIT_ROOT=$AUDIT_ROOT BAND_CODE_ROOT=$BAND_CODE_ROOT NUM_WORKERS=$NUM_WORKERS ${M:+M=$M} ${SIGNAL_CACHE_DIR:+SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic algorithms on CUDA need it, as in run_band_reconstruction.sh
export HYDRA_FULL_ERROR=1
# Gate 1: the tests of this read and of the code it reuses.
python -m pytest -q tests/test_class_dictionary_core.py tests/test_class_dictionary.py \
  tests/test_band_code_readout_core.py tests/test_band_code_readout.py \
  tests/test_band_sweep_core.py tests/test_band_sweep.py tests/test_band_map_core.py tests/test_band_map.py \
  tests/test_band_reconstruction_core.py tests/test_band_reconstruction.py \
  tests/test_reconstruction_baselines_core.py tests/test_reconstruction_baselines.py \
  tests/test_frozen_readout_core.py tests/test_encoder_check_core.py tests/test_covariate_audit_core.py \
  2>&1 | tee -a "$ROOT/tests.log"

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/class_dictionary.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

CORPUS_ROOTS=(--root "$ROOT" --sweep-root "$SWEEP_ROOT" --readout-root "$READOUT_ROOT")
if [[ "$STAGE" == all ]]; then
  step split "${CORPUS_ROOTS[@]}"
  for fold in 0 1 2 3 4; do
    for seed in 1 2; do
      for arm in epi non A B epi-start non-start; do
        step fit "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --arm "$arm" --seed "$seed"
      done
    done
    step score "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold"
  done
  step gates --root "$ROOT" --readout-root "$READOUT_ROOT"
  if ! python -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "pass")' "$ROOT/gates.json"; then
    echo "ERROR: a gate failed (see $ROOT/gates.json); no score is read" >&2
    cat "$ROOT/gates.json"
    exit 1
  fi
fi
step read --root "$ROOT" --readout-root "$READOUT_ROOT" --audit-root "$AUDIT_ROOT" --band-code-root "$BAND_CODE_ROOT"
cat "$ROOT/class_dictionary_decision.json"
