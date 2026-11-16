#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Hinge redesign of the class-masked separable dictionary (documentation/masked_hinge_redesign_protocol.md): the
#* hinge starts at epoch 11, ramps over 5 epochs and acts through a step of its own. Two stages on one root. Run
#* inside the HPC container from code/, on a GPU node:
#*   STAGE=pilot FIRST_ROOT=<root of the first masked run> SWEEP_ROOT=<sweep root> READOUT_ROOT=<pass 1 root> \
#*     M=<40 s audit> SMOKE_DIR=<its cache dir> nohup bash run_masked_hinge.sh > masked_hinge_pilot.log 2>&1 &
#*   STAGE=full ROOT=<the pilot root> FIRST_ROOT=... SWEEP_ROOT=... READOUT_ROOT=... AUDIT_ROOT=<covariate audit root> \
#*     BAND_CODE_ROOT=<band-code readout root> M=... SMOKE_DIR=... nohup bash run_masked_hinge.sh > masked_hinge_full.log 2>&1 &
#* STAGE=pilot: the tests, the splits, the first pilot dictionary (sup, strength 1) and its gate, the other 7 pilot
#* dictionaries (fold 0, seed 1, strengths 1, 10, 100, 1000, arms sup and perm), then the reads and the rule
#* (pilot.json: go, no_working_hinge or blocked). STAGE=full refuses a root whose pilot did not give go; it trains
#* sup2 and perm2 at the pilot's strength (5 folds, seeds 1 and 2; fold 0 at seed 1 is copied from the pilot), scores
#* every fold, checks gates 2-8 (gates.json) and reads the rule (masked_hinge_decision.json). A rerun with the same
#* ROOT skips finished steps. STAGE=read repeats only the read.
#* Environment: STAGE=pilot|full|read, ROOT, FIRST_ROOT, SWEEP_ROOT, READOUT_ROOT, AUDIT_ROOT, BAND_CODE_ROOT, M,
#* SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,17p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:?Set STAGE to pilot, full or read}"
NUM_WORKERS="${NUM_WORKERS:-4}"
[[ "$STAGE" =~ ^(pilot|full|read)$ ]] || { echo "ERROR: STAGE must be pilot, full or read" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
if [[ "$STAGE" == pilot ]]; then
  ROOT="${ROOT:-$SCRIPT_DIR/logs/masked_hinge/run_$(date +%Y%m%d_%H%M%S)}"
else
  ROOT="${ROOT:?Set ROOT to the root of the pilot}"
  [[ -f "$ROOT/pilot.json" ]] || { echo "ERROR: $ROOT holds no pilot.json" >&2; exit 2; }
fi
FIRST_ROOT="${FIRST_ROOT:?Set FIRST_ROOT to the root of the first masked run}"
SWEEP_ROOT="${SWEEP_ROOT:?Set SWEEP_ROOT to the sweep root (the rows reference)}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
[[ -f "$FIRST_ROOT/dictionaries/mask_s1_f0/sae_state.pt" && -f "$FIRST_ROOT/splits/fold0.json" ]] || {
  echo "ERROR: $FIRST_ROOT holds no reference dictionary mask_s1_f0 and split of fold 0" >&2; exit 2; }
[[ -d "$SWEEP_ROOT/scores" ]] || { echo "ERROR: $SWEEP_ROOT holds no sweep scores" >&2; exit 2; }
[[ -f "$READOUT_ROOT/folds.json" && -f "$READOUT_ROOT/patients.csv" ]] || {
  echo "ERROR: $READOUT_ROOT holds no folds.json and patients.csv" >&2; exit 2; }
FIRST_ROOT=$(cd -- "$FIRST_ROOT" && pwd)
SWEEP_ROOT=$(cd -- "$SWEEP_ROOT" && pwd)
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
if [[ "$STAGE" != pilot ]]; then
  AUDIT_ROOT="${AUDIT_ROOT:?Set AUDIT_ROOT to the covariate audit root}"
  BAND_CODE_ROOT="${BAND_CODE_ROOT:?Set BAND_CODE_ROOT to the band-code readout root}"
  [[ -f "$AUDIT_ROOT/drug_exposure.csv" && -f "$AUDIT_ROOT/covariates.csv" ]] || {
    echo "ERROR: $AUDIT_ROOT holds no drug_exposure.csv and covariates.csv" >&2; exit 2; }
  [[ -f "$BAND_CODE_ROOT/scores/Band-scaled.csv" ]] || { echo "ERROR: $BAND_CODE_ROOT holds no Band-scaled scores" >&2; exit 2; }
  [[ -f "$READOUT_ROOT/scores/Spectral.csv" ]] || { echo "ERROR: $READOUT_ROOT holds no Spectral scores" >&2; exit 2; }
  for fold in 0 1 2 3 4; do
    [[ -f "$FIRST_ROOT/counts/fold${fold}.npz" ]] || { echo "ERROR: $FIRST_ROOT holds no counts of fold $fold" >&2; exit 2; }
  done
  AUDIT_ROOT=$(cd -- "$AUDIT_ROOT" && pwd)
  BAND_CODE_ROOT=$(cd -- "$BAND_CODE_ROOT" && pwd)
fi
CORPUS_ARGS=()
if [[ "$STAGE" != read ]]; then
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
echo "$STAMP STAGE=$STAGE FIRST_ROOT=$FIRST_ROOT SWEEP_ROOT=$SWEEP_ROOT READOUT_ROOT=$READOUT_ROOT ${AUDIT_ROOT:+AUDIT_ROOT=$AUDIT_ROOT} ${BAND_CODE_ROOT:+BAND_CODE_ROOT=$BAND_CODE_ROOT} NUM_WORKERS=$NUM_WORKERS ${M:+M=$M} ${SIGNAL_CACHE_DIR:+SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic algorithms on CUDA need it, as in run_band_reconstruction.sh
export HYDRA_FULL_ERROR=1

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/masked_hinge.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  local status=${PIPESTATUS[0]}  # inside an if, set -e is off: return the status of python, not of echo
  echo "$(date '+%F %T') done  $* (status $status)"
  return "$status"
}

field() {  # field <json file> <key>: the value of a top-level key
  python -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

if [[ "$STAGE" != read ]]; then
  # Gate 1: the tests of this run and of the code it reuses.
  python -m pytest -q tests/test_masked_separable_sae.py tests/test_masked_dictionary_core.py tests/test_masked_dictionary.py \
    tests/test_masked_hinge.py tests/test_separable_sae.py tests/test_waveform_sae.py tests/test_patient_waveform.py \
    tests/test_spectral_decomposition_core.py tests/test_class_dictionary_core.py tests/test_class_dictionary.py \
    tests/test_band_code_readout_core.py tests/test_band_code_readout.py \
    tests/test_band_sweep_core.py tests/test_band_sweep.py tests/test_band_map_core.py tests/test_band_map.py \
    tests/test_band_reconstruction_core.py tests/test_band_reconstruction.py \
    tests/test_reconstruction_baselines_core.py tests/test_reconstruction_baselines.py \
    tests/test_frozen_readout_core.py tests/test_encoder_check_core.py tests/test_covariate_audit_core.py \
    2>&1 | tee -a "$ROOT/tests_$STAGE.log"
fi

CORPUS_ROOTS=(--root "$ROOT" --sweep-root "$SWEEP_ROOT" --readout-root "$READOUT_ROOT")
if [[ "$STAGE" == pilot ]]; then
  step split "${CORPUS_ROOTS[@]}"
  # Stage-1 gate 2: one fit first. The pilot stops if its warm start has no leak or too few atoms in use.
  step pilot-fit "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --arm sup --eta 1
  if ! step pilot-check --root "$ROOT"; then
    echo "ERROR: the first pilot fit failed its gate (see $ROOT/pilot_first_fit.json); the pilot stops" >&2
    exit 1
  fi
  step pilot-fit "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --arm perm --eta 1
  for eta in 10 100 1000; do
    for arm in sup perm; do
      step pilot-fit "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --arm "$arm" --eta "$eta"
    done
  done
  step pilot-read "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --first-root "$FIRST_ROOT"
  cat "$ROOT/pilot.json"
  exit 0
fi

OUTCOME=$(field "$ROOT/pilot.json" outcome)
[[ "$OUTCOME" == go ]] || { echo "ERROR: the pilot of $ROOT gave $OUTCOME; stage 2 does not run" >&2; exit 1; }
if [[ "$STAGE" == full ]]; then
  for fold in 0 1 2 3 4; do
    for seed in 1 2; do
      for arm in sup2 perm2; do
        step fit "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold" --arm "$arm" --seed "$seed"
      done
    done
    step score "${CORPUS_ROOTS[@]}" "${CORPUS_ARGS[@]}" --fold "$fold"
  done
  step gates --root "$ROOT" --readout-root "$READOUT_ROOT"
fi
# The read opens no score unless the gates passed; it then records the outcome blocked or collapsed.
step read --root "$ROOT" --readout-root "$READOUT_ROOT" --audit-root "$AUDIT_ROOT" --band-code-root "$BAND_CODE_ROOT" \
  --first-root "$FIRST_ROOT"
cat "$ROOT/masked_hinge_decision.json"
[[ "$(field "$ROOT/gates.json" status)" == pass ]] || { echo "ERROR: a gate failed (see $ROOT/gates.json)" >&2; exit 1; }
