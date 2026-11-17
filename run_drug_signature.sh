#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Within-patient drug protocol (documentation/drug_signature_detection_protocol.md): when a patient's drug list
#* changes between sessions, does the band power of that patient change in the direction that the literature reports?
#* Three stages on one root; no training; the 39 test patients stay locked. Run inside the HPC container from code/:
#*   STAGE=inventory READOUT_ROOT=<pass 1 root> AUDIT_ROOT=<covariate audit root> M=<40 s audit> SMOKE_DIR=<its cache smoke dir> bash run_drug_signature.sh
#*   STAGE=features ROOT=<the same root> READOUT_ROOT=<pass 1 root> M=<40 s audit> SMOKE_DIR=<its cache smoke dir> bash run_drug_signature.sh
#*   STAGE=test ROOT=<the same root> READOUT_ROOT=<pass 1 root> bash run_drug_signature.sh
#* inventory: the tests (gate 1; also before the test stage), the key of every segment (no signal is read), then the drugs and the side of every
#* session and the size of every contrast (drug_inventory.json; no EEG, no score). features: band power and alpha
#* frequency of every segment (one pass over the stored signals). test: gates 2-6 (gates.json), then the paired tests
#* and the rule (drug_signature_decision.json, drug_signature_summary.csv); a failed gate gives the outcome blocked.
#* Environment: ROOT, READOUT_ROOT, AUDIT_ROOT, STAGE=inventory|features|test, M, SMOKE_DIR or SIGNAL_CACHE_DIR,
#* DATA_DIR, METADATA (default <DATA_DIR>/v3.0.0/DOCS/metadata_v00r.xlsx), NUM_WORKERS=4.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,14p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-inventory}"
NUM_WORKERS="${NUM_WORKERS:-4}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
[[ "$STAGE" =~ ^(inventory|features|test)$ ]] || { echo "ERROR: STAGE must be inventory, features or test" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ -f "$READOUT_ROOT/folds.json" ]] || { echo "ERROR: $READOUT_ROOT holds no folds.json" >&2; exit 2; }
[[ -f "$READOUT_ROOT/features/Spectral-all.npz" && -f "$READOUT_ROOT/features/Spectral.npz" ]] || {
  echo "ERROR: $READOUT_ROOT holds no Spectral features" >&2; exit 2; }
if [[ "$STAGE" == inventory ]]; then
  ROOT="${ROOT:-$SCRIPT_DIR/logs/drug_signature/signature_$(date +%Y%m%d_%H%M%S)}"
  AUDIT_ROOT="${AUDIT_ROOT:?Set AUDIT_ROOT to the covariate audit root}"
  [[ -f "$AUDIT_ROOT/plan_recordings.csv" && -f "$AUDIT_ROOT/read_segments.csv" ]] || {
    echo "ERROR: $AUDIT_ROOT holds no plan_recordings.csv and read_segments.csv" >&2; exit 2; }
  AUDIT_ROOT=$(cd -- "$AUDIT_ROOT" && pwd)
  METADATA="${METADATA:-$DATA_DIR/v3.0.0/DOCS/metadata_v00r.xlsx}"
  [[ -f "$METADATA" ]] || { echo "ERROR: spreadsheet not found: $METADATA" >&2; exit 2; }
  METADATA=$(cd -- "$(dirname -- "$METADATA")" && pwd)/$(basename -- "$METADATA")
else
  ROOT="${ROOT:?Set ROOT to the root of STAGE=inventory}"
  [[ -f "$ROOT/drug_inventory.json" && -f "$ROOT/segment_keys.csv" ]] || {
    echo "ERROR: run STAGE=inventory on this root first: $ROOT" >&2; exit 2; }
fi
if [[ "$STAGE" == test ]]; then
  [[ -f "$ROOT/segment_features.npz" ]] || { echo "ERROR: run STAGE=features on this root first: $ROOT" >&2; exit 2; }
else
  M="${M:?Set M to the 40 s offset audit directory}"
  if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
    [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || {
      echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
    SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
  fi
  for dir in "$M" "$SIGNAL_CACHE_DIR" "$DATA_DIR"; do
    [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
  done
  M=$(cd -- "$M" && pwd)
  SIGNAL_CACHE_DIR=$(cd -- "$SIGNAL_CACHE_DIR" && pwd)
  DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
fi
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE READOUT_ROOT=$READOUT_ROOT AUDIT_ROOT=${AUDIT_ROOT:-} M=${M:-} METADATA=${METADATA:-}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/drug_signature.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  local status=${PIPESTATUS[0]}
  echo "$(date '+%F %T') done  $*"
  return "$status"
}
corpus=(--root "$ROOT" --readout-root "$READOUT_ROOT" --manifests "${M:-}" --data-dir "$DATA_DIR"
  --signal-cache-dir "${SIGNAL_CACHE_DIR:-}" --num-workers "$NUM_WORKERS")

if [[ "$STAGE" != features ]]; then
  # Gate 1: the tests of this protocol and of the code it reuses. A failure stops the script here.
  python -m pytest -q tests/test_drug_signature_core.py tests/test_drug_signature.py \
    tests/test_covariate_audit_core.py tests/test_covariate_audit.py tests/test_frozen_readout_core.py \
    2>&1 | tee -a "$ROOT/tests.log"
fi
if [[ "$STAGE" == inventory ]]; then
  step keys "${corpus[@]}"
  step inventory --root "$ROOT" --readout-root "$READOUT_ROOT" --audit-root "$AUDIT_ROOT" --metadata "$METADATA"
  cat "$ROOT/drug_inventory.json"
  echo "Inventory: $ROOT/drug_inventory.json. It holds no EEG and no score. Next: STAGE=features ROOT=$ROOT"
elif [[ "$STAGE" == features ]]; then
  step features "${corpus[@]}"
  echo "Features: $ROOT/segment_features.npz. Next: STAGE=test ROOT=$ROOT"
else
  if [[ -f "$ROOT/drug_signature_decision.json" ]]; then
    cp -- "$ROOT/drug_signature_decision.json" "$ROOT/drug_signature_decision_before_$STAMP.json"  # keep an earlier read
  fi
  step gates --root "$ROOT" --readout-root "$READOUT_ROOT"
  step test --root "$ROOT"
  cat "$ROOT/drug_signature_decision.json"
fi
