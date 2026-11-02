#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Covariate audit (documentation/covariate_audit_protocol.md): does the saved Spectral score of frozen readout
#* pass 1 hold on pairs matched for age and sex? CPU only; reads no signal. Two stages on one root, because the
#* addendum (the drug mapping) must be fixed from the inventory before any read by class.
#*
#* Run inside the HPC container from code/:
#*   STAGE=inventory READOUT_ROOT=<pass 1 root> M=<40 s audit> SMOKE_DIR=<its cache_smoke_P512_* dir> bash run_covariate_audit.sh
#*   STAGE=read ROOT=<the same root> READOUT_ROOT=<pass 1 root> MAPPING=<mapping json> bash run_covariate_audit.sh
#*   STAGE=drug ROOT=<the same root> READOUT_ROOT=<pass 1 root> MAPPING=<mapping json> bash run_covariate_audit.sh
#* Environment: ROOT, READOUT_ROOT, STAGE=inventory|read|drug, MAPPING, DRUG_CLASSES, M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR,
#* METADATA (default <DATA_DIR>/v3.0.0/DOCS/metadata_v00r.xlsx), NUM_WORKERS=4, HEADER_ROW=0, SUBJECT_COLUMN,
#* SESSION_COLUMN (the last three only if the first inventory could not find the join).
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,13p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-inventory}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HEADER_ROW="${HEADER_ROW:-0}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/covariate_audit/audit_$(date +%Y%m%d_%H%M%S)}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
[[ "$STAGE" == inventory || "$STAGE" == read || "$STAGE" == drug ]] || { echo "ERROR: STAGE must be inventory, read or drug" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ "$HEADER_ROW" =~ ^[0-9]+$ ]] || { echo "ERROR: bad HEADER_ROW" >&2; exit 2; }
for dir in "$READOUT_ROOT" "$DATA_DIR"; do
  [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
done
METADATA="${METADATA:-$DATA_DIR/v3.0.0/DOCS/metadata_v00r.xlsx}"
[[ -f "$METADATA" ]] || { echo "ERROR: spreadsheet not found: $METADATA" >&2; exit 2; }
if [[ "$STAGE" == inventory ]]; then
  M="${M:?Set M to the 40 s offset audit directory}"
  if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
    [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
    SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
  fi
  for dir in "$M" "$SIGNAL_CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
  done
else
  MAPPING="${MAPPING:?Set MAPPING to the medication mapping of the addendum}"
  [[ -f "$MAPPING" ]] || { echo "ERROR: mapping not found: $MAPPING" >&2; exit 2; }
  MAPPING=$(cd -- "$(dirname -- "$MAPPING")" && pwd)/$(basename -- "$MAPPING")
  [[ -f "$ROOT/read_segments.csv" ]] || { echo "ERROR: run STAGE=inventory on this root first: $ROOT" >&2; exit 2; }
fi
if [[ "$STAGE" == drug ]]; then
  [[ -f "$ROOT/audit_decision.json" ]] || { echo "ERROR: run STAGE=read on this root first: $ROOT" >&2; exit 2; }
  DRUG_CLASSES="${DRUG_CLASSES:-$SCRIPT_DIR/configs/covariate_audit/drug_classes.json}"
  [[ -f "$DRUG_CLASSES" ]] || { echo "ERROR: drug classes not found: $DRUG_CLASSES" >&2; exit 2; }
  DRUG_CLASSES=$(cd -- "$(dirname -- "$DRUG_CLASSES")" && pwd)/$(basename -- "$DRUG_CLASSES")
fi
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
METADATA=$(cd -- "$(dirname -- "$METADATA")" && pwd)/$(basename -- "$METADATA")
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE READOUT_ROOT=$READOUT_ROOT DATA_DIR=$DATA_DIR METADATA=$METADATA MAPPING=${MAPPING:-}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_covariate_audit_core.py tests/test_covariate_audit.py 2>&1 | tee -a "$ROOT/tests.log"

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/covariate_audit.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

if [[ "$STAGE" == inventory ]]; then
  extra=()
  [[ -n ${SUBJECT_COLUMN:-} ]] && extra+=(--subject-column "$SUBJECT_COLUMN")
  [[ -n ${SESSION_COLUMN:-} ]] && extra+=(--session-column "$SESSION_COLUMN")
  step inventory --root "$ROOT" --readout-root "$READOUT_ROOT" --data-dir "$DATA_DIR" --metadata "$METADATA" \
    --manifests "$(cd -- "$M" && pwd)" --signal-cache-dir "$(cd -- "$SIGNAL_CACHE_DIR" && pwd)" \
    --num-workers "$NUM_WORKERS" --header-row "$HEADER_ROW" "${extra[@]}"
  echo "Inventory: $ROOT/inventory.json. It holds no label, no score and no value of age or sex."
elif [[ "$STAGE" == drug ]]; then
  step drug-read --root "$ROOT" --readout-root "$READOUT_ROOT" --data-dir "$DATA_DIR" --metadata "$METADATA" \
    --mapping "$MAPPING" --drug-classes "$DRUG_CLASSES"
  cat "$ROOT/drug_read_decision.json"
else
  if [[ -f "$ROOT/audit_decision.json" ]]; then
    cp -- "$ROOT/audit_decision.json" "$ROOT/audit_decision_before_$STAMP.json"  # keep the record of an earlier read
  fi
  step extract --root "$ROOT"
  step read --root "$ROOT" --readout-root "$READOUT_ROOT" --data-dir "$DATA_DIR" --metadata "$METADATA" --mapping "$MAPPING"
  cat "$ROOT/audit_decision.json"
fi
