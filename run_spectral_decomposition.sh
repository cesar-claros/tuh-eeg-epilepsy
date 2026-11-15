#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Spectral decomposition and score fusion (documentation/spectral_decomposition_fusion_protocol.md): does the
#* rank-mean of the saved Spectral, HYDRA and REVE scores read above Spectral, and which part of the 200 Spectral
#* features carries its score? CPU only: it reads saved files and no signal. One root holds one read. Run inside the
#* project container from code/:
#*   READOUT_ROOT=<pass 1 root> AUDIT_ROOT=<covariate audit root> ENCODER_ROOT=<encoder check root> \
#*     bash run_spectral_decomposition.sh
#* It runs the tests (gate 1), checks the saved features and fits the head of pass 1 on the 26 arms, checks gates 2-6
#* (gates.json) and stops if one fails, then reads the rule (fusion_decision.json, decomposition_summary.csv). A rerun
#* with the same ROOT skips finished arms. STAGE=read repeats only the read. STAGE=lead (same ROOT, after the read)
#* checks the post-hoc lead of the read (documentation/small_arm_lead_check_protocol.md): lead_decision.json.
#* Environment: ROOT, READOUT_ROOT, AUDIT_ROOT, ENCODER_ROOT, STAGE=all|read|lead, N_BOOT=10000.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,13p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
N_BOOT="${N_BOOT:-10000}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/spectral_decomposition/read_$(date +%Y%m%d_%H%M%S)}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
AUDIT_ROOT="${AUDIT_ROOT:?Set AUDIT_ROOT to the covariate audit root}"
ENCODER_ROOT="${ENCODER_ROOT:?Set ENCODER_ROOT to the root of the encoder check}"
[[ "$STAGE" =~ ^(all|read|lead)$ ]] || { echo "ERROR: STAGE must be all, read or lead" >&2; exit 2; }
if [[ "$STAGE" == lead ]]; then
  [[ -n ${ROOT:-} && -f "$ROOT/decomposition_summary.csv" ]] || {
    echo "ERROR: STAGE=lead needs the ROOT of a finished read" >&2; exit 2; }
fi
[[ "$N_BOOT" =~ ^[0-9]+$ ]] || { echo "ERROR: bad N_BOOT" >&2; exit 2; }
[[ -f "$READOUT_ROOT/folds.json" && -f "$READOUT_ROOT/patients.csv" ]] || {
  echo "ERROR: $READOUT_ROOT holds no folds.json and patients.csv" >&2; exit 2; }
[[ -f "$READOUT_ROOT/features/Spectral.npz" ]] || { echo "ERROR: $READOUT_ROOT holds no Spectral features" >&2; exit 2; }
[[ -f "$READOUT_ROOT/scores/Spectral.csv" && -f "$READOUT_ROOT/scores/HYDRA.csv" ]] || {
  echo "ERROR: $READOUT_ROOT holds no Spectral and HYDRA scores" >&2; exit 2; }
[[ -f "$AUDIT_ROOT/drug_exposure.csv" && -f "$AUDIT_ROOT/covariates.csv" ]] || {
  echo "ERROR: $AUDIT_ROOT holds no drug_exposure.csv and covariates.csv" >&2; exit 2; }
[[ -f "$ENCODER_ROOT/scores/D.csv" ]] || { echo "ERROR: $ENCODER_ROOT holds no scores/D.csv" >&2; exit 2; }
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
AUDIT_ROOT=$(cd -- "$AUDIT_ROOT" && pwd)
ENCODER_ROOT=$(cd -- "$ENCODER_ROOT" && pwd)
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE READOUT_ROOT=$READOUT_ROOT AUDIT_ROOT=$AUDIT_ROOT ENCODER_ROOT=$ENCODER_ROOT N_BOOT=$N_BOOT" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
# Gate 1: the tests of this read and of the code it reuses.
python -m pytest -q tests/test_spectral_decomposition_core.py tests/test_spectral_decomposition.py \
  tests/test_frozen_readout_core.py tests/test_encoder_check_core.py tests/test_covariate_audit_core.py \
  2>&1 | tee -a "$ROOT/tests.log"

step() {
  echo "$(date '+%F %T') start $*"
  python -u src/spectral_decomposition.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

if [[ "$STAGE" == all ]]; then
  step fit --root "$ROOT" --readout-root "$READOUT_ROOT"
  step gates --root "$ROOT" --readout-root "$READOUT_ROOT" --encoder-root "$ENCODER_ROOT"
  if ! python -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "pass")' "$ROOT/gates.json"; then
    echo "ERROR: a gate failed (see $ROOT/gates.json); no score is read" >&2
    cat "$ROOT/gates.json"
    exit 1
  fi
fi
if [[ "$STAGE" == lead ]]; then
  step lead --root "$ROOT" --readout-root "$READOUT_ROOT" --audit-root "$AUDIT_ROOT" --encoder-root "$ENCODER_ROOT"
  cat "$ROOT/lead_decision.json"
  exit 0
fi
step read --root "$ROOT" --readout-root "$READOUT_ROOT" --audit-root "$AUDIT_ROOT" --encoder-root "$ENCODER_ROOT" \
  --n-boot "$N_BOOT"
cat "$ROOT/fusion_decision.json"
