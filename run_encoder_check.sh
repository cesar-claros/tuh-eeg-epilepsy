#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Frozen encoder check (documentation/frozen_encoder_check_protocol.md): does REVE read above band power on our
#* patients, and does it add to the maintenance drug? One root holds one check. Run inside the HPC container from code/:
#*   STAGE=step0 READOUT_ROOT=<pass 1 root> AUDIT_ROOT=<audit root> bash run_encoder_check.sh
#*   STAGE=smoke ROOT=<the root> READOUT_ROOT=... AUDIT_ROOT=... M=<40 s audit> SMOKE_DIR=<its cache dir> bash run_encoder_check.sh
#*   STAGE=extract ROOT=<the root> READOUT_ROOT=... AUDIT_ROOT=... M=... SMOKE_DIR=... nohup bash run_encoder_check.sh > encoder_extract.log 2>&1 &
#*   STAGE=gates ROOT=<the root> READOUT_ROOT=... AUDIT_ROOT=... bash run_encoder_check.sh
#*   STAGE=read ROOT=<the root> READOUT_ROOT=... AUDIT_ROOT=... bash run_encoder_check.sh
#* step0, gates and read need CPU only; smoke and extract need a GPU node and access to the REVE weights: register on
#* the authors' Hugging Face repository, then set HF_TOKEN in the environment. The token is never written to a file.
#* Environment: ROOT, READOUT_ROOT, AUDIT_ROOT, STAGE, M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4,
#* DEVICE=cuda, SPLITS=1000, ACCEPT_REVIEW=0 (1 goes on after a miss of read H, once the preprocessing is inspected).
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,14p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-step0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SPLITS="${SPLITS:-1000}"
ACCEPT_REVIEW="${ACCEPT_REVIEW:-0}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/encoder_check/check_$(date +%Y%m%d_%H%M%S)}"
READOUT_ROOT="${READOUT_ROOT:?Set READOUT_ROOT to the root of frozen readout pass 1}"
AUDIT_ROOT="${AUDIT_ROOT:?Set AUDIT_ROOT to the root of the covariate audit}"
[[ "$STAGE" =~ ^(step0|smoke|extract|gates|read)$ ]] || { echo "ERROR: STAGE must be step0, smoke, extract, gates or read" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad NUM_WORKERS" >&2; exit 2; }
[[ "$SPLITS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad SPLITS" >&2; exit 2; }
[[ "$ACCEPT_REVIEW" =~ ^[01]$ ]] || { echo "ERROR: ACCEPT_REVIEW must be 0 or 1" >&2; exit 2; }
for dir in "$READOUT_ROOT" "$AUDIT_ROOT"; do
  [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
done
READOUT_ROOT=$(cd -- "$READOUT_ROOT" && pwd)
AUDIT_ROOT=$(cd -- "$AUDIT_ROOT" && pwd)
CORPUS_ARGS=()
if [[ "$STAGE" == smoke || "$STAGE" == extract ]]; then
  M="${M:?Set M to the 40 s offset audit directory}"
  DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
  if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
    [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
    SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
  fi
  for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
  done
  [[ -n ${HF_TOKEN:-} ]] || echo "WARNING: HF_TOKEN is not set; the gated REVE weights need it unless they are cached." >&2
  CORPUS_ARGS=(--manifests "$(cd -- "$M" && pwd)" --data-dir "$(cd -- "$DATA_DIR" && pwd)"
    --signal-cache-dir "$(cd -- "$SIGNAL_CACHE_DIR" && pwd)" --num-workers "$NUM_WORKERS" --device "$DEVICE")
fi
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE READOUT_ROOT=$READOUT_ROOT AUDIT_ROOT=$AUDIT_ROOT DEVICE=$DEVICE SPLITS=$SPLITS ACCEPT_REVIEW=$ACCEPT_REVIEW" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_encoder_check_core.py tests/test_encoder_check.py 2>&1 | tee -a "$ROOT/tests.log"

ROOTS=(--root "$ROOT" --readout-root "$READOUT_ROOT" --audit-root "$AUDIT_ROOT")
step() {
  echo "$(date '+%F %T') start $*"
  python -u src/encoder_check.py "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $*"
}

case "$STAGE" in
  step0)
    step step0 "${ROOTS[@]}" --splits "$SPLITS"
    cat "$ROOT/step0.json"
    ;;
  smoke)
    step smoke "${ROOTS[@]}" "${CORPUS_ARGS[@]}"
    ;;
  extract)
    step extract "${ROOTS[@]}" "${CORPUS_ARGS[@]}"
    ;;
  gates)
    step gates "${ROOTS[@]}"
    cat "$ROOT/gates.json"
    ;;
  read)
    extra=()
    [[ "$ACCEPT_REVIEW" == 1 ]] && extra+=(--accept-review)
    step read "${ROOTS[@]}" --splits "$SPLITS" ${extra[@]+"${extra[@]}"}
    cat "$ROOT/encoder_decision.json"
    ;;
esac
