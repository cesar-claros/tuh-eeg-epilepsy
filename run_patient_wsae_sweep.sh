#!/usr/bin/env bash
# Run inside the HPC container on an allocated GPU node.
# Usage: bash run_patient_wsae_sweep.sh /path/to/fresh_uncapped_audit
# RUN_DIR=/path/to/existing/campaign resumes completed stages after interruption.
# STAGE=prepare stops after the three pretrained/calibrated references.
# STAGE=sweep requires those references; the default STAGE=all does both.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash $0 /path/to/fresh_uncapped_audit"
  echo "Environment: M, DATA_DIR, RUN_DIR, NUM_WORKERS=4, STAGE=all|prepare|sweep"
  echo "Fixed protocol: seeds 0,1,2; 30/50/30 epochs; 32 probes; five relative weights."
  exit 0
fi
if (( $# > 1 )); then
  echo "ERROR: provide one manifest directory." >&2
  exit 2
fi
M="${1:-${M:-}}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
STAGE="${STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ -z "$M" || ! -d "$M" || ! -d "$DATA_DIR" ]]; then
  echo "ERROR: provide an existing audit directory and corpus parent DATA_DIR." >&2
  exit 2
fi
if [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ || ! "$STAGE" =~ ^(all|prepare|sweep)$ ]]; then
  echo "ERROR: NUM_WORKERS must be nonnegative; STAGE must be all, prepare or sweep." >&2
  exit 2
fi
M=$(cd -- "$M" && pwd)
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
# Resolve a supplied output path relative to the caller, before changing directory.
if [[ -n ${RUN_DIR:-} ]]; then
  mkdir -p -- "$RUN_DIR"
  RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
elif [[ "$STAGE" == sweep ]]; then
  echo "ERROR: STAGE=sweep requires RUN_DIR from a completed preparation." >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z ${RUN_DIR:-} ]]; then
  mkdir -p logs/waveform_patient
  RUN_DIR=$(mktemp -d "$PWD/logs/waveform_patient/P512_linear_sweep_XXXXXXXX")
fi
echo "Campaign: $RUN_DIR"
echo "Seeds: 0 1 2; multipliers: 0.03 0.1 0.3 1 3; stage: $STAGE"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1
# Save a new console/test log on each invocation, including resumed invocations.
INVOCATION=$(mktemp -d "$RUN_DIR/invocation_XXXXXXXX")
python -m pytest -q \
  tests/test_patient_waveform.py tests/test_waveform_full_segments.py \
  tests/test_waveform_sae.py tests/test_count_pool_segments.py \
  tests/test_patient_waveform_sweep.py \
  2>&1 | tee "$INVOCATION/tests.log"

python -u src/run_patient_waveform_sweep.py \
  --manifests "$M" --data-dir "$DATA_DIR" --output "$RUN_DIR" \
  --num-workers "$NUM_WORKERS" --stage "$STAGE" \
  2>&1 | tee "$INVOCATION/console.log"

echo "Stage $STAGE completed. Results: $RUN_DIR"
