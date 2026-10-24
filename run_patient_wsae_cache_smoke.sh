#!/usr/bin/env bash
# Run inside the HPC container on an allocated GPU node.
# Usage: bash run_patient_wsae_cache_smoke.sh /path/to/fresh_uncapped_audit
# CACHE_PARENT optionally places a new cache on fast storage. Existing caches are never cleared.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash $0 /path/to/fresh_uncapped_audit"
  echo "Environment: M, DATA_DIR, RUN_DIR (new), CACHE_PARENT, NUM_WORKERS=4"
  echo "Two reconstruction epochs: seed 0, L=256, P=512, R=1, complete validation, fresh cache."
  exit 0
fi
if (( $# > 1 )); then
  echo "ERROR: provide one manifest directory." >&2
  exit 2
fi
M="${1:-${M:-}}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ -z "$M" || ! -d "$M" || ! -d "$DATA_DIR" || ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: provide existing audit/DATA_DIR directories and nonnegative NUM_WORKERS." >&2
  exit 2
fi
M=$(cd -- "$M" && pwd)
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
if [[ -n ${RUN_DIR:-} ]]; then
  mkdir -p -- "$(dirname -- "$RUN_DIR")"
  mkdir -- "$RUN_DIR"  # Refuse an existing run instead of mixing cold/warm measurements.
  RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
fi
if [[ -n ${CACHE_PARENT:-} ]]; then
  mkdir -p -- "$CACHE_PARENT"
  CACHE_PARENT=$(cd -- "$CACHE_PARENT" && pwd)
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z ${RUN_DIR:-} ]]; then
  mkdir -p logs/waveform_patient
  RUN_DIR=$(mktemp -d "$PWD/logs/waveform_patient/cache_smoke_P512_XXXXXXXX")
fi
CACHE_PARENT="${CACHE_PARENT:-$RUN_DIR}"
# Python creates this leaf exclusively. Ignore any inherited SIGNAL_CACHE_DIR to guarantee a cold start.
CACHE_DIR="$CACHE_PARENT/$(basename -- "$RUN_DIR")_signals"
if [[ -e "$CACHE_DIR" || -L "$CACHE_DIR" ]]; then
  echo "ERROR: cache path already exists; choose a new RUN_DIR or CACHE_PARENT: $CACHE_DIR" >&2
  exit 2
fi
echo "Results: $RUN_DIR"
echo "Fresh signal cache: $CACHE_DIR"
printf '%s\n' "$CACHE_DIR" > "$RUN_DIR/signal_cache_path.txt"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_signal_cache.py 2>&1 | tee "$RUN_DIR/tests.log"
python -u src/run_patient_waveform_cache_smoke.py \
  --manifests "$M" --data-dir "$DATA_DIR" --output "$RUN_DIR" \
  --signal-cache-dir "$CACHE_DIR" --num-workers "$NUM_WORKERS" \
  2>&1 | tee "$RUN_DIR/console.log"

echo "Cache reuse passed. Timing comparison: $RUN_DIR/validation_cache_comparison.csv"
echo "Summary: $RUN_DIR/cache_smoke_summary.json"
echo "The first validation includes cache writes; the second must make zero preprocessing calls."
echo "For the full sweep, use a new RUN_DIR and SIGNAL_CACHE_DIR=$CACHE_DIR"
