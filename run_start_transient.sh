#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Recording-start transient (documentation/recording_start_transient_protocol.md): per-second rms,
#* line length and clip fraction of the cached first window (0-120 s) of every train/validation
#* recording, against the same seconds of its second window (120-240 s); curves by class and native
#* rate, and the proposed skip_start_s from the decision rule in src/start_transient.py. Reads the
#* campaign signal cache only (refuses to rebuild it) and never reads test signals. CPU suffices.
#*
#* Run inside the HPC container from code/:
#*   bash run_start_transient.sh                     # campaign audit and cache by default
#*   bash run_start_transient.sh /path/to/audit      # another audit with a matching cache
#* Environment: M (or argument 1), SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, NUM_WORKERS=4, OUT (new).
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,12p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M="${1:-${M:-$SCRIPT_DIR/logs/audits/wsae_all_segments_fresh42_20261023_210759}}"
SMOKE_DIR="${SMOKE_DIR:-$SCRIPT_DIR/logs/waveform_patient/cache_smoke_P512_7cqc27GG}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUT="${OUT:-$SCRIPT_DIR/logs/start_transient/start_transient_$(date +%Y%m%d_%H%M%S)}"
if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
  [[ -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
  SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
fi
for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
  [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
done
cd -- "$SCRIPT_DIR"
mkdir -p -- "$(dirname -- "$OUT")"
python -m pytest -q tests/test_start_transient.py
python -u src/start_transient.py --manifests "$M" --data-dir "$DATA_DIR" --signal-cache-dir "$SIGNAL_CACHE_DIR" \
  --output "$OUT" --num-workers "$NUM_WORKERS" 2>&1 | tee "$OUT.log"
git rev-parse HEAD > "$OUT/sha.txt"
echo "Results: $OUT (start_transient_decision.json, start_transient.png)"
