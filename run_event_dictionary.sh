#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Events corpus protocol (documentation/event_corpus_dictionary_protocol.md): do dictionary waveforms mark the
#* annotated events of the TUH EEG Events Corpus (v2.0.1)? Only stage 0 exists: the user decided that the inventory
#* comes first, and the later stages are built after their confirmation. Run inside the HPC container from code/:
#*   STAGE=inventory EVENTS_DIR=<the folder that holds edf/> bash run_event_dictionary.sh
#* inventory: the tests, then the header of every file and every .rec annotation file. No signal is read. It writes
#* files.csv, patients.csv, folds.json and event_inventory.json (counts per set and class, interval lengths, folds,
#* and the floor for each class). CPU, minutes.
#* Environment: ROOT, STAGE=inventory, EVENTS_DIR (default <DATA_DIR>/tuh_eeg_events/v2.0.1), DATA_DIR.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,10p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-inventory}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
EVENTS_DIR="${EVENTS_DIR:-$DATA_DIR/tuh_eeg_events/v2.0.1}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/event_dictionary/events_$(date +%Y%m%d_%H%M%S)}"
[[ "$STAGE" == inventory ]] || { echo "ERROR: only STAGE=inventory exists" >&2; exit 2; }
[[ -d "$EVENTS_DIR/edf/train" && -d "$EVENTS_DIR/edf/eval" ]] || {
  echo "ERROR: $EVENTS_DIR holds no edf/train and edf/eval" >&2; exit 2; }
EVENTS_DIR=$(cd -- "$EVENTS_DIR" && pwd)
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE EVENTS_DIR=$EVENTS_DIR" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
python -m pytest -q tests/test_event_dictionary.py 2>&1 | tee -a "$ROOT/tests.log"
echo "$(date '+%F %T') start inventory"
python -u src/event_dictionary.py inventory --root "$ROOT" --events-dir "$EVENTS_DIR" 2>&1 | tee -a "$ROOT/run.log"
echo "$(date '+%F %T') done  inventory"
cat "$ROOT/event_inventory.json"
echo "Inventory: $ROOT. It read headers and annotation files only."
