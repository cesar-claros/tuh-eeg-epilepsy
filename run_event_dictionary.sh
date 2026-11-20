#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Events corpus protocol (documentation/event_corpus_dictionary_protocol.md): do dictionary waveforms mark the
#* annotated events of the TUH EEG Events Corpus (v2.0.1), and do they find them in records of 5 minutes? Stages 0
#* (inventory) and 1 (pilot) exist; stage 2 (new dictionaries) is built after the pilot has run. Run inside the HPC
#* container from code/:
#*   STAGE=inventory EVENTS_DIR=<the folder that holds edf/> bash run_event_dictionary.sh
#*   STAGE=pilot ROOT=<the inventory root> EVENTS_DIR=<...> SWEEP_ROOT=<the band sweep root> M=<40 s audit> \
#*     SMOKE_DIR=<its cache smoke dir> nohup bash run_event_dictionary.sh > event_pilot.log 2>&1 &
#* inventory (CPU, minutes): the tests, then headers and .rec annotation files only; writes files.csv, patients.csv,
#* folds.json and event_inventory.json. pilot (GPU node, about 2 h, no training): the tests, the bad-channel sidecars
#* of the training files, the signal cache, the reader check on the epilepsy corpus (gate 3), the three reads of every
#* unit for the adopted dictionaries of the epilepsy corpus, random waveforms and the simple measures, with the
#* time-alignment check (gate 4), then pilot.json and pilot_summary.csv. A rerun with the same ROOT keeps finished
#* sidecars and cached files. The evaluation set is not read.
#* Environment: ROOT, STAGE=inventory|pilot, EVENTS_DIR (default <DATA_DIR>/v2.0.1), DATA_DIR (the data root, which
#* also holds the epilepsy corpus), SWEEP_ROOT, M, SMOKE_DIR or SIGNAL_CACHE_DIR, N_JOBS=8, NUM_WORKERS=4,
#* DEVICE=cuda, N_BOOT=10000.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,18p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-inventory}"
N_JOBS="${N_JOBS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
N_BOOT="${N_BOOT:-10000}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
EVENTS_DIR="${EVENTS_DIR:-$DATA_DIR/v2.0.1}"
[[ "$STAGE" =~ ^(inventory|pilot)$ ]] || { echo "ERROR: STAGE must be inventory or pilot" >&2; exit 2; }
for value in "$N_JOBS" "$NUM_WORKERS" "$N_BOOT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "ERROR: bad N_JOBS, NUM_WORKERS or N_BOOT" >&2; exit 2; }
done
[[ -d "$EVENTS_DIR/edf/train" && -d "$EVENTS_DIR/edf/eval" ]] || {
  echo "ERROR: $EVENTS_DIR holds no edf/train and edf/eval" >&2; exit 2; }
EVENTS_DIR=$(cd -- "$EVENTS_DIR" && pwd)
if [[ "$STAGE" == inventory ]]; then
  ROOT="${ROOT:-$SCRIPT_DIR/logs/event_dictionary/events_$(date +%Y%m%d_%H%M%S)}"
else
  ROOT="${ROOT:?Set ROOT to the root of STAGE=inventory (the pass with records of 5 minutes)}"
  [[ -f "$ROOT/files.csv" && -f "$ROOT/folds.json" ]] || {
    echo "ERROR: run STAGE=inventory on this root first: $ROOT" >&2; exit 2; }
  SWEEP_ROOT="${SWEEP_ROOT:?Set SWEEP_ROOT to the band sweep root with the adopted dictionaries}"
  [[ -f "$SWEEP_ROOT/dictionaries/K64_L128_D8_P512_R4_E30_s1_f0/sae_time_atoms.npy" ]] || {
    echo "ERROR: $SWEEP_ROOT holds no adopted dictionaries" >&2; exit 2; }
  SWEEP_ROOT=$(cd -- "$SWEEP_ROOT" && pwd)
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
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE EVENTS_DIR=$EVENTS_DIR SWEEP_ROOT=${SWEEP_ROOT:-} M=${M:-}" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export HYDRA_FULL_ERROR=1
# Gate 1: the tests of this protocol. A failure stops the script here.
python -m pytest -q tests/test_event_dictionary.py tests/test_event_pilot.py 2>&1 | tee -a "$ROOT/tests.log"

step() {
  local script=$1
  shift
  echo "$(date '+%F %T') start $script $*"
  python -u "src/$script.py" "$@" 2>&1 | tee -a "$ROOT/run.log"
  echo "$(date '+%F %T') done  $script $1"
}

if [[ "$STAGE" == inventory ]]; then
  step event_dictionary inventory --root "$ROOT" --events-dir "$EVENTS_DIR"
  cat "$ROOT/event_inventory.json"
  echo "Inventory: $ROOT. It read headers and annotation files only."
  exit 0
fi

# The bad-channel sidecars of the training files; a file that has one is skipped. The evaluation set is not read.
echo "$(date '+%F %T') start bad-channel sidecars"
find "$EVENTS_DIR/edf/train" -name '*.edf' -print0 | sort -z \
  | xargs -0 python -u src/precompute_badchannels.py --n_jobs "$N_JOBS" --edf 2>&1 | tee -a "$ROOT/run.log"
echo "$(date '+%F %T') done  bad-channel sidecars"
step event_pilot cache --root "$ROOT" --events-dir "$EVENTS_DIR" --n-jobs "$N_JOBS"
step event_pilot reader-check --root "$ROOT" --manifests "$M" --data-dir "$DATA_DIR" \
  --signal-cache-dir "$SIGNAL_CACHE_DIR" --num-workers "$NUM_WORKERS"
step event_pilot features --root "$ROOT" --events-dir "$EVENTS_DIR" --sweep-root "$SWEEP_ROOT" --device "$DEVICE"
if [[ -f "$ROOT/pilot.json" ]]; then
  cp -- "$ROOT/pilot.json" "$ROOT/pilot_before_$STAMP.json"  # keep the record of an earlier read
fi
step event_pilot read --root "$ROOT" --n-boot "$N_BOOT"
cat "$ROOT/pilot.json" | head -60
echo "Pilot: $ROOT/pilot.json and $ROOT/pilot_summary.csv. The pilot is descriptive."
