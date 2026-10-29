#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Gate 1 of the separable atom SAE probe (documentation/separable_atom_sae_protocol.md, section 4):
#* recovery of planted rank-1 spike-and-wave events by format 5 (f5) and the separable arm (s4ext),
#* at amplitudes 0 (event-free control), 3 and 6, seeds 0 and 1, 5 epochs. Stage 1a uses AR(1)
#* background with artifacts (no corpus); stage 1b uses real segments of non-epilepsy patients from
#* the campaign cache (GPU). Arm names: {1a|1b}_{f5|s4ext}_a{0|3|6}_s{seed}. After each stage,
#* src/separable_synthetic.py summarize applies the fixed rules (synthetic_summary.csv,
#* synthetic_decision.json). Stage 1b runs only if stage 1a lets it (or FORCE_1B=1).
#*
#* Run inside the HPC container from code/ (stage 1b on a GPU node):
#*   M=<40 s audit> SMOKE_DIR=<its cache_smoke_P512_* dir> nohup bash run_separable_synthetic.sh > synthetic.log 2>&1 &
#*   STAGE=1a bash run_separable_synthetic.sh                  # stage 1a only, no corpus
#*   STAGE=1b ROOT=<root with stage 1a> M=... SMOKE_DIR=... bash run_separable_synthetic.sh
#* Environment: STAGE=all|1a|1b, ROOT, EPOCHS=5, DEVICE=auto (1a), NUM_WORKERS=4, M, SMOKE_DIR or
#* SIGNAL_CACHE_DIR, DATA_DIR, FORCE_1B=0, PRE_EMPHASIS=none|diff (10.11), INIT=patches|random (10.12),
#* RESEED=0|1 (stage 1a only, 10.13), SEEDS="0 1" ("2 3" for the replication of 10.13).
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,17p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
EPOCHS="${EPOCHS:-5}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE_1B="${FORCE_1B:-0}"
PRE_EMPHASIS="${PRE_EMPHASIS:-none}"
INIT="${INIT:-patches}"
RESEED="${RESEED:-0}"
SEEDS="${SEEDS:-0 1}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/separable_synthetic/gate1_$(date +%Y%m%d_%H%M%S)}"
[[ "$STAGE" =~ ^(all|1a|1b)$ ]] || { echo "ERROR: STAGE must be all, 1a or 1b" >&2; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad EPOCHS or NUM_WORKERS" >&2; exit 2; }
[[ "$PRE_EMPHASIS" =~ ^(none|diff)$ ]] || { echo "ERROR: PRE_EMPHASIS must be none or diff" >&2; exit 2; }
[[ "$INIT" =~ ^(patches|random)$ ]] || { echo "ERROR: INIT must be patches or random" >&2; exit 2; }
[[ "$RESEED" =~ ^[01]$ ]] || { echo "ERROR: RESEED must be 0 or 1" >&2; exit 2; }
[[ "$SEEDS" =~ ^[0-9]+( [0-9]+)*$ ]] || { echo "ERROR: SEEDS must be seeds separated by spaces" >&2; exit 2; }
cd -- "$SCRIPT_DIR"
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
# One arm setting per root: summarize, reread and compare read every arm of a root together.
marker="$ROOT/arm_settings.txt"
settings="PRE_EMPHASIS=$PRE_EMPHASIS INIT=$INIT RESEED=$RESEED"
if [[ -f "$marker" ]]; then
  [[ "$(cat "$marker")" == "$settings" ]] || { echo "ERROR: $ROOT holds $(cat "$marker") arms; use a new ROOT" >&2; exit 2; }
elif compgen -G "$ROOT/1?_*/synthetic_readout.json" > /dev/null && [[ "$settings" != "PRE_EMPHASIS=none INIT=patches RESEED=0" ]]; then
  echo "ERROR: $ROOT holds arms of PRE_EMPHASIS=none INIT=patches RESEED=0; use a new ROOT" >&2
  exit 2
fi
echo "$settings" > "$marker"
STAMP=$(date +%Y%m%d_%H%M%S)
failures="$ROOT/failures_$STAMP.txt"
: > "$failures"
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree
echo "$STAMP STAGE=$STAGE EPOCHS=$EPOCHS DEVICE=$DEVICE NUM_WORKERS=$NUM_WORKERS PRE_EMPHASIS=$PRE_EMPHASIS INIT=$INIT RESEED=$RESEED SEEDS=$SEEDS" >> "$ROOT/settings.txt"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_separable_synthetic.py tests/test_separable_sae.py 2>&1 | tee -a "$ROOT/tests.log"

run_stage() {
  local stage=$1
  shift
  for seed in $SEEDS; do
    for amplitude in 0 3 6; do
      for arm in f5 s4ext; do
        local name="${stage}_${arm}_a${amplitude}_s${seed}"
        if [[ -f "$ROOT/$name/synthetic_readout.json" ]]; then
          echo "$(date '+%F %T') skip  $name (finished)"
          continue
        fi
        if [[ -e "$ROOT/$name" ]]; then
          echo "ERROR: $ROOT/$name exists without results; move it away to rerun" | tee -a "$failures"
          continue
        fi
        echo "$(date '+%F %T') start $name"
        if ! python -u src/separable_synthetic.py run --output "$ROOT/$name" --epochs "$EPOCHS" \
            --num-workers "$NUM_WORKERS" --pre-emphasis "$PRE_EMPHASIS" --init "$INIT" --reseed "$RESEED" "$@" > "$ROOT/$name.log" 2>&1; then
          echo "FAILED $name" | tee -a "$failures"
          continue
        fi
        echo "$(date '+%F %T') done  $name"
      done
    done
  done
  python -u src/separable_synthetic.py summarize --root "$ROOT" 2>&1 | tee "$ROOT/summary_$stage.log"
}

if [[ "$STAGE" != 1b ]]; then
  run_stage 1a --device "$DEVICE"
  outcome=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['1a']['outcome'])" "$ROOT/synthetic_decision.json")
  if [[ "$STAGE" == all && "$outcome" != proceed_to_1b && "$FORCE_1B" != 1 ]]; then
    echo "Stage 1a outcome: $outcome. Stage 1b not started (set FORCE_1B=1 to run it anyway)."
    exit 0
  fi
fi
if [[ "$STAGE" != 1a ]]; then
  M="${M:?Set M to the 40 s offset audit directory}"
  DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
  if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
    [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
    SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
  fi
  for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
  done
  echo "$STAMP M=$M SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR DATA_DIR=$DATA_DIR" >> "$ROOT/settings.txt"
  run_stage 1b --manifests "$(cd -- "$M" && pwd)" --data-dir "$(cd -- "$DATA_DIR" && pwd)" \
    --signal-cache-dir "$(cd -- "$SIGNAL_CACHE_DIR" && pwd)"
fi
