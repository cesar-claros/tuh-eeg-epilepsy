#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Reconstruction-only sweep of K (atoms), R (atoms kept per pooling window) and the initialization
#* (documentation/waveform_kr_init_sweep_protocol.md). Every arm uses the campaign configuration
#* (L=256, P=512, patient sampling, format 5, exact-ratio resampling) on the 40 s offset audit and
#* reuses its signal cache; arms run one after the other, then src/waveform_kr_sweep.py summarize
#* applies the fixed decision rules (kr_sweep_summary.csv, kr_sweep_decision.json, kr_sweep.png).
#* Arm names: K{K}_R{R}_{patches|random}_s{seed}[_frozen]; frozen arms run one epoch with lr=0.
#* The default order runs the R=1 row and the initialization arms first (about 2.5 h of about 6 h at 10 epochs).
#*
#* Run inside the HPC container from code/ on a GPU node:
#*   M=<40 s audit> SMOKE_DIR=<its cache_smoke_P512_* dir> nohup bash run_waveform_kr_sweep.sh > kr_sweep.log 2>&1 &
#*   ARMS="K16_R1_patches_s0 K32_R1_patches_s0" EPOCHS=10 M=... SMOKE_DIR=... bash run_waveform_kr_sweep.sh
#*   ROOT=<existing sweep root> ... bash run_waveform_kr_sweep.sh   # add arms to a root; finished arms are skipped
#* Environment: M, SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, EPOCHS=10, ARMS, NUM_WORKERS=4, ROOT.
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,15p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M="${M:?Set M to the 40 s offset audit directory}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
EPOCHS="${EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/waveform_kr_sweep/kr_sweep_$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-K16_R1_patches_s0 K32_R1_patches_s0 K64_R1_patches_s0 K128_R1_patches_s0
  K32_R1_random_s0 K64_R1_random_s0 K128_R1_random_s0 K64_R1_patches_s1 K64_R1_random_s1
  K64_R1_patches_s0_frozen K64_R1_random_s0_frozen
  K16_R2_patches_s0 K32_R2_patches_s0 K64_R2_patches_s0 K128_R2_patches_s0
  K16_R4_patches_s0 K32_R4_patches_s0 K64_R4_patches_s0 K128_R4_patches_s0 K64_R4_random_s0}"
if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
  [[ -n ${SMOKE_DIR:-} && -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
  SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
fi
for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
  [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
done
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad EPOCHS or NUM_WORKERS" >&2; exit 2; }
cd -- "$SCRIPT_DIR"
M=$(cd -- "$M" && pwd)
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
SIGNAL_CACHE_DIR=$(cd -- "$SIGNAL_CACHE_DIR" && pwd)
mkdir -p -- "$ROOT"
ROOT=$(cd -- "$ROOT" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
echo "$STAMP M=$M SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR EPOCHS=$EPOCHS NUM_WORKERS=$NUM_WORKERS ARMS=$(tr -s ' \n' ' ' <<< "$ARMS")" >> "$ROOT/settings.txt"
failures="$ROOT/failures_$STAMP.txt"  # one per invocation: an earlier failure does not fail a rerun
: > "$failures"
git rev-parse HEAD >> "$ROOT/sha.txt"
git diff > "$ROOT/dirty_$STAMP.diff"  # empty file = clean tree

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1
python -m pytest -q tests/test_waveform_kr_sweep.py tests/test_waveform_determinism.py \
  "tests/test_patient_waveform.py::test_patient_reconstruction_accepts_more_than_one_atom_per_window" 2>&1 | tee -a "$ROOT/tests.log"

for arm in $ARMS; do
  if [[ -f "$ROOT/$arm/sae_training.csv" ]]; then
    echo "$(date '+%F %T') skip  $arm (finished)"
    continue
  fi
  if [[ -e "$ROOT/$arm" ]]; then
    echo "ERROR: $ROOT/$arm exists without results; move it away to rerun" | tee -a "$failures"
    continue
  fi
  echo "$(date '+%F %T') start $arm"
  if ! python -u src/waveform_kr_sweep.py run --manifests "$M" --data-dir "$DATA_DIR" \
      --signal-cache-dir "$SIGNAL_CACHE_DIR" --output "$ROOT/$arm" --epochs "$EPOCHS" \
      --num-workers "$NUM_WORKERS" > "$ROOT/$arm.log" 2>&1; then
    echo "FAILED $arm" | tee -a "$failures"
    continue
  fi
  echo "$(date '+%F %T') done  $arm"
done

python -u src/waveform_kr_sweep.py summarize --root "$ROOT" 2>&1 | tee "$ROOT/summary.log"
echo "Sweep root: $ROOT"
[[ -s "$failures" ]] && { echo "some arms failed:"; cat "$failures"; exit 1; }
exit 0
