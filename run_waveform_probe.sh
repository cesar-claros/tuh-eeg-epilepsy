#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* First format-5 probe (documentation/waveform_format5_probe_protocol.md):
#* atom competition per pooling window (format 5, this checkout) against the format-4 rule
#* (R windows per atom and segment, commit BASELINE in a detached worktree), both at R=1 with
#* the campaign geometry L=256, P=512, K=64: at most 60 against 64 coefficients per segment.
#*   f5_r1_s{seed}, f4_r1_s{seed}      trained arms, EPOCHS epochs, campaign learning rate
#*   f5_r1_frozen_s0, f4_r1_frozen_s0  frozen controls: lr=0 for one epoch (initial dictionary)
#* A seed fixes the initial atoms and the patient draws, so the two rules are paired per seed.
#* Every arm uses make_config of src/run_patient_waveform_sweep.py (patient sampling, seed-42
#* split, full validation each epoch) and reuses the campaign signal cache; an arm refuses to
#* rebuild the cache. Test signals are never loaded. The decision rules are applied at the end
#* by `python src/waveform_probe.py summarize` (probe_summary.csv, probe_decision.json).
#*
#* Run inside the HPC container from code/ on a GPU node:
#*   nohup bash run_waveform_probe.sh /path/to/fresh_uncapped_audit > waveform_probe.log 2>&1 &
#*   EPOCHS=3 SEEDS="0" bash run_waveform_probe.sh "$AUDIT_ALL"      # shorter, one seed
#* Environment: M (or argument 1), SMOKE_DIR or SIGNAL_CACHE_DIR, DATA_DIR, EPOCHS=5, SEEDS="0 1",
#* BASELINE=2ca8a66, NUM_WORKERS=4, ROOT (new directory).
#*----------------------------------------------------------------------------*
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '3,19p' "$0" | sed 's/^#\* \{0,1\}//'
  exit 0
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M="${1:-${M:-$SCRIPT_DIR/logs/audits/wsae_all_segments_fresh42_20261023_210759}}"
SMOKE_DIR="${SMOKE_DIR:-$SCRIPT_DIR/logs/waveform_patient/cache_smoke_P512_7cqc27GG}"
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
EPOCHS="${EPOCHS:-5}"
SEEDS="${SEEDS:-0 1}"
BASELINE="${BASELINE:-2ca8a66}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROOT="${ROOT:-$SCRIPT_DIR/logs/waveform_probe/f5_vs_f4_$(date +%Y%m%d_%H%M%S)}"
if [[ -z ${SIGNAL_CACHE_DIR:-} ]]; then
  [[ -f "$SMOKE_DIR/signal_cache_path.txt" ]] || { echo "ERROR: set SIGNAL_CACHE_DIR or SMOKE_DIR" >&2; exit 2; }
  SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
fi

# ---- preflight ---------------------------------------------------------------------
for dir in "$M" "$DATA_DIR" "$SIGNAL_CACHE_DIR"; do
  [[ -d "$dir" ]] || { echo "ERROR: directory not found: $dir" >&2; exit 2; }
done
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "ERROR: bad EPOCHS or NUM_WORKERS" >&2; exit 2; }
[[ " $SEEDS " == *" 0 "* ]] || { echo "ERROR: SEEDS must include 0 (the frozen controls use seed 0)" >&2; exit 2; }
cd -- "$SCRIPT_DIR"
grep -q "^CHECKPOINT_FORMAT = 5$" src/models/components/waveform_sae.py || { echo "ERROR: this checkout is not format 5" >&2; exit 2; }
git cat-file -e "${BASELINE}^{commit}" || { echo "ERROR: baseline commit $BASELINE not found; git fetch first" >&2; exit 2; }
M=$(cd -- "$M" && pwd)
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
SIGNAL_CACHE_DIR=$(cd -- "$SIGNAL_CACHE_DIR" && pwd)
mkdir -p -- "$(dirname -- "$ROOT")"
mkdir -- "$ROOT"  # refuse an existing root
ROOT=$(cd -- "$ROOT" && pwd)

# ---- archive the code state, then the baseline worktree -----------------------------
git rev-parse HEAD > "$ROOT/sha.txt"
git diff > "$ROOT/dirty.diff"  # empty file = clean working tree
BASE_DIR="$ROOT/code_$BASELINE"
git worktree add --detach "$BASE_DIR" "$BASELINE" > "$ROOT/worktree.log" 2>&1
cp src/waveform_probe.py "$BASE_DIR/src/waveform_probe.py"  # same driver, format-4 model code
grep -q "^CHECKPOINT_FORMAT = 4$" "$BASE_DIR/src/models/components/waveform_sae.py" || { echo "ERROR: $BASELINE is not format 4" >&2; exit 2; }
echo "M=$M DATA_DIR=$DATA_DIR SIGNAL_CACHE_DIR=$SIGNAL_CACHE_DIR EPOCHS=$EPOCHS SEEDS=$SEEDS BASELINE=$BASELINE NUM_WORKERS=$NUM_WORKERS" > "$ROOT/settings.txt"
failures="$ROOT/failures.txt"
: > "$failures"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HYDRA_FULL_ERROR=1

# The arms run under strict CUDA determinism; format 5 (argsort and scatter over atoms) must pass
# the CUDA determinism regressions first. The probe's own decision logic is checked here too.
python -m pytest -q tests/test_waveform_determinism.py tests/test_waveform_probe.py 2>&1 | tee "$ROOT/tests.log"

run_arm() {  # name, code directory, seed, epochs, extra arguments
  local name=$1 code=$2 seed=$3 epochs=$4
  shift 4
  echo "$(date '+%F %T') start $name"
  if ! (cd -- "$code" && python -u src/waveform_probe.py run --manifests "$M" --data-dir "$DATA_DIR" \
        --signal-cache-dir "$SIGNAL_CACHE_DIR" --output "$ROOT/$name" --seed "$seed" --top-r 1 \
        --epochs "$epochs" --num-workers "$NUM_WORKERS" "$@") > "$ROOT/$name.log" 2>&1; then
    echo "FAILED $name" | tee -a "$failures"
    return 0
  fi
  echo "$(date '+%F %T') done  $name"
}

# Seed 0 and the frozen controls first, so a partial run can already be read.
for seed in $SEEDS; do
  run_arm "f5_r1_s$seed" "$SCRIPT_DIR" "$seed" "$EPOCHS"
  run_arm "f4_r1_s$seed" "$BASE_DIR" "$seed" "$EPOCHS"
  if [[ $seed == 0 ]]; then
    run_arm f5_r1_frozen_s0 "$SCRIPT_DIR" 0 1 --lr 0
    run_arm f4_r1_frozen_s0 "$BASE_DIR" 0 1 --lr 0
  fi
done

python -u src/waveform_probe.py summarize --root "$ROOT" 2>&1 | tee "$ROOT/summary.log"
echo "done $(date '+%F %T')" > "$ROOT/done.txt"
echo "Probe root: $ROOT"
echo "Remove the baseline worktree after reading: git worktree remove --force $BASE_DIR"
[[ -s "$failures" ]] && { echo "some arms failed:"; cat "$failures"; exit 1; }
exit 0
