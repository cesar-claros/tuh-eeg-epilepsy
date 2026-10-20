#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* Waveform SAE, first EEG screen (documentation/waveform_sae_implementation.md, section 6):
#* three dictionary fits on the SAME interpolated manifests, run one after the other.
#*   P32_unit   pool_len=32,  atom_norm=unit   (dense tiling)
#*   P32_free   pool_len=32,  atom_norm=free   (the loss may shrink atoms: the gain check)
#*   P256_unit  pool_len=256, atom_norm=unit   (event regime, one activation per atom per second)
#* K=64, L=80, R=1, 256 Hz, TCP bipolar, 1-45 Hz, data.interpolate_bad_channels=true. No
#* classifier, no test windows loaded. One output root per invocation (never reused).
#* Per arm: <name>.log, sae_state.pt, sae_atoms*.npy, sae_training.csv, sae_atom_stats.csv,
#* sae_channels.txt, windows_*.csv, .hydra/. The root holds sha.txt, dirty.diff, settings.txt,
#* failures.txt (empty = all arms ran) and done.txt.
#*
#* Run inside the HPC container, from the code/ repo root (GPU node); EDIT DATA_DIR and M or
#* pass them. The probe of 2026-10-19 took about one minute per epoch (loader bound), so
#* EPOCHS=20 is about 20 minutes per arm.
#*   nohup bash run_wsae_arms.sh > wsae_arms.log 2>&1 &
#*   EPOCHS=10 ARMS="P32_unit" nohup bash run_wsae_arms.sh > wsae_arms.log 2>&1 &
#*----------------------------------------------------------------------------*
set -euo pipefail

DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"   # <-- EDIT
M="${M:-logs/manifests/wsae_seed42_interp}"                                                   # <-- EDIT: dir with windows_{train,val,test}.csv
ROOT="${ROOT:-logs/waveform_sae/$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-P32_unit P32_free P256_unit}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"            # dictionary seed (feature.random_state); the split is fixed by the manifests
NUM_WORKERS="${NUM_WORKERS:-4}"   # the probe node offered 5 CPUs; 7 workers oversubscribe it
DEVICE="${DEVICE:-cuda}"

cd "$(dirname "$0")"

arm_flags() {  # empty output = unknown arm
  case "$1" in
    P32_unit)  echo "feature.spec.pool_len=32" ;;
    P32_free)  echo "feature.spec.pool_len=32 feature.spec.atom_norm=free" ;;
    P256_unit) echo "feature.spec.pool_len=256" ;;
    P256_free) echo "feature.spec.pool_len=256 feature.spec.atom_norm=free" ;;
    *) echo "" ;;
  esac
}

# ---- preflight ---------------------------------------------------------------------
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
for split in train val test; do
  [ -f "$M/windows_$split.csv" ] || { echo "ERROR: M must hold windows_{train,val,test}.csv (got '$M')"; exit 1; }
done
for arm in $ARMS; do
  [ -n "$(arm_flags "$arm")" ] || { echo "ERROR: unknown arm '$arm' (known: P32_unit P32_free P256_unit P256_free)"; exit 1; }
  if [ -d "$ROOT/$arm" ] && [ -n "$(ls -A "$ROOT/$arm")" ]; then
    echo "ERROR: $ROOT/$arm already holds files; choose another ROOT"; exit 1
  fi
done
mkdir -p "$ROOT"

# ---- archive the code state and the settings ---------------------------------------
git rev-parse HEAD > "$ROOT/sha.txt"
git diff > "$ROOT/dirty.diff"            # empty file = clean working tree
cp "$M"/windows_{train,val,test}.csv "$ROOT"/
echo "ARMS=$ARMS EPOCHS=$EPOCHS SEED=$SEED NUM_WORKERS=$NUM_WORKERS DEVICE=$DEVICE M=$M" > "$ROOT/settings.txt"
failures="$ROOT/failures.txt"
: > "$failures"

common=(
  feature=waveform_sae "feature.train_spec.epochs=$EPOCHS" "feature.random_state=$SEED" "feature.device=$DEVICE"
  "data.data_dir=$DATA_DIR" data.lazy_loading=true data.signal_mode=bipolar data.target_sfreq=256
  'data.filter_freq=[1,45]' data.interpolate_bad_channels=true "data.num_workers=$NUM_WORKERS"
  "data.windows_train_csv=$M/windows_train.csv" "data.windows_val_csv=$M/windows_val.csv"
  "data.windows_test_csv=$M/windows_test.csv"
)

# ---- the arms, one after the other -------------------------------------------------
for arm in $ARMS; do
  run="$ROOT/$arm"
  mkdir -p "$run"
  echo "$(date '+%F %T') start $arm"
  # shellcheck disable=SC2046
  if ! python src/train_sae.py "${common[@]}" $(arm_flags "$arm") "output_dir=$run" > "$run/$arm.log" 2>&1; then
    echo "FAILED $arm" | tee -a "$failures"; continue
  fi
  echo "$(date '+%F %T') done  $arm"
  grep "Waveform SAE epoch" "$run/$arm.log" | tail -1
done
echo "done $(date '+%F %T')" > "$ROOT/done.txt"
[ -s "$failures" ] && { echo "some arms failed:"; cat "$failures"; exit 1; }
echo "all arms done under $ROOT"
