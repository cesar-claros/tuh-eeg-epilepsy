#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* ShapeConv SAE, Phase 2 of documentation/shapeconv_sae_experimental_proposal.md:
#* the EEG development screen. One frozen manifest set (train/val/test windows CSVs
#* from a reference run with the SAME data settings), dictionary fitting on training
#* subjects only, a fixed compute budget, and seven arms at K=64, L=80:
#*   N05 none/0.5 (control)   D05 D10 D20 diff/{0.5,1,2}   A05 A10 A20 diff_ar(16)/{0.5,1,2}
#* Every arm: 256 Hz, TCP bipolar, 1-45 Hz band-pass, the repair flags below applied
#* identically, extraction thresholds calibrated on NEGATIVE TRAINING windows at
#* CALIBRATE_FA activations per atom per usable channel-minute (Phase 1 synthetic
#* setting: 0.1), then full-window diagnostics on the VALIDATION split
#* (src/sae_diagnostics.py). Test windows are never encoded here. No classifier is
#* fitted here (Phase 3).
#*
#* Manifests: src/dump_windows.py writes windows_{train,val,test}.csv plus
#* manifest_provenance.json from the SAME data settings without loading signal:
#*   python src/dump_windows.py data.lazy_loading=true data.signal_mode=bipolar \
#*     data.filter_freq=[1,45] data.seed=42 output_dir=logs/manifests/phase2_seed42
#* (any train run with these settings writes the same three files). The repair flags
#* filter the plan, so the dump and every arm must use the same INTERPOLATE_BAD /
#* DROP_BAD_SEGMENTS. Reusing manifests needs data.lazy_loading=true; every Python
#* entry point asserts that the three plans are non-empty and subject-disjoint before
#* fitting. EDIT DATA_DIR and MANIFEST_DIR.
#*
#* Outputs: one root per invocation (OUT_ROOT, default logs/sae_phase2/<date>_<time>),
#* never reused: an arm directory that already holds files stops the script. Per arm,
#* train_sae.log, diagnostics.log, sae_state.pt and the arrays, the resolved Hydra
#* config under .hydra/, and diag_val/ with the diagnostics. The root holds sha.txt,
#* dirty.diff, env.txt (library versions), threads.txt, gpu.txt, settings.txt, the
#* copied manifests, failures.txt and summary.txt. The exit status is nonzero when any
#* arm failed, so a wrapper can gate on it.
#*
#* Run inside the HPC container, from the code/ repo root (GPU node):
#*   nohup bash run_sae_phase2.sh > sae_phase2.log 2>&1 &
#* Resource probe first (one epoch, both pre-emphasis families, separate root):
#*   ARMS="D10 A10" EPOCHS=1 OUT_ROOT=logs/sae_phase2/probe_$(date +%Y%m%d_%H%M%S) bash run_sae_phase2.sh
#*----------------------------------------------------------------------------*
set -euo pipefail

DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"   # <-- EDIT
MANIFEST_DIR="${MANIFEST_DIR:-}"                                                             # <-- EDIT: dir with windows_{train,val,test}.csv
OUT_ROOT="${OUT_ROOT:-logs/sae_phase2/$(date +%Y%m%d_%H%M%S)}"
ARMS="${ARMS:-N05 D05 D10 D20 A05 A10 A20}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"            # dictionary seed (feature.random_state); the split seed is fixed by the manifests
N_ATOMS="${N_ATOMS:-64}"
ATOM_LEN="${ATOM_LEN:-80}"
AR_ORDER="${AR_ORDER:-16}"
CALIBRATE_FA="${CALIBRATE_FA:-0.1}"
CROP_BATCH="${CROP_BATCH:-1024}"
# Signal repair, chosen ONCE and applied to every arm (record the choice with the results).
INTERPOLATE_BAD="${INTERPOLATE_BAD:-false}"
DROP_BAD_SEGMENTS="${DROP_BAD_SEGMENTS:-false}"
DEVICE="${DEVICE:-auto}"

cd "$(dirname "$0")"

arm_flags() {  # pre-emphasis and lambda of an arm; empty output = unknown arm
  case "$1" in
    N05) echo "feature.spec.pre_emphasis=none feature.train_spec.lam=0.5" ;;
    D05) echo "feature.spec.pre_emphasis=diff feature.train_spec.lam=0.5" ;;
    D10) echo "feature.spec.pre_emphasis=diff feature.train_spec.lam=1.0" ;;
    D20) echo "feature.spec.pre_emphasis=diff feature.train_spec.lam=2.0" ;;
    A05) echo "feature.spec.pre_emphasis=diff_ar feature.train_spec.lam=0.5" ;;
    A10) echo "feature.spec.pre_emphasis=diff_ar feature.train_spec.lam=1.0" ;;
    A20) echo "feature.spec.pre_emphasis=diff_ar feature.train_spec.lam=2.0" ;;
    *) echo "" ;;
  esac
}

# ---- preflight: inputs, arm names, fresh output directories ----------------------
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
for split in train val test; do
  [ -n "$MANIFEST_DIR" ] && [ -f "$MANIFEST_DIR/windows_$split.csv" ] || {
    echo "ERROR: MANIFEST_DIR must hold windows_{train,val,test}.csv (got '$MANIFEST_DIR')"; exit 1; }
done
for arm in $ARMS; do
  [ -n "$(arm_flags "$arm")" ] || { echo "ERROR: unknown arm '$arm' (known: N05 D05 D10 D20 A05 A10 A20)"; exit 1; }
  if [ -d "$OUT_ROOT/$arm" ] && [ -n "$(ls -A "$OUT_ROOT/$arm")" ]; then
    echo "ERROR: $OUT_ROOT/$arm already holds files; choose another OUT_ROOT"; exit 1
  fi
done
mkdir -p "$OUT_ROOT"

# ---- archive the code state, the environment and the inputs ----------------------
git rev-parse HEAD > "$OUT_ROOT/sha.txt"
git diff > "$OUT_ROOT/dirty.diff"            # empty file = clean working tree
git status --short > "$OUT_ROOT/status.txt"
python - > "$OUT_ROOT/env.txt" <<'EOF'
import platform, sklearn, torch, numpy, mne, polars
print(f"python {platform.python_version()}")
print(f"torch {torch.__version__} threads {torch.get_num_threads()} cuda {torch.cuda.is_available()}")
print(f"numpy {numpy.__version__} sklearn {sklearn.__version__} mne {mne.__version__} polars {polars.__version__}")
print(platform.platform())
EOF
cp "$MANIFEST_DIR"/windows_{train,val,test}.csv "$OUT_ROOT"/
env | grep -E "^(OMP|MKL|TORCH|CUDA)_" | sort > "$OUT_ROOT/threads.txt" || true
nvidia-smi -L > "$OUT_ROOT/gpu.txt" 2>/dev/null || echo "no GPU visible" > "$OUT_ROOT/gpu.txt"
echo "ARMS=$ARMS EPOCHS=$EPOCHS SEED=$SEED K=$N_ATOMS L=$ATOM_LEN AR=$AR_ORDER FA=$CALIBRATE_FA CROP_BATCH=$CROP_BATCH INTERPOLATE_BAD=$INTERPOLATE_BAD DROP_BAD_SEGMENTS=$DROP_BAD_SEGMENTS MANIFEST_DIR=$MANIFEST_DIR" | tee "$OUT_ROOT/settings.txt"
failures="$OUT_ROOT/failures.txt"
: > "$failures"

common=(
  "data.data_dir=$DATA_DIR" data.lazy_loading=true
  "data.windows_train_csv=$MANIFEST_DIR/windows_train.csv"
  "data.windows_val_csv=$MANIFEST_DIR/windows_val.csv"
  "data.windows_test_csv=$MANIFEST_DIR/windows_test.csv"
  data.signal_mode=bipolar data.target_sfreq=256 'data.filter_freq=[1,45]'
  "data.interpolate_bad_channels=$INTERPOLATE_BAD" "data.drop_bad_segments=$DROP_BAD_SEGMENTS"
  feature.spec.mode=shrink "feature.spec.n_atoms=$N_ATOMS" "feature.spec.atom_len=$ATOM_LEN"
  "feature.spec.ar_order=$AR_ORDER" "feature.calibrate_fa=$CALIBRATE_FA" feature.calibrate_label=0
  "feature.train_spec.epochs=$EPOCHS" "feature.train_spec.crop_batch=$CROP_BATCH"
  "feature.random_state=$SEED" "feature.device=$DEVICE"
)

# ---- the arms: dictionary fit + calibration, then validation diagnostics ----------
for arm in $ARMS; do
  run="$OUT_ROOT/$arm"
  mkdir -p "$run"
  # shellcheck disable=SC2046
  if ! python src/train_sae.py "${common[@]}" $(arm_flags "$arm") "output_dir=$run" > "$run/train_sae.log" 2>&1; then
    echo "FAILED train_sae $arm" | tee -a "$failures"; continue
  fi
  # shellcheck disable=SC2046
  if ! python src/sae_diagnostics.py "${common[@]}" $(arm_flags "$arm") \
       "feature.pretrained=$run/sae_state.pt" split=val "output_dir=$run/diag_val" > "$run/diagnostics.log" 2>&1; then
    echo "FAILED diagnostics $arm" | tee -a "$failures"
  fi
done

# ---- summary: one block per arm ---------------------------------------------------
summary="$OUT_ROOT/summary.txt"
: > "$summary"
for arm in $ARMS; do
  run="$OUT_ROOT/$arm"
  {
    echo "== $arm"
    tail -1 "$run/sae_training.csv" 2>/dev/null | sed 's/^/  last epoch (csv): /' || true
    grep -E "Fitted AR|Calibrated per-atom thresholds" "$run/train_sae.log" | tail -2 || true
    grep -E "SAE diagnostics on" "$run/diagnostics.log" | tail -1 || true
  } >> "$summary"
done
cat "$summary"
echo
if [ -s "$failures" ]; then
  echo "Failures (see the arm logs under $OUT_ROOT):"; cat "$failures"; exit 1
fi
echo "All arms done. Outputs under $OUT_ROOT"
