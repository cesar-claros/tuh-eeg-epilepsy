#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* ShapeConv SAE, Phase 3 of documentation/shapeconv_sae_experimental_proposal.md:
#* patient-level development readout with controls (src/readout.py), test untouched.
#*
#* Arms, all on the SAME manifests, data settings and readout protocol (train-only
#* fits, L2 logistic regression over C in {0.1, 1, 10}, inverse window-count subject
#* weights, subject-mean pooling, selection by validation subject AUROC, ties toward
#* smaller C):
#*   the seven dictionaries of a Phase 2 screen root (DICT_ROOT: N05 D05 D10 D20 A05 A10 A20),
#*   hydra      HYDRA random kernels, feature seed HYDRA_SEED,
#*   spectral   log band power per channel (delta, theta, alpha, beta, low gamma),
#*   kmeans_{none,diff,diff_ar}  the initialized dictionary (epochs 0) of each domain with the
#*              same training levers, gate, band limit and calibration as the screen.
#* Then the shortlist rule of the proposal: per domain (none, diff, diff_ar) the lambda
#* with the highest validation subject AUROC maximized over C, exact ties toward larger
#* lambda then smaller C. The test split is never loaded by any arm.
#*
#* Run inside the HPC container, from the code/ repo root (GPU node), e.g.:
#*   DICT_ROOT=logs/sae_phase2/20260920_170734/20260920_170734 MANIFEST_DIR=logs/manifests/phase2_seed42 \
#*     nohup bash run_sae_phase3.sh > sae_phase3.log 2>&1 &
#*----------------------------------------------------------------------------*
set -euo pipefail

DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"   # <-- EDIT
MANIFEST_DIR="${MANIFEST_DIR:-}"                                                             # <-- EDIT
DICT_ROOT="${DICT_ROOT:-}"                                                                   # <-- EDIT: Phase 2 root with N05 ... A20
OUT_ROOT="${OUT_ROOT:-logs/sae_phase3/$(date +%Y%m%d_%H%M%S)}"
SAE_ARMS="${SAE_ARMS:-N05 D05 D10 D20 A05 A10 A20}"
CONTROLS="${CONTROLS:-hydra spectral kmeans_none kmeans_diff kmeans_diff_ar}"
HYDRA_SEED="${HYDRA_SEED:-42}"
SEED="${SEED:-42}"
N_ATOMS="${N_ATOMS:-64}"
ATOM_LEN="${ATOM_LEN:-80}"
AR_ORDER="${AR_ORDER:-16}"
CALIBRATE_FA="${CALIBRATE_FA:-0.1}"
CROP_ENERGY_RATIO="${CROP_ENERGY_RATIO:-10}"
AR_FIT="${AR_FIT:-row_normalized}"
ATOM_LOWPASS_HZ="${ATOM_LOWPASS_HZ:-45}"
INTERPOLATE_BAD="${INTERPOLATE_BAD:-false}"
DROP_BAD_SEGMENTS="${DROP_BAD_SEGMENTS:-false}"
DEVICE="${DEVICE:-auto}"

cd "$(dirname "$0")"

pre_emphasis_of() {  # domain of a screen arm or a kmeans control; empty = unknown
  case "$1" in
    N05|kmeans_none) echo none ;;
    D05|D10|D20|kmeans_diff) echo diff ;;
    A05|A10|A20|kmeans_diff_ar) echo diff_ar ;;
    *) echo "" ;;
  esac
}

# ---- preflight ---------------------------------------------------------------------
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1; }
for split in train val test; do
  [ -n "$MANIFEST_DIR" ] && [ -f "$MANIFEST_DIR/windows_$split.csv" ] || {
    echo "ERROR: MANIFEST_DIR must hold windows_{train,val,test}.csv (got '$MANIFEST_DIR')"; exit 1; }
done
for arm in $SAE_ARMS; do
  [ -n "$(pre_emphasis_of "$arm")" ] || { echo "ERROR: unknown SAE arm '$arm'"; exit 1; }
  [ -f "$DICT_ROOT/$arm/sae_state.pt" ] || { echo "ERROR: $DICT_ROOT/$arm/sae_state.pt not found"; exit 1; }
done
for arm in $CONTROLS; do
  case "$arm" in hydra|spectral|kmeans_none|kmeans_diff|kmeans_diff_ar) ;; *) echo "ERROR: unknown control '$arm'"; exit 1 ;; esac
done
[ -d "$OUT_ROOT" ] && [ -n "$(ls -A "$OUT_ROOT")" ] && { echo "ERROR: $OUT_ROOT already holds files"; exit 1; }
mkdir -p "$OUT_ROOT"

# ---- archive -----------------------------------------------------------------------
git rev-parse HEAD > "$OUT_ROOT/sha.txt"
git diff > "$OUT_ROOT/dirty.diff"
git status --short > "$OUT_ROOT/status.txt"
python - > "$OUT_ROOT/env.txt" <<'EOF'
import platform, sklearn, torch, numpy, mne, polars
print(f"python {platform.python_version()}")
print(f"torch {torch.__version__} threads {torch.get_num_threads()} cuda {torch.cuda.is_available()}")
print(f"numpy {numpy.__version__} sklearn {sklearn.__version__} mne {mne.__version__} polars {polars.__version__}")
print(platform.platform())
EOF
cp "$MANIFEST_DIR"/windows_{train,val,test}.csv "$OUT_ROOT"/
[ -f "$DICT_ROOT/settings.txt" ] && cp "$DICT_ROOT/settings.txt" "$OUT_ROOT/dictionaries_settings.txt"
echo "DICT_ROOT=$DICT_ROOT SAE_ARMS=$SAE_ARMS CONTROLS=$CONTROLS HYDRA_SEED=$HYDRA_SEED SEED=$SEED K=$N_ATOMS L=$ATOM_LEN AR=$AR_ORDER FA=$CALIBRATE_FA CROP_ENERGY_RATIO=$CROP_ENERGY_RATIO AR_FIT=$AR_FIT ATOM_LOWPASS_HZ=$ATOM_LOWPASS_HZ INTERPOLATE_BAD=$INTERPOLATE_BAD DROP_BAD_SEGMENTS=$DROP_BAD_SEGMENTS MANIFEST_DIR=$MANIFEST_DIR" | tee "$OUT_ROOT/settings.txt"
failures="$OUT_ROOT/failures.txt"
: > "$failures"

common=(
  "data.data_dir=$DATA_DIR" data.lazy_loading=true
  "data.windows_train_csv=$MANIFEST_DIR/windows_train.csv"
  "data.windows_val_csv=$MANIFEST_DIR/windows_val.csv"
  "data.windows_test_csv=$MANIFEST_DIR/windows_test.csv"
  data.signal_mode=bipolar data.target_sfreq=256 'data.filter_freq=[1,45]'
  "data.interpolate_bad_channels=$INTERPOLATE_BAD" "data.drop_bad_segments=$DROP_BAD_SEGMENTS"
)
sae_common=(
  feature=shapeconv_sae feature.spec.mode=shrink "feature.spec.n_atoms=$N_ATOMS" "feature.spec.atom_len=$ATOM_LEN"
  "feature.spec.ar_order=$AR_ORDER" "feature.spec.ar_fit=$AR_FIT" "feature.device=$DEVICE"
)

run_arm() {  # name, extra hydra overrides...
  local name="$1"; shift
  local run="$OUT_ROOT/$name"
  mkdir -p "$run"
  if ! python src/readout.py "${common[@]}" "arm=$name" "$@" "output_dir=$run" > "$run/readout.log" 2>&1; then
    echo "FAILED readout $name" | tee -a "$failures"
  fi
}

# ---- the learned dictionaries -------------------------------------------------------
for arm in $SAE_ARMS; do
  run_arm "$arm" features=sae "${sae_common[@]}" "feature.spec.pre_emphasis=$(pre_emphasis_of "$arm")" \
    "feature.pretrained=$DICT_ROOT/$arm/sae_state.pt"
done

# ---- the controls -------------------------------------------------------------------
for arm in $CONTROLS; do
  case "$arm" in
    hydra)    run_arm hydra features=hydra feature=hydra_transformer "feature.random_state=$HYDRA_SEED" "feature.device=$DEVICE" ;;
    spectral) run_arm spectral features=spectral feature=spectral "feature.device=$DEVICE" ;;
    kmeans_*) run_arm "$arm" features=kmeans "${sae_common[@]}" "feature.spec.pre_emphasis=$(pre_emphasis_of "$arm")" \
                feature.train_spec.epochs=0 feature.train_spec.lam=0.5 "feature.random_state=$SEED" \
                "feature.train_spec.max_crop_energy_ratio=$CROP_ENERGY_RATIO" \
                "feature.train_spec.atom_lowpass_hz=$ATOM_LOWPASS_HZ" feature.train_spec.sfreq=256 \
                "feature.calibrate_fa=$CALIBRATE_FA" feature.calibrate_label=0 ;;
  esac
done

# ---- summary and shortlist ----------------------------------------------------------
python - "$OUT_ROOT" "$SAE_ARMS $CONTROLS" <<'EOF' | tee "$OUT_ROOT/summary.txt"
import json, sys
from pathlib import Path
import polars as pl
root, arms = Path(sys.argv[1]), sys.argv[2].split()
print(f"{'arm':14s} {'C=0.1':>7s} {'C=1':>7s} {'C=10':>7s}   selected C  val subj AUROC (train)  val window AUROC  dim")
best = {}
for arm in arms:
    grid_path = root / arm / "readout_grid.csv"
    if not grid_path.exists():
        print(f"{arm:14s} (missing)"); continue
    grid = pl.read_csv(grid_path); summary = json.loads((root / arm / "readout_summary.json").read_text())
    by_c = {float(r["C"]): r for r in grid.to_dicts()}
    sel = summary["selected"]
    print(f"{arm:14s} " + " ".join(f"{by_c[c]['val_subject_auroc']:7.3f}" for c in (0.1, 1.0, 10.0))
          + f"   {summary['selected_C']:>10g}  {sel['val_subject_auroc']:.3f} ({sel['train_subject_auroc']:.3f})"
          + f"          {sel['val_window_auroc']:.3f}        {summary['feature_dim']}")
    best[arm] = (sel["val_subject_auroc"], summary["selected_C"])
domains = {"none": ["N05"], "diff": ["D05", "D10", "D20"], "diff_ar": ["A05", "A10", "A20"]}
lam = {"N05": 0.5, "D05": 0.5, "D10": 1.0, "D20": 2.0, "A05": 0.5, "A10": 1.0, "A20": 2.0}
print("\nshortlist per domain (max validation subject AUROC over C; exact ties toward larger lambda, then smaller C):")
for domain, members in domains.items():
    present = [m for m in members if m in best]
    if not present:
        continue
    chosen = max(present, key=lambda m: (round(best[m][0], 12), lam[m], -best[m][1]))
    print(f"  {domain:8s} -> {chosen} (lambda {lam[chosen]:g}, C {best[chosen][1]:g}, val subject AUROC {best[chosen][0]:.3f})")
EOF
echo
if [ -s "$failures" ]; then echo "Failures (see the arm logs under $OUT_ROOT):"; cat "$failures"; exit 1; fi
echo "All arms done. Outputs under $OUT_ROOT (test split never loaded)"
