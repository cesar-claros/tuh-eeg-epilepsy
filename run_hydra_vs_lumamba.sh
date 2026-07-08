#!/usr/bin/env bash
#*----------------------------------------------------------------------------*
#* HYDRA side of the same-windows LuMamba-vs-HYDRA comparison.
#*
#* For each window length x seed, HYDRA reuses the EXACT windows LuMamba exported
#* ($MANIFEST_ROOT/w<ws>_s<seed>/windows_{train,val,test}.csv), fits the classifier on TRAIN
#* only (no train+val merge), and calibrates the subject decision threshold on VAL (max
#* balanced accuracy) before scoring test. Bipolar montage + 60 Hz notch to match LuMamba's
#* preprocessing. Each run writes performance.csv under $OUT_ROOT/w<ws>_s<seed>/.
#*
#* Run the LuMamba sweep FIRST (it produces the manifests). EDIT DATA_DIR + MANIFEST_ROOT, then
#* from the code/ repo root:
#*   nohup bash run_hydra_vs_lumamba.sh > hydra_vs_lumamba.log 2>&1 &
#* Override inline, e.g.:  WINDOWS="30 60" SEEDS="0 1 2" bash run_hydra_vs_lumamba.sh
#*----------------------------------------------------------------------------*
set -euo pipefail

DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data/v3.0.0}"  # <-- EDIT: has 00_epilepsy/ 01_no_epilepsy/
MANIFEST_ROOT="${MANIFEST_ROOT:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/BioFoundation/manifests}"  # <-- LuMamba manifests
OUT_ROOT="${OUT_ROOT:-logs/hydra_vs_lumamba}"

WINDOWS="${WINDOWS:-15 30 45 60}"
SEEDS="${SEEDS:-0 1 2 3 4}"

cd "$(dirname "$0")"
[ -d "$DATA_DIR" ] || { echo "ERROR: DATA_DIR not found: $DATA_DIR (edit run_hydra_vs_lumamba.sh)"; exit 1; }

echo "DATA_DIR=$DATA_DIR | MANIFEST_ROOT=$MANIFEST_ROOT | OUT_ROOT=$OUT_ROOT"
echo "windows=[$WINDOWS] seeds=[$SEEDS]"

for WS in $WINDOWS; do
    for SEED in $SEEDS; do
        MAN="$MANIFEST_ROOT/w${WS}_s${SEED}"
        OUT="$OUT_ROOT/w${WS}_s${SEED}"
        if [ ! -f "$MAN/windows_test.csv" ]; then
            echo "!! missing $MAN/windows_test.csv (run the LuMamba sweep first); skipping"
            continue
        fi
        if [ -f "$OUT/performance.csv" ]; then
            echo "[skip] $OUT/performance.csv already exists"
            continue
        fi
        echo "==================== HYDRA w=${WS}s seed=${SEED} ===================="
        # Same windows (from CSVs), bipolar + 60 Hz notch to match LuMamba, train-only fit with
        # val threshold calibration. data.seed only seeds HYDRA's random kernels (the split is
        # fixed by the CSVs), set = seed for reproducibility.
        python src/train.py \
            data.data_dir="$DATA_DIR" \
            data.lazy_loading=true \
            data.signal_mode=bipolar \
            data.notch_freqs="[60]" \
            data.windows_train_csv="$MAN/windows_train.csv" \
            data.windows_val_csv="$MAN/windows_val.csv" \
            data.windows_test_csv="$MAN/windows_test.csv" \
            data.seed="$SEED" \
            trainer.merge_train_val=false \
            trainer.calibrate_threshold=true \
            hydra.run.dir="$OUT"
    done
done

echo "HYDRA runs done. Per-run metrics are in $OUT_ROOT/w<ws>_s<seed>/performance.csv"
echo "Aggregated subject-level test table:"
python src/aggregate_performance.py --runs_root "$OUT_ROOT" --split test --level subject \
    --metrics balanced_accuracy roc_auc sensitivity specificity || true
