#!/usr/bin/env bash
# Run inside the HPC container on an allocated GPU node.
# Usage: bash run_patient_wsae_smoke.sh /path/to/fresh_uncapped_audit
# Alternatively: M=/path/to/fresh_uncapped_audit bash run_patient_wsae_smoke.sh
# DATA_DIR may override the corpus parent directory below.
# Schedule: 1 pretraining epoch, 2 head epochs, 2 probes, 1 joint epoch.
# Geometry: L=256, P=512, R=1; linear patient classifier.
set -euo pipefail

usage() {
  echo "Usage: bash $0 /path/to/fresh_uncapped_audit"
  echo "Alternatively, set M to the audit directory. DATA_DIR may override the corpus parent."
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

M="${1:-${M:-}}"
if [[ -z "$M" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -d "$M" ]]; then
  echo "ERROR: audit directory not found: $M" >&2
  exit 1
fi
# Resolve the caller's manifest path before changing to the repository directory.
M=$(cd -- "$M" && pwd)
DATA_DIR="${DATA_DIR:-/work/cniel/sw/singularity_containers/tuh-eeg-epilepsy/project/data}"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: corpus parent directory not found: $DATA_DIR" >&2
  exit 1
fi
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for split in train val test; do
  if [[ ! -f "$M/windows_${split}.csv" ]]; then
    echo "ERROR: missing manifest: $M/windows_${split}.csv" >&2
    exit 1
  fi
done

python - "$M" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads((Path(sys.argv[1]) / "audit_provenance.json").read_text())
checks = {
    "complete eligible segment scope": p["scope"] == "all_metadata_eligible_complete_segments",
    "no balancing or caps": p["balance_per_subject"] is False and p["max_windows_per_subject"] is None,
    "fresh stratified splits with seed 42": (
        p["patient_split"]["mode"] == "fresh_stratified" and p["patient_split"]["seed"] == 42
    ),
    "nonoverlapping 120-second segments from offset zero": (
        p["window_s"] == p["stride_s"] == 120 and p["skip_start_s"] == 0
    ),
    "1994 eligible recordings and 195 patients": p["recordings"] == 1994 and p["subjects"] == 195,
}
for description, passed in checks.items():
    if not passed:
        raise SystemExit(f"Audit provenance check failed: {description}")
print(f"Verified: {p['subjects']} patients, {p['segments']} segments")
PY

mkdir -p logs/waveform_patient
RUN=$(mktemp -d "$PWD/logs/waveform_patient/smoke_P512_linear_XXXXXXXX")
echo "Outputs: $RUN"

# pipefail ensures that a pytest failure stops the script despite tee succeeding.
python -m pytest -q \
  tests/test_patient_waveform.py \
  tests/test_waveform_full_segments.py \
  tests/test_waveform_sae.py \
  tests/test_count_pool_segments.py \
  2>&1 | tee "$RUN/tests.log"

HYDRA_FULL_ERROR=1 python -u src/train_sae.py \
  --config-name train_patient_waveform \
  "data.data_dir=$DATA_DIR" \
  "data.windows_train_csv=$M/windows_train.csv" \
  "data.windows_val_csv=$M/windows_val.csv" \
  "data.windows_test_csv=$M/windows_test.csv" \
  data.num_workers=4 feature.device=cuda \
  feature.random_state=42 seed=42 \
  feature.spec.atom_len=256 \
  feature.spec.pool_len=512 \
  feature.spec.top_r=1 \
  feature.train_spec.patients_per_batch=8 \
  feature.train_spec.segments_per_patient=4 \
  feature.train_spec.segment_batch=2 \
  feature.train_spec.epochs=1 \
  'supervised.hidden_dims=[]' \
  supervised.warmup_epochs=2 \
  supervised.calibration_batches=2 \
  supervised.joint_epochs=1 \
  supervised.ramp_epochs=1 \
  supervised.lambda_center=null \
  supervised.lambda_multiplier=0.1 \
  "output_dir=$RUN" \
  2>&1 | tee "$RUN/console.log"

echo "Smoke test completed. Outputs: $RUN"
