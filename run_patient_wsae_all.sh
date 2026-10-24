#!/usr/bin/env bash
# Launch the complete P=512 campaign inside the HPC container on an allocated GPU.
# Each invocation creates a new campaign and reuses the verified smoke signal cache.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: bash $0 [fresh_uncapped_audit_directory]"
  echo "Defaults: smoke cache_smoke_P512_7cqc27GG; audit wsae_all_segments_fresh42_20261023_210759."
  echo "Environment: SMOKE_DIR, AUDIT_ALL, DATA_DIR, NUM_WORKERS=4."
  echo "Always creates a new RUN_DIR and runs STAGE=all."
  echo "Seeds 0,1,2; P=512, R=1, linear head; 30/50/30 epochs; 32 probes/seed; five lambda arms."
  exit 0
fi
if (( $# > 1 )); then
  echo "ERROR: provide at most one audit directory." >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SMOKE_DIR="${SMOKE_DIR:-$SCRIPT_DIR/logs/waveform_patient/cache_smoke_P512_7cqc27GG}"
AUDIT_ALL="${1:-${AUDIT_ALL:-$SCRIPT_DIR/logs/audits/wsae_all_segments_fresh42_20261023_210759}}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
NUM_WORKERS="${NUM_WORKERS:-4}"
if [[ ! -d "$SMOKE_DIR" || ! -d "$AUDIT_ALL" || ! -d "$DATA_DIR" || ! "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SMOKE_DIR, audit and DATA_DIR must exist; NUM_WORKERS must be nonnegative." >&2
  exit 2
fi
SMOKE_DIR=$(cd -- "$SMOKE_DIR" && pwd)
AUDIT_ALL=$(cd -- "$AUDIT_ALL" && pwd)
DATA_DIR=$(cd -- "$DATA_DIR" && pwd)
SIGNAL_CACHE_DIR=$(cat "$SMOKE_DIR/signal_cache_path.txt")
if [[ ! -d "$SIGNAL_CACHE_DIR" ]]; then
  echo "ERROR: the smoke signal cache is unavailable: $SIGNAL_CACHE_DIR" >&2
  exit 2
fi

# Read metadata only: check successful reuse and the retained cache before launching tests/training.
python - "$SMOKE_DIR" "$SIGNAL_CACHE_DIR" "$AUDIT_ALL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

smoke, cache, audit = map(Path, sys.argv[1:])
summary = json.loads((smoke / "cache_smoke_summary.json").read_text())
passes = summary.get("validation", [])
if summary.get("status") != "passed" or len(passes) != 2 or not all(row["verified"] for row in passes):
    raise SystemExit("ERROR: two verified smoke validation passes are required.")
warm = passes[1]
if warm["preprocessing_calls"] != 0 or warm["cache_entries_after"] != warm["segments"]:
    raise SystemExit("ERROR: smoke validation did not confirm complete cache reuse.")
identities = json.loads((smoke / "signal_cache.json").read_text())
for split in ("train", "val"):
    if not (cache / identities[split]).is_dir():
        raise SystemExit(f"ERROR: retained {split} cache namespace is missing.")
expected = {f"{index:08d}.npz" for index in range(warm["segments"])}
if {path.name for path in (cache / identities["val"]).glob("*.npz")} != expected:
    raise SystemExit("ERROR: validation cache entries are missing or changed since the smoke run.")
provenance = json.loads((smoke / "smoke_provenance.json").read_text())
for split in ("train", "val", "test"):
    name = f"windows_{split}.csv"
    with (audit / name).open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if checksum != provenance["manifests"][name]:
        raise SystemExit(f"ERROR: {name} differs from the verified smoke manifests.")
PY

mkdir -p "$SCRIPT_DIR/logs/waveform_patient"
RUN_DIR=$(mktemp -d "$SCRIPT_DIR/logs/waveform_patient/P512_linear_long_cached_XXXXXXXX")
export RUN_DIR SIGNAL_CACHE_DIR DATA_DIR NUM_WORKERS
export STAGE=all
echo "New campaign: $RUN_DIR"
echo "Verified smoke: $SMOKE_DIR"
echo "Reusing signals: $SIGNAL_CACHE_DIR"
echo "Tests -> 3 references (30 pretrain / 50 head warm-up / 32 probes) -> 15 joint arms (30 epochs each)."
echo "Lambda multipliers: 0.03, 0.1, 0.3, 1, 3."
cd -- "$SCRIPT_DIR"
exec bash "$SCRIPT_DIR/run_patient_wsae_sweep.sh" "$AUDIT_ALL"
