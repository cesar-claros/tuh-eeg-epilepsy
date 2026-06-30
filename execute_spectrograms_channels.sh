#!/bin/bash
# Generate per-channel spectrogram grids (spectrogram_channels.py) for the top-N most
# anomalous recordings per native sampling rate, collected in one folder for inspection.
#
# Source: the per-rate anomaly rankings written by rank_psd_anomaly.py in
# diagnostics/psd/ (psd_anomaly-sfreq<rate>-...csv). For each rate it takes the top-N
# recordings (by the 'anomaly' column) and runs spectrogram_channels.py with the SAME
# flags the rankings were built with (bipolar, native rate, 60/120 Hz notch), writing
# <rec>_spectrograms.png straight into diagnostics/spectrograms/.
#
# Companion to execute_psd_segments.sh (same work list, different view): psd_segments
# shows one channel-averaged spectrogram plus hot time segments; this shows the full
# spectrogram of every channel, to see which channel carries an anomalous band.
#
# Run from inside the container, from code/ (it reads the EDFs named in the CSVs, which
# are absolute container paths). No precomputed PSD sidecar is needed.

set -u -o pipefail
cd "$(dirname "$0")"                  # run from code/

CSV_DIR="diagnostics/psd"            # where the rank_psd_anomaly.py CSVs live
OUT_DIR="diagnostics/spectrograms"   # where all the per-channel spectrograms are collected
SEG_FLAGS=(--bipolar --native --notch_freqs 60 120)   # MUST match the rankings' processing
# Extra spectrogram_channels.py options applied to every plot, e.g. ('--fmax' '80').
PLOT_FLAGS=()

# Per native rate (Hz): how many of the most anomalous recordings to plot. Edit freely;
# a count above the number available just plots all of them. (Available as of writing:
# 250 -> 593, 256 -> 1563, 400 -> 51, 1000 -> 20.)
TOPS=(
  "250:100"
  "256:100"
  "400:0"
  "1000:0"
)

mkdir -p "$OUT_DIR"

# ---- collect the (rate, edf) work list from the CSVs ----
edfs=()
rates=()
for spec in "${TOPS[@]}"; do
  rate="${spec%%:*}"; n="${spec#*:}"
  csv=$(ls "${CSV_DIR}"/psd_anomaly-sfreq${rate}-*.csv 2>/dev/null | head -1)
  if [[ -z "$csv" ]]; then
    echo "!! no anomaly CSV for ${rate} Hz in ${CSV_DIR} (skipping)"; continue
  fi
  # Column indices from the header, so we are robust to column reordering.
  hdr=$(head -1 "$csv" | tr -d '\r')
  pcol=$(echo "$hdr" | tr ',' '\n' | grep -n '^path$'    | cut -d: -f1)
  acol=$(echo "$hdr" | tr ',' '\n' | grep -n '^anomaly$' | cut -d: -f1)
  if [[ -z "$pcol" || -z "$acol" ]]; then
    echo "!! ${csv} missing a path/anomaly column (skipping)"; continue
  fi
  # Data rows sorted by anomaly descending (re-sorted here in case the file was edited),
  # take the top N, emit the path column.
  mapfile -t top < <(tail -n +2 "$csv" | tr -d '\r' \
      | sort -t, -k"${acol}","${acol}" -gr | head -n "$n" \
      | awk -F, -v c="$pcol" '{print $c}')
  echo "== ${rate} Hz: ${#top[@]} recordings from $(basename "$csv") =="
  for e in "${top[@]}"; do edfs+=("$e"); rates+=("$rate"); done
done

total=${#edfs[@]}
if (( total == 0 )); then echo "No recordings to plot."; exit 1; fi
echo "Total: ${total} recordings -> ${OUT_DIR}/"

# ---- generate the plots ----
i=0; ok=0; fail=0; skip=0
for idx in "${!edfs[@]}"; do
  edf="${edfs[$idx]}"; rate="${rates[$idx]}"
  i=$((i + 1))
  stem=$(basename "$edf" .edf)
  out="${OUT_DIR}/${stem}_spectrograms.png"
  if [[ -s "$out" ]]; then
    echo "[$i/$total] skip (exists): ${stem}"; skip=$((skip + 1)); continue
  fi
  if [[ ! -f "$edf" ]]; then
    echo "[$i/$total] !! EDF not found: ${edf}"; fail=$((fail + 1)); continue
  fi
  echo "=== [$i/$total] ${rate} Hz  ${stem} ==="
  if python src/spectrogram_channels.py --edf "$edf" "${SEG_FLAGS[@]}" ${PLOT_FLAGS[@]+"${PLOT_FLAGS[@]}"} --out "$out"; then
    ok=$((ok + 1))
  else
    echo "!!! FAILED: ${stem} (continuing)"; fail=$((fail + 1))
  fi
done

echo "Done. ${ok} plotted, ${skip} skipped, ${fail} failed -> ${OUT_DIR}/"
