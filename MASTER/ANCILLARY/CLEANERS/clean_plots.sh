#!/bin/bash
# Clean all PLOTS directories inside STATIONS/* (up to 5 levels)
# and report accurate per-directory + total freed space.

set -euo pipefail
LC_ALL=C   # consistent decimals in awk/printf

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
clean_plots.sh
Purges the contents of every PLOTS directory beneath STATIONS and reports disk
space reclaimed.

Usage:
  clean_plots.sh

Options:
  -h, --help    Show this help message and exit.

The script keeps the PLOTS directories themselves but deletes all files and
subdirectories inside each one. Run only when plot exports are no longer needed.
EOF
  exit 0
fi

BASE_DIR="$HOME/DATAFLOW_v3/STATIONS"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: Directory $BASE_DIR does not exist."
  exit 1
fi

echo "Cleaning all PLOTS directories under: $BASE_DIR"

total_before=0
total_after=0
total_freed=0

# Find PLOTS dirs up to 5 levels deep (note: -maxdepth before tests)
# Use -print0 for robust path handling
mapfile -d '' plots_dirs < <(find "$BASE_DIR" -maxdepth 5 -type d -name 'PLOTS' -print0)

if (( ${#plots_dirs[@]} == 0 )); then
  echo "No PLOTS directories found."
  exit 0
fi

for dir in "${plots_dirs[@]}"; do
  # Size before (bytes) – use du -sb for accuracy
  before_bytes=$(du -sb "$dir" | awk '{print $1}')
  total_before=$(( total_before + before_bytes ))

  echo "→ Cleaning $dir"

  # Remove everything inside PLOTS (files + hidden + subdirs), keep the PLOTS dir itself
  # Using find -mindepth 1 -delete handles hidden files and nested dirs safely
  if [[ -d "$dir" ]]; then
    # If directory not writable, try to make it writable to allow deletion
    chmod u+w "$dir" >/dev/null 2>&1 || true
    find "$dir" -mindepth 1 -exec chmod -R u+w {} + >/dev/null 2>&1 || true
    find "$dir" -mindepth 1 -delete
  fi

  # Size after (bytes)
  after_bytes=$(du -sb "$dir" | awk '{print $1}')
  total_after=$(( total_after + after_bytes ))

  freed_bytes=$(( before_bytes - after_bytes ))

  # Per-directory summary
  before_gb=$(awk -v v="$before_bytes" 'BEGIN{printf "%.3f", v/1000000000}')
  after_gb=$(awk -v v="$after_bytes"  'BEGIN{printf "%.3f", v/1000000000}')
  freed_gb=$(awk -v v="$freed_bytes"  'BEGIN{printf "%.3f", v/1000000000}')
  echo "   Size before: ${before_gb} GB | after: ${after_gb} GB | freed: ${freed_gb} GB"
done

total_freed=$(( total_before - total_after ))

# Totals
tb_gb=$(awk -v v="$total_before" 'BEGIN{printf "%.3f", v/1000000000}')
ta_gb=$(awk -v v="$total_after"  'BEGIN{printf "%.3f", v/1000000000}')
tf_gb=$(awk -v v="$total_freed"  'BEGIN{printf "%.3f", v/1000000000}')
tb_gib=$(awk -v v="$total_before" 'BEGIN{printf "%.3f", v/1024/1024/1024}')
ta_gib=$(awk -v v="$total_after"  'BEGIN{printf "%.3f", v/1024/1024/1024}')
tf_gib=$(awk -v v="$total_freed"  'BEGIN{printf "%.3f", v/1024/1024/1024}')

echo "All PLOTS directories cleaned."
echo "Limpieza completada."
echo "   Total antes:    ${tb_gb} GB  (${tb_gib} GiB)"
echo "   Total después:  ${ta_gb} GB  (${ta_gib} GiB)"
echo "   Total liberado: ${tf_gb} GB  (${tf_gib} GiB)"
