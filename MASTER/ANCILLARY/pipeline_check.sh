#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# check_station_files.sh
#
# Verify the key CSVs for each requested station.  Output columns:
#   File (relative path), Size (IEC units), Last modified (YYYY-MM-DD HH:MM:SS)
#
# Usage
#   $ bash check_station_files.sh          # stations 1 2 3 4
#   $ bash check_station_files.sh 2 4      # only stations 2 and 4
# ---------------------------------------------------------------------------

set -euo pipefail

BASE="/home/mingo/DATAFLOW_v3/STATIONS"

FILES=(
  "FIRST_STAGE/EVENT_DATA/raw_to_list_metadata.csv"
  "FIRST_STAGE/EVENT_DATA/event_accumulator_metadata.csv"
  "FIRST_STAGE/EVENT_DATA/big_event_data.csv"
  "FIRST_STAGE/LAB_LOGS/big_log_lab_data.csv"
  "FIRST_STAGE/COPERNICUS/big_copernicus_data.csv"
  "SECOND_STAGE/total_data_table.csv"
)

# ---------------------------------------------------------------------------
# stations to check: arguments or default 1-4
# ---------------------------------------------------------------------------
if (( $# )); then
  STATIONS=("$@")
else
  STATIONS=(1 2 3 4)
fi

# ---------------------------------------------------------------------------
# helper: human-readable size
# ---------------------------------------------------------------------------
hr_size() {
  numfmt --to=iec --format="%.1f" "$1" 2>/dev/null || echo "${1}B"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
for st in "${STATIONS[@]}"; do
  st_id=$(printf "%02d" "$st")              # "01", "02", …
  root="${BASE}/MINGO${st_id}"              # …/MINGO01, MINGO02, …

  printf "\n\033[1m=====  STATION %s  =====\033[0m\n" "$st"
  printf "%-70s %10s   %s\n" "File" "Size" "Last modified"
  printf "%-70s %10s   %s\n" "----" "----" "--------------"

  for rel in "${FILES[@]}"; do
    path="${root}/${rel}"
    if [[ -f "$path" ]]; then
      bytes=$(stat -c %s "$path")
      mtime=$(stat -c %y "$path" | cut -d'.' -f1)
      printf "%-70s %10s   %s\n" "$rel" "$(hr_size "$bytes")" "$mtime"
    else
      printf "%-70s %10s   %s\n" "$rel" "MISSING" "-"
    fi
  done
done
