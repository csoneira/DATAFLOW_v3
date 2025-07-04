#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# check_station_dirs.sh  –  file-count monitor with colour-coded recency
# ---------------------------------------------------------------------------

set -euo pipefail

##############################################################################
# Configuration
##############################################################################
BASE="/home/mingo/DATAFLOW_v3/STATIONS"

DIRS=(
  "/ZERO_STAGE/COMPRESSED_HLDS"
  "/ZERO_STAGE/UNCOMPRESSED_HLDS"
  "/ZERO_STAGE/SENT_TO_RAW_TO_LIST_PIPELINE"

  "/FIRST_STAGE/EVENT_DATA/RAW"

  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/PROCESSING_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/COMPLETED_DIRECTORY"

  "/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY"

  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_UNPROCESSED"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_PROCESSING"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_COMPLETED"

  "/FIRST_STAGE/EVENT_DATA/ACC_EVENTS_DIRECTORY"
)

FRESH_SEC=300       # < 5 min  → green
STALE_SEC=3600      # ≥ 60 min → orange
RECENT_SEC=3600     # files modified within last hour

##############################################################################
# ANSI colours (disable with --no-color)
##############################################################################
USE_COLOR=true
if [[ ${1:-} == "--no-color" ]]; then
  USE_COLOR=false
  shift
fi

if $USE_COLOR; then
  CLR_BOLD="\033[1m"; CLR_RESET="\033[0m"
  CLR_GREEN="\033[0;32m"; CLR_PURPLE="\033[0;35m"
  CLR_ORANGE="\033[0;33m"; CLR_RED="\033[0;31m"
else
  CLR_BOLD="" CLR_RESET=""
  CLR_GREEN="" CLR_PURPLE="" CLR_ORANGE="" CLR_RED=""
fi

##############################################################################
# Stations (positional args or default 1-4)
##############################################################################
if (( $# )); then
  STATIONS=("$@")
else
  STATIONS=(1 2 3 4)
fi

##############################################################################
# Header
##############################################################################
printf "%-80s %10s %5s   %s\n" "Directory" "Files" "<1h" "Last modified"
printf "%-80s %10s %5s   %s\n" "---------" "-----" "---" "--------------"

##############################################################################
# Main loop
##############################################################################
now=$(date +%s)

for st in "${STATIONS[@]}"; do
  st_id=$(printf "%02d" "$st")
  root="${BASE}/MINGO${st_id}"

  printf "${CLR_BOLD}miniTRASGO %s${CLR_RESET}\n" "$st"

  for rel in "${DIRS[@]}"; do
    path="${root}/${rel}"

    if [[ -d "$path" ]]; then
      # total and recent file counts
      file_count=$(find "$path" -type f -printf '.' | wc -c)
      recent_count=$(find "$path" -type f -printf '%T@\n' |
                     awk -v now="$now" -v lim="$RECENT_SEC" \
                         '{if (now-$1<=lim) c++} END{print c+0}')

      # newest mtime
      if (( file_count > 0 )); then
        mtime_sec=$(find "$path" -type f -printf '%T@\n' | sort -n | tail -1)
      else
        mtime_sec=$(stat -c %Y "$path")
      fi
      mtime_sec=${mtime_sec%.*}
      mtime_str=$(date -d @"$mtime_sec" '+%Y-%m-%d %H:%M:%S')

      # colour by recency of newest file
      age=$(( now - mtime_sec ))
      if   (( age >= STALE_SEC )); then colour=$CLR_ORANGE
      elif (( age >= FRESH_SEC )); then colour=$CLR_PURPLE
      else                              colour=$CLR_GREEN
      fi

      printf "%-80s %10d ${colour}%5d${CLR_RESET}   ${colour}%s${CLR_RESET}\n" \
             "    $rel" "$file_count" "$recent_count" "$mtime_str"
    else
      printf "%-80s ${CLR_RED}%10s %5s${CLR_RESET}   -\n" \
             "    $rel" "MISSING" "-"
    fi
  done
done

##############################################################################
# Legend
##############################################################################
printf "\n"
printf %b "Legend:\n"
printf %b "  ${CLR_GREEN}green${CLR_RESET}  — newest file < 5 min old\n"
printf %b "  ${CLR_PURPLE}purple${CLR_RESET} — 5–60 min old\n"
printf %b "  ${CLR_ORANGE}orange${CLR_RESET} — ≥ 60 min old\n"
printf %b "  ${CLR_RED}red${CLR_RESET}    — directory missing\n"
printf "  “Recent” = files modified within the last hour (same colour key)\n"
