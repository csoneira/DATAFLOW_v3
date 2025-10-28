#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# check_station_files.sh
# ---------------------------------------------------------------------------

set -euo pipefail

##############################################################################
# Configuration
##############################################################################
BASE="$HOME/DATAFLOW_v3/STATIONS"

FILES=(
  "STAGE_1/EVENT_DATA/raw_to_list_metadata.csv"
  "STAGE_1/EVENT_DATA/event_accumulator_metadata.csv"
  "STAGE_1/EVENT_DATA/big_event_data.csv"
  "STAGE_1/LAB_LOGS/big_log_lab_data.csv"
  "STAGE_1/COPERNICUS/big_copernicus_data.csv"
  "STAGE_2/total_data_table.csv"
)

FRESH_SEC=300       # < 5 min     → green
STALE_SEC=3600      # > 60 min    → orange

##############################################################################
# ANSI colour definitions (can be disabled with --no-color)
##############################################################################
USE_COLOR=true
show_help() {
  cat <<'EOF'
pipeline_check.sh
Reports freshness and size of key pipeline outputs for each station.

Usage:
  pipeline_check.sh [options] [station ...]

Options:
  -h, --help    Show this help message and exit.
      --no-color  Disable ANSI colour output.

Provide station numbers (e.g. 2 4) to limit the report; otherwise stations
1–4 are inspected. Files newer than 5 minutes are highlighted green, older
than 60 minutes orange, and missing files red.
EOF
}

if [[ ${1:-} == "--no-color" ]]; then
  USE_COLOR=false
  shift
elif [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  show_help
  exit 0
fi

if $USE_COLOR; then
  CLR_BOLD="\033[1m"
  CLR_RESET="\033[0m"
  CLR_GREEN="\033[0;32m"
  CLR_PURPLE="\033[0;35m"
  CLR_ORANGE="\033[0;33m"
  CLR_RED="\033[0;31m"
else
  CLR_BOLD="" CLR_RESET=""
  CLR_GREEN="" CLR_PURPLE="" CLR_ORANGE="" CLR_RED=""
fi

##############################################################################
# Human-readable size helper
##############################################################################
hr_size() {
  numfmt --to=iec --format="%.1f" "$1" 2>/dev/null || echo "${1}B"
}

##############################################################################
# Stations: positional args or default 1-4
##############################################################################
if (( $# )); then
  STATIONS=("$@")
else
  STATIONS=(1 2 3 4)
fi

##############################################################################
# Header
##############################################################################
printf "%-60s %10s   %s\n" "File" "Size" "Last modified"
printf "%-60s %10s   %s\n" "----" "----" "--------------"

##############################################################################
# Main loop
##############################################################################
now=$(date +%s)

for st in "${STATIONS[@]}"; do
  st_id=$(printf "%02d" "$st")
  root="${BASE}/MINGO${st_id}"

  printf "${CLR_BOLD}miniTRASGO %s${CLR_RESET}\n" "$st"

  for rel in "${FILES[@]}"; do
    path="${root}/${rel}"
    if [[ -f "$path" ]]; then
      bytes=$(stat -c %s "$path")
      mtime_sec=$(stat -c %Y "$path")
      mtime_str=$(date -d @"$mtime_sec" +'%Y-%m-%d %H:%M:%S')

      age=$(( now - mtime_sec ))
      if   (( age >= STALE_SEC )); then colour=$CLR_ORANGE
      elif (( age >= FRESH_SEC )); then colour=$CLR_PURPLE
      else                              colour=$CLR_GREEN
      fi

      printf "%-60s %10s   ${colour}%s${CLR_RESET}\n" \
             "    $rel" "$(hr_size "$bytes")" "$mtime_str"
    else
      printf "%-60s ${CLR_RED}%10s${CLR_RESET}   -\n" \
             "    $rel" "MISSING"
    fi
  done
done

##############################################################################
# Legend
##############################################################################
printf "\n"
printf %b "Legend:\n"
printf %b "  ${CLR_GREEN}green${CLR_RESET}  — updated < 1 min\n"
printf %b "  ${CLR_PURPLE}purple${CLR_RESET} — 1–60 min old\n"
printf %b "  ${CLR_ORANGE}orange${CLR_RESET} — > 60 min old\n"
printf %b "  ${CLR_RED}red${CLR_RESET}    — file missing\n"
