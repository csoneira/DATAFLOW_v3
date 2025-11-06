#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Configuration
##############################################################################
BASE="$HOME/DATAFLOW_v3/STATIONS"

# Track the STEP_1/TASK_* layout introduced in the Stage 1 refactor. Building the
# directory list programmatically keeps the watchlist aligned as tasks evolve.
declare -a DIRS=(
  "/STAGE_0/NEW_FILES"
  "/STAGE_0/REPROCESSING/INPUT_FILES"
  "/STAGE_0_to_1"
)

declare -a SHORT_DIRS=(
  "/STAGE_0/NEW_FILES"
  "/STAGE_0_to_1"
)

TASK_COUNT=5
TASK_INPUT_STATES=(
  "INPUT_FILES/UNPROCESSED_DIRECTORY"
  "INPUT_FILES/PROCESSING_DIRECTORY"
  "INPUT_FILES/ERROR_DIRECTORY"
  "INPUT_FILES/COMPLETED_DIRECTORY"
)

for ((task=1; task<=TASK_COUNT; task++)); do
  for state in "${TASK_INPUT_STATES[@]}"; do
    DIRS+=("/STAGE_1/EVENT_DATA/STEP_1/TASK_${task}/${state}")
  done
  DIRS+=("/STAGE_1/EVENT_DATA/STEP_1/TASK_${task}/OUTPUT_FILES")

  SHORT_DIRS+=("/STAGE_1/EVENT_DATA/STEP_1/TASK_${task}/INPUT_FILES/UNPROCESSED_DIRECTORY")
  SHORT_DIRS+=("/STAGE_1/EVENT_DATA/STEP_1/TASK_${task}/INPUT_FILES/ERROR_DIRECTORY")
done

# DIRS+=(
#   "/STAGE_1/EVENT_DATA/STEP_1_TO_2_OUTPUT"
#   "/STAGE_1/EVENT_DATA/STEP_2/LIST_TO_ACC/INPUT_FILES/UNPROCESSED"
#   "/STAGE_1/EVENT_DATA/STEP_2/LIST_TO_ACC/INPUT_FILES/PROCESSING"
#   "/STAGE_1/EVENT_DATA/STEP_2/LIST_TO_ACC/INPUT_FILES/ERROR_DIRECTORY"
#   "/STAGE_1/EVENT_DATA/STEP_2/LIST_TO_ACC/INPUT_FILES/REJECTED"
#   "/STAGE_1/EVENT_DATA/STEP_2/LIST_TO_ACC/INPUT_FILES/COMPLETED"
#   "/STAGE_1/EVENT_DATA/STEP_2/OUTPUT_FILES"
# )

# SHORT_DIRS+=(
#   "/STAGE_1/EVENT_DATA/STEP_1_TO_2_OUTPUT"
#   "/STAGE_1/EVENT_DATA/STEP_2/LIST_TO_ACC/INPUT_FILES/ERROR_DIRECTORY"
#   "/STAGE_1/EVENT_DATA/STEP_2/OUTPUT_FILES"
# )

##############################################################################
# ANSI colours (disable with --no-color)
##############################################################################
USE_COLOR=true                 # default: colourised output

# Collect non-option words (station numbers) here
declare -a POSITIONAL=()

show_help() {
  cat <<'EOF'
file_flow_check.sh
Summarises file ages across the STAGE_0 and STAGE_1 pipeline directories.

Usage:
  file_flow_check.sh [options] [station ...]

Options:
  -h, --help    Show this help message and exit.
      --no-color  Disable ANSI colours in the output.
  -s, --short    Show a trimmed set of directories for quick checks.

By default the script inspects stations 1–4. You can pass one or more
station numbers as positional arguments to limit the report.
EOF
}

while (( $# )); do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --no-color)
      USE_COLOR=false
      ;;
    -s|--short)
      DIRS=("${SHORT_DIRS[@]}")
      ;;
    --) shift; break ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    *)  POSITIONAL+=("$1") ;;
  esac
  shift
done
set -- "${POSITIONAL[@]}"

if $USE_COLOR; then
  CLR_BOLD="\033[1m"; CLR_RESET="\033[0m"
  CLR_RED="\033[0;31m";   CLR_YELLOW="\033[1;33m"
  CLR_GREEN="\033[0;32m"; CLR_BLUE="\033[0;34m"
  CLR_PURPLE="\033[0;35m"; CLR_ORANGE="\033[0;36m"
else
  CLR_BOLD="" CLR_RESET=""
  CLR_RED="" CLR_YELLOW="" CLR_GREEN="" CLR_BLUE="" CLR_PURPLE=""
  CLR_ORANGE=""
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
printf "%-80s %5s %5s %5s %5s %5s %5s\n" \
       "Directory" "<1m" "1-5m" "5-60m" "1-24h" ">24h" "Total"
printf "%-80s %5s %5s %5s %5s %5s %5s\n" \
       "---------" "---" "----" "-----" "-----" "----" "----"

##############################################################################
# Main loop
##############################################################################
for st in "${STATIONS[@]}"; do
  st_id=$(printf "%02d" "$st")
  root="${BASE}/MINGO${st_id}"

  printf "${CLR_BOLD}miniTRASGO %s${CLR_RESET}\n" "$st"

  for rel in "${DIRS[@]}"; do
    path="${root}/${rel}"

    if [[ -d "$path" ]]; then
      now=$(date +%s)

      # single find|awk pass per directory
      read -r c0 c1 c2 c3 c4 <<< "$(find "$path" -type f -printf '%T@\n' |
        awk -v now="$now" '
          BEGIN{for(i=0;i<5;i++)c[i]=0}
          {
            age=now-$1
            if (age<60)          c[0]++
            else if (age<300)    c[1]++
            else if (age<3600)   c[2]++
            else if (age<86400)  c[3]++
            else                 c[4]++
          }
          END{printf "%d %d %d %d %d", c[0],c[1],c[2],c[3],c[4]}
        ')"

      total=$(( c0 + c1 + c2 + c3 + c4 ))

      printf "%-80s ${CLR_ORANGE}%5d${CLR_RESET} ${CLR_YELLOW}%5d${CLR_RESET} ${CLR_GREEN}%5d${CLR_RESET} ${CLR_BLUE}%5d${CLR_RESET} ${CLR_PURPLE}%5d${CLR_RESET} %5d\n" \
             "    $rel" "$c0" "$c1" "$c2" "$c3" "$c4" "$total"
    else
      printf "%-80s ${CLR_RED}%5s${CLR_RESET} ${CLR_RED}%5s${CLR_RESET} ${CLR_RED}%5s${CLR_RESET} ${CLR_RED}%5s${CLR_RESET} ${CLR_RED}%5s${CLR_RESET} %5s\n" \
             "    $rel" "-" "-" "-" "-" "-" "-"
    fi
  done
done

##############################################################################
# Legend
##############################################################################
printf "\n"
printf %b "Legend:\n"
printf %b "  ${CLR_ORANGE}bold${CLR_RESET}   — files < 1 min old\n"
printf %b "  ${CLR_YELLOW}yellow${CLR_RESET} — 1 – 5 min old\n"
printf %b "  ${CLR_GREEN}green${CLR_RESET}  — 5 – 60 min old\n"
printf %b "  ${CLR_BLUE}blue${CLR_RESET}   — 1 – 24 h old\n"
printf %b "  ${CLR_PURPLE}purple${CLR_RESET} — > 24 h old\n"
printf %b "  ${CLR_RED}red${CLR_RESET}    — directory missing\n"
