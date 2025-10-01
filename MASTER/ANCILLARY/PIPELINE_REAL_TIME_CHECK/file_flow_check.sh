#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Configuration
##############################################################################
BASE="$HOME/DATAFLOW_v3/STATIONS"

DIRS=(
  "/ZERO_STAGE/COMPRESSED_HLDS"
  "/ZERO_STAGE/UNCOMPRESSED_HLDS"
  "/ZERO_STAGE/SENT_TO_RAW_TO_LIST_PIPELINE"

  "/FIRST_STAGE/EVENT_DATA/RAW"

  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/PROCESSING_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/ERROR_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/COMPLETED_DIRECTORY"

  "/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY"

  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_UNPROCESSED"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_PROCESSING"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ERROR_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_COMPLETED"

  "/FIRST_STAGE/EVENT_DATA/ACC_EVENTS_DIRECTORY"
)

SHORT_DIRS=(
  "/ZERO_STAGE/COMPRESSED_HLDS"
  "/ZERO_STAGE/UNCOMPRESSED_HLDS"
  # "/ZERO_STAGE/SENT_TO_RAW_TO_LIST_PIPELINE"

  # "/FIRST_STAGE/EVENT_DATA/RAW"

  # "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY"
  # "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/PROCESSING_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/ERROR_DIRECTORY"
  # "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/COMPLETED_DIRECTORY"

  "/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY"

  # "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_UNPROCESSED"
  # "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_PROCESSING"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ERROR_DIRECTORY"
  # "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ACC_COMPLETED"

  "/FIRST_STAGE/EVENT_DATA/ACC_EVENTS_DIRECTORY"
)

##############################################################################
# ANSI colours (disable with --no-color)
##############################################################################
USE_COLOR=true                 # default: colourised output

# Collect non-option words (station numbers) here
declare -a POSITIONAL=()

while (( $# )); do
  case "$1" in
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
