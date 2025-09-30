#!/usr/bin/env bash
# file_flow_bars_ascii.sh — live ASCII bar chart with '#', '+', '·'
# Usage:
#   ./file_flow_bars_ascii.sh [--no-color|--color] [-s|--short] \
#     [-w WIDTH] [-i SECONDS] [--young-ch '#'] [--old-ch '+'] [--pad-ch '·'] [STATION ...]
# Example:
#   ./file_flow_bars_ascii.sh -s -w 70 -i 2 1 2 3 4

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
  "/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/ERROR_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/LIST_EVENTS_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/LIST_TO_ACC/ACC_FILES/ERROR_DIRECTORY"
  "/FIRST_STAGE/EVENT_DATA/ACC_EVENTS_DIRECTORY"
)

# Defaults
BAR_WIDTH=60
INTERVAL=5
USE_COLOR=false           # default to plain ASCII
YOUNG_CH="#"
OLD_CH="+"
PAD_CH="·"                # you can switch to '.' if you want pure ASCII

##############################################################################
# Options
##############################################################################
declare -a POSITIONAL=()
while (( $# )); do
  case "$1" in
    --no-color) USE_COLOR=false ;;
    --color)    USE_COLOR=true ;;
    -s|--short) DIRS=("${SHORT_DIRS[@]}") ;;
    -w|--width) BAR_WIDTH="${2:-60}"; shift ;;
    -i|--interval) INTERVAL="${2:-2}"; shift ;;
    --young-ch)  YOUNG_CH="${2:-#}"; shift ;;
    --old-ch)    OLD_CH="${2:-+}"; shift ;;
    --pad-ch)    PAD_CH="${2:-·}"; shift ;;
    --) shift; break ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    *)  POSITIONAL+=("$1") ;;
  esac
  shift
done
set -- "${POSITIONAL[@]}"

##############################################################################
# Colors (optional; only for labels/counts, never for bars)
##############################################################################
if $USE_COLOR; then
  CLR_RESET=$'\033[0m'; CLR_BOLD=$'\033[1m'
  CLR_RED=$'\033[0;31m'; CLR_DIM=$'\033[2m'
else
  CLR_RESET=""; CLR_BOLD=""; CLR_RED=""; CLR_DIM=""
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
# Helpers
##############################################################################
trap 'printf "%s" "$CLR_RESET"; tput cnorm 2>/dev/null || true; exit 0' INT TERM

repeat() {  # repeat N CH (works with UTF-8 like '·' too)
  local n=$1 ch=${2:-}
  (( n>0 )) || return 0
  local bar
  printf -v bar '%*s' "$n" ''   # make N spaces
  bar=${bar// /$ch}             # replace each space with CH
  printf '%s' "$bar"
}

scale_len() { # value max width -> integer length (rounded)
  local value=$1 max=$2 width=$3
  if (( max <= 0 || width <= 0 || value <= 0 )); then
    echo 0; return
  fi
  local num=$(( value * width ))
  local len=$(( num / max ))
  local rem=$(( num % max ))
  (( rem * 2 >= max )) && ((len++))
  echo "$len"
}

collect_counts() { # path -> echo "c0 c1 c2 c3 c4 total"
  local path=$1
  local now; now=$(date +%s)
  local out
  out=$(find "$path" -type f -printf '%T@\n' 2>/dev/null | \
    awk -v now="$now" '
      BEGIN{for(i=0;i<5;i++)c[i]=0}
      { age=now-$1
        if (age<60)          c[0]++
        else if (age<300)    c[1]++
        else if (age<3600)   c[2]++
        else if (age<86400)  c[3]++
        else                 c[4]++
      }
      END{printf "%d %d %d %d %d", c[0],c[1],c[2],c[3],c[4]}
    ')
  local a b c d e
  read -r a b c d e <<<"$out"
  local tot=$(( a+b+c+d+e ))
  echo "$a $b $c $d $e $tot"
}

draw_row() { # rel c0 c1 c2 c3 c4 total max_total
  local rel=$1 c0=$2 c1=$3 c2=$4 c3=$5 c4=$6 tot=$7 max_total=$8

  local young=$(( c0 + c1 + c2 ))
  local bar_len; bar_len=$(scale_len "$tot" "$max_total" "$BAR_WIDTH")
  local young_len old_len pad_len
  if (( tot > 0 )); then
    young_len=$(( (young * bar_len + tot/2) / tot ))  # rounded
  else
    young_len=0
  fi
  old_len=$(( bar_len - young_len ))
  pad_len=$(( BAR_WIDTH - bar_len ))

  # label truncation
  local label="    $rel"
  local max_label=80
  if (( ${#label} > max_label )); then
    label="${label:0:max_label-1}…"
  fi

  # Build bar: ####+++++····
  local bar=""
  (( young_len > 0 )) && bar+="$(repeat "$young_len" "$YOUNG_CH")"
  (( old_len   > 0 )) && bar+="$(repeat "$old_len"   "$OLD_CH")"
  (( pad_len   > 0 )) && bar+="$(repeat "$pad_len"   "$PAD_CH")"

  # Print line (keep counts compact; never use numbers as the bar itself)
  printf "%-82s %s  %s%5d%s  y<1h:%3d  o≥1h:%3d\n" \
    "$label" "$bar" "$CLR_BOLD" "$tot" "$CLR_RESET" "$young" "$((tot-young))"
}

draw_missing_row() { # rel
  local rel=$1
  local label="    $rel"
  local max_label=80
  if (( ${#label} > max_label )); then
    label="${label:0:max_label-1}…"
  fi
  printf "%-82s %s  %sMISSING%s\n" \
    "$label" "$(repeat "$BAR_WIDTH" "$PAD_CH")" "$CLR_RED" "$CLR_RESET"
}

##############################################################################
# Main loop
##############################################################################
tput civis 2>/dev/null || true

while :; do
  printf '\033[H\033[2J'   # clear screen
  printf "%sFile Flow (ASCII) — %s — width=%d%s\n" "$CLR_BOLD" "$(date '+%Y-%m-%d %H:%M:%S')" "$BAR_WIDTH" "$CLR_RESET"
  printf "Bar legend: '%s'=<1h (new)  '%s'=≥1h (old)  '%s'=pad | Scale: full bar = max total across all stations/dirs\n\n" \
    "$YOUNG_CH" "$OLD_CH" "$PAD_CH"

  # Collect first pass to determine global max
  declare -a ROWS=()   # "rel|c0|c1|c2|c3|c4|tot|missing"
  max_total=0

  for st in "${STATIONS[@]}"; do
    st_id=$(printf "%02d" "$st")
    root="${BASE}/MINGO${st_id}"
    # Station header
    ROWS+=( "__HDR__|miniTRASGO $st" )

    for rel in "${DIRS[@]}"; do
      path="${root}/${rel}"
      if [[ -d "$path" ]]; then
        read -r c0 c1 c2 c3 c4 tot < <(collect_counts "$path")
        ROWS+=( "$rel|$c0|$c1|$c2|$c3|$c4|$tot|0" )
        (( tot > max_total )) && max_total=$tot
      else
        ROWS+=( "$rel|0|0|0|0|0|0|1" )
      fi
    done
  done
  (( max_total == 0 )) && max_total=1

  # Render
  for row in "${ROWS[@]}"; do
    IFS='|' read -r a b c d e f g h <<<"$row"
    if [[ "$a" == "__HDR__" ]]; then
      printf "\n%s%s%s\n" "$CLR_BOLD" "$b" "$CLR_RESET"
      continue
    fi
    rel="$a"; c0="$b"; c1="$c"; c2="$d"; c3="$e"; c4="$f"; tot="$g"; missing="$h"
    if [[ "$missing" == "1" ]]; then
      draw_missing_row "$rel"
    else
      draw_row "$rel" "$c0" "$c1" "$c2" "$c3" "$c4" "$tot" "$max_total"
    fi
  done

  # Footer
  printf "\nNotes: Younger-than-1-hour files are emphasized with '%s'. Older are '%s'.\n" "$YOUNG_CH" "$OLD_CH"
  printf "Tips: use -s, --no-color/--color, -w WIDTH, -i SEC, --young-ch, --old-ch, --pad-ch\n"

  sleep "$INTERVAL"
done
