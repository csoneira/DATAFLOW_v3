#!/usr/bin/env bash
set -euo pipefail

# Self-execute in watch mode unless running as internal call
if [[ "${SNAPSHOT_INTERNAL:-0}" -ne 1 ]]; then
  exec env SNAPSHOT_INTERNAL=1 watch -n 1 -t "$0" "$@"
fi

# Configuration
LOG_DIR="$HOME/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS"
LINES=11
DEFAULT_COLS=4
JOBS=(log_bring_reprocessing_files log_unpack_reprocessing_files raw_to_list_events ev_accumulator)

# Option parsing
cols=$DEFAULT_COLS
while getopts "c:" opt; do
  case $opt in
    c) cols=$OPTARG ;;
    *) echo "Usage: $0 [-c columns] [station ...]"; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Station list
STATIONS=("$@")
[[ ${#STATIONS[@]} -eq 0 ]] && STATIONS=(1 2 3 4)

# Layout
BLOCK_HEIGHT=$((LINES + 2))
sep='  '
term_w=$(tput cols)
col_w=$(( (term_w - (cols - 1) * ${#sep}) / cols ))

# Build padded block
build_block() {
  local job=$1 file=$2 now=$(date +%s)
  local hdr payload ts

  if [[ -f $file ]]; then
    ts=$(date -r "$file" '+%H:%M:%S')
    hdr="[$job] (updated ${ts})"
    payload=$(tail -n "$LINES" "$file")
  else
    hdr="[$job] (missing)"
    payload=''
  fi

  {
    printf '%s\n' "$hdr"
    [[ -n $payload ]] && printf '%s\n' "$payload"
    local printed=$(( $(grep -c '' <<<"$payload") + 1 ))
    for ((i=printed+1; i<=BLOCK_HEIGHT; i++)); do echo; done
  }
}

# Main output
for st in "${STATIONS[@]}"; do
  printf '\nSTATION %d\n' "$st"
  printf -- '%*s\n' "$term_w" '' | tr ' ' '─'

  declare -a blocks=()
  for job in "${JOBS[@]}"; do
    file="${LOG_DIR}/${job}_${st}.log"
    blocks+=("$(build_block "$job" "$file")")
  done

  total_jobs=${#blocks[@]}
  rows=$(( (total_jobs + cols - 1) / cols ))

  for ((r=0; r<rows; r++)); do
    for ((line=0; line<BLOCK_HEIGHT; line++)); do
      for ((c=0; c<cols; c++)); do
        idx=$(( r + c*rows ))
        if (( idx < total_jobs )); then
          line_text=$(printf '%s\n' "${blocks[idx]}" | sed -n "$((line+1))p")
          printf '%-*.*s' "$col_w" "$col_w" "$line_text"
        else
          printf '%-*s' "$col_w" ''
        fi
        (( c < cols - 1 )) && printf '%s' "$sep"
      done
      printf '\n'
    done
    printf '%s\n' "$(printf '─%.0s' $(seq 1 "$term_w"))"
  done
done
