#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/OUTPUT_FILES"
mkdir -p "$OUTPUT_ROOT"

to_epoch() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo 0
    return
  fi
  if ! date -d "$value" '+%s' >/dev/null 2>&1; then
    echo 0
    return
  fi
  date -d "$value" '+%s'
}

declare -a TARGET_STATIONS=()
declare -A SEEN_STATIONS=()
if (( $# == 0 )); then
  TARGET_STATIONS=("01" "02" "03" "04")
  for code in "${TARGET_STATIONS[@]}"; do
    SEEN_STATIONS[$code]=1
  done
else
  for arg in "$@"; do
    if [[ ! "$arg" =~ ^[0-9]+$ ]]; then
      log "Invalid station identifier: $arg"
      exit 1
    fi
    arg=$((10#$arg))
    if (( arg < 1 || arg > 4 )); then
      log "Station identifier must be between 1 and 4 (got $arg)"
      exit 1
    fi
    code="$(printf '%02d' "$arg")"
    if [[ -n ${SEEN_STATIONS[$code]:-} ]]; then
      continue
    fi
    SEEN_STATIONS[$code]=1
    TARGET_STATIONS+=("$code")
  done
fi

process_station() {
  local station_code="$1"
  local station="MINGO${station_code}"
  local task_output="$REPO_ROOT/STATIONS/${station}/STAGE_1/EVENT_DATA/STEP_3/TASK_2/OUTPUT_FILES"
  local processed_csv="$OUTPUT_ROOT/${station}_processed_basenames.csv"
  local hld_csv="$REPO_ROOT/STATIONS/${station}/STAGE_0/REPROCESSING/STEP_1/METADATA/hld_files_brought.csv"
  local dat_csv="$REPO_ROOT/STATIONS/${station}/STAGE_0/REPROCESSING/STEP_2/METADATA/dat_files_unpacked.csv"

  if [[ ! -d "$task_output" ]]; then
    log "[${station}] Task 2 output directory not found: $task_output (skipping)"
    return
  fi

  mkdir -p "$(dirname "$hld_csv")" "$(dirname "$dat_csv")"

  local -A BASENAME_TS=()
  local -A BASENAME_SRC=()

  while IFS= read -r -d '' csv_file; do
    local source_line exec_line basenames_raw exec_raw
    local -a basenames=()
    local -a exec_values=()
    source_line=$(grep -m1 '^# *source_basenames=' "$csv_file" || true)
    exec_line=$(grep -m1 '^# *execution_date=' "$csv_file" || true)
    [[ -z "$source_line" ]] && continue

    basenames_raw=${source_line#*=}
    basenames_raw=${basenames_raw//[$'\r\n']/}
    IFS=',' read -r -a basenames <<< "$basenames_raw"
    if (( ${#basenames[@]} == 0 )); then
      continue
    fi

    if [[ -n "$exec_line" ]]; then
      exec_raw=${exec_line#*=}
      exec_raw=${exec_raw//[$'\r\n']/}
      IFS=',' read -r -a exec_values <<< "$exec_raw"
    fi
    local exec_count=${#exec_values[@]}
    local file_timestamp
    file_timestamp=$(date -d "@$(stat -c %Y "$csv_file")" '+%Y-%m-%d %H:%M:%S')

    for idx in "${!basenames[@]}"; do
      local base ts map_index candidate existing_ts existing_epoch new_epoch
      base=$(echo "${basenames[$idx]}" | tr -d '[:space:]')
      [[ -z "$base" ]] && continue
      ts="$file_timestamp"
      if (( exec_count > 0 )); then
        map_index=$idx
        if (( map_index >= exec_count )); then
          map_index=$((exec_count - 1))
        fi
        candidate=$(echo "${exec_values[$map_index]}" | sed 's/^ *//;s/ *$//')
        [[ -n "$candidate" ]] && ts="$candidate"
      fi
      existing_ts="${BASENAME_TS[$base]:-}"
      if [[ -z "$existing_ts" ]]; then
        BASENAME_TS[$base]="$ts"
        BASENAME_SRC[$base]="$csv_file"
        continue
      fi
      existing_epoch=$(to_epoch "$existing_ts")
      new_epoch=$(to_epoch "$ts")
      if (( new_epoch != 0 && (existing_epoch == 0 || new_epoch < existing_epoch) )); then
        BASENAME_TS[$base]="$ts"
        BASENAME_SRC[$base]="$csv_file"
      fi
    done
done < <(find "$task_output" -type f -name '*.csv' -print0 | sort -z)

  if (( ${#BASENAME_TS[@]} == 0 )); then
    log "[${station}] No basenames discovered under $task_output"
    return
  fi

  local sorted_basenames
  sorted_basenames=$(printf '%s\n' "${!BASENAME_TS[@]}" | sort)

  {
    echo "basename,execution_timestamp,source_csv"
    while IFS= read -r base; do
      printf '%s,%s,%s\n' "$base" "${BASENAME_TS[$base]}" "${BASENAME_SRC[$base]}"
    done <<< "$sorted_basenames"
  } > "$processed_csv"
  log "[${station}] Processed metadata saved to $processed_csv"

  log "[${station}] Refreshing $hld_csv and $dat_csv with discovered basenames"
  {
    echo "hld_name,bring_timesamp"
    while IFS= read -r base; do
      printf '%s,%s\n' "$base" "${BASENAME_TS[$base]}"
    done <<< "$sorted_basenames"
  } > "$hld_csv"

  {
    echo "dat_name,execution_timestamp,execution_duration_s"
    while IFS= read -r base; do
      printf '%s,%s,0\n' "$base" "${BASENAME_TS[$base]}"
    done <<< "$sorted_basenames"
  } > "$dat_csv"

  log "[${station}] Execution CSVs updated based on processed basenames"
}

for code in "${TARGET_STATIONS[@]}"; do
  process_station "$code"
done
