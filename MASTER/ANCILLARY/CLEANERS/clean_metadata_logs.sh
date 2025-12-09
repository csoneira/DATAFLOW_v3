#!/bin/bash
# Remove stage metadata CSV trackers to force fresh pulls.

set -euo pipefail

ROOT="${HOME}/DATAFLOW_v3"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

purge_pattern() {
  local pattern="$1"
  shopt -s nullglob
  local matches=($pattern)
  shopt -u nullglob

  if ((${#matches[@]} == 0)); then
    log "No matches for pattern: $pattern"
    return
  fi

  for file in "${matches[@]}"; do
    if [[ -f "$file" ]]; then
      rm -f -- "$file"
      log "Removed: $file"
    fi
  done
}

log "Starting metadata CSV cleanup..."

purge_pattern "${ROOT}/STATIONS/MINGO0*/STAGE_0/NEW_FILES/METADATA/raw_files_brought.csv"
purge_pattern "${ROOT}/STATIONS/MINGO0*/STAGE_0/REPROCESSING/STEP_1/METADATA/hld_files_brought.csv"
purge_pattern "${ROOT}/STATIONS/MINGO0*/STAGE_0/REPROCESSING/STEP_2/METADATA/dat_files_unpacked.csv"
#purge_pattern "${ROOT}/STATIONS/MINGO0*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/METADATA/task_*_metadata_execution.csv"

log "Metadata CSV cleanup complete."
