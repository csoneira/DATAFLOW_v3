#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

station_from_path() {
  local path="$1" rel
  if [[ "$path" == "$STATIONS_ROOT/"* ]]; then
    rel=${path#"$STATIONS_ROOT"/}
    echo "${rel%%/*}"
  else
    echo "UNKNOWN"
  fi
}

usage() {
  cat <<'EOF'
Usage: flush_from_unprocessed.sh [--dry-run]

Removes files under STATIONS/*/UNPROCESSED* whose basename (after the first
underscore, extension trimmed) appears in any *_processed_basenames.csv under
MASTER/ANCILLARY/PIPELINE_OPERATIONS/UPDATE_EXECUTION_CSVS/OUTPUT_FILES.

Options:
  --dry-run   Show which files would be removed without deleting them.
  -h,--help   Show this help message.
EOF
}

DRY_RUN=false
while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PROCESSED_DIR="$REPO_ROOT/MASTER/ANCILLARY/PIPELINE_OPERATIONS/UPDATE_EXECUTION_CSVS/OUTPUT_FILES"
STATIONS_ROOT="$REPO_ROOT/STATIONS"

if [[ ! -d "$PROCESSED_DIR" ]]; then
  log "Processed basenames directory not found: $PROCESSED_DIR"
  exit 1
fi

if [[ ! -d "$STATIONS_ROOT" ]]; then
  log "Stations root not found: $STATIONS_ROOT"
  exit 1
fi

declare -A PROCESSED=()
while IFS= read -r -d '' csv_file; do
  while IFS=, read -r base _rest; do
    base=${base//$'\r'/}
    [[ -z "$base" || "$base" == "basename" ]] && continue
    PROCESSED["$base"]=1
  done < "$csv_file"
done < <(find "$PROCESSED_DIR" -type f -name '*_processed_basenames.csv' -print0)

if (( ${#PROCESSED[@]} == 0 )); then
  log "No processed basenames loaded from $PROCESSED_DIR"
  exit 1
fi
log "Loaded ${#PROCESSED[@]} processed basenames"

dir_count=0
file_count=0
removed_count=0
declare -A MATCH_COUNTS=()

while IFS= read -r -d '' unprocessed_dir; do
((++dir_count))
  while IFS= read -r -d '' file_path; do
    ((++file_count))
    filename="$(basename "$file_path")"
    stem="${filename%.*}"
    candidate="$stem"
    if [[ "$stem" == *_* ]]; then
      candidate="${stem#*_}"
    fi
    candidate=${candidate//$'\r'/}
    if [[ -n ${PROCESSED[$candidate]:-} ]]; then
      station="$(station_from_path "$file_path")"
      ((++MATCH_COUNTS["$station"]))
      if $DRY_RUN; then
        log "[DRY-RUN] Would remove $file_path"
      else
        rm -f "$file_path"
        log "Removed $file_path"
      fi
      ((++removed_count))
    fi
  done < <(find "$unprocessed_dir" -type f -print0)
done < <(find "$STATIONS_ROOT" -type d \( -name 'UNPROCESSED' -o -name 'UNPROCESSED_DIRECTORY' \) -print0)

if (( ${#MATCH_COUNTS[@]} > 0 )); then
  log "Per-station files to flush (dry-run: $DRY_RUN):"
  while IFS=$'\t' read -r count station; do
    log "  $station: $count"
  done < <(
    for station in "${!MATCH_COUNTS[@]}"; do
      printf '%d\t%s\n' "${MATCH_COUNTS[$station]}" "$station"
    done | sort -rn -k1,1
  )
else
  log "No matching files found in any station."
fi

log "Checked $file_count files across $dir_count UNPROCESSED directories; removed $removed_count files (dry-run: $DRY_RUN)"
