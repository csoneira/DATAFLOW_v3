#!/usr/bin/env bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: OPERATIONS/MAINTENANCE/CLEANERS/clean_dataflow.sh
# Purpose: Clean dataflow.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash OPERATIONS/MAINTENANCE/CLEANERS/clean_dataflow.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

set -euo pipefail

LC_ALL=C
shopt -s dotglob nullglob

COMPACT=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DATAFLOW_ROOT_DEFAULT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
DATAFLOW_ROOT="${DATAFLOW_CLEAN_ROOT:-$DATAFLOW_ROOT_DEFAULT}"
DATAFLOW_PARENT="$(dirname -- "$DATAFLOW_ROOT")"
LOCK_FILE="${DATAFLOW_CLEAN_LOCK_FILE:-/tmp/dataflow_clean_dataflow.lock}"

log_ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_warn() {
  printf '[%s] [CLEAN_DATAFLOW] [WARN] %s\n' "$(log_ts)" "$*" >&2
}

log_info() {
  printf '%s\n' "$*"
}

log_detail() {
  if [[ "$COMPACT" != true ]]; then
    printf '%s\n' "$*"
  fi
}

usage() {
  cat <<'EOF'
clean_dataflow.sh
Unified cleaner for DATAFLOW_v3 artefacts (COMPLETED_DIRECTORY exports, plot bundles, and Stage-0 buffers).

Usage:
  clean_dataflow.sh [--force|-f] [--threshold|-t <percent>] [--select|-s <list>] [--compact|-c]

Options:
  -h, --help             Show this help message and exit.
  -f, --force            Skip the disk usage threshold check.
                         When --select is omitted, defaults to: temps,plots,completed.
  -t, --threshold <pct>  Override the disk usage threshold (0-100, default 50).
  -s, --select <list>    Comma-separated list of cleanups to run (temps,plots,completed,cronlogs).
                         May be repeated. Defaults to all when omitted.
  -c, --compact          Compact output for chat/notification consumers.

Examples:
  clean_dataflow.sh
  clean_dataflow.sh --threshold 65 --select plots,completed
  clean_dataflow.sh --force -s temps
  clean_dataflow.sh --force --compact
EOF
}

DEFAULT_SELECTION=(temps plots completed cronlogs)
FORCE_DEFAULT_SELECTION=(temps plots completed)
declare -A VALID_TYPES=([temps]=1 [plots]=1 [completed]=1 [cronlogs]=1)

STATIONS_BASE="${DATAFLOW_CLEAN_STATIONS_BASE:-$DATAFLOW_ROOT/STATIONS}"
TEMP_ROOTS=(
  "$DATAFLOW_ROOT"
  "${DATAFLOW_CLEAN_SAFE_ROOT:-$DATAFLOW_PARENT/SAFE_DATAFLOW_v3}"
)
CRON_LOG_DIR="${DATAFLOW_CLEAN_CRON_LOG_DIR:-$DATAFLOW_ROOT/OPERATIONS_RUNTIME/CRON_LOGS}"
SIM_JUNK_BASE="${DATAFLOW_CLEAN_SIM_JUNK_BASE:-$DATAFLOW_PARENT/SIMULATION_DATA_JUNK}"
PLOTS_KEEP_FRESHEST="${DATAFLOW_CLEAN_PLOTS_KEEP_FRESHEST:-5}"

declare -A TYPE_BEFORE=()
declare -A TYPE_AFTER=()
declare -A TYPE_FREED=()
declare -A TYPE_COUNTS=()
QUEUE_SIDECAR_BEFORE=0
QUEUE_SIDECAR_AFTER=0
QUEUE_SIDECAR_FREED=0
QUEUE_SIDECAR_COUNT=0

join_by() {
  local sep="$1"
  shift || { printf ""; return 0; }
  local first="$1"
  shift
  printf "%s" "$first"
  for item in "$@"; do
    printf "%s%s" "$sep" "$item"
  done
}

format_bytes() {
  local bytes="${1:-0}"
  if [[ -z "$bytes" ]]; then
    bytes=0
  fi
  local abs=$(( bytes < 0 ? -bytes : bytes ))
  local decimal
  decimal=$(awk -v b="$abs" 'BEGIN{printf "%.3f", b/1000000000}')
  local binary
  binary=$(awk -v b="$abs" 'BEGIN{printf "%.3f", b/1024/1024/1024}')
  if (( bytes < 0 )); then
    printf "-%s GB (-%s GiB)" "$decimal" "$binary"
  else
    printf "%s GB (%s GiB)" "$decimal" "$binary"
  fi
}

validate_threshold() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo "Threshold requires a numeric value." >&2
    exit 1
  fi
  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Threshold must be numeric between 0 and 100: $value" >&2
    exit 1
  fi
  local formatted
  if ! formatted=$(LC_ALL=C awk -v v="$value" 'BEGIN{if (v < 0 || v > 100) exit 1; printf "%.2f", v}'); then
    echo "Threshold must be between 0 and 100: $value" >&2
    exit 1
  fi
  printf "%s" "$formatted"
}

disk_usage_percent() {
  df -P /home | awk 'NR==2 {gsub("%","",$5); print $5}'
}

disk_usage_summary() {
  df -h /home | awk 'NR==2 {printf "%s used (%s / %s)", $5, $3, $2}'
}

label_for_type() {
  case "$1" in
    temps) echo "temps";;
    plots) echo "plots";;
    completed) echo "completed/error directories";;
    cronlogs) echo "cron logs";;
    *) echo "$1";;
  esac
}

is_metadata_path() {
  local path="${1:-}"
  local upper="${path^^}"
  [[ "$upper" == *"/METADATA/"* || "$upper" == *"/METADATA" ]]
}

clean_completed() {
  local type="completed"
  local -a dirs=()
  if [[ ! -d "$STATIONS_BASE" ]]; then
    log_info "Skipping completed cleanup: $STATIONS_BASE not found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  local -a patterns=(
    '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/COMPLETED_DIRECTORY'
    '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/ERROR_DIRECTORY'
    '*/STAGE_1/EVENT_DATA/STEP_1/TASK_1/ANCILLARY/REJECTED_FILES'
    '*/STAGE_1/EVENT_DATA/STEP_2/INPUT_FILES/COMPLETED'
    '*/STAGE_1/EVENT_DATA/STEP_2/INPUT_FILES/ERROR_DIRECTORY'
    '*/STAGE_1/EVENT_DATA/STEP_3/TASK_2/INPUT_FILES/COMPLETED'
    '*/STAGE_1/LAB_LOGS/STEP_*/INPUT_FILES/COMPLETED'
    '*/STAGE_1/LAB_LOGS/STEP_*/INPUT_FILES/ERROR*'
    '*/STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/COMPLETED'
    '*/STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/ERROR*'
  )

  declare -A seen_dirs=()
  for pattern in "${patterns[@]}"; do
    while IFS= read -r -d '' dir; do
      [[ -d "$dir" ]] || continue
      [[ -n ${seen_dirs["$dir"]:-} ]] && continue
      if is_metadata_path "$dir"; then
        log_warn "Skipping metadata path during completed cleanup: $dir"
        continue
      fi
      seen_dirs["$dir"]=1
      dirs+=("$dir")
    done < <(find "$STATIONS_BASE" -type d -path "$pattern" -print0 2>/dev/null)
  done

  if (( ${#dirs[@]} == 0 )); then
    log_info "No completed or error directories found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  # Single batched du before any deletion — avoids 2×N separate du calls
  local total_before
  total_before=$(du -sb "${dirs[@]}" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')

  for dir in "${dirs[@]}"; do
    log_detail "--> Cleaning $dir"
    chmod -R u+w "$dir" 2>/dev/null || true
    find "$dir" -mindepth 1 -delete 2>/dev/null || true
  done

  # Directories are empty after deletion; after ≈ 0
  local total_after=0
  local freed=$((total_before - total_after))
  TYPE_BEFORE["$type"]=$total_before
  TYPE_AFTER["$type"]=$total_after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=${#dirs[@]}

  log_info "Completed/Error directories cleaned: ${#dirs[@]}"
  log_info "   Size before: $(format_bytes "$total_before")"
  log_info "   Size after:  $(format_bytes "$total_after")"
  log_info "   Freed:       $(format_bytes "$freed")"
}

clean_step1_queue_sidecars() {
  local -a files=()
  if [[ ! -d "$STATIONS_BASE" ]]; then
    log_info "Skipping Step-1 queue sidecar cleanup: $STATIONS_BASE not found."
    QUEUE_SIDECAR_BEFORE=0
    QUEUE_SIDECAR_AFTER=0
    QUEUE_SIDECAR_FREED=0
    QUEUE_SIDECAR_COUNT=0
    return 0
  fi

  while IFS= read -r -d '' file; do
    [[ -f "$file" ]] || continue
    files+=("$file")
  done < <(
    find "$STATIONS_BASE" -type f \
      \( \
        -path '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/UNPROCESSED_DIRECTORY/removed_channel_values_*' -o \
        -path '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/UNPROCESSED_DIRECTORY/removed_rows_*' -o \
        -path '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/PROCESSING_DIRECTORY/removed_channel_values_*' -o \
        -path '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/PROCESSING_DIRECTORY/removed_rows_*' \
      \) -print0 2>/dev/null
  )

  if (( ${#files[@]} == 0 )); then
    log_info "No misplaced Step-1 queue sidecars found."
    QUEUE_SIDECAR_BEFORE=0
    QUEUE_SIDECAR_AFTER=0
    QUEUE_SIDECAR_FREED=0
    QUEUE_SIDECAR_COUNT=0
    return 0
  fi

  local total_before
  total_before=$(du -cb "${files[@]}" 2>/dev/null | awk 'END{print $1+0}')

  for file in "${files[@]}"; do
    log_detail "--> Removing misplaced queue sidecar $file"
    chmod u+w "$file" 2>/dev/null || true
    rm -f "$file" 2>/dev/null || true
  done

  QUEUE_SIDECAR_BEFORE=$total_before
  QUEUE_SIDECAR_AFTER=0
  QUEUE_SIDECAR_FREED=$total_before
  QUEUE_SIDECAR_COUNT=${#files[@]}

  log_info "Misplaced Step-1 queue sidecars cleaned: ${#files[@]}"
  log_info "   Size before: $(format_bytes "$QUEUE_SIDECAR_BEFORE")"
  log_info "   Size after:  $(format_bytes "$QUEUE_SIDECAR_AFTER")"
  log_info "   Freed:       $(format_bytes "$QUEUE_SIDECAR_FREED")"
}

clean_cronlogs() {
  local type="cronlogs"
  local dir="$CRON_LOG_DIR"

  if [[ ! -d "$dir" ]]; then
    log_info "Cron logs directory not found: $dir"
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  local before after freed count failed
  before=$(du -sb "$dir" | awk '{print $1}')
  count=0
  failed=0

  log_detail "--> Truncating cron log files under $dir (keeping paths/inodes)"
  chmod -R u+w "$dir" >/dev/null 2>&1 || true
  while IFS= read -r -d '' file; do
    if : >"$file" 2>/dev/null; then
      count=$((count + 1))
    else
      failed=$((failed + 1))
    fi
  done < <(find "$dir" -type f -print0)

  after=$(du -sb "$dir" | awk '{print $1}')
  freed=$((before - after))

  TYPE_BEFORE["$type"]=$before
  TYPE_AFTER["$type"]=$after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=${count:-0}

  log_info "Cron logs truncated: ${count:-0} file(s)"
  if (( failed > 0 )); then
    log_info "   Failed to truncate: ${failed} file(s)"
  fi
  log_info "   Size before: $(format_bytes "$before")"
  log_info "   Size after:  $(format_bytes "$after")"
  log_info "   Freed:       $(format_bytes "$freed")"
}

clean_plots() {
  local type="plots"
  local -a dirs=()
  declare -A seen_dirs=()
  if [[ ! -d "$STATIONS_BASE" ]]; then
    log_info "Skipping plots cleanup: $STATIONS_BASE not found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  # Single find covering both PLOTS dirs and DEBUG_PLOTS fallback in one tree walk.
  # DEBUG_PLOTS entries whose parent PLOTS is already queued are removed below.
  while IFS= read -r -d '' dir; do
    [[ -n ${seen_dirs["$dir"]:-} ]] && continue
    if is_metadata_path "$dir"; then
      log_warn "Skipping metadata path during plots cleanup: $dir"
      continue
    fi
    seen_dirs["$dir"]=1
    dirs+=("$dir")
  done < <(find "$STATIONS_BASE" -maxdepth 10 -type d \
    \( -name 'PLOTS' -o -path '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/PLOTS/DEBUG_PLOTS' \) \
    -print0 2>/dev/null)

  # Post-filter: drop any DEBUG_PLOTS whose parent PLOTS is also in the list
  local -a filtered_dirs=()
  local parent_plot_dir
  for dir in "${dirs[@]}"; do
    if [[ "$dir" == */DEBUG_PLOTS ]]; then
      parent_plot_dir="${dir%/DEBUG_PLOTS}"
      if [[ -n ${seen_dirs["$parent_plot_dir"]:-} ]]; then
        continue
      fi
    fi
    filtered_dirs+=("$dir")
  done
  dirs=("${filtered_dirs[@]}")

  if [[ ! "$PLOTS_KEEP_FRESHEST" =~ ^[0-9]+$ ]]; then
    log_warn "Invalid DATAFLOW_CLEAN_PLOTS_KEEP_FRESHEST=$PLOTS_KEEP_FRESHEST. Falling back to 5."
    PLOTS_KEEP_FRESHEST=5
  fi

  if (( ${#dirs[@]} == 0 )); then
    log_info "No PLOTS directories found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  # Single batched du before deletion — avoids 2×N separate du calls
  local total_before
  total_before=$(du -sb "${dirs[@]}" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
  local total_deleted=0
  local kept_per_dir=$PLOTS_KEEP_FRESHEST

  for dir in "${dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
      log_warn "Skipping vanished plots path: $dir"
      continue
    fi
    log_detail "--> Cleaning $dir (keep newest $kept_per_dir file(s))"
    chmod -R u+w "$dir" 2>/dev/null || true

    local -a ranked_entries=()
    local entry path
    local idx=0
    while IFS= read -r -d '' entry; do
      ranked_entries+=("$entry")
    done < <(find "$dir" -type f -printf '%T@|%p\0' 2>/dev/null | sort -z -t '|' -k1,1nr)

    if (( ${#ranked_entries[@]} <= kept_per_dir )); then
      continue
    fi

    for entry in "${ranked_entries[@]}"; do
      path="${entry#*|}"
      if (( idx < kept_per_dir )); then
        idx=$((idx + 1))
        continue
      fi
      if rm -f -- "$path" 2>/dev/null; then
        total_deleted=$((total_deleted + 1))
      else
        log_warn "Failed to remove old plot file: $path"
      fi
      idx=$((idx + 1))
    done

    # Remove empty subdirectories left after trimming old files.
    find "$dir" -mindepth 1 -type d -empty -delete 2>/dev/null || true
  done

  local total_after
  total_after=$(du -sb "${dirs[@]}" 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
  local freed=$((total_before - total_after))
  TYPE_BEFORE["$type"]=$total_before
  TYPE_AFTER["$type"]=$total_after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=${#dirs[@]}

  log_info "PLOTS directories cleaned: ${#dirs[@]} (kept newest $kept_per_dir file(s) per directory)"
  log_info "   Old files deleted: $total_deleted"
  log_info "   Total before: $(format_bytes "$total_before")"
  log_info "   Total after:  $(format_bytes "$total_after")"
  log_info "   Total freed:  $(format_bytes "$freed")"
}

clean_temps() {
  local type="temps"
  local -a roots=("${TEMP_ROOTS[@]}")
  local -a rel_targets=(
    "varData/tmp_mi0*"
    "rawData/dat/removed/*"
    "asci/removed/*"
    "varData/*"
    "rawData/dat/done/*"
  )
  local -a patterns=()
  declare -A seen_bases=()

  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue

    # Single find covering both varData and rawData in one tree walk
    while IFS= read -r -d '' found_dir; do
      local base
      base="$(dirname "$found_dir")"
      if [[ -n ${seen_bases["$base"]:-} ]]; then
        continue
      fi
      seen_bases["$base"]=1
      for rel in "${rel_targets[@]}"; do
        patterns+=("$base/$rel")
      done
    done < <(find "$root" -type d \( -name 'varData' -o -path '*/rawData' \) -print0 2>/dev/null)
  done

  if (( ${#patterns[@]} == 0 )); then
    log_info "No Stage_0 data buffers found under DATAFLOW_v3 or SAFE_DATAFLOW_v3."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  declare -A seen_paths=()
  local total_before=0
  local total_after=0
  local removed=0

  for pattern in "${patterns[@]}"; do
    while IFS= read -r match; do
      [[ -n "$match" ]] || continue
      if [[ -n ${seen_paths["$match"]:-} ]]; then
        continue
      fi
      seen_paths["$match"]=1

      if [[ ! -e "$match" ]]; then
        continue
      fi
      if is_metadata_path "$match"; then
        log_warn "Skipping metadata path during temp cleanup: $match"
        continue
      fi

      local before after delta
      before=$(du -sb "$match" | awk '{print $1}')
      total_before=$((total_before + before))
      log_detail "--> Removing $match"
      chmod -R u+w "$match" >/dev/null 2>&1 || true
      if ! rm -rf -- "$match"; then
        log_warn "unable to remove $match (check permissions)"
      fi
      if [[ ! -e "$match" ]]; then
        ((++removed))
      fi

      if [[ -e "$match" ]]; then
        after=$(du -sb "$match" | awk '{print $1}')
      else
        after=0
      fi
      total_after=$((total_after + after))
      delta=$((before - after))
      log_detail "   Freed $(format_bytes "$delta")"
      if [[ -e "$match" ]]; then
        log_warn "Item still present after cleanup: $match"
      fi
    done < <(compgen -G "$pattern" || true)
  done

  local freed=$((total_before - total_after))
  TYPE_BEFORE["$type"]=$total_before
  TYPE_AFTER["$type"]=$total_after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=$removed

  if (( removed == 0 )); then
    log_info "No temporary files found to delete."
  else
    log_info "Temporary artefacts removed: $removed item(s)"
    log_info "   Total before: $(format_bytes "$total_before")"
    log_info "   Total after:  $(format_bytes "$total_after")"
    log_info "   Total freed:  $(format_bytes "$freed")"
  fi
}

FORCE=false
THRESHOLD=$(validate_threshold "50")
declare -a SELECTION_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--force)
      FORCE=true
      shift
      ;;
    -c|--compact)
      COMPACT=true
      shift
      ;;
    -t|--threshold)
      if [[ $# -lt 2 ]]; then
        echo "Option --threshold requires a value." >&2
        exit 1
      fi
      THRESHOLD=$(validate_threshold "$2")
      shift 2
      ;;
    -s|--select)
      if [[ $# -lt 2 ]]; then
        echo "Option --select requires a value." >&2
        exit 1
      fi
      SELECTION_ARGS+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

declare -a SELECTED_TYPES=()
declare -A SEEN_TYPES=()

if ((${#SELECTION_ARGS[@]} == 0)); then
  if [[ "$FORCE" == true ]]; then
    for type in "${FORCE_DEFAULT_SELECTION[@]}"; do
      SELECTED_TYPES+=("$type")
      SEEN_TYPES["$type"]=1
    done
  else
    for type in "${DEFAULT_SELECTION[@]}"; do
      SELECTED_TYPES+=("$type")
      SEEN_TYPES["$type"]=1
    done
  fi
else
  for entry in "${SELECTION_ARGS[@]}"; do
    IFS=',' read -ra tokens <<<"$entry"
    for token in "${tokens[@]}"; do
      token=${token,,}
      token=${token// /}
      [[ -z "$token" ]] && continue
      if [[ "$token" == "all" ]]; then
        for t in "${DEFAULT_SELECTION[@]}"; do
          if [[ -z ${SEEN_TYPES["$t"]:-} ]]; then
            SELECTED_TYPES+=("$t")
            SEEN_TYPES["$t"]=1
          fi
        done
        continue
      fi
      if [[ -z ${VALID_TYPES["$token"]:-} ]]; then
        echo "Unknown value for --select: $token" >&2
        exit 1
      fi
      if [[ -z ${SEEN_TYPES["$token"]:-} ]]; then
        SELECTED_TYPES+=("$token")
        SEEN_TYPES["$token"]=1
      fi
    done
  done
  if ((${#SELECTED_TYPES[@]} == 0)); then
    for type in "${DEFAULT_SELECTION[@]}"; do
      SELECTED_TYPES+=("$type")
      SEEN_TYPES["$type"]=1
    done
  fi
fi

# Acquire exclusive lock to prevent concurrent runs (LOCK_FILE was declared above)
exec {LOCK_FD}>"$LOCK_FILE" 2>/dev/null || { log_warn "Cannot open lock file: $LOCK_FILE"; exit 1; }
if ! flock -n "$LOCK_FD"; then
  log_warn "Another clean_dataflow instance is already running (lock: $LOCK_FILE). Exiting."
  exit 0
fi
trap 'flock -u "$LOCK_FD" 2>/dev/null || true; rm -f "$LOCK_FILE" 2>/dev/null || true' EXIT

log_info "Selected cleanups: $(join_by ', ' "${SELECTED_TYPES[@]}")"
log_info "Disk usage before cleaning: $(disk_usage_summary)"

if [[ "$FORCE" == true ]]; then
  if ((${#SELECTION_ARGS[@]} == 0)); then
    log_info "Force default selection active: $(join_by ', ' "${FORCE_DEFAULT_SELECTION[@]}")"
  fi
  log_info "Force flag enabled; skipping disk usage threshold check."
else
  usage_percent=$(disk_usage_percent)
  if [[ -z "$usage_percent" ]]; then
    echo "Unable to determine disk usage for /home." >&2
    exit 1
  fi
  log_info "Threshold: ${THRESHOLD}%"
  should_clean=$(awk -v usage="$usage_percent" -v threshold="$THRESHOLD" 'BEGIN{usage+=0; threshold+=0; if (usage >= threshold) print 1; else print 0}')
  if [[ "$should_clean" -eq 0 ]]; then
    log_info "Disk usage ${usage_percent}% is below the threshold (${THRESHOLD}%). Use --force to override."
    exit 0
  fi
  log_info "Disk usage ${usage_percent}% exceeds threshold ${THRESHOLD}%. Proceeding with cleanup."
fi

log_info ""
log_info "=== Cleaning misplaced Step-1 queue sidecars ==="
clean_step1_queue_sidecars

for type in "${SELECTED_TYPES[@]}"; do
  log_info ""
  case "$type" in
    temps)
      log_info "=== Cleaning Stage-0 temporary buffers ==="
      clean_temps
      ;;
    plots)
      log_info "=== Cleaning plot exports ==="
      clean_plots
      ;;
    completed)
      log_info "=== Cleaning COMPLETED_DIRECTORY exports ==="
      clean_completed
      ;;
    cronlogs)
      log_info "=== Cleaning cron execution logs ==="
      clean_cronlogs
      ;;
  esac
done

overall_before=0
overall_after=0

for type in "${SELECTED_TYPES[@]}"; do
  before=${TYPE_BEFORE["$type"]:-0}
  after=${TYPE_AFTER["$type"]:-0}
  overall_before=$((overall_before + before))
  overall_after=$((overall_after + after))
done
overall_before=$((overall_before + QUEUE_SIDECAR_BEFORE))
overall_after=$((overall_after + QUEUE_SIDECAR_AFTER))

overall_freed=$((overall_before - overall_after))

log_info ""
log_info "Summary:"
log_info "  - Step-1 queue sidecars: $(format_bytes "$QUEUE_SIDECAR_FREED") freed across ${QUEUE_SIDECAR_COUNT} item(s)"
for type in "${SELECTED_TYPES[@]}"; do
  label=$(label_for_type "$type")
  freed=${TYPE_FREED["$type"]:-0}
  count=${TYPE_COUNTS["$type"]:-0}
  log_info "  - ${label}: $(format_bytes "$freed") freed across ${count} item(s)"
done
log_info "  Total reclaimed: $(format_bytes "$overall_freed")"


# Clean SIMULATION_DATA_JUNK MINGO_DIGITAL_TWIN/INTERSTEPS directories
SIM_JUNK_BASE="$HOME/SIMULATION_DATA_JUNK"
if [[ -d "$SIM_JUNK_BASE" ]]; then
  log_info "Cleaning SIMULATION_DATA_JUNK/*/MINGO_DIGITAL_TWIN/INTERSTEPS directories..."
  count=0
  freed=0
  while IFS= read -r -d '' dir; do
    if [[ -d "$dir" ]]; then
      size_before=$(du -sb "$dir" 2>/dev/null | awk '{print $1}')
      rm -rf "$dir" 2>/dev/null || true
      count=$((count + 1))
      freed=$((freed + size_before))
      log_detail "  Removed $dir ($(format_bytes "$size_before"))"
    fi
  done < <(find "$SIM_JUNK_BASE" -mindepth 3 -maxdepth 3 -type d -path '*/MINGO_DIGITAL_TWIN/INTERSTEPS' -print0 2>/dev/null)
  log_info "  - SIMULATION_DATA_JUNK INTERSTEPS: $(format_bytes "$freed") freed across $count item(s)"
else
  log_info "SIMULATION_DATA_JUNK base directory not found: $SIM_JUNK_BASE"
fi

# Clean SIMULATION_DATA_JUNK MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES directories
if [[ -d "$SIM_JUNK_BASE" ]]; then
  log_info "Cleaning SIMULATION_DATA_JUNK/*/MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES directories..."
  count=0
  freed=0
  while IFS= read -r -d '' dir; do
    if [[ -d "$dir" ]]; then
      size_before=$(du -sb "$dir" 2>/dev/null | awk '{print $1}')
      rm -rf "$dir" 2>/dev/null || true
      count=$((count + 1))
      freed=$((freed + size_before))
      log_detail "  Removed $dir ($(format_bytes "$size_before"))"
    fi
  done < <(find "$SIM_JUNK_BASE" -mindepth 4 -maxdepth 4 -type d -path '*/MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES' -print0 2>/dev/null)
  log_info "  - SIMULATION_DATA_JUNK SIMULATED_DATA/FILES: $(format_bytes "$freed") freed across $count item(s)"
else
  log_info "SIMULATION_DATA_JUNK base directory not found: $SIM_JUNK_BASE"
fi

log_info ""
log_info "Disk usage after cleaning: $(disk_usage_summary)"
