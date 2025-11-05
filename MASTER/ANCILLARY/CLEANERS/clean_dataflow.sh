#!/usr/bin/env bash
set -euo pipefail

LC_ALL=C
shopt -s dotglob nullglob

usage() {
  cat <<'EOF'
clean_completed.sh
Unified cleaner for DATAFLOW_v3 artefacts (COMPLETED_DIRECTORY exports, plot bundles, and Stage-0 buffers).

Usage:
  clean_completed.sh [--force|-f] [--threshold|-t <percent>] [--select|-s <list>]

Options:
  -h, --help             Show this help message and exit.
  -f, --force            Skip the disk usage threshold check.
  -t, --threshold <pct>  Override the disk usage threshold (0-100, default 50).
  -s, --select <list>    Comma-separated list of cleanups to run (temps,plots,completed,cronlogs).
                         May be repeated. Defaults to all when omitted.

Examples:
  clean_completed.sh
  clean_completed.sh --threshold 65 --select plots,completed
  clean_completed.sh --force -s temps
EOF
}

DEFAULT_SELECTION=(temps plots completed cronlogs)
declare -A VALID_TYPES=([temps]=1 [plots]=1 [completed]=1 [cronlogs]=1)

STATIONS_BASE="$HOME/DATAFLOW_v3/STATIONS"
TEMP_ROOTS=(
  "$HOME/DATAFLOW_v3"
  "$HOME/SAFE_DATAFLOW_v3"
)
CRON_LOG_DIR="$HOME/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS"

declare -A TYPE_BEFORE=()
declare -A TYPE_AFTER=()
declare -A TYPE_FREED=()
declare -A TYPE_COUNTS=()

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
  if ! formatted=$(awk -v v="$value" 'BEGIN{if (v < 0 || v > 100) exit 1; printf "%.2f", v}'); then
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
    completed) echo "completed directories";;
    cronlogs) echo "cron logs";;
    *) echo "$1";;
  esac
}

clean_completed() {
  local type="completed"
  local -a dirs=()
  if [[ ! -d "$STATIONS_BASE" ]]; then
    echo "Skipping completed cleanup: $STATIONS_BASE not found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  while IFS= read -r -d '' dir; do
    dirs+=("$dir")
  done < <(find "$STATIONS_BASE" -type d -path '*/STAGE_1/EVENT_DATA/STEP_1/TASK_*/INPUT_FILES/COMPLETED_DIRECTORY' -print0 2>/dev/null)

  if (( ${#dirs[@]} == 0 )); then
    echo "No COMPLETED_DIRECTORY directories found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  local total_before=0
  local total_after=0

  for dir in "${dirs[@]}"; do
    local before after delta
    before=$(du -sb "$dir" | awk '{print $1}')
    total_before=$((total_before + before))

    echo "→ Cleaning $dir"
    chmod -R u+w "$dir" >/dev/null 2>&1 || true
    find "$dir" -mindepth 1 -delete

    after=$(du -sb "$dir" | awk '{print $1}')
    total_after=$((total_after + after))
    delta=$((before - after))
    echo "   Freed $(format_bytes "$delta")"
  done

  local freed=$((total_before - total_after))
  TYPE_BEFORE["$type"]=$total_before
  TYPE_AFTER["$type"]=$total_after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=${#dirs[@]}

  echo "Completed directories cleaned: ${#dirs[@]}"
  echo "   Size before: $(format_bytes "$total_before")"
  echo "   Size after:  $(format_bytes "$total_after")"
  echo "   Freed:       $(format_bytes "$freed")"
}

clean_cronlogs() {
  local type="cronlogs"
  local dir="$CRON_LOG_DIR"

  if [[ ! -d "$dir" ]]; then
    echo "Cron logs directory not found: $dir"
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  local before after freed count
  before=$(du -sb "$dir" | awk '{print $1}')
  count=$(find "$dir" -mindepth 1 -print | wc -l | awk '{print $1}')

  echo "→ Cleaning $dir"
  chmod -R u+w "$dir" >/dev/null 2>&1 || true
  find "$dir" -mindepth 1 -delete

  after=$(du -sb "$dir" | awk '{print $1}')
  freed=$((before - after))

  TYPE_BEFORE["$type"]=$before
  TYPE_AFTER["$type"]=$after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=${count:-0}

  echo "Cron logs cleared: ${count:-0} item(s)"
  echo "   Size before: $(format_bytes "$before")"
  echo "   Size after:  $(format_bytes "$after")"
  echo "   Freed:       $(format_bytes "$freed")"
}

clean_plots() {
  local type="plots"
  local -a dirs=()
  if [[ ! -d "$STATIONS_BASE" ]]; then
    echo "Skipping plots cleanup: $STATIONS_BASE not found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  while IFS= read -r -d '' dir; do
    dirs+=("$dir")
  done < <(find "$STATIONS_BASE" -maxdepth 8 -type d -name 'PLOTS' -print0 2>/dev/null)

  if (( ${#dirs[@]} == 0 )); then
    echo "No PLOTS directories found."
    TYPE_BEFORE["$type"]=0
    TYPE_AFTER["$type"]=0
    TYPE_FREED["$type"]=0
    TYPE_COUNTS["$type"]=0
    return 0
  fi

  local total_before=0
  local total_after=0

  for dir in "${dirs[@]}"; do
    local before after delta
    before=$(du -sb "$dir" | awk '{print $1}')
    total_before=$((total_before + before))

    echo "→ Cleaning $dir"
    chmod -R u+w "$dir" >/dev/null 2>&1 || true
    find "$dir" -mindepth 1 -delete

    after=$(du -sb "$dir" | awk '{print $1}')
    total_after=$((total_after + after))
    delta=$((before - after))
    echo "   Size before: $(format_bytes "$before")"
    echo "   Size after:  $(format_bytes "$after")"
    echo "   Freed:       $(format_bytes "$delta")"
  done

  local freed=$((total_before - total_after))
  TYPE_BEFORE["$type"]=$total_before
  TYPE_AFTER["$type"]=$total_after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=${#dirs[@]}

  echo "PLOTS directories cleaned: ${#dirs[@]}"
  echo "   Total before: $(format_bytes "$total_before")"
  echo "   Total after:  $(format_bytes "$total_after")"
  echo "   Total freed:  $(format_bytes "$freed")"
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

    while IFS= read -r -d '' var_dir; do
      local base
      base="$(dirname "$var_dir")"
      if [[ -n ${seen_bases["$base"]:-} ]]; then
        continue
      fi
      seen_bases["$base"]=1
      for rel in "${rel_targets[@]}"; do
        patterns+=("$base/$rel")
      done
    done < <(find "$root" -type d -name 'varData' -print0 2>/dev/null)

    while IFS= read -r -d '' raw_dir; do
      local base
      base="$(dirname "$raw_dir")"
      if [[ -n ${seen_bases["$base"]:-} ]]; then
        continue
      fi
      seen_bases["$base"]=1
      for rel in "${rel_targets[@]}"; do
        patterns+=("$base/$rel")
      done
    done < <(find "$root" -type d -path '*/rawData' -print0 2>/dev/null)
  done

  if (( ${#patterns[@]} == 0 )); then
    echo "No Stage_0 data buffers found under DATAFLOW_v3 or SAFE_DATAFLOW_v3."
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

      local before after delta
      before=$(du -sb "$match" | awk '{print $1}')
      total_before=$((total_before + before))
      echo "→ Removing $match"
      chmod -R u+w "$match" >/dev/null 2>&1 || true
      if ! rm -rf -- "$match"; then
        echo "   Warning: unable to remove $match (check permissions)" >&2
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
      echo "   Freed $(format_bytes "$delta")"
      if [[ -e "$match" ]]; then
        echo "   Item still present; manual follow-up may be required." >&2
      fi
    done < <(compgen -G "$pattern" || true)
  done

  local freed=$((total_before - total_after))
  TYPE_BEFORE["$type"]=$total_before
  TYPE_AFTER["$type"]=$total_after
  TYPE_FREED["$type"]=$freed
  TYPE_COUNTS["$type"]=$removed

  if (( removed == 0 )); then
    echo "No temporary files found to delete."
  else
    echo "Temporary artefacts removed: $removed item(s)"
    echo "   Total before: $(format_bytes "$total_before")"
    echo "   Total after:  $(format_bytes "$total_after")"
    echo "   Total freed:  $(format_bytes "$freed")"
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
  for type in "${DEFAULT_SELECTION[@]}"; do
    SELECTED_TYPES+=("$type")
    SEEN_TYPES["$type"]=1
  done
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

if [[ "$FORCE" == true ]]; then
  SELECTED_TYPES=("${DEFAULT_SELECTION[@]}")
fi

echo "Selected cleanups: $(join_by ', ' "${SELECTED_TYPES[@]}")"
echo "Disk usage before cleaning: $(disk_usage_summary)"

if [[ "$FORCE" == true ]]; then
  echo "Force flag enabled; cleaning all cleanup types and skipping disk usage threshold check."
else
  usage_percent=$(disk_usage_percent)
  if [[ -z "$usage_percent" ]]; then
    echo "Unable to determine disk usage for /home." >&2
    exit 1
  fi
  echo "Threshold: ${THRESHOLD}%"
  should_clean=$(awk -v usage="$usage_percent" -v threshold="$THRESHOLD" 'BEGIN{if (usage >= threshold) print 1; else print 0}')
  if [[ "$should_clean" -eq 0 ]]; then
    echo "Disk usage ${usage_percent}% is below the threshold (${THRESHOLD}%). Use --force to override."
    exit 0
  fi
  echo "Disk usage ${usage_percent}% exceeds threshold ${THRESHOLD}%. Proceeding with cleanup."
fi

for type in "${SELECTED_TYPES[@]}"; do
  echo
  case "$type" in
    temps)
      echo "=== Cleaning Stage-0 temporary buffers ==="
      clean_temps
      ;;
    plots)
      echo "=== Cleaning plot exports ==="
      clean_plots
      ;;
    completed)
      echo "=== Cleaning COMPLETED_DIRECTORY exports ==="
      clean_completed
      ;;
    cronlogs)
      echo "=== Cleaning cron execution logs ==="
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

overall_freed=$((overall_before - overall_after))

echo
echo "Summary:"
for type in "${SELECTED_TYPES[@]}"; do
  label=$(label_for_type "$type")
  freed=${TYPE_FREED["$type"]:-0}
  count=${TYPE_COUNTS["$type"]:-0}
  echo "  - ${label}: $(format_bytes "$freed") freed across ${count} item(s)"
done
echo "  Total reclaimed: $(format_bytes "$overall_freed")"

echo
echo "Disk usage after cleaning: $(disk_usage_summary)"
