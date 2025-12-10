#!/usr/bin/env bash
# Purge cron log files under EXECUTION_LOGS/CRON_LOGS.
set -euo pipefail

usage() {
  cat <<'EOF'
erase_cron_logs.sh [--dry-run]

Deletes every file inside ~/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS to keep the cron
logs directory from growing indefinitely.

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
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

LOG_DIR="/home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS"

if [[ ! -d "$LOG_DIR" ]]; then
  echo "Cron logs directory not found: $LOG_DIR" >&2
  exit 1
fi

mapfile -t files < <(find "$LOG_DIR" -mindepth 1 -maxdepth 1 -type f -print)

if (( ${#files[@]} == 0 )); then
  echo "No files to remove under $LOG_DIR."
  exit 0
fi

echo "Found ${#files[@]} file(s) in $LOG_DIR."

for path in "${files[@]}"; do
  if $DRY_RUN; then
    echo "[DRY-RUN] Would remove $path"
  else
    rm -f "$path"
    echo "Removed $path"
  fi
done

echo "Cron log cleanup completed (dry-run: $DRY_RUN)."
