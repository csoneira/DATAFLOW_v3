#!/usr/bin/env bash
# =============================================================================
# Script: LOG_BACKUP/install_log_backup_cron.sh
# Purpose: Install a 12-hour cron entry for the log backup runner.
# =============================================================================

set -euo pipefail

show_help() {
  cat <<'EOF'
install_log_backup_cron.sh

Installs a cron entry for LOG_BACKUP/run_log_backup.sh.

Usage:
  bash install_log_backup_cron.sh [options]

Options:
  --schedule SPEC       Cron schedule. Default: 0 */12 * * *
  --runner PATH         Override runner path.
  --log-path PATH       Override cron stdout/stderr log path.
  --print-only          Print the cron line without installing it.
  -h, --help            Show this help text.
EOF
}

require_option_value() {
  local option_name="$1"
  local option_value="${2:-}"
  if [[ -z "$option_value" ]]; then
    echo "Missing value for $option_name" >&2
    exit 1
  fi
}

BACKUP_ROOT="${LOG_BACKUP_ROOT:-$HOME/DATAFLOW_v3/LOG_BACKUP}"
RUNNER_PATH="${LOG_BACKUP_RUNNER_PATH:-$BACKUP_ROOT/run_log_backup.sh}"
CRON_LOG_PATH="${LOG_BACKUP_CRON_LOG_PATH:-$BACKUP_ROOT/runtime/cron/log_backup.cron.log}"
CRON_SCHEDULE="${LOG_BACKUP_CRON_SCHEDULE:-0 */12 * * *}"
PRINT_ONLY=false
CRON_MARKER="# DATAFLOW_v3_LOG_BACKUP"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --schedule)
      require_option_value "$1" "${2:-}"
      CRON_SCHEDULE="$2"
      shift 2
      ;;
    --runner)
      require_option_value "$1" "${2:-}"
      RUNNER_PATH="$2"
      shift 2
      ;;
    --log-path)
      require_option_value "$1" "${2:-}"
      CRON_LOG_PATH="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$RUNNER_PATH" ]]; then
  echo "Runner not found: $RUNNER_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$CRON_LOG_PATH")"

CRON_LINE="$CRON_SCHEDULE /usr/bin/env bash $RUNNER_PATH >> $CRON_LOG_PATH 2>&1 $CRON_MARKER"

if $PRINT_ONLY; then
  printf '%s\n' "$CRON_LINE"
  exit 0
fi

existing_crontab="$(mktemp)"
new_crontab="$(mktemp)"
trap 'rm -f "$existing_crontab" "$new_crontab"' EXIT

crontab -l 2>/dev/null > "$existing_crontab" || true
grep -Fv "$CRON_MARKER" "$existing_crontab" > "$new_crontab" || true
printf '%s\n' "$CRON_LINE" >> "$new_crontab"
crontab "$new_crontab"

printf 'Installed cron entry:\n%s\n' "$CRON_LINE"
