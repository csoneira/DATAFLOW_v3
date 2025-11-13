#!/usr/bin/env bash
#
# Ensures mingo_analysis_bot.py is running. If not, starts it detached and logs the
# outcome so cron invocations stay silent unless something fails.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
BOT_SCRIPT="${BOT_SCRIPT:-/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TELEGRAM_BOT/mingo_analysis_bot.py}"
LOG_DIR="${LOG_DIR:-/home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/bot_analysis.log}"
PROCESS_PATTERN="${PYTHON_BIN} ${BOT_SCRIPT}"

mkdir -p "$LOG_DIR"

timestamp() {
  date --iso-8601=seconds
}

log() {
  printf '%s %s\n' "$(timestamp)" "$*" >>"$LOG_FILE"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "ERROR: python binary not found at $PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$BOT_SCRIPT" ]]; then
  log "ERROR: bot script missing at $BOT_SCRIPT"
  exit 1
fi

if /usr/bin/pgrep -fx "$PROCESS_PATTERN" >/dev/null 2>&1; then
  log "Bot already running."
  exit 0
fi

log "Bot not running; launching..."
nohup "$PYTHON_BIN" "$BOT_SCRIPT" >>"$LOG_FILE" 2>&1 &
pid=$!
log "Bot started with PID $pid."

exit 0
