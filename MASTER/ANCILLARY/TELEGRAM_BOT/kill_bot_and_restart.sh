#!/usr/bin/env bash
# kill_bot_and_restart.sh
# Edit SERVICE_NAME or BOT_CMD / BOT_NAME below to match your bot setup.

set -euo pipefail

# ----- CONFIGURE BELOW -----
SERVICE_NAME=""                          # e.g. my-telegram-bot.service (leave empty if not using systemd)
BOT_CMD="/usr/bin/python3 /home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TELEGRAM_BOT/persistent_telegram_bot_check.sh"   # full command to start the bot (used when SERVICE_NAME is empty)
BOT_NAME="mingo_analysis_bot.py"                # pattern used to find running bot with pgrep -f (fallback if BOT_CMD is not unique)
LOG_FILE="/home/mingo/DATAFLOW_v3/EXECUTION_LOGS/NOHUP_LOGS/kill_bot_nohup.log"      # where nohup will append logs
WAIT_SEC=1                               # seconds to wait for graceful shutdown
# ----------------------------

die(){ echo "$*" >&2; exit 1; }

if [ -n "$SERVICE_NAME" ] && command -v systemctl >/dev/null 2>&1; then
    echo "Restarting systemd service: $SERVICE_NAME"
    sudo systemctl restart "$SERVICE_NAME"
    exit 0
fi

# Ensure BOT_CMD is set if not using systemd
if [ -z "$BOT_CMD" ]; then
    die "BOT_CMD is not configured. Edit the script and set BOT_CMD or SERVICE_NAME."
fi

PATTERN="$BOT_NAME"
if [ -z "$PATTERN" ]; then
    PATTERN="$BOT_CMD"
fi

# find running pids
pids=$(pgrep -f "$PATTERN" || true)

if [ -n "$pids" ]; then
    echo "Stopping bot processes matching '$PATTERN': $pids"
    kill $pids || true

    # wait for graceful shutdown
    for i in $(seq 1 "$WAIT_SEC"); do
        sleep 1
        remaining=$(pgrep -f "$PATTERN" || true)
        [ -z "$remaining" ] && break
    done

    remaining=$(pgrep -f "$PATTERN" || true)
    if [ -n "$remaining" ]; then
        echo "Processes still running after ${WAIT_SEC}s, forcing kill: $remaining"
        kill -9 $remaining || true
    fi
else
    echo "No running bot processes found for pattern: '$PATTERN'"
fi

# ensure log file directory exists
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

echo "Starting bot with command: $BOT_CMD"
# start in background and detach
nohup bash -c "$BOT_CMD" >>"$LOG_FILE" 2>&1 &

echo "Bot restarted. Logs appended to: $LOG_FILE"