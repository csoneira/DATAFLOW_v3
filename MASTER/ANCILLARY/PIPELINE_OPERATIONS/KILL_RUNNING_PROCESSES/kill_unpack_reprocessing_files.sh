#!/usr/bin/env bash
# Kill any running unpack_reprocessing_files.sh processes so cron can restart them.
set -euo pipefail
pattern="unpack_reprocessing_files.sh"
this_pid=$$
mapfile -t pids < <(pgrep -f "$pattern" || true)
if (( ${#pids[@]} == 0 )); then
  echo "No $pattern processes found."
  exit 0
fi
for pid in "${pids[@]}"; do
  [[ "$pid" == "$this_pid" ]] && continue
  if kill "$pid" 2>/dev/null; then
    echo "Sent SIGTERM to PID $pid"
  else
    echo "Failed to kill PID $pid" >&2
  fi
done
