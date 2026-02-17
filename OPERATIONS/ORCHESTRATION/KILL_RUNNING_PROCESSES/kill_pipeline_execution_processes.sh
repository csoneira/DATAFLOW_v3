#!/usr/bin/env bash
# Kill running bash/python scripts referenced in add_to_crontab.info without touching unrelated processes.
set -euo pipefail

BASE_DIR="/home/mingo/DATAFLOW_v3"
CRON_FILE="$BASE_DIR/add_to_crontab.info"
this_pid=$$

if [[ ! -f "$CRON_FILE" ]]; then
  echo "Cron file not found: $CRON_FILE" >&2
  exit 1
fi

declare -A scripts=()
while IFS= read -r line; do
  # Skip comments/blank/env lines
  [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]] && continue
  [[ "$line" == *"="* && "${line%%=*}" != "$line" ]] && continue

  # Strip the first 5 cron fields to get the command
  command=$(echo "$line" | awk '{ $1=$2=$3=$4=$5=""; sub(/^ +/,""); print }')
  [[ -z "$command" ]] && continue

  read -ra parts <<<"$command"
  [[ ${#parts[@]} -eq 0 ]] && continue

  launcher=${parts[0]}
  script=""

  case "$launcher" in
    /bin/bash|bash)
      [[ ${#parts[@]} -ge 2 ]] || continue
      script="${parts[1]}"
      ;;
    python|python3|/usr/bin/python|/usr/bin/python3|/usr/bin/env)
      # Find the first non-flag argument (the script path)
      for ((i=1; i<${#parts[@]}; i++)); do
        token="${parts[$i]}"
        [[ "$token" == "python" || "$token" == "python3" ]] && continue
        [[ "$token" == -* ]] && continue
        script="$token"
        break
      done
      ;;
    *)
      continue
      ;;
  esac

  [[ -z "$script" ]] && continue

  # Normalize relative paths to absolute (assume relative to BASE_DIR)
  if [[ "$script" != /* ]]; then
    script="$BASE_DIR/$script"
  fi

  scripts["$script"]=1
done < "$CRON_FILE"

if (( ${#scripts[@]} == 0 )); then
  echo "No target scripts found in $CRON_FILE."
  exit 0
fi

declare -A kill_list=()
for script in "${!scripts[@]}"; do
  # Look for running processes whose command contains the script path
  while IFS= read -r pid; do
    [[ -z "$pid" || "$pid" == "$this_pid" ]] && continue
    kill_list["$pid"]="$script"
  done < <(pgrep -f -- "$script" || true)
done

if (( ${#kill_list[@]} == 0 )); then
  echo "No matching processes found for scripts in $CRON_FILE."
  exit 0
fi

for pid in "${!kill_list[@]}"; do
  script="${kill_list[$pid]}"
  if kill "$pid" 2>/dev/null; then
    echo "Sent SIGTERM to PID $pid (script match: $script)"
  else
    echo "Failed to kill PID $pid (script match: $script)" >&2
  fi
done

# Add this
echo "Killing the guide_raw_to_corrected.sh process specifically"
sudo pgrep -f 'bash /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh' | xargs -r kill