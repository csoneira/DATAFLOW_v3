#!/bin/bash

# log_file="${LOG_FILE:-~/cron_logs/bring_and_analyze_events_${station}.log}"
# mkdir -p "$(dirname "$log_file")"

# Station specific -----------------------------
if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
raw_to_list_events.sh
Launches the STAGE_0_to_1-to-LIST processing stage for a single station.

Usage:
  raw_to_list_events.sh <station>

Options:
  -h, --help    Show this help message and exit.

The script schedules the STAGE_0_to_1-LIST pipeline for the given station (1-4),
ensuring only one instance runs concurrently per station and updates status
tracking as files move through the queue.
EOF
  exit 0
fi

if [ -z "$1" ]; then
  echo "Error: No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

# echo '------------------------------------------------------'
# echo "bring_and_analyze_events.sh started on: $(date '+%Y-%m-%d %H:%M:%S')"

station=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
    MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""
# STATUS_CSV=""

# If $1 is not 1, 2, 3, 4, exit
if [[ ! "$station" =~ ^[1-4]$ ]]; then
  echo "Error: Invalid station number. Please provide a number between 1 and 4."
  exit 1
fi

# echo "Station: $station"
# ----------------------------------------------


# --------------------------------------------------------------------------------------------
# Prevent overlapping runs per station using a lock ------------------------------------------
# --------------------------------------------------------------------------------------------
LOCK_DIR="$HOME/DATAFLOW_v3/EXECUTION_LOGS/LOCKS"
mkdir -p "$LOCK_DIR"
LOCK_FILE="$LOCK_DIR/step1_station_${station}.lock"

# Open FD 9 for flock; closing FD releases lock automatically
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date)] Another STEP_1 pipeline is already running for station $station (lock: $LOCK_FILE). Exiting."
  exit 0
fi

cleanup_lock() {
  flock -u 9
  rm -f "$LOCK_FILE"
}
trap cleanup_lock EXIT

echo "------------------------------------------------------"
echo "raw_to_list_events.sh started on: $(date)"
echo "Station: $station"
echo "Lock acquired at $LOCK_FILE"
echo "------------------------------------------------------"


# dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# # Define base working directory
# station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station"
base_working_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station/STAGE_1/EVENT_DATA"

mkdir -p "$base_working_directory"
# STATUS_CSV="$base_working_directory/raw_to_list_events_status.csv"
# if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
#     echo "Warning: unable to record status in $STATUS_CSV" >&2
#     STATUS_TIMESTAMP=""
# fi

# finish() {
#     local exit_code="$1"
#     if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
#         python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
#     fi
# }

# trap 'finish $?' EXIT

# # Additional paths
# mingo_direction="mingo0$station"

TASK_SCRIPTS=(
  "$SCRIPT_DIR/TASK_1/script_1_raw_to_clean.py"
  "$SCRIPT_DIR/TASK_2/script_2_clean_to_cal.py"
  "$SCRIPT_DIR/TASK_3/script_3_cal_to_list.py"
  "$SCRIPT_DIR/TASK_4/script_4_list_to_fit.py"
  "$SCRIPT_DIR/TASK_5/script_5_fit_to_corr.py"
)

max_cpu_usage_pct() {
  # Returns overall CPU utilization over ~1s window (any core peak is close to overall when busy)
  local line1 line2
  line1=$(grep '^cpu ' /proc/stat) || { echo 0; return; }
  sleep 1
  line2=$(grep '^cpu ' /proc/stat) || { echo 0; return; }

  local _ u1 n1 s1 i1 w1 irq1 sirq1 st1 stl1 u2 n2 s2 i2 w2 irq2 sirq2 st2 stl2
  read -r _ u1 n1 s1 i1 w1 irq1 sirq1 st1 stl1 _ <<<"$line1"
  read -r _ u2 n2 s2 i2 w2 irq2 sirq2 st2 stl2 _ <<<"$line2"

  local idle=$(( (i2 - i1) + (w2 - w1) ))
  local total=$(( (u2-u1) + (n2-n1) + (s2-s1) + (i2-i1) + (w2-w1) + (irq2-irq1) + (sirq2-sirq1) + (st2-st1) + (stl2-stl1) ))
  [[ $total -le 0 ]] && { echo 0; return; }
  local busy_pct=$(( (100 * (total - idle)) / total ))
  echo "$busy_pct"
}

wait_for_resources() {
  local swap_limit_pct=35
  local swap_limit_kb=$((4 * 1024 * 1024)) # 4 GB
  while true; do
    read -r mem_total mem_avail swap_total swap_free < <(awk '/MemTotal:/ {t=$2} /MemAvailable:/ {a=$2} /SwapTotal:/ {st=$2} /SwapFree:/ {sf=$2} END {print t, a, st, sf}' /proc/meminfo)
    if [[ -z "${mem_total:-}" || -z "${mem_avail:-}" || "$mem_total" -eq 0 ]]; then
      echo "Warning: unable to read memory info; continuing."
      return
    fi
    mem_used_pct=$(( (100 * (mem_total - mem_avail)) / mem_total ))
    swap_used_pct=0
    swap_used_kb=0
    if [[ -n "${swap_total:-}" && "$swap_total" -gt 0 ]]; then
      swap_used_pct=$(( (100 * (swap_total - swap_free)) / swap_total ))
      swap_used_kb=$((swap_total - swap_free))
    fi

    max_cpu_pct=$(max_cpu_usage_pct || echo 0)
    if [[ -z "${max_cpu_pct:-}" ]]; then
      max_cpu_pct=0
    fi

    if (( mem_used_pct < 90 && swap_used_pct < swap_limit_pct && swap_used_kb < swap_limit_kb && max_cpu_pct < 95 )); then
      echo "Resources OK: Mem ${mem_used_pct}% (avail ${mem_avail}k/${mem_total}k) / Swap ${swap_used_pct}% (${swap_used_kb}k used) / Max CPU ${max_cpu_pct}%."
      return
    fi

    echo "Waiting: Mem ${mem_used_pct}% (avail ${mem_avail}k/${mem_total}k, limit <90), Swap ${swap_used_pct}% (${swap_used_kb}k used, limits <${swap_limit_pct}% and <${swap_limit_kb}k), Max CPU ${max_cpu_pct}% (limit <95)."
    sleep 15
  done
}

is_task_running() {
  local script_path="$1"
  while IFS= read -r line; do
    # Each line contains "PID COMMAND"; we only need the command portion.
    local cmd="${line#* }"
    if [[ "$cmd" == *"$script_path"* && "$cmd" == *" $station"* ]]; then
      echo "$line"
      return 0
    fi
  done < <(ps -eo pid=,args=)
  return 1
}

echo '------------------------------------------------------'
echo '------------------------------------------------------'

iteration=1
while true; do
  echo "Pipeline iteration $iteration started at: $(date '+%Y-%m-%d %H:%M:%S')"
  for task_script in "${TASK_SCRIPTS[@]}"; do
    if [[ ! -x "$task_script" ]]; then
      echo "Warning: task script $task_script not found or not executable. Skipping."
      continue
    fi

    if running_line=$(is_task_running "$task_script"); then
      echo "Skipping $(basename "$task_script") because it is already running: $running_line"
      echo '------------------------------------------------------'
      continue
    fi

    wait_for_resources
    echo "Running $(basename "$task_script")..."
    if ! python3 -u "$task_script" "$station"; then
      echo "Task $(basename "$task_script") failed; aborting pipeline."
      exit 1
    fi
    echo '------------------------------------------------------'
  done
  echo "Pipeline iteration $iteration completed at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo '------------------------------------------------------'
  iteration=$((iteration + 1))
done
