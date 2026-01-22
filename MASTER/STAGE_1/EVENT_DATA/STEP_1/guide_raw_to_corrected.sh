#!/bin/bash

# log_file="${LOG_FILE:-~/cron_logs/bring_and_analyze_events_${station}.log}"
# mkdir -p "$(dirname "$log_file")"

# Station specific -----------------------------
original_args=("$@")
original_args_string="$*"
print_help() {
  cat <<'EOF'
guide_raw_to_corrected.sh
Launches the STAGE_1 EVENT_DATA STEP_1 pipeline for a station.

Usage:
  guide_raw_to_corrected.sh [--station <list>] [--task <list>] [--run-anyway]

Options:
  -s, --station   Station numbers (1-4). Comma-separated or space-separated. Default: all.
  -t, --task      Task IDs to run: 1-5 or "all" (default: all tasks).
  --run-anyway    Skip the check that prevents overlapping guide_raw_to_corrected.sh runs.
  -h, --help      Show this help message and exit.

When multiple stations and tasks are provided, tasks are executed in task-major
order (e.g., stations 1,2 and tasks 3,4,5 run as: 1-3, 2-3, 1-4, 2-4, 1-5, 2-5),
looping continuously. If another guide_raw_to_corrected.sh is already running
for a station, that station is skipped for this invocation.
EOF
}

station_filter_raw="all"
task_filter_raw="all"
run_anyway=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--station)
      station_filter_raw="${2:-}"
      shift 2
      ;;
    -t|--task)
      task_filter_raw="${2:-}"
      shift 2
      ;;
    --run-anyway)
      run_anyway=true
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

# # If the time is 00 in minutes, run pgrep -f 'bash /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh' | xargs -r kill
# if [[ "$(date +%M)" == "00" ]]; then
#   echo "Killing old instances of guide_raw_to_corrected.sh"
#   pgrep -f 'bash /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh' | xargs -r kill
# fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
    MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""
# STATUS_CSV=""

TASK_SCRIPTS=(
  "$SCRIPT_DIR/TASK_1/script_1_raw_to_clean.py"
  "$SCRIPT_DIR/TASK_2/script_2_clean_to_cal.py"
  "$SCRIPT_DIR/TASK_3/script_3_cal_to_list.py"
  "$SCRIPT_DIR/TASK_4/script_4_list_to_fit.py"
  "$SCRIPT_DIR/TASK_5/script_5_fit_to_corr.py"
)

TASK_LABELS=(
  "raw_to_clean"
  "clean_to_cal"
  "cal_to_list"
  "list_to_fit"
  "fit_to_corr"
)

TRAFFIC_LIGHT_DIR="$HOME/DATAFLOW_v3/EXECUTION_LOGS/TRAFFIC_LIGHT"
TRAFFIC_QUEUE_FILE="$TRAFFIC_LIGHT_DIR/stage1_step1_station_task_queue.txt"

build_default_queue_order() {
  local -n out_arr=$1
  out_arr=()
  for task_id in "${ALL_TASK_IDS[@]}"; do
    for station in "${ALL_STATIONS[@]}"; do
      out_arr+=("${station}-${task_id}")
    done
  done
}

sanitize_queue_file() {
  local -a default_pairs sanitized
  build_default_queue_order default_pairs

  declare -A valid_map=()
  for pair in "${default_pairs[@]}"; do
    valid_map["$pair"]=1
  done

  sanitized=()
  declare -A seen_pairs=()
  if [[ -s "$TRAFFIC_QUEUE_FILE" ]]; then
    while IFS= read -r line; do
      line="${line//$'\r'/}"
      line="${line//[$'\t ']/}"
      [[ -z "$line" ]] && continue
      if [[ -n "${valid_map[$line]:-}" && -z "${seen_pairs[$line]:-}" ]]; then
        sanitized+=("$line")
        seen_pairs["$line"]=1
      fi
    done <"$TRAFFIC_QUEUE_FILE"
  fi

  for pair in "${default_pairs[@]}"; do
    if [[ -z "${seen_pairs[$pair]:-}" ]]; then
      sanitized+=("$pair")
    fi
  done

  if [[ ${#sanitized[@]} -eq 0 ]]; then
    sanitized=("${default_pairs[@]}")
  fi

  {
    for pair in "${sanitized[@]}"; do
      echo "$pair"
    done
  } >"$TRAFFIC_QUEUE_FILE"
}

traffic_light_queue_init() {
  mkdir -p "$TRAFFIC_LIGHT_DIR"
  sanitize_queue_file
}

load_queue_snapshot() {
  local -n out_arr=$1
  out_arr=()
  [[ -f "$TRAFFIC_QUEUE_FILE" ]] || sanitize_queue_file
  if [[ -f "$TRAFFIC_QUEUE_FILE" ]]; then
    while IFS= read -r line; do
      line="${line//$'\r'/}"
      line="${line//[$'\t ']/}"
      [[ -z "$line" ]] && continue
      out_arr+=("$line")
    done <"$TRAFFIC_QUEUE_FILE"
  fi
}

build_iteration_pairs_from_queue() {
  local -n out_arr=$1
  local -A requested_map=()
  for pair in "${execution_pairs[@]}"; do
    requested_map["$pair"]=0
  done

  local -a queue_snapshot
  load_queue_snapshot queue_snapshot

  out_arr=()
  for pair in "${queue_snapshot[@]}"; do
    if [[ -n "${requested_map[$pair]:-}" ]]; then
      out_arr+=("$pair")
      requested_map["$pair"]=1
    fi
  done

  for pair in "${execution_pairs[@]}"; do
    if [[ "${requested_map[$pair]}" -ne 1 ]]; then
      out_arr+=("$pair")
    fi
  done

  if [[ ${#out_arr[@]} -eq 0 ]]; then
    out_arr=("${execution_pairs[@]}")
  fi
}

rotate_pair_to_queue_end() {
  local pair="$1"
  [[ -f "$TRAFFIC_QUEUE_FILE" ]] || sanitize_queue_file
  local -a snapshot new_order
  snapshot=()
  if [[ -f "$TRAFFIC_QUEUE_FILE" ]]; then
    while IFS= read -r line; do
      line="${line//$'\r'/}"
      line="${line//[$'\t ']/}"
      [[ -z "$line" ]] && continue
      snapshot+=("$line")
    done <"$TRAFFIC_QUEUE_FILE"
  fi

  local moved=false
  for entry in "${snapshot[@]}"; do
    if [[ "$entry" == "$pair" && "$moved" == false ]]; then
      moved=true
      continue
    fi
    new_order+=("$entry")
  done
  new_order+=("$pair")

  {
    for entry in "${new_order[@]}"; do
      echo "$entry"
    done
  } >"$TRAFFIC_QUEUE_FILE"
}

parse_list() {
  local raw="$1"
  local default_array_name="$2"
  local out_var="$3"

  # Use the provided defaults if raw is empty/all
  if [[ -z "$raw" || "$raw" == "all" ]]; then
    eval "$out_var=(\"\${${default_array_name}[@]}\")"
    return
  fi

  local tmp=()
  IFS=', ' read -r -a parts <<<"$raw"
  for p in "${parts[@]}"; do
    [[ -z "$p" ]] && continue
    tmp+=("$p")
  done
  eval "$out_var=(\"\${tmp[@]}\")"
}

ALL_STATIONS=(1 2 3 4)
ALL_TASK_IDS=(1 2 3 4 5)

LOCK_BASE_DIR="$HOME/DATAFLOW_v3/EXECUTION_LOGS/LOCKS/guide_raw_to_corrected"
ARG_LOCK_FILE=""
ARG_LOCK_FD=""
LOCK_SIGNATURE_DESC=""

format_original_args() {
  if (( ${#original_args[@]} == 0 )); then
    printf '[no explicit args]'
    return
  fi
  local rendered=()
  local arg
  for arg in "${original_args[@]}"; do
    rendered+=("$(printf '%q' "$arg")")
  done
  printf '%s' "${rendered[*]}"
}

prepare_argument_signature() {
  local payload=""
  if (( ${#original_args[@]} == 0 )); then
    payload="__default__"
    LOCK_SIGNATURE_DESC="[no explicit args]"
  else
    LOCK_SIGNATURE_DESC="$(format_original_args)"
    payload=$(printf '%s\037' "${original_args[@]}")
  fi
  local hash
  hash=$(printf '%s' "$payload" | md5sum | awk '{print $1}')
  ARG_LOCK_FILE="${LOCK_BASE_DIR}/${hash}.lock"
}

acquire_argument_lock() {
  prepare_argument_signature
  mkdir -p "$LOCK_BASE_DIR"
  exec {ARG_LOCK_FD}> "$ARG_LOCK_FILE"
  if ! flock -n "$ARG_LOCK_FD"; then
    echo "Another guide_raw_to_corrected.sh with identical arguments (${LOCK_SIGNATURE_DESC}) is already running."
    echo "Lock file: ${ARG_LOCK_FILE}"
    echo "Use --run-anyway to override this safety check."
    return 1
  fi
  echo "Acquired run lock for argument set ${LOCK_SIGNATURE_DESC} (lock: ${ARG_LOCK_FILE})."
  return 0
}

release_argument_lock() {
  if [[ -n "${ARG_LOCK_FD:-}" ]]; then
    flock -u "$ARG_LOCK_FD" 2>/dev/null || true
    eval "exec ${ARG_LOCK_FD}>&-" 2>/dev/null || true
    ARG_LOCK_FD=""
    if [[ -n "${ARG_LOCK_FILE:-}" ]]; then
      rm -f "$ARG_LOCK_FILE"
      ARG_LOCK_FILE=""
    fi
  fi
}

# Collect the PID chain (this shell plus ancestors spawned by cron/bash -c) so
# the duplicate detector can ignore them when scanning for real other runs.
SELF_INVOCATION_PIDS=()

collect_self_invocation_pids() {
  local pid="$$"
  local guard=0

  while [[ -n "$pid" && "$pid" =~ ^[0-9]+$ && $pid -gt 1 && $guard -lt 20 ]]; do
    SELF_INVOCATION_PIDS+=("$pid")
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    guard=$((guard + 1))
    [[ -z "$pid" ]] && break
  done

  if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
    SELF_INVOCATION_PIDS+=("$pid")
  fi

  if [[ -n "${PPID:-}" && "$PPID" =~ ^[0-9]+$ ]]; then
    SELF_INVOCATION_PIDS+=("$PPID")
  fi
}

collect_self_invocation_pids

pid_is_self_or_ancestor() {
  local candidate="$1"
  for spid in "${SELF_INVOCATION_PIDS[@]}"; do
    if [[ -n "$spid" && "$candidate" == "$spid" ]]; then
      return 0
    fi
  done
  return 1
}

parse_list "$station_filter_raw" ALL_STATIONS stations_requested
parse_list "$task_filter_raw" ALL_TASK_IDS tasks_requested

validate_stations() {
  local arr=("$@")
  local validated=()
  for s in "${arr[@]}"; do
    if [[ "$s" =~ ^[0-4]$ ]]; then
      validated+=("$s")
    else
      echo "Warning: ignoring invalid station '$s' (must be 1-8)" >&2
    fi
  done
  if [[ ${#validated[@]} -eq 0 ]]; then
    echo "Error: no valid stations to process." >&2
    exit 1
  fi
  stations_requested=("${validated[@]}")
}

validate_tasks() {
  local arr=("$@")
  local validated=()
  for t in "${arr[@]}"; do
    if [[ "$t" =~ ^[1-5]$ ]]; then
      validated+=("$t")
    else
      echo "Warning: ignoring invalid task '$t' (must be 1-5)" >&2
    fi
  done
  if [[ ${#validated[@]} -eq 0 ]]; then
    echo "Error: no valid tasks to process." >&2
    exit 1
  fi
  tasks_requested=("${validated[@]}")
}

validate_stations "${stations_requested[@]}"
validate_tasks "${tasks_requested[@]}"

# Max runtime watchdog (minutes); override with STEP1_MAX_RUNTIME_MIN env
MAX_RUNTIME_MIN=${STEP1_MAX_RUNTIME_MIN:-120}
watchdog_pid=""


cleanup() {
  if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
  fi
  release_argument_lock
}
trap cleanup EXIT

echo "------------------------------------------------------"
echo "guide_raw_to_corrected.sh started on: $(date)"
echo "Stations: ${stations_requested[*]}"
echo "Tasks: ${tasks_requested[*]}"
echo -n "Execution order: "
execution_pairs=()
for task_id in "${tasks_requested[@]}"; do
  for st in "${stations_requested[@]}"; do
    execution_pairs+=("${st}-${task_id}")
  done
done
echo "${execution_pairs[*]}"
echo "------------------------------------------------------"


# dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

config_file="$HOME/DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml"

# Resource gate defaults (overridden by config_global.yaml if present)
mem_limit_pct=90
swap_limit_pct=70
swap_limit_kb=$((4 * 1024 * 1024)) # 4 GB
cpu_limit_pct=95

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

load_resource_limits() {
  local cfg_path="$config_file"
  local result
  if [[ -f "$cfg_path" ]]; then
    if result=$(python3 - "$cfg_path" <<'PY'
import sys
cfg_path = sys.argv[1]
try:
    import yaml
except ImportError:
    print(",,,,".strip(","))
    sys.exit(0)
try:
    data = yaml.safe_load(open(cfg_path)) or {}
except Exception:
    print(",,,,".strip(","))
    sys.exit(0)
node = data.get("event_data_resource_limits") or {}
def val(key):
    v = node.get(key)
    return "" if v is None else v
print(f"{val('mem_used_pct_max')},{val('swap_used_pct_max')},{val('swap_used_kb_max')},{val('cpu_used_pct_max')}")
PY
    ); then
      IFS=',' read -r cfg_mem cfg_swap_pct cfg_swap_kb cfg_cpu <<<"$result"
      if [[ $cfg_mem =~ ^[0-9]+$ ]]; then mem_limit_pct=$cfg_mem; fi
      if [[ $cfg_swap_pct =~ ^[0-9]+$ ]]; then swap_limit_pct=$cfg_swap_pct; fi
      if [[ $cfg_swap_kb =~ ^[0-9]+$ ]]; then swap_limit_kb=$cfg_swap_kb; fi
      if [[ $cfg_cpu =~ ^[0-9]+$ ]]; then cpu_limit_pct=$cfg_cpu; fi
    fi
  fi
}

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
  read -r mem_total mem_avail swap_total swap_free < <(awk '/MemTotal:/ {t=$2} /MemAvailable:/ {a=$2} /SwapTotal:/ {st=$2} /SwapFree:/ {sf=$2} END {print t, a, st, sf}' /proc/meminfo)
  if [[ -z "${mem_total:-}" || -z "${mem_avail:-}" || "$mem_total" -eq 0 ]]; then
    echo "Warning: unable to read memory info; continuing."
    return 0
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

  if (( mem_used_pct < mem_limit_pct && swap_used_pct < swap_limit_pct && swap_used_kb < swap_limit_kb && max_cpu_pct < cpu_limit_pct )); then
    echo "Resources OK: Mem ${mem_used_pct}% (avail ${mem_avail}k/${mem_total}k) / Swap ${swap_used_pct}% (${swap_used_kb}k used) / Max CPU ${max_cpu_pct}%."
    return 0
  fi

  echo "Resources high: Mem ${mem_used_pct}% (avail ${mem_avail}k/${mem_total}k, limit <${mem_limit_pct}), Swap ${swap_used_pct}% (${swap_used_kb}k used, limits <${swap_limit_pct}% and <${swap_limit_kb}k), Max CPU ${max_cpu_pct}% (limit <${cpu_limit_pct}). Skipping this station/task this loop."
  return 1
}

pipeline_already_running() {
  local target_station="$1"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local pid="${line%% *}"
    local cmd="${line#* }"
    [[ -z "$pid" || "$pid" == "$line" ]] && continue
    [[ ! "$pid" =~ ^[0-9]+$ ]] && continue

    if pid_is_self_or_ancestor "$pid"; then
      continue
    fi
    [[ "$cmd" != *"guide_raw_to_corrected.sh"* ]] && continue

    if command_targets_station "$cmd" "$target_station"; then
      echo "$pid $cmd"
      return 0
    fi
  done < <(ps -eo pid=,args=)
  return 1
}

station_value_matches_target() {
  local value="$1"
  local target="$2"
  local lowered="${value,,}"
  [[ -z "$value" ]] && return 1
  if [[ "$lowered" == "all" ]]; then
    return 0
  fi
  IFS=',' read -ra parts <<<"$value"
  local part cleaned
  for part in "${parts[@]}"; do
    cleaned="${part//[^0-9]/}"
    [[ -z "$cleaned" ]] && continue
    if (( 10#$cleaned == 10#$target )); then
      return 0
    fi
  done
  return 1
}

command_targets_station() {
  local cmd="$1"
  local target="$2"
  local expect_value=false
  local saw_station_arg=false
  local -a tokens=()
  # shellcheck disable=SC2206
  read -ra tokens <<< "$cmd"
  local token value
  for token in "${tokens[@]}"; do
    if $expect_value; then
      expect_value=false
      saw_station_arg=true
      [[ -z "$token" ]] && continue
      if station_value_matches_target "$token" "$target"; then
        return 0
      fi
      continue
    fi
    case "$token" in
      --station|-s)
        expect_value=true
        continue
        ;;
      --station=*)
        saw_station_arg=true
        value="${token#*=}"
        if station_value_matches_target "$value" "$target"; then
          return 0
        fi
        ;;
      -s*)
        saw_station_arg=true
        value="${token#-s}"
        if [[ -n "$value" ]] && station_value_matches_target "$value" "$target"; then
          return 0
        fi
        ;;
    esac
  done
  if [[ "$saw_station_arg" != true ]]; then
    # Default behavior processes all stations.
    return 0
  fi
  return 1
}


load_resource_limits

echo '------------------------------------------------------'
echo '------------------------------------------------------'

traffic_light_queue_init

iteration=1
if [[ "$run_anyway" != true ]]; then
  echo "Ensuring no other guide_raw_to_corrected.sh is running with identical arguments..."
  if ! acquire_argument_lock; then
    exit 0
  fi
else
  echo "--run-anyway set; skipping same-argument locking."
fi


while true; do
  echo "Pipeline iteration $iteration started at: $(date '+%Y-%m-%d %H:%M:%S')"
  build_iteration_pairs_from_queue iteration_pairs
  echo "Traffic-light queue snapshot: ${iteration_pairs[*]}"
  for pair in "${iteration_pairs[@]}"; do
    IFS='-' read -r station task_id <<<"$pair"
    task_index=$((task_id - 1))
    task_script="${TASK_SCRIPTS[$task_index]}"
    task_label="${TASK_LABELS[$task_index]}"
    echo "Considering station $station task${task_id} (${task_label}) ..."
    rotate_pair_to_queue_end "$pair"
    if pipeline_already_running "$station"; then
      echo "Station $station already handled by another guide_raw_to_corrected.sh; exiting to avoid waiting indefinitely."
      echo '------------------------------------------------------'
      exit 0
    fi
    if [[ ! -x "$task_script" ]]; then
      echo "Warning: task script $task_script not found or not executable. Skipping."
      continue
    fi
    if ! wait_for_resources; then
      echo "Skipping station $station task${task_id} (${task_label}) this loop due to high load."
      echo '------------------------------------------------------'
      continue
    fi
    echo "Running station $station task${task_id} (${task_label}) ..."

    if ! python3 -u "$task_script" "$station"; then
      echo "Task $(basename "$task_script") failed for station $station; continuing to next pair."
      echo '------------------------------------------------------'
      continue
    fi
    echo "Completed station $station task${task_id} (${task_label})."
    echo '------------------------------------------------------'
  done
  echo "Pipeline iteration $iteration completed at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo '------------------------------------------------------'
  iteration=$((iteration + 1))
  echo "Starting next iteration..."
done
