#!/usr/bin/env bash

# Emergency process guard for cron:
# if the pipeline accumulates too many duplicated/pending jobs, stop them.

set -u

BASE_DIR="/home/mingo/DATAFLOW_v3"
LOCK_FILE="/tmp/dataflow_stop_execution.lock"
LOG_DIR="${BASE_DIR}/EXECUTION_LOGS/CRON_LOGS/ANCILLARY/CLEANERS"
INTERNAL_LOG="${LOG_DIR}/stop_execution_internal.log"

mkdir -p "${LOG_DIR}"

{
  flock -n 9 || exit 0

  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  user_name="${USER:-$(id -un)}"
  proc_threshold="${PROC_THRESHOLD:-25}"
  dup_threshold="${DUP_THRESHOLD:-5}"
  load_factor="${LOAD_FACTOR:-2.0}"
  grace_seconds="${GRACE_SECONDS:-6}"
  dry_run="${DRY_RUN:-0}"

  cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  load_1m="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"
  load_trigger="$(awk -v c="${cpu_count}" -v f="${load_factor}" 'BEGIN { printf "%.2f", c * f }')"

  declare -a patterns=(
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/run_step.sh"
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_0/step_0_setup_to_blank.py"
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py"
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/ANCILLARY/sanitize_sim_runs.py"
    "/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/ANCILLARY/reset_param_mesh_from_final.py"
  )

  declare -A pid_to_cmd=()
  for pattern in "${patterns[@]}"; do
    while IFS= read -r line; do
      [ -z "${line}" ] && continue
      pid="${line%% *}"
      cmd="${line#* }"
      [ -z "${pid}" ] && continue
      pid_to_cmd["${pid}"]="${cmd}"
    done < <(pgrep -u "${user_name}" -f -a "${pattern}" || true)
  done

  unset "pid_to_cmd[$$]" 2>/dev/null || true
  unset "pid_to_cmd[$PPID]" 2>/dev/null || true

  declare -A dup_counts=()
  for cmd in "${pid_to_cmd[@]}"; do
    read -r a1 a2 a3 a4 _ <<< "${cmd}"
    key="${a1}"
    if [[ "${a1}" == *python* ]]; then
      if [[ "${a2:-}" == -* ]]; then
        key="${a3:-${a1}}"
      else
        key="${a2:-${a1}}"
      fi
    elif [[ "${a1}" == */bash || "${a1}" == "bash" ]]; then
      if [ "${a2:-}" = "-c" ]; then
        if [[ "${a3:-}" == */bash || "${a3:-}" == "bash" ]]; then
          key="${a4:-${a1}}"
        else
          key="${a3:-${a1}}"
        fi
      else
        key="${a2:-${a1}}"
      fi
    fi
    [[ "${key}" == -* || -z "${key}" ]] && key="${a1}"
    dup_counts["${key}"]=$(( ${dup_counts["${key}"]:-0} + 1 ))
  done

  dup_trip=0
  dup_summary=""
  for key in "${!dup_counts[@]}"; do
    count="${dup_counts[${key}]}"
    if [ "${count}" -ge "${dup_threshold}" ]; then
      dup_trip=1
      dup_summary="${dup_summary}${key}=${count}; "
    fi
  done

  match_count="${#pid_to_cmd[@]}"
  load_trip="$(awk -v l="${load_1m}" -v t="${load_trigger}" 'BEGIN {print (l >= t) ? 1 : 0}')"

  should_kill=0
  reasons=()
  if [ "${match_count}" -ge "${proc_threshold}" ]; then
    should_kill=1
    reasons+=("match_count=${match_count}>=${proc_threshold}")
  fi
  if [ "${load_trip}" -eq 1 ] && [ "${match_count}" -gt 0 ]; then
    should_kill=1
    reasons+=("load_1m=${load_1m}>=${load_trigger}")
  fi
  if [ "${dup_trip}" -eq 1 ]; then
    should_kill=1
    reasons+=("duplicates>=${dup_threshold}: ${dup_summary}")
  fi

  if [ "${should_kill}" -eq 0 ]; then
    echo "[${ts}] Healthy: matched=${match_count}, load_1m=${load_1m}, cpu=${cpu_count}"
    exit 0
  fi

  echo "[${ts}] Triggered emergency stop (${reasons[*]})"
  if [ "${match_count}" -eq 0 ]; then
    echo "[${ts}] No matching processes found after trigger; exiting."
    exit 0
  fi

  pids=( "${!pid_to_cmd[@]}" )
  IFS=$'\n' pids=( $(printf '%s\n' "${pids[@]}" | sort -n) )
  unset IFS

  echo "[${ts}] Candidate PID count: ${#pids[@]}"
  for pid in "${pids[@]}"; do
    echo "[${ts}] PID=${pid} CMD=${pid_to_cmd[${pid}]}"
  done

  if [ "${dry_run}" = "1" ]; then
    echo "[${ts}] DRY_RUN=1, no process terminated."
    exit 0
  fi

  kill -TERM "${pids[@]}" 2>/dev/null || true
  sleep "${grace_seconds}"

  still_running=()
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      still_running+=( "${pid}" )
    fi
  done

  if [ "${#still_running[@]}" -gt 0 ]; then
    echo "[${ts}] Escalating to SIGKILL for ${#still_running[@]} PIDs."
    kill -KILL "${still_running[@]}" 2>/dev/null || true
  fi

  echo "[${ts}] stop_execution.sh completed."
} 9>"${LOCK_FILE}" >> "${INTERNAL_LOG}" 2>&1
