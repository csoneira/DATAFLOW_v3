#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/DATAFLOW_v3"
DT_DIR="${ROOT_DIR}/MINGO_DIGITAL_TWIN"

# locks used by the pipeline.  the main cycle holds SIM_MAIN_PIPELINE_LOCK
# when running steps 0–10; the final formatter uses a separate lock so that
# long final-stage invocations don't block the next cycle from starting.  the
# independent cron job also uses the final lock.
SIM_MAIN_PIPELINE_LOCK="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_main_pipeline.lock"
SIM_FINAL_LOCK="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_final.lock"

STEP0_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_0/step_0_setup_to_blank.py"
RUN_STEP_SCRIPT="${DT_DIR}/run_step.sh"
STEP_FINAL_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py"
REPAIR_MESH_IDS_SCRIPT="${DT_DIR}/ANCILLARY/repair_param_mesh_step_ids.py"
PRUNE_MESH_SCRIPT="${DT_DIR}/ANCILLARY/prune_completed_param_mesh_rows.py"
SANITIZE_SCRIPT="${DT_DIR}/ANCILLARY/sanitize_sim_runs.py"
FREQUENCY_CONFIG_FILE_DEFAULT="${DT_DIR}/CONFIG_FILES/sim_main_pipeline_frequency.conf"
DEFAULT_MIN_INTERVAL_SECONDS=180
DEFAULT_LAST_RUN_STATE_FILE="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_main_pipeline.last_run_epoch"

RESOLVED_MIN_INTERVAL_SECONDS="${DEFAULT_MIN_INTERVAL_SECONDS}"
RESOLVED_LAST_RUN_FILE="${DEFAULT_LAST_RUN_STATE_FILE}"

log_ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
  printf '%s [SIM_CYCLE] [INFO] %s\n' "$(log_ts)" "$*"
}

log_warn() {
  printf '%s [SIM_CYCLE] [WARN] %s\n' "$(log_ts)" "$*"
}

is_nonneg_int() {
  local value="${1:-}"
  [[ "${value}" =~ ^[0-9]+$ ]]
}

load_frequency_gate_settings() {
  local config_file="${SIM_MAIN_CYCLE_FREQUENCY_CONFIG_FILE:-${FREQUENCY_CONFIG_FILE_DEFAULT}}"
  local raw_interval
  local env_interval_set=0
  local env_interval_value=""
  local env_last_run_set=0
  local env_last_run_value=""
  local env_ignore_set=0
  local env_ignore_value=""

  if [[ "${SIM_MAIN_CYCLE_MIN_INTERVAL_SECONDS+x}" == "x" ]]; then
    env_interval_set=1
    env_interval_value="${SIM_MAIN_CYCLE_MIN_INTERVAL_SECONDS}"
  fi
  if [[ "${SIM_MAIN_CYCLE_LAST_RUN_FILE+x}" == "x" ]]; then
    env_last_run_set=1
    env_last_run_value="${SIM_MAIN_CYCLE_LAST_RUN_FILE}"
  fi
  if [[ "${SIM_MAIN_CYCLE_IGNORE_FREQUENCY_GATE+x}" == "x" ]]; then
    env_ignore_set=1
    env_ignore_value="${SIM_MAIN_CYCLE_IGNORE_FREQUENCY_GATE}"
  fi

  if [[ -f "${config_file}" ]]; then
    # shellcheck disable=SC1090
    source "${config_file}"
  fi

  if (( env_interval_set )); then
    SIM_MAIN_CYCLE_MIN_INTERVAL_SECONDS="${env_interval_value}"
  fi
  if (( env_last_run_set )); then
    SIM_MAIN_CYCLE_LAST_RUN_FILE="${env_last_run_value}"
  fi
  if (( env_ignore_set )); then
    SIM_MAIN_CYCLE_IGNORE_FREQUENCY_GATE="${env_ignore_value}"
  fi

  raw_interval="${SIM_MAIN_CYCLE_MIN_INTERVAL_SECONDS:-${DEFAULT_MIN_INTERVAL_SECONDS}}"
  if ! is_nonneg_int "${raw_interval}"; then
    log_warn "frequency_gate status=invalid_interval value='${raw_interval}' fallback=${DEFAULT_MIN_INTERVAL_SECONDS}"
    raw_interval="${DEFAULT_MIN_INTERVAL_SECONDS}"
  fi

  RESOLVED_MIN_INTERVAL_SECONDS="${raw_interval}"
  RESOLVED_LAST_RUN_FILE="${SIM_MAIN_CYCLE_LAST_RUN_FILE:-${DEFAULT_LAST_RUN_STATE_FILE}}"

  log_info "frequency_gate status=config min_interval_s=${RESOLVED_MIN_INTERVAL_SECONDS} state_file=${RESOLVED_LAST_RUN_FILE}"
}

frequency_gate_allows_run() {
  local now_epoch
  local last_epoch
  local elapsed
  local remaining

  if [[ "${SIM_MAIN_CYCLE_IGNORE_FREQUENCY_GATE:-0}" == "1" ]]; then
    log_info "frequency_gate status=bypassed"
    return 0
  fi

  if (( RESOLVED_MIN_INTERVAL_SECONDS <= 0 )); then
    log_info "frequency_gate status=disabled min_interval_s=${RESOLVED_MIN_INTERVAL_SECONDS}"
    return 0
  fi

  now_epoch="$(date +%s)"
  if [[ -f "${RESOLVED_LAST_RUN_FILE}" ]]; then
    last_epoch="$(tr -d '[:space:]' < "${RESOLVED_LAST_RUN_FILE}" || true)"
    if is_nonneg_int "${last_epoch}"; then
      elapsed=$(( now_epoch - last_epoch ))
      if (( elapsed < 0 )); then
        log_warn "frequency_gate status=clock_skew now_epoch=${now_epoch} last_epoch=${last_epoch}"
      elif (( elapsed < RESOLVED_MIN_INTERVAL_SECONDS )); then
        remaining=$(( RESOLVED_MIN_INTERVAL_SECONDS - elapsed ))
        log_info "cycle status=skipped reason=frequency_gate min_interval_s=${RESOLVED_MIN_INTERVAL_SECONDS} elapsed_s=${elapsed} remaining_s=${remaining}"
        return 1
      fi
    else
      log_warn "frequency_gate status=invalid_state value='${last_epoch}' file=${RESOLVED_LAST_RUN_FILE}"
    fi
  fi

  mkdir -p "$(dirname "${RESOLVED_LAST_RUN_FILE}")"
  if ! printf '%s\n' "${now_epoch}" > "${RESOLVED_LAST_RUN_FILE}"; then
    log_warn "frequency_gate status=state_write_failed file=${RESOLVED_LAST_RUN_FILE}"
  fi

  return 0
}

run_stage() {
  local stage="$1"
  shift

  log_info "stage=${stage} status=start"
  if "$@"; then
    log_info "stage=${stage} status=ok"
    return 0
  fi

  local rc=$?
  log_warn "stage=${stage} status=failed rc=${rc}"
  return "${rc}"
}

# ---------------------------------------------------------------------------
# Cascade cleanup: once step N+1 has produced valid output, step N's
# intermediate SIM_RUN (which was step N+1's input) is no longer needed.
# Covers steps 3-9 where each SIM_RUN maps 1:1 to a downstream SIM_RUN
# (step IDs 4-10 are always 001).  Steps 1-2 fan out and are left to
# sanitize_sim_runs after done=1.
# ---------------------------------------------------------------------------
cleanup_consumed_intermediates() {
  local n upstream_dir downstream_dir
  local sim_run sim_name downstream_candidate manifest
  local cleaned=0 checked=0
  local intersteps="${DT_DIR}/INTERSTEPS"

  for n in $(seq 3 9); do
    upstream_dir="${intersteps}/STEP_${n}_TO_$((n + 1))"
    if (( n == 9 )); then
      downstream_dir="${intersteps}/STEP_10_TO_FINAL"
    else
      downstream_dir="${intersteps}/STEP_$((n + 1))_TO_$((n + 2))"
    fi
    [[ -d "$upstream_dir" ]] || continue
    [[ -d "$downstream_dir" ]] || continue

    for sim_run in "$upstream_dir"/SIM_RUN_*; do
      [[ -d "$sim_run" ]] || continue
      sim_name="$(basename "$sim_run")"
      checked=$((checked + 1))

      # Find a downstream SIM_RUN whose name starts with this one (+ "_").
      for downstream_candidate in "${downstream_dir}/${sim_name}_"*; do
        if [[ -d "$downstream_candidate" ]]; then
          # Verify the downstream step produced a valid output manifest.
          manifest="$(find "$downstream_candidate" -maxdepth 1 \
            -name "step_$((n + 1))_chunks.chunks.json" -type f 2>/dev/null | head -1)"
          if [[ -n "$manifest" && -s "$manifest" ]]; then
            rm -rf "$sim_run"
            cleaned=$((cleaned + 1))
            break
          fi
        fi
      done
    done
  done

  log_info "cascade_cleanup status=done checked=${checked} cleaned=${cleaned}"
}

main() {
  local failed=0

  load_frequency_gate_settings
  if ! frequency_gate_allows_run; then
    return 0
  fi

  log_info "cycle status=start"

  if ! run_stage "step_0" /usr/bin/env python3 "${STEP0_SCRIPT}"; then
    failed=1
  fi

  if ! run_stage "repair_mesh_step_ids" /usr/bin/env python3 "${REPAIR_MESH_IDS_SCRIPT}" --apply; then
    failed=1
  fi

  if ! run_stage "run_step_all" /usr/bin/env \
    RUN_STEP_STRICT_LINE_CLOSURE="${RUN_STEP_STRICT_LINE_CLOSURE:-1}" \
    RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES="${RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES:-1}" \
    RUN_STEP_CHECK_PARAM_MESH_CONSISTENCY="${RUN_STEP_CHECK_PARAM_MESH_CONSISTENCY:-0}" \
    /bin/bash "${RUN_STEP_SCRIPT}" all --no-plots; then
    failed=1
  fi

  # run final stage under its own lock so that a concurrently-scheduled
  # standalone job doesn't interfere with the main-cycle lock.  if the
  # external cron entry is already formatting, skip rather than wait – the
  # exporter will run again on the next minute.
  if ! flock -n "${SIM_FINAL_LOCK}" /usr/bin/env python3 "${STEP_FINAL_SCRIPT}"; then
    log_warn "step_final skipped because ${SIM_FINAL_LOCK} was busy"
  else
    if [[ $? -ne 0 ]]; then
      failed=1
    fi
  fi

  # Housekeeping only: keep param_mesh focused on pending work while serialized
  # inside the same simulation cycle lock. Do not fail the cycle on prune issues.
  if ! run_stage "prune_mesh_done_rows" /usr/bin/env python3 "${PRUNE_MESH_SCRIPT}"; then
    log_warn "stage=prune_mesh_done_rows status=non_fatal_failure"
  fi

  # Cascade cleanup: delete intermediate SIM_RUN data that has been consumed
  # by the next step.  This prevents storage exhaustion from accumulated
  # intermediates.  Non-fatal: do not fail the cycle on cleanup issues.
  cleanup_consumed_intermediates || log_warn "stage=cascade_cleanup status=non_fatal_failure"

  # Sanitize: delete SIM_RUNs for completed mesh rows and broken runs.
  # Integrated here so it runs every cycle instead of a separate cron job.
  if [[ -f "${SANITIZE_SCRIPT}" ]]; then
    if ! run_stage "sanitize" /usr/bin/env python3 "${SANITIZE_SCRIPT}" --apply --min-age-seconds 900; then
      log_warn "stage=sanitize status=non_fatal_failure"
    fi
  fi

  if [[ "${failed}" -eq 0 ]]; then
    log_info "cycle status=ok"
    return 0
  fi

  log_warn "cycle status=failed"
  return 1
}

main "$@"
