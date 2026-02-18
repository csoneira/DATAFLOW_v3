#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/DATAFLOW_v3"
DT_DIR="${ROOT_DIR}/MINGO_DIGITAL_TWIN"

STEP0_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_0/step_0_setup_to_blank.py"
RUN_STEP_SCRIPT="${DT_DIR}/run_step.sh"
STEP_FINAL_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py"
PRUNE_MESH_SCRIPT="${DT_DIR}/ANCILLARY/prune_completed_param_mesh_rows.py"
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

  if ! run_stage "run_step_all" /usr/bin/env \
    RUN_STEP_STRICT_LINE_CLOSURE="${RUN_STEP_STRICT_LINE_CLOSURE:-1}" \
    RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES="${RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES:-1}" \
    RUN_STEP_CHECK_PARAM_MESH_CONSISTENCY="${RUN_STEP_CHECK_PARAM_MESH_CONSISTENCY:-0}" \
    /bin/bash "${RUN_STEP_SCRIPT}" all --no-plots; then
    failed=1
  fi

  if ! run_stage "step_final" /usr/bin/env python3 "${STEP_FINAL_SCRIPT}"; then
    failed=1
  fi

  # Housekeeping only: keep param_mesh focused on pending work while serialized
  # inside the same simulation cycle lock. Do not fail the cycle on prune issues.
  if ! run_stage "prune_mesh_done_rows" /usr/bin/env python3 "${PRUNE_MESH_SCRIPT}"; then
    log_warn "stage=prune_mesh_done_rows status=non_fatal_failure"
  fi

  if [[ "${failed}" -eq 0 ]]; then
    log_info "cycle status=ok"
    return 0
  fi

  log_warn "cycle status=failed"
  return 1
}

main "$@"
