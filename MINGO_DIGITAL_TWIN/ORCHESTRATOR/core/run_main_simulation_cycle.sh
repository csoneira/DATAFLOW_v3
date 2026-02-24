#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Simulation orchestrator (cron-driven, two-phase).
#
# Phase 1 (enqueue, fast):
#   - Optional frequency gate
#   - Backpressure gate
#   - STEP_0 with --force (guarantees new mesh rows when under threshold)
#
# Phase 2 (processing, potentially long):
#   - run_step.sh all --no-plots
#   - STEP_FINAL (single attempt under sim_final.lock)
#   - housekeeping (non-fatal)
#
# Locks are internal and per-phase so long processing cannot starve enqueue.
# -----------------------------------------------------------------------------

ROOT_DIR="$HOME/DATAFLOW_v3"
DT_DIR="${ROOT_DIR}/MINGO_DIGITAL_TWIN"
ROOT_RUNTIME_DIR="${ROOT_DIR}/OPERATIONS_RUNTIME"

SIM_STRUCTURED_LOGS_ENABLED="${SIM_STRUCTURED_LOGS_ENABLED:-1}"
if [[ "${SIM_STRUCTURED_LOGS_ENABLED}" != "0" && "${SIM_STRUCTURED_LOGS_ENABLED}" != "1" ]]; then
  SIM_STRUCTURED_LOGS_ENABLED="1"
fi
SIM_CYCLE_STRUCTURED_LOG_PATH="${ROOT_RUNTIME_DIR}/CRON_LOGS/SIMULATION/STRUCTURED/sim_cycle.jsonl"
SIM_LOG_HELPER="${DT_DIR}/ORCHESTRATOR/helpers/sim_structured_logging.sh"
if [[ -f "${SIM_LOG_HELPER}" ]]; then
  # shellcheck disable=SC1090
  source "${SIM_LOG_HELPER}"
fi

SIM_ENQUEUE_LOCK="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_enqueue.lock"
SIM_PROCESSING_LOCK="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_processing.lock"
SIM_FINAL_LOCK="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_final.lock"

STEP0_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_0/step_0_setup_to_blank.py"
RUN_STEP_SCRIPT="${DT_DIR}/ORCHESTRATOR/core/run_step.sh"
STEP_FINAL_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py"
STEP_FINAL_CONFIG="${DT_DIR}/MASTER_STEPS/STEP_FINAL/config_step_final_physics.yaml"
REPAIR_MESH_IDS_SCRIPT="${DT_DIR}/ORCHESTRATOR/maintenance/repair_param_mesh_step_ids.py"
PRUNE_MESH_SCRIPT="${DT_DIR}/ORCHESTRATOR/maintenance/prune_completed_param_mesh_rows.py"
PRUNE_FINAL_SCRIPT="${DT_DIR}/ORCHESTRATOR/maintenance/prune_step_final_params.py"
SANITIZE_SCRIPT="${DT_DIR}/ORCHESTRATOR/maintenance/sanitize_sim_runs.py"
CASCADE_CLEANUP_HELPER="${DT_DIR}/ORCHESTRATOR/helpers/cascade_cleanup_intersteps.py"
FREQUENCY_CONFIG_FILE_DEFAULT="${DT_DIR}/CONFIG_FILES/sim_main_pipeline_frequency.conf"

PARAM_MESH_PATH="${DT_DIR}/INTERSTEPS/STEP_0_TO_1/param_mesh.csv"
SIMULATED_DATA_DIR="${DT_DIR}/SIMULATED_DATA"
SIMULATED_DATA_FILES_DIR="${SIMULATED_DATA_DIR}/FILES"
STATIONS_STEP1_DIR="${ROOT_DIR}/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1"

# Disabled by default: cron already fires every minute.
DEFAULT_MIN_INTERVAL_SECONDS=0
DEFAULT_LAST_RUN_STATE_FILE="${ROOT_DIR}/OPERATIONS_RUNTIME/LOCKS/cron/sim_main_pipeline.last_run_epoch"

RESOLVED_MIN_INTERVAL_SECONDS="${DEFAULT_MIN_INTERVAL_SECONDS}"
RESOLVED_LAST_RUN_FILE="${DEFAULT_LAST_RUN_STATE_FILE}"

emit_structured_log() {
  local level="$1"
  shift
  local message="$*"
  if ! declare -F sim_structured_log_emit >/dev/null 2>&1; then
    return 0
  fi
  sim_structured_log_emit "${SIM_CYCLE_STRUCTURED_LOG_PATH}" "sim_cycle" "${level}" "${message}" || true
}

log_ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
  printf '%s [SIM_CYCLE] [INFO] %s\n' "$(log_ts)" "$*"
  emit_structured_log "INFO" "$*"
}

log_warn() {
  printf '%s [SIM_CYCLE] [WARN] %s\n' "$(log_ts)" "$*"
  emit_structured_log "WARN" "$*"
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
  local env_backpressure_set=0
  local env_backpressure_value=""

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
  if [[ "${SIM_MAX_UNPROCESSED_FILES+x}" == "x" ]]; then
    env_backpressure_set=1
    env_backpressure_value="${SIM_MAX_UNPROCESSED_FILES}"
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
  if (( env_backpressure_set )); then
    SIM_MAX_UNPROCESSED_FILES="${env_backpressure_value}"
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
        log_info "enqueue status=skipped reason=frequency_gate min_interval_s=${RESOLVED_MIN_INTERVAL_SECONDS} elapsed_s=${elapsed} remaining_s=${remaining}"
        return 1
      fi
    else
      log_warn "frequency_gate status=invalid_state value='${last_epoch}' file=${RESOLVED_LAST_RUN_FILE}"
    fi
  fi

  return 0
}

frequency_gate_mark_run() {
  local run_epoch="${1:-$(date +%s)}"
  if (( RESOLVED_MIN_INTERVAL_SECONDS <= 0 )); then
    return 0
  fi
  mkdir -p "$(dirname "${RESOLVED_LAST_RUN_FILE}")"
  if ! printf '%s\n' "${run_epoch}" > "${RESOLVED_LAST_RUN_FILE}"; then
    log_warn "frequency_gate status=state_write_failed file=${RESOLVED_LAST_RUN_FILE}"
    return 1
  fi
  log_info "frequency_gate status=state_updated epoch=${run_epoch}"
  return 0
}

count_pending_files() {
  local n_sim_root=0 n_sim_files=0 n_unprocessed=0 n_processing=0 total=0

  if [[ -d "${SIMULATED_DATA_DIR}" ]]; then
    n_sim_root=$(find "${SIMULATED_DATA_DIR}" -maxdepth 1 -name "mi*.dat" -type f 2>/dev/null | wc -l || true)
  fi
  if [[ -d "${SIMULATED_DATA_FILES_DIR}" ]]; then
    n_sim_files=$(find "${SIMULATED_DATA_FILES_DIR}" -maxdepth 1 -name "mi*.dat" -type f 2>/dev/null | wc -l || true)
  fi
  if [[ -d "${STATIONS_STEP1_DIR}" ]]; then
    n_unprocessed=$(find "${STATIONS_STEP1_DIR}" -path "*/INPUT_FILES/UNPROCESSED_DIRECTORY/*" -type f 2>/dev/null | wc -l || true)
    n_processing=$(find "${STATIONS_STEP1_DIR}" -path "*/INPUT_FILES/PROCESSING_DIRECTORY/*" -type f 2>/dev/null | wc -l || true)
  fi

  total=$(( n_sim_root + n_sim_files + n_unprocessed + n_processing ))
  printf '%s %s %s %s %s\n' "${n_sim_root}" "${n_sim_files}" "${n_unprocessed}" "${n_processing}" "${total}"
}

backpressure_gate_allows_step0() {
  local threshold="${SIM_MAX_UNPROCESSED_FILES:-0}"
  local n_sim_root=0 n_sim_files=0 n_unprocessed=0 n_processing=0 total=0

  if ! is_nonneg_int "${threshold}"; then
    log_warn "backpressure_gate status=invalid_threshold value='${threshold}' fallback=0"
    threshold=0
  fi

  # shellcheck disable=SC2086
  read -r n_sim_root n_sim_files n_unprocessed n_processing total <<< "$(count_pending_files)"

  if (( threshold <= 0 )); then
    log_info "backpressure_gate status=disabled pending_total=${total} simulated_root=${n_sim_root} simulated_files=${n_sim_files} unprocessed=${n_unprocessed} processing=${n_processing}"
    return 0
  fi

  if (( total >= threshold )); then
    log_info "backpressure_gate status=blocked pending_total=${total} threshold=${threshold} simulated_root=${n_sim_root} simulated_files=${n_sim_files} unprocessed=${n_unprocessed} processing=${n_processing}"
    return 1
  fi

  log_info "backpressure_gate status=ok pending_total=${total} threshold=${threshold} simulated_root=${n_sim_root} simulated_files=${n_sim_files} unprocessed=${n_unprocessed} processing=${n_processing}"
  return 0
}

count_mesh_rows() {
  local mesh_path="${1:-${PARAM_MESH_PATH}}"
  if [[ ! -f "${mesh_path}" ]]; then
    echo "0"
    return 0
  fi
  awk 'NR>1 {count++} END {print count+0}' "${mesh_path}" 2>/dev/null || echo "0"
}

run_stage() {
  local stage="$1"
  shift
  local rc=0

  log_info "stage=${stage} status=start"
  if "$@"; then
    log_info "stage=${stage} status=ok"
    return 0
  fi

  rc=$?
  log_warn "stage=${stage} status=failed rc=${rc}"
  return "${rc}"
}

# Cascade cleanup: once step N+1 has produced valid output, step N's
# intermediate SIM_RUN (which was step N+1's input) is no longer needed.
cleanup_consumed_intermediates() {
  local output
  local rc

  if [[ ! -f "${CASCADE_CLEANUP_HELPER}" ]]; then
    log_warn "cascade_cleanup status=skipped reason=missing_helper helper=${CASCADE_CLEANUP_HELPER}"
    return 1
  fi

  if output="$(/usr/bin/env python3 "${CASCADE_CLEANUP_HELPER}" --intersteps "${DT_DIR}/INTERSTEPS" --mesh "${PARAM_MESH_PATH}" 2>&1)"; then
    log_info "cascade_cleanup status=done ${output}"
    return 0
  fi

  rc=$?
  log_warn "cascade_cleanup status=failed rc=${rc} output=${output}"
  return 1
}

run_enqueue_phase() {
  local mesh_before=0 mesh_after=0 rows_added=0
  local enqueue_failed=0

  mkdir -p "$(dirname "${SIM_ENQUEUE_LOCK}")"
  exec {enqueue_fd}> "${SIM_ENQUEUE_LOCK}"
  if ! flock -n "${enqueue_fd}"; then
    log_info "enqueue status=skipped reason=lock_busy lock=${SIM_ENQUEUE_LOCK}"
    exec {enqueue_fd}>&-
    return 0
  fi

  if ! frequency_gate_allows_run; then
    exec {enqueue_fd}>&-
    return 0
  fi

  if backpressure_gate_allows_step0; then
    mesh_before="$(count_mesh_rows "${PARAM_MESH_PATH}")"
    log_info "stage=step_0 status=start mode=force"
    if /usr/bin/env python3 "${STEP0_SCRIPT}" --force; then
      mesh_after="$(count_mesh_rows "${PARAM_MESH_PATH}")"
      rows_added=$(( mesh_after - mesh_before ))
      if (( rows_added <= 0 )); then
        log_warn "stage=step_0 status=failed reason=no_rows_added rows_before=${mesh_before} rows_after=${mesh_after}"
        enqueue_failed=1
      else
        log_info "stage=step_0 status=ok rows_before=${mesh_before} rows_after=${mesh_after} rows_added=${rows_added}"
      fi
    else
      log_warn "stage=step_0 status=failed"
      enqueue_failed=1
    fi
  else
    log_info "stage=step_0 status=skipped reason=backpressure"
  fi

  frequency_gate_mark_run "$(date +%s)" || true
  exec {enqueue_fd}>&-

  if (( enqueue_failed == 0 )); then
    return 0
  fi
  return 1
}

run_processing_phase() {
  local failed=0
  local requested_include_final="${RUN_STEP_INCLUDE_FINAL_IN_ALL:-0}"
  local run_step_autoclean_conflicts="${SIM_RUN_STEP_AUTOCLEAN_CONFLICTS:-1}"
  local run_step_autoclean_min_age_s="${SIM_RUN_STEP_AUTOCLEAN_MIN_AGE_S:-300}"
  local post_sanitize_enabled="${SIM_POST_SANITIZE_ENABLED:-1}"
  local post_sanitize_min_age_s="${SIM_POST_SANITIZE_MIN_AGE_S:-900}"
  local processing_runs_step_final="${SIM_PROCESSING_RUN_STEP_FINAL:-1}"
  local run_step_stuck_alerts_enabled="${RUN_STEP_STUCK_ALERTS_ENABLED:-1}"
  local run_step_stuck_alert_min_cycles="${RUN_STEP_STUCK_ALERT_MIN_CYCLES:-3}"
  local run_step_stuck_alert_repeat_cycles="${RUN_STEP_STUCK_ALERT_REPEAT_CYCLES:-60}"
  local run_step_stuck_alert_min_observation_interval_s="${RUN_STEP_STUCK_ALERT_MIN_OBSERVATION_INTERVAL_S:-45}"

  if [[ "${run_step_autoclean_conflicts}" != "0" && "${run_step_autoclean_conflicts}" != "1" ]]; then
    run_step_autoclean_conflicts="1"
  fi
  if ! is_nonneg_int "${run_step_autoclean_min_age_s}"; then
    run_step_autoclean_min_age_s="300"
  fi
  if [[ "${post_sanitize_enabled}" != "0" && "${post_sanitize_enabled}" != "1" ]]; then
    post_sanitize_enabled="1"
  fi
  if ! is_nonneg_int "${post_sanitize_min_age_s}"; then
    post_sanitize_min_age_s="900"
  fi
  if [[ "${processing_runs_step_final}" != "0" && "${processing_runs_step_final}" != "1" ]]; then
    processing_runs_step_final="1"
  fi
  if [[ "${run_step_stuck_alerts_enabled}" != "0" && "${run_step_stuck_alerts_enabled}" != "1" ]]; then
    run_step_stuck_alerts_enabled="1"
  fi
  if ! is_nonneg_int "${run_step_stuck_alert_min_cycles}"; then
    run_step_stuck_alert_min_cycles="3"
  fi
  if ! is_nonneg_int "${run_step_stuck_alert_repeat_cycles}"; then
    run_step_stuck_alert_repeat_cycles="60"
  fi
  if ! is_nonneg_int "${run_step_stuck_alert_min_observation_interval_s}"; then
    run_step_stuck_alert_min_observation_interval_s="45"
  fi

  mkdir -p "$(dirname "${SIM_PROCESSING_LOCK}")"
  exec {processing_fd}> "${SIM_PROCESSING_LOCK}"
  if ! flock -n "${processing_fd}"; then
    log_info "processing status=skipped reason=lock_busy lock=${SIM_PROCESSING_LOCK}"
    exec {processing_fd}>&-
    return 0
  fi

  if [[ "${SIM_ENABLE_REPAIR_MESH_IDS:-0}" == "1" ]]; then
    if ! run_stage "repair_mesh_step_ids" /usr/bin/env python3 "${REPAIR_MESH_IDS_SCRIPT}" --apply; then
      log_warn "stage=repair_mesh_step_ids status=non_fatal_failure"
    fi
  else
    log_info "stage=repair_mesh_step_ids status=skipped reason=disabled"
  fi

  if [[ "${requested_include_final}" != "0" ]]; then
    log_warn "stage=run_step_all invariant=force_disable_step_final requested_value=${requested_include_final}"
  fi
  log_info "sanitize_policy pre_enabled=${run_step_autoclean_conflicts} pre_min_age_s=${run_step_autoclean_min_age_s} post_enabled=${post_sanitize_enabled} post_min_age_s=${post_sanitize_min_age_s}"
  log_info "alert_policy stuck_alerts_enabled=${run_step_stuck_alerts_enabled} min_cycles=${run_step_stuck_alert_min_cycles} repeat_cycles=${run_step_stuck_alert_repeat_cycles} min_observation_interval_s=${run_step_stuck_alert_min_observation_interval_s}"

  # Invariant: the processing phase runs STEP_FINAL exactly once below under
  # sim_final.lock, so run_step.sh all must never invoke STEP_FINAL itself.
  if ! run_stage "run_step_all" /usr/bin/env \
    RUN_STEP_STRICT_LINE_CLOSURE="${RUN_STEP_STRICT_LINE_CLOSURE:-0}" \
    RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES="${RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES:-0}" \
    RUN_STEP_CHECK_PARAM_MESH_CONSISTENCY="${RUN_STEP_CHECK_PARAM_MESH_CONSISTENCY:-0}" \
    RUN_STEP_AUTOCLEAN_CONFLICTS="${run_step_autoclean_conflicts}" \
    RUN_STEP_AUTOCLEAN_MIN_AGE_S="${run_step_autoclean_min_age_s}" \
    RUN_STEP_STUCK_ALERTS_ENABLED="${run_step_stuck_alerts_enabled}" \
    RUN_STEP_STUCK_ALERT_MIN_CYCLES="${run_step_stuck_alert_min_cycles}" \
    RUN_STEP_STUCK_ALERT_REPEAT_CYCLES="${run_step_stuck_alert_repeat_cycles}" \
    RUN_STEP_STUCK_ALERT_MIN_OBSERVATION_INTERVAL_S="${run_step_stuck_alert_min_observation_interval_s}" \
    RUN_STEP_INCLUDE_FINAL_IN_ALL=0 \
    /bin/bash "${RUN_STEP_SCRIPT}" all --no-plots; then
    failed=1
  fi

  if [[ "${processing_runs_step_final}" == "1" ]]; then
    log_info "stage=step_final status=start lock=${SIM_FINAL_LOCK}"
    if flock -n -E 200 "${SIM_FINAL_LOCK}" /usr/bin/env python3 "${STEP_FINAL_SCRIPT}" --config "${STEP_FINAL_CONFIG}"; then
      log_info "stage=step_final status=ok"
    else
      rc=$?
      if (( rc == 200 )); then
        log_info "stage=step_final status=skipped reason=lock_busy lock=${SIM_FINAL_LOCK}"
      else
        log_warn "stage=step_final status=failed rc=${rc}"
        failed=1
      fi
    fi
  else
    log_info "stage=step_final status=skipped reason=disabled_by_config"
  fi

  if ! run_stage "prune_mesh_done_rows" /usr/bin/env python3 "${PRUNE_MESH_SCRIPT}"; then
    log_warn "stage=prune_mesh_done_rows status=non_fatal_failure"
  fi

  if [[ -f "${PRUNE_FINAL_SCRIPT}" ]]; then
    if ! run_stage "prune_final_params" /usr/bin/env python3 "${PRUNE_FINAL_SCRIPT}"; then
      log_warn "stage=prune_final_params status=non_fatal_failure"
    fi
  fi

  cleanup_consumed_intermediates || log_warn "stage=cascade_cleanup status=non_fatal_failure"

  if [[ "${post_sanitize_enabled}" == "1" ]]; then
    if [[ -f "${SANITIZE_SCRIPT}" ]]; then
      if ! run_stage "sanitize" /usr/bin/env python3 "${SANITIZE_SCRIPT}" --apply --min-age-seconds "${post_sanitize_min_age_s}"; then
        log_warn "stage=sanitize status=non_fatal_failure"
      fi
    else
      log_info "stage=sanitize status=skipped reason=missing_script"
    fi
  else
    log_info "stage=sanitize status=skipped reason=disabled"
  fi

  exec {processing_fd}>&-
  if (( failed == 0 )); then
    return 0
  fi
  return 1
}

main() {
  local failed=0

  load_frequency_gate_settings
  log_info "cycle status=start"

  if ! run_enqueue_phase; then
    failed=1
  fi

  if ! run_processing_phase; then
    failed=1
  fi

  if (( failed == 0 )); then
    log_info "cycle status=ok"
    return 0
  fi

  log_warn "cycle status=failed"
  return 1
}

main "$@"
