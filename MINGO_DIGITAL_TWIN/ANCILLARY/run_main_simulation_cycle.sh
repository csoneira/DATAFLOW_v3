#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/DATAFLOW_v3"
DT_DIR="${ROOT_DIR}/MINGO_DIGITAL_TWIN"

STEP0_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_0/step_0_setup_to_blank.py"
RUN_STEP_SCRIPT="${DT_DIR}/run_step.sh"
STEP_FINAL_SCRIPT="${DT_DIR}/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py"

log_ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
  printf '%s [SIM_CYCLE] [INFO] %s\n' "$(log_ts)" "$*"
}

log_warn() {
  printf '%s [SIM_CYCLE] [WARN] %s\n' "$(log_ts)" "$*"
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

  log_info "cycle status=start"

  if ! run_stage "step_0" /usr/bin/env python3 "${STEP0_SCRIPT}"; then
    failed=1
  fi

  if ! run_stage "run_step_all" /usr/bin/env \
    RUN_STEP_STRICT_LINE_CLOSURE="${RUN_STEP_STRICT_LINE_CLOSURE:-1}" \
    RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES="${RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES:-1}" \
    /bin/bash "${RUN_STEP_SCRIPT}" all --no-plots; then
    failed=1
  fi

  if ! run_stage "step_final" /usr/bin/env python3 "${STEP_FINAL_SCRIPT}"; then
    failed=1
  fi

  if [[ "${failed}" -eq 0 ]]; then
    log_info "cycle status=ok"
    return 0
  fi

  log_warn "cycle status=failed"
  return 1
}

main "$@"
