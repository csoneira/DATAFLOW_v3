#!/usr/bin/env bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_step_final_cron.sh
# Purpose: Run standalone STEP_FINAL without starving a pending interstep reset.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-07-28
# Runtime: bash
# Usage: bash MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_step_final_cron.sh
# Inputs: STEP_10 intersteps, STEP_FINAL configuration, and runtime reset state.
# Outputs: STEP_FINAL products and logs from the invoked Python process.
# Notes: A pending reset owns priority over standalone STEP_FINAL.
# =============================================================================

set -euo pipefail

ROOT_DIR="${DATAFLOW_ROOT:-${HOME}/DATAFLOW_v3}"
RESET_MARKER="${SIM_INTERSTEPS_RESET_MARKER:-${ROOT_DIR}/OPERATIONS/OPERATIONS_RUNTIME/STATE/sim_intersteps_reset_needed.flag}"
FINAL_LOCK="${SIM_FINAL_LOCK:-${ROOT_DIR}/OPERATIONS/OPERATIONS_RUNTIME/LOCKS/cron/sim_final.lock}"
STEP_FINAL_SCRIPT="${SIM_STEP_FINAL_SCRIPT:-${ROOT_DIR}/MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py}"
STEP_FINAL_CONFIG="${SIM_STEP_FINAL_CONFIG:-${ROOT_DIR}/MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/config_step_final_physics.yaml}"
PYTHON_BIN="${SIM_PYTHON_BIN:-/usr/bin/python3}"

# The main controller needs final_lock together with its enqueue and processing
# locks to clear stale intersteps. If a reset is pending, do not compete for the
# final lock; the main controller will clear the marker after the reset.
if [[ -f "${RESET_MARKER}" ]]; then
  exit 0
fi

mkdir -p "$(dirname "${FINAL_LOCK}")"
exec /usr/bin/flock -n "${FINAL_LOCK}" \
  /usr/bin/env "${PYTHON_BIN}" "${STEP_FINAL_SCRIPT}" --config "${STEP_FINAL_CONFIG}"
