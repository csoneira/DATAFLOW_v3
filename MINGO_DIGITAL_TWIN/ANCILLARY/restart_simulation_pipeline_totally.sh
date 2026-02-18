#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HOME}/DATAFLOW_v3"
DT_DIR="${ROOT_DIR}/MINGO_DIGITAL_TWIN"
JUNK_ROOT="${HOME}/SIMULATION_DATA_JUNK"
STAMP="$(date '+%Y_%m_%d_%H_%M_%S')"
ARCHIVE_DIR="${JUNK_ROOT}/${STAMP}"

MINGO00_DIR="${ROOT_DIR}/STATIONS/MINGO00"
SIMULATED_DATA_DIR="${DT_DIR}/SIMULATED_DATA"
INTERSTEPS_DIR="${DT_DIR}/INTERSTEPS"

# Avoid collisions if called multiple times in the same second.
if [[ -e "${ARCHIVE_DIR}" ]]; then
  ARCHIVE_DIR="${ARCHIVE_DIR}_$$"
fi

mkdir -p \
  "${ARCHIVE_DIR}/STATIONS" \
  "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN/SIMULATED_DATA" \
  "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN"

if [[ -e "${MINGO00_DIR}" ]]; then
  mv "${MINGO00_DIR}" "${ARCHIVE_DIR}/STATIONS/"
  echo "[moved] ${MINGO00_DIR} -> ${ARCHIVE_DIR}/STATIONS/"
else
  echo "[skip] ${MINGO00_DIR} does not exist"
fi

if [[ -d "${SIMULATED_DATA_DIR}" ]]; then
  shopt -s dotglob nullglob
  simulated_items=( "${SIMULATED_DATA_DIR}"/* )
  shopt -u dotglob nullglob

  if (( ${#simulated_items[@]} > 0 )); then
    mv "${simulated_items[@]}" "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN/SIMULATED_DATA/"
    echo "[moved] ${SIMULATED_DATA_DIR}/* -> ${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN/SIMULATED_DATA/"
  else
    echo "[skip] ${SIMULATED_DATA_DIR} is already empty"
  fi
else
  echo "[skip] ${SIMULATED_DATA_DIR} does not exist"
fi

if [[ -e "${INTERSTEPS_DIR}" ]]; then
  mv "${INTERSTEPS_DIR}" "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN/"
  echo "[moved] ${INTERSTEPS_DIR} -> ${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN/"
else
  echo "[skip] ${INTERSTEPS_DIR} does not exist"
fi

echo "[ok] Pipeline simulation data reset complete"
echo "[ok] Archive created at: ${ARCHIVE_DIR}"
