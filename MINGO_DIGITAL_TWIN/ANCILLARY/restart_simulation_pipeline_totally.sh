#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Restart simulation pipeline from a clean state while archiving previous outputs.

What gets archived/moved out:
  - STATIONS/MINGO00
  - MINGO_DIGITAL_TWIN/SIMULATED_DATA
  - MINGO_DIGITAL_TWIN/INTERSTEPS

What gets recreated empty:
  - STATIONS/MINGO00
  - MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES
  - MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1

Usage:
  restart_simulation_pipeline_totally.sh [--yes] [--archive-root DIR] [--root DIR]

Options:
  --yes               Skip confirmation prompt.
  --archive-root DIR  Archive base directory (default: $HOME/SIMULATION_DATA_JUNK).
  --root DIR          DATAFLOW root (default: $HOME/DATAFLOW_v3).
  -h, --help          Show this help.
EOF
}

ROOT_DIR="${HOME}/DATAFLOW_v3"
ARCHIVE_ROOT="${HOME}/SIMULATION_DATA_JUNK"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --archive-root)
      ARCHIVE_ROOT="$2"
      shift 2
      ;;
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

DT_DIR="${ROOT_DIR}/MINGO_DIGITAL_TWIN"
MINGO00_DIR="${ROOT_DIR}/STATIONS/MINGO00"
SIMULATED_DATA_DIR="${DT_DIR}/SIMULATED_DATA"
INTERSTEPS_DIR="${DT_DIR}/INTERSTEPS"

STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
ARCHIVE_DIR="${ARCHIVE_ROOT}/restart_${STAMP}"
if [[ -e "${ARCHIVE_DIR}" ]]; then
  ARCHIVE_DIR="${ARCHIVE_DIR}_$$"
fi

for required in "${ROOT_DIR}" "${DT_DIR}"; do
  if [[ ! -d "${required}" ]]; then
    echo "[error] Required directory does not exist: ${required}" >&2
    exit 1
  fi
done

mkdir -p \
  "${ARCHIVE_DIR}/STATIONS" \
  "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN"

if [[ ${ASSUME_YES} -ne 1 ]]; then
  echo "This will archive and reset simulation outputs:"
  echo "  - ${MINGO00_DIR}"
  echo "  - ${SIMULATED_DATA_DIR}"
  echo "  - ${INTERSTEPS_DIR}"
  echo "Archive destination: ${ARCHIVE_DIR}"
  read -r -p "Continue? [y/N] " reply
  case "${reply}" in
    y|Y|yes|YES) ;;
    *)
      echo "[abort] No changes made."
      exit 0
      ;;
  esac
fi

move_if_exists() {
  local src="$1"
  local dst_dir="$2"
  if [[ -e "${src}" ]]; then
    mv "${src}" "${dst_dir}/"
    echo "[moved] ${src} -> ${dst_dir}/"
  else
    echo "[skip] ${src} does not exist"
  fi
}

move_if_exists "${MINGO00_DIR}" "${ARCHIVE_DIR}/STATIONS"
move_if_exists "${SIMULATED_DATA_DIR}" "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN"
move_if_exists "${INTERSTEPS_DIR}" "${ARCHIVE_DIR}/MINGO_DIGITAL_TWIN"

mkdir -p "${MINGO00_DIR}"
mkdir -p "${SIMULATED_DATA_DIR}/FILES"
mkdir -p "${INTERSTEPS_DIR}/STEP_0_TO_1"

{
  echo "restart_timestamp_utc=${STAMP}"
  echo "dataflow_root=${ROOT_DIR}"
  echo "archive_dir=${ARCHIVE_DIR}"
  echo "moved_station_dir=STATIONS/MINGO00"
  echo "moved_simulated_data_dir=MINGO_DIGITAL_TWIN/SIMULATED_DATA"
  echo "moved_intersteps_dir=MINGO_DIGITAL_TWIN/INTERSTEPS"
  echo "recreated_station_dir=STATIONS/MINGO00"
  echo "recreated_simulated_data_dir=MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES"
  echo "recreated_intersteps_dir=MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1"
} > "${ARCHIVE_DIR}/restart_manifest.txt"

echo "[ok] Clean restart reset completed"
echo "[ok] Archive created at: ${ARCHIVE_DIR}"
echo "[next] Update z geometry config(s), then run STEP 0 and subsequent steps."
