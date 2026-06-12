#!/usr/bin/env bash
set -euo pipefail

BASE="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATION_OUTPUTS/INTERSTEPS"
JOBS="${JOBS:-$(nproc)}"

echo "Target base directory:"
echo "${BASE}"
echo

if [[ ! -d "${BASE}" ]]; then
  echo "Base directory does not exist. Nothing to delete."
  exit 0
fi

echo "Deleting matching pickle chunks:"
echo "${BASE}/STEP_*_TO_*/SIM_RUN_*/step_*_chunks/part_*.pkl"
echo "Parallel jobs: ${JOBS}"
echo

find "${BASE}" \
  -type f \
  -path "${BASE}/STEP_*_TO_*/SIM_RUN_*/step_*_chunks/part_*.pkl" \
  -print0 \
| xargs -0 -r -n 1000 -P "${JOBS}" rm -f --

echo
echo "Removing entire INTERSTEPS directory:"
echo "${BASE}"

rm -rf --one-file-system -- "${BASE}"

echo "Done."