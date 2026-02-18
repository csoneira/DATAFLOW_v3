#!/usr/bin/env bash

set -u

DATAFLOW_ROOT="${DATAFLOW_ROOT:-$HOME/DATAFLOW_v3}"
PLOTTERS_ROOT="$DATAFLOW_ROOT/MASTER/ANCILLARY/PLOTTERS"

failures=0

run_plotter() {
  local label="$1"
  shift

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${label}"
  "$@"
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  ${label} (exit ${rc})"
    failures=$((failures + 1))
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${label}"
  fi
}

run_plotter \
  "definitive_execution_plotter" \
  python3 -u "$PLOTTERS_ROOT/DEFINITIVE_EXECUTION/definitive_execution_plotter.py"

run_plotter \
  "filter_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/FILTER/filter_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/FILTER/filter_metadata_config.json"

run_plotter \
  "rates_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/RATES/rates_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/RATES/rates_metadata_config.json"

run_plotter \
  "execution_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/EXECUTION/execution_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/EXECUTION/execution_metadata_config.json"

if [[ $failures -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH with ${failures} failure(s)"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH all plotters OK"
