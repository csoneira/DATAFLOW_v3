#!/usr/bin/env bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: MASTER/ANCILLARY/PLOTTERS/create_metadata_plots.sh
# Purpose: Create metadata plots.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash MASTER/ANCILLARY/PLOTTERS/create_metadata_plots.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

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
  "trigger_rates_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/TRIGGER_RATES/trigger_rate_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/TRIGGER_RATES/trigger_rate_metadata_config.json"

run_plotter \
  "noise_control_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/NOISE_CONTROL/noise_control_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/NOISE_CONTROL/noise_control_metadata_config.json"

run_plotter \
  "rates_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/RATES/rates_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/RATES/rates_metadata_config.json"

run_plotter \
  "execution_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/EXECUTION/execution_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/EXECUTION/execution_metadata_config.json"

run_plotter \
  "efficiency_metadata_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/EFFICIENCIES/efficiency_metadata_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/EFFICIENCIES/efficiency_metadata_config.json"

run_plotter \
  "efficiencies_three_to_four_plotter" \
  python3 -u "$PLOTTERS_ROOT/METADATA/EFFICIENCIES_THREE_TO_FOUR/efficiencies_three_to_four_plotter.py" \
  --config "$PLOTTERS_ROOT/METADATA/EFFICIENCIES_THREE_TO_FOUR/efficiencies_three_to_four_config.json"

run_plotter \
  "simulated_data_evolution_plotter" \
  python3 -u "$PLOTTERS_ROOT/SIMULATED_DATA_EVOLUTION/simulated_data_evolution_plotter.py" \
  --config "$PLOTTERS_ROOT/SIMULATED_DATA_EVOLUTION/simulated_data_evolution_config.json"

if [[ $failures -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH with ${failures} failure(s)"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH all plotters OK"
