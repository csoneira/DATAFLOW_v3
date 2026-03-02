#!/bin/bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: FOR_MINGO_SYSTEMS/station_automation_scripts/daq_data_scripts/cron_join_last_day.sh
# Purpose: Designed to merge the files of only the previous day of the current adquisition.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash FOR_MINGO_SYSTEMS/station_automation_scripts/daq_data_scripts/cron_join_last_day.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

# Designed to merge the files of only the previous day of the current adquisition.
if [[ "$1" == '-h' ]]; then
  echo "Usage:"
  echo "A function that asks for a date range and joins and compresses all data inside that range."
  exit 0
fi

# Get the current date in YYMMDD format
current_date=$(date +'%y%m%d')
# Calculate the previous day's date
previous_date=$(date -d "$current_date - 1 day" +'%y%m%d')

echo "============================================="
echo "Joining data from $previous_date"
echo "============================================="

bash ~/caye_software/daq_data_scripts/join.sh $previous_date $previous_date
