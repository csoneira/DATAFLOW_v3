#!/bin/bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: FOR_MINGO_SYSTEMS/station_automation_scripts/logs_scripts/cron_logs_join.sh
# Purpose: Cron logs join.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash FOR_MINGO_SYSTEMS/station_automation_scripts/logs_scripts/cron_logs_join.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

cd /home/rpcuser/logs/'done'

pwd

rm merged_Flow0.txt
rm merged_hv0.txt
rm merged_sensors_bus0.txt
rm merged_sensors_bus1.txt
rm merged_rates.txt

echo 'Starting Flow merge'

cat Flow0* >> merged_Flow0.txt

echo 'Flow done'

echo 'Starting HV merge'

cat hv0* >> merged_hv0.txt
sed -i 's/T/ /' merged_hv0.txt

echo 'HV done'

echo 'Starting environmental sensors merge'

cat sensors_bus0* >> merged_sensors_bus0.txt
cat sensors_bus1* >> merged_sensors_bus1.txt

sed -i 's/T/ /' merged_sensors_bus*
sed -i 's/;//' merged_sensors_bus*
sed -i 's/nan nan nan nan //' merged_sensors_bus*

echo 'Done'

echo 'Starting TRB rate merge'

cat rates* >> merged_rates.txt
sed -i 's/T/ /' merged_rates*
sed -i 's/;//' merged_rates*

echo 'TRB rate merge done'
