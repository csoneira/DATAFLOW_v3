#!/bin/bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: FOR_MINGO_SYSTEMS/station_automation_scripts/logs_scripts/cron_logs_clean.sh
# Purpose: Cron logs clean.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash FOR_MINGO_SYSTEMS/station_automation_scripts/logs_scripts/cron_logs_clean.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

cd /home/rpcuser/logs/

pwd

cat Flow0* > clean_Flow0.txt
echo "Flow file done"

cat hv0* > clean_hv0.txt
sed -i 's/T/ /' clean_hv0.txt
echo "HV file done"

cat sensors_bus0* > clean_sensors_bus0.txt
cat sensors_bus1* > clean_sensors_bus1.txt

sed -i 's/T/ /' clean_sensors_bus*
sed -i 's/;//' clean_sensors_bus*
sed -i 's/nan nan nan nan //' clean_sensors_bus*
echo "Environment files done"

cat rates* > clean_rates.txt
sed -i 's/T/ /' clean_rates*
sed -i 's/;//' clean_rates*
echo "TRB rate file done"
