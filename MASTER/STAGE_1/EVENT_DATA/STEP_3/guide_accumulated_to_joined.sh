#!/bin/bash

if [[ $# -ne 1 || "$1" =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
guide_accumulated_to_joined.sh
Guides accumulated event data files to be joined into final output files.
Usage:
  guide_accumulated_to_joined.sh <station>
Options:
  -h, --help    Show this help message and exit.
Pass the station number (1-4). The script runs the distributor and joiner for that station.
EOF
  exit 0
fi

station=$1
if [[ ! "$station" =~ ^[1-4]$ ]]; then
  echo "Error: Invalid station number. Please provide a number between 1 and 4."
  exit 1
fi

script_name=$(basename "$0")
lockfile="/tmp/${script_name}_${station}.lock"

# Acquire exclusive lock so we do not run overlapping instances for the same station.
exec 200>"$lockfile"
if ! flock -n 200; then
  echo "------------------------------------------------------"
  echo "$(date): $script_name is already running for station $station (lock held in $lockfile). Exiting."
  echo "------------------------------------------------------"
  exit 1
fi
trap 'rm -f "$lockfile"' EXIT
echo "$(date) - Lock acquired ($lockfile). Proceeding with station $station."

python3 $HOME/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_3/TASK_1/accumulated_distributor.py $station
python3 $HOME/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_3/TASK_2/distributed_joiner.py $station
