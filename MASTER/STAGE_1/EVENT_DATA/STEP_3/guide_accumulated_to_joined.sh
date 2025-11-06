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

python3 $HOME/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_3/TASK_1/accumulated_distributor.py $station
python3 $HOME/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_3/TASK_2/distributed_joiner.py $station