#!/bin/bash

# Number of times to run the script
NUM_RUNS=400

# Path to the Python script
#SCRIPT_PATH="/home/cayetano/DATAFLOW_v3/MASTER/FIRST_STAGE/EVENT_DATA/Backbone/event_accumulator.py"

# Loop 400 times
for i in $(seq 1 $NUM_RUNS); do
    echo "Running iteration $i..."
    python3 /home/cayetano/DATAFLOW_v3/MASTER/FIRST_STAGE/EVENT_DATA/Backbone/event_accumulator.py 1
done

echo "All runs completed."
