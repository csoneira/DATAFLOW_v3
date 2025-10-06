#!/bin/bash

# log_file="${LOG_FILE:-~/cron_logs/bring_and_analyze_events_${station}.log}"
# mkdir -p "$(dirname "$log_file")"

# Station specific -----------------------------
if [ -z "$1" ]; then
  echo "Error: No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

# echo '------------------------------------------------------'
# echo "bring_and_analyze_events.sh started on: $(date '+%Y-%m-%d %H:%M:%S')"

station=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
    MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""
STATUS_CSV=""

# If $1 is not 1, 2, 3, 4, exit
if [[ ! "$station" =~ ^[1-4]$ ]]; then
  echo "Error: Invalid station number. Please provide a number between 1 and 4."
  exit 1
fi

# echo "Station: $station"
# ----------------------------------------------


# --------------------------------------------------------------------------------------------
# Prevent the script from running multiple instances -----------------------------------------
# --------------------------------------------------------------------------------------------

# Variables
script_name=$(basename "$0")
script_args="$*"
current_pid=$$

# Debug: Check for running processes
# echo "$(date) - Checking for existing processes of $script_name with args $script_args"
# ps -eo pid,cmd | grep "[b]ash .*/$script_name"

# Get all running instances of the script *with the same argument*, but exclude the current process
# for pid in $(ps -eo pid,cmd | grep "[b]ash .*/$script_name" | awk '{print $1}'); do
for pid in $(ps -eo pid,cmd | grep "[b]ash .*/$script_name" | grep -v "bin/bash -c" | awk '{print $1}'); do
    if [[ "$pid" != "$current_pid" ]]; then
        cmdline=$(ps -p "$pid" -o args=)
        # echo "$(date) - Found running process: PID $pid - $cmdline"
        if [[ "$cmdline" == *"$script_name $script_args"* ]]; then
            echo "------------------------------------------------------"
            echo "$(date): The script $script_name with arguments '$script_args' is already running (PID: $pid). Exiting."
            echo "------------------------------------------------------"
            exit 1
        fi
    fi
done

# If no duplicate process is found, continue
echo "$(date) - No running instance found. Proceeding..."

# Variables
# script_name=$(basename "$0")
# script_args="$*"
# current_pid=$$

# # Get all running instances of the script (excluding itself)
# # for pid in $(pgrep -f "bash .*/$script_name $script_args"); do
# for pid in $(pgrep -f "bash .*/$script_name $script_args" | grep -v $$); do
#     if [ "$pid" != "$current_pid" ]; then
#         cmdline=$(ps -p "$pid" -o args=)
#         if [[ "$cmdline" == *"$script_name"* && "$cmdline" == *"$script_args"* ]]; then
#             echo "------------------------------------------------------"
#             echo "$(date): The script $script_name with arguments '$script_args' is already running (PID: $pid). Exiting."
#             echo "------------------------------------------------------"
#             exit 1
#         fi
#     fi
# done

# If no duplicate process is found, continue
echo "------------------------------------------------------"
echo "raw_to_list_events.sh started on: $(date)"
echo "Station: $script_args"
echo "Running the script..."
echo "------------------------------------------------------"
# --------------------------------------------------------------------------------------------


# dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# # Define base working directory
# station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station"
base_working_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA"

mkdir -p "$base_working_directory"
STATUS_CSV="$base_working_directory/raw_to_list_events_status.csv"
if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
    echo "Warning: unable to record status in $STATUS_CSV" >&2
    STATUS_TIMESTAMP=""
fi

finish() {
    local exit_code="$1"
    if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
        python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
    fi
}

trap 'finish $?' EXIT

# # Define directories
# local_destination="$base_working_directory/RAW"
# storage_directory="$base_working_directory/RAW_TO_LIST"

# # Additional paths
# mingo_direction="mingo0$station"

raw_to_list_directory="$HOME/DATAFLOW_v3/MASTER/FIRST_STAGE/EVENT_DATA/Backbone/raw_to_list.py"
# event_accumulator_directory="$HOME/DATAFLOW_v3/MASTER/FIRST_STAGE/EVENT_DATA/Backbone/event_accumulator.py"

exclude_list_file="$base_working_directory/tmp/exclude_list.txt"

# # Create necessary directories
# mkdir -p "$station_directory"
# mkdir -p "$base_working_directory/tmp"
# mkdir -p "$local_destination"
# mkdir -p "$storage_directory"

echo '------------------------------------------------------'
echo '------------------------------------------------------'

# Process the data: raw_to_list.py
echo "Processing .dat files with Python script (raw_to_list.py)..."
python3 -u "$raw_to_list_directory" "$station"

echo '------------------------------------------------------'
echo "raw_to_list.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
