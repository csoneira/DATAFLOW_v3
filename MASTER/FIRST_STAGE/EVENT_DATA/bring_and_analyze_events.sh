env > /tmp/cron_env.log
echo "$(date): Running bring_and_analyze_events.sh $script_args" > /tmp/script_debug.log

#!/bin/bash

# Station specific -----------------------------
if [ -z "$1" ]; then
  echo "Error: No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

# echo '------------------------------------------------------'
# echo "bring_and_analyze_events.sh started on: $(date '+%Y-%m-%d %H:%M:%S')"

station=$1
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
echo "bring_and_analyze_events.sh started on: $(date)"
echo "Station: $script_args"
echo "Running the script..."
echo "------------------------------------------------------"
# --------------------------------------------------------------------------------------------


dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# Define base working directory
base_working_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA"

# Define directories
local_destination="$base_working_directory/RAW"
storage_directory="$base_working_directory/RAW_TO_LIST"

# Additional paths
mingo_direction="mingo0$station"

raw_to_list_directory="$HOME/DATAFLOW_v3/MASTER/FIRST_STAGE/EVENT_DATA/Backbone/raw_to_list.py"
event_accumulator_directory="$HOME/DATAFLOW_v3/MASTER/FIRST_STAGE/EVENT_DATA/Backbone/event_accumulator.py"

exclude_list_file="$base_working_directory/tmp/exclude_list.txt"

# Create necessary directories
mkdir -p "$base_working_directory/tmp"
mkdir -p "$local_destination"
mkdir -p "$storage_directory"

# Ensure exclude_list_file exists
if [ ! -f "$exclude_list_file" ]; then
    touch "$exclude_list_file"
    # echo "Created exclude list file at: $exclude_list_file"
else
    # echo "Exclude list file already exists: $exclude_list_file"
    echo 
fi


# Generating exclude list
echo "Generating exclude list from processed files..."
echo "Searching in: $storage_directory"

find "$storage_directory" -type f -name '*.dat' -exec basename {} \; > "$exclude_list_file"
# echo "Exclude list saved to: $exclude_list_file"


# Fetch all data
echo "Fetching data from $mingo_direction to $local_destination, excluding already processed files..."

echo '------------------------------------------------------'

if [[ -s "$exclude_list_file" ]]; then
  rsync -avz --exclude-from="$exclude_list_file" "$mingo_direction:$dat_files_directory"/*.dat "$local_destination"
  rm "$exclude_list_file"
  rm -r "$base_working_directory/tmp"
else
  echo "No exclude list found. Fetching all files..."
  rsync -avz "$mingo_direction:$dat_files_directory"/*.dat "$local_destination"
fi

echo '------------------------------------------------------'

# Process the data: raw_to_list.py
echo "Processing .dat files with Python script (raw_to_list.py)..."
python3 "$raw_to_list_directory" "$station"

# Process the data: event_accumulator.py
echo "Processing list files with Python script..."
python3 "$event_accumulator_directory" "$station"

echo '------------------------------------------------------'
echo "bring_and_analyze_events.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'