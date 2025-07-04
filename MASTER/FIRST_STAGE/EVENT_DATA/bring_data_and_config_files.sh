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
echo "bring_and_analyze_events.sh started on: $(date)"
echo "Station: $script_args"
echo "Running the script..."
echo "------------------------------------------------------"
# --------------------------------------------------------------------------------------------


dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# Define base working directory
station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station"
base_working_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA"

# Define directories
local_destination="$base_working_directory/RAW"
storage_directory="$base_working_directory/RAW_TO_LIST"

# Additional paths
mingo_direction="mingo0$station"

exclude_list_file="$base_working_directory/tmp/exclude_list.txt"

# Create necessary directories
mkdir -p "$station_directory"
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
  echo "Files currently available on $mingo_direction:"
  ssh "$mingo_direction" "ls -lh $dat_files_directory"/*.dat

  echo "Fetching new data files, excluding those already processed..."
  rsync -avz --exclude-from="$exclude_list_file" "$mingo_direction:$dat_files_directory"/*.dat "$local_destination"

  rm "$exclude_list_file"
  rm -r "$base_working_directory/tmp"
else
  echo "No exclude list found. Fetching all files..."
  echo "Files currently available on $mingo_direction:"
  ssh "$mingo_direction" "ls -lh $dat_files_directory"/*.dat

  rsync -avz "$mingo_direction:$dat_files_directory"/*.dat "$local_destination"
fi


echo '------------------------------------------------------'
echo '------------------------------------------------------'

# Bring the input files from the logbook
echo "Bringing the input files from the logbook..."

# Google Sheet ID (common for all stations)
SHEET_ID="1ato36QkIXCxFkDT_LtAaLjPP7pvLcor-xZAP4fy00l0"

# Mapping of station numbers to their respective GIDs
declare -A STATION_GID_MAP
STATION_GID_MAP[1]="1331842924"
STATION_GID_MAP[2]="600987525"
STATION_GID_MAP[3]="376764978"
STATION_GID_MAP[4]="1268265225"

# Get the corresponding GID
GID=${STATION_GID_MAP[$station]}

# Define output file path
OUTPUT_FILE="$station_directory/input_file_mingo0${station}.csv"

# Download the file using wget with minimal console output
echo "Downloading logbook for Station $station..."
wget -q --show-progress --no-check-certificate \
     "https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv&id=${SHEET_ID}&gid=${GID}" \
     -O "${OUTPUT_FILE}"


# Check if download was successful
if [[ $? -eq 0 ]]; then
    echo "Download completed. Data saved at ${OUTPUT_FILE}."
else
    echo "Error: Download failed. Continuing execution..."
fi


echo '------------------------------------------------------'
echo "bring_data_and_config_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'