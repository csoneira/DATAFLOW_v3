#!/bin/bash



# Usage: ./reprocessing.sh <station> <start_date> <end_date>
# station: Station number (1, 2, 3, or 4)
# start_date: Start date in YYMMDD format
# end_date: End date in YYMMDD format
# If no start_date and end_date are provided, the script will skip steps 1–4 and directly move to step 5.

# Check if the station is provided
if [ -z "$1" ]; then
    echo "Error: No station provided. Usage: ./reprocessing.sh <station> <start_date> <end_date>"
    exit 1
fi


# --------------------------------------------------------------------------------------------
# Prevent the script from running multiple instances -----------------------------------------
# --------------------------------------------------------------------------------------------

# # Variables
# script_name=$(basename "$0")
# script_args="$*"
# current_pid=$$

# # Check for duplicate process
# for pid in $(pgrep -f "$script_name"); do
#     if [ "$pid" != "$current_pid" ]; then
#         # Compare arguments of the found process with the current one
#         cmdline=$(cat /proc/$pid/cmdline | tr '\0' ' ')
#         if [[ "$cmdline" == *"$script_name"* && "$cmdline" == *"$script_args"* ]]; then
#             echo "------------------------------------------------------"
#             echo "bring_and_analyze_events.sh started on: $(date)"
#             echo "Station: $script_args"
#             echo "The script $script_name with arguments '$script_args' is already running (PID: $pid). Exiting."
#             echo "------------------------------------------------------"
#             exit 1
#         fi
#     fi
# done

# # If no duplicate process is found, continue
# echo "------------------------------------------------------"
# echo "bring_and_analyze_events.sh started on: $(date)"
# echo "Station: $script_args"
# echo "Running the script..."
# echo "------------------------------------------------------"


# Variables
# script_name=$(basename "$0")
# script_args="$*"
# current_pid=$$

# Debug: Check for running processes
# echo "$(date) - Checking for existing processes of $script_name with args $script_args"
# ps -eo pid,cmd | grep "[b]ash .*/$script_name"

# Get all running instances of the script *with the same argument*, but exclude the current process
# for pid in $(ps -eo pid,cmd | grep "[b]ash .*/$script_name" | awk '{print $1}'); do
#     if [[ "$pid" != "$current_pid" ]]; then
#         cmdline=$(ps -p "$pid" -o args=)
#         # echo "$(date) - Found running process: PID $pid - $cmdline"
#         if [[ "$cmdline" == *"$script_name $script_args"* ]]; then
#             echo "------------------------------------------------------"
#             echo "$(date): The script $script_name with arguments '$script_args' is already running (PID: $pid). Exiting."
#             echo "------------------------------------------------------"
#             exit 1
#         fi
#     fi
# done

script_name=$(basename "$0")
script_args="$*"
station="$1"
lock_file="/tmp/${script_name}_${station}.lock"

# exec > /tmp/reprocessing_${station}.log 2>&1
echo "$(date): Script started"

# Check for existing lock file
if [ -f "$lock_file" ]; then
    echo "***************************************************"
    echo "***************************************************"
    echo "$(date): Script is already running. Exiting."
    echo "***************************************************"
    echo "***************************************************"
    exit 1
fi

echo "$(date): No other instances found. Executing script..."
# Create lock file
touch "$lock_file"
# Ensure lock file is removed on script exit
trap 'rm -f "$lock_file"' EXIT
# Main script logic
echo "$(date): No other instances found. Executing script..."


# --------------------------------------------------------------------------------------------


# Check if dates are provided, set flag for skipping steps
skip_steps=0
if [ -z "$2" ] || [ -z "$3" ]; then
    echo "Warning: No dates provided. Skipping steps 1–2 and jumping to step 3."
    skip_steps=1
fi

station=$1 # Station number: 1, 2, 3, or 4

base_directory=/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0$station/ZERO_STAGE

hld_input_directory=/media/externalDisk/gate/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory=/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci

first_stage_raw_directory=/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY

compressed_directory=$base_directory/COMPRESSED_HLDS
uncompressed_directory=$base_directory/UNCOMPRESSED_HLDS
processed_directory=$base_directory/ASCII
moved_directory=$base_directory/MOVED_ASCII

# Create necessary directories
mkdir -p $compressed_directory
mkdir -p $uncompressed_directory
mkdir -p $processed_directory
mkdir -p $moved_directory

# Step 1. Convert date range to YYDDDHHMMSS format if dates are provided
if [ "$skip_steps" -eq 0 ]; then
    start=$2
    end=$3

    start_DOY=$(date -d "20${start:0:2}-${start:2:2}-${start:4:2}" +%y%j)
    end_DOY=$(date -d "20${end:0:2}-${end:2:2}-${end:4:2}" +%y%j)

    echo "Date range converted to DOY format: $start_DOY to $end_DOY"
fi

# Step 2. Collect compressed HLDs from backup server
# if [ "$skip_steps" -eq 0 ]; then
#     echo "Collecting compressed HLDs from backup server..."
#     scp "rpcuser@backuplip:/local/experiments/MINGOS/MINGO0$station/mi0${station}*{$start_DOY..$end_DOY}*.hld*" $compressed_directory/
# fi

if [ "$skip_steps" -eq 0 ]; then
    echo "Collecting compressed HLDs from backup server..."
    
    # List all matching files first to provide immediate feedback
    echo "Listing files to be transferred:"
    ssh backuplip "ls /local/experiments/MINGOS/MINGO0$station/mi0${station}*{$start_DOY..$end_DOY}*.hld*" || {
        echo "Error: No files found on the server. Check the date range and station."
        # exit 1
    }

    ssh backuplip "ls /local/experiments/MINGOS/MINGO0$station/minI${station}*{$start_DOY..$end_DOY}*.hld*" || {
        echo "Error: No files found on the server. Check the date range and station."
        # exit 1
    }
    
    # Transfer files one by one with progress feedback
    echo "Transferring files:"
    for file in $(ssh backuplip "ls /local/experiments/MINGOS/MINGO0$station/mi0${station}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null); do
        echo "Transferring $file ..."
        scp backuplip:"$file" "$compressed_directory/" || {
            echo "Error: Failed to transfer $file"
            # exit 1
        }
    done

    for file in $(ssh backuplip "ls /local/experiments/MINGOS/MINGO0$station/minI${station}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null); do
        echo "Transferring $file ..."
        scp backuplip:"$file" "$compressed_directory/" || {
            echo "Error: Failed to transfer $file"
            # exit 1
        }
    done
    echo "All files transferred successfully."
fi


# Step 3. Uncompress the HLD files
if [ "$skip_steps" -eq 0 ]; then
    echo "Uncompressing HLD files..."
    for file in $compressed_directory/*.tar.gz; do
        tar -xvzf "$file" --strip-components=3 -C $uncompressed_directory
    done

    # Remove the compressed files
    rm $compressed_directory/*.tar.gz
fi

# Step 4. Move uncompressed files to the HLD input directory
if [ "$skip_steps" -gt 0 ]; then
    echo "Moving uncompressed files to HLD input directory..."
    mv $uncompressed_directory/* $hld_input_directory/
fi

# Step 5. Execute unpacking
echo "***************************************************"
echo "Executing unpacking process..."
export RPCSYSTEM=mingo0$station
# export RPCRUNMODE=oneRun
export RPCRUNMODE=False
/home/cayetano/gate/bin/unpack.sh

# Step 6. Move ASCII files to the processed directory
echo "Moving ASCII files to processed directory..."
mv $asci_output_directory/* $processed_directory/

# Step 7. Copy ASCII files to the first-stage raw directory
echo "Copying ASCII files to the first-stage unprocessed directory..."
cp -n $processed_directory/* $first_stage_raw_directory/


# Step 8. Move ASCII files to an already used directory
echo "Moving ASCII files to a directory for copied files..."
mv "$processed_directory"/* "$moved_directory"/

echo "Reprocessing completed successfully!"
