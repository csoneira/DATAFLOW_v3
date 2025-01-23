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

# Check if dates are provided, set flag for skipping steps
skip_steps=0
if [ -z "$2" ] || [ -z "$3" ]; then
    echo "Warning: No dates provided. Skipping steps 1–4 and jumping to step 5."
    skip_steps=1
fi

station=$1 # Station number: 1, 2, 3, or 4

base_directory=/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0$station/ZERO_STAGE

hld_input_directory=/media/externalDisk/gate/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory=/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci

first_stage_raw_directory=/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA/RAW

compressed_directory=$base_directory/COMPRESSED_HLDS
uncompressed_directory=$base_directory/UNCOMPRESSED_HLDS
processed_directory=$base_directory/ASCII

# Create necessary directories
mkdir -p $compressed_directory
mkdir -p $uncompressed_directory
mkdir -p $processed_directory

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
if [ "$skip_steps" -eq 0 ]; then
    echo "Moving uncompressed files to HLD input directory..."
    mv $uncompressed_directory/* $hld_input_directory/
fi

# Step 5. Execute unpacking
echo "Executing unpacking process..."
export RPCSYSTEM=mingo0$station
export RPCRUNMODE=oneRun
/home/cayetano/gate/bin/unpack.sh

# Step 6. Move ASCII files to the processed directory
echo "Moving ASCII files to processed directory..."
mv $asci_output_directory/* $processed_directory/

# Step 7. Copy ASCII files to the first-stage raw directory
echo "Copying ASCII files to the first-stage raw directory..."
cp $processed_directory/* $first_stage_raw_directory/

echo "Reprocessing completed successfully!"
