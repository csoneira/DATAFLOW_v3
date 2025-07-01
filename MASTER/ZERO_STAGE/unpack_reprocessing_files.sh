#!/bin/bash
# Usage: ./unpack_reprocessing_files.sh <station>
# Example: ./unpack_reprocessing_files.sh 1

if [ $# -ne 1 ]; then
    echo "Usage: $0 <station>"
    exit 1
fi

station=$1

base_directory=/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE
compressed_directory=${base_directory}/COMPRESSED_HLDS
uncompressed_directory=${base_directory}/UNCOMPRESSED_HLDS
processed_directory=${base_directory}/ASCII
moved_directory=${base_directory}/MOVED_ASCII

hld_input_directory=/media/externalDisk/gate/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory=/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci
first_stage_raw_directory=/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY

mkdir -p "$uncompressed_directory" "$processed_directory" "$moved_directory"

echo "Unpacking HLD tarballs..."
for file in "$compressed_directory"/*.tar.gz; do
    [ -e "$file" ] || continue
    tar -xvzf "$file" --strip-components=3 -C "$uncompressed_directory"
done

rm -f "$compressed_directory"/*.tar.gz


echo "Moving unpacked HLDs to DAQ input directory..."
mv "$uncompressed_directory"/* "$hld_input_directory/"

echo "Running unpacking..."
export RPCSYSTEM=mingo0$station
export RPCRUNMODE=False # Other option is oneRun
/home/cayetano/gate/bin/unpack.sh

echo "Moving ASCII files to processed and unprocessed directories..."
mv "$asci_output_directory"/* "$processed_directory/"
cp -n "$processed_directory"/* "$first_stage_raw_directory/"
mv "$processed_directory"/* "$moved_directory/"

echo "Unpacking and ASCII file handling completed."
