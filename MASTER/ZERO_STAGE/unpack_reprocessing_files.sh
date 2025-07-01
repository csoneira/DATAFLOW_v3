#!/bin/bash
# Usage: ./unpack_reprocessing_files.sh <station>
# Example: ./unpack_reprocessing_files.sh 1

if [ $# -ne 1 ]; then
    echo "Usage: $0 <station>"
    exit 1
fi

random_file=false  # set to true to enable random selection

station=$1

base_directory=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE
compressed_directory=${base_directory}/COMPRESSED_HLDS
uncompressed_directory=${base_directory}/UNCOMPRESSED_HLDS
# processed_directory=${base_directory}/ANCILLARY_DIRECTORY
moved_directory=${base_directory}/SENT_TO_RAW_TO_LIST_PIPELINE

hld_input_directory=/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory=/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/asci
first_stage_raw_directory=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY

# mkdir -p "$uncompressed_directory" "$processed_directory" "$moved_directory"
mkdir -p "$uncompressed_directory" "$moved_directory" "$first_stage_raw_directory"

echo ""
echo "Unpacking HLD tarballs..."
for file in "$compressed_directory"/*.tar.gz; do
    [ -e "$file" ] || continue
    tar -xvzf "$file" --strip-components=3 -C "$uncompressed_directory"
done

rm -f "$compressed_directory"/*.tar.gz

# If there are hlds in hld_input_directory, move them to $hld_input_directory/removed
if [ -d "$hld_input_directory" ]; then
    echo "Moving existing HLD files to removed directory..."
    mkdir -p "$hld_input_directory/removed"
    mv "$hld_input_directory"/*.hld* "$hld_input_directory/removed/" 2>/dev/null
fi

# If there are hlds in hld_input_directory, move them to $hld_input_directory/removed
if [ -d "$asci_output_directory" ]; then
    echo "Moving existing dat files to removed directory..."
    mkdir -p "$asci_output_directory/removed"
    mv "$asci_output_directory"/*.dat* "$asci_output_directory/removed/" 2>/dev/null
fi


# Choose one file to unpack
echo "Selecting one HLD file to unpack..."

shopt -s nullglob
hld_files=("$uncompressed_directory"/*.hld)

if [ ${#hld_files[@]} -eq 0 ]; then
    echo "No HLD files found in $uncompressed_directory"
    exit 1
fi

if [ "$random_file" = true ]; then
    selected_file="${hld_files[RANDOM % ${#hld_files[@]}]}"
else
    IFS=$'\n' sorted=($(sort <<<"${hld_files[*]}"))
    unset IFS
    selected_file="${sorted[0]}"
fi

echo "Selected HLD file: $(basename "$selected_file")"

# Move selected file to HLD input directory
mv "$selected_file" "$hld_input_directory/"

# Extract the numeric timestamp part assuming fixed format: mi0<station><YYJJJHHMMSS>
# Example: mi0124324083227 → timestamp is 324083227 (YYJJJHHMMSS)
filename=$(basename "$selected_file")
name_no_ext="${filename%.hld}"

prefix="${name_no_ext:0:${#name_no_ext}-2}"  # everything except last 2 chars
ss="${name_no_ext: -2}"                     # last 2 chars (SS)

ss_val=$((10#$ss))  # parse safely as decimal

if (( ss_val < 30 )); then
    ss_new=$(printf "%02d" $((ss_val + 1)))
else
    ss_new=$(printf "%02d" $((ss_val - 1)))
fi

new_filename="${prefix}${ss_new}.hld"

echo "Original file: $filename"
echo "Copied as:     $new_filename"
cp "$hld_input_directory/$filename" "$hld_input_directory/$new_filename"


# echo empty line to create break lines in the terminal
echo ""
echo ""
echo "Running unpacking..."
export RPCSYSTEM=mingo0$station
export RPCRUNMODE=oneRun # Other option is oneRun 
/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/bin/unpack.sh
# /media/externalDisk/gate/bin/unpack.sh
echo ""
echo ""

echo "Moving dat files to destiny folders, RAW included..."
cp "$asci_output_directory"/*.dat "$first_stage_raw_directory/"
mv "$asci_output_directory"/*.dat "$moved_directory/"

# If there are hlds in hld_input_directory, move them to $hld_input_directory/removed
if [ -d "$hld_input_directory" ]; then
    echo "Moving existing HLD files to removed directory..."
    mkdir -p "$hld_input_directory/removed"
    mv "$hld_input_directory"/*.hld* "$hld_input_directory/removed/" 2>/dev/null
fi

# If there are hlds in hld_input_directory, move them to $hld_input_directory/removed
if [ -d "$asci_output_directory" ]; then
    echo "Moving existing dat files to removed directory..."
    mkdir -p "$asci_output_directory/removed"
    mv "$asci_output_directory"/*.dat* "$asci_output_directory/removed/" 2>/dev/null
fi

echo "Unpacking and ASCII file handling completed."
