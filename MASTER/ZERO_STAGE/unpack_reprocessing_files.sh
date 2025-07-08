#!/bin/bash
# Usage: ./unpack_reprocessing_files.sh <station>
# Example: ./unpack_reprocessing_files.sh 1

if [ $# -ne 1 ]; then
    echo "Usage: $0 <station>"
    exit 1
fi

random_file=false  # set to true to enable random selection

station=$1


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
echo "unpack_reprocessing_files.sh started on: $(date)"
echo "Station: $script_args"
echo "Running the script..."
echo "------------------------------------------------------"

# --------------------------------------------------------------------------------------------

base_directory=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE
compressed_directory=${base_directory}/COMPRESSED_HLDS
uncompressed_directory=${base_directory}/UNCOMPRESSED_HLDS
# processed_directory=${base_directory}/ANCILLARY_DIRECTORY
moved_directory=${base_directory}/SENT_TO_RAW_TO_LIST_PIPELINE

hld_input_directory=/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory=/home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/asci
first_stage_raw_directory=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA/RAW
first_stage_base=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA
first_stage_base_deep=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES

# mkdir -p "$uncompressed_directory" "$processed_directory" "$moved_directory"
echo "Creating necessary directories..."
mkdir -p "$uncompressed_directory" "$moved_directory" "$first_stage_raw_directory"

# Now Copy to first_stage_raw_directory every file from moved_directory that is not in any first_stage_base subdirectory (recursively)

printf "\n>> Moving files from:\n    %s\n--> to:\n    %s\n-> only if not already present in:\n    %s, %s, or %s\n\n" \
"$moved_directory" "$first_stage_raw_directory" "$first_stage_base" "$first_stage_raw_directory" "$first_stage_base_deep"

# Get list of all existing filenames under FIRST_STAGE/EVENT_DATA recursively
# mapfile -t existing_files < <(find "$first_stage_base" -type f -exec basename {} \; | sort -u)

mapfile -t existing_files < <(
    find "$first_stage_raw_directory" "$first_stage_base_deep" \
    -type f -printf '%f\n' | sort -u
)


# Copy files from moved_directory only if not already present (by name) in first_stage_base
find "$moved_directory" -maxdepth 1 -type f | while read -r src_file; do
    fname=$(basename "$src_file")
    if ! printf "%s\n" "${existing_files[@]}" | grep -Fxq "$fname"; then
        echo "Moving $fname to RAW directory"
        cp "$src_file" "$first_stage_raw_directory/"
        # Update mdate to current time of the copied file
        touch "$first_stage_raw_directory/$fname"
    # else
        # echo "Skipping $fname — already exists in EVENT_DATA."
    fi
done

# echo ""
# echo "Unpacking HLD tarballs..."
# for file in "$compressed_directory"/*.tar.gz; do
#     [ -e "$file" ] || continue
#     tar -xvzf "$file" --strip-components=3 -C "$uncompressed_directory"
# done

echo ""
echo "Unpacking HLD tarballs and removing archives..."
for file in "$compressed_directory"/*.tar.gz; do
    [ -e "$file" ] || continue
    if tar -xvzf "$file" --strip-components=3 -C "$uncompressed_directory"; then
        echo "Successfully untared $file"
        # Touch the last untared file to update their modification date in this iteration
        last_untared_file=$(ls -t "$uncompressed_directory"/*.hld | head -n 1)
        touch "$last_untared_file"
        echo "Updated modification date for $last_untared_file"
        rm "$file"
    else
        echo "Warning: Failed to unpack $file" >&2
    fi
done

# rm -f "$compressed_directory"/*.tar.gz

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
# List all the files that will be moved
find "$asci_output_directory" -maxdepth 1 -type f -name '*.dat' -print

for file in "$asci_output_directory"/*.dat; do
    touch "$file"
done

echo ""
cp "$asci_output_directory"/*.dat "$first_stage_raw_directory/"
mv "$asci_output_directory"/*.dat "$moved_directory/"

# # Update mdate to current time of the copied and moved files
# for file in "$first_stage_raw_directory"/*.dat; do
#     touch "$file"
# done

# for file in "$moved_directory"/*.dat; do
#     touch "$file"
# done

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


# Reordering the unpcaked files if needed according to its stations

BASE_ROOT="/home/mingo/DATAFLOW_v3/STATIONS"
SUBDIRS=("COMPRESSED_HLDS" "UNCOMPRESSED_HLDS" "SENT_TO_RAW_TO_LIST_PIPELINE")

# Loop through station numbers 1 to 4
for station in {1..4}; do
    station_id="0${station}"
    station_dir="${BASE_ROOT}/MINGO${station_id}/ZERO_STAGE"

    for subdir in "${SUBDIRS[@]}"; do
        current_dir="${station_dir}/${subdir}"
        [[ -d "$current_dir" ]] || continue

        find "$current_dir" -maxdepth 1 -type f -name "mi0*.dat" | while read -r file; do
            filename=$(basename "$file")
            file_station=${filename:2:2}  # Extract "0X" from "mi0X..."
            
            # If file_station does not match current station, move it
            if [[ "$file_station" != "$station_id" ]]; then
                target_dir="${BASE_ROOT}/MINGO${file_station}/ZERO_STAGE/${subdir}"
                echo "→ Moving $filename from MINGO${station_id}/${subdir} to MINGO${file_station}/${subdir}"
                mkdir -p "$target_dir"
                mv "$file" "$target_dir/"
            fi
        done
    done
done



echo '------------------------------------------------------'
echo "unpack_reprocessing_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'