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

base_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE"
compressed_directory=${base_directory}/COMPRESSED_HLDS
uncompressed_directory=${base_directory}/UNCOMPRESSED_HLDS
# processed_directory=${base_directory}/ANCILLARY_DIRECTORY
moved_directory=${base_directory}/SENT_TO_RAW_TO_LIST_PIPELINE

csv_path="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/database_status_${station}.csv"
csv_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
csv_header="basename,start_date,hld_remote_add_date,hld_local_add_date,dat_add_date,list_ev_name,list_ev_add_date,acc_name,acc_add_date,merge_add_date"

ensure_csv() {
    if [[ ! -f "$csv_path" ]]; then
        printf '%s\n' "$csv_header" > "$csv_path"
    elif [[ ! -s "$csv_path" ]]; then
        printf '%s\n' "$csv_header" > "$csv_path"
    else
        local current_header
        current_header=$(head -n1 "$csv_path")
        if [[ "$current_header" != "$csv_header" ]]; then
            local upgrade_tmp
            upgrade_tmp=$(mktemp)
            {
                printf '%s\n' "$csv_header"
                tail -n +2 "$csv_path" | awk -F',' -v OFS=',' '{ while (NF < 10) { $(NF+1)="" } if (NF > 10) { NF=10 } print }'
            } > "$upgrade_tmp"
            mv "$upgrade_tmp" "$csv_path"
        fi
    fi
}

ensure_csv

strip_suffix() {
    local name="$1"
    name=${name%.hld.tar.gz}
    name=${name%.hld-tar-gz}
    name=${name%.tar.gz}
    name=${name%.hld}
    name=${name%.dat}
    printf '%s' "$name"
}

compute_start_date() {
    local name="$1"
    local base
    base=$(strip_suffix "$name")
    if [[ $base =~ ([0-9]{11})$ ]]; then
        local digits=${BASH_REMATCH[1]}
        local yy=${digits:0:2}
        local doy=${digits:2:3}
        local hhmmss=${digits:5:6}
        local hh=${hhmmss:0:2}
        local mm=${hhmmss:2:2}
        local ss=${hhmmss:4:2}
        local year=$((2000 + 10#$yy))
        local offset=$((10#$doy - 1))
        (( offset < 0 )) && offset=0
        date -d "${year}-01-01 +${offset} days ${hh}:${mm}:${ss}" '+%Y-%m-%d_%H.%M.%S' 2>/dev/null || printf ''
    else
        printf ''
    fi
}

declare -A csv_rows=()
if [[ -s "$csv_path" ]]; then
    while IFS=',' read -r existing_basename _; do
        [[ -z "$existing_basename" || "$existing_basename" == "basename" ]] && continue
        existing_basename=${existing_basename//$'\r'/}
        csv_rows["$existing_basename"]=1
    done < "$csv_path"
fi

append_row_if_missing() {
    local base="$1"
    local remote_date="$2"
    local local_date="$3"
    local dat_date="$4"
    [[ -z "$base" ]] && return
    if [[ -n ${csv_rows["$base"]+_} ]]; then
        return
    fi
    local start_value
    start_value=$(compute_start_date "$base")
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$base" "$start_value" "$remote_date" "$local_date" "$dat_date" "" "" "" "" "" >> "$csv_path"
    csv_rows["$base"]=1
}

hld_input_directory=$HOME/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/rawData/dat
asci_output_directory=$HOME/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/asci
first_stage_raw_directory=$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA/RAW
first_stage_base=$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA
first_stage_base_deep=$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/EVENT_DATA/RAW_TO_LIST/RAW_TO_LIST_FILES

# mkdir -p "$uncompressed_directory" "$processed_directory" "$moved_directory"
echo "Creating necessary directories..."
mkdir -p "$uncompressed_directory" "$moved_directory" "$first_stage_raw_directory"

pipeline_before=$(mktemp)
pipeline_after=""
pipeline_new=""

cleanup_pipeline() {
    rm -f "$pipeline_before"
    [[ -n "$pipeline_after" ]] && rm -f "$pipeline_after"
    [[ -n "$pipeline_new" ]] && rm -f "$pipeline_new"
}

trap cleanup_pipeline EXIT

find "$moved_directory" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' | sort -u > "$pipeline_before"

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

selected_base=$(basename "${selected_file%.hld}")
append_row_if_missing "$selected_base" "" "$csv_timestamp" ""

awk -F',' -v OFS=',' -v key="$selected_base" -v ts="$csv_timestamp" '
    NR == 1 { print; next }
    {
        if ($1 == key && $4 == "") {
            $4 = ts
        }
        print
    }
' "$csv_path" > "${csv_path}.tmp" && mv "${csv_path}.tmp" "$csv_path"

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
$HOME/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/bin/unpack.sh
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

pipeline_after=$(mktemp)
find "$moved_directory" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' | sort -u > "$pipeline_after"

pipeline_new=$(mktemp)
comm -13 "$pipeline_before" "$pipeline_after" > "$pipeline_new"

if [[ -s "$pipeline_after" ]]; then
    while IFS= read -r dat_entry; do
        dat_entry=${dat_entry//$'\r'/}
        [[ -z "$dat_entry" ]] && continue
        dat_base=$(strip_suffix "$dat_entry")
        append_row_if_missing "$dat_base" "" "$csv_timestamp" "$csv_timestamp"
    done < "$pipeline_after"
fi

if [[ -s "$pipeline_after" || -s "$pipeline_new" ]]; then
    awk -F',' -v OFS=',' -v all="$pipeline_after" -v new="$pipeline_new" -v ts="$csv_timestamp" '
        function canonical(name) {
            gsub(/\r/, "", name)
            sub(/\.hld\.tar\.gz$/, "", name)
            sub(/\.hld-tar-gz$/, "", name)
            sub(/\.tar\.gz$/, "", name)
            sub(/\.hld$/, "", name)
            sub(/\.dat$/, "", name)
            return name
        }
        BEGIN {
            if (all != "") {
                while ((getline line < all) > 0) {
                    if (line == "") { continue }
                    roots_all[canonical(line)] = 1
                }
                close(all)
            }
            if (new != "") {
                while ((getline line < new) > 0) {
                    if (line == "") { continue }
                    roots_new[canonical(line)] = 1
                }
                close(new)
            }
        }
        NR == 1 { print; next }
        {
            base = canonical($1)
            if (base == "") {
                print
                next
            }
            if (base in roots_new) {
                if ($4 == "") {
                    $4 = ts
                }
                $5 = ts
            } else if (base in roots_all) {
                if ($4 == "") {
                    $4 = ts
                }
                if ($5 == "") {
                    $5 = ts
                }
            }
            print
        }
    ' "$csv_path" > "${csv_path}.tmp" && mv "${csv_path}.tmp" "$csv_path"
fi

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

BASE_ROOT="$HOME/DATAFLOW_v3/STATIONS"
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
