#!/bin/bash
# Usage: ./unpack_reprocessing_files.sh <station>
# Example: ./unpack_reprocessing_files.sh 1

# What to really change when the directory is changed:
#    unpack.sh, of course, the cd should lead to software
#    initConf.m, the HOME line, THAT MUST END WITH A SLASH
#    

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
    cat <<'EOF'
unpack_reprocessing_files.sh
Unpacks compressed HLD archives and prepares data for STAGE_0 processing.

Usage:
  unpack_reprocessing_files.sh <station> [--loop|-l]

Options:
  -h, --help    Show this help message and exit.
  -l, --loop    Process every pending HLD sequentially (repeat single-run workflow).

Provide the numeric station identifier (1-4). The script ensures only one
instance runs per-station and operates on files queued in STAGE_0 buffers.
EOF
    exit 0
fi

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 <station> [--loop|-l]"
    exit 1
fi

random_file=false  # set to true to enable random selection

original_args="$*"
station=$1
shift

loop_mode=false

while (( $# > 0 )); do
    case "$1" in
        --loop|-l)
            loop_mode=true
            ;;
        *)
            echo "Usage: $0 <station> [--loop|-l]"
            exit 1
            ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
    MASTER_DIR="$(dirname "${MASTER_DIR}")"
done


# --------------------------------------------------------------------------------------------
# Prevent the script from running multiple instances -----------------------------------------
# --------------------------------------------------------------------------------------------

# Variables
script_name=$(basename "$0")
script_args="$original_args"
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
echo "Station: $station (loop_mode=$loop_mode)"
echo "Running the script..."
echo "------------------------------------------------------"

# --------------------------------------------------------------------------------------------

station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}"
reprocessing_directory="${station_directory}/STAGE_0/REPROCESSING"
input_directory="${reprocessing_directory}/INPUT_FILES"
compressed_directory="${input_directory}/COMPRESSED_HLDS"
uncompressed_directory="${input_directory}/UNCOMPRESSED_HLDS"
metadata_directory="${reprocessing_directory}/METADATA"
stage0_to_1_directory="${station_directory}/STAGE_0_to_1"

mkdir -p "$compressed_directory" "$uncompressed_directory" "$metadata_directory" "$stage0_to_1_directory"

dat_unpacked_csv="${metadata_directory}/dat_files_unpacked.csv"
dat_unpacked_header="dat_name,execution_timestamp,execution_duration_s"

csv_path="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/database_status_${station}.csv"
csv_header="basename,start_date,hld_remote_add_date,hld_local_add_date,dat_add_date,list_ev_name,list_ev_add_date,acc_name,acc_add_date,merge_add_date"

ensure_dat_unpacked_csv() {
    if [[ ! -f "$dat_unpacked_csv" || ! -s "$dat_unpacked_csv" ]]; then
        printf '%s\n' "$dat_unpacked_header" > "$dat_unpacked_csv"
    fi
}

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

hld_input_directory=$HOME/DATAFLOW_v3/MASTER/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/rawData/dat # <--------------------------------------------
asci_output_directory=$HOME/DATAFLOW_v3/MASTER/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/system/devices/TRB3/data/daqData/asci # <--------------------------------------------
dest_directory="$stage0_to_1_directory"

process_single_hld() {
    local script_start_time script_start_epoch script_end_epoch script_duration
    local csv_timestamp
    local -a new_dat_files=()

    script_start_time="$(date '+%Y-%m-%d %H:%M:%S')"
    script_start_epoch=$(date +%s)
    csv_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "Creating necessary directories..."
    mkdir -p "$uncompressed_directory" "$dest_directory"

    echo ""
    echo "Unpacking HLD tarballs and removing archives..."
    shopt -s nullglob
    for file in "$compressed_directory"/*.tar.gz; do
        [ -e "$file" ] || continue
        if tar -xvzf "$file" --strip-components=3 -C "$uncompressed_directory"; then
            echo "Successfully untared $file"
            local last_untared_file
            last_untared_file=$(ls -t "$uncompressed_directory"/*.hld 2>/dev/null | head -n 1)
            if [[ -n "$last_untared_file" ]]; then
                touch "$last_untared_file"
                echo "Updated modification date for $last_untared_file"
            fi
            rm "$file"
        else
            echo "Warning: Failed to unpack $file" >&2
        fi
    done
    shopt -u nullglob

    if [ -d "$hld_input_directory" ]; then
        echo "Moving existing HLD files to removed directory..."
        mkdir -p "$hld_input_directory/removed"
        mv "$hld_input_directory"/*.hld* "$hld_input_directory/removed/" 2>/dev/null
    fi

    if [ -d "$asci_output_directory" ]; then
        echo "Moving existing dat files to removed directory..."
        mkdir -p "$asci_output_directory/removed"
        mv "$asci_output_directory"/*.dat* "$asci_output_directory/removed/" 2>/dev/null
    fi

    echo "Selecting one HLD file to unpack..."
    shopt -s nullglob
    local hld_files=("$uncompressed_directory"/*.hld)
    shopt -u nullglob

    if [ ${#hld_files[@]} -eq 0 ]; then
        echo "No HLD files found in $uncompressed_directory"
        return 1
    fi

    local selected_file
    if [ "$random_file" = true ]; then
        selected_file="${hld_files[RANDOM % ${#hld_files[@]}]}"
    else
        local -a sorted=()
        IFS=$'\n' sorted=($(sort <<<"${hld_files[*]}"))
        unset IFS
        selected_file="${sorted[0]}"
    fi

    echo "Selected HLD file: $(basename "$selected_file")"

    local selected_base
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

    mv "$selected_file" "$hld_input_directory/"

    local filename name_no_ext prefix ss ss_val ss_new new_filename
    filename=$(basename "$selected_file")
    name_no_ext="${filename%.hld}"
    prefix="${name_no_ext:0:${#name_no_ext}-2}"
    ss="${name_no_ext: -2}"
    ss_val=$((10#$ss))

    if (( ss_val < 30 )); then
        ss_new=$(printf "%02d" $((ss_val + 1)))
    else
        ss_new=$(printf "%02d" $((ss_val - 1)))
    fi

    new_filename="${prefix}${ss_new}.hld"

    echo "Original file: $filename"
    echo "Copied as:     $new_filename"
    cp "$hld_input_directory/$filename" "$hld_input_directory/$new_filename"

    echo ""
    echo ""
    echo "Running unpacking..."
    export RPCSYSTEM=mingo0$station
    export RPCRUNMODE=oneRun

    "$HOME/DATAFLOW_v3/MASTER/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/bin/unpack.sh"

    echo ""
    echo ""

    echo "Moving dat files into STAGE_0_to_1..."
    find "$asci_output_directory" -maxdepth 1 -type f -name '*.dat' -print

    local stage0_before stage0_after stage0_new
    stage0_before=$(mktemp)
    find "$dest_directory" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' | sort -u > "$stage0_before"

    for file in "$asci_output_directory"/*.dat; do
        [ -e "$file" ] || continue
        local fname
        fname=$(basename "$file")
        touch "$file"
        if [[ -e "$dest_directory/$fname" ]]; then
            echo "Skipping $fname — already present in STAGE_0_to_1."
            continue
        fi
        if mv "$file" "$dest_directory/$fname"; then
            touch "$dest_directory/$fname"
            echo "Moved $fname to STAGE_0_to_1."
        else
            echo "Warning: failed to move $fname into STAGE_0_to_1." >&2
        fi
    done

    stage0_after=$(mktemp)
    find "$dest_directory" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' | sort -u > "$stage0_after"

    stage0_new=$(mktemp)
    comm -13 "$stage0_before" "$stage0_after" > "$stage0_new"

    if [[ -s "$stage0_new" ]]; then
        while IFS= read -r dat_entry; do
            dat_entry=${dat_entry//$'\r'/}
            [[ -z "$dat_entry" ]] && continue
            new_dat_files+=("$dat_entry")
        done < "$stage0_new"
    fi

    if [[ -s "$stage0_after" ]]; then
        while IFS= read -r dat_entry; do
            dat_entry=${dat_entry//$'\r'/}
            [[ -z "$dat_entry" ]] && continue
            local dat_base
            dat_base=$(strip_suffix "$dat_entry")
            append_row_if_missing "$dat_base" "" "$csv_timestamp" "$csv_timestamp"
        done < "$stage0_after"
    fi

    if [[ -s "$stage0_after" || -s "$stage0_new" ]]; then
        awk -F',' -v OFS=',' -v all="$stage0_after" -v new="$stage0_new" -v ts="$csv_timestamp" '
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

    rm -f "$stage0_before" "$stage0_after" "$stage0_new"

    if [ -d "$hld_input_directory" ]; then
        echo "Moving existing HLD files to removed directory..."
        mkdir -p "$hld_input_directory/removed"
        mv "$hld_input_directory"/*.hld* "$hld_input_directory/removed/" 2>/dev/null
    fi

    if [ -d "$asci_output_directory" ]; then
        echo "Moving existing dat files to removed directory..."
        mkdir -p "$asci_output_directory/removed"
        mv "$asci_output_directory"/*.dat* "$asci_output_directory/removed/" 2>/dev/null
    fi

    local BASE_ROOT SUBDIRS
    BASE_ROOT="$HOME/DATAFLOW_v3/STATIONS"
    SUBDIRS=(
        "STAGE_0/REPROCESSING/INPUT_FILES/COMPRESSED_HLDS"
        "STAGE_0/REPROCESSING/INPUT_FILES/UNCOMPRESSED_HLDS"
        "STAGE_0_to_1"
    )

    local station_loop
    for station_loop in {1..4}; do
        local station_id
        station_id=$(printf "%02d" "$station_loop")
        local station_dir
        station_dir="${BASE_ROOT}/MINGO${station_id}"

        local subdir
        for subdir in "${SUBDIRS[@]}"; do
            local current_dir
            current_dir="${station_dir}/${subdir}"
            [[ -d "$current_dir" ]] || continue

            find "$current_dir" -maxdepth 1 -type f \( \
                -name "mi0*.dat" -o \
                -name "mi0*.hld" -o \
                -name "mi0*.hld.tar.gz" -o \
                -name "mi0*.hld-tar-gz" \
            \) | while read -r file; do
                local filename file_station target_dir
                filename=$(basename "$file")
                file_station=${filename:2:2}
                if [[ "$file_station" != "$station_id" ]]; then
                    target_dir="${BASE_ROOT}/MINGO${file_station}/${subdir}"
                    echo "→ Moving $filename from MINGO${station_id}/${subdir} to MINGO${file_station}/${subdir}"
                    mkdir -p "$target_dir"
                    mv "$file" "$target_dir/"
                fi
            done
        done
    done

    script_end_epoch=$(date +%s)
    script_duration=$((script_end_epoch - script_start_epoch))
    if (( script_duration < 0 )); then
        script_duration=0
    fi

    if (( ${#new_dat_files[@]} > 0 )); then
        ensure_dat_unpacked_csv
        local dat_name
        for dat_name in "${new_dat_files[@]}"; do
            printf '%s,%s,%s\n' "$dat_name" "$script_start_time" "$script_duration" >> "$dat_unpacked_csv"
        done
    fi

    echo '------------------------------------------------------'
    echo "unpack_reprocessing_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
    echo '------------------------------------------------------'

    return 0
}

if $loop_mode; then
    iteration=0
    while true; do
        if process_single_hld; then
            ((iteration++))
            continue
        fi
        if (( iteration == 0 )); then
            exit 1
        fi
        break
    done
else
    if ! process_single_hld; then
        exit 1
    fi
fi
