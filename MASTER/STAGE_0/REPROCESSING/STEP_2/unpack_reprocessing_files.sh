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
  unpack_reprocessing_files.sh <station> [--loop|-l] [--newest|-n]

Options:
  -h, --help       Show this help message and exit.
  -l, --loop       Process every pending HLD sequentially (repeat single-run workflow).
  -n, --newest     Select the newest pending HLD (after normalizing its prefix).

Provide the numeric station identifier (1-4). The script ensures only one
instance runs per-station and operates on files queued in STAGE_0 buffers.
EOF
    exit 0
fi

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 <station> [--loop|-l] [--newest|-n]"
    exit 1
fi

random_file=false  # set to true to enable random selection

original_args="$*"
station=$1
shift

if [[ ! "$station" =~ ^[0-9]+$ ]]; then
    echo "Station identifier must be a number between 1 and 4."
    exit 1
fi

station=$((10#$station))
if (( station < 1 || station > 4 )); then
    echo "Station identifier must be between 1 and 4."
    exit 1
fi

station_code=$(printf "%02d" "$station")
station_prefix=$(printf "mi0%d" "$station")
station_prefix_lower="${station_prefix,,}"

loop_mode=false
newest_mode=false

while (( $# > 0 )); do
    case "$1" in
        --loop|-l)
            loop_mode=true
            ;;
        --newest|-n)
            newest_mode=true
            ;;
        *)
            echo "Usage: $0 <station> [--loop|-l] [--newest|-n]"
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
echo "Station: ${station_code} (loop_mode=$loop_mode, newest_mode=$newest_mode)"
echo "Running the script..."

shared_lock_file="${MASTER_DIR}/STAGE_0/REPROCESSING/STEP_2/.unpack_shared.lock"
exec {shared_lock_fd}> "$shared_lock_file"
if ! flock -n "$shared_lock_fd"; then
    echo "$(date) - Another station is using the shared unpacker input; waiting for lock..."
    flock "$shared_lock_fd"
fi
echo "$(date) - Acquired shared unpacker lock."
cleanup_shared_lock() {
    if [[ -n ${shared_lock_fd:-} ]]; then
        flock -u "$shared_lock_fd"
        exec {shared_lock_fd}>&-
    fi
}
trap cleanup_shared_lock EXIT

echo "------------------------------------------------------"

# --------------------------------------------------------------------------------------------

STATIONS_BASE="$HOME/DATAFLOW_v3/STATIONS"
station_directory="${STATIONS_BASE}/MINGO${station_code}"
reprocessing_directory="${station_directory}/STAGE_0/REPROCESSING/STEP_2"
step_1_output_directory="${station_directory}/STAGE_0/REPROCESSING/STEP_1/OUTPUT_FILES"
step_1_output_compressed="${step_1_output_directory}/COMPRESSED_HLDS"
step_1_output_uncompressed="${step_1_output_directory}/UNCOMPRESSED_HLDS"
input_directory="${reprocessing_directory}/INPUT_FILES"
unprocessed_uncompressed="${input_directory}/UNPROCESSED"
completed_uncompressed="${input_directory}/COMPLETED"
processing_directory="${input_directory}/PROCESSING"
error_directory="${input_directory}/ERROR"
metadata_directory="${reprocessing_directory}/METADATA"
stage0_to_1_directory="${station_directory}/STAGE_0_to_1"
# csv_path="${station_directory}/database_status_${station_code}.csv"

mkdir -p \
    "$unprocessed_uncompressed" \
    "$completed_uncompressed" \
    "$processing_directory" \
    "$error_directory" \
    "$metadata_directory" \
    "$stage0_to_1_directory"

dat_unpacked_csv="${metadata_directory}/dat_files_unpacked.csv"
dat_unpacked_header="dat_name,execution_timestamp,execution_duration_s"
declare -A dat_unpacked_basenames=()

# csv_path="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/database_status_${station}.csv"
csv_header="basename,start_date,hld_remote_add_date,hld_local_add_date,dat_add_date,list_ev_name,list_ev_add_date,acc_name,acc_add_date,merge_add_date"

ensure_dat_unpacked_csv() {
    if [[ ! -f "$dat_unpacked_csv" || ! -s "$dat_unpacked_csv" ]]; then
        printf '%s\n' "$dat_unpacked_header" > "$dat_unpacked_csv"
    fi
}

load_dat_unpacked_basenames() {
    dat_unpacked_basenames=()
    if [[ ! -s "$dat_unpacked_csv" ]]; then
        return
    fi
    local first=true
    while IFS=',' read -r dat_name _rest; do
        if $first; then
            first=false
            continue
        fi
        dat_name=${dat_name//$'\r'/}
        [[ -z "$dat_name" ]] && continue
        local base
        base=$(strip_suffix "$dat_name")
        [[ -z "$base" ]] && base="$dat_name"
        dat_unpacked_basenames["$base"]=1
    done < "$dat_unpacked_csv"
}

# ensure_csv() {
#     if [[ ! -f "$csv_path" || ! -s "$csv_path" ]]; then
#         printf '%s\n' "$csv_header" > "$csv_path"
#         return
#     fi
#     local current_header
#     current_header=$(head -n1 "$csv_path")
#     if [[ "$current_header" != "$csv_header" ]]; then
#         local upgrade_tmp
#         upgrade_tmp=$(mktemp)
#         {
#             printf '%s\n' "$csv_header"
#             tail -n +2 "$csv_path" | awk -F',' -v OFS=',' '{ while (NF < 10) { $(NF+1)="" } if (NF > 10) { NF=10 } print }'
#         } > "$upgrade_tmp"
#         mv "$upgrade_tmp" "$csv_path"
#     fi
# }

# ensure_csv

strip_suffix() {
    local name="$1"
    name=${name%.hld.tar.gz}
    name=${name%.hld-tar-gz}
    name=${name%.tar.gz}
    name=${name%.hld}
    name=${name%.dat}
    printf '%s' "$name"
}

basename_time_key() {
    local name="$1"
    local base
    base=$(strip_suffix "$name")
    if [[ $base =~ ([0-9]{11})$ ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    else
        printf ''
    fi
}

order_key_after_prefix() {
    local base="$1"
    local normalized="${base,,}"
    if [[ $normalized == mini* ]]; then
        normalized="mi01${normalized:4}"
    fi
    if [[ -n "$station_prefix_lower" && $normalized == ${station_prefix_lower}* ]]; then
        printf '%s' "${normalized:${#station_prefix_lower}}"
        return 0
    fi
    local key
    key=$(basename_time_key "$normalized")
    if [[ -n "$key" ]]; then
        printf '%s' "$key"
        return 0
    fi
    printf '%s' "$normalized"
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

extract_station_code_from_name() {
    local artifact="${1##*/}"
    local lowered="${artifact,,}"
    if [[ "$lowered" =~ ^mini ]]; then
        printf '01'
        return 0
    fi
    if [[ "$lowered" =~ ^mi0([0-9]) ]]; then
        local digit="${BASH_REMATCH[1]}"
        case "$digit" in
            1|2|3|4)
                printf '%02d' "$digit"
                return 0
                ;;
        esac
    fi
    return 1
}

relocate_artifact_to_station() {
    local file_path="$1"
    local target_code="$2"
    local relative_subdir="$3"
    if [[ -z "$file_path" || -z "$target_code" || -z "$relative_subdir" ]]; then
        return 1
    fi
    local target_dir="${STATIONS_BASE}/MINGO${target_code}/${relative_subdir}"
    mkdir -p "$target_dir"
    local destination="${target_dir}/$(basename "$file_path")"
    echo "--> Redirecting $(basename "$file_path") to ${target_dir}"
    mv "$file_path" "$destination"
}

# declare -A csv_rows=()
# if [[ -s "$csv_path" ]]; then
#     while IFS=',' read -r existing_basename _; do
#         [[ -z "$existing_basename" || "$existing_basename" == "basename" ]] && continue
#         existing_basename=${existing_basename//$'\r'/}
#         csv_rows["$existing_basename"]=1
#     done < "$csv_path"
# fi

# append_row_if_missing() {
#     local base="$1"
#     local remote_date="$2"
#     local local_date="$3"
#     local dat_date="$4"
#     [[ -z "$base" ]] && return
#     if [[ -n ${csv_rows["$base"]+_} ]]; then
#         return
#     fi
#     local start_value
#     start_value=$(compute_start_date "$base")
#     printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
#         "$base" "$start_value" "$remote_date" "$local_date" "$dat_date" "" "" "" "" "" >> "$csv_path"
#     csv_rows["$base"]=1
# }

move_step1_outputs_to_unprocessed() {
    local moved_any=false
    shopt -s nullglob

    if [[ -d "$step_1_output_compressed" ]]; then
        for archive in "$step_1_output_compressed"/*.tar.gz "$step_1_output_compressed"/*.hld-tar-gz; do
            [[ -e "$archive" ]] || continue
            echo "Extracting $(basename "$archive") into UNPROCESSED..."
            if tar -xvzf "$archive" --strip-components=3 -C "$unprocessed_uncompressed"; then
                moved_any=true
                rm -f "$archive"
            else
                echo "Warning: failed to extract $(basename "$archive")." >&2
            fi
        done
    fi

    if [[ -d "$step_1_output_uncompressed" ]]; then
        for file in "$step_1_output_uncompressed"/*.hld "$step_1_output_uncompressed"/*.HLD; do
            [[ -e "$file" ]] || continue
            local target
            target="$unprocessed_uncompressed/$(basename "$file")"
            if [[ -e "$target" ]]; then
                echo "Skipping $(basename "$file") — already present in UNPROCESSED."
                continue
            fi
            mv "$file" "$target"
            moved_any=true
        done
    fi

    shopt -u nullglob

    if $moved_any; then
        echo "Moved STEP_1 outputs into UNPROCESSED queue."
    fi
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

    load_dat_unpacked_basenames
    echo "Loaded ${#dat_unpacked_basenames[@]} previously unpacked basenames as exclusions."

    echo "Creating necessary directories..."
    mkdir -p \
        "$unprocessed_uncompressed" \
        "$completed_uncompressed" \
        "$processing_directory" \
        "$error_directory" \
        "$dest_directory" \
        "$hld_input_directory" \
        "$asci_output_directory"

    move_step1_outputs_to_unprocessed

    echo ""
    echo "Ready to process uncompressed HLD files from UNPROCESSED..."

    shopt -s nullglob
    local -a stale_processing=("$processing_directory"/*.hld "$processing_directory"/*.HLD)
    if (( ${#stale_processing[@]} > 0 )); then
        echo "Found file(s) already in PROCESSING; moving them to ERROR."
        for stale in "${stale_processing[@]}"; do
            [[ -e "$stale" ]] || continue
            local stale_name
            stale_name=$(basename "$stale")
            mv "$stale" "$error_directory/$stale_name"
            echo "  -> $stale_name moved to ERROR."
        done
    fi
    shopt -u nullglob

    if [ -d "$hld_input_directory" ]; then
        echo "Clearing leftover HLD files from unpacker input..."
        shopt -s nullglob
        for leftover in "$hld_input_directory"/*.hld*; do
            [[ -e "$leftover" ]] || continue
            local name
            name=$(basename "$leftover")
            if [[ -e "$completed_uncompressed/$name" ]]; then
                rm -f "$completed_uncompressed/$name"
            fi
            mv "$leftover" "$completed_uncompressed/$name"
        done
        shopt -u nullglob
    fi

    if [ -d "$asci_output_directory" ]; then
        echo "Moving existing dat files to removed directory..."
        mkdir -p "$asci_output_directory/removed"
        mv "$asci_output_directory"/*.dat* "$asci_output_directory/removed/" 2>/dev/null
    fi

    echo "Selecting one HLD file to unpack..."
    local relocated_candidates=false
    shopt -s nullglob
    local -a candidate_files=("$unprocessed_uncompressed"/*.hld "$unprocessed_uncompressed"/*.HLD)
    local source_stage="UNPROCESSED"

    if (( ${#candidate_files[@]} == 0 )); then
        echo "UNPROCESSED is empty; checking PROCESSING."
        local -a processing_candidates=("$processing_directory"/*.hld "$processing_directory"/*.HLD)
        if (( ${#processing_candidates[@]} > 0 )); then
            for proc in "${processing_candidates[@]}"; do
                [[ -e "$proc" ]] || continue
                local proc_name
                proc_name=$(basename "$proc")
                echo "  -> Found $proc_name in PROCESSING; moving to ERROR."
                mv "$proc" "$error_directory/$proc_name"
            done
        fi
        echo "UNPROCESSED and PROCESSING are empty so we are done for now."
        # candidate_files=("$completed_uncompressed"/*.hld "$completed_uncompressed"/*.HLD)
        # candidate_files=("$completed_uncompressed"/*.hld "$completed_uncompressed"/*.HLD)
        # source_stage="COMPLETED"
    fi
    shopt -u nullglob

    if (( ${#candidate_files[@]} > 0 )); then
        local -a filtered_candidates=()
        local candidate
        for candidate in "${candidate_files[@]}"; do
            [[ -e "$candidate" ]] || continue
            local candidate_station=""
            candidate_station=$(extract_station_code_from_name "$candidate" 2>/dev/null || true)
            if [[ -n "$candidate_station" && "$candidate_station" != "$station_code" ]]; then
                if relocate_artifact_to_station "$candidate" "$candidate_station" "STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/UNPROCESSED"; then
                    relocated_candidates=true
                    continue
                else
                    echo "Warning: failed to relocate $(basename "$candidate") to station $candidate_station." >&2
                fi
            fi
            local candidate_base
            candidate_base=$(strip_suffix "$(basename "$candidate")")
            [[ -z "$candidate_base" ]] && candidate_base="$(basename "$candidate")"
            if [[ -n ${dat_unpacked_basenames["$candidate_base"]+_} ]]; then
                echo "  -> Skipping $(basename "$candidate") (already recorded in dat_files_unpacked.csv)."
                continue
            fi
            filtered_candidates+=("$candidate")
        done
        candidate_files=("${filtered_candidates[@]}")
    fi

    if (( ${#candidate_files[@]} == 0 )); then
        if [[ "$relocated_candidates" == true ]]; then
            echo "All HLD files in the queue belonged to other stations and were reassigned."
        fi
        echo "No HLD files available in UNPROCESSED or PROCESSING."
        return 1
    fi

    local selected_file
    if [ "$random_file" = true ]; then
        selected_file="${candidate_files[RANDOM % ${#candidate_files[@]}]}"
    elif $newest_mode; then
        local best_candidate=""
        local best_key=""
        local path base order_key
        for path in "${candidate_files[@]}"; do
            [[ -e "$path" ]] || continue
            base=$(strip_suffix "$(basename "$path")")
            [[ -z "$base" ]] && base="$(basename "$path")"
            order_key=$(order_key_after_prefix "$base")
            if [[ -z "$best_candidate" || "$order_key" > "$best_key" ]]; then
                best_candidate="$path"
                best_key="$order_key"
            fi
        done
        if [[ -z "$best_candidate" ]]; then
            best_candidate="${candidate_files[0]}"
            best_key=""
        fi
        selected_file="$best_candidate"
        echo "--newest flag active; newest pending basename selected: $(basename "$selected_file")"
    else
        local -a sorted=()
        IFS=$'\n' sorted=($(printf '%s\n' "${candidate_files[@]}" | sort -u))
        unset IFS
        selected_file="${sorted[0]}"
    fi

    echo "Selected HLD file: $(basename "$selected_file") [source: $source_stage]"

    local selected_base
    selected_base=$(basename "${selected_file%.hld}")
    # append_row_if_missing "$selected_base" "" "$csv_timestamp" ""

    # awk -F',' -v OFS=',' -v key="$selected_base" -v ts="$csv_timestamp" '
    #     NR == 1 { print; next }
    #     {
    #         if ($1 == key && $4 == "") {
    #             $4 = ts
    #         }
    #         print
    #     }
    # ' "$csv_path" > "${csv_path}.tmp" && mv "${csv_path}.tmp" "$csv_path"

    local filename name_no_ext prefix ss ss_val ss_new new_filename
    filename=$(basename "$selected_file")
    local processing_path="$processing_directory/$filename"

    if [[ -e "$processing_path" ]]; then
        echo "Warning: $filename already existed in PROCESSING; moving old copy to ERROR."
        mv "$processing_path" "$error_directory/$filename"
    fi

    if ! mv "$selected_file" "$processing_path"; then
        echo "Warning: failed to move $filename into PROCESSING." >&2
        return 1
    fi

    if ! cp "$processing_path" "$hld_input_directory/$filename"; then
        echo "Warning: failed to copy $filename into unpacker input directory." >&2
        return 1
    fi

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
    export RPCSYSTEM="mingo${station_code}"
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
        local fname target_station_code destination_dir
        fname=$(basename "$file")
        touch "$file"
        target_station_code=$(extract_station_code_from_name "$fname" 2>/dev/null || true)
        destination_dir="$dest_directory"
        if [[ -n "$target_station_code" && "$target_station_code" != "$station_code" ]]; then
            destination_dir="${STATIONS_BASE}/MINGO${target_station_code}/STAGE_0_to_1"
            echo "--> Redirecting $fname to ${destination_dir} (belongs to station ${target_station_code})."
        fi
        if [[ -e "$destination_dir/$fname" ]]; then
            if [[ "$destination_dir" == "$dest_directory" ]]; then
                echo "Skipping $fname — already present in STAGE_0_to_1."
            else
                echo "Skipping $fname — already present in ${destination_dir}."
            fi
            continue
        fi
        mkdir -p "$destination_dir"
        if mv "$file" "$destination_dir/$fname"; then
            touch "$destination_dir/$fname"
            if [[ "$destination_dir" == "$dest_directory" ]]; then
                echo "Moved $fname to STAGE_0_to_1."
            else
                echo "Moved $fname to ${destination_dir}."
            fi
        else
            echo "Warning: failed to move $fname into ${destination_dir}." >&2
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
            # append_row_if_missing "$dat_base" "" "$csv_timestamp" "$csv_timestamp"
        done < "$stage0_after"
    fi

    # if [[ -s "$stage0_after" || -s "$stage0_new" ]]; then
    #     awk -F',' -v OFS=',' -v all="$stage0_after" -v new="$stage0_new" -v ts="$csv_timestamp" '
    #         function canonical(name) {
    #             gsub(/\r/, "", name)
    #             sub(/\.hld\.tar\.gz$/, "", name)
    #             sub(/\.hld-tar-gz$/, "", name)
    #             sub(/\.tar\.gz$/, "", name)
    #             sub(/\.hld$/, "", name)
    #             sub(/\.dat$/, "", name)
    #             return name
    #         }
    #         BEGIN {
    #             if (all != "") {
    #                 while ((getline line < all) > 0) {
    #                     if (line == "") { continue }
    #                     roots_all[canonical(line)] = 1
    #                 }
    #                 close(all)
    #             }
    #             if (new != "") {
    #                 while ((getline line < new) > 0) {
    #                     if (line == "") { continue }
    #                     roots_new[canonical(line)] = 1
    #                 }
    #                 close(new)
    #             }
    #         }
    #         NR == 1 { print; next }
    #         {
    #             base = canonical($1)
    #             if (base == "") {
    #                 print
    #                 next
    #             }
    #             if (base in roots_new) {
    #                 if ($4 == "") {
    #                     $4 = ts
    #                 }
    #                 $5 = ts
    #             } else if (base in roots_all) {
    #                 if ($4 == "") {
    #                     $4 = ts
    #                 }
    #                 if ($5 == "") {
    #                     $5 = ts
    #                 }
    #             }
    #             print
    #         }
    #     ' "$csv_path" > "${csv_path}.tmp" && mv "${csv_path}.tmp" "$csv_path"
    # fi

    rm -f "$stage0_before" "$stage0_after" "$stage0_new"

    if [ -d "$hld_input_directory" ]; then
        echo "Archiving processed HLD files into COMPLETED..."
        shopt -s nullglob
        for processed in "$hld_input_directory"/*.hld*; do
            [[ -e "$processed" ]] || continue
            local name
            name=$(basename "$processed")
            if [[ -e "$completed_uncompressed/$name" ]]; then
                rm -f "$completed_uncompressed/$name"
            fi
            mv "$processed" "$completed_uncompressed/$name"
        done
        shopt -u nullglob
    fi

    shopt -s nullglob
    for processed in "$processing_directory"/*.hld "$processing_directory"/*.HLD; do
        [[ -e "$processed" ]] || continue
        local name
        name=$(basename "$processed")
        if [[ -e "$completed_uncompressed/$name" ]]; then
            rm -f "$completed_uncompressed/$name"
        fi
        mv "$processed" "$completed_uncompressed/$name"
    done
    shopt -u nullglob

    if [ -d "$asci_output_directory" ]; then
        echo "Moving existing dat files to removed directory..."
        mkdir -p "$asci_output_directory/removed"
        mv "$asci_output_directory"/*.dat* "$asci_output_directory/removed/" 2>/dev/null
    fi

    local BASE_ROOT SUBDIRS
    BASE_ROOT="$STATIONS_BASE"
    SUBDIRS=(
        "STAGE_0/REPROCESSING/STEP_1/OUTPUT_FILES/COMPRESSED_HLDS"
        "STAGE_0/REPROCESSING/STEP_1/OUTPUT_FILES/UNCOMPRESSED_HLDS"
        "STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/UNPROCESSED"
        "STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/COMPLETED"
        "STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/PROCESSING"
        "STAGE_0/REPROCESSING/STEP_2/INPUT_FILES/ERROR"
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
                    echo "--> Moving $filename from MINGO${station_id}/${subdir} to MINGO${file_station}/${subdir}"
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
        local dat_name dat_base
        for dat_name in "${new_dat_files[@]}"; do
            dat_base=$(strip_suffix "$dat_name")
            [[ -z "$dat_base" ]] && dat_base="$dat_name"
            printf '%s,%s,%s\n' "$dat_base" "$script_start_time" "$script_duration" >> "$dat_unpacked_csv"
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
