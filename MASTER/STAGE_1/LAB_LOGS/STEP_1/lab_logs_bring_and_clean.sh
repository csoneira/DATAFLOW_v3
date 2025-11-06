#!/bin/bash

# ----------------------------------------------
# Only this changes between mingos and computers
if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
lab_logs_bring_and_clean.sh
Pulls lab log files from a station, cleans new ones, and stages them for STEP_2.

Usage:
  lab_logs_bring_and_clean.sh [-a|--all] <station>

Options:
  -a, --all    Re-clean files already marked as completed.
  -h, --help    Show this help message and exit.

Pass the station number (1-4). The script fetches raw logs via rsync, runs the
cleaner, and keeps a record of processed files so they are not re-cleaned unless
the --all flag is provided.
EOF
  exit 0
fi

include_completed=false
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--all)
      include_completed=true
      shift
      ;;
    -h|--help)
      cat <<'EOF'
lab_logs_bring_and_clean.sh
Pulls lab log files from a station, cleans new ones, and stages them for STEP_2.

Usage:
  lab_logs_bring_and_clean.sh [-a|--all] <station>

Options:
  -a, --all    Re-clean files already marked as completed.
  -h, --help   Show this help message and exit.
EOF
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ ${#args[@]} -lt 1 ]]; then
  echo "Error: No station provided."
  echo "Usage: $0 [-a|--all] <station>"
  exit 1
fi

station=${args[0]}
echo "Station: $station"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
  MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
# STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
# STATUS_TIMESTAMP=""
# STATUS_CSV=""

# ----------------------------------------------

# Additional paths
mingo_direction="mingo0$station"

python_script_path="$HOME/DATAFLOW_v3/MASTER/STAGE_1/LAB_LOGS/STEP_2/log_aggregate_and_join.py"

LAB_LOGS_ROOT="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/STAGE_1/LAB_LOGS"
STEP1_ROOT="${LAB_LOGS_ROOT}/STEP_1"
STEP1_INPUT="${STEP1_ROOT}/INPUT_FILES"
STEP1_INPUT_UNPROCESSED="${STEP1_INPUT}/UNPROCESSED"
STEP1_INPUT_COMPLETED="${STEP1_INPUT}/COMPLETED"
STEP1_OUTPUT="${STEP1_ROOT}/OUTPUT_FILES"

mkdir -p "${STEP1_ROOT}" "${STEP1_INPUT_UNPROCESSED}" "${STEP1_INPUT_COMPLETED}" "${STEP1_OUTPUT}"

# STATUS_CSV="${STEP1_ROOT}/log_bring_and_clean_status.csv"
# if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
#   echo "Warning: unable to record status in $STATUS_CSV" >&2
#   STATUS_TIMESTAMP=""
# fi

# finish() {
#   local exit_code="$1"
#   if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
#     python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
#   fi
# }

# trap 'finish $?' EXIT

local_destination="${STEP1_INPUT_UNPROCESSED}"
OUTPUT_DIR="${STEP1_OUTPUT}"


echo '--------------------------- bash script starts ---------------------------'

# Sync data from the remote server
EXCLUDES=(
    "--exclude=/clean_*"
    "--exclude=/done/clean_*"
    "--exclude=/done/merged_*"
)

declare -A SEEN_RELATIVE_PATHS

if [[ -d "${STEP1_INPUT_UNPROCESSED}" ]]; then
    while IFS= read -r -d '' file; do
        rel="${file#${STEP1_INPUT_UNPROCESSED}/}"
        [[ -n "$rel" ]] || continue
        SEEN_RELATIVE_PATHS["$rel"]=1
    done < <(find "${STEP1_INPUT_UNPROCESSED}" -type f -print0 2>/dev/null || true)
fi

if [[ -d "${STEP1_INPUT_COMPLETED}" ]]; then
    while IFS= read -r -d '' file; do
        rel="${file#${STEP1_INPUT_COMPLETED}/}"
        [[ -n "$rel" ]] || continue
        SEEN_RELATIVE_PATHS["$rel"]=1
    done < <(find "${STEP1_INPUT_COMPLETED}" -type f -print0 2>/dev/null || true)
fi

for rel_path in "${!SEEN_RELATIVE_PATHS[@]}"; do
    EXCLUDES+=("--exclude=/${rel_path}")
done

rsync -avz "${EXCLUDES[@]}" \
    "$mingo_direction:/home/rpcuser/logs/" "${local_destination}/"

echo 'Received data from remote computer'

declare -A COLUMN_COUNTS
COLUMN_COUNTS["Flow0_"]=5
COLUMN_COUNTS["hv0_"]=22
COLUMN_COUNTS["Odroid_"]=4
COLUMN_COUNTS["rates_"]=12
COLUMN_COUNTS["sensors_bus0_"]=8
COLUMN_COUNTS["sensors_bus1_"]=8

move_to_completed() {
    local file_path=$1
    if [[ "${file_path}" != "${STEP1_INPUT_UNPROCESSED}/"* ]]; then
        return
    fi
    local relative_path="${file_path#${STEP1_INPUT_UNPROCESSED}/}"
    local destination="${STEP1_INPUT_COMPLETED}/${relative_path}"
    mkdir -p "$(dirname "$destination")"
    if [[ -e "$destination" ]]; then
        rm -f "$destination"
    fi
    mv "$file_path" "$destination"
}

process_file() {
    local file=$1
    local move_after=$2
    local filename=$(basename "$file")
    local output_file="$OUTPUT_DIR/$filename"

    # Check if the file needs to be processed
    if [[ -f "$output_file" ]]; then
        # Compare modification timestamps
        local source_mtime=$(stat -c %Y "$file")
        local processed_mtime=$(stat -c %Y "$output_file")
        
        #source_mtime_save=$(stat -c %Y "$file")
        #processed_mtime_save=$(stat -c %Y "$output_file")
        
        if [[ $source_mtime -le $processed_mtime ]]; then
            #echo "File $filename is already processed and up-to-date. Skipping."
            if [[ "${move_after}" == "1" ]]; then
                move_to_completed "$file"
            fi
            return
        fi
    fi

    # Process the file
    for prefix in "${!COLUMN_COUNTS[@]}"; do
        if [[ $filename == $prefix* ]]; then
            local column_count=${COLUMN_COUNTS[$prefix]}
            awk -v col_count=$column_count -v output_file="$output_file" -v file="$file" '
		    BEGIN { OFS=" "; invalid_count=0; valid_count=0 }
		    {
			  gsub(/T/, " ", $1);      # Replace T with space
			  gsub(/[,;]/, " ");       # Replace commas and semicolons with space
			  gsub(/  +/, " ");        # Replace multiple spaces with a single space
			  if (NF >= col_count) {   # Keep rows with at least the expected number of fields
				valid_count++;
				print $0 > output_file
			  } else {
				invalid_count++;
			  }
		    }
		    END {
			  if (invalid_count > 0) {   # Only print the message if invalid rows were found
				print "Processed: " valid_count " valid rows, " invalid_count " discarded rows." > "/dev/stderr"
				print "Processed " file " into " output_file > "/dev/stderr"
			  }
		    }
	' "$file"
            
            #echo "Processed $file into $output_file."
	#echo $source_mtime_save
	#echo $processed_mtime_save
	#echo '-------------------'
            if [[ "${move_after}" == "1" && "${file}" == "${STEP1_INPUT_UNPROCESSED}/"* ]]; then
                move_to_completed "$file"
            fi
            return
        fi
    done

    #echo "Unknown file prefix: $filename. Skipping $file."
}

process_directory() {
    local dir=$1
    local move_after=$2
    [[ -d "$dir" ]] || return

    while IFS= read -r -d '' file; do
        process_file "$file" "$move_after"
    done < <(find "$dir" -type f -print0)
}

process_directory "${STEP1_INPUT_UNPROCESSED}" "1"

if [[ "${include_completed}" == true ]]; then
    process_directory "${STEP1_INPUT_COMPLETED}" "0"
fi

# Clean up any empty directories left in UNPROCESSED after moving files
cleanup_empty_dirs() {
    local base_dir=$1
    [[ -d "$base_dir" ]] || return
    find "$base_dir" -type d -empty -delete 2>/dev/null || true
}

cleanup_empty_dirs "${STEP1_INPUT_UNPROCESSED}"
echo "Files cleaned into $OUTPUT_DIR"

echo '------------------------------------------------------'
echo "log_bring_and_clean.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
