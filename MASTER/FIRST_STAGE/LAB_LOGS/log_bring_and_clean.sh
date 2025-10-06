#!/bin/bash

# ----------------------------------------------
# Only this changes between mingos and computers
if [ -z "$1" ]; then
  echo "Error: No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

station=$1
echo "Station: $station"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
  MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""
STATUS_CSV=""

# ----------------------------------------------

# Additional paths
mingo_direction="mingo0$station"

python_script_path="$HOME/DATAFLOW_v3/MASTER/FIRST_STAGE/LAB_LOGS/log_aggregate_and_join.py"

base_working_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/FIRST_STAGE/LAB_LOGS"

mkdir -p "${base_working_directory}"
STATUS_CSV="${base_working_directory}/log_bring_and_clean_status.csv"
if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
  echo "Warning: unable to record status in $STATUS_CSV" >&2
  STATUS_TIMESTAMP=""
fi

finish() {
  local exit_code="$1"
  if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
    python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
  fi
}

trap 'finish $?' EXIT

local_destination="${base_working_directory}/RAW_LOGS"
DONE_DIR="${local_destination}/done"
OUTPUT_DIR="${base_working_directory}/CLEAN_LOGS"

mkdir -p "${local_destination}" "${DONE_DIR}" "${OUTPUT_DIR}"


echo '--------------------------- bash script starts ---------------------------'

# Sync data from the remote server
rsync -avz --delete \
    --exclude='/clean_*' \
    --exclude='/done/clean_*' \
    --exclude='/done/merged_*' \
    $mingo_direction:/home/rpcuser/logs/ $local_destination

echo 'Received data from remote computer'

mkdir -p "$OUTPUT_DIR"

declare -A COLUMN_COUNTS
COLUMN_COUNTS["Flow0_"]=5
COLUMN_COUNTS["hv0_"]=22
COLUMN_COUNTS["Odroid_"]=4
COLUMN_COUNTS["rates_"]=12
COLUMN_COUNTS["sensors_bus0_"]=8
COLUMN_COUNTS["sensors_bus1_"]=8

process_file() {
    local file=$1
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
            return
        fi
    done

    #echo "Unknown file prefix: $filename. Skipping $file."
}

process_directory() {
    local dir=$1
    for file in "$dir"/*; do
        if [[ -f "$file" ]]; then
            process_file "$file"
        fi
    done
}

process_directory "$local_destination"
process_directory "$DONE_DIR"

echo "Files cleaned into $OUTPUT_DIR"

# Call the python joiner execution
python3 -u $python_script_path "$station"

echo '------------------------------------------------------'
echo "log_bring_and_clean.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
