#!/bin/bash

# log_file="${LOG_FILE:-~/cron_logs/bring_and_analyze_events_${station}.log}"
# mkdir -p "$(dirname "$log_file")"

# Station specific -----------------------------
if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
bring_data_and_config_files.sh
Synchronises STAGE_0_to_1 event data and configuration files from stations into STAGE_0_to_1.

Usage:
  bring_data_and_config_files.sh <station>

Options:
  -h, --help    Show this help message and exit.

Provide the station identifier (1-4). The script prevents concurrent runs for
the same station and updates status tracking CSVs as it pulls files.
EOF
  exit 0
fi

if [ -z "$1" ]; then
  echo "Error: No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

# echo '------------------------------------------------------'
# echo "bring_and_analyze_events.sh started on: $(date '+%Y-%m-%d %H:%M:%S')"

station=$1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
  MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""
STATUS_CSV=""

# If $1 is not 1, 2, 3, 4, exit
if [[ ! "$station" =~ ^[1-4]$ ]]; then
  echo "Error: Invalid station number. Please provide a number between 1 and 4."
  exit 1
fi

# --------------------------------------------------------------------------------------------
# Prevent the script from running multiple instances -----------------------------------------
# --------------------------------------------------------------------------------------------

# Variables
script_name=$(basename "$0")
script_args="$*"
current_pid=$$

# # Get all running instances of the script *with the same argument*, but exclude the current process
# for pid in $(ps -eo pid,cmd | grep "[b]ash .*/$script_name" | grep -v "bin/bash -c" | awk '{print $1}'); do
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

# Get all running instances of the script *with the same argument*, but exclude the current process
for pid in $(ps -eo pid,cmd | grep "[b]ash .*/$script_name" | grep -v "bin/bash -c" | awk '{print $1}'); do
    if [[ "$pid" != "$current_pid" ]]; then
        cmdline=$(ps -p "$pid" -o args=)
        # echo "$(date) - Found running process: PID $pid - $cmdline"
        if [[ "$cmdline" == *"$script_name"* ]]; then
            echo "------------------------------------------------------"
            echo "$(date): The script $script_name is already running (PID: $pid). Exiting."
            echo "------------------------------------------------------"
            exit 1
        fi
    fi
done

# If no duplicate process is found, continue
echo "$(date) - No running instance found. Proceeding..."


# If no duplicate process is found, continue
echo "------------------------------------------------------"
echo "bring_data_and_config_files.sh started on: $(date)"
echo "Station: $script_args"
echo "Running the script..."
echo "------------------------------------------------------"
# --------------------------------------------------------------------------------------------


dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# Additional paths
mingo_direction="mingo0$station"

station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station"
stage0_directory="$station_directory/STAGE_0/NEW_FILES"
stage0_to_1_directory="$station_directory/STAGE_0_to_1"
metadata_directory="$stage0_directory/METADATA"
raw_directory="$stage0_to_1_directory"

mkdir -p "$station_directory" "$stage0_directory" "$stage0_to_1_directory" "$metadata_directory"

# STATUS_CSV="$metadata_directory/bring_data_and_config_files_status.csv"
# if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
#   echo "Warning: unable to record status in $STATUS_CSV" >&2
#   STATUS_TIMESTAMP=""
# fi

log_csv="$metadata_directory/raw_files_brought.csv"
log_csv_header="filename,bring_timestamp"
run_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

before_list=""
after_list=""
new_list=""
rsync_file_list=""

cleanup() {
  for tmp in "$before_list" "$after_list" "$new_list" "$rsync_file_list"; do
    [[ -n "$tmp" ]] && rm -f "$tmp"
  done
}

# finish() {
#   local exit_code="$1"
#   cleanup
#   if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
#     python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
#   fi
# }

trap 'finish $?' EXIT

ensure_log_csv() {
  if [[ ! -f "$log_csv" ]]; then
    printf '%s\n' "$log_csv_header" > "$log_csv"
  elif [[ ! -s "$log_csv" ]]; then
    printf '%s\n' "$log_csv_header" > "$log_csv"
  else
    local current_header
    current_header=$(head -n1 "$log_csv")
    if [[ "$current_header" != "$log_csv_header" ]]; then
      local upgrade_tmp
      upgrade_tmp=$(mktemp)
      printf '%s\n' "$log_csv_header" > "$upgrade_tmp"
      tail -n +2 "$log_csv" >> "$upgrade_tmp"
      mv "$upgrade_tmp" "$log_csv"
    fi
  fi
}

declare -A logged_files=()
load_logged_files() {
  if [[ ! -s "$log_csv" ]]; then
    return
  fi
  while IFS=',' read -r filename _; do
    filename=${filename//$'\r'/}
    [[ -z "$filename" || "$filename" == "filename" ]] && continue
    logged_files["$filename"]=1
  done < "$log_csv"
}

register_brought_file() {
  local filename="$1"
  [[ -z "$filename" ]] && return
  if [[ -n ${logged_files["$filename"]+_} ]]; then
    return
  fi
  printf '%s,%s\n' "$filename" "$run_timestamp" >> "$log_csv"
  logged_files["$filename"]=1
}

ensure_log_csv
load_logged_files

# Fetch all data
echo "Fetching data from $mingo_direction to $raw_directory..."
echo '------------------------------------------------------'

before_list=$(mktemp)
# Strictly match files ending in .dat (not .dat*)
find "$raw_directory" -maxdepth 1 -type f \
  -regextype posix-extended -regex '.*/[^/]+\.dat$' \
  -printf '%f\n' | sort -u > "$before_list"

echo "Files currently available on $mingo_direction:"
remote_list_cmd=$(printf 'cd %q && find . -maxdepth 1 -type f -regextype posix-extended -regex %s -printf %s | sort' \
  "$dat_files_directory" "'.*/[^/]+\\.dat$'" "'%P\n'")
ssh "$mingo_direction" "$remote_list_cmd" 2>/dev/null || echo "No .dat files found or listing unavailable."

rsync_file_list=$(mktemp)
remote_find_cmd=$(printf 'cd %q && find . -maxdepth 1 -type f -regextype posix-extended -regex %s -printf %s' \
  "$dat_files_directory" "'.*/[^/]+\\.dat$'" "'%P\0'")
if ssh "$mingo_direction" "$remote_find_cmd" > "$rsync_file_list"; then
  if [[ -s "$rsync_file_list" ]]; then
    rsync -avz --ignore-existing \
      --files-from="$rsync_file_list" \
      --from0 \
      "$mingo_direction:$dat_files_directory/" \
      "$raw_directory/" || {
      echo "Warning: rsync encountered an error while fetching data." >&2
    }
  else
    echo "No .dat files found to transfer."
  fi
else
  echo "Warning: unable to build .dat file list from remote host." >&2
fi

after_list=$(mktemp)
# Strictly match files ending in .dat (not .dat*)
find "$raw_directory" -maxdepth 1 -type f \
  -regextype posix-extended -regex '.*/[^/]+\.dat$' \
  -printf '%f\n' | sort -u > "$after_list"

new_list=$(mktemp)
comm -13 "$before_list" "$after_list" > "$new_list"

if [[ -s "$new_list" ]]; then
  while IFS= read -r dat_entry; do
    dat_entry=${dat_entry//$'\r'/}
    [[ -z "$dat_entry" ]] && continue
    register_brought_file "$dat_entry"
  done < "$new_list"
  new_count=$(grep -c '' "$new_list")
  echo "Registered $new_count new file(s) in $log_csv."
else
  echo "No new files transferred."
fi


echo '------------------------------------------------------'
echo '------------------------------------------------------'

# Bring the input files from the logbook
echo "Bringing the input files from the logbook..."

# # Google Sheet ID (common for all stations)
# SHEET_ID="1ato36QkIXCxFkDT_LtAaLjPP7pvLcor-xZAP4fy00l0"

# # Mapping of station numbers to their respective GIDs
# declare -A STATION_GID_MAP
# STATION_GID_MAP[1]="1331842924"
# STATION_GID_MAP[2]="600987525"
# STATION_GID_MAP[3]="376764978"
# STATION_GID_MAP[4]="1268265225"

# # Get the corresponding GID
# GID=${STATION_GID_MAP[$station]}

# ---------------------------------------------
# Read IDs from YAML (requires PyYAML in Python)
# ---------------------------------------------
CONFIG_FILE="$HOME/DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml"
SHEET_ID=$(yq -r '.logbook.sheet_id' "$CONFIG_FILE")
GID=$(yq -r ".logbook.gid_by_station.\"$station\"" "$CONFIG_FILE")

# Basic validation
if [[ -z "$SHEET_ID" || -z "$GID" ]]; then
  echo "Error: Could not read SHEET_ID or GID for station $station from $CONFIG_FILE"
  exit 1
fi

echo $SHEET_ID
echo $GID

# Define output file path
OUTPUT_FILE="$station_directory/input_file_mingo0${station}.csv"

# Download the file using wget with minimal console output
echo "Downloading logbook for Station $station..."
wget -q --show-progress --no-check-certificate \
     "https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv&id=${SHEET_ID}&gid=${GID}" \
     -O "${OUTPUT_FILE}"


# Check if download was successful
if [[ $? -eq 0 ]]; then
    echo "Download completed. Data saved at ${OUTPUT_FILE}."
else
    echo "Error: Download failed. Continuing execution..."
fi


echo '------------------------------------------------------'
echo "bring_data_and_config_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
