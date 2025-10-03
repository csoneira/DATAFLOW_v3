#!/bin/bash

# log_file="${LOG_FILE:-~/cron_logs/bring_and_analyze_events_${station}.log}"
# mkdir -p "$(dirname "$log_file")"

# Station specific -----------------------------
if [ -z "$1" ]; then
  echo "Error: No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

# echo '------------------------------------------------------'
# echo "bring_and_analyze_events.sh started on: $(date '+%Y-%m-%d %H:%M:%S')"

station=$1

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

# Get all running instances of the script *with the same argument*, but exclude the current process
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


# If no duplicate process is found, continue
echo "------------------------------------------------------"
echo "bring_data_and_config_files.sh started on: $(date)"
echo "Station: $script_args"
echo "Running the script..."
echo "------------------------------------------------------"
# --------------------------------------------------------------------------------------------


dat_files_directory="/media/externalDisk/gate/system/devices/TRB3/data/daqData/asci"

# Define base working directory
station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station"
base_working_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station/FIRST_STAGE/EVENT_DATA"
csv_path="$station_directory/database_status_${station}.csv"
csv_header="basename,start_date,hld_remote_add_date,hld_local_add_date,dat_add_date,list_ev_name,list_ev_add_date,acc_name,acc_add_date,merge_add_date"
csv_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

# Define directories
local_destination="$base_working_directory/RAW"
storage_directory="$base_working_directory/RAW_TO_LIST"

# Additional paths
mingo_direction="mingo0$station"

exclude_list_file="$base_working_directory/tmp/exclude_list.txt"
csv_exclude_file=""
before_list=""
after_list=""
new_list=""

cleanup() {
  rm -f "$exclude_list_file" "$csv_exclude_file" "$before_list" "$after_list" "$new_list"
}

trap cleanup EXIT

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

strip_suffix() {
  local name="$1"
  name=${name%.dat}
  name=${name%.hld.tar.gz}
  name=${name%.hld-tar-gz}
  name=${name%.hld}
  printf '%s' "$name"
}

compute_start_date() {
  local base="$1"
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

ensure_csv

declare -A csv_rows=()
if [[ -s "$csv_path" ]]; then
  while IFS=',' read -r base _; do
    [[ -z "$base" || "$base" == "basename" ]] && continue
    base=${base//$'\r'/}
    csv_rows["$base"]=1
  done < "$csv_path"
fi

append_row_if_missing() {
  local base="$1"
  local dat_date="$2"
  [[ -z "$base" ]] && return
  if [[ -n ${csv_rows["$base"]+_} ]]; then
    return
  fi
  local start_value
  start_value=$(compute_start_date "$base")
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$base" "$start_value" "" "" "$dat_date" "" "" "" "" "" >> "$csv_path"
  csv_rows["$base"]=1
}

csv_exclude_file=$(mktemp)
if [[ -s "$csv_path" ]]; then
  awk -F',' 'NR>1 && $5 != "" {gsub(/\r/,"",$1); if ($1!="") print $1".dat"}' "$csv_path" >> "$csv_exclude_file"
fi

# Create necessary directories
mkdir -p "$station_directory"
mkdir -p "$base_working_directory/tmp"
mkdir -p "$local_destination"
mkdir -p "$storage_directory"

printf '' > "$exclude_list_file"
if [[ -s "$csv_exclude_file" ]]; then
  cat "$csv_exclude_file" >> "$exclude_list_file"
fi
find "$storage_directory" -type f -name '*.dat' -printf '%f\n' >> "$exclude_list_file"
find "$local_destination" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' >> "$exclude_list_file"
sort -u "$exclude_list_file" -o "$exclude_list_file"


# Fetch all data
echo "Fetching data from $mingo_direction to $local_destination, excluding already processed files..."

echo '------------------------------------------------------'

before_list=$(mktemp)
find "$local_destination" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' | sort -u > "$before_list"

echo "Files currently available on $mingo_direction:"
ssh "$mingo_direction" "ls -lh $dat_files_directory"/*.dat

rsync -avz --exclude-from="$exclude_list_file" "$mingo_direction:$dat_files_directory"/*.dat "$local_destination"

after_list=$(mktemp)
find "$local_destination" -maxdepth 1 -type f -name '*.dat' -printf '%f\n' | sort -u > "$after_list"

new_list=$(mktemp)
comm -13 "$before_list" "$after_list" > "$new_list"

if [[ -s "$new_list" ]]; then
  while IFS= read -r dat_entry; do
    dat_entry=${dat_entry//$'\r'/}
    [[ -z "$dat_entry" ]] && continue
    dat_base=$(strip_suffix "$dat_entry")
    append_row_if_missing "$dat_base" "$csv_timestamp"
  done < "$new_list"

  awk -F',' -v OFS=',' -v list="$new_list" -v ts="$csv_timestamp" '
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
      while ((getline line < list) > 0) {
        line = canonical(line)
        if (line != "") {
          seen[line] = 1
        }
      }
      close(list)
    }
    NR == 1 { print; next }
    {
      base = canonical($1)
      if (base in seen) {
        $5 = ts
      }
      print
    }
  ' "$csv_path" > "${csv_path}.tmp" && mv "${csv_path}.tmp" "$csv_path"
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
CONFIG_FILE="$HOME/DATAFLOW_v3/MASTER/config.yaml"
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
