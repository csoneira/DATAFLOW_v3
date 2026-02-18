#!/bin/bash

# log_file="${LOG_FILE:-~/cron_logs/bring_and_analyze_events_${station}.log}"
# mkdir -p "$(dirname "$log_file")"

log_ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_info() {
  printf '[%s] [NEW_FILES] %s\n' "$(log_ts)" "$*"
}

log_warn() {
  printf '[%s] [NEW_FILES] [WARN] %s\n' "$(log_ts)" "$*" >&2
}

log_error() {
  printf '[%s] [NEW_FILES] [ERROR] %s\n' "$(log_ts)" "$*" >&2
}

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
  log_error "No station provided."
  echo "Usage: $0 <station>"
  exit 1
fi

# echo '------------------------------------------------------'
# echo "bring_and_analyze_events.sh started on: $(date '+%Y-%m-%d %H:%M:%S')"

station=$1

# Ensure snap-installed utilities such as yq are available even under cron
export PATH="/snap/bin:$PATH"

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
  log_error "Invalid station number. Please provide a number between 1 and 4."
  exit 1
fi

NEW_FILES_CONFIG_FILE="$HOME/DATAFLOW_v3/MASTER/CONFIG_FILES/STAGE_0/NEW_FILES/config_new_files.yaml"

is_new_files_station_enabled() {
  local station_id="$1"
  local cfg_path="$NEW_FILES_CONFIG_FILE"
  [[ -f "$cfg_path" ]] || return 0

  local decision
  decision=$(python3 - "$cfg_path" "$station_id" <<'PY' || true
import sys

cfg_path, station_raw = sys.argv[1:3]

try:
  import yaml
except ImportError:
  print("1")
  raise SystemExit(0)

def parse_int(value):
  try:
    return int(str(value).strip())
  except Exception:
    return None

def normalize_stations(value):
  if value is None:
    return None
  if isinstance(value, str):
    text = value.strip().lower()
    if not text:
      return []
    if text == "all":
      return "all"
    out = []
    for token in text.replace(";", ",").split(","):
      token = token.strip()
      if not token:
        continue
      try:
        out.append(int(token))
      except Exception:
        continue
    return out
  if isinstance(value, (list, tuple, set)):
    out = []
    for item in value:
      if isinstance(item, str):
        parsed = normalize_stations(item)
        if parsed == "all":
          return "all"
        if isinstance(parsed, list):
          out.extend(parsed)
          continue
      try:
        out.append(int(item))
      except Exception:
        continue
    return out
  if isinstance(value, (int, float)):
    try:
      return [int(value)]
    except Exception:
      return []
  return []

station_id = parse_int(station_raw)
if station_id is None:
  print("1")
  raise SystemExit(0)

try:
  with open(cfg_path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
except Exception:
  data = {}

node = data.get("new_files_run_matrix")
if not isinstance(node, dict):
  print("1")
  raise SystemExit(0)

if not bool(node.get("enabled", False)):
  print("1")
  raise SystemExit(0)

mode = str(node.get("mode", "whitelist")).strip().lower()
stations_node = node.get("stations")
if isinstance(stations_node, dict):
  station_list = normalize_stations(stations_node.get(str(station_id)))
  if station_list is None:
    station_list = normalize_stations(stations_node.get("0"))
else:
  station_list = normalize_stations(stations_node)

default_stations = normalize_stations(node.get("default_stations", []))

if mode == "blacklist":
  disabled = False
  if station_list is None:
    if default_stations == "all":
      disabled = True
    elif isinstance(default_stations, list) and station_id in default_stations:
      disabled = True
  elif station_list == "all":
    disabled = True
  elif isinstance(station_list, list) and station_id in station_list:
    disabled = True
  print("0" if disabled else "1")
  raise SystemExit(0)

allowed = False
if station_list is None:
  if default_stations == "all":
    allowed = True
  elif isinstance(default_stations, list) and station_id in default_stations:
    allowed = True
elif station_list == "all":
  allowed = True
elif isinstance(station_list, list) and station_id in station_list:
  allowed = True

print("1" if allowed else "0")
PY
  )

  [[ "$decision" != "0" ]]
}

if ! is_new_files_station_enabled "$station"; then
  log_info "Skipping station ${station}: disabled by new_files_run_matrix."
  exit 0
fi

date_range_filter_enabled=false
declare -a date_ranges_start_epochs=()
declare -a date_ranges_end_epochs=()
declare -a date_ranges_display_pairs=()

load_new_files_date_ranges() {
  local result
  if ! result=$(python3 - "$NEW_FILES_CONFIG_FILE" <<'PY'
import sys
import shlex
import datetime as dt

try:
    import yaml
except ImportError:
    yaml = None

cfg_path = sys.argv[1]

def load_yaml(path):
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def normalize_iso(text: str) -> str:
    text = text.strip()
    if text.endswith("Z") and "T" in text:
        text = text[:-1] + "+00:00"
    return text

def parse_epoch(value, is_end=False):
    if value in ("", None):
        return None
    text = normalize_iso(str(value))
    if not text:
        return None
    parsed = None
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            if fmt is None:
                parsed = dt.datetime.fromisoformat(text)
            else:
                parsed = dt.datetime.strptime(text, fmt)
            break
        except Exception:
            parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    if is_end and "T" not in text and " " not in text:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return int(parsed.timestamp())

ranges = []

def add_interval(start_raw, end_raw):
    start_epoch = parse_epoch(start_raw, is_end=False)
    end_epoch = parse_epoch(end_raw, is_end=True)
    if start_epoch is None and end_epoch is None:
        return
    start_label = "" if start_raw in (None, "") else str(start_raw)
    end_label = "" if end_raw in (None, "") else str(end_raw)
    ranges.append((start_epoch, end_epoch, start_label, end_label))

def collect(source):
    if not isinstance(source, dict):
        return
    legacy = source.get("date_range")
    if isinstance(legacy, dict):
        add_interval(legacy.get("start"), legacy.get("end"))
    range_list = source.get("date_ranges")
    if isinstance(range_list, list):
        for item in range_list:
            if isinstance(item, dict):
                add_interval(item.get("start"), item.get("end"))
    nested = source.get("new_files_date_selection")
    if isinstance(nested, dict):
        collect(nested)

collect(load_yaml(cfg_path))

deduped = []
seen = set()
for item in ranges:
    if item in seen:
        continue
    seen.add(item)
    deduped.append(item)

epochs = ";".join(
    f"{'' if start is None else start},{'' if end is None else end}"
    for start, end, _, _ in deduped
)
labels = ";".join(f"{a}|{b}" for _, _, a, b in deduped)
print(f"NEW_FILES_DATE_RANGES_EPOCHS={shlex.quote(epochs)}")
print(f"NEW_FILES_DATE_RANGES_LABELS={shlex.quote(labels)}")
PY
  ); then
    return 0
  fi

  eval "$result"
  local serialized="${NEW_FILES_DATE_RANGES_EPOCHS:-}"
  local labels="${NEW_FILES_DATE_RANGES_LABELS:-}"
  date_ranges_start_epochs=()
  date_ranges_end_epochs=()
  date_ranges_display_pairs=()
  date_range_filter_enabled=false

  if [[ -n "$serialized" ]]; then
    local -a epoch_chunks=()
    local -a label_chunks=()
    IFS=';' read -r -a epoch_chunks <<< "$serialized"
    IFS=';' read -r -a label_chunks <<< "$labels"
    local idx
    for idx in "${!epoch_chunks[@]}"; do
      local chunk="${epoch_chunks[idx]}"
      [[ -z "$chunk" || "$chunk" != *,* ]] && continue
      local start_epoch="${chunk%%,*}"
      local end_epoch="${chunk#*,}"
      [[ -n "$start_epoch" && ! "$start_epoch" =~ ^[0-9]+$ ]] && continue
      [[ -n "$end_epoch" && ! "$end_epoch" =~ ^[0-9]+$ ]] && continue
      [[ -z "$start_epoch" && -z "$end_epoch" ]] && continue
      date_ranges_start_epochs+=("$start_epoch")
      date_ranges_end_epochs+=("$end_epoch")

      local label_chunk="${label_chunks[idx]:-}"
      local start_label=""
      local end_label=""
      if [[ "$label_chunk" == *"|"* ]]; then
        start_label="${label_chunk%%|*}"
        end_label="${label_chunk#*|}"
      fi
      [[ -z "$start_label" ]] && start_label="-inf"
      [[ -z "$end_label" ]] && end_label="+inf"
      date_ranges_display_pairs+=("${start_label} to ${end_label}")
    done
  fi

  if (( ${#date_ranges_start_epochs[@]} > 0 )); then
    date_range_filter_enabled=true
    local range_display
    range_display="$(printf '%s; ' "${date_ranges_display_pairs[@]}")"
    range_display="${range_display%; }"
    log_info "Date range filtering enabled (${#date_ranges_start_epochs[@]} interval(s)): ${range_display}."
  fi
}

extract_file_epoch_from_name() {
  local file_path="$1"
  local name="${file_path##*/}"
  name="${name%.dat}"
  if [[ "$name" =~ ([0-9]{11})$ ]]; then
    local key="${BASH_REMATCH[1]}"
    local yy="${key:0:2}"
    local doy="${key:2:3}"
    local hh="${key:5:2}"
    local mm="${key:7:2}"
    local ss="${key:9:2}"
    if (( 10#$doy < 1 || 10#$doy > 366 )); then
      return 1
    fi
    local day_index=$((10#$doy - 1))
    date -u -d "20${yy}-01-01 +${day_index} days ${hh}:${mm}:${ss}" +%s 2>/dev/null
    return $?
  fi
  return 1
}

filename_matches_date_ranges() {
  local file_path="$1"
  if ! $date_range_filter_enabled; then
    return 0
  fi
  local file_epoch
  file_epoch=$(extract_file_epoch_from_name "$file_path") || return 1
  local idx
  for idx in "${!date_ranges_start_epochs[@]}"; do
    local start_epoch="${date_ranges_start_epochs[idx]}"
    local end_epoch="${date_ranges_end_epochs[idx]}"
    if [[ -n "$start_epoch" && "$file_epoch" -lt "$start_epoch" ]]; then
      continue
    fi
    if [[ -n "$end_epoch" && "$file_epoch" -gt "$end_epoch" ]]; then
      continue
    fi
    return 0
  done
  return 1
}

load_new_files_date_ranges

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
            log_warn "The script $script_name is already running (pid=$pid). Exiting."
            exit 1
        fi
    fi
done

# If no duplicate process is found, continue
log_info "No running instance found. Proceeding."


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
# Shared SSH/Rsync options so cron never hangs awaiting passwords
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)
RSYNC_RSH_CMD="ssh -o BatchMode=yes -o ConnectTimeout=15"

station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0$station"
config_file_directory="$HOME/DATAFLOW_v3/MASTER/CONFIG_FILES/STAGE_0/NEW_FILES/ONLINE_RUN_DICTIONARY/STATION_$station"
stage0_directory="$station_directory/STAGE_0/NEW_FILES"
stage0_to_1_directory="$station_directory/STAGE_0_to_1"
metadata_directory="$stage0_directory/METADATA"
raw_directory="$stage0_to_1_directory"
# Reject list shared with STATIONS/MINGO0*/STAGE_0/NEW_FILES/METADATA/raw_files_brought.csv
reject_list_csv="$metadata_directory/raw_files_brought.csv"

mkdir -p "$station_directory" "$stage0_directory" "$stage0_to_1_directory" "$metadata_directory" "$config_file_directory"

# STATUS_CSV="$metadata_directory/bring_data_and_config_files_status.csv"
# if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
#   echo "Warning: unable to record status in $STATUS_CSV" >&2
#   STATUS_TIMESTAMP=""
# fi

log_csv="$reject_list_csv"
log_csv_header="filename,bring_timestamp"
run_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"

before_list=""
after_list=""
new_list=""
rsync_file_list=""
filtered_rsync_file_list=""

cleanup() {
  for tmp in "$before_list" "$after_list" "$new_list" "$rsync_file_list" "$filtered_rsync_file_list"; do
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

# trap 'finish $?' EXIT

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
echo "Using reject list at $log_csv"

connection_ok=0
echo "Checking connectivity to $mingo_direction..."
if ssh "${SSH_OPTS[@]}" "$mingo_direction" "true" >/dev/null 2>&1; then
  connection_ok=1
  echo "Connection to $mingo_direction confirmed."
else
  log_error "Unable to establish SSH connection with $mingo_direction. Skipping data fetch for this run."
fi

if [[ $connection_ok -eq 1 ]]; then
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
  if ! ssh "${SSH_OPTS[@]}" "$mingo_direction" "$remote_list_cmd" 2>/dev/null; then
    echo "No .dat files found or listing unavailable."
  fi

  rsync_file_list=$(mktemp)
  filtered_rsync_file_list=$(mktemp)
  remote_find_cmd=$(printf 'cd %q && find . -maxdepth 1 -type f -regextype posix-extended -regex %s -printf %s' \
    "$dat_files_directory" "'.*/[^/]+\\.dat$'" "'%P\0'")
  if ssh "${SSH_OPTS[@]}" "$mingo_direction" "$remote_find_cmd" > "$rsync_file_list"; then
    if [[ -s "$rsync_file_list" ]]; then
      : > "$filtered_rsync_file_list"
      while IFS= read -r -d '' candidate; do
        candidate=${candidate//$'\r'/}
        candidate=${candidate#./}
        [[ -z "$candidate" ]] && continue
        if [[ -n ${logged_files["$candidate"]+_} ]]; then
          continue
        fi
        if ! filename_matches_date_ranges "$candidate"; then
          continue
        fi
        printf '%s\0' "$candidate" >> "$filtered_rsync_file_list"
      done < "$rsync_file_list"

      if [[ -s "$filtered_rsync_file_list" ]]; then
        if ! RSYNC_RSH="$RSYNC_RSH_CMD" rsync -avz --ignore-existing \
          --files-from="$filtered_rsync_file_list" \
          --from0 \
          "$mingo_direction:$dat_files_directory/" \
          "$raw_directory/"; then
          log_warn "rsync encountered an error while fetching data."
        fi
      else
        echo "No .dat files eligible for transfer after excluding logged entries."
      fi
    else
      echo "No .dat files found to transfer."
    fi
  else
    log_warn "unable to build .dat file list from remote host."
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
else
  echo "Skipping data fetch because $mingo_direction is unreachable."
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
CONFIG_FILE="$NEW_FILES_CONFIG_FILE"
SHEET_ID=$(yq -r '.logbook.sheet_id' "$CONFIG_FILE")
GID=$(yq -r ".logbook.gid_by_station.\"$station\"" "$CONFIG_FILE")

# Basic validation
if [[ -z "$SHEET_ID" || -z "$GID" ]]; then
  log_error "Could not read SHEET_ID or GID for station $station from $CONFIG_FILE"
  exit 1
fi

echo $SHEET_ID
echo $GID

# Define output file path
OUTPUT_FILE="$config_file_directory/input_file_mingo0${station}.csv"

# Download the file using wget with minimal console output (write to a temp file then move atomically)
echo "Downloading logbook for Station $station..."
TMP_DL_FILE="$(mktemp "${OUTPUT_FILE}.tmp.XXXXXXXX")"
if wget -q --show-progress --no-check-certificate \
     "https://docs.google.com/spreadsheets/d/${SHEET_ID}/export?format=csv&id=${SHEET_ID}&gid=${GID}" \
     -O "${TMP_DL_FILE}"; then
    mv "${TMP_DL_FILE}" "${OUTPUT_FILE}"
    echo "Download completed. Data saved at ${OUTPUT_FILE}."
else
    rm -f "${TMP_DL_FILE}"
    log_error "Download failed. Continuing execution."
fi


echo '------------------------------------------------------'
echo "bring_data_and_config_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
