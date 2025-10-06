#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bring_reprocessing_files.sh
#   Fetch HLD data from backuplip, writing
#     * *.hld.tar.gz or *.hld-tar-gz  → ZERO_STAGE/COMPRESSED_HLDS
#     * *.hld                         → ZERO_STAGE/UNCOMPRESSED_HLDS
# ---------------------------------------------------------------------------

set -e  # Exit on command failure
set -u  # Error on undefined variables
set -o pipefail  # Fail on any part of a pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
  MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""

finish() {
  local exit_code="$1"
  if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
    python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
  fi
}
trap 'finish $?' EXIT

##############################################################################
# Parse arguments
##############################################################################
if (( $# < 2 )); then
  echo "Usage: $0 <station> YYMMDD YYMMDD | --random|-r"
  exit 1
fi

station="$1"; shift

if [[ ${1:-} =~ ^(--random|-r)$ ]]; then
  epoch_start=$(date -d '2023-07-01 00:00:00' +%s)
  epoch_end=$(  date -d 'today -5 days 00:00:00' +%s)
  rand_epoch=$(shuf -i "${epoch_start}-${epoch_end}" -n1)
  rand_ymd=$(date -d "@${rand_epoch}" +%y%m%d)
  start="$rand_ymd"; end="$rand_ymd"
  echo "Random day selected: $rand_ymd"
else
  if (( $# != 2 )); then
    echo "Usage: $0 <station> YYMMDD YYMMDD | --random|-r"
    exit 1
  fi
  start="$1"; end="$2"
fi

##############################################################################
# Date conversion
##############################################################################
start_DOY=$(date -d "20${start:0:2}-${start:2:2}-${start:4:2}" +%y%j)
end_DOY=$(  date -d "20${end:0:2}-${end:2:2}-${end:4:2}"   +%y%j)

##############################################################################
# Target directories
##############################################################################
base_dir="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE"
compressed_directory="${base_dir}/COMPRESSED_HLDS"
uncompressed_directory="${base_dir}/UNCOMPRESSED_HLDS"
mkdir -p "$compressed_directory" "$uncompressed_directory"

STATUS_CSV="${base_dir}/bring_reprocessing_files_status.csv"
if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
  echo "Warning: unable to record status in $STATUS_CSV" >&2
  STATUS_TIMESTAMP=""
fi

remote_dir="/local/experiments/MINGOS/MINGO0${station}/"
remote_user="rpcuser"
csv_path="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/database_status_${station}.csv"
csv_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
csv_header="basename,start_date,hld_remote_add_date,hld_local_add_date,dat_add_date,list_ev_name,list_ev_add_date,acc_name,acc_add_date,merge_add_date"
patterns=("mi0${station}" "minI${station}")

# Snapshot existing archives to detect fresh downloads after rsync completes
tmp_before=$(mktemp)
tmp_remote=$(mktemp)
tmp_remote_sorted=""
tmp_after=""
tmp_new=""
exclude_file=""

cleanup() {
  rm -f "$tmp_before" "$tmp_remote"
  [[ -n "$tmp_remote_sorted" ]] && rm -f "$tmp_remote_sorted"
  [[ -n "$tmp_after" ]] && rm -f "$tmp_after"
  [[ -n "$tmp_new" ]] && rm -f "$tmp_new"
  [[ -n "$exclude_file" ]] && rm -f "$exclude_file"
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
    local date_value
    date_value=$(date -d "${year}-01-01 +${offset} days ${hh}:${mm}:${ss}" '+%Y-%m-%d_%H.%M.%S' 2>/dev/null) || date_value=""
    printf '%s' "$date_value"
  else
    printf ''
  fi
}

ensure_csv

declare -A existing_rows=()
if [[ -s "$csv_path" ]]; then
  while IFS=',' read -r existing_basename _; do
    [[ -z "$existing_basename" || "$existing_basename" == "basename" ]] && continue
    existing_basename=${existing_basename//$'\r'/}
    existing_rows["$existing_basename"]=1
  done < "$csv_path"
fi

if ssh -o BatchMode=yes "${remote_user}@backuplip" "cd ${remote_dir} && ls -1" > "$tmp_remote" 2>/dev/null; then
  :
else
  echo "Warning: unable to list remote directory ${remote_user}@backuplip:${remote_dir}" >&2
fi

tmp_remote_sorted=$(mktemp)
sort -u "$tmp_remote" > "$tmp_remote_sorted"

while IFS= read -r remote_entry; do
  remote_entry=${remote_entry//$'\r'/}
  [[ -z "$remote_entry" ]] && continue
  [[ $remote_entry =~ \.hld ]] || continue
  base=$(strip_suffix "$remote_entry")
  [[ -z "$base" ]] && continue
  [[ $base =~ ^(mi|minI) ]] || continue
  if [[ -n ${existing_rows["$base"]+_} ]]; then
    continue
  fi
  start_value=$(compute_start_date "$base")
  printf '%s,%s,%s,,,,,,,\n' "$base" "$start_value" "$csv_timestamp" >> "$csv_path"
  existing_rows["$base"]=1
done < "$tmp_remote_sorted"

exclude_file=$(mktemp)
if [[ -s "$csv_path" ]]; then
  tail -n +2 "$csv_path" | while IFS=',' read -r base_name _; do
    base_name=${base_name//$'\r'/}
    [[ -z "$base_name" ]] && continue
    printf '%s.hld.tar.gz\n' "$base_name" >> "$exclude_file"
    printf '%s.hld-tar-gz\n' "$base_name" >> "$exclude_file"
    printf '%s.hld\n' "$base_name" >> "$exclude_file"
  done
fi

find "$base_dir" -type f -name '*.hld*' -printf '%f\n' >> "$exclude_file"
sort -u "$exclude_file" -o "$exclude_file"

find "$compressed_directory" -maxdepth 1 -type f \
  \( -name '*.hld*.tar.gz' -o -name '*.hld-tar-gz' \) \
  -printf '%f\n' | sort -u > "$tmp_before"

echo "Fetching HLD files for MINGO0$station between $start_DOY and $end_DOY..."
echo

##############################################################################
# Transfer loop
##############################################################################
for pattern in "${patterns[@]}"; do
  echo "Pattern: $pattern"

  for doy in $(seq "$start_DOY" "$end_DOY"); do
    echo "  DOY $doy"

    # -------------------- 1. COMPRESSED  (.hld.tar.gz | .hld-tar-gz) --------
    rsync -avz --progress \
      --include="${pattern}*${doy}*.hld*.tar.gz" \
      --include="${pattern}*${doy}*.hld-tar-gz" \
      --exclude-from="$exclude_file" \
      --exclude='*' \
      --ignore-existing --no-compress \
      "${remote_user}@backuplip:${remote_dir}" "$compressed_directory/"

    # touch freshly-downloaded compressed files
    find "$compressed_directory" -type f \( \
           -name "${pattern}*${doy}*.hld*.tar.gz" -o \
           -name "${pattern}*${doy}*.hld-tar-gz" \) -exec touch {} +

    # -------------------- 2. UNCOMPRESSED (.hld only) -----------------------
    rsync -avz --progress \
      --include="${pattern}*${doy}*.hld" \
      --exclude='*.tar.gz' \
      --exclude='*-tar-gz' \
      --exclude-from="$exclude_file" \
      --exclude='*' \
      --ignore-existing --no-compress \
      "${remote_user}@backuplip:${remote_dir}" "$uncompressed_directory/"

    # touch freshly-downloaded uncompressed files
    find "$uncompressed_directory" -type f -name "${pattern}*${doy}*.hld" -exec touch {} +
  done

done

tmp_after=$(mktemp)
find "$compressed_directory" -maxdepth 1 -type f \
  \( -name '*.hld*.tar.gz' -o -name '*.hld-tar-gz' \) \
  -printf '%f\n' | sort -u > "$tmp_after"

tmp_new=$(mktemp)
comm -13 "$tmp_before" "$tmp_after" > "$tmp_new"

if [[ -s "$tmp_new" ]]; then
  awk -F',' -v OFS=',' -v newlist="$tmp_new" -v ts="$csv_timestamp" '
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
      while ((getline line < newlist) > 0) {
        line = canonical(line)
        if (line != "") {
          new[line] = 1
        }
      }
      close(newlist)
    }
    NR == 1 { print; next }
    {
      key = canonical($1)
      if (key in new) {
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

echo
echo '------------------------------------------------------'
echo "bring_reprocessing_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
