#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bring_reprocessing_files.sh
#   Fetch HLD data from backuplip, writing
#     * *.hld.tar.gz or *.hld-tar-gz  → STAGE_0/REPROCESSING/INPUT_FILES/COMPRESSED_HLDS
#     * *.hld                         → STAGE_0/REPROCESSING/INPUT_FILES/UNCOMPRESSED_HLDS
# ---------------------------------------------------------------------------

set -e  # Exit on command failure
set -u  # Error on undefined variables
set -o pipefail  # Fail on any part of a pipeline

log_info() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_DIR="$SCRIPT_DIR"
while [[ "${MASTER_DIR}" != "/" && "$(basename "${MASTER_DIR}")" != "MASTER" ]]; do
  MASTER_DIR="$(dirname "${MASTER_DIR}")"
done
STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
STATUS_TIMESTAMP=""

# finish() {
#   local exit_code="$1"
#   if [[ ${exit_code} -eq 0 && -n "${STATUS_TIMESTAMP:-}" && -n "${STATUS_CSV:-}" ]]; then
#     python3 "$STATUS_HELPER" complete "$STATUS_CSV" "$STATUS_TIMESTAMP" >/dev/null 2>&1 || true
#   fi
# }
# trap 'finish $?' EXIT

##############################################################################
# Parse arguments
##############################################################################
usage() {
  echo "Usage: $0 <station> YYMMDD YYMMDD | --random/-r"
  exit 1
}

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
bring_reprocessing_files.sh
Fetches HLD data from backuplip into the STAGE_0 buffers for a station.

Usage:
  bring_reprocessing_files.sh <station> YYMMDD YYMMDD
  bring_reprocessing_files.sh <station> --random/-r

Options:
  -h, --help    Show this help message and exit.

The random mode selects a pending day automatically; otherwise provide a
start and end date in YYMMDD format for the desired range.
EOF
  exit 0
fi

if (( $# < 2 )); then
  usage
fi

station="$1"; shift

random_mode=false
start_arg=""
end_arg=""

case "$#" in
  1)
    if [[ ${1:-} =~ ^(--random|-r)$ ]]; then
      random_mode=true
    else
      usage
    fi
    ;;
  2)
    start_arg="$1"
    end_arg="$2"
    ;;
  *)
    usage
    ;;
esac

##############################################################################
# Target directories
##############################################################################
station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}"
reprocessing_directory="${station_directory}/STAGE_0/REPROCESSING"
input_directory="${reprocessing_directory}/INPUT_FILES"
compressed_directory="${input_directory}/COMPRESSED_HLDS"
uncompressed_directory="${input_directory}/UNCOMPRESSED_HLDS"
metadata_directory="${reprocessing_directory}/METADATA"
mkdir -p "$compressed_directory" "$uncompressed_directory" "$metadata_directory"

brought_csv="${metadata_directory}/hld_files_brought.csv"
brought_csv_header="hld_name,bring_timesamp"

# STATUS_CSV="${metadata_directory}/bring_reprocessing_files_status.csv"
# if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
#   echo "Warning: unable to record status in $STATUS_CSV" >&2
#   STATUS_TIMESTAMP=""
# fi

remote_dir="/local/experiments/MINGOS/MINGO0${station}/"
remote_user="rpcuser"
csv_path="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}/database_status_${station}.csv"
csv_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
csv_header="basename,start_date,hld_remote_add_date,hld_local_add_date,dat_add_date,list_ev_name,list_ev_add_date,acc_name,acc_add_date,merge_add_date"

# Snapshot existing archives to detect fresh downloads after rsync completes
tmp_before_compressed=$(mktemp)
tmp_before_uncompressed=$(mktemp)
tmp_remote=$(mktemp)
tmp_remote_sorted=""
tmp_after_compressed=""
tmp_after_uncompressed=""
tmp_new_compressed=""
tmp_new_uncompressed=""
compressed_list_file=""
uncompressed_list_file=""
new_downloads_file=""

cleanup() {
  rm -f "$tmp_before_compressed" "$tmp_before_uncompressed" "$tmp_remote"
  [[ -n "$tmp_remote_sorted" ]] && rm -f "$tmp_remote_sorted"
  [[ -n "$tmp_after_compressed" ]] && rm -f "$tmp_after_compressed"
  [[ -n "$tmp_after_uncompressed" ]] && rm -f "$tmp_after_uncompressed"
  [[ -n "$tmp_new_compressed" ]] && rm -f "$tmp_new_compressed"
  [[ -n "$tmp_new_uncompressed" ]] && rm -f "$tmp_new_uncompressed"
  [[ -n "$compressed_list_file" ]] && rm -f "$compressed_list_file"
  [[ -n "$uncompressed_list_file" ]] && rm -f "$uncompressed_list_file"
  [[ -n "$new_downloads_file" ]] && rm -f "$new_downloads_file"
}
trap cleanup EXIT

ensure_brought_csv() {
  if [[ ! -f "$brought_csv" || ! -s "$brought_csv" ]]; then
    printf '%s\n' "$brought_csv_header" > "$brought_csv"
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

extract_doy() {
  local name="$1"
  if [[ $name =~ ([0-9]{11})$ ]]; then
    local digits=${BASH_REMATCH[1]}
    printf '%s' "${digits:0:5}"
  else
    printf ''
  fi
}

doy_to_epoch() {
  local doy="$1"
  if [[ ! $doy =~ ^[0-9]{5}$ ]]; then
    return 1
  fi
  local yy=${doy:0:2}
  local ddd=${doy:2:3}
  local year=$((2000 + 10#$yy))
  local offset=$((10#$ddd - 1))
  (( offset < 0 )) && offset=0
  date -d "${year}-01-01 +${offset} days" +%s
}

doy_to_ymd() {
  local doy="$1"
  local epoch
  epoch=$(doy_to_epoch "$doy") || return 1
  date -d "@${epoch}" +%y%m%d
}

declare -a target_doys=()
declare -a target_ymds=()
declare -A target_doy_lookup=()

build_target_days() {
  local start_ymd="$1"
  local end_ymd="$2"

  local start_epoch
  local end_epoch
  start_epoch=$(date -d "20${start_ymd:0:2}-${start_ymd:2:2}-${start_ymd:4:2}" +%s)
  end_epoch=$(date -d "20${end_ymd:0:2}-${end_ymd:2:2}-${end_ymd:4:2}" +%s)
  if (( start_epoch > end_epoch )); then
    log_info "Error: start date ${start_ymd} is after end date ${end_ymd}" >&2
    return 1
  fi

  target_doys=()
  target_ymds=()
  target_doy_lookup=()

  local current_epoch="$start_epoch"
  local one_day=86400
  while (( current_epoch <= end_epoch )); do
    local ymd
    local doy
    ymd=$(date -d "@${current_epoch}" +%y%m%d) || return 1
    doy=$(date -d "@${current_epoch}" +%y%j) || return 1
    target_ymds+=("$ymd")
    target_doys+=("$doy")
    target_doy_lookup["$doy"]=1
    current_epoch=$(( current_epoch + one_day ))
  done

  return 0
}

# ensure_csv

# log_info "CSV initialized at ${csv_path}"

declare -A csv_rows=()
declare -A downloaded_bases=()
if [[ -s "$csv_path" ]]; then
  {
    read -r _header || true
    while IFS=',' read -r base_name start_date remote_added local_added dat_added remaining; do
      [[ -z "$base_name" ]] && continue
      base_name=${base_name//$'\r'/}
      local_added=${local_added//$'\r'/}
      csv_rows["$base_name"]=1
      if [[ -n "$local_added" ]]; then
        downloaded_bases["$base_name"]=1
      fi
    done
  } < "$csv_path"
fi
log_info "Loaded CSV rows: total=${#csv_rows[@]}, with local copies=${#downloaded_bases[@]}"

declare -A local_files=()
while IFS= read -r local_entry; do
  [[ -z "$local_entry" ]] && continue
  local_files["$local_entry"]=1
done < <(find "$input_directory" -type f -name '*.hld*' -printf '%f\n')
log_info "Local HLD files currently buffered: ${#local_files[@]}"

declare -A base_to_doy=()
declare -A base_compressed_files=()
declare -A base_uncompressed_files=()

log_info "Listing remote files from ${remote_user}@backuplip:${remote_dir} ..."

if ssh -o BatchMode=yes "${remote_user}@backuplip" "cd ${remote_dir} && ls -1" > "$tmp_remote" 2>/dev/null; then
  log_info "Remote listing retrieved."
else
  log_info "Warning: unable to list remote directory ${remote_user}@backuplip:${remote_dir}" >&2
fi

tmp_remote_sorted=$(mktemp)
sort -u "$tmp_remote" > "$tmp_remote_sorted"
remote_entry_count=$(wc -l < "$tmp_remote_sorted" | tr -d '[:space:]')
log_info "Remote entries discovered: ${remote_entry_count}"

while IFS= read -r remote_entry; do
  remote_entry=${remote_entry//$'\r'/}
  [[ -z "$remote_entry" ]] && continue
  [[ $remote_entry =~ \.hld ]] || continue
  base=$(strip_suffix "$remote_entry")
  [[ -z "$base" ]] && continue
  [[ $base =~ ^(mi|minI) ]] || continue
  doy=$(extract_doy "$base")
  [[ -z "$doy" ]] && continue

  base_to_doy["$base"]="$doy"

  case "$remote_entry" in
    *.hld.tar.gz|*.hld-tar-gz)
      base_compressed_files["$base"]+="${remote_entry}"$'\n'
      ;;
    *.hld)
      [[ "$remote_entry" =~ \.tar\.gz$ ]] && continue
      [[ "$remote_entry" =~ -tar-gz$ ]] && continue
      base_uncompressed_files["$base"]+="${remote_entry}"$'\n'
      ;;
    *)
      continue
      ;;
  esac

  if [[ -n ${csv_rows["$base"]+_} ]]; then
    continue
  fi
  start_value=$(compute_start_date "$base")
  # printf '%s,%s,%s,,,,,,,\n' "$base" "$start_value" "$csv_timestamp" >> "$csv_path"
  csv_rows["$base"]=1
done < "$tmp_remote_sorted"

log_info "Remote parsing complete: bases=${#base_to_doy[@]} compressed_candidates=${#base_compressed_files[@]} uncompressed_candidates=${#base_uncompressed_files[@]}"

declare -A pending_doys=()
for base in "${!base_to_doy[@]}"; do
  [[ -z ${base_to_doy["$base"]+_} ]] && continue
  if [[ -n ${downloaded_bases["$base"]+_} ]]; then
    continue
  fi
  pending_doys["${base_to_doy["$base"]}"]=1
done
log_info "Pending DOYs not yet downloaded: ${#pending_doys[@]}"

start=""
end=""

if $random_mode; then
  epoch_start=$(date -d '2023-07-01 00:00:00' +%s)
  epoch_end=$(date -d 'today -5 days 00:00:00' +%s)
  mapfile -t candidate_doys < <(
    for doy in "${!pending_doys[@]}"; do
      epoch=$(doy_to_epoch "$doy" 2>/dev/null) || continue
      if (( epoch >= epoch_start && epoch <= epoch_end )); then
        printf '%s\n' "$doy"
      fi
    done | sort -u
  )
  if (( ${#candidate_doys[@]} == 0 )); then
    log_info "No pending HLD days eligible for random selection." >&2
    exit 0
  fi
  selected_doy=$(printf '%s\n' "${candidate_doys[@]}" | shuf -n1)
  selected_ymd=$(doy_to_ymd "$selected_doy")
  if [[ -z "$selected_ymd" ]]; then
    log_info "Failed to convert DOY ${selected_doy} to date." >&2
    exit 1
  fi
  start="$selected_ymd"
  end="$selected_ymd"
  log_info "Random day selected: $selected_ymd"
else
  start="$start_arg"
  end="$end_arg"
fi

if [[ ! $start =~ ^[0-9]{6}$ || ! $end =~ ^[0-9]{6}$ ]]; then
  log_info "Dates must be provided as YYMMDD; received start=${start} end=${end}" >&2
  exit 1
fi

if ! build_target_days "$start" "$end"; then
  exit 1
fi
log_info "Requested range ${start}-${end} produced ${#target_doys[@]} target day(s)"

if (( ${#target_doys[@]} == 0 )); then
  log_info "No target days found in the requested range (${start}-${end})." >&2
  exit 0
fi

last_index=$(( ${#target_doys[@]} - 1 ))
start_DOY="${target_doys[0]}"
end_DOY="${target_doys[$last_index]}"
start_label="${target_ymds[0]}"
end_label="${target_ymds[$last_index]}"

compressed_list_file=$(mktemp)
uncompressed_list_file=$(mktemp)
for base in "${!base_to_doy[@]}"; do
  doy="${base_to_doy["$base"]}"
  [[ -n ${target_doy_lookup["$doy"]+_} ]] || continue
  if [[ -n ${downloaded_bases["$base"]+_} ]]; then
    continue
  fi
  if [[ -n ${base_compressed_files["$base"]+_} ]]; then
    while IFS= read -r filename; do
      [[ -z "$filename" ]] && continue
      if [[ -n ${local_files["$filename"]+_} ]]; then
        continue
      fi
      printf '%s\n' "$filename" >> "$compressed_list_file"
    done <<< "${base_compressed_files["$base"]}"
  fi
  if [[ -n ${base_uncompressed_files["$base"]+_} ]]; then
    while IFS= read -r filename; do
      [[ -z "$filename" ]] && continue
      if [[ -n ${local_files["$filename"]+_} ]]; then
        continue
      fi
      printf '%s\n' "$filename" >> "$uncompressed_list_file"
    done <<< "${base_uncompressed_files["$base"]}"
  fi
done

if [[ -s "$compressed_list_file" ]]; then
  sort -u "$compressed_list_file" -o "$compressed_list_file"
fi
if [[ -s "$uncompressed_list_file" ]]; then
  sort -u "$uncompressed_list_file" -o "$uncompressed_list_file"
fi

compressed_count=0
uncompressed_count=0
if [[ -s "$compressed_list_file" ]]; then
  compressed_count=$(wc -l < "$compressed_list_file" | tr -d '[:space:]')
fi
if [[ -s "$uncompressed_list_file" ]]; then
  uncompressed_count=$(wc -l < "$uncompressed_list_file" | tr -d '[:space:]')
fi

if [[ "$start_label" == "$end_label" ]]; then
  log_info "Fetching HLD files for MINGO0${station} on ${start_label} (DOY ${start_DOY})"
else
  log_info "Fetching HLD files for MINGO0${station} from ${start_label} to ${end_label} (DOY ${start_DOY}-${end_DOY})"
fi
log_info "  Planned compressed downloads : $compressed_count"
log_info "  Planned uncompressed downloads: $uncompressed_count"
log_info ""

find "$compressed_directory" -maxdepth 1 -type f \
  \( -name '*.hld*.tar.gz' -o -name '*.hld-tar-gz' \) \
  -printf '%f\n' | sort -u > "$tmp_before_compressed"

find "$uncompressed_directory" -maxdepth 1 -type f \
  -name '*.hld' \
  -printf '%f\n' | sort -u > "$tmp_before_uncompressed"

if [[ -s "$compressed_list_file" ]]; then
  echo "Starting compressed transfers..."
  if ! rsync -av --info=progress2 \
      --files-from="$compressed_list_file" \
      --ignore-missing-args \
      --ignore-existing --no-compress \
      "${remote_user}@backuplip:${remote_dir}" "$compressed_directory/"; then
    rsync_status=$?
    if (( rsync_status != 23 && rsync_status != 24 )); then
      exit "$rsync_status"
    fi
    echo "Warning: rsync reported status $rsync_status while transferring compressed files; continuing." >&2
  fi
  while IFS= read -r filename; do
    [[ -z "$filename" ]] && continue
    target_path="${compressed_directory}/${filename}"
    if [[ -f "$target_path" ]]; then
      touch "$target_path"
    fi
  done < "$compressed_list_file"
fi

if [[ -s "$uncompressed_list_file" ]]; then
  echo "Starting uncompressed transfers..."
  if ! rsync -av --info=progress2 \
      --files-from="$uncompressed_list_file" \
      --ignore-missing-args \
      --ignore-existing --no-compress \
      "${remote_user}@backuplip:${remote_dir}" "$uncompressed_directory/"; then
    rsync_status=$?
    if (( rsync_status != 23 && rsync_status != 24 )); then
      exit "$rsync_status"
    fi
    echo "Warning: rsync reported status $rsync_status while transferring uncompressed files; continuing." >&2
  fi
  while IFS= read -r filename; do
    [[ -z "$filename" ]] && continue
    target_path="${uncompressed_directory}/${filename}"
    if [[ -f "$target_path" ]]; then
      touch "$target_path"
    fi
  done < "$uncompressed_list_file"
fi

if (( compressed_count == 0 && uncompressed_count == 0 )); then
  echo "No files matched the requested range; nothing to transfer."
fi

tmp_after_compressed=$(mktemp)
find "$compressed_directory" -maxdepth 1 -type f \
  \( -name '*.hld*.tar.gz' -o -name '*.hld-tar-gz' \) \
  -printf '%f\n' | sort -u > "$tmp_after_compressed"

tmp_after_uncompressed=$(mktemp)
find "$uncompressed_directory" -maxdepth 1 -type f \
  -name '*.hld' \
  -printf '%f\n' | sort -u > "$tmp_after_uncompressed"

tmp_new_compressed=$(mktemp)
comm -13 "$tmp_before_compressed" "$tmp_after_compressed" > "$tmp_new_compressed"

tmp_new_uncompressed=$(mktemp)
comm -13 "$tmp_before_uncompressed" "$tmp_after_uncompressed" > "$tmp_new_uncompressed"

if [[ -s "$tmp_new_compressed" || -s "$tmp_new_uncompressed" ]]; then
  new_downloads_file=$(mktemp)
  [[ -s "$tmp_new_compressed" ]] && cat "$tmp_new_compressed" >> "$new_downloads_file"
  [[ -s "$tmp_new_uncompressed" ]] && cat "$tmp_new_uncompressed" >> "$new_downloads_file"
  ensure_brought_csv
  while IFS= read -r brought_file; do
    [[ -z "$brought_file" ]] && continue
    printf '%s,%s\n' "$brought_file" "$csv_timestamp" >> "$brought_csv"
  done < "$new_downloads_file"
fi

# if [[ -s "$tmp_new" ]]; then
#   awk -F',' -v OFS=',' -v newlist="$tmp_new" -v ts="$csv_timestamp" '
#     function canonical(name) {
#       gsub(/\r/, "", name)
#       sub(/\.hld\.tar\.gz$/, "", name)
#       sub(/\.hld-tar-gz$/, "", name)
#       sub(/\.tar\.gz$/, "", name)
#       sub(/\.hld$/, "", name)
#       sub(/\.dat$/, "", name)
#       return name
#     }
#     BEGIN {
#       while ((getline line < newlist) > 0) {
#         line = canonical(line)
#         if (line != "") {
#           new[line] = 1
#         }
#       }
#       close(newlist)
#     }
#     NR == 1 { print; next }
#     {
#       key = canonical($1)
#       if (key in new) {
#         if ($4 == "") {
#           $4 = ts
#         }
#         if ($5 == "") {
#           $5 = ts
#         }
#       }
#       print
#     }
#   ' "$csv_path" > "${csv_path}.tmp" && mv "${csv_path}.tmp" "$csv_path"
# fi

echo
echo '------------------------------------------------------'
echo "bring_reprocessing_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
