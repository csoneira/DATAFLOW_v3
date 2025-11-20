#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bring_reprocessing_files.sh
#   Fetch HLD data from backuplip, writing
#     * *.hld.tar.gz or *.hld-tar-gz  --> STAGE_0/REPROCESSING/INPUT_FILES/COMPRESSED_HLDS
#     * *.hld                         --> STAGE_0/REPROCESSING/INPUT_FILES/UNCOMPRESSED_HLDS
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
# STATUS_HELPER="${MASTER_DIR}/common/status_csv.py"
# STATUS_TIMESTAMP=""

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
  cat <<'EOF'
Usage:
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot]
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot] --random|-r
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot] --newest|-n
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot] <start YYMMDD> <end YYMMDD>

Options:
  -h, --help            Show this help message and exit.
  -r, --random          Select a single random basename from metadata and download it.
  -n, --newest          Select the newest (lexicographically last) unprocessed basename.
  -m, --refresh-metadata
                        Refresh the metadata CSV with the latest list of basenames
                        from the remote host. Can be combined with --random or a
                        date range, or used alone.
      --plot            Generate a histogram (PDF) of time differences between
                        consecutive metadata entries in STEP_1/PLOTS. Can be run
                        by itself using the existing metadata CSV or combined
                        with other flags.
EOF
  exit 1
}

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  cat <<'EOF'
bring_reprocessing_files.sh
Fetches HLD data from backuplip into the STAGE_0 buffers for a station.

Usage:
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot]
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot] --random/-r
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot] --newest/-n
  bring_reprocessing_files.sh <station> [--refresh-metadata|-m] [--plot] YYMMDD YYMMDD

Options:
  -h, --help            Show this help message and exit.
  -r, --random          Select a single random basename from metadata and download it.
  -n, --newest          Select the newest (lexicographically last) unprocessed basename.
  -m, --refresh-metadata
                        Refresh the metadata CSV with the latest list of basenames
                        from the remote host. Can be combined with --random or a
                        date range, or used alone.
      --plot            Generate a histogram (PDF) of time differences between
                        consecutive metadata entries in STEP_1/PLOTS. Can be run
                        by itself using the existing metadata CSV or combined
                        with other flags.

The random mode selects a pending day automatically; otherwise provide a
start and end date in YYMMDD format for the desired range.
EOF
  exit 0
fi

if (( $# < 1 )); then
  usage
fi

station="$1"
shift

random_mode=false
newest_mode=false
refresh_metadata=false
plot_hist=false
start_arg=""
end_arg=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh-metadata|-m)
      refresh_metadata=true
      shift
      ;;
    --plot)
      plot_hist=true
      shift
      ;;
    --random|-r)
      if [[ -n "$start_arg" || -n "$end_arg" || $newest_mode == true ]]; then
        usage
      fi
      random_mode=true
      shift
      ;;
    --newest|-n)
      if [[ -n "$start_arg" || -n "$end_arg" || $random_mode == true ]]; then
        usage
      fi
      newest_mode=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      if [[ "$1" =~ ^[0-9]{6}$ ]]; then
        if $random_mode || $newest_mode; then
          usage
        fi
        if [[ -z "$start_arg" ]]; then
          start_arg="$1"
        elif [[ -z "$end_arg" ]]; then
          end_arg="$1"
        else
          usage
        fi
        shift
      else
        usage
      fi
      ;;
  esac
done

perform_download=true
if $random_mode || $newest_mode; then
  if [[ -n "$start_arg" || -n "$end_arg" ]]; then
    usage
  fi
else
  if [[ -z "$start_arg" || -z "$end_arg" ]]; then
    perform_download=false
  fi
fi

if ! $refresh_metadata && ! $perform_download && ! $plot_hist; then
  usage
fi

##############################################################################
# Target directories
##############################################################################
station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}"
reprocessing_directory="${station_directory}/STAGE_0/REPROCESSING/STEP_1"
input_directory="${reprocessing_directory}/INPUT_FILES"
output_directory="${reprocessing_directory}/OUTPUT_FILES"
compressed_directory="${output_directory}/COMPRESSED_HLDS"
uncompressed_directory="${output_directory}/UNCOMPRESSED_HLDS"
metadata_directory="${reprocessing_directory}/METADATA"
mkdir -p "$input_directory" "$compressed_directory" "$uncompressed_directory" "$metadata_directory"

step0_directory="${station_directory}/STAGE_0/REPROCESSING/STEP_0"
step0_metadata_directory="${step0_directory}/METADATA"
step0_output_directory="${step0_directory}/OUTPUT_FILES"

brought_csv="${metadata_directory}/hld_files_brought.csv"
brought_csv_header="hld_name,bring_timesamp"

# STATUS_CSV="${metadata_directory}/bring_reprocessing_files_status.csv"
# if ! STATUS_TIMESTAMP="$(python3 "$STATUS_HELPER" append "$STATUS_CSV")"; then
#   echo "Warning: unable to record status in $STATUS_CSV" >&2
#   STATUS_TIMESTAMP=""
# fi

remote_dir="/local/experiments/MINGOS/MINGO0${station}/"
remote_user="rpcuser"
printf -v remote_dir_escaped '%q' "$remote_dir"
clean_metadata_csv="${step0_output_directory}/clean_remote_database_${station}.csv"
working_metadata_csv="${input_directory}/clean_remote_database_${station}.csv"
csv_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
csv_header="basename"

metadata_prep_script="${MASTER_DIR}/STAGE_0/REPROCESSING/STEP_0/prepare_reprocessing_metadata.sh"
if $refresh_metadata || $plot_hist; then
  if [[ ! -x "$metadata_prep_script" ]]; then
    log_info "Metadata preparation script not found at $metadata_prep_script" >&2
    exit 1
  fi
  step0_args=("$station")
  $refresh_metadata && step0_args+=("--refresh-metadata")
  $plot_hist && step0_args+=("--plot")
  "$metadata_prep_script" "${step0_args[@]}"
fi

# Snapshot existing archives to detect fresh downloads after rsync completes
tmp_before_compressed=$(mktemp)
tmp_before_uncompressed=$(mktemp)
tmp_after_compressed=""
tmp_after_uncompressed=""
tmp_new_compressed=""
tmp_new_uncompressed=""
compressed_list_file=""
uncompressed_list_file=""
new_downloads_file=""

cleanup() {
  rm -f "$tmp_before_compressed" "$tmp_before_uncompressed"
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

declare -A brought_hld_records=()
load_brought_hld_records() {
  if [[ ! -s "$brought_csv" ]]; then
    return
  fi
  while IFS=',' read -r hld_name _; do
    hld_name=${hld_name//$'\r'/}
    [[ -z "$hld_name" || "$hld_name" == "hld_name" ]] && continue
    brought_hld_records["$hld_name"]=1
  done < "$brought_csv"
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

time_key_to_epoch() {
  local key="$1"
  if [[ ! $key =~ ^[0-9]{11}$ ]]; then
    return 1
  fi
  local yy=${key:0:2}
  local doy=${key:2:3}
  local hh=${key:5:2}
  local mm=${key:7:2}
  local ss=${key:9:2}
  local day_index=$((10#$doy - 1))
  if (( day_index < 0 )); then
    return 1
  fi
  date -d "20${yy}-01-01 +${day_index} days ${hh}:${mm}:${ss}" +%s 2>/dev/null
}

basename_to_epoch() {
  local name="$1"
  local key
  key=$(basename_time_key "$name")
  if [[ -z "$key" ]]; then
    return 1
  fi
  time_key_to_epoch "$key"
}

newest_mode_order_key() {
  local base="$1"
  local normalized="$base"
  if [[ $normalized == minI* ]]; then
    normalized="mi01${normalized:4}"
  fi
  local prefix="mi0${station}"
  if [[ $normalized == "${prefix}"* ]]; then
    printf '%s' "${normalized:${#prefix}}"
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

ymd_to_epoch() {
  local ymd="$1"
  if [[ ! $ymd =~ ^[0-9]{6}$ ]]; then
    return 1
  fi
  local yy=${ymd:0:2}
  local mm=${ymd:2:2}
  local dd=${ymd:4:2}
  date -d "20${yy}-${mm}-${dd}" +%s
}

ymd_to_bound_key() {
  local ymd="$1"
  local suffix="$2"
  local doy
  doy=$(date -d "20${ymd:0:2}-${ymd:2:2}-${ymd:4:2}" +%y%j 2>/dev/null) || return 1
  printf '%s%s' "$doy" "$suffix"
}

if [[ ! -s "$clean_metadata_csv" ]]; then
  log_info "Clean metadata CSV not found at ${clean_metadata_csv}. Run $(basename "$metadata_prep_script") --refresh-metadata first." >&2
  exit 1
fi

if ! cp "$clean_metadata_csv" "$working_metadata_csv"; then
  log_info "Unable to copy metadata CSV into ${working_metadata_csv}." >&2
  exit 1
fi
csv_path="$working_metadata_csv"
log_info "Using metadata from ${csv_path}"

declare -a metadata_basenames=()
if [[ -s "$csv_path" ]]; then
  {
    read -r _header || true
    while IFS=',' read -r base_name _rest; do
      base_name=${base_name//$'\r'/}
      [[ -z "$base_name" || "$base_name" == "$csv_header" ]] && continue
      metadata_basenames+=("$base_name")
    done
  } < "$csv_path"
fi

if (( ${#metadata_basenames[@]} == 0 )); then
  log_info "Metadata CSV ${csv_path} does not list any basenames; nothing to do."
  exit 0
fi
log_info "Loaded ${#metadata_basenames[@]} basenames from metadata CSV."

if ! $perform_download; then
  if $plot_hist; then
    log_info "No download requested; plot-only run complete."
  else
    log_info "No download requested; exiting after metadata handling."
  fi
  exit 0
fi

declare -A brought_basenames=()
if [[ -s "$brought_csv" ]]; then
  {
    read -r _header || true
    while IFS=',' read -r recorded_name bring_ts; do
      recorded_name=${recorded_name//$'\r'/}
      [[ -z "$recorded_name" || "$recorded_name" == "hld_name" ]] && continue
      brought_basenames["$recorded_name"]=1
    done
  } < "$brought_csv"
fi
log_info "Basenames already recorded as brought: ${#brought_basenames[@]}"

log_info "Selecting basenames to download..."

declare -a selected_bases=()

if $random_mode; then
  mapfile -t random_candidates < <(
    for base in "${metadata_basenames[@]}"; do
      [[ -n ${brought_basenames["$base"]+_} ]] && continue
      printf '%s\n' "$base"
    done
  )
  if (( ${#random_candidates[@]} == 0 )); then
    random_candidates=("${metadata_basenames[@]}")
  fi
  if (( ${#random_candidates[@]} == 0 )); then
    log_info "No basenames available for random selection."
    exit 0
  fi
  selected_bases=($(printf '%s\n' "${random_candidates[@]}" | shuf -n1))
  log_info "Randomly selected basename: ${selected_bases[0]}"
elif $newest_mode; then
  mapfile -t newest_candidates < <(
    for base in "${metadata_basenames[@]}"; do
      [[ -n ${brought_basenames["$base"]+_} ]] && continue
      printf '%s\n' "$base"
    done
  )
  if (( ${#newest_candidates[@]} == 0 )); then
    log_info "--newest requested but all metadata basenames are already recorded in hld_files_brought.csv."
    exit 0
  fi
  newest_base=""
  newest_key=""
  for base in "${newest_candidates[@]}"; do
    order_key=$(newest_mode_order_key "$base")
    if [[ -z "$newest_base" || "$order_key" > "$newest_key" ]]; then
      newest_base="$base"
      newest_key="$order_key"
    fi
  done
  if [[ -z "$newest_base" ]]; then
    newest_base=$(printf '%s\n' "${newest_candidates[@]}" | sort | tail -n1)
    newest_key=""
  fi
  selected_bases=("$newest_base")
  if [[ -n "$newest_key" ]]; then
    log_info "--newest flag active; newest unprocessed basename selected: ${selected_bases[0]} (key ${newest_key})"
  else
    log_info "--newest flag active; newest unprocessed basename selected: ${selected_bases[0]}"
  fi
else
  start="$start_arg"
  end="$end_arg"

  if [[ ! $start =~ ^[0-9]{6}$ || ! $end =~ ^[0-9]{6}$ ]]; then
    log_info "Dates must be provided as YYMMDD; received start=${start} end=${end}" >&2
    exit 1
  fi

  start_key=$(ymd_to_bound_key "$start" "000000") || {
    log_info "Unable to compute comparison key for start date ${start}" >&2
    exit 1
  }
  end_key=$(ymd_to_bound_key "$end" "235959") || {
    log_info "Unable to compute comparison key for end date ${end}" >&2
    exit 1
  }
  start_bound="mi0${station}${start_key}"
  end_bound="mi0${station}${end_key}"

  for base in "${metadata_basenames[@]}"; do
    [[ -n ${brought_basenames["$base"]+_} ]] && continue
    [[ "$base" < "$start_bound" ]] && continue
    [[ "$base" > "$end_bound" ]] && continue
    selected_bases+=("$base")
  done

  if (( ${#selected_bases[@]} == 0 )); then
    log_info "No basenames remain to download for the requested range ${start}-${end}; all listed entries are already recorded in hld_files_brought.csv."
    exit 0
  fi

  log_info "Range ${start}-${end} matched ${#selected_bases[@]} basename(s)."
fi

declare -A compressed_lookup=()
declare -A uncompressed_lookup=()

for base in "${selected_bases[@]}"; do
  log_info "Inspecting remote files for basename ${base}"
  printf -v base_prefix '%q' "$base"
  remote_listing=$(ssh -o BatchMode=yes "${remote_user}@backuplip" "cd ${remote_dir_escaped} && ls -1 ${base_prefix}*" 2>/dev/null || true)
  if [[ -z "$remote_listing" ]]; then
    log_info "  No remote files found for basename ${base}" >&2
    continue
  fi
  while IFS= read -r remote_entry; do
    remote_entry=${remote_entry//$'\r'/}
    [[ -z "$remote_entry" ]] && continue
    base_entry=$(strip_suffix "$remote_entry")
    if [[ -n ${brought_basenames["$base_entry"]+_} ]]; then
      continue
    fi
    case "$remote_entry" in
      *.hld.tar.gz|*.hld-tar-gz)
        compressed_lookup["$remote_entry"]=1
        ;;
      *.hld)
        [[ "$remote_entry" =~ \.tar\.gz$ ]] && continue
        [[ "$remote_entry" =~ -tar-gz$ ]] && continue
        uncompressed_lookup["$remote_entry"]=1
        ;;
      *)
        continue
        ;;
    esac
  done <<< "$remote_listing"
done

if (( ${#compressed_lookup[@]} == 0 && ${#uncompressed_lookup[@]} == 0 )); then
  log_info "No remote files matched the selected basenames; nothing to transfer."
  exit 0
fi

if [[ -n "$compressed_list_file" ]]; then
  rm -f "$compressed_list_file"
fi
compressed_list_file=""
if (( ${#compressed_lookup[@]} > 0 )); then
  compressed_list_file=$(mktemp)
  printf '%s\n' "${!compressed_lookup[@]}" | sort -u > "$compressed_list_file"
fi

if [[ -n "$uncompressed_list_file" ]]; then
  rm -f "$uncompressed_list_file"
fi
uncompressed_list_file=""
if (( ${#uncompressed_lookup[@]} > 0 )); then
  uncompressed_list_file=$(mktemp)
  printf '%s\n' "${!uncompressed_lookup[@]}" | sort -u > "$uncompressed_list_file"
fi

compressed_count=0
uncompressed_count=0
if [[ -s "$compressed_list_file" ]]; then
  compressed_count=$(wc -l < "$compressed_list_file" | tr -d '[:space:]')
fi
if [[ -s "$uncompressed_list_file" ]]; then
  uncompressed_count=$(wc -l < "$uncompressed_list_file" | tr -d '[:space:]')
fi

if $random_mode; then
  log_info "Fetching HLD files for MINGO0${station} using randomly selected basename ${selected_bases[0]}"
elif $newest_mode; then
  log_info "Fetching HLD files for MINGO0${station} using newest basename ${selected_bases[0]} (--newest)"
else
  log_info "Fetching HLD files for MINGO0${station} for range ${start}-${end}"
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
    base=$(strip_suffix "$brought_file")
    if [[ -z "$base" ]]; then
      continue
    fi
    if [[ -n ${brought_basenames["$base"]+_} ]]; then
      continue
    fi
    printf '%s,%s\n' "$base" "$csv_timestamp" >> "$brought_csv"
    brought_basenames["$base"]=1
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
