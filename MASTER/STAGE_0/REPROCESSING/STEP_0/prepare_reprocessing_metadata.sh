#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# prepare_reprocessing_metadata.sh
#   Refresh and clean the STEP_0 metadata CSVs for STAGE_0 reprocessing.
#   Stores raw listings and filtered versions plus optional diagnostic plots.
# ---------------------------------------------------------------------------

set -euo pipefail

log_info() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'USAGE'
Usage:
  prepare_reprocessing_metadata.sh <station> [--refresh-metadata|-m] [--plot]

Options:
  -h, --help            Show this help message and exit.
  -m, --refresh-metadata
                        Fetch the latest remote listing and update the metadata CSVs.
      --plot            Generate the metadata delta histogram PDF using the
                        current clean metadata CSV.
USAGE
  exit 1
}

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
  usage
fi

if (( $# < 1 )); then
  usage
fi

station="$1"
shift

refresh_metadata=false
plot_hist=false

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
    -h|--help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

if ! $refresh_metadata && ! $plot_hist; then
  usage
fi

station_directory="$HOME/DATAFLOW_v3/STATIONS/MINGO0${station}"
step0_directory="${station_directory}/STAGE_0/REPROCESSING/STEP_0"
metadata_directory="${step0_directory}/METADATA"
input_directory="${step0_directory}/INPUT_FILES"
output_directory="${step0_directory}/OUTPUT_FILES"
plots_directory="${step0_directory}/PLOTS"
mkdir -p "$metadata_directory" "$input_directory" "$output_directory" "$plots_directory"

remote_user="rpcuser"
remote_dir="/local/experiments/MINGOS/MINGO0${station}/"
printf -v remote_dir_escaped '%q' "$remote_dir"

raw_csv="${input_directory}/remote_database_${station}.csv"
clean_csv="${output_directory}/clean_remote_database_${station}.csv"
metadata_tracking_csv="${metadata_directory}/metadata_refresh_history.csv"
metadata_tracking_header="executed_at,remote_file_count,clean_file_count"
csv_header="basename"

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

filter_close_metadata_entries() {
  local csv_file="$1"
  if [[ ! -s "$csv_file" ]]; then
    return 0
  fi

  local header
  if ! header=$(head -n1 "$csv_file"); then
    return 1
  fi

  local -a basenames=()
  while IFS= read -r line; do
    line=${line//$'\r'/}
    [[ -z "$line" ]] && continue
    basenames+=("$line")
  done < <(tail -n +2 "$csv_file")

  local total=${#basenames[@]}
  if (( total == 0 )); then
    return 0
  fi

  local -a keep_flags
  keep_flags=()
  for ((i = 0; i < total; i++)); do
    keep_flags[i]=1
  done

  for ((i = 0; i < total - 1; i++)); do
    local prev="${basenames[i]}"
    local next="${basenames[i + 1]}"
    [[ -z "$prev" || -z "$next" ]] && continue
    local prev_epoch next_epoch
    if ! prev_epoch=$(basename_to_epoch "$prev"); then
      continue
    fi
    if ! next_epoch=$(basename_to_epoch "$next"); then
      continue
    fi
    local delta=$((next_epoch - prev_epoch))
    if (( delta >= 0 && delta < 600 )); then
      keep_flags[i]=0
    fi
  done

  local tmp_filtered
  tmp_filtered=$(mktemp) || return 1
  printf '%s\n' "$header" > "$tmp_filtered"
  local removed=0
  for ((i = 0; i < total; i++)); do
    if (( keep_flags[i] == 1 )); then
      printf '%s\n' "${basenames[i]}" >>"$tmp_filtered"
    else
      ((removed++))
    fi
  done

  if (( removed > 0 )); then
    mv "$tmp_filtered" "$csv_file"
    log_info "Metadata filter removed ${removed} basename(s) closer than 10 minutes apart."
  else
    rm -f "$tmp_filtered"
  fi
}

ensure_metadata_tracking_csv() {
  if [[ ! -f "$metadata_tracking_csv" || ! -s "$metadata_tracking_csv" ]]; then
    printf '%s\n' "$metadata_tracking_header" > "$metadata_tracking_csv"
  fi
}

record_metadata_history() {
  local raw_total="$1"
  local clean_total="$2"
  ensure_metadata_tracking_csv
  printf '%s,%s,%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$raw_total" "$clean_total" >> "$metadata_tracking_csv"
}

plot_metadata_time_deltas() {
  local csv_file="$1"
  local station_label="$2"
  local plot_dir="$3"

  if [[ ! -s "$csv_file" ]]; then
    log_info "Metadata CSV ${csv_file} is empty; skipping histogram generation."
    return 1
  fi

  mkdir -p "$plot_dir" || return 1
  local plot_path="${plot_dir}/metadata_time_differences_${station_label}.pdf"

  if PLOT_CSV="$csv_file" PLOT_PATH="$plot_path" STATION_LABEL="$station_label" python3 <<'PY'
import csv
import datetime as dt
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = os.environ["PLOT_CSV"]
plot_path = os.environ["PLOT_PATH"]
station_label = os.environ.get("STATION_LABEL", "")

basenames = []
with open(csv_path, newline="") as handle:
  reader = csv.reader(handle)
  next(reader, None)
  for row in reader:
    if not row:
      continue
    base = row[0].strip()
    if base:
      basenames.append(base)

pattern = re.compile(r"([0-9]{11})$")
timestamps = []
for base in basenames:
  match = pattern.search(base)
  if not match:
    continue
  key = match.group(1)
  yy = int(key[0:2])
  doy = int(key[2:5])
  hh = int(key[5:7])
  mm = int(key[7:9])
  ss = int(key[9:11])
  if doy < 1 or doy > 366:
    continue
  year = 2000 + yy
  ts = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
  timestamps.append(ts)

timestamps.sort()

title = f"Metadata time differences for {station_label}" if station_label else "Metadata time differences"

def render_message(message: str) -> None:
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
  ax.set_axis_off()
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(plot_path, format="pdf")
  plt.close(fig)

if len(timestamps) < 2:
  render_message("Not enough metadata entries\nto compute time differences.")
else:
  diffs = []
  for idx in range(len(timestamps) - 1):
    delta = timestamps[idx + 1] - timestamps[idx]
    minutes = delta.total_seconds() / 60.0
    if 0 <= minutes <= 120:
      diffs.append(minutes)

  if not diffs:
    render_message("No metadata time differences\nwithin 0-120 minutes.")
  else:
    bins = "auto" if len(diffs) > 1 else 1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diffs, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.set_xlim(0, 120)
    ax.set_title(title)
    ax.set_xlabel("Minutes between consecutive files (0-120)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, format="pdf")
    plt.close(fig)
PY
  then
    log_info "Histogram of metadata time differences saved to ${plot_path}"
    return 0
  else
    log_info "Failed to create metadata histogram at ${plot_path}" >&2
    return 1
  fi
}

refresh_metadata_csv() {
  log_info "Refreshing metadata CSV for MINGO0${station} from remote listing..."

  local tmp_listing
  tmp_listing=$(mktemp) || return 1
  if ! ssh -o BatchMode=yes "${remote_user}@backuplip" "cd ${remote_dir_escaped} && ls -1" > "$tmp_listing"; then
    log_info "Warning: unable to refresh metadata; could not list remote directory ${remote_user}@backuplip:${remote_dir}" >&2
    rm -f "$tmp_listing"
    return 1
  fi

  declare -A refreshed_bases=()
  local remote_entry
  while IFS= read -r remote_entry; do
    remote_entry=${remote_entry//$'\r'/}
    [[ -z "$remote_entry" ]] && continue
    [[ $remote_entry =~ \.hld ]] || continue
    local base
    base=$(strip_suffix "$remote_entry")
    [[ -z "$base" ]] && continue
    [[ $base =~ ^(mi|minI) ]] || continue
    refreshed_bases["$base"]=1
  done < "$tmp_listing"
  rm -f "$tmp_listing"

  if (( ${#refreshed_bases[@]} == 0 )); then
    log_info "Metadata refresh found no basenames; existing CSV left unchanged."
    return 1
  fi

  local tmp_sorted
  tmp_sorted=$(mktemp) || return 1
  printf '%s\n' "${!refreshed_bases[@]}" | sort > "$tmp_sorted"

  {
    printf '%s\n' "$csv_header"
    cat "$tmp_sorted"
  } > "$raw_csv"

  cp "$raw_csv" "$clean_csv"
  if ! filter_close_metadata_entries "$clean_csv"; then
    log_info "Metadata CSV filter could not be applied; continuing with unfiltered entries."
  fi

  local raw_count clean_count
  raw_count=$(tail -n +2 "$raw_csv" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')
  clean_count=$(tail -n +2 "$clean_csv" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')
  log_info "Metadata CSV refreshed: raw=${raw_count} entries, clean=${clean_count} entries."

  record_metadata_history "$raw_count" "$clean_count"

  rm -f "$tmp_sorted"
  return 0
}

if $refresh_metadata; then
  if refresh_metadata_csv; then
    log_info "Metadata refresh completed."
  else
    log_info "Metadata refresh could not update the CSV; continuing with existing entries."
  fi
fi

if $plot_hist; then
  if [[ ! -s "$clean_csv" ]]; then
    log_info "Clean metadata CSV ${clean_csv} is missing; run with --refresh-metadata first." >&2
    exit 1
  fi
  plot_metadata_time_deltas "$clean_csv" "MINGO0${station}" "$plots_directory" || true
fi

log_info "STEP_0 metadata preparation finished."
