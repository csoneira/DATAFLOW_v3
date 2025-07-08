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
base_dir=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE
compressed_directory="${base_dir}/COMPRESSED_HLDS"
uncompressed_directory="${base_dir}/UNCOMPRESSED_HLDS"
mkdir -p "$compressed_directory" "$uncompressed_directory"

echo "Fetching HLD files for MINGO0$station between $start_DOY and $end_DOY..."
echo

##############################################################################
# Transfer loop
##############################################################################
for pattern in "mi0${station}" "minI${station}"; do
  echo "Pattern: $pattern"

  # -------------------------------------------------------------------------
  # Build exclusion list – filenames already present anywhere in ZERO_STAGE
  # -------------------------------------------------------------------------
  exclude_file=$(mktemp)
  find "$base_dir" -type f -name "*.hld*" -printf "%f\n" > "$exclude_file"

  remote_dir="/local/experiments/MINGOS/MINGO0${station}/"

  for doy in $(seq "$start_DOY" "$end_DOY"); do
    echo "  DOY $doy"

    # -------------------- 1. COMPRESSED  (.hld.tar.gz | .hld-tar-gz) --------
    rsync -avz --progress \
      --include="${pattern}*${doy}*.hld*.tar.gz" \
      --include="${pattern}*${doy}*.hld-tar-gz" \
      --exclude-from="$exclude_file" \
      --exclude='*' \
      --ignore-existing --no-compress \
      "backuplip:${remote_dir}" "$compressed_directory/"

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
      "backuplip:${remote_dir}" "$uncompressed_directory/"

    # touch freshly-downloaded uncompressed files
    find "$uncompressed_directory" -type f -name "${pattern}*${doy}*.hld" -exec touch {} +
  done

  rm -f "$exclude_file"
done

echo
echo '------------------------------------------------------'
echo "bring_reprocessing_files.sh completed on: $(date '+%Y-%m-%d %H:%M:%S')"
echo '------------------------------------------------------'
