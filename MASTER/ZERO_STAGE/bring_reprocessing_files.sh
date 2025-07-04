#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bring_reprocessing_files.sh
#
# Usage examples
#   ./bring_reprocessing_files.sh <station> YYMMDD YYMMDD   # explicit range
#   ./bring_reprocessing_files.sh <station> --random        # random day
#   ./bring_reprocessing_files.sh <station> -r              # idem
# ---------------------------------------------------------------------------

set -euo pipefail

##############################################################################
# Parse arguments
##############################################################################
if (( $# < 2 )); then
  echo "Usage: $0 <station> YYMMDD YYMMDD | --random|-r"
  exit 1
fi

station="$1"
shift                                   # consume station

if [[ ${1:-} =~ ^(--random|-r)$ ]]; then
  # -------------------------------------------------------------------------
  # RANDOM-DAY MODE  (one date chosen uniformly between 2023-07-01 and today-5d)
  # -------------------------------------------------------------------------
  epoch_start=$(date -d '2023-07-01 00:00:00' +%s)
  epoch_end=$(date -d 'today -5 days 00:00:00' +%s)

  rand_epoch=$(shuf -i "${epoch_start}-${epoch_end}" -n1)
  rand_ymd=$(date -d "@${rand_epoch}" +%y%m%d)

  start="${rand_ymd}"
  end="${rand_ymd}"
  echo "Random day selected: $rand_ymd"
else
  # -------------------------------------------------------------------------
  # EXPLICIT YYYYMMDD RANGE
  # -------------------------------------------------------------------------
  if (( $# != 2 )); then
    echo "Usage: $0 <station> YYMMDD YYMMDD | --random|-r"
    exit 1
  fi
  start="$1"
  end="$2"
fi

##############################################################################
# Convert to day-of-year format (YYJJJ)
##############################################################################
start_DOY=$(date -d "20${start:0:2}-${start:2:2}-${start:4:2}" +%y%j)
end_DOY=$(  date -d "20${end:0:2}-${end:2:2}-${end:4:2}"   +%y%j)

##############################################################################
# Target directory preparation
##############################################################################
compressed_directory=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE/COMPRESSED_HLDS
mkdir -p "$compressed_directory"

echo "Fetching HLD files for MINGO0$station between $start_DOY and $end_DOY..."

##############################################################################
# Transfer loop
##############################################################################
for pattern in "mi0${station}" "minI${station}"; do
  ssh backuplip "ls /local/experiments/MINGOS/MINGO0${station}/${pattern}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null || continue
  for file in $(ssh backuplip "ls /local/experiments/MINGOS/MINGO0${station}/${pattern}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null); do
    echo "Transferring $file ..."
    scp "backuplip:$file" "$compressed_directory/" || echo "Failed: $file"
    # Modify the time metadata of modification time of the file to the current time
    touch "$compressed_directory/$(basename "$file")"
  done
done

echo "Download completed."
