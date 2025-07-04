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

# for pattern in "mi0${station}" "minI${station}"; do
#   echo ""
#   echo "Listing all the files"
#   ssh backuplip "ls /local/experiments/MINGOS/MINGO0${station}/${pattern}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null || continue
#   echo ""

#   while IFS= read -r file; do
#     echo "Transferring $file ..."
#     if scp "backuplip:$file" "$compressed_directory/"; then
#     # if scp backuplip:"'$file'"  "$compressed_directory/"; then
#       # update mtime to now
#       touch "$compressed_directory/$(basename "$file")"
#     else
#       echo "Failed: $file"
#     fi
#   done < <(ssh backuplip "ls -1 /local/experiments/MINGOS/MINGO0${station}/${pattern}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null)
# done



for pattern in "mi0${station}" "minI${station}"; do
  echo "Fetching pattern: $pattern"

  # Build exclude list from local files to avoid re-transfer
  exclude_file=$(mktemp)
  find /home/mingo/DATAFLOW_v3/STATIONS/MINGO0*/ZERO_STAGE -type f -name "*.hld*" -printf "%f\n" > "$exclude_file"

  # Print the exclude list contents
  echo "Excluding files listed in: $exclude_file"
  echo "Excluding files already present in ZERO_STAGE/COMPRESSED_HLDS"
  cat "$exclude_file"
  echo ""

  echo "Syncing files for station MINGO0${station} from $start_DOY to $end_DOY"

  for doy in $(seq "$start_DOY" "$end_DOY"); do
    echo "Syncing files for DOY: $doy"

    # Remote directory
    remote_dir="/local/experiments/MINGOS/MINGO0${station}/"

    # Use rsync to pull only the matching files
    rsync -avz --progress \
      --include="${pattern}*${doy}*.hld*" \
      --exclude-from="$exclude_file" \
      --exclude='*' \
      --ignore-existing \
      --no-compress \
      "backuplip:${remote_dir}" "$compressed_directory/"
  done

  rm -f "$exclude_file"
done




echo "Download completed."
