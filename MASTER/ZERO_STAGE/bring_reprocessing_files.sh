#!/bin/bash
# Usage: ./bring_reprocessing_files.sh <station> <start_date> <end_date>
# Example: ./bring_reprocessing_files.sh 1 250626 250701

if [ $# -ne 3 ]; then
    echo "Usage: $0 <station> <start_date YYMMDD> <end_date YYMMDD>"
    exit 1
fi

station=$1
start=$2
end=$3

start_DOY=$(date -d "20${start:0:2}-${start:2:2}-${start:4:2}" +%y%j)
end_DOY=$(date -d "20${end:0:2}-${end:2:2}-${end:4:2}" +%y%j)

compressed_directory=/home/mingo/DATAFLOW_v3/STATIONS/MINGO0${station}/ZERO_STAGE/COMPRESSED_HLDS
mkdir -p "$compressed_directory"

echo "Fetching HLD files for MINGO0$station between $start_DOY and $end_DOY..."

for pattern in "mi0${station}" "minI${station}"; do
    ssh backuplip "ls /local/experiments/MINGOS/MINGO0${station}/${pattern}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null || continue
    for file in $(ssh backuplip "ls /local/experiments/MINGOS/MINGO0${station}/${pattern}*{$start_DOY..$end_DOY}*.hld*" 2>/dev/null); do
        echo "Transferring $file ..."
        scp "backuplip:$file" "$compressed_directory/" || echo "Failed: $file"
    done
done

echo "Download completed."
