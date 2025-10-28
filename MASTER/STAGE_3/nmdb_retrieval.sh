#!/bin/bash

# Put in crontab the following line to run this script every day at 2 AM:
# 0 2 * * * /bin/bash /home/mingo/DATAFLOW_v3/MASTER/STAGE_3/nmdb_retrieval.sh >> /home/mingo/DATAFLOW_v3/MASTER/STAGE_3/update_log.txt 2>&1

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then
    cat <<'EOF'
nmdb_retrieval.sh
Downloads and maintains the NMDB combined data CSV used in STAGE_3 analyses.

Usage:
  nmdb_retrieval.sh

Options:
  -h, --help    Show this help message and exit.

The script fetches data using URLs listed in address.conf, replaces the last
100 days within nmdb_combined.csv, and logs the outcome under STAGE_3.
EOF
    exit 0
fi

# ==== CONFIGURATION ====
BASE_DIR="$HOME/DATAFLOW_v3/MASTER/STAGE_3"
CSV_FILE="$BASE_DIR/nmdb_combined.csv"
TMP_FILE="$BASE_DIR/nmdb_tmp.csv"
ADDRESS_FILE="$BASE_DIR/address.conf"

mkdir -p "$BASE_DIR"

# ==== DOWNLOAD DATA ====
wget -np -q -O "$TMP_FILE" -i "$ADDRESS_FILE"

# Abort if the download failed or file is empty
if [[ ! -s "$TMP_FILE" ]]; then
    echo "[ERROR] Download failed or empty file: $(date)" >> "$BASE_DIR/update_error.log"
    exit 1
fi

# ==== INITIALIZE CSV IF NOT EXISTS ====
if [[ ! -f "$CSV_FILE" ]]; then
    mv "$TMP_FILE" "$CSV_FILE"
    echo "[INFO] CSV initialized: $(date)" >> "$BASE_DIR/update_log.txt"
    exit 0
fi

# ==== REPLACE LAST 100 DAYS ====
# Get ISO timestamp 100 days ago
CUTOFF_DATE=$(date -u --date="100 days ago" +%Y-%m-%dT%H:%M:%S)

# Extract header lines (lines starting with '#')
grep '^#' "$CSV_FILE" > "$CSV_FILE.tmp"

# Keep rows earlier than cutoff
awk -F';' -v cutoff="$CUTOFF_DATE" '($1 < cutoff && $1 !~ /^#/) || /^#/' "$CSV_FILE" >> "$CSV_FILE.tmp"

# Append all new data
awk '($1 !~ /^#/) && NF > 1' "$TMP_FILE" >> "$CSV_FILE.tmp"

# Replace original file
mv "$CSV_FILE.tmp" "$CSV_FILE"
rm -f "$TMP_FILE"

echo "[INFO] CSV updated with last 100 days replaced: $(date)" >> "$BASE_DIR/update_log.txt"
