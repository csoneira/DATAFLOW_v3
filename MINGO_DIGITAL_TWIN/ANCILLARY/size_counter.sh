#!/bin/bash

set -euo pipefail

intersteps_dir="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS"

if [[ ! -d "$intersteps_dir" ]]; then
  echo "Directory not found: $intersteps_dir" >&2
  exit 1
fi

shopt -s nullglob
step_dirs=("$intersteps_dir"/STEP_*_TO_*/)
shopt -u nullglob

if [[ ${#step_dirs[@]} -eq 0 ]]; then
  echo "No STEP_X_TO_Y directories found in $intersteps_dir"
  exit 0
fi

step_names=()
for dir in "${step_dirs[@]}"; do
  step_names+=("$(basename "$dir")")
done

IFS=$'\n' read -r -d '' -a sorted_names < <(printf '%s\n' "${step_names[@]}" | sort -V && printf '\0')

total_bytes=0
total_files=0

printf "\n%-20s %12s %12s\n" "STEP" "FILES" "SIZE (GB)"
printf "%-20s %12s %12s\n" "--------------------" "------------" "------------"

for name in "${sorted_names[@]}"; do
  dir="$intersteps_dir/$name"
  file_count=$(find "$dir" -type f | wc -l | tr -d ' ')
  size_bytes=$(du -sb "$dir" | awk '{print $1}')
  size_gb=$(awk -v bytes="$size_bytes" 'BEGIN { printf "%.3f", bytes / (1024^3) }')
  total_bytes=$((total_bytes + size_bytes))
  total_files=$((total_files + file_count))
  printf "%-20s %12d %12s\n" "$name" "$file_count" "$size_gb"
done

total_gb=$(awk -v bytes="$total_bytes" 'BEGIN { printf "%.3f", bytes / (1024^3) }')
printf "%-20s %12s %12s\n" "--------------------" "------------" "------------"
printf "%-20s %12d %12s\n\n" "TOTAL" "$total_files" "$total_gb"
