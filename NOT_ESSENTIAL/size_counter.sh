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
total_dirs=0
total_expected_dirs=0
total_expected_bytes=0

expected_map="$(python3 - <<'PY'
import pandas as pd
from pathlib import Path

mesh_path = Path("/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv")
if not mesh_path.exists():
    raise SystemExit(0)
mesh = pd.read_csv(mesh_path)
if "done" in mesh.columns:
    mesh = mesh[mesh["done"].fillna(0).astype(int) != 1]
step1 = mesh["step_1_id"].astype(str).nunique() if "step_1_id" in mesh.columns else 0
step2 = mesh["step_2_id"].astype(str).nunique() if "step_2_id" in mesh.columns else 0
step3 = mesh["step_3_id"].astype(str).nunique() if "step_3_id" in mesh.columns else 0
counts = {
    "STEP_1_TO_2": step1,
    "STEP_2_TO_3": step1 * step2,
    "STEP_3_TO_4": step1 * step2 * step3,
}
total = counts["STEP_3_TO_4"]
for step in range(4, 11):
    counts[f"STEP_{step}_TO_{step + 1}"] = total
for key, val in counts.items():
    print(f"{key}={val}")
PY
)"

printf "\n%-20s %14s %20s %10s\n" "STEP" "DIRS NOW/EXP" "SIZE GB NOW/EXP" "% DONE"
printf "%-20s %14s %20s %10s\n" "--------------------" "--------------" "--------------------" "----------"

for name in "${sorted_names[@]}"; do
  dir="$intersteps_dir/$name"
  dir_count=$(find "$dir" -maxdepth 1 -type d -name 'SIM_RUN_*' | wc -l | tr -d ' ')
  size_bytes=$(du -sb "$dir" | awk '{print $1}')
  size_gb=$(awk -v bytes="$size_bytes" 'BEGIN { printf "%.3f", bytes / (1024^3) }')
  expected_dirs=0
  if [[ -n "$expected_map" ]]; then
    expected_dirs=$(printf '%s\n' "$expected_map" | awk -F= -v key="$name" '$1==key{print $2}')
    expected_dirs=${expected_dirs:-0}
  fi
  if [[ "$dir_count" -gt 0 ]]; then
    avg_bytes=$(awk -v bytes="$size_bytes" -v dirs="$dir_count" 'BEGIN { printf "%.0f", bytes / dirs }')
    expected_bytes=$(awk -v avg="$avg_bytes" -v exp="$expected_dirs" 'BEGIN { printf "%.0f", avg * exp }')
  else
    expected_bytes=0
  fi
  size_expected_gb=$(awk -v bytes="$expected_bytes" 'BEGIN { printf "%.3f", bytes / (1024^3) }')
  if [[ "$expected_dirs" -gt 0 ]]; then
    pct=$(awk -v now="$dir_count" -v exp="$expected_dirs" 'BEGIN { printf "%.1f", (now/exp)*100 }')
  else
    pct="0.0"
  fi
  total_bytes=$((total_bytes + size_bytes))
  total_expected_bytes=$((total_expected_bytes + expected_bytes))
  total_dirs=$((total_dirs + dir_count))
  total_expected_dirs=$((total_expected_dirs + expected_dirs))
  printf "%-20s %6d/%-7d %9s/%-9s %9s%%\n" "$name" "$dir_count" "$expected_dirs" "$size_gb" "$size_expected_gb" "$pct"
done

total_gb=$(awk -v bytes="$total_bytes" 'BEGIN { printf "%.3f", bytes / (1024^3) }')
total_expected_gb=$(awk -v bytes="$total_expected_bytes" 'BEGIN { printf "%.3f", bytes / (1024^3) }')
if [[ "$total_expected_dirs" -gt 0 ]]; then
  total_pct=$(awk -v now="$total_dirs" -v exp="$total_expected_dirs" 'BEGIN { printf "%.1f", (now/exp)*100 }')
else
  total_pct="0.0"
fi
printf "%-20s %14s %20s %10s\n" "--------------------" "--------------" "--------------------" "----------"
printf "%-20s %6d/%-7d %9s/%-9s %9s%%\n\n" "TOTAL" "$total_dirs" "$total_expected_dirs" "$total_gb" "$total_expected_gb" "$total_pct"
