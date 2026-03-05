#!/usr/bin/env bash
set -euo pipefail

QA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_all_steps] Starting QUALITY_ASSURANCE runners"

mapfile -t step_runners < <(find "$QA_DIR" -maxdepth 2 -mindepth 2 -type f -path "$QA_DIR/STEP_*/run_tasks_*.sh" | sort)

if (( ${#step_runners[@]} == 0 )); then
  echo "[run_all_steps] No STEP_*/run_tasks_*.sh scripts found in $QA_DIR"
  exit 0
fi

for runner in "${step_runners[@]}"; do
  echo "[run_all_steps] Running ${runner#$QA_DIR/}"
  bash "$runner"
done

echo "[run_all_steps] Completed QUALITY_ASSURANCE runners"
