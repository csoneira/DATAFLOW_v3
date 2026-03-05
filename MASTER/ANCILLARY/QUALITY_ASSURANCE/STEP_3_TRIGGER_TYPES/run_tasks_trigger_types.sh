#!/usr/bin/env bash
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_tasks_trigger_types] Starting $(basename "$STEP_DIR")"

mapfile -t task_dirs < <(find "$STEP_DIR" -maxdepth 1 -mindepth 1 -type d -name 'TASK_*' | sort)

if (( ${#task_dirs[@]} == 0 )); then
  echo "[run_tasks_trigger_types] No TASK_* directories found in $STEP_DIR"
  exit 0
fi

for task_dir in "${task_dirs[@]}"; do
  mapfile -t py_scripts < <(find "$task_dir" -maxdepth 1 -type f -name '*.py' | sort)
  if (( ${#py_scripts[@]} == 0 )); then
    echo "[run_tasks_trigger_types] No Python scripts found in $(basename "$task_dir")"
    continue
  fi

  for py_script in "${py_scripts[@]}"; do
    echo "[run_tasks_trigger_types] Running ${py_script#$STEP_DIR/}"
    python3 "$py_script"
  done
done

echo "[run_tasks_trigger_types] Completed $(basename "$STEP_DIR")"
