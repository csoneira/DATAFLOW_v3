#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: run_simulation_plots.sh [--parallel] [--help] [ARGS...]
Runs:
  - MESH: MESH/plot_param_mesh.py
  - TIME: SIMULATION_TIME/plot_simulation_time.py

Options:
  -p, --parallel    run both scripts in parallel
  -h, --help        show this help

Any extra ARGS are forwarded to both Python scripts.
USAGE
}

PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--parallel) PARALLEL=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_SCRIPT="$SCRIPT_DIR/MESH/plot_param_mesh.py"
TIME_SCRIPT="$SCRIPT_DIR/SIMULATION_TIME/plot_simulation_time.py"
PYTHON="$(command -v python3 || command -v python || true)"

if [[ -z "$PYTHON" ]]; then
  echo "Error: python3 (or python) not found in PATH" >&2
  exit 1
fi

for f in "$MESH_SCRIPT" "$TIME_SCRIPT"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: required script not found: $f" >&2
    exit 2
  fi
done

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOG_DIR/run_simulation_plots-$TIMESTAMP.log"

echo "Start: $(date)" | tee -a "$LOGFILE"
echo "Using PYTHON: $PYTHON" | tee -a "$LOGFILE"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}" | tee -a "$LOGFILE"
else
  echo "Extra args: <none>" | tee -a "$LOGFILE"
fi

run_one() {
  local name="$1"; shift
  local script="$1"; shift
  echo "=== [$name] $(date) ===" | tee -a "$LOGFILE"
  cd "$SCRIPT_DIR"
  if "$PYTHON" "$script" "$@" 2>&1 | tee -a "$LOGFILE"; then
    status=0
  else
    status=${PIPESTATUS[0]:-1}
  fi
  echo "=== [$name] exit: $status ===" | tee -a "$LOGFILE"
  return $status
}

if [[ "$PARALLEL" == true ]]; then
  run_one "mesh" "$MESH_SCRIPT" "${EXTRA_ARGS[@]}" &
  PID_MESH=$!
  run_one "time" "$TIME_SCRIPT" "${EXTRA_ARGS[@]}" &
  PID_TIME=$!
  wait $PID_MESH; STATUS_MESH=$?
  wait $PID_TIME; STATUS_TIME=$?
  echo "mesh:$STATUS_MESH time:$STATUS_TIME" | tee -a "$LOGFILE"
  exit $(( STATUS_MESH || STATUS_TIME ))
else
  run_one "mesh" "$MESH_SCRIPT" "${EXTRA_ARGS[@]}"
  run_one "time" "$TIME_SCRIPT" "${EXTRA_ARGS[@]}"
fi

echo "Finished: $(date)" | tee -a "$LOGFILE"
exit 0
