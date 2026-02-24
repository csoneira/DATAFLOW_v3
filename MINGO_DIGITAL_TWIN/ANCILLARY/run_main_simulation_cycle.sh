#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint. Canonical location:
#   MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_main_simulation_cycle.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../ORCHESTRATOR/core/run_main_simulation_cycle.sh" "$@"
