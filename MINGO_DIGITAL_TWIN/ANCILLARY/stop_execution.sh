#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. Canonical location:
#   MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/stop_execution.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../ORCHESTRATOR/maintenance/stop_execution.sh" "$@"
