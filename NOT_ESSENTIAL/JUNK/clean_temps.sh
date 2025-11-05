#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper to retain backwards compatibility with the legacy cleaner name.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/clean_completed.sh" --select temps "$@"
