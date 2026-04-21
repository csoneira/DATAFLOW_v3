#!/usr/bin/env bash
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_DIR="$(cd "$STEP_DIR/.." && pwd)"

python3 "$QA_DIR/run_family_step.py" "$STEP_DIR"
