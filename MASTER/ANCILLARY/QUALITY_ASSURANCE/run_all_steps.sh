#!/usr/bin/env bash
set -euo pipefail

QA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_all_steps] Running QUALITY_ASSURANCE pipeline and final aggregation"
python3 "$QA_DIR/orchestrate_quality_assurance.py"
echo "[run_all_steps] Completed QUALITY_ASSURANCE pipeline"
