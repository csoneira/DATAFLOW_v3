#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTION_CONFIG="${1:-$SCRIPT_DIR/configs/selection.yaml}"
GATE_CONFIG="${2:-$SCRIPT_DIR/configs/gates.yaml}"

cd "$SCRIPT_DIR"

echo "Running collector with:"
echo "  selection config: $SELECTION_CONFIG"
echo "  gate config: $GATE_CONFIG"
python3 -m src.main \
  --selection-config "$SELECTION_CONFIG" \
  --gate-config "$GATE_CONFIG"

echo
echo "Running diagnostics and plotting with:"
echo "  selection config: $SELECTION_CONFIG"
echo "  gate config: $GATE_CONFIG"
python3 -m src.diagnostics_main \
  --selection-config "$SELECTION_CONFIG" \
  --gate-config "$GATE_CONFIG"

echo
echo "STEP_2_DETOUR pipeline finished."
