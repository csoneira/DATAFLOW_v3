#!/bin/bash
# Run dictionary inference on MINGO01 TASK_1 data.
# Usage: bash run_inference.sh [config.json]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.json}"

echo "============================================"
echo " INFERENCE_DICTIONARY_ON_DATA"
echo " Running inference with config: $CONFIG"
echo "============================================"

python3 "$SCRIPT_DIR/infer_from_dictionary.py" --config "$CONFIG"
