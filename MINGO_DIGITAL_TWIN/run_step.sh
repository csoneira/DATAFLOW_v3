#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <step_number>"
  exit 1
fi

STEP="$1"
DT="$(cd "$(dirname "$0")" && pwd)"

case "$STEP" in
  1) python3 "$DT/STEP_1/step_1_blank_to_generated.py" --config "$DT/STEP_1/config_step_1.yaml" ;;
  2) python3 "$DT/STEP_2/step_2_generated_to_crossing.py" --config "$DT/STEP_2/config_step_2.yaml" ;;
  3) python3 "$DT/STEP_3/step_3_crossing_to_hit.py" --config "$DT/STEP_3/config_step_3.yaml" ;;
  4) python3 "$DT/STEP_4/step_4_hit_to_measured.py" --config "$DT/STEP_4/config_step_4.yaml" ;;
  5) python3 "$DT/STEP_5/step_5_measured_to_triggered.py" --config "$DT/STEP_5/config_step_5.yaml" ;;
  6) python3 "$DT/STEP_6/step_6_triggered_to_timing.py" --config "$DT/STEP_6/config_step_6.yaml" ;;
  7) python3 "$DT/STEP_7/step_7_timing_to_calibrated.py" --config "$DT/STEP_7/config_step_7.yaml" ;;
  8) python3 "$DT/STEP_8/step_8_calibrated_to_threshold.py" --config "$DT/STEP_8/config_step_8.yaml" ;;
  9) python3 "$DT/STEP_9/step_9_threshold_to_trigger.py" --config "$DT/STEP_9/config_step_9.yaml" ;;
  10) python3 "$DT/STEP_10/step_10_triggered_to_jitter.py" --config "$DT/STEP_10/config_step_10.yaml" ;;
  11) python3 "$DT/STEP_11/step_11_daq_to_detector_format.py" --config "$DT/STEP_11/config_step_11.yaml" ;;
  *)
    echo "Unknown step: $STEP"
    exit 1
    ;;
 esac
