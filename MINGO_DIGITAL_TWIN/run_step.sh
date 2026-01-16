#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <step_number|all|final|--from> [--no-plots|--plot-only|--final]"
}

NO_PLOTS=""
PLOT_ONLY=""
FINAL_STEP=""
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--no-plots" ]]; then
    NO_PLOTS="--no-plots"
  elif [[ "$arg" == "--plot-only" ]]; then
    PLOT_ONLY="--plot-only"
  elif [[ "$arg" == "--final" ]]; then
    FINAL_STEP="--final"
  else
    ARGS+=("$arg")
  fi
done

if [[ ${#ARGS[@]} -lt 1 ]]; then
  usage
  exit 1
fi

STEP="${ARGS[0]}"
DT="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "$NO_PLOTS" && -n "$PLOT_ONLY" ]]; then
  echo "Error: --no-plots and --plot-only cannot be used together."
  exit 1
fi

run_step() {
  case "$1" in
    1) python3 "$DT/MASTER_STEPS/STEP_1/step_1_blank_to_generated.py" --config "$DT/MASTER_STEPS/STEP_1/config_step_1_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    2) python3 "$DT/MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py" --config "$DT/MASTER_STEPS/STEP_2/config_step_2_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    3) python3 "$DT/MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py" --config "$DT/MASTER_STEPS/STEP_3/config_step_3_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    4) python3 "$DT/MASTER_STEPS/STEP_4/step_4_hit_to_measured.py" --config "$DT/MASTER_STEPS/STEP_4/config_step_4_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    5) python3 "$DT/MASTER_STEPS/STEP_5/step_5_measured_to_triggered.py" --config "$DT/MASTER_STEPS/STEP_5/config_step_5_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    6) python3 "$DT/MASTER_STEPS/STEP_6/step_6_triggered_to_timing.py" --config "$DT/MASTER_STEPS/STEP_6/config_step_6_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    7) python3 "$DT/MASTER_STEPS/STEP_7/step_7_timing_to_calibrated.py" --config "$DT/MASTER_STEPS/STEP_7/config_step_7_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    8) python3 "$DT/MASTER_STEPS/STEP_8/step_8_calibrated_to_threshold.py" --config "$DT/MASTER_STEPS/STEP_8/config_step_8_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    9) python3 "$DT/MASTER_STEPS/STEP_9/step_9_threshold_to_trigger.py" --config "$DT/MASTER_STEPS/STEP_9/config_step_9_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    10) python3 "$DT/MASTER_STEPS/STEP_10/step_10_triggered_to_jitter.py" --config "$DT/MASTER_STEPS/STEP_10/config_step_10_physics.yaml" $NO_PLOTS $PLOT_ONLY ;;
    final)
      if [[ -z "$FINAL_STEP" ]]; then
        echo "Error: final step requires --final flag."
        exit 1
      fi
      python3 "$DT/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py" --config "$DT/MASTER_STEPS/STEP_FINAL/config_step_final_physics.yaml" $NO_PLOTS $PLOT_ONLY
      ;;
    *)
      echo "Unknown step: $1"
      exit 1
      ;;
  esac
}

case "$STEP" in
  all)
    start_time=$(date +%s)
    for step in $(seq 1 10); do
      run_step "$step"
    done
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "All steps completed in ${elapsed}s"
    ;;
  --from)
    start_step="${ARGS[1]:-}"
    if [[ -z "$start_step" ]]; then
      echo "Usage: $0 --from <step_number> [--no-plots]"
      exit 1
    fi
    start_time=$(date +%s)
    for step in $(seq "$start_step" 10); do
      run_step "$step"
    done
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Steps ${start_step}-12 completed in ${elapsed}s"
    ;;
  *)
    run_step "$STEP"
    ;;
esac
