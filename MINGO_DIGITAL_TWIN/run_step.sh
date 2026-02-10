#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_step.sh <step_number|all|final|from> [options]
  run_step.sh from <step_number> [options]
  run_step.sh -c|--continuous [options]

Options:
  --no-plots           Skip plot generation
  --plot-only          Only generate plots from existing outputs
  --loop               Repeat the selected run in a loop
  --force              Recompute even if SIM_RUN exists
  -c, --continuous     Run "all" in a loop with a lock to prevent overlaps (implies --no-plots)
  -fc, --force-continuous  Terminate the active continuous run and start a new one
  -h, --help           Show this help and exit

Notes:
  -c/--continuous implies "all", "--loop", and "--no-plots" and uses a lock in /tmp.
  --force-continuous is only valid with -c/--continuous.
EOF
}

log_ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
  printf '%s [INFO] [run_step] %s\n' "$(log_ts)" "$*"
}

log_warn() {
  printf '%s [WARN] [run_step] %s\n' "$(log_ts)" "$*" >&2
}

log_error() {
  printf '%s [ERROR] [run_step] %s\n' "$(log_ts)" "$*" >&2
}

simulation_time_csv_path() {
  printf '%s/PLOTTERS/SIMULATION_TIME/simulation_execution_times.csv' "$DT"
}

ensure_simulation_time_csv() {
  local csv_path
  local header
  local current_header
  local backup
  csv_path="$(simulation_time_csv_path)"
  header='timestamp_utc,elapsed_seconds'
  mkdir -p "$(dirname "$csv_path")"
  if [[ ! -f "$csv_path" ]]; then
    printf '%s\n' "$header" > "$csv_path"
    return 0
  fi
  current_header="$(head -n 1 "$csv_path" 2>/dev/null || true)"
  if [[ "$current_header" != "$header" ]]; then
    backup="${csv_path}.legacy_$(date -u +%Y%m%dT%H%M%SZ).csv"
    cp "$csv_path" "$backup" 2>/dev/null || true
    printf '%s\n' "$header" > "$csv_path"
    log_warn "simulation timing CSV had legacy format; migrated to simple format and backup was saved at $backup"
  fi
}

append_simulation_time_row() {
  local elapsed="$1"
  local csv_path
  csv_path="$(simulation_time_csv_path)"
  ensure_simulation_time_csv
  printf '%s,%s\n' \
    "$(log_ts)" \
    "$elapsed" >> "$csv_path"
}

NO_PLOTS=""
PLOT_ONLY=""
FINAL_STEP=""
LOOP=""
FORCE=""
CONTINUOUS=""
FORCE_CONTINUOUS=""
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    usage
    exit 0
  elif [[ "$arg" == "--no-plots" ]]; then
    NO_PLOTS="--no-plots"
  elif [[ "$arg" == "--plot-only" ]]; then
    PLOT_ONLY="--plot-only"
  elif [[ "$arg" == "--final" ]]; then
    FINAL_STEP="--final"
  elif [[ "$arg" == "--loop" ]]; then
    LOOP="1"
  elif [[ "$arg" == "--force" ]]; then
    FORCE="--force"
  elif [[ "$arg" == "-c" || "$arg" == "--continuous" ]]; then
    CONTINUOUS="1"
  elif [[ "$arg" == "-fc" || "$arg" == "--force-continuous" ]]; then
    FORCE_CONTINUOUS="1"
  else
    ARGS+=("$arg")
  fi
done

if [[ -n "$FORCE_CONTINUOUS" && -z "$CONTINUOUS" ]]; then
  log_error "--force-continuous requires -c/--continuous."
  usage
  exit 1
fi

if [[ -z "$CONTINUOUS" && ${#ARGS[@]} -lt 1 ]]; then
  usage
  exit 1
fi

if [[ -n "$CONTINUOUS" ]]; then
  STEP="all"
  LOOP="1"
  NO_PLOTS="--no-plots"
else
  STEP="${ARGS[0]}"
fi
DT="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "$NO_PLOTS" && -n "$PLOT_ONLY" ]]; then
  log_error "--no-plots and --plot-only cannot be used together."
  exit 1
fi

if [[ -n "$CONTINUOUS" ]]; then
  LOCK_DIR="/tmp/mingo_digital_twin_run_step_continuous.lock"
  PID_FILE="$LOCK_DIR/pid"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [[ -n "$FORCE_CONTINUOUS" ]]; then
      if [[ -f "$PID_FILE" ]]; then
        LOCK_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
          CMDLINE="$(tr '\0' ' ' < "/proc/$LOCK_PID/cmdline" 2>/dev/null || true)"
          if [[ "$CMDLINE" == *"run_step.sh"* ]]; then
            kill "$LOCK_PID" 2>/dev/null || true
            sleep 1
            if kill -0 "$LOCK_PID" 2>/dev/null; then
              kill -9 "$LOCK_PID" 2>/dev/null || true
            fi
          fi
        fi
      fi
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR"
    else
      if [[ -f "$PID_FILE" ]]; then
        LOCK_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$LOCK_PID" ]] && kill -0 "$LOCK_PID" 2>/dev/null; then
          CMDLINE="$(tr '\0' ' ' < "/proc/$LOCK_PID/cmdline" 2>/dev/null || true)"
          if [[ "$CMDLINE" == *"run_step.sh"* ]]; then
            LOCK_AGE_S="$(ps -o etimes= -p "$LOCK_PID" 2>/dev/null | tr -d ' ' || true)"
            [[ -z "$LOCK_AGE_S" ]] && LOCK_AGE_S="unknown"
            log_warn "Continuous operation already running; pid=$LOCK_PID etimes_s=$LOCK_AGE_S lock_dir=$LOCK_DIR pid_file=$PID_FILE cmdline=$CMDLINE"
            exit 0
          fi
        fi
      fi
      log_warn "Stale continuous lock detected; removing lock_dir=$LOCK_DIR pid_file=$PID_FILE and continuing"
      rm -rf "$LOCK_DIR"
      mkdir "$LOCK_DIR"
    fi
  fi
  echo "$$" > "$PID_FILE"
  cleanup_continuous_lock() {
    rm -f "$PID_FILE" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
  }
  trap cleanup_continuous_lock EXIT INT TERM
fi

run_step() {
  case "$1" in
    1) python3 "$DT/MASTER_STEPS/STEP_1/step_1_blank_to_generated.py" --config "$DT/MASTER_STEPS/STEP_1/config_step_1_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    2) python3 "$DT/MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py" --config "$DT/MASTER_STEPS/STEP_2/config_step_2_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    3) python3 "$DT/MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py" --config "$DT/MASTER_STEPS/STEP_3/config_step_3_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    4) python3 "$DT/MASTER_STEPS/STEP_4/step_4_hit_to_measured.py" --config "$DT/MASTER_STEPS/STEP_4/config_step_4_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    5) python3 "$DT/MASTER_STEPS/STEP_5/step_5_measured_to_triggered.py" --config "$DT/MASTER_STEPS/STEP_5/config_step_5_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    6) python3 "$DT/MASTER_STEPS/STEP_6/step_6_triggered_to_timing.py" --config "$DT/MASTER_STEPS/STEP_6/config_step_6_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    7) python3 "$DT/MASTER_STEPS/STEP_7/step_7_timing_to_uncalibrated.py" --config "$DT/MASTER_STEPS/STEP_7/config_step_7_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    8) python3 "$DT/MASTER_STEPS/STEP_8/step_8_uncalibrated_to_threshold.py" --config "$DT/MASTER_STEPS/STEP_8/config_step_8_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    9) python3 "$DT/MASTER_STEPS/STEP_9/step_9_threshold_to_trigger.py" --config "$DT/MASTER_STEPS/STEP_9/config_step_9_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    10) python3 "$DT/MASTER_STEPS/STEP_10/step_10_triggered_to_jitter.py" --config "$DT/MASTER_STEPS/STEP_10/config_step_10_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE ;;
    final)
      python3 "$DT/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py" --config "$DT/MASTER_STEPS/STEP_FINAL/config_step_final_physics.yaml" $NO_PLOTS $PLOT_ONLY $FORCE
      ;;
    *)
      log_error "Unknown step: $1"
      exit 1
      ;;
  esac
}

while true; do
  cycle_start_epoch=$(date +%s)
  cycle_end_epoch="$cycle_start_epoch"
  cycle_elapsed=0
  cycle_should_record=0
  step_failed="0"

  if [[ -n "$CONTINUOUS" ]]; then
    log_info "continuous loop start"
  fi
  case "$STEP" in
    all)
      cycle_should_record=1
      failed_steps=0
      for step in $(seq 1 10); do
        if ! run_step "$step"; then
          log_warn "step=$step failed; continuing"
          failed_steps=$((failed_steps + 1))
        fi
      done
      cycle_end_epoch=$(date +%s)
      cycle_elapsed=$((cycle_end_epoch - cycle_start_epoch))
      log_info "all steps completed in ${cycle_elapsed}s"
      ;;
    from)
      start_step="${ARGS[1]:-}"
      if [[ -z "$start_step" ]]; then
        log_error "Usage: $0 from <step_number> [--no-plots]"
        exit 1
      fi
      cycle_should_record=1
      for step in $(seq "$start_step" 10); do
        if ! run_step "$step"; then
          log_warn "step=$step failed; continuing"
        fi
      done
      cycle_end_epoch=$(date +%s)
      cycle_elapsed=$((cycle_end_epoch - cycle_start_epoch))
      log_info "steps ${start_step}-10 completed in ${cycle_elapsed}s"
      ;;
    *)
      if ! run_step "$STEP"; then
        step_failed="1"
      fi
      cycle_end_epoch=$(date +%s)
      cycle_elapsed=$((cycle_end_epoch - cycle_start_epoch))
      ;;
  esac

  if [[ "$cycle_should_record" == "1" ]]; then
    if ! append_simulation_time_row "$cycle_elapsed"; then
      log_warn "failed to append simulation timing row to $(simulation_time_csv_path)"
    fi
  fi

  if [[ "$step_failed" == "1" ]]; then
    exit 1
  fi

  if [[ -z "$LOOP" ]]; then
    break
  fi
  log_info "loop enabled; restarting"
done
