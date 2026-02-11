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
  --debug              Keep full step stdout/stderr (default in continuous mode is concise logs)
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
  header='exec_time_s,step,timestamp_utc'
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
  local step="$2"
  local csv_path
  csv_path="$(simulation_time_csv_path)"
  ensure_simulation_time_csv
  printf '%s,%s,%s\n' \
    "$elapsed" \
    "$step" \
    "$(log_ts)" >> "$csv_path"
}

elapsed_seconds_between() {
  local start_ts="$1"
  local end_ts="$2"
  LC_ALL=C awk -v s="$start_ts" -v e="$end_ts" 'BEGIN{d=e-s; if (d < 0) d=0; printf "%.6f", d}'
}

NO_PLOTS=""
PLOT_ONLY=""
FINAL_STEP=""
LOOP=""
FORCE=""
CONTINUOUS=""
FORCE_CONTINUOUS=""
DEBUG=""
QUIET_CONTINUOUS=""
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
  elif [[ "$arg" == "--debug" ]]; then
    DEBUG="1"
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
WORK_CACHE_PATH="/tmp/mingo_digital_twin_run_step_work_cache.csv"
if [[ -n "$CONTINUOUS" && -z "$DEBUG" ]]; then
  QUIET_CONTINUOUS="1"
fi

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
  terminate_direct_children() {
    pkill -TERM -P "$$" 2>/dev/null || true
  }
  handle_termination_signal() {
    local code="$1"
    terminate_direct_children
    cleanup_continuous_lock
    exit "$code"
  }
  trap cleanup_continuous_lock EXIT
  trap 'handle_termination_signal 130' INT
  trap 'handle_termination_signal 143' TERM
fi

run_step() {
  local step="$1"
  local -a cmd
  local tmp_log
  local failure_log
  local last_error
  local rc
  case "$step" in
    1) cmd=(python3 "$DT/MASTER_STEPS/STEP_1/step_1_blank_to_generated.py" --config "$DT/MASTER_STEPS/STEP_1/config_step_1_physics.yaml") ;;
    2) cmd=(python3 "$DT/MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py" --config "$DT/MASTER_STEPS/STEP_2/config_step_2_physics.yaml") ;;
    3) cmd=(python3 "$DT/MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py" --config "$DT/MASTER_STEPS/STEP_3/config_step_3_physics.yaml") ;;
    4) cmd=(python3 "$DT/MASTER_STEPS/STEP_4/step_4_hit_to_measured.py" --config "$DT/MASTER_STEPS/STEP_4/config_step_4_physics.yaml") ;;
    5) cmd=(python3 "$DT/MASTER_STEPS/STEP_5/step_5_measured_to_triggered.py" --config "$DT/MASTER_STEPS/STEP_5/config_step_5_physics.yaml") ;;
    6) cmd=(python3 "$DT/MASTER_STEPS/STEP_6/step_6_triggered_to_timing.py" --config "$DT/MASTER_STEPS/STEP_6/config_step_6_physics.yaml") ;;
    7) cmd=(python3 "$DT/MASTER_STEPS/STEP_7/step_7_timing_to_uncalibrated.py" --config "$DT/MASTER_STEPS/STEP_7/config_step_7_physics.yaml") ;;
    8) cmd=(python3 "$DT/MASTER_STEPS/STEP_8/step_8_uncalibrated_to_threshold.py" --config "$DT/MASTER_STEPS/STEP_8/config_step_8_physics.yaml") ;;
    9) cmd=(python3 "$DT/MASTER_STEPS/STEP_9/step_9_threshold_to_trigger.py" --config "$DT/MASTER_STEPS/STEP_9/config_step_9_physics.yaml") ;;
    10) cmd=(python3 "$DT/MASTER_STEPS/STEP_10/step_10_triggered_to_jitter.py" --config "$DT/MASTER_STEPS/STEP_10/config_step_10_physics.yaml") ;;
    final) cmd=(python3 "$DT/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py" --config "$DT/MASTER_STEPS/STEP_FINAL/config_step_final_physics.yaml") ;;
    *)
      log_error "Unknown step: $step"
      exit 1
      ;;
  esac
  [[ -n "$NO_PLOTS" ]] && cmd+=("$NO_PLOTS")
  [[ -n "$PLOT_ONLY" ]] && cmd+=("$PLOT_ONLY")
  [[ -n "$FORCE" ]] && cmd+=("$FORCE")

  if [[ -z "$QUIET_CONTINUOUS" ]]; then
    "${cmd[@]}"
    return $?
  fi

  tmp_log="$(mktemp "/tmp/mingo_digital_twin_step_${step}.XXXXXX.log")"
  if "${cmd[@]}" >"$tmp_log" 2>&1; then
    rm -f "$tmp_log"
    return 0
  fi
  rc=$?
  failure_log="/tmp/mingo_digital_twin_last_step_${step}.log"
  cp "$tmp_log" "$failure_log" 2>/dev/null || true
  last_error="$(awk 'NF {line=$0} END {print line}' "$tmp_log" 2>/dev/null || true)"
  if [[ -z "$last_error" ]]; then
    last_error="(no output from step process)"
  fi
  log_warn "step=$step failed rc=$rc; last_line=\"$last_error\" (full log: $failure_log, use --debug for verbose output)"
  rm -f "$tmp_log"
  return "$rc"
}

step_output_dir_for_step() {
  case "$1" in
    1) printf '%s/INTERSTEPS/STEP_1_TO_2' "$DT" ;;
    2) printf '%s/INTERSTEPS/STEP_2_TO_3' "$DT" ;;
    3) printf '%s/INTERSTEPS/STEP_3_TO_4' "$DT" ;;
    4) printf '%s/INTERSTEPS/STEP_4_TO_5' "$DT" ;;
    5) printf '%s/INTERSTEPS/STEP_5_TO_6' "$DT" ;;
    6) printf '%s/INTERSTEPS/STEP_6_TO_7' "$DT" ;;
    7) printf '%s/INTERSTEPS/STEP_7_TO_8' "$DT" ;;
    8) printf '%s/INTERSTEPS/STEP_8_TO_9' "$DT" ;;
    9) printf '%s/INTERSTEPS/STEP_9_TO_10' "$DT" ;;
    10) printf '%s/INTERSTEPS/STEP_10_TO_FINAL' "$DT" ;;
    *) return 1 ;;
  esac
}

count_sim_run_dirs() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    printf '0\n'
    return 0
  fi
  find "$dir" -mindepth 1 -maxdepth 1 -type d -name 'SIM_RUN_*' | wc -l | tr -d ' '
  printf '\n'
}

step_has_cached_work() {
  local step="$1"
  local has_work
  if [[ ! -f "$WORK_CACHE_PATH" ]]; then
    return 0
  fi
  has_work="$(awk -F, -v step="$step" 'NR>1 && $1==step {print $2; exit}' "$WORK_CACHE_PATH")"
  if [[ -z "$has_work" ]]; then
    return 0
  fi
  [[ "$has_work" == "1" ]]
}

refresh_step_work_cache() {
  local mesh_path="$DT/INTERSTEPS/STEP_0_TO_1/param_mesh.csv"
  local intersteps_dir="$DT/INTERSTEPS"
  python3 - "$mesh_path" "$intersteps_dir" "$WORK_CACHE_PATH" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

mesh_path = Path(sys.argv[1])
intersteps_dir = Path(sys.argv[2])
out_csv_path = Path(sys.argv[3])

if not mesh_path.exists():
    raise FileNotFoundError(f"param_mesh.csv not found: {mesh_path}")

mesh = pd.read_csv(mesh_path)
if "done" in mesh.columns:
    mesh = mesh[mesh["done"].fillna(0).astype(int) != 1]

step_cols = [f"step_{idx}_id" for idx in range(1, 11)]
for col in step_cols:
    if col not in mesh.columns:
        mesh[col] = ""


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        return f"{int(float(value)):03d}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "<na>"} else text


for col in step_cols:
    mesh[col] = mesh[col].map(normalize_id)


def output_dir_for_step(step_num: int) -> Path:
    if step_num == 10:
        return intersteps_dir / "STEP_10_TO_FINAL"
    return intersteps_dir / f"STEP_{step_num}_TO_{step_num + 1}"


def parse_sim_run_ids(path: Path) -> tuple[str, ...] | None:
    name = path.name
    if not name.startswith("SIM_RUN_"):
        return None
    raw = name[len("SIM_RUN_") :].split("_")
    if not raw:
        return None
    normalized = []
    for val in raw:
        norm = normalize_id(val)
        if norm:
            normalized.append(norm)
    return tuple(normalized)


rows: list[tuple[int, int, int, int, int, int]] = []

for step_num in range(1, 11):
    prefix_cols = step_cols[: step_num - 1]
    current_col = step_cols[step_num - 1]
    needed_by_prefix: dict[tuple[str, ...], int] = {}

    if step_num == 1:
        needed_by_prefix[tuple()] = int(mesh[current_col][mesh[current_col] != ""].nunique())
    else:
        subset = mesh[prefix_cols + [current_col]].copy()
        valid = subset[current_col] != ""
        for col in prefix_cols:
            valid &= subset[col] != ""
        subset = subset[valid]
        if not subset.empty:
            grouped = subset.groupby(prefix_cols, dropna=False)[current_col].nunique()
            for key, count in grouped.items():
                if not isinstance(key, tuple):
                    key = (key,)
                needed_by_prefix[tuple(str(k) for k in key)] = int(count)

    produced_by_prefix: Counter[tuple[str, ...]] = Counter()
    produced_dirs = 0
    for sim_dir in output_dir_for_step(step_num).glob("SIM_RUN_*"):
        ids = parse_sim_run_ids(sim_dir)
        if ids is None or len(ids) < step_num:
            continue
        produced_dirs += 1
        produced_by_prefix[tuple(ids[: step_num - 1])] += 1

    available_prefixes: set[tuple[str, ...]] = set()
    if step_num == 1:
        available_prefixes.add(tuple())
    else:
        upstream_dir = output_dir_for_step(step_num - 1)
        for sim_dir in upstream_dir.glob("SIM_RUN_*"):
            ids = parse_sim_run_ids(sim_dir)
            if ids is None or len(ids) < step_num - 1:
                continue
            available_prefixes.add(tuple(ids[: step_num - 1]))

    pending_prefixes = 0
    for prefix in available_prefixes:
        needed = needed_by_prefix.get(prefix, 0)
        if needed <= 0:
            continue
        if produced_by_prefix.get(prefix, 0) < needed:
            pending_prefixes += 1

    has_work = 1 if pending_prefixes > 0 else 0
    expected_dirs = int(sum(needed_by_prefix.values()))
    rows.append(
        (
            step_num,
            has_work,
            pending_prefixes,
            len(available_prefixes),
            produced_dirs,
            expected_dirs,
        )
    )

out_csv_path.parent.mkdir(parents=True, exist_ok=True)
with out_csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "step",
            "has_work",
            "pending_prefixes",
            "available_prefixes",
            "produced_dirs",
            "expected_dirs",
        ]
    )
    writer.writerows(rows)
PY
}

refresh_work_cache_or_disable() {
  if refresh_step_work_cache; then
    if [[ -n "$DEBUG" ]]; then
      log_info "work cache refreshed: $WORK_CACHE_PATH"
    fi
    return 0
  fi
  log_warn "failed to refresh work cache; continuing without cache"
  rm -f "$WORK_CACHE_PATH" 2>/dev/null || true
  return 0
}

run_step_with_progress() {
  local step="$1"
  local output_dir
  local before_count
  local after_count
  local start_epoch
  local end_epoch
  local elapsed
  local elapsed_log
  local rc
  output_dir="$(step_output_dir_for_step "$step")"
  before_count="$(count_sim_run_dirs "$output_dir")"
  start_epoch=$(date +%s.%N)
  if run_step "$step"; then
    rc=0
  else
    rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    end_epoch=$(date +%s.%N)
    elapsed="$(elapsed_seconds_between "$start_epoch" "$end_epoch")"
    elapsed_log="$(LC_ALL=C awk -v d="$elapsed" 'BEGIN{printf "%.3f", d}')"
    after_count="$(count_sim_run_dirs "$output_dir")"
    append_simulation_time_row "$elapsed" "$step" || true
    log_warn "step=$step status=failed rc=$rc dirs=${before_count}->${after_count} elapsed_s=$elapsed_log"
    return 2
  fi
  end_epoch=$(date +%s.%N)
  elapsed="$(elapsed_seconds_between "$start_epoch" "$end_epoch")"
  elapsed_log="$(LC_ALL=C awk -v d="$elapsed" 'BEGIN{printf "%.3f", d}')"
  append_simulation_time_row "$elapsed" "$step" || true
  after_count="$(count_sim_run_dirs "$output_dir")"
  if [[ "$after_count" -gt "$before_count" ]]; then
    log_info "step=$step status=progress dirs=${before_count}->${after_count} elapsed_s=$elapsed_log"
    return 0
  fi
  if [[ -n "$DEBUG" ]]; then
    log_info "step=$step status=noop dirs=${before_count}->${after_count} elapsed_s=$elapsed_log"
  fi
  return 1
}

while true; do
  cycle_start_epoch=$(date +%s)
  cycle_end_epoch="$cycle_start_epoch"
  step_failed="0"

  if [[ -n "$CONTINUOUS" ]]; then
    log_info "continuous loop start"
  fi
  case "$STEP" in
    all)
      failed_steps=0
      refresh_work_cache_or_disable
      lower_step=9
      while [[ "$lower_step" -ge 1 ]]; do
        top_rc=1
        if step_has_cached_work 10; then
          if run_step_with_progress 10; then
            refresh_work_cache_or_disable
            continue
          fi
          top_rc=$?
        elif [[ -n "$DEBUG" ]]; then
          log_info "step=10 status=cache-skip"
        fi
        if [[ "$top_rc" -eq 2 ]]; then
          log_warn "step=10 failed; continuing"
          failed_steps=$((failed_steps + 1))
        fi

        chain_progress=0
        for step in $(seq "$lower_step" 10); do
          if ! step_has_cached_work "$step"; then
            if [[ -n "$DEBUG" ]]; then
              log_info "step=$step status=cache-skip"
            fi
            continue
          fi
          if run_step_with_progress "$step"; then
            chain_progress=1
            refresh_work_cache_or_disable
            continue
          fi
          rc=$?
          if [[ "$rc" -eq 2 ]]; then
            log_warn "step=$step failed; continuing"
            failed_steps=$((failed_steps + 1))
          fi
        done
        if [[ "$chain_progress" -eq 0 ]]; then
          lower_step=$((lower_step - 1))
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

  if [[ "$step_failed" == "1" ]]; then
    exit 1
  fi

  if [[ -z "$LOOP" ]]; then
    break
  fi
  log_info "loop enabled; restarting"
done
