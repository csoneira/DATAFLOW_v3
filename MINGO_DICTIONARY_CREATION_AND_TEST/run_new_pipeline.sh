#!/usr/bin/env bash
# ===========================================================================
# run_pipeline.sh — Run the inference dictionary validation pipeline.
#
# Usage:
#   ./run_pipeline.sh                  # Run all steps
#   ./run_pipeline.sh 1.1 1.2          # Run specific steps
#   ./run_pipeline.sh --from 2.1       # Run from a specific step onwards
#   ./run_pipeline.sh --list           # List available steps
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.json"

# Preferred layout: STEP_* directories directly under SCRIPT_DIR.
# Legacy layout fallback: SCRIPT_DIR/STEPS/STEP_*.
if [[ -d "${SCRIPT_DIR}/STEP_1_SETUP" && -d "${SCRIPT_DIR}/STEP_2_INFERENCE" && -d "${SCRIPT_DIR}/STEP_3_SYNTHETIC_TIME_SERIES" ]]; then
    STEP_ROOT="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/STEPS/STEP_1_SETUP" && -d "${SCRIPT_DIR}/STEPS/STEP_2_INFERENCE" && -d "${SCRIPT_DIR}/STEPS/STEP_3_SYNTHETIC_TIME_SERIES" ]]; then
    STEP_ROOT="${SCRIPT_DIR}/STEPS"
else
    echo "ERROR: Could not locate STEP directories under ${SCRIPT_DIR} or ${SCRIPT_DIR}/STEPS" >&2
    exit 1
fi

# Step definitions: label → script
declare -A STEP_SCRIPTS
STEP_SCRIPTS["1.1"]="${STEP_ROOT}/STEP_1_SETUP/STEP_1_1_COLLECT_DATA/collect_data.py"
STEP_SCRIPTS["1.2"]="${STEP_ROOT}/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/build_dictionary.py"
STEP_SCRIPTS["2.1"]="${STEP_ROOT}/STEP_2_INFERENCE/STEP_2_1_ESTIMATE_PARAMS/estimate_and_plot.py"
STEP_SCRIPTS["2.2"]="${STEP_ROOT}/STEP_2_INFERENCE/STEP_2_2_VALIDATION/validate_solution.py"
STEP_SCRIPTS["2.3"]="${STEP_ROOT}/STEP_2_INFERENCE/STEP_2_3_UNCERTAINTY/build_uncertainty_lut.py"
STEP_SCRIPTS["3.1"]="${STEP_ROOT}/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_1_TIME_SERIES_CREATION/create_time_series.py"
STEP_SCRIPTS["3.2"]="${STEP_ROOT}/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_2_SYNTHETIC_TIME_SERIES/synthetic_time_series.py"
STEP_SCRIPTS["3.3"]="${STEP_ROOT}/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_3_CORRECTION/correction_by_inference.py"

# Ordered list
STEP_ORDER=("1.1" "1.2" "2.1" "2.2" "2.3" "3.1" "3.2" "3.3")

# Step descriptions
declare -A STEP_DESC
STEP_DESC["1.1"]="Collect simulated data & match with simulation parameters"
STEP_DESC["1.2"]="Build dictionary and dataset (filter outliers, select entries)"
STEP_DESC["2.1"]="Estimate parameters (inverse problem via dictionary matching)"
STEP_DESC["2.2"]="Validate solution (estimated vs simulated, error analysis)"
STEP_DESC["2.3"]="Uncertainty assessment (build LUT from error distributions)"
STEP_DESC["3.1"]="Create synthetic (flux, efficiency) time series"
STEP_DESC["3.2"]="Build synthetic dataset from time series and dictionary"
STEP_DESC["3.3"]="Apply inference correction and uncertainty to synthetic dataset"

# ── Helpers ────────────────────────────────────────────────────────────

list_steps() {
    echo "Available steps:"
    for step in "${STEP_ORDER[@]}"; do
        echo "  ${step}  —  ${STEP_DESC[$step]}"
    done
}

print_help() {
    local script_name
    script_name="$(basename "$0")"
    cat <<EOF
Usage:
  ./${script_name}                 Run all steps
  ./${script_name} <step...>       Run specific steps (e.g. 1.1 2.3)
  ./${script_name} --from <step>   Run from a specific step onwards
  ./${script_name} --list          List available steps
  ./${script_name} -h|--help       Show this help message
EOF
}

run_step() {
    local step="$1"
    local script="${STEP_SCRIPTS[$step]}"
    if [[ ! -f "$script" ]]; then
        echo "ERROR: Script not found: $script" >&2
        return 1
    fi
    echo ""
    echo "=================================================================="
    echo "  STEP ${step}: ${STEP_DESC[$step]}"
    echo "  Script: ${script}"
    echo "=================================================================="
    python3 "$script" --config "$CONFIG"
    echo "  ✓ Step ${step} completed."
}

# ── Argument parsing ──────────────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
    list_steps
    exit 0
fi

if [[ "${1:-}" == "--from" ]]; then
    FROM_STEP="${2:-}"
    if [[ -z "$FROM_STEP" ]]; then
        echo "ERROR: --from requires a step number (e.g. --from 2.1)" >&2
        exit 1
    fi
    FOUND=0
    for step in "${STEP_ORDER[@]}"; do
        if [[ "$step" == "$FROM_STEP" ]]; then
            FOUND=1
        fi
        if [[ $FOUND -eq 1 ]]; then
            run_step "$step"
        fi
    done
    if [[ $FOUND -eq 0 ]]; then
        echo "ERROR: Unknown step: $FROM_STEP" >&2
        list_steps
        exit 1
    fi
    exit 0
fi

# If specific steps are given, run only those
if [[ $# -gt 0 ]]; then
    for step in "$@"; do
        if [[ -z "${STEP_SCRIPTS[$step]:-}" ]]; then
            echo "ERROR: Unknown step: $step" >&2
            list_steps
            exit 1
        fi
        run_step "$step"
    done
    exit 0
fi

# Default: run all steps
echo "Running full pipeline..."
for step in "${STEP_ORDER[@]}"; do
    run_step "$step"
done

echo ""
echo "Pipeline completed successfully."
