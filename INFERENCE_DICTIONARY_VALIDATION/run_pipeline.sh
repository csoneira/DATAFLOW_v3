#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE DICTIONARY VALIDATION — Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs all six steps of the validation pipeline in order.
#
#   STEP 1  Build Dictionary        — build param-metadata dictionary from
#                                      simulated data files
#   STEP 2  Validate Simulation     — compare estimated vs simulated
#                                      efficiencies per plane
#   STEP 3  Relative Error          — compute relative errors, apply quality
#                                      cuts, produce filtered reference
#   STEP 4  Self-Consistency        — match every data sample against the
#                                      dictionary and recover (flux, eff)
#   STEP 5  Uncertainty Limits      — calibrate uncertainty curves and
#                                      dictionary coverage diagnostics
#   STEP 6  Uncertainty LUT         — build the 3-D empirical uncertainty
#                                      look-up table for inference
#
# Shared libraries (not steps, imported by the scripts above):
#   msv_utils.py          — common helpers (logging, plotting, config, …)
#   uncertainty_lut.py    — LUT loader + trilinear interpolation class
#
# Usage:
#   ./run_pipeline.sh              # run all steps
#   ./run_pipeline.sh  3 4 6       # run only steps 3, 4 and 6
#   ./run_pipeline.sh  --from 4    # run steps 4, 5, 6
#
# Each step is idempotent: re-running it overwrites its own output/.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

# ── colours for terminal output ──────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'   # no colour

# ── helpers ──────────────────────────────────────────────────────────────

banner() {
    echo ""
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  STEP $1  —  $2${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
}

run_step() {
    local step_num="$1"
    local step_dir="$2"
    local script="$3"
    local desc="$4"
    shift 4
    # remaining args ($@) are passed to the python script

    banner "$step_num" "$desc"
    echo -e "  Script: ${BOLD}${step_dir}/${script}${NC}"
    if [[ $# -gt 0 ]]; then
        echo -e "  Args:   $*"
    fi
    echo ""

    if ! "$PYTHON" "${step_dir}/${script}" "$@" ; then
        echo -e "${RED}${BOLD}  ✗  STEP ${step_num} FAILED${NC}"
        exit 1
    fi
    echo -e "${GREEN}${BOLD}  ✓  STEP ${step_num} complete${NC}"
}

# ── parse arguments ─────────────────────────────────────────────────────

STEPS_TO_RUN=()

if [[ $# -eq 0 ]]; then
    # Run all steps
    STEPS_TO_RUN=(1 2 3 4 5 6)
elif [[ "$1" == "--from" ]]; then
    from_step="${2:-1}"
    for s in 1 2 3 4 5 6; do
        if (( s >= from_step )); then
            STEPS_TO_RUN+=("$s")
        fi
    done
else
    # Explicit list of step numbers
    STEPS_TO_RUN=("$@")
fi

echo -e "${BOLD}Pipeline steps to run: ${STEPS_TO_RUN[*]}${NC}"
echo ""

# ── execute requested steps ─────────────────────────────────────────────

for step in "${STEPS_TO_RUN[@]}"; do
    case "$step" in
        1)
            run_step 1 \
                STEP_1_BUILD_DICTIONARY \
                build_dictionary.py \
                "Build Dictionary"
            ;;
        2)
            run_step 2 \
                STEP_2_VALIDATE_SIMULATION \
                validate_simulation_vs_parameters.py \
                "Validate Simulation"
            ;;
        3)
            run_step 3 \
                STEP_3_RELATIVE_ERROR \
                compute_relative_error.py \
                "Compute Relative Error"
            ;;
        4)
            run_step 4 \
                STEP_4_SELF_CONSISTENCY \
                self_consistency_r2.py \
                "Self-Consistency (all samples)" \
                --all
            ;;
        5)
            run_step 5 \
                STEP_5_UNCERTAINTY_LIMITS \
                compute_uncertainty_limits.py \
                "Uncertainty Limits"
            ;;
        6)
            run_step 6 \
                STEP_6_UNCERTAINTY_LUT \
                build_uncertainty_lut.py \
                "Uncertainty LUT"
            ;;
        *)
            echo -e "${RED}Unknown step: ${step}  (valid: 1-6)${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Pipeline complete  (steps: ${STEPS_TO_RUN[*]})${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
