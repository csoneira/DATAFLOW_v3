#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# METRIC COMPARISON PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs STEPs 1–3 once (shared), then runs STEP 4 → 6 → 7 for each of the
# four score metrics (l2, chi2, poisson, r2).  At the end, a comparison
# table is printed and saved to COMPARISON/metric_comparison.csv.
#
# Output layout (per metric):
#   STEP_4_SELF_CONSISTENCY/<metric>/OUTPUTS/{FILES,PLOTS}/
#   STEP_6_UNCERTAINTY_LUT/<metric>/OUTPUTS/{FILES,PLOTS}/ (LUT in FILES/lut/)
#   STEP_7_SIMULATED_DEMO/<metric>/OUTPUTS/{FILES,PLOTS}/
#   COMPARISON/                                    (final summary)
#
# Usage:
#   ./run_metric_comparison.sh                     # all 4 metrics
#   ./run_metric_comparison.sh l2 chi2             # only those two
#   ./run_metric_comparison.sh --skip-shared l2    # skip STEPs 1–3
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

ALL_METRICS=(l2 chi2 poisson r2)

# ── colours ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

# ── parse arguments ─────────────────────────────────────────────────────
SKIP_SHARED=false
METRICS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-shared) SKIP_SHARED=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--skip-shared] [metric1 metric2 ...]"
            echo "  Metrics: l2 chi2 poisson r2  (default: all four)"
            exit 0
            ;;
        *)
            if [[ " ${ALL_METRICS[*]} " == *" $1 "* ]]; then
                METRICS+=("$1")
            else
                echo -e "${RED}Unknown metric: $1  (valid: ${ALL_METRICS[*]})${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ ${#METRICS[@]} -eq 0 ]]; then
    METRICS=("${ALL_METRICS[@]}")
fi

echo -e "${BOLD}Metrics to compare: ${METRICS[*]}${NC}"
echo ""

# ── helpers ──────────────────────────────────────────────────────────────

banner() {
    echo ""
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
}

run_py() {
    local label="$1"; shift
    echo -e "  ${BOLD}${label}${NC}"
    if ! "$PYTHON" "$@" ; then
        echo -e "${RED}${BOLD}  ✗  FAILED: ${label}${NC}"
        exit 1
    fi
    echo -e "${GREEN}  ✓  ${label}${NC}"
}

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    echo $(( now - start ))
}

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Shared steps (1–3)  —  run once, metric-independent
# ═══════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_SHARED" == false ]]; then
    banner "PHASE 1 — Shared steps (1 → 3)"

    run_py "STEP 1: Build Dictionary" \
        STEP_1_BUILD_DICTIONARY/build_dictionary.py

    run_py "STEP 2: Validate Simulation" \
        STEP_2_VALIDATE_SIMULATION/validate_simulation_vs_parameters.py

    run_py "STEP 3: Relative Error + Deduplication" \
        STEP_3_RELATIVE_ERROR/compute_relative_error.py
else
    echo -e "${YELLOW}Skipping shared steps 1–3 (--skip-shared)${NC}"
fi

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Per-metric steps (4 → 6 → 7)
# ═══════════════════════════════════════════════════════════════════════════

TOTAL_START=$(date +%s)

for metric in "${METRICS[@]}"; do
    METRIC_START=$(date +%s)
    banner "PHASE 2 — Metric: ${metric}"

    STEP4_OUT="STEP_4_SELF_CONSISTENCY/${metric}"
    STEP6_OUT="STEP_6_UNCERTAINTY_LUT/${metric}"
    STEP7_OUT="STEP_7_SIMULATED_DEMO/${metric}"

    mkdir -p "$STEP4_OUT" "$STEP6_OUT" "$STEP7_OUT"

    # Choose appropriate feature scaling for each metric:
    #   l2, r2   → zscore  (equal feature weight, scale-invariant)
    #   chi2     → none    (uses max(y,1) as variance; z-scores break this)
    #   poisson  → none    (needs raw positive values for log-likelihood)
    case "$metric" in
        l2|r2) SCALE="zscore" ;;
        chi2|poisson) SCALE="none" ;;
    esac

    # ── STEP 4: Self-consistency ──────────────────────────────────────
    run_py "STEP 4 [${metric}]: Self-Consistency (all, scale=${SCALE})" \
        STEP_4_SELF_CONSISTENCY/self_consistency_r2.py \
        --all \
        --score-metric "$metric" \
        --metric-scale "$SCALE" \
        --out-dir "$STEP4_OUT"

    # ── STEP 6: Uncertainty LUT ───────────────────────────────────────
    run_py "STEP 6 [${metric}]: Uncertainty LUT" \
        STEP_6_UNCERTAINTY_LUT/build_uncertainty_lut.py \
        --all-results-csv "${STEP4_OUT}/OUTPUTS/FILES/all_samples_results.csv" \
        --out-dir "$STEP6_OUT"

    # ── STEP 7: Simulated Demo ────────────────────────────────────────
    run_py "STEP 7 [${metric}]: Simulated Demo" \
        STEP_7_SIMULATED_DEMO/simulated_uncertainty_demo.py \
        --all-results-csv "${STEP4_OUT}/OUTPUTS/FILES/all_samples_results.csv" \
        --lut-dir "${STEP6_OUT}/OUTPUTS/FILES/lut" \
        --out-dir "$STEP7_OUT"

    DT=$(elapsed_since "$METRIC_START")
    echo -e "${GREEN}${BOLD}  ✓  Metric '${metric}' complete  (${DT}s)${NC}"
done

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Comparison
# ═══════════════════════════════════════════════════════════════════════════

banner "PHASE 3 — Comparison"

COMPARISON_DIR="${SCRIPT_DIR}/COMPARISON"
mkdir -p "$COMPARISON_DIR"

# Build the list of metric flags for the Python script
METRIC_ARGS=""
for m in "${METRICS[@]}"; do
    METRIC_ARGS="${METRIC_ARGS} ${m}"
done

SCRIPT_DIR="$SCRIPT_DIR" "$PYTHON" - "${COMPARISON_DIR}" ${METRIC_ARGS} <<'PYEOF'
import sys, json, os
from pathlib import Path
import csv

comp_dir = Path(sys.argv[1])
metrics = sys.argv[2:]
base = Path(os.environ.get("SCRIPT_DIR", str(comp_dir.parent)))

rows = []
for metric in metrics:
    summary_path = base / f"STEP_7_SIMULATED_DEMO/{metric}/OUTPUTS/FILES/demo_summary.json"
    step4_csv = base / f"STEP_4_SELF_CONSISTENCY/{metric}/OUTPUTS/FILES/all_samples_results.csv"

    if not summary_path.exists():
        print(f"  WARNING: {summary_path} not found, skipping {metric}")
        continue

    with open(summary_path) as f:
        s = json.load(f)

    raw = s.get("coverage_without_exact_self_matches", s.get("coverage_all_rows", {}))
    cal = s.get("coverage_without_exact_self_matches_calibrated", {})
    calib = s.get("sigma_calibration", {})

    rows.append({
        "metric": metric,
        "n_samples": raw.get("n_samples", "?"),
        "med_flux_err_%": f"{raw.get('median_abs_flux_rel_error_pct', 0):.3f}",
        "p68_flux_err_%": f"{raw.get('p68_abs_flux_rel_error_pct', 0):.3f}",
        "med_eff_err_%": f"{raw.get('median_abs_eff_rel_error_pct', 0):.3f}",
        "p68_eff_err_%": f"{raw.get('p68_abs_eff_rel_error_pct', 0):.3f}",
        "cov_flux_1s_raw": f"{raw.get('coverage_flux_1sigma', 0)*100:.1f}%",
        "cov_eff_1s_raw": f"{raw.get('coverage_eff_1sigma', 0)*100:.1f}%",
        "cov_flux_1s_cal": f"{cal.get('coverage_flux_1sigma', 0)*100:.1f}%",
        "cov_eff_1s_cal": f"{cal.get('coverage_eff_1sigma', 0)*100:.1f}%",
        "cov_flux_2s_cal": f"{cal.get('coverage_flux_2sigma', 0)*100:.1f}%",
        "cov_eff_2s_cal": f"{cal.get('coverage_eff_2sigma', 0)*100:.1f}%",
        "scale_flux": f"{calib.get('scale_flux', 0):.3f}",
        "scale_eff": f"{calib.get('scale_eff', 0):.3f}",
        "med_err/sigma_flux": f"{raw.get('median_error_over_sigma_flux', 0):.3f}",
        "med_err/sigma_eff": f"{raw.get('median_error_over_sigma_eff', 0):.3f}",
    })

if not rows:
    print("  No results found!")
    sys.exit(1)

# ── Print table ──
keys = list(rows[0].keys())
col_widths = {k: max(len(k), max(len(r[k]) if isinstance(r[k], str) else len(str(r[k])) for r in rows)) for k in keys}

header = " | ".join(k.ljust(col_widths[k]) for k in keys)
sep = "-+-".join("-" * col_widths[k] for k in keys)

print()
print(header)
print(sep)
for r in rows:
    vals = " | ".join(str(r[k]).ljust(col_widths[k]) for k in keys)
    print(vals)
print()

# ── Highlight best ──
best_metric_flux = min(rows, key=lambda r: float(r["med_flux_err_%"]))
best_metric_eff  = min(rows, key=lambda r: float(r["med_eff_err_%"]))
print(f"  Best median flux error:  {best_metric_flux['metric']}  ({best_metric_flux['med_flux_err_%']}%)")
print(f"  Best median eff error:   {best_metric_eff['metric']}  ({best_metric_eff['med_eff_err_%']}%)")

# closest to 68% raw coverage
best_cov_flux = min(rows, key=lambda r: abs(float(r["cov_flux_1s_raw"].rstrip('%')) - 68.0))
print(f"  Best raw 1σ flux coverage (closest to 68%): {best_cov_flux['metric']}  ({best_cov_flux['cov_flux_1s_raw']})")

# smallest calibration scale factor (closest to 1.0 = best-calibrated)
best_scale = min(rows, key=lambda r: abs(float(r["scale_flux"]) - 1.0))
print(f"  Most naturally calibrated (σ_flux scale closest to 1.0): {best_scale['metric']}  ({best_scale['scale_flux']})")
print()

# ── Save CSV ──
csv_path = comp_dir / "metric_comparison.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(f"  Saved: {csv_path}")
print()
PYEOF

TOTAL_DT=$(elapsed_since "$TOTAL_START")

echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Metric comparison complete  (${TOTAL_DT}s total)${NC}"
echo -e "${GREEN}${BOLD}  Results: COMPARISON/metric_comparison.csv${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
