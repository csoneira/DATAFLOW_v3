# TASK3 Noise Baseline Analysis

This analysis estimates **MINGO01 noise** relative to a **MINGO00 noiseless baseline** using
`TASK_3` activation metadata matrices.

## Contents

- `task3_noise_baseline_analysis.py`: main analysis script
- `OUTPUTS/`: all generated plots and CSV summaries

## What it does

1. Loads `task_3_metadata_activation.csv` from station folders.
2. Detects complete 4x4 matrix families from columns like `..._P1_to_P1` ... `..._P4_to_P4`.
3. Computes station means for baseline (`MINGO00`) and target (`MINGO01`).
4. Produces for each matrix family:
   - baseline matrix,
   - target matrix,
   - delta matrix (`MINGO01 - MINGO00`).
5. Produces matrix-cell column plots (16 panels) for baseline vs target.
6. Writes ranked CSV tables for strongest differences.

## Run

From repository root (or any location):

```bash
python3 MASTER/ANCILLARY/TASK3_NOISE_BASELINE_ANALYSIS/task3_noise_baseline_analysis.py
```

Optional:

```bash
python3 MASTER/ANCILLARY/TASK3_NOISE_BASELINE_ANALYSIS/task3_noise_baseline_analysis.py \
  --baseline MINGO00 \
  --target MINGO01 \
  --family all \
  --z-threshold 3.0 \
  --delta-threshold 0.03
```

## Key outputs

- `OUTPUTS/matrix_cell_difference_summary.csv`
- `OUTPUTS/matrix_cell_significant_noise_candidates.csv`
- `OUTPUTS/matrix_activation_plane_*_MINGO01_minus_MINGO00.png`
- `OUTPUTS/matrix_columns_activation_plane_*_MINGO00_vs_MINGO01.png`
