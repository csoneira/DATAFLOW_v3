# SIM Validation (MASTER_VS_SIMULATION)

This module validates the consistency between:

1. Simulation input parameters (from files generated for `MINGO00`)
2. Measured metadata/rates obtained after running STEP_1 analysis

It consumes the dictionary produced by:

`MASTER_VS_SIMULATION/DICTIONARY_CREATOR/STEP_1_BUILD/output/task_01/param_metadata_dictionary.csv`

## What It Produces

- `validation_table.csv`: per-file table with:
  - simulated parameters: `flux_cm2_min`, `cos_n`, simulated efficiencies (`eff_sim_p1..p4`)
  - measured trigger/rate columns (`raw_tt_*`)
  - estimated efficiencies (`eff_est_p1..p4`)
  - `generated_events_count` (auto-detected from dictionary, typically `selected_rows`)
  - signed residuals (`eff_resid_p1..p4` = estimated - simulated)
  - relative errors (`eff_rel_err_p1..p4` = residual / simulated)
  - absolute errors (`eff_abs_err_p1..p4`)
  - `global_trigger_rate_hz` (sum of available `raw_tt_*_rate_hz`)

- `summary_metrics.csv`:
  - per-plane efficiency comparison metrics (`MAE`, `RMSE`, `bias`, `corr`)
  - correlation of `flux_cm2_min` vs `global_trigger_rate_hz`

- Plots:
  - `scatter_flux_vs_global_trigger_rate.png`
  - `scatter_eff_sim_vs_estimated.png`
  - `scatter_residual_relative_error_planes_2_3.png` (`|relative error| <= 5%` by default)
  - `scatter_colored_sim_eff_relerr_eventcount_planes_2_3.png`
  - `contour_eff_vs_flux_cosn2_identical_eff.png` (filtered by `cos_n` and equal plane efficiencies)

## Efficiency Estimator

Default estimator (recommended):

- `eff_est_plane_i = N4 / (N4 + N3_missing_i)`

where:
- `N4 = raw_tt_1234_count`
- `N3_missing_1 = raw_tt_234_count`
- `N3_missing_2 = raw_tt_134_count`
- `N3_missing_3 = raw_tt_124_count`
- `N3_missing_4 = raw_tt_123_count`

Alternative method is available for compatibility with previous "quick" formulations:

- `1 - N3_missing_i / N4`

## Run

```bash
python3 /home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION/SIM_VALIDATION/validate_simulation_vs_parameters.py
```

With explicit config:

```bash
python3 /home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION/SIM_VALIDATION/validate_simulation_vs_parameters.py \
  --config /home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION/SIM_VALIDATION/config_validation.json
```

Contour-specific overrides (optional):

```bash
python3 /home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION/SIM_VALIDATION/validate_simulation_vs_parameters.py \
  --contour-cos-n 2.0 \
  --contour-cos-tol 1e-9 \
  --contour-equal-eff-tol 1e-9 \
  --contour-value-col global_trigger_rate_hz
```

Relative-error filter override (optional):

```bash
python3 /home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION/SIM_VALIDATION/validate_simulation_vs_parameters.py \
  --relerr-max-abs 0.05
```
