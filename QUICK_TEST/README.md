# QUICK_TEST

## Purpose

`quick_test_flux_matrix.py` is a **quick application script** for real metadata.

It is used to estimate flux from:
- `global_rate_hz`
- `reference_efficiency = (eff_2 + eff_3) / 2`

using the affine formula already calibrated in:

`INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/PURELY_LINEAR/PLOTS/00_linear_summary.txt`

Important: this script **does not calibrate** the matrix/model.  
Calibration is done in `PURELY_LINEAR`. This script only extracts and applies.

## What the script does

1. Reads real metadata (default: MINGO01 TASK_1 metadata).
2. Computes:
   - `rate_1234_hz = raw_tt_1234_rate_hz`
   - `eff_2 = 1 - raw_tt_134_rate_hz / raw_tt_1234_rate_hz`
   - `eff_3 = 1 - raw_tt_124_rate_hz / raw_tt_1234_rate_hz`
   - `reference_efficiency = 0.5 * (eff_2 + eff_3)`
3. Builds `global_rate_hz` from:
   - `events_per_second_global_rate` if present
   - fallback to `raw_tt_1234_rate_hz` only where needed
4. Detects station and active z-configuration `(P1,P2,P3,P4)` from:
   - `MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY/STATION_X/input_file_mingoXX.csv`
5. Extracts coefficients `(m11, m12, t1)` from the PURELY_LINEAR summary.
6. Applies:
   - `flux_est = m11*global_rate_hz + m12*reference_efficiency + t1`
7. Writes CSV, info TXT, and plots.

## Default run

```bash
python3 /home/mingo/DATAFLOW_v3/QUICK_TEST/quick_test_flux_matrix.py
```

## Main outputs

- `QUICK_TEST/quick_test_flux_output.csv`
- `QUICK_TEST/quick_test_apply_info.txt`
- `QUICK_TEST/PLOTS/01_inputs_over_time.png`
- `QUICK_TEST/PLOTS/02_flux_est_over_time.png`

## Key interpretation

- `global_rate_hz` and `raw_tt_1234_rate_hz` are not assumed to be identical.
- Efficiency is always computed with `raw_tt_1234_rate_hz` in the denominator.
- The matrix/formula is valid only as the local linear approximation produced by the selected PURELY_LINEAR calibration.
