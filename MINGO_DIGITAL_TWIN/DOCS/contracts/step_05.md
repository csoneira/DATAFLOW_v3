# STEP 05 Interface Contract (Hit -> Signal)

## Purpose
Derive time-difference and charge-difference observables from per-strip measurements.

## Required inputs
- Input data (from STEP 04):
  - `event_id`.
  - `X_mea_i_sj`, `Y_mea_i_sj`, `T_sum_meas_i_sj`.
- Config inputs:
  - `qdiff_frac`.
  - Optional `c_mm_per_ns` (otherwise inherited from upstream metadata).

## Output schema
Outputs:
- `INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/step_5.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- Per-plane per-strip:
  - `Y_mea_i_sj` (arb).
  - `T_sum_meas_i_sj` (ns).
  - `T_diff_i_sj` (ns): `X_mea * (3 / (2 * c_mm_per_ns))`.
  - `q_diff_i_sj` (arb): Gaussian noise with sigma `qdiff_frac * Y_mea`.

## Behavior
- `q_diff` is zero where `Y_mea <= 0`.
- `T_diff` follows `X_mea` and is NaN when `X_mea` is NaN.

## Metadata
- Common fields plus `source_dataset` and `step_5_id`.

## Failure modes
- Missing strip columns are skipped without warnings.
