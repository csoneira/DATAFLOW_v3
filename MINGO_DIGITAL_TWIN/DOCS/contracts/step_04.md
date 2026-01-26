# STEP 04 Interface Contract (Avalanche -> Hit)

## Purpose
Induce per-strip signals from avalanche centroids and sizes.

## Required inputs
- Input data (from STEP 03):
  - `event_id`.
  - `avalanche_size_electrons_i`, `avalanche_x_i`, `avalanche_y_i`.
  - `T_sum_i_ns` (for time assignment).
- Config inputs:
  - `charge_share_points`, `avalanche_width_mm`, `width_scale_exponent`, `width_scale_max`.
  - `x_noise_mm`, `time_sigma_ns`.

## Output schema
Outputs:
- `INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/step_4.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- `tt_hit` (string): planes with any strip hit.
- Per-plane per-strip (i = 1..4, j = 1..4):
  - `Y_mea_i_sj` (arb): induced charge on strip j.
  - `X_mea_i_sj` (mm): x position (NaN if no hit on strip j).
  - `T_sum_meas_i_sj` (ns): noisy sum time (NaN if no hit on strip j).

## Behavior
- A disk-overlap model computes the fraction of avalanche charge per strip.
- `Y_mea` is charge (not a Y position).
- `X_mea` and `T_sum_meas` are only assigned for strips with `Y_mea > 0`.

## Metadata
- Common fields plus `source_dataset` and `step_4_id`.

## Failure modes
- Missing avalanche columns for a plane cause that plane to be skipped.
