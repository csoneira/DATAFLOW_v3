# STEP 04 Interface Contract (Avalanche -> Hit)

## Purpose
Induce per-strip signals from avalanche centroids and sizes, producing strip-level charge and time measurements.

## Required inputs
- Input data (from STEP 03):
  - `event_id` (int)
  - `avalanche_size_electrons_i`, `avalanche_x_i`, `avalanche_y_i` for planes i = 1..4.
  - `T_sum_i_ns` for planes i = 1..4 (used to build per-strip times).
- Config inputs:
  - strip geometry (from `STEP_SHARED.get_strip_geometry`), charge sharing settings, `time_sigma_ns`, `x_noise`.
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
All STEP 03 columns plus the following:
- `avalanche_width_scale_i` (unitless): width scale factor for plane i.
- `avalanche_scaled_width_i` (mm): induced width used for charge sharing.
- Per-plane, per-strip columns (plane i = 1..4, strip j = 1..4):
  - `Y_mea_i_sj` (arb charge): induced charge on strip j.
  - `X_mea_i_sj` (mm): inferred x position along strip j (NaN if no hit).
  - `T_sum_meas_i_sj` (ns): measured sum-time per strip (NaN if no hit).
- `tt_hit` (string): concatenation of planes with at least one strip hit.

Time reference: `T_sum_meas_i_sj` is derived from `T_sum_i_ns` with added Gaussian noise; time zero remains the earliest plane crossing per event.

## Invariants & checks
- `Y_mea_i_sj` is >= 0; if 0, `X_mea_i_sj` and `T_sum_meas_i_sj` are NaN.
- If any `Y_mea_i_sj` > 0 for plane i, then plane i appears in `tt_hit`.

## Failure modes & validation behavior
- Missing avalanche inputs for a plane cause that plane to be skipped.
- No explicit validation beyond basic numeric operations; NaNs propagate for missing inputs.
