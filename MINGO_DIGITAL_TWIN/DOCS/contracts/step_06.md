# STEP 06 Interface Contract (Signal -> Front/Back)

## Purpose
Convert per-strip sum/difference observables into end-specific times and charges (front/back).

## Required inputs
- Input data (from STEP 05):
  - `event_id` (int)
  - `T_sum_meas_i_sj` (ns), `T_diff_i_sj` (ns)
  - `Y_mea_i_sj` (arb charge), `q_diff_i_sj` (arb charge)
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
Retained columns:
- `event_id` (int)
- `T_thick_s` (s) if present upstream

Derived columns:
- `T_front_i_sj` (ns): `T_sum_meas_i_sj - T_diff_i_sj`.
- `T_back_i_sj` (ns): `T_sum_meas_i_sj + T_diff_i_sj`.
- `Q_front_i_sj` (arb charge): `Y_mea_i_sj - q_diff_i_sj`.
- `Q_back_i_sj` (arb charge): `Y_mea_i_sj + q_diff_i_sj`.

Time reference: same event time origin as STEP 02; only per-strip differences are introduced.

## Invariants & checks
- For finite `T_sum_meas_i_sj` and `T_diff_i_sj`, `(T_front_i_sj + T_back_i_sj) / 2 == T_sum_meas_i_sj`.
- For finite `Y_mea_i_sj` and `q_diff_i_sj`, `(Q_front_i_sj + Q_back_i_sj) / 2 == Y_mea_i_sj`.

## Failure modes & validation behavior
- Missing required strip columns cause that strip to be skipped.
- No explicit warnings are emitted; NaNs propagate for missing inputs.
