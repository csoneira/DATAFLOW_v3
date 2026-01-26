# STEP 06 Interface Contract (Signal -> Front/Back)

## Purpose
Convert sum/difference observables into front/back times and charges.

## Required inputs
- Input data (from STEP 05):
  - `event_id`.
  - `T_sum_meas_i_sj`, `T_diff_i_sj`.
  - `Y_mea_i_sj`, `q_diff_i_sj`.

## Output schema
Outputs:
- `INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/step_6.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- Per-plane per-strip:
  - `T_front_i_sj`, `T_back_i_sj` (ns).
  - `Q_front_i_sj`, `Q_back_i_sj` (arb).

## Behavior
- `T_front = T_sum_meas - T_diff`.
- `T_back  = T_sum_meas + T_diff`.
- `Q_front = Y_mea - q_diff`.
- `Q_back  = Y_mea + q_diff`.

## Metadata
- Common fields plus `source_dataset` and `step_6_id`.

## Failure modes
- Missing strip columns are skipped without warnings.
