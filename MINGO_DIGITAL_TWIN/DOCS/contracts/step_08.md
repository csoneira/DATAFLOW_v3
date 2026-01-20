# STEP 08 Interface Contract (Calibrated -> Threshold)

## Purpose
Apply front-end electronics effects: per-channel time jitter, charge-to-time conversion, and thresholding.

## Required inputs
- Input data (from STEP 07):
  - `event_id` (int)
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`.
- Config inputs:
  - `t_fee_sigma_ns` (time jitter per channel).
  - `q_to_time_factor`, `qfront_offsets`, `qback_offsets` (charge-to-time conversion).
  - `threshold` (minimum Q to keep channel active).
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
- No new columns are added.
- `T_front_i_sj` and `T_back_i_sj` are updated with Gaussian jitter.
- `Q_front_i_sj` and `Q_back_i_sj` are converted to time-walk units and thresholded:
  - values below `threshold` are set to 0.

Time reference: same as STEP 07, with additional per-channel FEE timing jitter.

## Invariants & checks
- `Q_front_i_sj` and `Q_back_i_sj` are zero for channels below threshold.
- Converted Q values reflect `q_to_time_factor` and per-channel offsets when above threshold.

## Failure modes & validation behavior
- Missing input columns are skipped without warnings.
- No explicit validation is performed on threshold sign or offset array dimensions.
