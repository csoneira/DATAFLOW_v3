# STEP 07 Interface Contract (Front/Back -> Calibrated)

## Purpose
Apply per-channel connector/cable timing offsets (uncalibration/decalibration) to front/back times.

## Required inputs
- Input data (from STEP 06):
  - `event_id` (int)
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`.
- Config inputs:
  - `tfront_offsets` and `tback_offsets` arrays (4x4, ns).
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
- No new columns are added.
- `T_front_i_sj` and `T_back_i_sj` are updated in-place with the configured offsets.
- Charges are preserved unchanged.

Time reference: same as STEP 06, plus per-channel cable offsets.

## Invariants & checks
- For nonzero, finite times, `T_front_i_sj` and `T_back_i_sj` increase by the configured offsets.

## Failure modes & validation behavior
- Offsets with incorrect dimensions may raise `IndexError` during application.
- Missing input columns are skipped without warnings.
