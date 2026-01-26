# STEP 07 Interface Contract (Front/Back -> Uncalibrated)

## Purpose
Apply per-channel cable/connector offsets to front/back times.

## Required inputs
- Input data (from STEP 06):
  - `event_id`.
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`.
- Config inputs:
  - `tfront_offsets`, `tback_offsets` (4x4 arrays, ns).

## Output schema
Outputs:
- `INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/step_7.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- Per-plane per-strip: `T_front`, `T_back`, `Q_front`, `Q_back`.

## Behavior
- Offsets are added to nonzero, finite `T_front` and `T_back` values.
- Charges are preserved unchanged.

## Metadata
- Common fields plus `source_dataset` and `step_7_id`.

## Failure modes
- Offset arrays with incorrect dimensions may raise `IndexError`.
- Missing strip columns are skipped without warnings.
