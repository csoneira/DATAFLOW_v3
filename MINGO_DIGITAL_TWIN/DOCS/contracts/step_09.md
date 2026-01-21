# STEP 09 Interface Contract (Threshold -> Trigger)

## Purpose
Apply coincidence trigger logic and retain only events that satisfy configured plane combinations.

## Required inputs
- Input data (from STEP 08):
  - `event_id` (int)
  - `Q_front_i_sj` and/or `Q_back_i_sj` to determine per-plane activity.
- Config inputs:
  - `trigger_combinations` (list of strings such as "1234").
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
- Output is a subset of input rows that pass the trigger.
- Retained columns:
  - `event_id` (int)
  - `T_thick_s` (s) if present upstream
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`
- Added column: `tt_trigger` (string): concatenation of active planes for each retained event.

Time reference: unchanged from STEP 08.

## Invariants & checks
- Every output row satisfies at least one configured trigger combination.
- `tt_trigger` contains only digits in {1,2,3,4}.

## Failure modes & validation behavior
- If `trigger_combinations` is empty, output will be empty (no events retained).
- Missing Q columns for a plane imply that plane cannot contribute to triggering.
