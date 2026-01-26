# STEP 09 Interface Contract (Threshold -> Trigger)

## Purpose
Apply coincidence trigger logic and retain only passing events.

## Required inputs
- Input data (from STEP 08):
  - `event_id`.
  - `Q_front_i_sj` and/or `Q_back_i_sj`.
- Config inputs:
  - `trigger_combinations` (list of plane strings, e.g., "12").

## Output schema
Outputs:
- `INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/step_9.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- Per-plane per-strip: `T_front`, `T_back`, `Q_front`, `Q_back`.
- `tt_trigger` (string): concatenation of active planes.

## Behavior
- A plane is active if any strip has `Q_front > 0` or `Q_back > 0`.
- `tt_trigger` is a string of active planes in ascending order.
- An event passes if any trigger string is a subset of the active planes.
- Events that do not pass are dropped.

## Metadata
- Common fields plus `source_dataset` and `step_9_id`.

## Failure modes
- Empty `trigger_combinations` yields empty output.
- Missing Q columns for a plane prevent that plane from contributing to the trigger.
