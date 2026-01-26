# STEP 02 Interface Contract (Generated -> Crossing)

## Purpose
Propagate muons through geometry and compute per-plane crossing positions and times.

## Required inputs
- Input data (from STEP 01):
  - `event_id`, `X_gen`, `Y_gen`, `Z_gen`, `Theta_gen`, `Phi_gen`.
  - `T_thick_s` is preserved if present.
- Config inputs:
  - `z_positions` (4 values) or `random` (param mesh).
  - `bounds_mm` (x/y acceptance).
  - `c_mm_per_ns` (speed of light in mm/ns).

## Output schema
Outputs:
- `INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/step_2.(pkl|csv|chunks.json)`

Columns:
- `event_id` (int)
- `T_thick_s` (s)
- Per-plane (i = 1..4):
  - `X_gen_i`, `Y_gen_i` (mm)
  - `Z_gen_i` (mm)
  - `T_sum_i_ns` (ns)
- `tt_crossing` (string): planes with in-bounds crossings.

## Behavior
- Crossings are computed by straight-line projection and filtered by `bounds_mm`.
- `T_sum_i_ns` is normalized so the earliest valid plane per event is at time 0.
- Events with no valid crossings are dropped.

## Metadata
Stored in `.meta.json` or chunk manifest:
- Common fields plus `z_positions_mm`, `z_positions_raw_mm`.
- `param_set_id`, `param_date`, `param_row_id`, `step_1_id`, `step_2_id` (mesh-driven).

## Failure modes
- Missing mesh rows for `random` z positions raise `ValueError`.
- If all candidate step_2_id combinations are already simulated, the step is skipped.
