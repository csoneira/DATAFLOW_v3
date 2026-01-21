# STEP 02 Interface Contract (Generated -> Crossing)

## Purpose
Propagate generated muons through station geometry and compute per-plane crossing positions and times.

## Required inputs
- Input data (from STEP 01):
  - `event_id` (int)
  - `X_gen`, `Y_gen`, `Z_gen` (mm)
  - `Theta_gen`, `Phi_gen` (rad)
  - `T0_ns` (ns)
  - `T_thick_s` (s) is optional and preserved if present.
- Config inputs:
  - Geometry registry/station config and `geometry_id` selection.
  - `c_mm_per_ns` (defaults to 299.792458 if not in config).
  - `bounds_mm` for active area acceptance.
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
Retained columns:
- `event_id` (int)
- `T_thick_s` (s) if present upstream

Per-plane columns (plane index i = 1..4):
- `X_gen_i` (mm): projected crossing x for plane i (NaN if out of bounds).
- `Y_gen_i` (mm): projected crossing y for plane i (NaN if out of bounds).
- `Z_gen_i` (mm): plane z position for plane i (NaN if out of bounds).
- `T_sum_i_ns` (ns): flight time to plane i, shifted so the earliest valid plane has time 0.
- `tt_crossing` (string): concatenation of plane indices where the muon crosses within bounds (NaN if none).

Time reference: `T_sum_i_ns` is relative to the earliest in-bounds plane crossing within each event.

## Invariants & checks
- If `X_gen_i` or `Y_gen_i` is NaN, then `Z_gen_i` and `T_sum_i_ns` are NaN for that plane.
- `tt_crossing` contains only digits in {1,2,3,4} and reflects planes with non-NaN crossings.
- `T_sum_i_ns` are all >= 0 for valid planes (after normalization).

## Failure modes & validation behavior
- Missing geometry or station config raises `FileNotFoundError`.
- If all planes are out of bounds, the row is dropped and not written to output.
- No explicit warnings are emitted for out-of-geometry cases; values are set to NaN and filtered.
