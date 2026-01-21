# STEP 01 Interface Contract (Blank -> Generated)

## Purpose
Generate primary muon parameters (position, direction, and thick-time tags) for downstream transport and detector simulation.

## Required inputs
- Input data: none (this step is the pipeline start).
- Config inputs:
  - `config_step_1_physics.yaml` and `config_step_1_runtime.yaml`.
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
- `event_id` (int): persistent per-event identifier (0-based, unique within the run).
- `X_gen` (mm): generated x position at the generation plane.
- `Y_gen` (mm): generated y position at the generation plane.
- `Z_gen` (mm): generated z position (constant per run).
- `Theta_gen` (rad): polar angle from +Z, in [0, pi/2].
- `Phi_gen` (rad): azimuth in [-pi, pi].
- `T_thick_s` (s): wall-clock-like time for thick-rate sequencing (0 if unused).

Time reference: this step does not persist a per-event time origin; downstream times are defined relative to the earliest plane crossing in STEP 2.

## Invariants & checks
- `event_id` is unique within the output file and 0-based.
- `Z_gen` is constant for all rows in a run.
- `Theta_gen` and `Phi_gen` are finite for all rows.
- `T_thick_s` is non-negative; may be all zeros if thick-rate mode is off.

## Failure modes & validation behavior
- Missing or invalid config files raise `FileNotFoundError` or `ValueError`.
- If `drop_last_second` is enabled, events from the last thick-time second are removed (informational print).
- No explicit column validation is performed beyond config sanity checks.
