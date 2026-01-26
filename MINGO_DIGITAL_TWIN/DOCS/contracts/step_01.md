# STEP 01 Interface Contract (Blank -> Generated)

## Purpose
Generate primary muon parameters (position, direction, thick-time tags) for downstream transport.

## Required inputs
- Input data: none (pipeline start).
- Config inputs:
  - `config_step_1_physics.yaml` and `config_step_1_runtime.yaml`.
  - If `cos_n` or `flux_cm2_min` is `random`, `param_mesh.csv` is required.

## Output schema
Outputs:
- `INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.(pkl|csv)`
- Optional chunk manifest: `muon_sample_<N>.chunks.json`

Columns:
- `event_id` (int): 0-based unique identifier.
- `X_gen`, `Y_gen`, `Z_gen` (mm).
- `Theta_gen`, `Phi_gen` (rad).
- `T_thick_s` (s): integer second tag; may be all zeros if rate is zero.

## Behavior
- Sampling uses uniform x/y, uniform phi, and `Theta_gen = arccos(U^(1/(cos_n+1)))`.
- Thick-time tags use a Poisson count-per-second process derived from `flux_cm2_min`.
- Events from the final incomplete thick-time second are dropped.

## Metadata
Stored in `.meta.json` or chunk manifest:
- `step`, `created_at`, `config`, `runtime_config`, `sim_run`.
- `config_hash`, `upstream_hash`.
- `param_set_id`, `param_date`, `param_row_id`, `step_1_id` (when mesh-driven).
- `param_mesh_path`.

## Failure modes
- Missing configs or param mesh raise `FileNotFoundError`/`ValueError`.
- If all candidate mesh rows are already simulated, the step aborts with a message.
