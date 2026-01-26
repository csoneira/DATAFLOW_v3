# STEP 00 Interface Contract (Setup -> Parameter Mesh)

## Purpose
Generate or extend `param_mesh.csv` used for step ID selection and parameter scans.

## Required inputs
- Config inputs:
  - `config_step_0_physics.yaml`
  - `config_step_0_runtime.yaml`
- External data:
  - Station configuration CSVs under `station_config_root` (P1..P4 columns).

## Output schema
Outputs:
- `INTERSTEPS/STEP_0_TO_1/param_mesh.csv`
- `INTERSTEPS/STEP_0_TO_1/param_mesh_metadata.json`

`param_mesh.csv` columns (core):
- `done` (int): 0/1 completion flag.
- `step_1_id`..`step_10_id` (string): 3-digit step IDs.
- `cos_n`, `flux_cm2_min`.
- `eff_p1..eff_p4`.
- `z_p1..z_p4` (mm).
- Optional: `param_set_id`, `param_date`.

## Behavior
- Samples parameter values within configured ranges.
- Assigns step IDs by unique combinations.
- Appends new rows unless existing mesh is incomplete (unless `--force`).

## Metadata
`param_mesh_metadata.json` includes:
- `created_at`, `updated_at`, `row_count`.
- `repeat_samples`, `shared_columns`, `expand_z_positions`.
- `step` and configs.

## Failure modes
- Missing station configs or empty geometry lists raise `FileNotFoundError`/`ValueError`.
- Invalid ranges or shared column settings raise `ValueError`.
