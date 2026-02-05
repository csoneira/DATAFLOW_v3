# STEP FINAL Interface Contract (DAQ -> Station .dat)

## Purpose
Format STEP 10 outputs into station-style .dat files and register outputs.

## Required inputs
- Input data (from STEP 10):
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`.
  - `T_thick_s` (optional; used for timestamp offsets).
- Config inputs:
  - Runtime selection (`input_dir`, `input_sim_run`, `input_collect`).
  - Sampling (`target_rows`, `payload_sampling`).
  - Output control (`files_per_station_conf`, `rate_hz`).
  - Parameter mesh location (`param_mesh_dir`).

## Output schema
Outputs:
- `SIMULATED_DATA/mi00YYDDDHHMMSS.dat` (ASCII text).
- `SIMULATED_DATA/step_final_output_registry.json`.
- `SIMULATED_DATA/step_final_simulation_params.csv`.

Each .dat file may begin with a comment line:
`# param_hash=<sha256>` matching the `param_hash` column in
`step_final_simulation_params.csv`.

Each .dat line contains:
- Timestamp header: `YYYY MM DD HH MM SS 1`.
- 64 channel values ordered by plane [4,3,2,1], field [T_front, T_back, Q_front, Q_back],
  strip [1..4].

## Behavior
- Samples rows from STEP 10 via reservoir (`random`) or contiguous block
  (`sequential_random_start`).
- If `T_thick_s` is present, timestamps are offsets from a base date; otherwise
  inter-arrival times follow an exponential distribution with mean `1 / rate_hz`.
- Updates `param_mesh.csv` with `param_set_id`, `param_date`, and `done=1` for
  matched parameter rows.

## Metadata
- Registry entries include:
  - `param_set_id`, `param_date`, `source_dataset`, `target_rows`, `selected_rows`.
  - `baseline_meta` (upstream metadata snapshot).

## Failure modes
- Missing upstream metadata (z positions or efficiencies) raises `ValueError`.
- Inconsistent presence of `T_thick_s` across input chunks raises `ValueError`.
- Ambiguous param mesh matches raise `ValueError`.
