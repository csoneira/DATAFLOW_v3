# STEP 0 (Parameter Mesh Setup)

Purpose:
- Build or extend `INTERSTEPS/STEP_0_TO_1/param_mesh.csv` with sampled parameters.
- Assign step IDs (`step_1_id`..`step_10_id`) that define SIM_RUN naming downstream.

Inputs:
- Physics config: `config_step_0_physics.yaml`
- Runtime config: `config_step_0_runtime.yaml`
- Station config CSVs under `station_config_root` (P1..P4 plane positions).

Outputs:
- `INTERSTEPS/STEP_0_TO_1/param_mesh.csv`
- `INTERSTEPS/STEP_0_TO_1/param_mesh_metadata.json`

Key behavior:
- Samples `cos_n`, `flux_cm2_min`, efficiencies, and z-plane tuples.
- Supports `repeat_samples`, `shared_columns`, and `expand_z_positions` for scan design.
- Appends only when all existing rows are marked `done=1` unless `--force` is provided.

Run:
```
python3 step_0_setup_to_blank.py --config config_step_0_physics.yaml
```

Notes:
- Step IDs are normalized to 3-digit strings.
- `step_1_id` is based on (cos_n, flux_cm2_min), `step_2_id` on (z_p1..z_p4),
  `step_3_id` on (eff_p1..eff_p4). Steps 4..10 default to "001".
