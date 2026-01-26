# STEP FINAL (DAQ -> Station .dat)

Purpose:
- Format STEP 10 data into station-style .dat files and register outputs.

Inputs:
- Data: `INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/step_10.(pkl|csv|chunks.json)`
- Physics config: `config_step_final_physics.yaml` (currently empty)
- Runtime config: `config_step_final_runtime.yaml`

Outputs:
- `SIMULATED_DATA/mi00YYDDDHHMMSS.dat`
- `SIMULATED_DATA/step_final_output_registry.json`
- `SIMULATED_DATA/step_final_simulation_params.csv`

Algorithm highlights:
- Samples `target_rows` events using reservoir or sequential sampling.
- Generates timestamps from `T_thick_s` if present; otherwise uses exponential spacing.
- Writes 71 fields per line: timestamp header + 64 channel values.

Run:
```
python3 step_final_daq_to_station_dat.py --config config_step_final_physics.yaml
python3 step_final_daq_to_station_dat.py --config config_step_final_physics.yaml --runtime-config config_step_final_runtime.yaml
```

Notes:
- `input_sim_run` accepts explicit SIM_RUN, `latest`, `random`, or `all`.
- `input_collect` controls how multiple SIM_RUN inputs are matched.
- STEP FINAL assigns `param_set_id` and `param_date` in the parameter mesh.
