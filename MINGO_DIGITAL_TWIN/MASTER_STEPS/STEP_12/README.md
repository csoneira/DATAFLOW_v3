Step 12 (Detector Format with Station Timing)

Purpose:
- Select a station/conf pair for the geometry_id, pick a random start time within its date range,
  and assign per-event timestamps using a Poisson process (rate_hz).

Inputs:
- config:
- config_step_12_physics.yaml
- config_step_12_runtime.yaml
- Step 11 outputs in ../../INTERSTEPS/STEP_11_TO_12/SIM_RUN_<N>
- geometry_map_all.csv from Step 2 outputs (via geometry_map_dir/geometry_map_sim_run)
  - target_rows: number of events to sample into the output file
  - input_collect: baseline_only | matching
  - when Step 11 is chunked, multiple .dat inputs are sampled together

Outputs:
- ../../SIMULATED_DATA/mi0XYYDDDHHMMSS.dat
- step_13_output_registry.json (in SIMULATED_DATA)
  - When input_collect=matching, rows are sampled across matching Step 11 outputs with the same
    config_hash and upstream_hash as the baseline input_sim_run.

Run:
- python3 step_12_detector_to_station_dat.py --config config_step_12_physics.yaml
