Step 12 (Detector Format with Station Timing)

Purpose:
- Select a station/conf pair for the geometry_id, pick a random start time within its date range,
  and assign per-event timestamps using a Poisson process (rate_hz).

Inputs:
- config: config_step_12.yaml
- Step 11 outputs in ../STEP_11_TO_12/SIM_RUN_<N>
- geometry_map_all.csv from Step 2 outputs (via geometry_map_dir/geometry_map_sim_run)

Outputs:
- ../STEP_12_TO_13/sim_run_registry.json
- ../STEP_12_TO_13/SIM_RUN_<N>/mi0XYYDDDHHMMSS.dat
- ../STEP_12_TO_13/SIM_RUN_<N>/mi0XYYDDDHHMMSS.dat.meta.json

Run:
- python3 step_12_detector_to_station_dat.py --config config_step_12.yaml
