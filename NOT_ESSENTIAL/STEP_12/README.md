Step 12 (Detector Text -> Station Dat) [deprecated]

Purpose:
- Assign station/conf time ranges, apply Poisson timestamps, and emit station .dat files.

Inputs:
- config:
  - config_step_12_physics.yaml
  - config_step_12_runtime.yaml
- data: INTERSTEPS/STEP_11_TO_12/SIM_RUN_<N>/*.dat

Outputs:
- SIMULATED_DATA/mi0XYYDDDHHMMSS.dat

Run:
- Deprecated: use STEP_FINAL instead.
- python3 ../STEP_FINAL/step_final_daq_to_station_dat.py --config ../STEP_FINAL/config_step_final_physics.yaml
