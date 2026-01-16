Step 11 (DAQ -> Detector Text) [deprecated]

Purpose:
- Serialize per-event data into detector-style text rows.

Inputs:
- config:
  - config_step_11_physics.yaml
  - config_step_11_runtime.yaml
- data: INTERSTEPS/STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_11_TO_12/SIM_RUN_<N>/mi00YYDDDHHMMSS.dat

Run:
- Deprecated: use STEP_FINAL instead.
- python3 ../STEP_FINAL/step_final_daq_to_station_dat.py --config ../STEP_FINAL/config_step_final_physics.yaml
