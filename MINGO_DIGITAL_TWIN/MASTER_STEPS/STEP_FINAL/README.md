STEP_FINAL: DAQ -> Station .dat (final formatting)

Inputs
- data: INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/geom_*_daq.(pkl|csv|chunks.json)
- config: config_step_final_physics.yaml
- runtime: config_step_final_runtime.yaml

Outputs
- SIMULATED_DATA/mi0XYYDDDHHMMSS.dat
- SIMULATED_DATA/step_13_output_registry.json

Usage
- python3 step_final_daq_to_station_dat.py --config config_step_final_physics.yaml
- python3 step_final_daq_to_station_dat.py --config config_step_final_physics.yaml --runtime-config config_step_final_runtime.yaml

Notes
- This step emits ASCII for the first time and does not create SIM_RUN output directories.
