STEP_FINAL: DAQ -> Station .dat (event builder formatting)

Inputs
- data: INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/geom_*_daq.(pkl|csv|chunks.json)
- config: config_step_final_physics.yaml
- runtime: config_step_final_runtime.yaml

Outputs
- SIMULATED_DATA/SIM_RUN_<N>/STATION_<S>_CONFIG_<C>/mi0XYYDDDHHMMSS.dat
- SIMULATED_DATA/step_final_output_registry.json

Usage
- python3 step_final_daq_to_station_dat.py --config config_step_final_physics.yaml
- python3 step_final_daq_to_station_dat.py --config config_step_final_physics.yaml --runtime-config config_step_final_runtime.yaml

Notes
- This step emulates the event builder: formats TDC events into the station .dat layout.
- This step emits ASCII for the first time and does not create SIM_RUN output directories.
- input_sim_run and geometry_map_sim_run accept explicit SIM_RUN_<N>, latest, random, or all (runtime config).
- files_per_station_conf controls how many files to generate per (station, conf) within each SIM_RUN.
- payload_sampling can be random (reservoir) or sequential_random_start (contiguous rows).
