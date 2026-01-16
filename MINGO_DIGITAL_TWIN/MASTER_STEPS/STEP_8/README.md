Step 8 (Threshold / FEE)

Purpose:
- Apply charge threshold and front-end conversion to time units.

Inputs:
- config:
  - config_step_8_physics.yaml
  - config_step_8_runtime.yaml
- data: INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.(pkl|csv)
- INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/PLOTS/geom_<G>_threshold_plots.pdf

Run:
- python3 step_8_calibrated_to_threshold.py --config config_step_8_physics.yaml
- python3 step_8_calibrated_to_threshold.py --config config_step_8_physics.yaml --runtime-config config_step_8_runtime.yaml
- python3 step_8_calibrated_to_threshold.py --config config_step_8_physics.yaml --plot-only
