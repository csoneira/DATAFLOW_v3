Step 6 (Signal -> Front/Back)

Purpose:
- Propagate strip signals to both ends, producing front/back times and charges (pre-uncalibration).

Inputs:
- config:
  - config_step_6_physics.yaml
  - config_step_6_runtime.yaml
- data: INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/step_5.(pkl|csv|chunks.json)

Outputs:
- INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/step_6.(pkl|csv|chunks.json)
- INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/PLOTS/step_6_plots.pdf

Run:
- python3 step_6_triggered_to_timing.py --config config_step_6_physics.yaml
- python3 step_6_triggered_to_timing.py --config config_step_6_physics.yaml --runtime-config config_step_6_runtime.yaml
- python3 step_6_triggered_to_timing.py --config config_step_6_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
