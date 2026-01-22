Step 10 (DAQ Timing)

Purpose:
- Apply TDC timing smear and DAQ clock jitter to coincidence events.

Inputs:
- config:
  - config_step_10_physics.yaml
  - config_step_10_runtime.yaml
- data: INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/step_9.(pkl|csv|chunks.json)

Outputs:
- INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/step_10.(pkl|csv|chunks.json)
- INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/PLOTS/step_10_plots.pdf

Run:
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml --runtime-config config_step_10_runtime.yaml
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
