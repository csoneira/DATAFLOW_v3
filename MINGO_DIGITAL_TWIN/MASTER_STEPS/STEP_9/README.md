Step 9 (Trigger)

Purpose:
- Apply coincidence trigger logic; keep only allowed plane combinations.

Inputs:
- config:
  - config_step_9_physics.yaml
  - config_step_9_runtime.yaml
- data: INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/step_8.(pkl|csv|chunks.json)

Outputs:
- INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/step_9.(pkl|csv|chunks.json)
- INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/PLOTS/step_9_plots.pdf

Run:
- python3 step_9_threshold_to_trigger.py --config config_step_9_physics.yaml
- python3 step_9_threshold_to_trigger.py --config config_step_9_physics.yaml --runtime-config config_step_9_runtime.yaml
- python3 step_9_threshold_to_trigger.py --config config_step_9_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
