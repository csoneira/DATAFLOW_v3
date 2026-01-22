Step 8 (Threshold / FEE, uncalibration)

Purpose:
- Simulate the FEE: apply charge thresholds and convert charge to timing (uncalibration/decalibration step).

Inputs:
- config:
  - config_step_8_physics.yaml
  - config_step_8_runtime.yaml
- data: INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/step_7.(pkl|csv|chunks.json)

Outputs:
- INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/step_8.(pkl|csv|chunks.json)
- INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/PLOTS/step_8_plots.pdf

Run:
- python3 step_8_uncalibrated_to_threshold.py --config config_step_8_physics.yaml
- python3 step_8_uncalibrated_to_threshold.py --config config_step_8_physics.yaml --runtime-config config_step_8_runtime.yaml
- python3 step_8_uncalibrated_to_threshold.py --config config_step_8_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
