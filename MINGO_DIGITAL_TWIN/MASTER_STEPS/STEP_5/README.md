Step 5 (Hit -> Signal)

Purpose:
- Complete readout quantities: charge imbalance and time-difference derived from X position within strips.

Inputs:
- config:
  - config_step_5_physics.yaml
  - config_step_5_runtime.yaml
- data: INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/step_4.(pkl|csv|chunks.json)

Outputs:
- INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/step_5.(pkl|csv|chunks.json)
- INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/PLOTS/step_5_plots.pdf

Run:
- python3 step_5_measured_to_triggered.py --config config_step_5_physics.yaml
- python3 step_5_measured_to_triggered.py --config config_step_5_physics.yaml --runtime-config config_step_5_runtime.yaml
- python3 step_5_measured_to_triggered.py --config config_step_5_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
