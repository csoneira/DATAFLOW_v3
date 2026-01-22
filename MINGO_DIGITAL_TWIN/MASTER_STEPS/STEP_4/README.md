Step 4 (Avalanche -> Hit)

Purpose:
- Use avalanche centers and sizes to estimate induction area and affected strips (readout starts here).

Inputs:
- config:
  - config_step_4_physics.yaml
  - config_step_4_runtime.yaml
- data: INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/step_3.(pkl|csv|chunks.json)

Outputs:
- INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/step_4.(pkl|csv|chunks.json)
- INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/PLOTS/step_4_plots.pdf

Run:
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --runtime-config config_step_4_runtime.yaml
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
