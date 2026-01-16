Step 4 (Avalanche -> Hit)

Purpose:
- Induce strip charges and measured positions/times per plane.

Inputs:
- config:
  - config_step_4_physics.yaml
  - config_step_4_runtime.yaml
- data: INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.(pkl|csv)
- INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/PLOTS/geom_<G>_hit_plots.pdf

Run:
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --runtime-config config_step_4_runtime.yaml
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --plot-only
