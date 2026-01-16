Step 3 (Crossing -> Avalanche)

Purpose:
- Apply efficiencies and avalanche model to generate avalanche size/position.

Inputs:
- config:
  - config_step_3_physics.yaml
  - config_step_3_runtime.yaml
- data: INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.(pkl|csv)
- INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/PLOTS/geom_<G>_avalanche_plots.pdf

Run:
- python3 step_3_crossing_to_hit.py --config config_step_3_physics.yaml
- python3 step_3_crossing_to_hit.py --config config_step_3_physics.yaml --runtime-config config_step_3_runtime.yaml
- python3 step_3_crossing_to_hit.py --config config_step_3_physics.yaml --plot-only
