Step 6 (Signal -> Front/Back)

Purpose:
- Build front/back timing and charge vectors per strip.

Inputs:
- config:
  - config_step_6_physics.yaml
  - config_step_6_runtime.yaml
- data: INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback.(pkl|csv)
- INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/PLOTS/geom_<G>_frontback_plots.pdf

Run:
- python3 step_6_triggered_to_timing.py --config config_step_6_physics.yaml
- python3 step_6_triggered_to_timing.py --config config_step_6_physics.yaml --runtime-config config_step_6_runtime.yaml
- python3 step_6_triggered_to_timing.py --config config_step_6_physics.yaml --plot-only
