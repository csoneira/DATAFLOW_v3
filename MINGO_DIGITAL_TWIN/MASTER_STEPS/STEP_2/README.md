Step 2 (Generated -> Crossing)

Purpose:
- Propagate muons through station geometry and compute plane crossings/times.

Inputs:
- config:
  - config_step_2_physics.yaml
  - config_step_2_runtime.yaml
- data: INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.(pkl|csv)
- INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/PLOTS/geom_<G>_plots.pdf

Run:
- python3 step_2_generated_to_crossing.py --config config_step_2_physics.yaml
- python3 step_2_generated_to_crossing.py --config config_step_2_physics.yaml --runtime-config config_step_2_runtime.yaml
- python3 step_2_generated_to_crossing.py --config config_step_2_physics.yaml --plot-only
