Step 3 (Crossing -> Avalanche)

Purpose:
- Simulate RPC gas ionization in the active volume (avalanche size, position, time); no readout effects.

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

Notes:
- efficiencies can be a single 4-value list or a list of 4-value lists; one is selected per run (seeded).
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
