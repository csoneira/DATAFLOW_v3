Step 2 (Generated -> Crossing)

Purpose:
- Propagate muon trajectories through the station geometry and compute per-plane incidence time and position.

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

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- normalize_to_first_plane is configured in the runtime config.
- The step skips if the matching SIM_RUN exists unless --force is provided.
