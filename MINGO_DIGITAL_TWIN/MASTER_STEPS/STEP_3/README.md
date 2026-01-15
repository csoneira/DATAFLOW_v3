Step 3 (Crossing to Avalanche)

Purpose:
- Generate avalanche ionizations and centers per plane.
- Poisson ionizations model intrinsic inefficiency with P(ion=0)=1-eff.

Inputs:
- config:
- config_step_3_physics.yaml
- config_step_3_runtime.yaml
- Step 2 outputs in ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_3_TO_4/sim_run_registry.json
- ../../INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.pkl (or .csv)
- ../../INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.pkl.meta.json
- ../../INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche_plots.pdf

Run:
- python3 step_3_crossing_to_hit.py --config config_step_3_physics.yaml
- python3 step_3_crossing_to_hit.py --config config_step_3_physics.yaml --plot-only
