Step 4 (Avalanche to Hit)

Purpose:
- Induce strip signals from avalanche centers and size.
- Produce per-strip X_mea, T_sum_meas, and Y_mea (qsum) vectors.

Inputs:
- config:
- config_step_4_physics.yaml
- config_step_4_runtime.yaml
- Step 3 outputs in ../../INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_4_TO_5/sim_run_registry.json
- ../../INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.pkl (or .csv)
- ../../INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.pkl.meta.json
- ../../INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit_plots.pdf

Run:
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml
- python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --plot-only
