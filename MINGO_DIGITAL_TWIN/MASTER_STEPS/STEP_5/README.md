Step 5 (Hit to Signal)

Purpose:
- Compute T_diff from per-strip X_mea and q_diff from qsum.

Inputs:
- config:
- config_step_5_physics.yaml
- config_step_5_runtime.yaml
- Step 4 outputs in ../../INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_5_TO_6/sim_run_registry.json
- ../../INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.pkl (or .csv)
- ../../INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.pkl.meta.json
- ../../INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal_plots.pdf

Run:
- python3 step_5_measured_to_triggered.py --config config_step_5_physics.yaml
