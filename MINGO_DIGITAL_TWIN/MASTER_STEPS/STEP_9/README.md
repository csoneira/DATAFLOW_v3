Step 9 (Trigger at DAQ)

Purpose:
- Apply trigger combinations using per-plane channel activity.

Inputs:
- config:
- config_step_9_physics.yaml
- config_step_9_runtime.yaml
- Step 8 outputs in ../../INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_9_TO_10/sim_run_registry.json
- ../../INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.pkl (or .csv)
- ../../INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.pkl.meta.json
- ../../INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered_plots.pdf

Run:
- python3 step_9_threshold_to_trigger.py --config config_step_9_physics.yaml
