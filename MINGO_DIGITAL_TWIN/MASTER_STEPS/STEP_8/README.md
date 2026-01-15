Step 8 (Threshold at FEE)

Purpose:
- Apply FEE effects: smear T_front/T_back and transform Q_front/Q_back to ns, then apply threshold.

Inputs:
- config:
- config_step_8_physics.yaml
- config_step_8_runtime.yaml
- Step 7 outputs in ../../INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_8_TO_9/sim_run_registry.json
- ../../INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.pkl (or .csv)
- ../../INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.pkl.meta.json
- ../../INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold_plots.pdf

Run:
- python3 step_8_calibrated_to_threshold.py --config config_step_8_physics.yaml
