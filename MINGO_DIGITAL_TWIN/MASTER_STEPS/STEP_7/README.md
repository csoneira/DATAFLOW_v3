Step 7 (Front/Back to Calibrated)

Purpose:
- Apply per-strip offsets to T_front/T_back (cable propagation).

Inputs:
- config: config_step_7.yaml
- Step 6 outputs in ../../INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_7_TO_8/sim_run_registry.json
- ../../INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.pkl (or .csv)
- ../../INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.pkl.meta.json
- ../../INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated_plots.pdf

Run:
- python3 step_7_timing_to_calibrated.py --config config_step_7.yaml
