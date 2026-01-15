Step 10 (DAQ Jitter)

Purpose:
- Apply per-event DAQ jitter to T_front/T_back when channel info is present.

Inputs:
- config:
- config_step_10_physics.yaml
- config_step_10_runtime.yaml
- Step 9 outputs in ../../INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N> (via input_sim_run)
  - chunk_rows: when set, process in chunks and write chunked outputs
  - plot_sample_rows: plot only a sample from the last full chunk

Outputs:
- ../../INTERSTEPS/STEP_10_TO_11/sim_run_registry.json
- ../../INTERSTEPS/STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq.pkl (or .csv)
- ../../INTERSTEPS/STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq.pkl.meta.json
- ../../INTERSTEPS/STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq_plots.pdf

Run:
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml
