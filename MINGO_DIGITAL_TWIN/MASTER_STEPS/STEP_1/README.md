Step 1 (Muon Sample)

Purpose:
- Generate independent muon start points and directions (X, Y, Z, theta, phi).

Inputs:
- config: config_step_1.yaml
  - chunk_rows: when set, write in batches to reduce memory usage (csv or pkl chunked output)
  - plot_sample_rows: plot only a sample from the last chunk when chunk_rows is set
  - partial chunks are dropped unless the dataset fits in a single chunk

Outputs:
- ../../INTERSTEPS/STEP_1_TO_2/sim_run_registry.json
- ../../INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.pkl (or .csv)
- ../../INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.pkl.meta.json
- ../../INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/plots/muon_sample_<N>_plots.pdf

Run:
- python3 step_1_blank_to_generated.py --config config_step_1.yaml
- python3 step_1_blank_to_generated.py --config config_step_1.yaml --plot-only
