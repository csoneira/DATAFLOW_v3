Step 1 (Muon Sample)

Purpose:
- Generate independent muon start points and directions (X, Y, Z, theta, phi).

Inputs:
- config: config_step_1.yaml

Outputs:
- ../STEP_1_TO_2/sim_run_registry.json
- ../STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.pkl (or .csv)
- ../STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.pkl.meta.json
- ../STEP_1_TO_2/SIM_RUN_<N>/plots/muon_sample_<N>_summary.pdf
- ../STEP_1_TO_2/SIM_RUN_<N>/plots/muon_sample_<N>_single.pdf

Run:
- python3 step_1_blank_to_generated.py --config config_step_1.yaml
- python3 step_1_blank_to_generated.py --config config_step_1.yaml --plot-only
