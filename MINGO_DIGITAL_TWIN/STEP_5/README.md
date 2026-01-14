Step 5 (Hit to Signal)

Purpose:
- Compute T_diff from per-strip X_mea and q_diff from qsum.

Inputs:
- config: config_step_5.yaml
- Step 4 outputs in ../STEP_4_TO_5/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_5_TO_6/sim_run_registry.json
- ../STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.pkl (or .csv)
- ../STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.pkl.meta.json
- ../STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal_summary.pdf

Run:
- python3 step_5_measured_to_triggered.py --config config_step_5.yaml
