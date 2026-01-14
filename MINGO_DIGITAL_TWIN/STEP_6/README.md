Step 6 (Signal to Front/Back)

Purpose:
- Compute T_front/T_back and Q_front/Q_back from T_sum_meas/T_diff and qsum/q_diff.

Inputs:
- config: config_step_6.yaml
- Step 5 outputs in ../STEP_5_TO_6/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_6_TO_7/sim_run_registry.json
- ../STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback.pkl (or .csv)
- ../STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback.pkl.meta.json
- ../STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback_summary.pdf

Run:
- python3 step_6_triggered_to_timing.py --config config_step_6.yaml
