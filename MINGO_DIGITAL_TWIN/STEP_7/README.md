Step 7 (Front/Back to Calibrated)

Purpose:
- Apply per-strip offsets to T_front/T_back and Q_front/Q_back.

Inputs:
- config: config_step_7.yaml
- Step 6 outputs in ../STEP_6_TO_7/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_7_TO_8/sim_run_registry.json
- ../STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.pkl (or .csv)
- ../STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.pkl.meta.json
- ../STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated_summary.pdf

Run:
- python3 step_7_timing_to_calibrated.py --config config_step_7.yaml
