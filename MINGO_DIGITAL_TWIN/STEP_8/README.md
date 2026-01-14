Step 8 (Threshold)

Purpose:
- Apply charge threshold to Q_front/Q_back; values below threshold become 0.

Inputs:
- config: config_step_8.yaml
- Step 7 outputs in ../STEP_7_TO_8/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_8_TO_9/sim_run_registry.json
- ../STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.pkl (or .csv)
- ../STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.pkl.meta.json
- ../STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold_summary.pdf

Run:
- python3 step_8_calibrated_to_threshold.py --config config_step_8.yaml
