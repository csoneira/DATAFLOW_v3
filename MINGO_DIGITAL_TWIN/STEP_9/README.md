Step 9 (Trigger at DAQ)

Purpose:
- Apply trigger combinations using per-plane channel activity.

Inputs:
- config: config_step_9.yaml
- Step 8 outputs in ../STEP_8_TO_9/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_9_TO_10/sim_run_registry.json
- ../STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.pkl (or .csv)
- ../STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.pkl.meta.json
- ../STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered_summary.pdf

Run:
- python3 step_9_threshold_to_trigger.py --config config_step_9.yaml
