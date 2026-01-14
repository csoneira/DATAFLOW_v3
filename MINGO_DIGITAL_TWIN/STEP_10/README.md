Step 10 (DAQ Jitter)

Purpose:
- Apply per-event DAQ jitter to T_front/T_back when channel info is present.

Inputs:
- config: config_step_10.yaml
- Step 9 outputs in ../STEP_9_TO_10/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_10_TO_11/sim_run_registry.json
- ../STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq.pkl (or .csv)
- ../STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq.pkl.meta.json
- ../STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq_summary.pdf

Run:
- python3 step_10_triggered_to_jitter.py --config config_step_10.yaml
