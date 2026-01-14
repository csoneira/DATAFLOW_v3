Step 4 (Avalanche to Hit)

Purpose:
- Induce strip signals from avalanche centers and size.
- Produce per-strip X_mea, T_sum_meas, and Y_mea (qsum) vectors.

Inputs:
- config: config_step_4.yaml
- Step 3 outputs in ../STEP_3_TO_4/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_4_TO_5/sim_run_registry.json
- ../STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.pkl (or .csv)
- ../STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.pkl.meta.json
- ../STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit_summary.pdf
- ../STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit_single.pdf

Run:
- python3 step_4_hit_to_measured.py --config config_step_4.yaml
- python3 step_4_hit_to_measured.py --config config_step_4.yaml --plot-only
