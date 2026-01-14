Step 3 (Crossing to Avalanche)

Purpose:
- Generate avalanche ionizations and centers per plane.
- Poisson ionizations model intrinsic inefficiency with P(ion=0)=1-eff.

Inputs:
- config: config_step_3.yaml
- Step 2 outputs in ../STEP_2_TO_3/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_3_TO_4/sim_run_registry.json
- ../STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.pkl (or .csv)
- ../STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.pkl.meta.json
- ../STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche_summary.pdf
- ../STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche_single.pdf

Run:
- python3 step_3_crossing_to_hit.py --config config_step_3.yaml
- python3 step_3_crossing_to_hit.py --config config_step_3.yaml --plot-only
