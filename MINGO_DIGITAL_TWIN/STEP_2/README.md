Step 2 (Geometry Expansion + Intersections)

Purpose:
- Build station/conf geometry map from ONLINE_RUN_DICTIONARY.
- For the selected geometry_id, intersect muon tracks with planes and save a per-geometry dataset.

Inputs:
- config: config_step_2.yaml
- muon sample from Step 1 (via input_dir/input_sim_run/input_basename or input_muon_sample)
- station configs: /home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY

Outputs:
- ../STEP_2_TO_3/sim_run_registry.json
- ../STEP_2_TO_3/SIM_RUN_<N>/geometry_registry.csv
- ../STEP_2_TO_3/SIM_RUN_<N>/geometry_registry.json
- ../STEP_2_TO_3/SIM_RUN_<N>/geometry_map_all.csv
- ../STEP_2_TO_3/SIM_RUN_<N>/geometry_map_all.json
- ../STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.pkl (or .csv)
- ../STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.pkl.meta.json
- ../STEP_2_TO_3/SIM_RUN_<N>/geom_<G>_summary.pdf
- ../STEP_2_TO_3/SIM_RUN_<N>/geom_<G>_single.pdf

Run:
- python3 step_2_generated_to_crossing.py --config config_step_2.yaml
- python3 step_2_generated_to_crossing.py --config config_step_2.yaml --plot-only
