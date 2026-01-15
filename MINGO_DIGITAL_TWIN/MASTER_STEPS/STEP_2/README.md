Step 2 (Geometry Expansion + Intersections)

Purpose:
- Build station/conf geometry map from ONLINE_RUN_DICTIONARY.
- For the selected geometry_id, intersect muon tracks with planes and save a per-geometry dataset.

Inputs:
- config:
- config_step_2_physics.yaml
- config_step_2_runtime.yaml
- muon sample from Step 1 (via input_dir/input_sim_run/input_basename or input_muon_sample)
- station configs: /home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY
  - chunk_rows: when set and output_format=csv, process in chunks to reduce memory
  - chunked PKL inputs: detects *.chunks.json from Step 1 and streams chunks to CSV
  - plot_sample_rows: plot only a sample from the last processed chunk
  - geometry_id: set to "random" to choose a geometry_id from the registry
  - partial chunks are dropped unless the dataset fits in a single chunk

Outputs:
- ../../INTERSTEPS/STEP_2_TO_3/sim_run_registry.json
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geometry_registry.csv
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geometry_registry.json
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geometry_map_all.csv
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geometry_map_all.json
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.pkl (or .csv)
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.pkl.meta.json
- ../../INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geom_<G>_plots.pdf

Run:
- python3 step_2_generated_to_crossing.py --config config_step_2_physics.yaml
- python3 step_2_generated_to_crossing.py --config config_step_2_physics.yaml --plot-only
