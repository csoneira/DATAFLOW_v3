Step 7 (Front/Back -> Uncalibrated)

Purpose:
- Apply cable/connector propagation delays (uncalibration/decalibration step).

Inputs:
- config:
  - config_step_7_physics.yaml
  - config_step_7_runtime.yaml
- data: INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.(pkl|csv)
- INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/PLOTS/geom_<G>_calibrated_plots.pdf

Run:
- python3 step_7_timing_to_uncalibrated.py --config config_step_7_physics.yaml
- python3 step_7_timing_to_uncalibrated.py --config config_step_7_physics.yaml --runtime-config config_step_7_runtime.yaml
- python3 step_7_timing_to_uncalibrated.py --config config_step_7_physics.yaml --plot-only

Notes:
- input_sim_run supports explicit SIM_RUN_<N>, latest, or random (runtime config).
- The step skips if the matching SIM_RUN exists unless --force is provided.
