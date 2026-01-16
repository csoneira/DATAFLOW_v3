Step 10 (DAQ Timing)

Purpose:
- Apply TDC timing smear and DAQ jitter to front/back times.

Inputs:
- config:
  - config_step_10_physics.yaml
  - config_step_10_runtime.yaml
- data: INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.(pkl|csv)

Outputs:
- INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/geom_<G>_daq.(pkl|csv)
- INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/PLOTS/geom_<G>_daq_plots.pdf

Run:
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml --runtime-config config_step_10_runtime.yaml
- python3 step_10_triggered_to_jitter.py --config config_step_10_physics.yaml --plot-only
