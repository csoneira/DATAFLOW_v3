Step 1 (Muon Sample)

Purpose:
- Generate primary muon parameters (position/direction/time) for the simulation.

Inputs:
- config:
  - config_step_1_physics.yaml
  - config_step_1_runtime.yaml
- data: No upstream input; uses physics parameters to generate tracks.

Outputs:
- INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.(pkl|csv)
- INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/PLOTS/muon_sample_<N>_plots.pdf

Run:
- python3 step_1_blank_to_generated.py --config config_step_1_physics.yaml
- python3 step_1_blank_to_generated.py --config config_step_1_physics.yaml --runtime-config config_step_1_runtime.yaml
- python3 step_1_blank_to_generated.py --config config_step_1_physics.yaml --plot-only
