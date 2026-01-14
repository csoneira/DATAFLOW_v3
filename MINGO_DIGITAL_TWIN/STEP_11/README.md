Step 11 (DAQ to Detector Format)

Purpose:
- Convert DAQ outputs into the detector text format used by real data files the date of the event (thick time) is set to a placeholder.

Inputs:
- config: config_step_11.yaml
- Step 10 outputs in ../STEP_10_TO_11/SIM_RUN_<N> (via input_sim_run)

Outputs:
- ../STEP_11_TO_12/sim_run_registry.json
- ../STEP_11_TO_12/SIM_RUN_<N>/mi00YYDDDHHMMSS.dat
- ../STEP_11_TO_12/SIM_RUN_<N>/mi00YYDDDHHMMSS.dat.meta.json

Format:
- YYYY MM DD HH MM SS event_type then vectors in order:
  T4_F, T4_B, Q4_F, Q4_B, T3_F, T3_B, Q3_F, Q3_B, T2_F, T2_B, Q2_F, Q2_B, T1_F, T1_B, Q1_F, Q1_B
  (each is 4 components per strip).

Run:
- python3 step_11_daq_to_detector_format.py --config config_step_11.yaml
