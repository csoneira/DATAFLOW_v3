MINGO_DIGITAL_TWIN Pipeline

Overview
- STEP_1: Blank -> Generated
  - Script: STEP_1/step_1_blank_to_generated.py
  - Output: STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.(pkl|csv)
  - Key fields: X_gen, Y_gen, Z_gen, Theta_gen, Phi_gen, T0_ns

- STEP_2: Generated -> Crossing
  - Script: STEP_2/step_2_generated_to_crossing.py
  - Output: STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.(pkl|csv)
  - Key fields: X_gen_i, Y_gen_i, Z_gen_i, T_sum_i_ns (relative), tt_crossing

- STEP_3: Crossing -> Avalanche
  - Script: STEP_3/step_3_crossing_to_hit.py
  - Output: STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.(pkl|csv)
  - Key fields: avalanche_ion_i, avalanche_exists_i, avalanche_x_i, avalanche_y_i, avalanche_qsum_i, tt_avalanche

- STEP_4: Avalanche -> Hit
  - Script: STEP_4/step_4_hit_to_measured.py
  - Output: STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.(pkl|csv)
  - Key fields: Y_mea_i_sj (qsum), X_mea_i_sj, T_sum_meas_i_sj, tt_hit

- STEP_5: Hit -> Signal (T_diff/q_diff)
  - Script: STEP_5/step_5_measured_to_triggered.py
  - Output: STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.(pkl|csv)
  - Key fields: T_diff_i_sj, q_diff_i_sj

- STEP_6: Signal -> Front/Back
  - Script: STEP_6/step_6_triggered_to_timing.py
  - Output: STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback.(pkl|csv)
  - Key fields: T_front_i_sj, T_back_i_sj, Q_front_i_sj, Q_back_i_sj

- STEP_7: Front/Back -> Calibrated
  - Script: STEP_7/step_7_timing_to_calibrated.py
  - Output: STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.(pkl|csv)
  - Key fields: calibrated T_front/T_back/Q_front/Q_back

- STEP_8: Threshold
  - Script: STEP_8/step_8_calibrated_to_threshold.py
  - Output: STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.(pkl|csv)
  - Key fields: thresholded Q_front/Q_back

- STEP_9: Trigger
  - Script: STEP_9/step_9_threshold_to_trigger.py
  - Output: STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.(pkl|csv)
  - Key fields: tt_trigger

- STEP_10: DAQ Jitter
  - Script: STEP_10/step_10_triggered_to_jitter.py
  - Output: STEP_10_TO_11/SIM_RUN_<N>/geom_<G>_daq.(pkl|csv)
  - Key fields: daq_jitter_ns, jittered T_front/T_back

- STEP_11: DAQ -> Detector Text
  - Script: STEP_11/step_11_daq_to_detector_format.py
  - Output: STEP_11_TO_12/SIM_RUN_<N>/mi00YYDDDHHMMSS.dat
  - Key fields: detector text format (date/time, event_type, per-strip T/Q)

- STEP_12: Detector Text -> Station Dat
  - Script: STEP_12/step_12_detector_to_station_dat.py
  - Output: STEP_12_TO_13/SIM_RUN_<N>/mi0XYYDDDHHMMSS.dat
  - Key fields: station/conf selection, Poisson timestamps, real-data filename format

Notes
- Geometry IDs are global across stations: identical (P1,P2,P3,P4) share geometry_id.
- Steps 1-4 write outputs under SIM_RUN_<N> directories; mappings live in sim_run_registry.json per STEP_X_TO_Y.
- Each step writes metadata to a .meta.json alongside outputs.
- Plots are emitted as rasterized PDFs in the step output folders when enabled.
