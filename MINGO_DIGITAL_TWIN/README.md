MINGO_DIGITAL_TWIN Pipeline

Overview
- The pipeline simulates detector data from idealized muon generation through DAQ formatting and station-style .dat output.
- Each step writes to INTERSTEPS/STEP_X_TO_Y/SIM_RUN_<N> and records metadata in sim_run_registry.json.
- Configuration is split into two files per step:
  - config_step_X_physics.yaml: physical parameters that define the simulation identity (used for sim-run hashing)
  - config_step_X_runtime.yaml: I/O, chunking, plotting, and other run controls (ignored for sim-run identity)

Directory layout
- MASTER_STEPS/STEP_<X>: step scripts and configs
- MASTER_STEPS/STEP_SHARED: shared utilities
- INTERSTEPS/STEP_<X>_TO_<Y>: outputs and sim_run registries
- SIMULATED_DATA: final station-style .dat outputs + step_13_output_registry.json

Pipeline summary (per step)
- STEP_1: Blank -> Generated
  - Script: MASTER_STEPS/STEP_1/step_1_blank_to_generated.py
  - Purpose: generate primary muon parameters (position/direction/time)
  - Output: INTERSTEPS/STEP_1_TO_2/SIM_RUN_<N>/muon_sample_<N>.(pkl|csv)
  - Key fields: X_gen, Y_gen, Z_gen, Theta_gen, Phi_gen, T0_ns

- STEP_2: Generated -> Crossing
  - Script: MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py
  - Purpose: propagate muons through station geometry, compute plane crossings and times
  - Output: INTERSTEPS/STEP_2_TO_3/SIM_RUN_<N>/geom_<G>.(pkl|csv)
  - Key fields: X_gen_i, Y_gen_i, Z_gen_i, T_sum_i_ns, tt_crossing

- STEP_3: Crossing -> Avalanche
  - Script: MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py
  - Purpose: apply efficiencies and avalanche model at each plane
  - Output: INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/geom_<G>_avalanche.(pkl|csv)
  - Key fields: avalanche_size_electrons_i, avalanche_x_i, avalanche_y_i, tt_avalanche

- STEP_4: Avalanche -> Hit
  - Script: MASTER_STEPS/STEP_4/step_4_hit_to_measured.py
  - Purpose: induce strip charges and measured positions/times per plane
  - Output: INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/geom_<G>_hit.(pkl|csv)
  - Key fields: Y_mea_i_sj (qsum), X_mea_i_sj, T_sum_meas_i_sj, tt_hit

- STEP_5: Hit -> Signal (T_diff/q_diff)
  - Script: MASTER_STEPS/STEP_5/step_5_measured_to_triggered.py
  - Purpose: compute per-strip time/charge differences from measured signals
  - Output: INTERSTEPS/STEP_5_TO_6/SIM_RUN_<N>/geom_<G>_signal.(pkl|csv)
  - Key fields: T_diff_i_sj, q_diff_i_sj

- STEP_6: Signal -> Front/Back
  - Script: MASTER_STEPS/STEP_6/step_6_triggered_to_timing.py
  - Purpose: build front/back timing and charge vectors per strip
  - Output: INTERSTEPS/STEP_6_TO_7/SIM_RUN_<N>/geom_<G>_frontback.(pkl|csv)
  - Key fields: T_front_i_sj, T_back_i_sj, Q_front_i_sj, Q_back_i_sj

- STEP_7: Front/Back -> Calibrated
  - Script: MASTER_STEPS/STEP_7/step_7_timing_to_calibrated.py
  - Purpose: apply per-strip timing offsets
  - Output: INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/geom_<G>_calibrated.(pkl|csv)
  - Key fields: calibrated T_front/T_back/Q_front/Q_back

- STEP_8: Threshold (FEE model)
  - Script: MASTER_STEPS/STEP_8/step_8_calibrated_to_threshold.py
  - Purpose: apply front-end thresholds and charge-to-time conversion
  - Output: INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/geom_<G>_threshold.(pkl|csv)
  - Key fields: thresholded Q_front/Q_back and converted time values

- STEP_9: Trigger
  - Script: MASTER_STEPS/STEP_9/step_9_threshold_to_trigger.py
  - Purpose: evaluate trigger combinations
  - Output: INTERSTEPS/STEP_9_TO_10/SIM_RUN_<N>/geom_<G>_triggered.(pkl|csv)
  - Key fields: tt_trigger

- STEP_10: DAQ timing model
  - Script: MASTER_STEPS/STEP_10/step_10_triggered_to_jitter.py
  - Purpose: apply TDC smear and DAQ jitter to T_front/T_back
  - Output: INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/geom_<G>_daq.(pkl|csv)
  - Key fields: daq_jitter_ns, jittered T_front/T_back

- STEP_FINAL: DAQ -> Station Dat (final formatting)
  - Script: MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py
  - Purpose: format detector text rows, assign station/conf time ranges, output station .dat
  - Output: SIMULATED_DATA/mi0XYYDDDHHMMSS.dat
  - Key fields: station/conf selection, date-based filenames, per-strip T/Q text format

Notes
- Geometry IDs are global across stations: identical (P1,P2,P3,P4) share geometry_id.
- Each step writes metadata to a .meta.json alongside outputs.
- Plots (if enabled) are saved as a single rasterized PDF per output with a *_plots.pdf suffix.
