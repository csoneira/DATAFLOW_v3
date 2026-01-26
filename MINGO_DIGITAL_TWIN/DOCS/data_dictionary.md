# Data Dictionary

This document lists the column-level data products produced by each step. For per-plane
and per-strip fields, use the naming conventions below.

## Naming conventions
- Plane index: i = 1..4
- Strip index: j = 1..4
- Per-plane fields: `<field>_i`
- Per-plane per-strip fields: `<field>_i_sj`
- Plane tags: `tt_*` are strings concatenating active plane indices in ascending order.

## Common columns
- `event_id` (int): 0-based identifier, preserved across steps.
- `T_thick_s` (s): integer second tag generated in STEP 1; used for timestamp offsets.

## STEP 1 output (muon generation)
- `X_gen` (mm): generated x coordinate at the generation plane.
- `Y_gen` (mm): generated y coordinate at the generation plane.
- `Z_gen` (mm): generated z coordinate (constant per run).
- `Theta_gen` (rad): polar angle from +Z.
- `Phi_gen` (rad): azimuth angle in X-Y plane.

Note: `T0_ns` exists internally but is not persisted in output.

## STEP 2 output (plane crossings)
Per-plane (i = 1..4):
- `X_gen_i` (mm): projected crossing x for plane i (NaN if out of bounds).
- `Y_gen_i` (mm): projected crossing y for plane i (NaN if out of bounds).
- `Z_gen_i` (mm): plane z position for plane i (NaN if out of bounds).
- `T_sum_i_ns` (ns): flight time to plane i (normalized so earliest valid plane is 0).

Per-event:
- `tt_crossing`: planes with in-bounds crossings.

## STEP 3 output (avalanche)
Per-plane (i = 1..4):
- `avalanche_ion_i` (count): primary ionizations (Poisson).
- `avalanche_exists_i` (bool): True when `avalanche_ion_i > 0`.
- `avalanche_x_i` (mm): avalanche x (NaN if no avalanche).
- `avalanche_y_i` (mm): avalanche y (NaN if no avalanche).
- `avalanche_size_electrons_i` (electrons): avalanche size after gain and smearing.
- `T_sum_i_ns` (ns): preserved from STEP 2.

Per-event:
- `tt_avalanche`: planes with avalanches.

## STEP 4 output (induced strip signals)
Per-plane, per-strip (i = 1..4, j = 1..4):
- `Y_mea_i_sj` (arb): induced charge on strip j.
- `X_mea_i_sj` (mm): measured x position (NaN if no hit on strip j).
- `T_sum_meas_i_sj` (ns): measured sum time (NaN if no hit on strip j).

Per-event:
- `tt_hit`: planes with any strip hit.

## STEP 5 output (time/charge differences)
Per-plane, per-strip:
- `Y_mea_i_sj` (arb): preserved from STEP 4.
- `T_sum_meas_i_sj` (ns): preserved from STEP 4.
- `T_diff_i_sj` (ns): time difference derived from X position.
- `q_diff_i_sj` (arb): charge imbalance noise term.

## STEP 6 output (front/back endpoints)
Per-plane, per-strip:
- `T_front_i_sj` (ns)
- `T_back_i_sj` (ns)
- `Q_front_i_sj` (arb)
- `Q_back_i_sj` (arb)

## STEP 7 output (cable offsets)
Same schema as STEP 6; `T_front_i_sj` and `T_back_i_sj` include per-channel offsets.

## STEP 8 output (FEE threshold)
Same schema as STEP 6; Q values are converted to time-walk units (ns) and thresholded.

## STEP 9 output (triggered events)
Same schema as STEP 6 plus:
- `tt_trigger`: planes active after thresholding.

## STEP 10 output (DAQ jitter)
Same schema as STEP 6 plus:
- `daq_jitter_ns` (ns): event-level jitter applied to active channels.

## STEP FINAL output (.dat format)
Each ASCII line contains:
- Timestamp header: `YYYY MM DD HH MM SS 1`.
- 64 channel values ordered by plane [4,3,2,1], field [T_front, T_back, Q_front, Q_back],
  strip [1..4].

See `DOCS/station_dat_format.md` for formatting rules and file naming.
