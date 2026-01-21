# Coordinate and Timing Conventions

This document is the authoritative reference for coordinate axes, plane/strip ordering, and time references used throughout MINGO_DIGITAL_TWIN.

## Coordinate system
- Units: positions are in millimeters (mm), times are in nanoseconds (ns) unless stated.
- Axes:
  - +Z is the nominal detector stack axis (plane separation axis).
  - +X and +Y span the detector plane.
- Angles:
  - `Theta_gen` is the polar angle from +Z (0 is vertical).
  - `Phi_gen` is the azimuth in the X-Y plane, measured from +X toward +Y.

## Plane ordering and station conventions
- Planes are indexed 1..4 and correspond to station geometry fields `P1..P4`.
- Plane z positions come from the station configuration table and are passed to STEP 2.
- `normalize_to_first_plane` (STEP 2 runtime config) shifts all plane z positions so plane 1 is at z = 0.

## Strip orientation and indexing
- Each plane has 4 strips indexed 1..4.
- Strip segmentation is along Y (strips are parallel to the X axis).
- Strip edges and centers are defined by `STEP_SHARED.get_strip_geometry`, which uses `Y_WIDTHS` with widths in mm.
- The along-strip coordinate used for `X_mea_i_sj` is the global X coordinate (mm) at the induction point.

## Front/back definition
- `T_diff_i_sj` is computed from `X_mea_i_sj` in STEP 5: `T_diff = X_mea * (3 / (2 * c_mm_per_ns))`.
- `T_front_i_sj` and `T_back_i_sj` are defined in STEP 6:
  - `T_front = T_sum_meas - T_diff`
  - `T_back = T_sum_meas + T_diff`
- As a result, positive `X_mea` implies `T_front < T_back`.

## Timing reference and offsets
- STEP 2 defines per-plane incidence times `T_sum_i_ns`, then subtracts the minimum valid plane time so that the earliest crossing in each event is at 0.
- STEP 4 builds `T_sum_meas_i_sj` by adding Gaussian noise (`time_sigma_ns`) to `T_sum_i_ns`.
- STEP 7 applies fixed connector/cable offsets to `T_front_i_sj` and `T_back_i_sj`.
- STEP 8 applies front-end electronics jitter (`t_fee_sigma_ns`) and converts Q values to time-walk units via `q_to_time_factor` and per-channel offsets.
- STEP 10 applies TDC smear (`tdc_sigma_ns`) and event-level jitter (`jitter_width_ns`) to active channels.

## Time-walk / charge units
- Up to STEP 6, `Y_mea_i_sj`, `Q_front_i_sj`, and `Q_back_i_sj` represent charge-like quantities in arbitrary units tied to avalanche size.
- STEP 8 converts `Q_front_i_sj` and `Q_back_i_sj` into time-walk units (ns) via `q_to_time_factor`; after STEP 8 these columns are no longer raw charge.

## Boundary reminders (physics vs electronics)
- Detector response begins at STEP 3 (avalanche microphysics).
- Readout coupling begins at STEP 4 (induction on strips).
- Electronics effects begin at STEP 8 (FEE threshold + time-walk).
- Trigger/DAQ event definition begins at STEP 9.
