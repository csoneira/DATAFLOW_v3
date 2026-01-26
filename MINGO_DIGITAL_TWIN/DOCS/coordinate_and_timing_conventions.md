# Coordinate and Timing Conventions

This document defines the coordinate axes, plane/strip ordering, and timing reference points
used throughout MINGO_DIGITAL_TWIN. It is intended to prevent sign errors and misinterpretation
of column meanings in analysis and downstream tooling.

## Units
- Positions: millimeters (mm).
- Times: nanoseconds (ns) unless explicitly stated otherwise.
- Thick-time tags: seconds (s) in `T_thick_s`.
- Charges: arbitrary units until STEP 8, after which Q values are time-walk units (ns).

## Coordinate system
- +Z: nominal detector stack axis (plane separation axis).
- +X, +Y: span the detector plane.
- `Theta_gen`: polar angle from +Z, in [0, pi/2].
- `Phi_gen`: azimuth in the X-Y plane, measured from +X toward +Y.

## Plane ordering and geometry
- Planes are indexed 1..4 (P1..P4) and correspond to the station geometry definitions.
- Step 2 accepts `z_positions` as a 4-tuple [z1, z2, z3, z4].
- If `normalize_to_first_plane` is true (Step 2 runtime), z positions are shifted so plane 1
  is at z = 0 (relative geometry preserved).

## Strip geometry and indexing
- Each plane has 4 strips indexed 1..4.
- Strip segmentation is along Y (strips are parallel to the X axis).
- Strip edges and centers are defined in `STEP_SHARED.get_strip_geometry` using:
  - Planes 1 and 3: widths [63, 63, 63, 98] mm.
  - Planes 2 and 4: widths [98, 63, 63, 63] mm.
- The along-strip coordinate used for `X_mea_i_sj` is the global X coordinate at the
  induction point with added Gaussian noise.

## Event timing reference
- STEP 2 computes per-plane flight times:
  `T_sum_i_ns = dz_i / (c_mm_per_ns * cos(theta))`.
- STEP 2 then shifts all per-plane times so the earliest valid plane crossing per event is 0.
  This normalized time is carried forward as the event reference time.

## Derived time/charge fields
- STEP 4 constructs `T_sum_meas_i_sj` by adding Gaussian noise (`time_sigma_ns`) to
  `T_sum_i_ns` on hit strips.
- STEP 5 defines the along-strip time difference:
  `T_diff = X_mea * (3 / (2 * c_mm_per_ns))`.
- STEP 6 defines front/back timing:
  - `T_front = T_sum_meas - T_diff`
  - `T_back  = T_sum_meas + T_diff`
- STEP 7 applies per-channel fixed offsets to `T_front` and `T_back`.
- STEP 8 adds front-end jitter (`t_fee_sigma_ns`) and converts Q values to time-walk units:
  `Q <- Q * q_to_time_factor + per-channel offset`, followed by thresholding.
- STEP 10 applies TDC smear (`tdc_sigma_ns`) and event-level jitter (`jitter_width_ns`) to
  active channels only.

## Thick-time tags
- `T_thick_s` is a per-event integer second tag derived from a Poisson count process
  in STEP 1. It is used by STEP FINAL to drive timestamp offsets when present.
- `T_thick_s` is not a physical transit time; it is a rate-sequencing tag.

## Trigger and plane tags
- `tt_crossing`, `tt_avalanche`, `tt_hit`, and `tt_trigger` are strings formed by
  concatenating plane indices (1..4) that are active for a given step.
- Ordering is always ascending (1 to 4). Presence is binary per plane.

## Charge units and conversion boundaries
- Up to STEP 7, `Y_mea`, `Q_front`, and `Q_back` are charge-like quantities derived from
  avalanche sizes.
- After STEP 8, `Q_front` and `Q_back` are time-walk units (ns) and are no longer raw charge.
