# STEP 05 Interface Contract (Hit -> Signal)

## Purpose
Derive per-strip time-difference and charge-difference observables from measured strip hits.

## Required inputs
- Input data (from STEP 04):
  - `event_id` (int)
  - `X_mea_i_sj` (mm), `Y_mea_i_sj` (arb charge), `T_sum_meas_i_sj` (ns) for planes i = 1..4, strips j = 1..4.
- Config inputs:
  - `c_mm_per_ns` (propagation speed for T_diff calculation).
  - `qdiff_frac` (fractional charge noise).
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
Retained columns:
- `event_id` (int)
- `T_thick_s` (s) if present upstream
- `Y_mea_i_sj` (arb charge)
- `T_sum_meas_i_sj` (ns)

Derived columns:
- `T_diff_i_sj` (ns): time-difference proxy from strip x-position.
- `q_diff_i_sj` (arb charge): charge-difference proxy (zero if no strip charge).

Time reference: `T_diff_i_sj` is relative to strip center (zero at X = 0) and preserves the STEP 02 time origin.

## Invariants & checks
- If `Y_mea_i_sj` == 0, then `q_diff_i_sj` == 0.
- `T_diff_i_sj` is finite where `X_mea_i_sj` is finite.

## Failure modes & validation behavior
- Missing required strip columns cause that strip to be skipped (no new columns).
- No explicit warnings are emitted; NaNs propagate for missing inputs.
