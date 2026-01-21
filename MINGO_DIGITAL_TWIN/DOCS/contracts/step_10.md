# STEP 10 Interface Contract (Triggered -> Jitter)

## Purpose
Apply TDC smear and DAQ clock jitter to front/back times for triggered events.

## Required inputs
- Input data (from STEP 09):
  - `event_id` (int)
  - `T_front_i_sj`, `T_back_i_sj` (ns)
  - `Q_front_i_sj`, `Q_back_i_sj` (used to determine active channels)
- Config inputs:
  - `tdc_sigma_ns` (Gaussian TDC smear)
  - `jitter_width_ns` (uniform clock jitter)
- Required metadata: none (metadata is produced by this step).

## Schema (guaranteed outputs)
Retained columns:
- `event_id` (int)
- `T_thick_s` (s) if present upstream
- `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`

Added column:
- `daq_jitter_ns` (ns): per-event jitter applied to active channels (0 if no active channels).

Time reference: unchanged from STEP 09, with additional TDC smear and event-level jitter.

## Invariants & checks
- `daq_jitter_ns` is uniform in [-jitter_width_ns/2, +jitter_width_ns/2] for active events.
- Only channels with `Q_front_i_sj` or `Q_back_i_sj` > 0 are jittered/smeared.

## Failure modes & validation behavior
- Missing input columns are skipped without warnings.
- No explicit validation is performed on jitter/smear parameter ranges.
