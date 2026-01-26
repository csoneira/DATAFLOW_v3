# STEP 10 Interface Contract (Triggered -> Jitter)

## Purpose
Apply TDC smear and event-level jitter to triggered events.

## Required inputs
- Input data (from STEP 09):
  - `event_id`.
  - `T_front_i_sj`, `T_back_i_sj`.
  - `Q_front_i_sj`, `Q_back_i_sj` (to determine active events).
- Config inputs:
  - `tdc_sigma_ns`, `jitter_width_ns`.

## Output schema
Outputs:
- `INTERSTEPS/STEP_10_TO_FINAL/SIM_RUN_<N>/step_10.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- Per-plane per-strip: `T_front`, `T_back`, `Q_front`, `Q_back`.
- `daq_jitter_ns` (ns): jitter applied to active events (0 for inactive events).

## Behavior
- An event is active if any strip has `Q_front > 0` or `Q_back > 0`.
- Active events receive Gaussian TDC smear and uniform jitter on all time channels.

## Metadata
- Common fields plus `source_dataset` and `step_10_id`.

## Failure modes
- Missing strip columns are skipped without warnings.
- No explicit validation is performed on jitter parameter ranges.
