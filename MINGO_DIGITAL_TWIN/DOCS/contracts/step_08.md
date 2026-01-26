# STEP 08 Interface Contract (Uncalibrated -> Threshold)

## Purpose
Apply front-end electronics effects: timing jitter, time-walk conversion, and thresholding.

## Required inputs
- Input data (from STEP 07):
  - `event_id`.
  - `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`.
- Config inputs:
  - `t_fee_sigma_ns`.
  - `q_to_time_factor`, `qfront_offsets`, `qback_offsets`.
  - `charge_threshold`.

## Output schema
Outputs:
- `INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/step_8.(pkl|csv|chunks.json)`

Columns:
- `event_id`, `T_thick_s`.
- Per-plane per-strip: `T_front`, `T_back`, `Q_front`, `Q_back`.

## Behavior
- `T_front` and `T_back` receive Gaussian jitter on all non-NaN entries.
- `Q_front` and `Q_back` are converted to time-walk units and offset per channel.
- Values below `charge_threshold` are set to 0.

## Metadata
- Common fields plus `source_dataset` and `step_8_id`.

## Failure modes
- Missing strip columns are skipped without warnings.
- No explicit validation is performed on threshold sign or offset dimensions.
