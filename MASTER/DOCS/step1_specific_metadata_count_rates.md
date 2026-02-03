# STEP_1 Specific Metadata: Count Normalization (Hz)

## Goal
Raw `*_count` values in `task_{task}_metadata_specific.csv` scale with file duration and are therefore hard to compare across files/stations/tasks.  
To make comparisons meaningful, we also store **per-second normalized versions** (Hz) plus the **denominator seconds** used for normalization so counts can be reconstructed if needed.

## Where This Is Saved
For each station and STEP_1 task:

- `STATIONS/MINGO0{station}/STAGE_1/EVENT_DATA/STEP_1/TASK_{task}/METADATA/task_{task}_metadata_specific.csv`

## Implementation (Tasks 1–5)
Implemented in all five STEP_1 task scripts:

- `MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py`
- `MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_2/script_2_clean_to_cal.py`
- `MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_3/script_3_cal_to_list.py`
- `MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_4/script_4_list_to_fit.py`
- `MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_5/script_5_fit_to_corr.py`

Each task already computes time coverage with `build_events_per_second_metadata(working_df)`.  
Immediately after that call, we now also run:

- `add_normalized_count_metadata(global_variables, global_variables["events_per_second_total_seconds"])`

## New Columns Added
### Denominator (always present)
- `count_rate_denominator_seconds`
  - Integer seconds used for normalization.
  - Comes from `events_per_second_total_seconds` for the final `working_df` of that task.

### Normalized versions
For every metadata key that looks like a count:

- For columns ending in `*_count`:
  - Default: add `*_rate_hz = _count / count_rate_denominator_seconds`
  - Exception for `events_per_second_{k}_count`:
    - Add `events_per_second_{k}_fraction = events_per_second_{k}_count / count_rate_denominator_seconds`
    - (Those `*_count` are "how many seconds had k events", so the normalized quantity is a fraction of time, not Hz.)

- For columns ending in `*_entries`, `*_entries_final`, `*_entries_initial`:
  - Add `<original_name>_rate_hz = entries / count_rate_denominator_seconds`

All normalized columns are written into the same `task_{task}_metadata_specific.csv` row.

## How To Reconstruct Counts
Given:

- `X_rate_hz`
- `count_rate_denominator_seconds`

You can recover the original count approximately as:

- `X_count ≈ X_rate_hz * count_rate_denominator_seconds`

For the events-per-second histogram fractions:

- `events_per_second_{k}_count ≈ events_per_second_{k}_fraction * count_rate_denominator_seconds`

## Important Notes / Caveats
- If `count_rate_denominator_seconds` is `0`, normalization is skipped for that row (rates are not generated).
  - This happens when `build_events_per_second_metadata()` cannot find a usable time column in `working_df`.
  - The current logic expects a time column named `datetime` or `Time` (or a DatetimeIndex).
- The denominator is computed **after the task’s filtering**, so it can differ across tasks if early/late events are removed.
  - This is often what you want (rates reflect what remains after filtering), but it means comparing raw counts across tasks is misleading.

