---
title: STEP_1 Metadata Count-Rate Normalization
description: Definition of count-normalization fields added to STEP_1 task metadata.
last_updated: 2026-02-24
status: active
---

# STEP_1 Metadata Count-Rate Normalization (Hz)

## Table of contents
- [Goal](#goal)
- [Output location](#output-location)
- [Implementation coverage](#implementation-coverage)
- [Columns added](#columns-added)
- [Reconstruction formulas](#reconstruction-formulas)
- [Caveats](#caveats)

## Goal
Raw `*_count` fields in `task_{task}_metadata_specific.csv` scale with file duration, making cross-file comparisons misleading.

To make metadata comparable, STEP_1 tasks also write:
- normalized rate/fraction fields,
- a denominator column (`count_rate_denominator_seconds`) so original counts can be reconstructed.

## Output location
Per station and task:
- `STATIONS/MINGO0{station}/STAGE_1/EVENT_DATA/STEP_1/TASK_{task}/METADATA/task_{task}_metadata_specific.csv`

## Implementation coverage
Applied in all STEP_1 task scripts:
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/script_2_clean_to_cal.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_3/script_3_cal_to_list.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_4/script_4_list_to_fit.py`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_5/script_5_fit_to_post.py`

The denominator is produced from:
- `build_events_per_second_metadata(working_df)`

Normalization is applied via:
- `add_normalized_count_metadata(global_variables, global_variables["events_per_second_total_seconds"])`

## Columns added

### Denominator
- `count_rate_denominator_seconds`
- Integer seconds used as normalization base.

### Normalized fields
- For `*_count`: add `*_rate_hz = *_count / count_rate_denominator_seconds`.
- Exception: `events_per_second_{k}_count` becomes `events_per_second_{k}_fraction` (fraction of time, not Hz).
- For `*_entries`, `*_entries_initial`, `*_entries_final`: add `<field>_rate_hz`.

## Reconstruction formulas
- `X_count ~= X_rate_hz * count_rate_denominator_seconds`
- `events_per_second_{k}_count ~= events_per_second_{k}_fraction * count_rate_denominator_seconds`

## Caveats
- If denominator is `0`, normalization is skipped.
- Denominator depends on post-filter time coverage inside each task, so values can differ between tasks for the same basename.
- Time-column assumptions (`datetime`, `Time`, or DatetimeIndex) should remain consistent; schema drift here silently changes normalization behavior.
