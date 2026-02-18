# MASTER Cron Expected Behaviour

Last updated: 2026-02-18  
Source of truth: `CONFIG/add_to_crontab.info`

## Scope
This document covers cron-triggered processes whose executed scripts live under:
- `/home/mingo/DATAFLOW_v3/MASTER/STAGES`
- `/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY`

It does not cover non-MASTER orchestration wrappers except where they gate MASTER jobs (for example, `resource_gate.sh`).

## Global Expectations
1. `MASTER_STAGE_ROOT` must resolve to `/home/mingo/DATAFLOW_v3/MASTER/STAGES`.
2. No cron execution should attempt literal unresolved paths like:
   - `$DATAFLOW_ROOT/MASTER/...`
   - `$HOME/DATAFLOW_v3/MASTER/...`
3. Jobs with `flock` must run as singletons per lock file.
4. Jobs wrapped with `resource_gate.sh` may defer execution under high resource pressure; this is expected and not a failure by itself.
5. Commented entries are intentionally disabled and should not run.

## Expected Process Inventory

| Cron Schedule | Process | Lock / Gate | Expected Behaviour | Primary Log |
|---|---|---|---|---|
| `*/10 * * * *` | `STAGE_0/SIMULATION/ingest_simulated_station_data.py` | `resource_gate(sim_ingest)` + `flock sim_ingest_station_data.lock` | Move simulated `.dat` files into `STATIONS/MINGO00/STAGE_0_to_1`; update imported basename registry | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/SIMULATION/ingest_simulated_station_data.log` |
| `*/15 * * * *` (stations `1..4`) | `STAGE_0/NEW_FILES/bring_data_and_config_files.sh <station>` | none | Pull station raw/config inputs; update station online-run dictionary; complete even when remote station unreachable | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/NEW_FILES/bring_data_and_config_files_<station>.log` |
| `*/5 * * * *` (stations `1..4`) | `STAGE_0/REPROCESSING/STEP_0/prepare_reprocessing_metadata.sh <station> --refresh-metadata` | none | Refresh/filter reprocessing metadata; no-op when metadata unchanged | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/REPROCESSING/STEP_0/log_prepare_reprocessing_metadata_<station>.log` |
| `* * * * *` (stations `1..4`) | `STAGE_0/REPROCESSING/STEP_1/bring_reprocessing_files.sh <station> -r` | `resource_gate(step0_reproc_bring_s*)` + per-station `flock` | Select random basename and fetch reprocessing `.hld(.tar.gz)` inputs | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/REPROCESSING/STEP_1/log_bring_reprocessing_files_<station>.log` |
| `* * * * *` (stations `1..4`) | `STAGE_0/REPROCESSING/STEP_2/unpack_reprocessing_files.sh <station> -p 3` | `resource_gate(step0_reproc_unpack_s*)` | Unpack HLD payloads, emit `.dat`, move to `STAGE_0_to_1`, archive processed inputs | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/REPROCESSING/STEP_2/log_unpack_reprocessing_files_<station>.log` |
| station `1`: `* 1 * * *`; station `2`: `* 2 * * *`; station `3`: `* 3 * * *`; station `4`: `* 4 * * *` | `STAGE_1/COPERNICUS/STEP_1/copernicus_bring.py <station>` | `resource_gate(copernicus_s*)` + per-station `flock` | Refresh Copernicus weather data; may no-op if already up to date | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/COPERNICUS/copernicus_bring_<station>.log` |
| `* * * * *` (stations `0..4`) | `STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh -s <station>` | per-station `flock` | Execute enabled task chain (typically tasks `1,2,3,4`) from raw to corrected outputs | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected_<station>.log` |
| `* * * * *` | `STAGE_1/EVENT_DATA/STEP_2/guide_corrected_to_accumulated.sh --no-loop` | `resource_gate(step1_step2_accumulate)` + `flock guide_corrected_to_accumulated_all.lock` | Single pass over stations, producing/updating accumulated products | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_2/guide_corrected_to_accumulated_all.log` |
| `* * * * *` | `STAGE_1/EVENT_DATA/STEP_3/guide_accumulated_to_joined.sh --no-loop` | `resource_gate(step1_step3_joined)` + `flock guide_accumulated_to_joined_all.lock` | Single pass join/distribution from accumulated data | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_3/guide_accumulated_to_joined_all.log` |
| `10/20/30/40 * * * *` (stations `1..4`) | `STAGE_1/LAB_LOGS/STEP_1/lab_logs_bring_and_clean.sh <station>` | none | Pull and clean lab logs per station | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/LAB_LOGS/STEP_1/lab_logs_bring_and_clean_<station>.log` |
| `20/30/40/50 * * * *` (stations `1..4`) | `STAGE_1/LAB_LOGS/STEP_2/lab_logs_merge.py <station>` | none | Merge cleaned lab logs per station; may no-op when no cleaned files exist | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/LAB_LOGS/STEP_2/lab_logs_merge_<station>.log` |
| `*/10 * * * *` | `MASTER/ANCILLARY/PLOTTERS/DEFINITIVE_EXECUTION/definitive_execution_plotter.py` | none | Refresh definitive execution status plot artifacts | `OPERATIONS_RUNTIME/CRON_LOGS/PLOTTERS/DEFINITIVE_EXECUTION/definitive_execution_plotter.log` |

## MASTER STEP_1 Queue Behaviour (Critical)
Expected for `guide_raw_to_corrected.sh`:
1. Each station invocation must use its own queue file in:
   - `OPERATIONS_RUNTIME/TRAFFIC_LIGHT/stage1_step1_station_task_queue_s<stations>_t<tasks>.txt`
2. Station 1 with whitelist `[1,2,3,4]` must eventually execute task 3 (not starve):
   - Log evidence: `Running station 1 task3 (cal_to_list)` and `Completed station 1 task3 (cal_to_list)`.

## Intentionally Disabled / Missing in Cron
1. `STAGE_3` NMDB retrieval is commented out.
2. No active `STAGE_2` cron job is currently defined in `add_to_crontab.info`.

## Machine-Check Assertions (for future automation)
1. `crontab -l` contains `MASTER_STAGE_ROOT=/home/mingo/DATAFLOW_v3/MASTER/STAGES`.
2. For each active MASTER job log, mtime is within `2x` expected cadence.
3. No recent log line matches:
   - `No such file or directory` with `$DATAFLOW_ROOT/MASTER` or `$HOME/DATAFLOW_v3/MASTER`.
4. For STEP_1 station 1 log, within a rolling window (for example 30 minutes), at least one `task3` run appears when `event_data_step1_run_matrix` enables `1-3`.
5. For all `flock`-guarded MASTER jobs, there is at most one active worker per lock.
