---
title: Cron and Scheduling
description: Expected cron behavior for MASTER and MINGO_DIGITAL_TWIN pipelines.
last_updated: 2026-02-24
status: active
source_of_truth: CONFIG/add_to_crontab.info
---

# Cron and Scheduling

## Table of contents
- [Scope](#scope)
- [Global expectations](#global-expectations)
- [MASTER process inventory](#master-process-inventory)
- [MINGO_DIGITAL_TWIN process inventory](#mingo_digital_twin-process-inventory)
- [Health checks](#health-checks)
- [Open checks and stale assumptions](#open-checks-and-stale-assumptions)

## Scope
This document replaces separate cron behavior notes and is now the canonical scheduling reference for:
- `MASTER/STAGES`
- `MASTER/ANCILLARY`
- `MINGO_DIGITAL_TWIN`

## Global expectations
1. Cron entries must resolve to real absolute paths at runtime.
2. Jobs guarded with `flock -n` may skip when lock contention occurs; this is expected.
3. `resource_gate.sh` skips are expected under memory/swap/CPU pressure and should be logged, not treated as code failures.
4. Long-running jobs must never rely on log-file deletion during cleanup; use truncation/rotation that preserves active file descriptors.
5. When cron behavior and this file disagree, `CONFIG/add_to_crontab.info` is authoritative.

## MASTER process inventory

| Schedule | Process | Lock / gate | Expected behavior | Primary log |
|---|---|---|---|---|
| `*/10 * * * *` | `STAGE_0/SIMULATION/ingest_simulated_station_data.py` | `resource_gate(sim_ingest)` + `flock sim_ingest_station_data.lock` | Ingest simulated `.dat` into station flow and track imported basenames | `OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/SIMULATION/ingest_simulated_station_data.log` |
| `*/15 * * * *` (stations `1..4`) | `STAGE_0/NEW_FILES/bring_data_and_config_files.sh <station>` | None | Pull new station raw/config files; no-op if unavailable | `.../STAGE_0/NEW_FILES/bring_data_and_config_files_<station>.log` |
| `*/5 * * * *` (stations `1..4`) | `STAGE_0/REPROCESSING/STEP_0/prepare_reprocessing_metadata.sh <station> --refresh-metadata` | None | Refresh reprocessing metadata | `.../REPROCESSING/STEP_0/log_prepare_reprocessing_metadata_<station>.log` |
| `* * * * *` (stations `1..4`) | `STAGE_0/REPROCESSING/STEP_1/bring_reprocessing_files.sh <station> -r` | `resource_gate(step0_reproc_bring_s*)` + per-station `flock` | Pull reprocessing HLD inputs | `.../REPROCESSING/STEP_1/log_bring_reprocessing_files_<station>.log` |
| `* * * * *` (stations `1..4`) | `STAGE_0/REPROCESSING/STEP_2/unpack_reprocessing_files.sh <station> -p 3` | `resource_gate(step0_reproc_unpack_s*)` | Unpack HLD, emit `.dat`, hand off to stage queue | `.../REPROCESSING/STEP_2/log_unpack_reprocessing_files_<station>.log` |
| station 1: `* 1 * * *`, station 2: `* 2 * * *`, station 3: `* 3 * * *`, station 4: `* 4 * * *` | `STAGE_1/COPERNICUS/STEP_1/copernicus_bring.py <station>` | `resource_gate(copernicus_s*)` + per-station `flock` | Refresh weather context, may no-op if current | `.../STAGE_1/COPERNICUS/copernicus_bring_<station>.log` |
| `* * * * *` (stations `0..4`) | `STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh -s <station>` | per-station `flock` | Run enabled STEP_1 task chain | `.../STEP_1/guide_raw_to_corrected_<station>.log` |
| `* * * * *` | `STAGE_1/EVENT_DATA/STEP_2/guide_corrected_to_accumulated.sh --no-loop` | `resource_gate(step1_step2_accumulate)` + `flock` | Accumulate outputs from corrected files | `.../STEP_2/guide_corrected_to_accumulated_all.log` |
| `* * * * *` | `STAGE_1/EVENT_DATA/STEP_3/guide_accumulated_to_joined.sh --no-loop` | `resource_gate(step1_step3_joined)` + `flock` | Join/distribute accumulated data | `.../STEP_3/guide_accumulated_to_joined_all.log` |
| `10,20,30,40 * * * *` (stations `1..4`) | `STAGE_1/LAB_LOGS/STEP_1/lab_logs_bring_and_clean.sh <station>` | None | Pull and clean lab logs | `.../LAB_LOGS/STEP_1/lab_logs_bring_and_clean_<station>.log` |
| `20,30,40,50 * * * *` (stations `1..4`) | `STAGE_1/LAB_LOGS/STEP_2/lab_logs_merge.py <station>` | None | Merge cleaned lab logs | `.../LAB_LOGS/STEP_2/lab_logs_merge_<station>.log` |
| `*/10 * * * *` | `MASTER/ANCILLARY/PLOTTERS/DEFINITIVE_EXECUTION/definitive_execution_plotter.py` | None | Refresh definitive execution plots | `OPERATIONS_RUNTIME/CRON_LOGS/PLOTTERS/DEFINITIVE_EXECUTION/definitive_execution_plotter.log` |

Critical queue expectation for STEP_1:
- `guide_raw_to_corrected.sh` should use per-station queue files under `OPERATIONS_RUNTIME/TRAFFIC_LIGHT/`.
- Station 1 should eventually execute task 3 when enabled; repeated starvation is abnormal.

## MINGO_DIGITAL_TWIN process inventory

| Schedule | Process | Lock / gate | Expected behavior | Primary log |
|---|---|---|---|---|
| `* * * * *` | `MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_main_simulation_cycle.sh` | `flock sim_main_pipeline.lock` + internal frequency/backpressure gate | Run cycle: STEP_0 -> STEP_1..10 -> STEP_FINAL -> pruning/sanitize | `OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/RUN/sim_main_pipeline_cycle.log` |
| `*/15 * * * *` | `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/sanitize_sim_runs.py --apply --min-age-seconds 900` | `flock sim_main_pipeline.lock` | Remove stale/broken `SIM_RUN_*` state | `OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/ANCILLARY/sanitize_sim_runs.log` |
| `25 6 * * *` | `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/ensure_sim_hashes.py --apply` | `flock sim_main_pipeline.lock` | Enforce hash consistency for simulated `.dat` and parquet | `OPERATIONS_RUNTIME/CRON_LOGS/ANCILLARY/ensure_sim_hashes.log` |
| `*/2 * * * *` | `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/stop_execution.sh` | internal `/tmp/dataflow_stop_execution.lock` | Emergency guard for runaway process/load conditions | `OPERATIONS_RUNTIME/CRON_LOGS/ANCILLARY/CLEANERS/stop_execution.log` |

Main-cycle stage ordering should be visible in logs as:
- `stage=step_0 status=start|ok|skipped`
- `stage=run_step_all status=start|ok|failed`
- `stage=step_final status=start|ok|failed`
- `stage=prune_mesh_done_rows status=start|ok|failed(non-fatal)`
- `cycle status=ok|skipped|failed`

## Health checks

```bash
service cron status
crontab -l

# MASTER quick checks
pgrep -af "guide_raw_to_corrected.sh -s"
stat -c '%y %n' /home/mingo/DATAFLOW_v3/OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected_{0,1,2,3,4}.log

# Digital twin quick checks
pgrep -af "run_main_simulation_cycle.sh|run_step.sh -c|step_final_daq_to_station_dat.py"
tail -n 80 /home/mingo/DATAFLOW_v3/OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/RUN/sim_main_pipeline_cycle.log
```

Expected signs of health:
- Log mtimes advance at roughly job cadence (allowing lock skips).
- No repeated `No such file or directory` for unresolved cron paths.
- No persistent stale lock without matching process.

## Open checks and stale assumptions
- `STAGE_2` cron scheduling on the MASTER side has historically been absent in some snapshots and should be verified in `CONFIG/add_to_crontab.info` before relying on this behavior.
- Simulation backpressure thresholds (`SIM_MAX_UNPROCESSED_FILES`) are environment-sensitive. Review when queue depth or hardware profile changes.
- If process-count explosions recur, re-verify `flock` coverage for every per-minute job (STEP_1, Copernicus, simulation loop).
