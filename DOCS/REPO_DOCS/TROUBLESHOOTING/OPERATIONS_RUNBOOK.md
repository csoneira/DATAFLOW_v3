---
title: Operations Runbook and Incident History
description: Consolidated troubleshooting reference for cron, pipeline staleness, and simulation integrity incidents.
last_updated: 2026-02-24
status: active
supersedes:
  - TROUBLESHOOTING_CRON_STEP1_GUIDE.md
  - cron_and_pipeline_status_check.md
  - PIPELINE_AUDIT_AND_STALENESS.md
  - STEP_2_param_mesh_pipeline_stall.md
  - simulated_file_missing_param_hash.md
  - copilot_api_problem.md
---

# Operations Runbook and Incident History

## Table of contents
- [Fast health checks](#fast-health-checks)
- [Pipeline state audit](#pipeline-state-audit)
- [Incident log](#incident-log)
- [Recovery procedures](#recovery-procedures)
- [Open risks and stale guidance flags](#open-risks-and-stale-guidance-flags)

## Fast health checks

```bash
service cron status
crontab -l

pgrep -af "guide_raw_to_corrected.sh -s"
pgrep -af "run_main_simulation_cycle.sh|run_step.sh -c|step_final_daq_to_station_dat.py"

stat -c '%y %n' /home/mingo/DATAFLOW_v3/OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected_{0,1,2,3,4}.log
stat -c '%y %n' /home/mingo/DATAFLOW_v3/OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/RUN/sim_main_pipeline_cycle.log
```

Expected:
- `cron` is active.
- One active worker per station for `guide_raw_to_corrected`.
- Simulation cycle log updates at expected cadence (accounting for gate/lock skips).

## Pipeline state audit
Use the audit tool for staleness and queue state diagnostics:

```bash
python3 OPERATIONS/OBSERVABILITY/AUDIT_PIPELINE_STATES/audit_pipeline_states.py --stations 0 --summary-only
python3 OPERATIONS/OBSERVABILITY/AUDIT_PIPELINE_STATES/audit_pipeline_states.py --stations 0 --stale-hours 24
python3 OPERATIONS/OBSERVABILITY/AUDIT_PIPELINE_STATES/audit_pipeline_states.py --stations 0 --scan-logs
```

Output folder:
- `OPERATIONS/OBSERVABILITY/AUDIT_PIPELINE_STATES/OUTPUT_FILES/<timestamp>/`

Key files:
- `summary_counts.csv`
- `anomalies.csv`
- `stale.csv`
- `report.html`

Canonical skip/reject lists that drive reprocessing behavior:
- `STATIONS/MINGO0{station}/STAGE_0/NEW_FILES/METADATA/raw_files_brought.csv`
- `STATIONS/MINGO0{station}/STAGE_0/REPROCESSING/STEP_1/METADATA/hld_files_brought.csv`
- `STATIONS/MINGO0{station}/STAGE_0/REPROCESSING/STEP_2/METADATA/dat_files_unpacked.csv`
- `OPERATIONS/DATA_MAINTENANCE/UPDATE_EXECUTION_CSVS/OUTPUT_FILES/MINGO0{station}_processed_basenames.csv`

## Incident log

### 2026-02-04: STEP_1 cron looked dead
- Symptom: `guide_raw_to_corrected_*.log` stopped updating.
- Root cause: `cron.service` was down (OOM-killed).
- Fix:
  - Restarted cron service.
  - Added `OPERATIONS/ORCHESTRATION/ENSURE_CRON_LOG_PATHS/ensure_cron_log_paths.sh` to cron.
- Verification: cron active + per-station STEP_1 processes + advancing log mtimes.

### 2026-02-04: Logs appeared stale while metadata advanced
- Symptom: status CSVs updated, visible cron logs looked frozen.
- Root cause: cleanup script deleted active cron log files; writers continued writing to unlinked inodes.
- Fix: changed `OPERATIONS/MAINTENANCE/CLEANERS/clean_dataflow.sh` to truncate logs instead of deleting paths.

### 2026-02-05: false self-lock exit loop in STEP_1 launcher
- Symptom: repeated "Acquired run lock" followed by immediate "already handled" exit.
- Root cause: PID parsing retained whitespace and misclassified parent process as foreign.
- Fix: trimmed PID and removed noisy raw `ps` log line in `guide_raw_to_corrected.sh`.

### 2026-02-05: process explosion (STEP_1)
- Symptom: thousands of `guide_raw_to_corrected.sh` processes, heavy swap use.
- Root cause: missing durable cron-level locking across minute launches.
- Fix:
  - killed runaway processes,
  - added per-station `flock` wrappers in cron entries.

### 2026-02-05: process explosion (Copernicus)
- Symptom: hundreds of `copernicus_bring.py` processes and rapid swap refill.
- Root cause: overlapping schedule without station-level lock.
- Fix: added per-station `flock` + `resource_gate` in cron entries.

### 2026-02-09: simulation continuous run blocked by stale `/tmp` lock
- Symptom: "continuous operation already running" with no actual running worker.
- Root cause: stale lock directory in `/tmp/mingo_digital_twin_run_step_continuous.lock`.
- Fix:
  - removed stale lock,
  - hardened `run_step.sh` to auto-recover stale lock states.

### 2026-02-12: transient simulation errors with auto-recovery
- Symptom:
  - STEP_2 intermittent `FileNotFoundError` on chunk files.
  - STEP_FINAL crash on empty `STEP_10_TO_FINAL` window.
- Fix:
  - STEP_2 and shared chunk readers now skip transient missing files.
  - STEP_FINAL handles temporary empty input windows without hard crash.
- Result: continuity restored, but errors should still be monitored as health debt.

### 2026-02-12 onward: simulated file hash mismatch / missing table rows
- Symptom: `Warning: Simulated file missing param_hash; using default z_positions.` and many `ensure_sim_hashes` mismatch candidates.
- Root causes:
  - hash normalization mismatch (`int` vs `float` through CSV round-trips),
  - non-atomic batch CSV writes causing missing registry rows on interruption.
- Fix:
  - normalized exact-integer floats before hashing,
  - atomic + incremental writes in STEP_FINAL,
  - repair tooling (`repair_orphan_hashes.py`) for existing data.

### 2026-02-16: STEP_2 -> STEP_3 stall on missing param-mesh upstreams
- Symptom: pending mesh rows but missing required `SIM_RUN_<step1>_<step2>` upstreams for STEP_3.
- Root cause: mismatch between param-mesh expectations and active STEP_2 generation mode.
- Fix direction:
  - explicit consistency checks,
  - manual or scheduler-assisted generation of missing upstream SIM_RUNs,
  - clearer visibility via work-cache and diagnostics.
- Follow-up: track with `ORCHESTRATOR/helpers/check_param_mesh_consistency.py`.

## Recovery procedures

### A) Stale or missing upstreams for pending param_mesh rows

```bash
PYTHONPATH=. python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/helpers/check_param_mesh_consistency.py \
  --mesh MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv \
  --intersteps MINGO_DIGITAL_TWIN/INTERSTEPS --step 3
```

If missing upstreams are reported:
1. Generate required STEP_2 outputs (manual targeted run or scheduler run).
2. Re-check consistency.
3. Confirm STEP_3 output appears in `INTERSTEPS/STEP_3_TO_4/`.

### B) Hash integrity repair

```bash
python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/repair_orphan_hashes.py
python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/repair_orphan_hashes.py --apply
python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/ensure_sim_hashes.py
```

### C) Pipeline staleness triage order
1. Check process presence and locks.
2. Check queue/list membership and reject lists.
3. Check ERROR directories and latest tracebacks.
4. Retry from the nearest failing step before full reset.

## Open risks and stale guidance flags
- Some historical notes relied on assistant-session state and contain phrasing such as "run this now" without reproducible acceptance criteria. Those instructions were intentionally removed from this canonical runbook.
- The old Copilot API troubleshooting note was incomplete and unrelated to core pipeline operations; keep editor/tooling notes in separate developer environment docs if needed.
- For high-volume cleanup jobs, periodically verify that no script reintroduces `find ... -delete` against active cron log trees.
