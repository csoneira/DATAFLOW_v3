# Troubleshooting Report: `guide_raw_to_corrected` Not Updating Cron Logs

Date: 2026-02-04  
Scope: STAGE_1 / EVENT_DATA / STEP_1 cron execution (`guide_raw_to_corrected.sh`)

## Summary

The STEP_1 cron jobs were not running because the **cron daemon itself was down**.  
Manual runs worked, which made it look like a script problem, but the root cause was service-level.

## What Happened

1. A redirection error was seen first (`No such file or directory` for a log file path), and log-path creation safeguards were added.
2. Even after log-path safeguards, STEP_1 logs stopped advancing.
3. Evidence showed cron activity had stalled:
   - `guide_raw_to_corrected_0.log` stopped at `2026-02-04 10:39:18 +0000`
   - several cron logs stopped around `2026-02-04 10:47:14 +0000`
   - no active `guide_raw_to_corrected.sh -s X` processes
4. Checking cron service state showed:
   - `cron.service` had been **failed (oom-kill)** since `Tue 2026-02-03 21:07:03 WET`
5. After restarting cron manually, jobs resumed:
   - `Active: active (running) since Wed 2026-02-04 11:01:53 WET`
   - STEP_1 logs started updating again (`~11:04 UTC` onwards)

## Root Cause

Primary root cause:
- `cron.service` was terminated by OOM kill and remained down.

Secondary/confusing factor:
- Earlier missing log directories caused shell redirection failures before scripts started, which masked diagnosis at the beginning.

## Fix Applied

1. Added a dedicated cron log-path checker script:
   - `MASTER/ANCILLARY/PIPELINE_OPERATIONS/ENSURE_CRON_LOG_PATHS/ensure_cron_log_paths.sh`
2. Added cron entry to run checker every minute (`--create-files --quiet`), so all log targets in `add_to_crontab.info` always exist.
3. Restored STEP_1/2/3 cron command lines to clean form (no inline `mkdir` workaround).
4. Restarted cron service manually (requires sudo/password in terminal).

## Verification

Use these checks:

```bash
service cron status
pgrep -af "guide_raw_to_corrected.sh -s"
stat -c '%y %n' /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected_{0,1,2,3,4}.log
```

Expected:
- `cron` is `active (running)`
- one long-lived process per station argument (`-s 0`, `-s 1`, etc.)
- log mtimes keep moving forward

## Notes

- `KeyboardInterrupt` in STEP_1 logs indicates a manual interrupt (`Ctrl+C`), not cron failure.
- If manual execution works but cron execution does not, always check cron daemon state early.

---

## Incident Addendum (2026-02-04): Metadata Moves, Cron Log Looks Stuck

### Symptom

- `task_1_metadata_status.csv` and task metadata kept updating.
- `guide_raw_to_corrected_*.log` only showed repetitive launcher messages (or appeared to miss detailed task output).

### Root Cause

Hourly cleanup was deleting files inside `EXECUTION_LOGS/CRON_LOGS` while long-lived STEP_1 processes were still running:

- cron entry:
  - `add_to_crontab.info` -> `clean_dataflow.sh --force`
- cleanup behavior before fix:
  - `MASTER/ANCILLARY/CLEANERS/clean_dataflow.sh` used `find "$CRON_LOG_DIR" -mindepth 1 -delete`

When a running process writes to a deleted log file descriptor, data goes to the old (unlinked) inode, not to the newly recreated visible log path. This makes processing real, but visible logs misleading.

### Fix Applied

Changed cron-log cleanup from **delete** to **truncate** so paths/inodes stay stable for running writers:

- Updated: `MASTER/ANCILLARY/CLEANERS/clean_dataflow.sh`
- `clean_cronlogs()` now truncates each file (`: > "$file"`) instead of deleting log files/directories.

### Prevention Rule

For any long-running cron process:

1. Do not delete active log files during cleanup.
2. Use truncation/rotation mechanisms that preserve writable file handles.
3. If logs look stale but metadata/status moves, check for deleted-log-handle behavior first.
