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

---

## Incident Addendum (2026-02-05): Lock Appears “Broken” (Repeated Lock Acquire + Immediate Exit)

### Symptom

- `guide_raw_to_corrected_0.log` (and peers) repeat the pattern:
  - “Acquired run lock …”
  - followed by `Station X already handled by another guide_raw_to_corrected.sh; exiting …`
  - a raw `ps` line with the cron launcher command shows up in the log.
- No lock file remains in `EXECUTION_LOGS/LOCKS/guide_raw_to_corrected`.

### Root Cause

`pipeline_already_running()` was parsing `ps -eo pid=,args=` output with a PID that included leading spaces.  
That prevented `pid_is_self_or_ancestor()` from recognizing the cron launcher (the script’s own parent) as “self,”
so the script thought it was *another* instance and exited immediately. The lock was not broken; the self‑detection was.

### Fix Applied

- Trim whitespace from the PID before comparison.
- Stop echoing the raw `ps` line into logs (kept as a TTY‑only debug message).

Updated file:
- `MASTER/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh`

### Verification

Run these checks:

```bash
pgrep -af "guide_raw_to_corrected.sh -s 0"
tail -n 5 /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected_0.log
```

Expected:
- Only one active process per station.
- No repeated “Acquired run lock …” followed by immediate “Station X already handled …” loops.

---

## Incident Addendum (2026-02-05): STEP_1 Process Explosion (Thousands of guide_raw_to_corrected.sh)

### Symptom

- `guide_raw_to_corrected_*.log` files updated extremely fast (hundreds of lines per minute).
- Memory and swap usage climbed quickly even though most station/task pairs were disabled.
- `pgrep -fc "guide_raw_to_corrected.sh"` returned a very large count (thousands).

### Root Cause

Cron was launching STEP_1 once per minute per station without a durable cron-level lock.
If a prior instance did not exit cleanly, the next minute spawned another instance.
Over time this created thousands of `guide_raw_to_corrected.sh` processes, which:

- Spammed logs with “disabled/omitted” messages.
- Consumed RAM and swap via sheer process count.

### Fix Applied

1. Killed all running `guide_raw_to_corrected.sh` processes.
2. Added cron-level `flock` guards so only one instance per station can run at a time.

Updated file:
- `add_to_crontab.info`

Example (new cron line form):
```
* * * * * /usr/bin/flock -n /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/LOCKS/cron/guide_raw_to_corrected_s0.lock /bin/bash /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh -s 0 >> /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected_0.log 2>&1
```

### Verification

```bash
pgrep -af "guide_raw_to_corrected.sh -s"
```

Expected:
- Only one active process per station (`-s 0` .. `-s 4`) plus the `flock` wrapper.

---

## Incident Addendum (2026-02-05): COPERNICUS Process Explosion (Swap Refills Quickly)

### Symptom

- Swap refilled to ~500 MB almost immediately after being cleared.
- System showed many `copernicus_bring.py` processes running simultaneously.
- `pgrep -fc "copernicus_bring.py"` returned a very large count (hundreds).

### Root Cause

Cron started `copernicus_bring.py` on a fixed schedule without a durable cron-level lock.
If a prior run did not exit in time, each scheduled tick launched another process.
Over time this created a large number of concurrent Copernicus fetchers, which:

- Consumed RAM and swap quickly.
- Produced misleading “system feels stuck” symptoms even after swap was cleared.

### Fix Applied

1. Manually killed all `copernicus_bring.py` processes.
2. Added cron-level `flock` guards (one per station) to prevent overlaps.
3. Wrapped Copernicus cron entries with the resource gate to skip runs when
   memory/swap/CPU are already high.

---

## Incident Addendum (2026-02-09): SIMULATION Continuous Run Stuck (Stale Lock)

### Symptom

- `cron_mingo_digital_twin_continuous.log` repeated:
  - "Continuous operation already running; exiting."
- No active `run_step.sh` process despite cron running every minute.
- STEP_0 log stayed empty and STEP_FINAL failed with missing `SIM_RUN_*` inputs.

### Root Cause

`run_step.sh -c` uses a lock directory in `/tmp`:
`/tmp/mingo_digital_twin_run_step_continuous.lock`. The PID inside the lock was
stale (process no longer running), so every new cron invocation exited early.

### Fix Applied

1. Removed the stale lock directory so cron could start a new continuous run.
2. Updated `run_step.sh` to auto-detect stale locks and recover:
   - If the PID is alive and `run_step.sh` is running, log a `[WARN]` with timestamp.
   - If the PID is missing or dead, remove the lock and continue.

Updated file:
- `MINGO_DIGITAL_TWIN/run_step.sh`

### Verification

```bash
pgrep -af "run_step.sh -c"
tail -n 5 /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/SIMULATION/RUN/cron_mingo_digital_twin_continuous.log
```

Expected:
- A running `run_step.sh -c` process.
- Log lines like `YYYY-MM-DDTHH:MM:SSZ run_step.sh continuous loop start`.
4. Added Copernicus to the process-count watchdog caps.

Updated file:
- `add_to_crontab.info`

Example (new cron line form):
```
* 2 * * * /bin/bash /home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/PIPELINE_OPERATIONS/RESOURCE_GATE/resource_gate.sh --tag copernicus_s2 --max-mem-pct 90 --max-swap-pct 80 --max-cpu-pct 90 -- /usr/bin/flock -n /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/LOCKS/cron/copernicus_bring_s2.lock /usr/bin/env python3 -u /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/COPERNICUS/STEP_1/copernicus_bring.py 2 >> /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/MAIN_ANALYSIS/STAGE_1/COPERNICUS/copernicus_bring_2.log 2>&1
```

### Verification

```bash
pgrep -af "copernicus_bring.py"
```

Expected:
- One active Copernicus process per station (1..4) plus the `flock` wrapper.
