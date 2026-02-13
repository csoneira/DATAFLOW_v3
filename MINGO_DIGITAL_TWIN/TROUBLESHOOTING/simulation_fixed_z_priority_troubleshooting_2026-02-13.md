# Simulation Troubleshooting: Fixed `z_positions` Priority and Stalls

Date: 2026-02-13 (UTC)

## Scope

This document covers the troubleshooting and fixes for the continuous simulation pipeline under:

- `/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN`

Focused files:

- `MASTER_STEPS/STEP_2/config_step_2_physics.yaml`
- `MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py`
- `run_step.sh`
- `ANCILLARY/obliterate_open_lines_for_fixed_z.py`
- `add_to_crontab.info`

## Symptoms Observed

- `STEP_2` failed repeatedly in cron with:
  - `ValueError: Unsupported input format:`
- No visible progress in downstream outputs for a period.
- Strict line closure prevented opening new `step_1` lines while fixed `z_positions` were desired.

## Root Causes

### 1) Strict closure + fixed z subset blocked opening new lines

- `strict_line_closure=1` means no new `step_1` lines can open while one active line still has pending rows.
- With fixed `z_positions` in `STEP_2`, many pending mesh rows in the active line were not relevant to the fixed geometry.
- Result: line remained open but not useful for the fixed-z objective, and new lines stayed blocked.

### 2) Incomplete `STEP_1_TO_2` input run triggered `STEP_2` crash

- At least one `SIM_RUN_*` in `INTERSTEPS/STEP_1_TO_2` had chunk files but no manifest:
  - had `muon_sample_*/chunks/*`
  - missing `muon_sample_*.chunks.json`
- `STEP_2` random input selection could hit this incomplete run and pass an invalid path to metadata/data loading.
- That produced:
  - `ValueError: Unsupported input format:`

## Fixes Applied

### A) Auto-obliterate non-interest active lines for fixed z

Added helper script:

- `MINGO_DIGITAL_TWIN/ANCILLARY/obliterate_open_lines_for_fixed_z.py`

Behavior:

- Reads fixed z from `MASTER_STEPS/STEP_2/config_step_2_physics.yaml`.
- Resolves matching fixed `step_2_id` in `param_mesh.csv`.
- Detects active open `step_1` lines.
- If an active line only has pending non-fixed `step_2_id` rows, marks those pending rows `done=1` (line gets cleared).
- Uses lock + atomic write protections.

### B) Integrated auto-obliteration into continuous orchestrator

Updated:

- `MINGO_DIGITAL_TWIN/run_step.sh`

Added env toggle:

- `RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES=1`

When strict closure blocks step 1, `run_step.sh` can now auto-unblock by invoking the helper script, then continue.

### C) Enabled feature in cron

Updated simulation cron command in:

- `add_to_crontab.info`

It now runs with:

- `/usr/bin/env RUN_STEP_OBLITERATE_UNINTERESTING_STEP1_LINES=1`

### D) Hardened `STEP_2` input path/meta handling

Updated:

- `MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py`

Key hardening:

- Incomplete chunk-only runs (missing `.chunks.json`) are treated as incomplete and skipped.
- Metadata load is defensive and returns `{}` instead of crashing on unsupported paths/bad files.
- Prevents repeated hard-fail loop on malformed input candidates.

## Verification That It Recovered

Cron log showed transition from repeated failure to sustained progress:

- Failure window ended around `2026-02-13T10:20:03Z`.
- Progress resumed at `2026-02-13T10:22:20Z` and continued through steps 3..10 repeatedly.

Process health also confirmed:

- Continuous runner process alive:
  - `run_step.sh -c --no-plots`
- Active child steps observed with high CPU.
- New `SIM_RUN_*` directories created across intersteps.

## Quick Health Checks

### 1) Confirm continuous runner and active step

```bash
ps -ef | rg 'run_step\.sh -c --no-plots|MASTER_STEPS/STEP_[0-9]+/step_'
```

### 2) Watch simulation progress log

```bash
tail -n 80 /home/mingo/DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS/SIMULATION/RUN/cron_mingo_digital_twin_continuous.log
```

Look for `status=progress` lines, not repeated `failed rc=1`.

### 3) Check strict-closure state cache

```bash
cat /tmp/mingo_digital_twin_run_step_state.csv
cat /tmp/mingo_digital_twin_run_step_stuck_lines.csv
```

### 4) Detect incomplete chunk runs in `STEP_1_TO_2`

```bash
for d in /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_1_TO_2/SIM_RUN_*; do
  for base in "$d"/muon_sample_*; do
    [ -d "$base/chunks" ] || continue
    [ -f "${base}.chunks.json" ] || echo "INCOMPLETE: $base (missing ${base}.chunks.json)"
  done
done
```

## Operational Notes

- `MINGO_DIGITAL_TWIN/ANCILLARY/size_and_expected_report.py` is source code; it is not expected to change while jobs run.
- Runtime health should be checked via:
  - cron logs
  - process list
  - new output directories/files in `INTERSTEPS`.
- If `run_step.sh` logic is edited, restart the long-running continuous process so the new script version is actually used.

## If It Happens Again (Short Playbook)

1. Check `cron_mingo_digital_twin_continuous.log` for repeated identical failures.
2. Check `/tmp/mingo_digital_twin_last_step_2.log` for current traceback.
3. Check for incomplete chunk-only inputs in `STEP_1_TO_2`.
4. Check strict closure cache (`/tmp/mingo_digital_twin_run_step_state.csv`).
5. If blocked and fixed-z priority is intended, run:
   - `python3 /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/ANCILLARY/obliterate_open_lines_for_fixed_z.py --apply`
6. Ensure continuous runner is restarted if control script was changed.
