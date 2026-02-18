# MINGO_DIGITAL_TWIN Cron Expected Behaviour

Last updated: 2026-02-18  
Source of truth: `CONFIG/add_to_crontab.info`

## Scope
This document covers cron-triggered processes whose executed scripts live under:
- `/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN`

## Global Expectations
1. Main simulation cycle is cron-triggered every minute but effectively rate-limited by frequency gating inside `run_main_simulation_cycle.sh`.
2. `sim_main_pipeline.lock` serializes:
   - main simulation cycle
   - `sanitize_sim_runs.py`
   - `ensure_sim_hashes.py`
3. Lock contention is expected; skipped runs due `flock -n` are acceptable.
4. `stop_execution.sh` may keep cron log empty while writing internal details to `stop_execution_internal.log`.

## Expected Process Inventory

| Cron Schedule | Process | Lock / Gate | Expected Behaviour | Primary Log |
|---|---|---|---|---|
| `* * * * *` | `MINGO_DIGITAL_TWIN/ANCILLARY/run_main_simulation_cycle.sh` | `flock sim_main_pipeline.lock` + internal frequency gate | Ordered cycle: `STEP_0 -> run_step.sh all --no-plots -> STEP_FINAL -> prune_completed_param_mesh_rows`; writes stage-level status lines | `OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/RUN/sim_main_pipeline_cycle.log` |
| `*/15 * * * *` | `MINGO_DIGITAL_TWIN/ANCILLARY/sanitize_sim_runs.py --apply --min-age-seconds 900` | `flock sim_main_pipeline.lock` | Deletes broken/stale `SIM_RUN_*` step outputs and keeps INTERSTEPS clean, respecting STEP_FINAL lock awareness | `OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/ANCILLARY/sanitize_sim_runs.log` |
| `25 6 * * *` | `MINGO_DIGITAL_TWIN/ANCILLARY/ensure_sim_hashes.py --apply` | `flock sim_main_pipeline.lock` | Enforces/repairs param-hash consistency in simulated `.dat` and parquet outputs; may delete mismatched files | `OPERATIONS_RUNTIME/CRON_LOGS/ANCILLARY/ensure_sim_hashes.log` |
| `*/2 * * * *` | `MINGO_DIGITAL_TWIN/ANCILLARY/stop_execution.sh` | internal `/tmp/dataflow_stop_execution.lock` | Emergency guard: if process duplication/load thresholds are exceeded, terminate matching simulation pipeline jobs and clean stale locks | `OPERATIONS_RUNTIME/CRON_LOGS/ANCILLARY/CLEANERS/stop_execution.log` (internal details: `.../stop_execution_internal.log`) |

## Main Simulation Cycle: Exact Runtime Behaviour
Expected from `run_main_simulation_cycle.sh`:
1. Load frequency-gate config from:
   - `MINGO_DIGITAL_TWIN/CONFIG_FILES/sim_main_pipeline_frequency.conf`
2. If gate says skip, cycle logs:
   - `cycle status=skipped reason=frequency_gate ...`
3. If running, stage order must be:
   - `stage=step_0 status=start|ok`
   - `stage=run_step_all status=start|ok`
   - `stage=step_final status=start|ok`
   - `stage=prune_mesh_done_rows status=start|ok` (non-fatal if it fails)
4. Final status must be one of:
   - `cycle status=ok`
   - `cycle status=failed`

## Health Signals
1. Healthy run evidence in main log:
   - `[SIM_CYCLE] [INFO] stage=... status=ok`
   - `STEP_FINAL` emits `.dat` save lines into `MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES`.
2. `sanitize_sim_runs.log` and `ensure_sim_hashes.log` may be empty if:
   - jobs were lock-skipped, or
   - no actionable work was found.
3. `stop_execution.log` can be empty by design; inspect:
   - `OPERATIONS_RUNTIME/CRON_LOGS/ANCILLARY/CLEANERS/stop_execution_internal.log`

## Machine-Check Assertions (for future automation)
1. `sim_main_pipeline_cycle.log` mtime is fresh (for example within `2x` of configured frequency gate interval).
2. Recent lines include stage transitions and either `cycle status=ok` or `cycle status=skipped reason=frequency_gate`.
3. No stale lock deadlock:
   - `sim_main_pipeline.lock` should not remain blocked with no matching process.
4. `stop_execution_internal.log` should periodically report either:
   - `Healthy: matched=...`
   - or explicit triggered emergency stop events.
5. When cycle runs successfully, new files should appear periodically under:
   - `MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES`

