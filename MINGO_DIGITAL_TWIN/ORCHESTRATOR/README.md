# Orchestrator Layout

This directory contains orchestration logic for the simulation pipeline.

## `core/`

Primary entrypoints:

- `run_main_simulation_cycle.sh` (minute-cycle controller for enqueue + processing)
- `run_step.sh` (step scheduler for STEP_1..STEP_10 + optional STEP_FINAL)

## `maintenance/`

Operational maintenance and repair scripts used by cron and/or core orchestrators:

- `ensure_sim_hashes.py`
- `prune_completed_param_mesh_rows.py`
- `prune_step_final_params.py`
- `repair_param_mesh_step_ids.py`
- `sanitize_sim_runs.py`
- `stop_execution.sh`

## `helpers/`

Non-critical helper tools used by scheduler preflight and troubleshooting:

- `check_param_mesh_consistency.py`
- `cascade_cleanup_intersteps.py`
- `refresh_step_work_cache.py`
- `obliterate_open_lines_for_fixed_z.py`
- `size_and_expected_report.py`

## `tests/`

Ancillary/orchestrator tests moved from the previous `ANCILLARY/tests` location.
