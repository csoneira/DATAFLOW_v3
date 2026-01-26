# Tools and Validation

This document describes built-in tools for inspecting, validating, and managing
MINGO_DIGITAL_TWIN outputs.

## Event microscope
Path: `tools/event_microscope.py`

Purpose:
- Trace a single event through STEP 1..10 using metadata chain resolution.
- Print per-plane and per-strip values for a selected event ID.

Example:
```
python3 tools/event_microscope.py --sim-run SIM_RUN_0001 --geometry-id 0 --event-id 150
```

Key options:
- `--sim-run`: SIM_RUN directory or `latest`/`random`.
- `--geometry-id`: plane geometry index.
- `--event-id`: target event ID.

## Timing closure check
Path: `tools/timing_closure.py`

Purpose:
- Verify algebraic consistency between STEP 5 and STEP 6 (front/back closure).
- Compute residuals across sampled events.

Example:
```
python3 tools/timing_closure.py --sim-run SIM_RUN_0001 --geometry-id 0 --max-events 50000 --fail-on-bad
```

## SIM_RUN summary
Path: `MASTER_STEPS/STEP_SHARED/sim_run_summary.py`

Purpose:
- Summarize row counts per SIM_RUN across INTERSTEPS.

Example:
```
python3 MASTER_STEPS/STEP_SHARED/sim_run_summary.py --root .
```

## Ancillary management scripts
- `ANCILLARY/size_and_expected_report.py`:
  - Estimates expected SIM_RUN counts from `param_mesh.csv` and reports storage usage.
- `ANCILLARY/sanitize_sim_runs.py`:
  - Removes SIM_RUN directories whose step ID combinations are marked done in the mesh.

These scripts are intended for batch management and may remove data when run with
`--apply` (use caution).
