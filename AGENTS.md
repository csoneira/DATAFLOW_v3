# AGENTS.md

This file provides guidance to coding agents (Claude Code and Codex) when working with code in this repository.

## Repository purpose

DATAFLOW_v3 hosts two side-by-side systems that share geometry and timing assumptions:

1. **Operational pipeline** (`MASTER/`, `STATIONS/`): ingests, cleans, and consolidates real cosmic-ray detector (miniTRASGO/MINGO) station data through four stages.
2. **MINGO digital twin** (`MINGO_DIGITAL_TWIN/`): generates synthetic RPC detector data by walking a muon event from generation through DAQ formatting in 11 deterministic steps.

## Setup

Install Python dependencies (Python 3.12, venv at `.venv/`):
```bash
pip install -r CONFIG/requirements.list
```
Also requires `yq` (YAML CLI tool) for Bash scripts.

All scripts assume paths rooted at `$HOME/DATAFLOW_v3`. Never embed absolute user-specific paths; use configured path references instead.

## Running the simulation pipeline

From within `MINGO_DIGITAL_TWIN/`:

```bash
# Generate parameter mesh (run once, or to expand the mesh)
python3 MASTER_STEPS/STEP_0/step_0_setup_to_blank.py \
  --config MASTER_STEPS/STEP_0/config_step_0_physics.yaml

# Run a single step
./run_step.sh 1

# Run all steps (1–10)
./run_step.sh all

# Run from a specific step onward
./run_step.sh --from 4 --no-plots

# Run STEP FINAL (DAQ → .dat formatting)
./run_step.sh final

# Main orchestration cycle (normally cron-driven)
./ORCHESTRATOR/core/run_main_simulation_cycle.sh
```

Each step script accepts `--config`, `--runtime-config`, `--plot-only`, `--no-plots`, and `--force` (overwrite existing SIM_RUN).

## Running tests

```bash
pytest MINGO_DIGITAL_TWIN/PLOTTERS/EXECUTION/MESH/tests/
```

## Simulation pipeline architecture

**Domain boundaries** (what changes at each group of steps):

| Steps | Domain |
|-------|--------|
| STEP 0 | Parameter mesh generation (`param_mesh.csv`) |
| STEP 1–2 | Muon generation and straight-line transport to detector planes |
| STEP 3 | RPC gas gap: ionization and avalanche |
| STEP 4–6 | Strip induction, charge sharing, front/back endpoint derivation |
| STEP 7–10 | Electronics: cable offsets, FEE model, trigger logic, TDC digitization |
| STEP FINAL | Format-only: write station-style `.dat` files |

**SIM_RUN naming and fan-out:**

- Steps 1–3 **fan out**: one upstream SIM_RUN produces multiple downstream SIM_RUNs (one per unique parameter group). SIM_RUN IDs are hierarchical: `SIM_RUN_{step1_id}_{step2_id}_{step3_id}`.
- Steps 4–10 are **1:1**: step ID is always `001`. The name extends the chain: `SIM_RUN_{...}_001_001_...`.
- SIM_RUN names are immutable once created; `--force` is required to overwrite.

**Intermediate data** lives in `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_N_TO_N+1/SIM_RUN_*/` (2–5 GB each). Cascade cleanup (steps 3–9) deletes consumed upstream intermediates after each step completes.

**Metadata:** every output has a `.meta.json` sidecar with `step`, `config`, `runtime_config`, `sim_run`, `config_hash`, `upstream_hash`, and lineage. Each INTERSTEP directory has a `sim_run_registry.json`. `ensure_sim_hashes.py` validates hashes daily.

**Backpressure:** `run_main_simulation_cycle.sh` skips STEP 0 if downstream has ≥ `SIM_MAX_UNPROCESSED_FILES` pending files (configured in `CONFIG_FILES/sim_main_pipeline_frequency.conf`). Set to 0 to disable.

## Operational pipeline architecture

Stages in `MASTER/STAGES/`:

- **STAGE 0**: rsync raw detector archives into per-station buffers
- **STAGE 1**: raw ASCII → cleaned event lists + aligned lab logs
- **STAGE 2**: pressure/temperature corrections, source merging
- **STAGE 3**: NMDB integration, enriched analytics tables

Station-specific trees live under `STATIONS/MINGO00`–`MINGO04`. Shared helpers are in `MASTER/common/`.

## Orchestration and operations

Scripts run via cron with `flock`-based locking; lock files, logs, and status markers live in `OPERATIONS_RUNTIME/`. Cron and tmux templates are in `FOR_MINGO_SYSTEMS/` and `CONFIG/add_to_crontab.info`.

Key operational scripts:
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/sanitize_sim_runs.py` – clean broken/orphaned SIM_RUNs (integrated into main cycle)
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/ensure_sim_hashes.py` – daily hash validation
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/prune_step_final_params.py` – housekeeping for final output catalog

## Hard rules (from CODEX.md)

- **Reproducibility**: results must be reproducible from committed code and config without manual steps.
- **Determinism**: set and document random seeds when randomness is used.
- **Configuration**: use existing config files and parameters; never hardcode values.
- **Outputs**: write derived outputs to designated locations; never overwrite raw inputs.
- **Never** modify or delete raw data files.
- **Never** silently change filtering, thresholds, or units without explicitly noting it.
- **Never** mix data from different runs without clear provenance.
- **Never** introduce new tools or workflows not already present in the repo.

Propose a plan before coding when changes span multiple files, affect data correctness, or alter outputs. For "done" to mean done, state clearly how correctness was validated.

## Key documentation

- Per-step interface contracts: `MINGO_DIGITAL_TWIN/DOCS/contracts/`
- Physics modeling details: `MINGO_DIGITAL_TWIN/DOCS/methods_overview.md`
- Column definitions per step: `MINGO_DIGITAL_TWIN/DOCS/data_dictionary.md`
- Config parameter reference: `MINGO_DIGITAL_TWIN/DOCS/config_reference.md`
- Coordinate and timing conventions: `MINGO_DIGITAL_TWIN/DOCS/coordinate_and_timing_conventions.md`
- Station `.dat` format spec: `MINGO_DIGITAL_TWIN/DOCS/station_dat_format.md`
- Pipeline orchestration details: `MINGO_DIGITAL_TWIN/DOCS/simulation_orchestration.md`
- Backpressure mechanism: `MINGO_DIGITAL_TWIN/DOCS/BACKPRESSURE_PIPELINE_DETAILS.md`
- Troubleshooting guides: `DOCS/REPO_DOCS/TROUBLESHOOTING/`
