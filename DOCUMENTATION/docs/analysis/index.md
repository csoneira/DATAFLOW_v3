# Analysis software suite

*Last updated: March 2026*

The miniTRASGO analysis codebase (hosted at
[github.com/cayesoneira/miniTRASGO-analysis](https://github.com/cayesoneira/miniTRASGO-analysis))
implements the operational pipeline that converts raw detector and log data
into calibrated event tables, environmental corrections, and physics
quantities.  This page summarises the high‑level architecture and points to
relevant components and documentation.

## Overview

The analysis software implements a modular, automated processing chain that
transforms raw station data into calibrated, physics‑ready products.  It is
composed of several conceptual layers:

- **Operational pipeline** (`MASTER/STAGES/`): four sequential stages that
  operate on per‑station directories (e.g. `STATIONS/MINGO01`):
  1. **STAGE 0** – ingest files from the station (raw archives, configuration,
     log snapshots) and prepare them for reprocessing.
  2. **STAGE 1** – clean and correct event data, applying pressure/temperature
     adjustments and merging laboratory log entries.
  3. **STAGE 2** – perform environmental corrections, merge NMDB cosmic ray
     reference data, and compute analytics tables such as flux rates.
  4. **STAGE 3** – final enrichment and export for external services (e.g.
     NMDB submission).

  Each stage is driven by a small shell wrapper script which in turn invokes a
  Python module; configuration is handled via YAML files in
  `MASTER/CONFIG_FILES`.  Details and troubleshooting guides for each stage
  reside under `MASTER/DOCS/`.

  Execution is typically handled by cron jobs defined in
  `CONFIG/add_to_crontab.info` and by wrapper scripts in `OPERATIONS/` which
  enforce resource limits (CPU/memory) and locking.  See
  `OPERATIONS/ORCHESTRATION` for the resource gate and watchdog utilities.

- **Station helpers** (`STATIONS/`): contains routines specific to each
  physical detector for bringing data from the remote RPC computers and for
  restarting services after network outages.

- **Simulation ingestion**: synthetic `.dat` files produced by the digital twin
  are treated identically to real hardware outputs.  An ingestion script under
  `STAGE_0/SIMULATION` copies them into the processing tree; they may be
  generated locally or pulled from the simulation server.  This allows
  algorithm development and dictionary validation to proceed without station
  access.

- **Dictionary-based inference**: flux and efficiency are estimated from
  measured rates using precomputed lookup tables built from the digital twin.
  The dictionary is loaded via `MASTER/common/simulated_data_utils.py` and is
  automatically invoked by rate‑calculation routines.  Refer to the
  [Dictionary correction](../dictionary/index.md) page for more.

## Directory structure and key artefacts

```
DATAFLOW_v3/
├─ MASTER/
│   ├─ STAGES/                # core processing scripts for each stage
│   ├─ CONFIG_FILES/          # YAML configuration used by the stages
│   ├─ DOCS/                  # documentation specific to the analysis code
│   └─ common/                # shared utilities (e.g. simulated_data_utils.py)
├─ STATIONS/                 # per-station helpers and data pointers
├─ OPERATIONS/               # orchestration & automation scripts
├─ MINGO_DIGITAL_TWIN/       # simulation (see relevant doc page)
└─ …
```

| Component | Location | Description |
|-----------|----------|-------------|
| Analysis code | `MASTER/` & `STATIONS/` within this repo | Core pipeline scripts, station-specific helpers, selection/config logic |
| Documentation | `MASTER/DOCS/` (see `README.md`) | Processing stage guides, troubleshooting, QA plots |
| Docker environment | `CONFIG_FILES/docker_analysis.yaml` | Reproducible runtime for analysts and CI |

This repository is the authoritative source for both the software and the
operational procedure; changes to `MASTER/STAGES` should be accompanied by
updates to `MASTER/DOCS` and appropriate regression tests.

| Component | Location | Description |
|-----------|----------|-------------|
| Analysis code | `MASTER/` & `STATIONS/` within this repo | Core pipeline scripts, station-specific helpers, selection/config logic |
| Documentation | `MASTER/DOCS/` (see `README.md`) | Processing stage guides, troubleshooting, QA plots |
| Docker environment | `CONFIG_FILES/docker_analysis.yaml` | Reproducible runtime for analysts and CI |

## Getting started

The analysis code has its own dedicated repository; the steps below assume you
are setting up a new analyst workstation.  Much of the same repository is
also checked out on the main analysis PC, which runs the cron-driven pipeline.

The modularity of the stages means you can inspect or run them individually.  A
few representative script names are:

- `MASTER/STAGES/STAGE_0/NEW_FILES/bring_data_and_config_files.sh`
- `MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/clean_event_data.py`
- `MASTER/STAGES/STAGE_1/LAB_LOGS/STEP_2/lab_logs_merge.py`
- `MASTER/STAGES/STAGE_2/NMDB/merge_nmdb.py`
- `MASTER/STAGES/STAGE_3/export_nmdb.py`

Configuration files are plain YAML; here is a minimal `stage_1_config.yaml`
excerpt showing some typical fields (taken from the repository):

```yaml
thresholds:
  tdc: 200   # ps
  rate: 0.5  # Hz per strip
geometry:
  plane_separation_cm: 30.0
  strip_count: 32
nmdb:
  api_key: '${CDSAPI_KEY}'
```

Before editing any configuration, run `python MASTER/common/config_loader.py
--validate <file>` to ensure your YAML is well-formed and contains all required
keys.  The pipeline will abort with a clear error message if a config check
fails.

```bash
# clone the repository
git clone https://github.com/cayesoneira/miniTRASGO-analysis.git
cd miniTRASGO-analysis

# create a Python virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ensure you have access to the DATAFLOW_v3 repo in a sibling directory
# (this is where data, config, and the digital twin live)
export DATAFLOW_ROOT=$HOME/DATAFLOW_v3

# run an individual Python stage on a local dataset
python MASTER/STAGES/STAGE_1/EVENT_DATA/STEP_1/clean_event_data.py \
    --input /path/to/raw/archive.tar.gz
```

For interactive exploration, the `MASTER/DOCS` directory contains example
notebooks and command references.  Analysts modifying the pipeline should use
`pytest` to run the unit/regression tests around `MASTER/common` utilities.

### Full pipeline test run

A convenience script `OPERATIONS/run_all.sh` can be used to exercise the
entire pipeline on a small synthetic dataset.  It sets up a temporary `STAGE_0`
tree, copies a handful of raw archives (checked into `MASTER/TEST_DATA/`), and
then runs through STAGE 1–3 sequentially.  Example invocation:

```bash
cd $DATAFLOW_ROOT
bash OPERATIONS/run_all.sh --stages 0 1 2 --stations 00 01
```

Logs from this run are written to `OPERATIONS_RUNTIME/CRON_LOGS/test_run/` and
the resulting analytics tables appear under `STATIONS/MINGO00/` etc.  This is a
handy way to verify that any code changes haven't broken the dataflow without
waiting for real station data to arrive.

### Environment variables and cron example

A handful of ancillary scripts, such as the documentation plot synchroniser
(`DOCUMENTATION/docs/assets/update_plots.sh`), are also scheduled from the
same crontab.  The `ADD_TO_CRONTAB.INFO` file already contains an entry that
runs this helper every 30 minutes; modifying that schedule or the script is a
routine way to keep the public website up to date with new diagnostic figures.

### Logs and troubleshooting

All cron‑launched jobs redirect stdout/stderr into the `OPERATIONS_RUNTIME`
log tree.  Logs are grouped by functional area (`ANCILLARY`, `SIMULATION`,
`MAIN_ANALYSIS`, `PLOTTERS`) and further subdivided by stage.  When investigating
a failure:

1. Determine which stage or script produced the error (the cron log filename
   usually contains the script name).
2. `tail -n 100` the corresponding log file; the stack trace or shell error
   message is usually there.
3. Check for stale lock files under `OPERATIONS_RUNTIME/LOCKS/cron`; remove
them only if you are certain no active pipeline is running (use
`pgrep -f run_main_simulation_cycle.sh` to verify).
4. Use the `error_finder.py` tool (`OPERATIONS/OBSERVABILITY/SEARCH_FOR_ERRORS`)
   to scan all logs for Python tracebacks and alert messages.

The `MASTER/DOCS` folder includes a more comprehensive runbook for analyzing
pipeline incidents; search for keywords such as "stale lock" or
"missing metadata" there.

Routine maintenance tasks (auto‑clear swap, solve stale locks, clean temp
files) are also defined in `add_to_crontab.info` and run every few minutes.

Several environment variables influence pipeline behaviour; these are set in
cron via `CONFIG/add_to_crontab.info` on the analysis PC (see that file for the
full list).  The key variables include:

- `DATAFLOW_ROOT` – root path of this repository (e.g. `/home/mingo/DATAFLOW_v3`).
- `MASTER_STAGE_ROOT` – shorthand for `$DATAFLOW_ROOT/MASTER/STAGES`.
- `SIM_PROCESSING_RUN_STEP_FINAL` – when set to zero, prevents automatic
  STEP_FINAL during the main simulation cycle (cron holds separate ownership).

A typical cron entry for running STAGE 0 ingestion looks like this:

```cron
*/10 * * * * /bin/bash $HOME/DATAFLOW_v3/OPERATIONS/ORCHESTRATION/RESOURCE_GATE/resource_gate.sh \
    --tag sim_ingest --max-mem-pct 90 --max-swap-pct 95 --max-cpu-pct 90 \
    -- /usr/bin/flock -n $HOME/DATAFLOW_v3/OPERATIONS_RUNTIME/LOCKS/cron/sim_ingest_station_data.lock \
    /usr/bin/env python3 $MASTER_STAGE_ROOT/STAGE_0/SIMULATION/ingest_simulated_station_data.py \
    >> $HOME/DATAFLOW_v3/OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/SIMULATION/ingest_simulated_station_data.log 2>&1
```

Copying and examining this file is often the fastest way to understand the
enterprise-wide processing schedule.

### Running the full pipeline locally

A `run_all.sh` helper in `OPERATIONS/` mimics the cron behaviour and may be
used to process a small subset of data for testing.  Consult
`MASTER/DOCS/` for instructions on configuring 
`SIMULATION/` ingestion, NMDB credentials, and output directories.

---

_Cross-reference:_ the [Home](../index.md) page, [Digital twin](../simulation/index.md),
and [Dictionary correction](../dictionary/index.md) pages all describe
components of the analysis workflow.

For detailed usage consult the `README.md` in each stage directory and the
`DOCS/` subtree.

---

_Cross-reference:_ the [Home](../index.md) page, [Digital twin](../simulation/index.md),
and [Dictionary correction](../dictionary/index.md) pages all describe
components of the analysis workflow.
