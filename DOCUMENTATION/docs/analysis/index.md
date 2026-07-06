# Analysis software suite

*Last updated: July 2026*

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

- **Operational pipeline** (`MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/`): four sequential stages that
  operate on per‑station directories (e.g. `MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO01`):
  1. **STAGE 0** – ingest files from the station (raw archives, configuration,
     log snapshots) and prepare them for reprocessing.
  2. **STAGE 1** - clean and correct event data, normalize log/Copernicus side
     products, and collect the complete Stage 1 handoff under each station's
     `STAGE_1_PRODUCTS/` directory. This directory is essential: it
     contains all data needed for further analysis after Stage 1, including the
     event parquet lake, task metadata, log products, and Copernicus products.
  3. **STAGE 2** - consume `STAGE_1_PRODUCTS`, accumulate/correct event data,
     join event/log/Copernicus sources, and build integrated Stage 2 tables.
  4. **STAGE 3** – final enrichment and export for external services (e.g.
     NMDB submission).

  Each stage is driven by a small shell wrapper script which in turn invokes a
  Python module; configuration is handled via YAML files in
  `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES`.  Details and troubleshooting guides for each stage
  reside under `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/DOCS/`.

  Execution is typically handled by cron jobs defined in
  `CONFIG/add_to_crontab.info` and by wrapper scripts in `OPERATIONS/OPERATIONS_SCRIPTS/` which
  enforce resource limits (CPU/memory) and locking.  See
  `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION` for the resource gate and watchdog utilities.

- **Station helpers** (`MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/`): contains routines specific to each
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
  The dictionary is loaded via `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/simulated_data_utils.py` and is
  automatically invoked by rate‑calculation routines.  Refer to the
  [Dictionary correction](../dictionary/index.md) page for more.

## Directory structure and key artefacts

```
DATAFLOW_v3/
├─ MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/
│   ├─ STAGES/                # core processing scripts for each stage
│   ├─ CONFIG_FILES/          # YAML configuration used by the stages
│   ├─ DOCS/                  # documentation specific to the analysis code
│   └─ common/                # shared utilities (e.g. simulated_data_utils.py)
├─ MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/                 # per-station helpers and data pointers
├─ OPERATIONS/OPERATIONS_SCRIPTS/               # orchestration & automation scripts
├─ MINGO_DIGITAL_TWIN/       # simulation (see relevant doc page)
└─ …
```

| Component | Location | Description |
|-----------|----------|-------------|
| Analysis code | `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/` & `MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/` within this repo | Core pipeline scripts, station-specific helpers, selection/config logic |
| Documentation | `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/DOCS/` (see `README.md`) | Processing stage guides, troubleshooting, QA plots |
| Docker environment | `CONFIG_FILES/docker_analysis.yaml` | Reproducible runtime for analysts and CI |
| Stage 1 product bundle | `MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO0X/STAGE_1_PRODUCTS/` | Essential handoff directory. Contains all data needed for further analysis after Stage 1: event parquet lake, task metadata, LOG_DATA products, and Copernicus products. |

This repository is the authoritative source for both the software and the
operational procedure; changes to `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES` should be accompanied by
updates to `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/DOCS` and appropriate regression tests.


## Getting started

The analysis code has its own dedicated repository; the steps below assume you
are setting up a new analyst workstation.  Much of the same repository is
also checked out on the main analysis PC, which runs the cron-driven pipeline.

The modularity of the stages means you can inspect or run them individually.  A
few representative script names are:

- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/NEW_FILES/bring_data_and_config_files.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_1/clean_event_data.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/LOG_DATA/STEP_2/lab_logs_merge.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_2/NMDB/merge_nmdb.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_3/export_nmdb.py`

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

Before editing any configuration, run `python MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/config_loader.py
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
python MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_1/clean_event_data.py \
    --input /path/to/raw/archive.tar.gz
```

For interactive exploration, the `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/DOCS` directory contains example
notebooks and command references.  Analysts modifying the pipeline should use
`pytest` to run the unit/regression tests around `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common` utilities.

### Full pipeline test run

A convenience script `OPERATIONS/OPERATIONS_SCRIPTS/run_all.sh` can be used to exercise the
entire pipeline on a small synthetic dataset.  It sets up a temporary `STAGE_0`
tree, copies a handful of raw archives (checked into `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/TEST_DATA/`), and
then runs through STAGE 1–3 sequentially.  Example invocation:

```bash
cd $DATAFLOW_ROOT
bash OPERATIONS/OPERATIONS_SCRIPTS/run_all.sh --stages 0 1 2 --stations 00 01
```

Logs from this run are written to `OPERATIONS/OPERATIONS_RUNTIME/CRON_LOGS/test_run/` and
the resulting analytics tables appear under `MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/` etc.  This is a
handy way to verify that any code changes haven't broken the dataflow without
waiting for real station data to arrive.

### Environment variables and cron example

A handful of ancillary scripts, such as the documentation plot synchroniser
(`DOCUMENTATION/docs/assets/update_plots.sh`), are also scheduled from the
same crontab.  The `ADD_TO_CRONTAB.INFO` file already contains an entry that
runs this helper every 30 minutes; modifying that schedule or the script is a
routine way to keep the public website up to date with new diagnostic figures.

### Logs and troubleshooting

All cron‑launched jobs redirect stdout/stderr into the `OPERATIONS/OPERATIONS_RUNTIME`
log tree.  Logs are grouped by functional area (`ANCILLARY`, `SIMULATION`,
`MAIN_ANALYSIS`, `PLOTTERS`) and further subdivided by stage.  When investigating
a failure:

1. Determine which stage or script produced the error (the cron log filename
   usually contains the script name).
2. `tail -n 100` the corresponding log file; the stack trace or shell error
   message is usually there.
3. Check for stale lock files under `OPERATIONS/OPERATIONS_RUNTIME/LOCKS/cron`; remove
them only if you are certain no active pipeline is running (use
`pgrep -f run_main_simulation_cycle.sh` to verify).
4. Use the `error_finder.py` tool (`OPERATIONS/OPERATIONS_SCRIPTS/OBSERVABILITY/SEARCH_FOR_ERRORS`)
   to scan all logs for Python tracebacks and alert messages.

The `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/DOCS` folder includes a more comprehensive runbook for analyzing
pipeline incidents; search for keywords such as "stale lock" or
"missing metadata" there.

Routine maintenance tasks (auto‑clear swap, solve stale locks, clean temp
files) are also defined in `add_to_crontab.info` and run every few minutes.

Several environment variables influence pipeline behaviour; these are set in
cron via `CONFIG/add_to_crontab.info` on the analysis PC (see that file for the
full list).  The key variables include:

- `DATAFLOW_ROOT` – root path of this repository (e.g. `/home/mingo/DATAFLOW_v3`).
- `MASTER_STAGE_ROOT` – shorthand for `$DATAFLOW_ROOT/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES`.
- `SIM_PROCESSING_RUN_STEP_FINAL` – when set to zero, prevents automatic
  STEP_FINAL during the main simulation cycle (cron holds separate ownership).

A typical cron entry for running STAGE 0 ingestion looks like this:

```cron
*/10 * * * * /bin/bash $HOME/DATAFLOW_v3/OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/RESOURCE_GATE/resource_gate.sh \
    --tag sim_ingest --max-mem-pct 90 --max-swap-pct 95 --max-cpu-pct 90 \
    -- /usr/bin/flock -n $HOME/DATAFLOW_v3/OPERATIONS/OPERATIONS_RUNTIME/LOCKS/cron/sim_ingest_station_data.lock \
    /usr/bin/env python3 $MASTER_STAGE_ROOT/STAGE_0/SIMULATION/ingest_simulated_station_data.py \
    >> $HOME/DATAFLOW_v3/OPERATIONS/OPERATIONS_RUNTIME/CRON_LOGS/MAIN_ANALYSIS/STAGE_0/SIMULATION/ingest_simulated_station_data.log 2>&1
```

Copying and examining this file is often the fastest way to understand the
enterprise-wide processing schedule.

### Running the full pipeline locally

A `run_all.sh` helper in `OPERATIONS/OPERATIONS_SCRIPTS/` mimics the cron behaviour and may be
used to process a small subset of data for testing.  Consult
`MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/DOCS/` for instructions on configuring
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
