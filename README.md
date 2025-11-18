# Unified Dataflow for miniTRASGO Cosmic Ray Network

## Overview
This repository hosts the pipeline used to ingest, process, and consolidate cosmic ray observations collected by the **miniTRASGO** network. It includes the bash and Python tooling that currently runs in production for four stations and the orchestration snippets (cron, tmux) needed to keep the jobs alive on-premise.

The project brings together:
- **Detector event streams** from each miniTRASGO station (miniTRASGO01–04).
- **Laboratory logbooks** produced by operators and local environmental probes.
- **Copernicus ERA5 reanalysis** products, used for atmospheric corrections.

## Architecture

### Execution topology

- `MASTER/` hosts the authoritative pipeline scripts, configuration snippets, and `common/` helpers that standardize logging (`execution_logger.py`), configuration loading, plotting, and the station status CSVs used by control dashboards.
- `STATIONS/MINGO0X/` mirrors the stage folders (`STAGE_0`, `STAGE_0_to_1`, `STAGE_1`, `STAGE_2`, optional `STAGE_3`) for each detector. It contains only artefacts and is rewritten by the MASTER scripts. `MASTER/STATIONS/` stores the skeleton used when provisioning a new station.
- Operational tooling lives under `MASTER/ANCILLARY/` (real-time pipeline checks, Telegram bot watchdogs, plotters, simulators, disk cleaners) and `FOR_MINGO_SYSTEMS/` (shell/tmux templates deployed to each host). Runtime metadata is collected inside `EXECUTION_LOGS/` (cron output, nohup logs, lock files).
- Shared data products are published to `GRAFANA_DATA/` (CSV/Parquet tables) and the `grafana/` provisioning tree, while `MASTER/STAGE_3/` retains NMDB downloads and analysis notebooks that are consumed by every station.

### Pipeline stages

1. **Stage 0 – Acquisition & buffering** (`MASTER/STAGE_0/NEW_FILES`, `MASTER/STAGE_0/REPROCESSING`, `STATIONS/<ID>/STAGE_0`): rsync/bring new detector archives, fan them into per-station buffers, and replay historical batches through `UNPACKER_ZERO_STAGE_FILES` when needed. `STAGE_0_to_1/` acts as the hand-off buffer once files clear validation.
2. **Stage 1 – Domain-specific processing** (`MASTER/STAGE_1`):
   - `EVENT_DATA/STEP_1..3/TASK_*` convert RAW ASCII into cleaned LIST files, pressure-corrected streams, and joined ACC outputs.
   - `LAB_LOGS/` harmonizes laboratory notebooks and housekeeping tables in two passes (raw ingestion + cleaning).
   - `COPERNICUS/` fetches ERA5 products via `.cdsapirc` credentials and formats them into per-station slices.
3. **Stage 2 – Corrections & integration** (`MASTER/STAGE_2/stage2_corrector.py`, `STATIONS/<ID>/STAGE_2`): merges detector, lab, and Copernicus tables, applies pressure/temperature corrections, and emits the `large_corrected_table.csv` that feeds Grafana as well as quicklook plots under `QUICKLOOK_OUTPUTS/`.
4. **Stage 3 – External context & analytics** (`MASTER/STAGE_3`): `nmdb_retrieval.sh` refreshes NMDB combined neutron monitor data, `rolling_pressure.py` prepares contextual pressure series, and `fine_analysis.py` aligns NMDB traces with each station's corrected table to create `STAGE_3/third_stage_table.csv` along with diagnostic figures.

Ancillary daemons (`MASTER/ANCILLARY/PIPELINE_REAL_TIME_CHECK`, `PIPELINE_OPERATIONS/UPDATE_EXECUTION_CSVS`, Telegram bot watchdogs) continuously inspect file counts, update execution CSV trackers, and push alerts when any stage stalls.

## Key capabilities

- **Zero stage (staging & unpacking)** – nightly rsync of detector archives, checksum/retimestamp scripts, and orderly fan-out into per-station working directories.
- **First stage (per-source processing)** – transformation of raw detector ASCII into filtered LIST/ACC files, cleaning and aggregation of lab logs, and batched download/formatting of Copernicus data.
- **Second stage (integration & correction)** – pressure/temperature corrections plus a wide merge that yields the "large table" consumed by Grafana dashboards and downstream science notebooks.
- **Third stage (context & analytics)** – merges NMDB neutron monitor references, rolling atmospheric baselines, and machine-learned trends to publish refined tables and figures under each station’s future `STAGE_3` directory.
- **Operations tooling** – helper scripts for cron scheduling, tmux session layout, and purging temporary products when disk pressure rises.

The existing implementation favors explicit directory choreography over workflow managers so that operators can reason about every intermediate artefact. While opinionated, the repository is designed to be reproducible when cloned onto a fresh host with the expected directory layout.

## Deployment footprint

Stations currently connected to the network:
- **MINGO01 – Madrid, Spain**
- **MINGO02 – Warsaw, Poland**
- **MINGO03 – Puebla, Mexico**
- **MINGO04 – Monterrey, Mexico**

Each station mirrors the same directory tree under `STATIONS/<ID>/` so that scripts can operate identically regardless of location. Most paths default to `/home/mingo/DATAFLOW_v3`, although this can be adapted by editing the environment variables at the top of the shell entrypoints.

## Repository layout

```
MASTER/
├── ANCILLARY/               # Cleaners, monitors, Telegram bot, plotters, simulators
├── common/                  # Shared utilities (config loader, execution logger, status dashboards)
├── CONFIG_FILES/            # Station-agnostic configs (rsync manifests, Copernicus credentials)
├── STAGE_0/
│   ├── NEW_FILES/           # bring_data_and_config_files.sh + supporting configs
│   └── REPROCESSING/        # Manual replays (STEP_0..2, UNPACKER_ZERO_STAGE_FILES)
├── STAGE_1/
│   ├── EVENT_DATA/             # RAW-->LIST-->ACC converters (STEP_1..3, TASK_*)
│   ├── LAB_LOGS/               # Logbook ingestion and cleaning scripts
│   └── COPERNICUS/             # ERA5 download and wrangling utilities
├── STAGE_2/               # Corrections + unified table builder
├── STAGE_3/               # NMDB retrieval, rolling pressure, fine analysis
├── STATIONS/              # Skeleton station tree used for bootstrapping
└── OUTPUT/, OLD_STAGE_2/  # Legacy exports and transitional artefacts

STATIONS/<ID>/
├── STAGE_0/                 # Station-local buffers (ASCII, HLDS, etc.)
├── STAGE_0_to_1/            # Files cleared from staging and ready for stage 1
├── STAGE_1/                 # Mirrors MASTER logic for per-station runs
└── STAGE_2/                 # Outputs ready for Grafana + quicklook plots

GRAFANA_DATA/                   # Published tables and dashboard assets
grafana/                        # Grafana provisioning (datasources, dashboards)
EXECUTION_LOGS/                 # Cron, nohup, and lock files for observability
FOR_MINGO_SYSTEMS/              # Shell/tmux bootstrap snippets deployed to stations
```

Use `top_large_dirs.sh` to inspect disk usage and the `clean_*.sh` utilities to prune transient files when needed.

## Getting started

### Requirements

- Linux host with passwordless SSH access to each miniTRASGO station (`mingo0X`).
- Python 3.9+ with the scientific stack listed in `requirements.list` (install via `pip install -r requirements.list`).
- Copernicus Climate Data Store account and configured `~/DATAFLOW_v3/MASTER/CONFIG_FILES/COPERNICUS/.cdsapirc` credentials for ERA5 downloads.
- Cron and tmux available on the processing node.

### Initial setup

1. **Clone the repository** onto the processing host and ensure the root matches the paths expected by the scripts (default `/home/mingo/DATAFLOW_v3`).
2. **Populate SSH config** entries for each station so that `ssh mingo0X` resolves without passwords. Test `rsync` connectivity before scheduling automated jobs.
3. **Install Python dependencies** within the environment used for the first and second stage scripts.
4. **Review configuration headers** inside the shell/Python entrypoints to adjust station IDs, root paths, or retention windows for your deployment.

### Operating the pipeline

1. Load the tmux layout from `add_to_tmux.info` (e.g., `tmux source-file add_to_tmux.info`) to prepare named panes for each stage.
2. Append the contents of `add_to_crontab.info` to the service user's crontab to trigger staging, processing, and integration jobs on schedule.
3. Inspect logs in each tmux pane or under the station directories to verify progress. Several scripts emit bannered stdout instead of structured logs, so saving the tmux history is recommended.
4. Use the helper scripts in `MASTER/STAGE_0/` for ad-hoc reprocessing when backfilling historical data or replaying failed days.

### Outputs

- **Unified CSV/Parquet tables** under `GRAFANA_DATA/` or `MASTER/STAGE_2/` for visualization and archival analysis.
- **Diagnostic plots** generated during the RAW-->LIST transformation, saved under each station’s `STAGE_1/EVENT_DATA/PLOTS/` subtree.
- **Intermediate artefacts** for auditability (raw lab logs, cleaned aggregates, Copernicus NetCDF downloads) maintained per station.

## Contributing

Improvements are welcome. Please open an issue or draft pull request describing the motivation, expected data impact, and testing performed. Given the operational nature of this codebase, coordinating changes with station operators is encouraged before merging.

## Support

For questions about deployment or data usage, contact the miniTRASGO operations team via `csoneira@ucm.es`.
