# Unified Dataflow for miniTRASGO Cosmic Ray Network

## Overview
This repository hosts the pipeline used to ingest, process, and consolidate cosmic ray observations collected by the **miniTRASGO** network. It includes the bash and Python tooling that currently runs in production for four stations and the orchestration snippets (cron, tmux) needed to keep the jobs alive on-premise.

The project brings together:
- **Detector event streams** from each miniTRASGO station (miniTRASGO01–04).
- **Laboratory logbooks** produced by operators and local environmental probes.
- **Copernicus ERA5 reanalysis** products, used for atmospheric corrections.

## Key capabilities

- **Zero stage (staging & unpacking)** – nightly rsync of detector archives, checksum/retimestamp scripts, and orderly fan-out into per-station working directories.
- **First stage (per-source processing)** – transformation of raw detector ASCII into filtered LIST/ACC files, cleaning and aggregation of lab logs, and batched download/formatting of Copernicus data.
- **Second stage (integration & correction)** – pressure/temperature corrections plus a wide merge that yields the "large table" consumed by Grafana dashboards and downstream science notebooks.
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
├── ZERO_STAGE/                 # Unpacking, deduplication, housekeeping
├── FIRST_STAGE/
│   ├── EVENT_DATA/             # RAW→LIST→ACC converters and their helpers
│   ├── LAB_LOGS/               # Logbook ingestion and cleaning scripts
│   └── COPERNICUS/             # ERA5 download and wrangling utilities
└── SECOND_STAGE/               # Corrections + unified table builder

STATIONS/<ID>/
├── ZERO_STAGE/                 # Station-local buffers (ASCII, HLDS, etc.)
├── FIRST_STAGE/                # Mirrors MASTER logic for per-station runs
└── SECOND_STAGE/               # Outputs ready for Grafana and archival use

GRAFANA_DATA/                   # Published tables and dashboard assets
TESTS/                          # Sample inputs and regression notebooks
```

Use `top_large_dirs.sh` to inspect disk usage and the `clean_*.sh` utilities to prune transient files when needed.

## Getting started

### Requirements

- Linux host with passwordless SSH access to each miniTRASGO station (`mingo0X`).
- Python 3.9+ with the scientific stack listed in `requirements.list` (install via `pip install -r requirements.list`).
- Copernicus Climate Data Store account and configured `~/.cdsapirc` credentials for ERA5 downloads.
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
4. Use the helper scripts in `MASTER/ZERO_STAGE/` for ad-hoc reprocessing when backfilling historical data or replaying failed days.

### Outputs

- **Unified CSV/Parquet tables** under `GRAFANA_DATA/` or `MASTER/SECOND_STAGE/` for visualization and archival analysis.
- **Diagnostic plots** generated during the RAW→LIST transformation, saved under each station’s `FIRST_STAGE/EVENT_DATA/PLOTS/` subtree.
- **Intermediate artefacts** for auditability (raw lab logs, cleaned aggregates, Copernicus NetCDF downloads) maintained per station.

## Contributing

Improvements are welcome. Please open an issue or draft pull request describing the motivation, expected data impact, and testing performed. Given the operational nature of this codebase, coordinating changes with station operators is encouraged before merging.

## Support

For questions about deployment or data usage, contact the miniTRASGO operations team via `csoneira@ucm.es`.
