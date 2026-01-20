# DATAFLOW_v3

## Overview
DATAFLOW_v3 hosts two connected systems:
1. The production miniTRASGO dataflow pipeline under `MASTER/` and `STATIONS/`, used to ingest, clean, and consolidate station data.
2. The MINGO digital twin simulator under `MINGO_DIGITAL_TWIN/`, used to generate synthetic RPC detector data from muon generation through DAQ formatting.

This repository keeps the operational pipeline and the simulator side-by-side so the same geometry and timing assumptions can be tested, validated, and tuned.

## Repository layout
- `MASTER/`: operational pipeline scripts, configs, and shared helpers
- `STATIONS/`: station-specific stage trees and outputs
- `FOR_MINGO_SYSTEMS/`: host bootstrap templates (cron and tmux snippets)
- `EXECUTION_LOGS/`: cron, nohup, and lock files
- `MINGO_DIGITAL_TWIN/`: simulation pipeline, tools, and docs
- `NOT_ESSENTIAL/`: archived or non-critical assets

## Operational pipeline (miniTRASGO)
The production pipeline ingests detector streams, cleans and merges lab logs, integrates Copernicus ERA5 context, and builds unified tables for monitoring and analysis.

Stages:
- Stage 0 (acquisition and buffering): rsync detector archives into per-station buffers.
- Stage 1 (domain processing): convert raw ASCII into cleaned event lists and align lab logs.
- Stage 2 (corrections and integration): apply pressure/temperature corrections and merge sources.
- Stage 3 (external context and analytics): incorporate NMDB data and build enriched tables.

Key locations:
- `MASTER/STAGE_0/`, `MASTER/STAGE_1/`, `MASTER/STAGE_2/`, `MASTER/STAGE_3/`
- `MASTER/ANCILLARY/`: monitors, bots, plotters, and utility scripts
- `FOR_MINGO_SYSTEMS/`: tmux and cron templates for production deployment

## Simulation pipeline (MINGO digital twin)
The digital twin models RPC detector response and electronics in a stepwise chain from muon generation to station-style outputs.

Highlights:
- Per-step outputs in `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_X_TO_Y/SIM_RUN_<N>`
- Metadata and sim-run hashing for reproducibility
- Persistent `event_id` for step alignment and debugging

For the full simulator documentation, including architecture, boundaries, and step-by-step details, see:
- `MINGO_DIGITAL_TWIN/README.md`

Key simulator docs:
- `MINGO_DIGITAL_TWIN/DOCS/contracts/`
- `MINGO_DIGITAL_TWIN/DOCS/coordinate_and_timing_conventions.md`

## Deployment and operations
- Operational scripts expect paths rooted at `/home/mingo/DATAFLOW_v3` by default.
- Cron and tmux templates live under `FOR_MINGO_SYSTEMS/` and are referenced by `add_to_crontab.info` and `add_to_tmux.info` in the repo root.

## Getting started
- Install Python dependencies from `requirements.list`.
- Review configuration headers in `MASTER/` scripts for station IDs and root paths.
- For simulation, follow the instructions in `MINGO_DIGITAL_TWIN/README.md`.

If you want a tighter deployment guide or a simplified quickstart for either system, say the word and I will add it.
