---
title: Project Backlog
description: Consolidated roadmap and task backlog for DATAFLOW_v3.
last_updated: 2026-02-24
status: active
supersedes:
  - to_do.md
  - potential_improvements.md
---

# Project Backlog

## Table of contents
- [Current priorities](#current-priorities)
- [Scientific validation priorities](#scientific-validation-priorities)
- [Engineering roadmap](#engineering-roadmap)
- [Completed highlights](#completed-highlights)
- [Stale or underspecified items](#stale-or-underspecified-items)

## Current priorities

### P0: stabilize operations
- [ ] Verify simulation cron logs remain reliable under cleanup jobs (no deleted active file handles).
- [ ] Keep `guide_raw_to_corrected` and Copernicus jobs singleton-safe via `flock` and periodic process-count checks.
- [ ] Maintain hash consistency in `MINGO_DIGITAL_TWIN/SIMULATED_DATA/step_final_simulation_params.csv`.

### P0: simulation-to-analysis consistency
- [ ] Validate that STEP_1 filters on simulated data retain near-100% expected purity for `MINGO00` noiseless runs.
- [ ] Confirm `param_mesh` upstream coverage for pending rows to avoid STEP_2/STEP_3 starvation.

### P1: inference workflow execution
- [ ] Run inference method on a bounded real-data window (candidate: HV scan period) and compare against simulation-derived expectations.
- [ ] Produce first uncertainty-aware correction report for a short real-data interval.

## Scientific validation priorities
- [ ] Validate simulation pipeline against physics expectations (rate, efficiency trends, geometry sensitivity).
- [ ] Validate inverse-method robustness: bias, resolution, and uncertainty calibration.
- [ ] Cross-check with independent datasets or external detector references where available.

## Engineering roadmap

### Configuration and orchestration
- [ ] Continue decomposition of monolithic configs into step-level runtime + physics configs.
- [ ] Keep orchestration behavior explicitly documented when new gates/locks are introduced.

### Testing and observability
- [ ] Add targeted unit tests for critical helpers (hash normalization, param-mesh consistency checks, lock and scheduler decisions).
- [ ] Expand integration checks for full chain (`STEP_0 -> STEP_FINAL -> ingest`).
- [ ] Keep `OPERATIONS/OBSERVABILITY/AUDIT_PIPELINE_STATES` outputs in regular operational checks.

### Code quality and maintainability
- [ ] Refactor high-churn scripts into shared utilities only when behavior remains unchanged.
- [ ] Standardize structured logging across long-running scripts.
- [ ] Keep CLI behavior consistent with [CLI help + verbose pattern](../PATTERNS/CLI_HELP_VERBOSE_PATTERN.md).

## Completed highlights
- [x] STEP_1 task scripts now expose consistent CLI `--help`/`--verbose` behavior.
- [x] Definitive execution plotting pipeline added and scheduled.
- [x] Cron-level locking added for major high-frequency jobs to prevent process explosions.
- [x] Hash mismatch and orphan-row handling was hardened in simulation output tracking.
- [x] STEP_1 metadata status/progress tracking implemented.

## Stale or underspecified items
These were carried from earlier notes and need explicit triage before execution:
- [ ] GPU feasibility study for STEP_1 scripts (CuPy migration): scope, benchmarks, and acceptance criteria are not yet defined.
- [ ] "Relax filters until almost all simulated events pass": needs formal quality constraints to avoid invalidating physics selection logic.
- [ ] "General modernization (CI/CD, Docker, large modularization)": valid long-term direction but currently broad; break into concrete milestones before implementation.
