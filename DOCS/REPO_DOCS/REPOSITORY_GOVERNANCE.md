---
title: Repository Governance
description: Engineering rules for reproducibility, deterministic behavior, and safe maintenance.
last_updated: 2026-02-24
status: active
supersedes:
  - CODEX.md
---

# Repository Governance

## Table of contents
- [Purpose](#purpose)
- [Hard requirements](#hard-requirements)
- [Implementation practices](#implementation-practices)
- [Verification standard](#verification-standard)
- [Documentation standard](#documentation-standard)

## Purpose
This document defines mandatory engineering rules for `DATAFLOW_v3` with emphasis on scientific reproducibility, operational stability, and auditable changes.

## Hard requirements
1. Reproducibility
- Outputs must be reproducible from committed code + tracked configuration without manual hidden steps.

2. Determinism
- Randomness must be explicitly seeded when deterministic replay is required.
- Any intentionally non-deterministic component must be documented in the corresponding step docs.

3. Configuration-driven behavior
- Use existing YAML/CSV/CONF configuration files.
- Do not hardcode station-specific paths, thresholds, or physics constants in scripts.

4. Data safety
- Never modify or delete raw source data as part of normal processing.
- Derived artifacts belong in designated output/intersteps locations.

5. Provenance
- Never mix datasets from different runs unless provenance is explicit and recoverable.
- Parameter lineage (`param_hash`, step IDs, metadata sidecars) must remain intact.

6. Operational compatibility
- Preserve lock semantics, cron behavior, and log paths when modifying orchestration code.
- Any change to scheduling/locking must be documented in [Cron and Scheduling](../BEHAVIOUR/CRON_AND_SCHEDULING.md).

## Implementation practices
- Keep diffs scoped and avoid formatting-only churn.
- Prefer small refactors over large rewrites unless explicitly requested.
- For multi-file behavior changes, document assumptions and risks before implementation.
- Keep naming consistent: use `STEP_N` in docs and code comments, not mixed variants (`stepN`, `step n`).

## Verification standard
A task is considered done only if:
1. Behavior is validated by tests, script execution, or a reproducible manual check.
2. Failures are reported with probable root cause and next action.
3. No workaround suppresses errors silently.

Minimum checks for pipeline-impacting changes:
- syntax/lint where available
- one representative runtime invocation
- relevant log verification for cron-managed components

## Documentation standard
- Canonical docs live under `DOCS/`, `MASTER/DOCS/`, `MINGO_DIGITAL_TWIN/DOCS/`, and `MINGO_DICTIONARY_CREATION_AND_TEST/DOCS/`.
- Use front matter (`title`, `description`, `last_updated`, `status`) for maintained docs.
- Include cross-links to code paths and related docs.
- Incident notes must include: date, symptom, root cause, fix, verification, and remaining risk.

If a rule is found to be incomplete, add a concrete and testable bullet in this file.
