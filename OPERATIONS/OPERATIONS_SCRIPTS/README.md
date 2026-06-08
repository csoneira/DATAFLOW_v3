# OPERATIONS

Centralized operational tooling for repository-wide and pipeline-wide concerns.

## Scope

Use this area for scripts/services that operate across multiple domains of the repo,
not logic tied to a single pipeline stage under `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/`, `MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/`, or simulation-only code.

## Structure

- `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/`
  - Runtime control, concurrency guards, and process orchestration.
- `OPERATIONS/OPERATIONS_SCRIPTS/OBSERVABILITY/`
  - Audits, error scanning, and operational visibility/reporting.
- `OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/`
  - Cross-pipeline data/index maintenance workflows.
- `OPERATIONS/OPERATIONS_SCRIPTS/MAINTENANCE/`
  - Generic system/repository housekeeping.
- `OPERATIONS/OPERATIONS_SCRIPTS/NOTIFICATIONS/`
  - Messaging and alerting integrations.

## Placement Rules

- Put a tool here only if it impacts multiple layers/components of the repository.
- Keep stage-specific scripts inside their owning stage directory.
- Prefer explicit absolute paths in cron-facing wrappers.
- Keep logs under `OPERATIONS/OPERATIONS_RUNTIME/` and lock files under `OPERATIONS/OPERATIONS_RUNTIME/LOCKS/`.
