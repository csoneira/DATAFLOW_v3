# OPERATIONS

Centralized operational tooling for repository-wide and pipeline-wide concerns.

## Scope

Use this area for scripts/services that operate across multiple domains of the repo,
not logic tied to a single pipeline stage under `MASTER/`, `STATIONS/`, or simulation-only code.

## Structure

- `OPERATIONS/ORCHESTRATION/`
  - Runtime control, concurrency guards, and process orchestration.
- `OPERATIONS/OBSERVABILITY/`
  - Audits, error scanning, and operational visibility/reporting.
- `OPERATIONS/DATA_MAINTENANCE/`
  - Cross-pipeline data/index maintenance workflows.
- `OPERATIONS/MAINTENANCE/`
  - Generic system/repository housekeeping.
- `OPERATIONS/NOTIFICATIONS/`
  - Messaging and alerting integrations.

## Placement Rules

- Put a tool here only if it impacts multiple layers/components of the repository.
- Keep stage-specific scripts inside their owning stage directory.
- Prefer explicit absolute paths in cron-facing wrappers.
- Keep logs under `OPERATIONS_RUNTIME/` and lock files under `OPERATIONS_RUNTIME/LOCKS/`.
