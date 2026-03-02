# Determinism and Provenance

## Determinism policy

- Use explicit random seeds where deterministic replay is required.
- Document intentional non-determinism at step level.
- Avoid hidden randomness in selection/ordering unless explicitly justified.

## Provenance requirements

Generated artifacts should preserve enough context to reconstruct origin:

- configuration snapshot and/or hash
- upstream lineage hash
- step IDs and run identifiers
- mesh row identifiers where applicable

## Simulation-specific checks

- Validate hash integrity with maintenance tooling.
- Keep `step_final_simulation_params.csv` aligned with generated files.
- Treat orphaned or hash-mismatched artifacts as operational debt, not normal state.

## Operational-specific checks

- Track queue membership and processed basenames to avoid accidental duplication.
- Keep cron and lock semantics unchanged unless explicitly reviewed.
- Validate stage boundaries before any replay/reprocessing action.

## Acceptance criteria for behavior changes

A behavior change is not complete until:

1. Validation was executed (test/script/manual reproducible check).
2. Failure modes and residual risk are documented.
3. Related runbook and standards pages are updated.

