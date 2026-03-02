# Documentation Lifecycle

This page defines how DATAFLOW_v3 documentation is maintained as code and operations evolve.

## Ownership model

- Architecture and software behavior pages: contributors changing `MASTER/`, `MINGO_DIGITAL_TWIN/`, or `OPERATIONS/`.
- Hardware/operation procedures: operators and hardware maintainers.
- Governance and standards pages: repository maintainers.

## Update triggers

Update docs in the same pull request when any of these change:

1. Pipeline behavior (stage sequencing, selection logic, thresholds, defaults)
2. Output formats or metadata fields
3. Cron scheduling, lock behavior, or maintenance procedures
4. Naming conventions, configuration keys, or determinism policy

## Minimum documentation bundle per behavior change

- One updated technical page in `DOCUMENTATION/docs/`.
- One updated canonical source doc when applicable (`DOCS/` or `MINGO_DIGITAL_TWIN/DOCS/`).
- Validation notes indicating how correctness was checked.

## Review checklist

Before merge:

- Links resolve.
- Commands are runnable as written (or explicitly marked as examples).
- New assumptions are explicit.
- Any non-deterministic behavior is documented.
- Incident/recovery steps include verification criteria.

## Release and upkeep cadence

- Quick updates: with each behavior-impacting PR.
- Structured audit: monthly docs review for stale procedures and broken links.
- Incident-driven updates: immediately after root cause and fix are confirmed.

## Suggested lightweight quality gates

```bash
# mkdocs build in docs environment
mkdocs build -f DOCUMENTATION/mkdocs.yml
```

Additionally, run a local markdown-link checker (or equivalent CI job) and resolve any broken references before release.
