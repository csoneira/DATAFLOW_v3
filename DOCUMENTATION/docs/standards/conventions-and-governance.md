# Conventions and Governance

## Mandatory engineering principles

1. Reproducibility from committed code + tracked configuration.
2. Configuration-driven behavior (no hidden hardcoded policy).
3. Data safety (never modify/delete raw source data in routine flows).
4. Explicit provenance for merged or transformed outputs.
5. Operational compatibility with existing lock/scheduling semantics.

## Naming and structure conventions

- Use `STEP_N` naming consistently in simulation docs and code comments.
- Keep per-step configuration under the corresponding `MASTER_STEPS/STEP_N/` path.
- Keep outputs in designated interstep/stage output trees.

## Configuration policy

- Use existing YAML/CSV/CONF configuration files.
- Avoid station-specific absolute paths in source code.
- Document any new config keys and defaults in docs that match implementation.

## CLI behavior patterns

For long-running scripts, follow the standardized CLI help/verbose pattern:

- `-h/--help`
- station argument conventions where relevant
- `-v/--verbose` plus environment fallback

Reference pattern:
- <https://github.com/csoneira/DATAFLOW_v3/blob/main/DOCS/PATTERNS/CLI_HELP_VERBOSE_PATTERN.md>

## Documentation standards

- Update docs in same PR for behavior-impacting code changes.
- Keep runbooks date-stamped with symptom, root cause, fix, and verification.
- Prefer concise but complete technical prose.

