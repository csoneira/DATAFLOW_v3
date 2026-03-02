# Code Structure Reference

This page provides a navigation map for developers working across analysis, simulation, and operations code.

## Top-level structure

```text
DATAFLOW_v3/
├─ MASTER/                      # Operational analysis stages + shared logic
│  ├─ STAGES/
│  ├─ ANCILLARY/
│  ├─ CONFIG_FILES/
│  └─ common/
├─ STATIONS/                    # Station-specific runtime trees
├─ MINGO_DIGITAL_TWIN/          # STEP_0..STEP_FINAL simulation stack
│  ├─ MASTER_STEPS/
│  ├─ ORCHESTRATOR/
│  ├─ INTERSTEPS/
│  ├─ SIMULATED_DATA/
│  └─ DOCS/
├─ MINGO_DICTIONARY_CREATION_AND_TEST/
├─ OPERATIONS/                  # Orchestration + observability utilities
├─ OPERATIONS_RUNTIME/          # Runtime state: logs, locks, status files
├─ DOCS/                        # Governance, behavior, and runbooks
└─ DOCUMENTATION/               # MkDocs source (this site)
```

## Documentation split

- `DOCUMENTATION/docs/`: curated user/developer documentation site.
- `DOCS/`: repository governance and operational behavior docs.
- `MINGO_DIGITAL_TWIN/DOCS/`: canonical simulation architecture/method contracts.

## Change impact guidance

- Changes under `MASTER/STAGES/` can affect real-data processing and cron jobs.
- Changes under `MINGO_DIGITAL_TWIN/MASTER_STEPS/` can affect simulation outputs and dictionary quality.
- Changes under `OPERATIONS/` can affect orchestration safety and incident response.

When behavior changes, update runbooks and standards in the same PR.

