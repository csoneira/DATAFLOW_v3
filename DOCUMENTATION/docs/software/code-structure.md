# Code Structure Reference

This page provides a navigation map for developers working across analysis, simulation, and operations code.

## Top-level structure

```text
DATAFLOW_v3/
├─ MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/                      # Mother analysis code (real + simulated input processing)
│  ├─ STAGES/
│  ├─ ANCILLARY/
│  ├─ CONFIG_FILES/
│  └─ common/
├─ MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/                    # Station-specific runtime trees and materialized outputs
├─ MINGO_DIGITAL_TWIN/          # STEP_0..STEP_FINAL simulation stack
│  ├─ MASTER_STEPS/
│  ├─ ORCHESTRATOR/
│  ├─ INTERSTEPS/
│  ├─ SIMULATED_DATA/
│  └─ DOCS/
├─ MINGO_DICTIONARY_CREATION_AND_TEST/
├─ OPERATIONS/OPERATIONS_SCRIPTS/                  # Orchestration + observability utilities
├─ OPERATIONS/OPERATIONS_RUNTIME/          # Runtime state: logs, locks, status files
├─ DOCS/                        # Governance, behavior, and runbooks
└─ DOCUMENTATION/               # MkDocs source (this site)
```

## Documentation split

- `DOCUMENTATION/docs/`: curated user/developer documentation site.
- `DOCS/`: repository governance and operational behavior docs.
- `MINGO_DIGITAL_TWIN/DOCS/`: canonical simulation architecture/method contracts.

## Change impact guidance

- Changes under `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/` can affect both real-data and simulated-data analysis processing, plus cron jobs.
- Changes under `MINGO_DIGITAL_TWIN/MASTER_STEPS/` can affect simulation outputs and dictionary quality.
- Changes under `OPERATIONS/OPERATIONS_SCRIPTS/` can affect orchestration safety and incident response.

When behavior changes, update runbooks and standards in the same PR.
