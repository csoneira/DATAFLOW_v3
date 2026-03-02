# DATAFLOW_v3 Documentation

DATAFLOW_v3 contains two coordinated systems:

1. The **operational pipeline** (`MASTER/`, `STATIONS/`) for real station data ingestion, correction, and analytics.
2. The **MINGO digital twin** (`MINGO_DIGITAL_TWIN/`) for synthetic RPC event generation from STEP_0 to STEP_FINAL.

This documentation is organized by subsystem and role so collaborators can move from onboarding to operations and troubleshooting without hunting across folders.

## Documentation map

- [Getting Started](getting-started/index.md): environment, dependencies, and common commands.
- [Collaborators](collaborators/index.md): team roles, institutions, and contact points.
- [Software](software/index.md): architecture, workflows, and code structure for analysis, simulation, and inference.
- [Hardware](hardware/index.md): detector stations, DAQ, infrastructure, and maintenance practices.
- [Operational Notes](operations/index.md): cron behavior, dataflow, maintenance scripts, and runtime checks.
- [Troubleshooting and FAQs](troubleshooting/index.md): recurring failure patterns and operator Q&A.
- [Conventions and Standards](standards/index.md): governance rules, reproducibility, determinism, and naming/config policies.
- [Publications and References](references/index.md): papers, reports, and source documentation pointers.
- [Appendices](appendices/index.md): dictionaries, contracts, glossary, and full contact list.
- [Legacy miniTRASGO Pages](legacy/index.md): archived context pages retained for continuity.

## Quick orientation

- New developer: start at [Getting Started](getting-started/index.md), then [Software](software/index.md).
- Operator: start at [Hardware](hardware/index.md), then [Operational Notes](operations/index.md).
- Analyst: start at [Software](software/index.md), [Appendices](appendices/data-dictionaries.md), and [References](references/publications-and-reports.md).
- Documentation maintainer: start at [Conventions and Standards](standards/index.md) and [Documentation Lifecycle](standards/documentation-lifecycle.md).

## Canonical source documents in this repository

The pages in this MkDocs site summarize and cross-link these maintained sources:

- `README.md` (repository overview)
- `DOCS/REPO_DOCS/REPOSITORY_GOVERNANCE.md`
- `DOCS/BEHAVIOUR/CRON_AND_SCHEDULING.md`
- `DOCS/REPO_DOCS/TROUBLESHOOTING/OPERATIONS_RUNBOOK.md`
- `MINGO_DIGITAL_TWIN/DOCS/README.md`
- `MINGO_DIGITAL_TWIN/DOCS/contracts/STEP_CONTRACTS.md`
