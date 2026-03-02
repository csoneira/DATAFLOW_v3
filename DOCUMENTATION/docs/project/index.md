# Project Dossier (Grant-Oriented)

DATAFLOW_v3 is the software backbone of a distributed cosmic-ray detector effort combining detector operations, simulation, and reconstruction in a single auditable platform.

## Project objective

Deliver a collaboration-grade analysis framework that:

1. Processes real station data with reproducible, operations-safe pipelines.
2. Produces simulation data with explicit physics/electronics lineage.
3. Reconstructs physical observables through dictionary-based inference where real and simulated domains meet.

## Why this project is technically strong

- Unified architecture: one analysis mother code (`MASTER`) for real and simulated data.
- Traceable simulation chain (`STEP_0` to `STEP_FINAL`) with metadata and hashing.
- Explicit bridge from simulation to reconstruction (dictionary-based inference).
- Operational discipline: scheduling, locking, observability, and runbook-driven recovery.

## High-level system view

![Dual-pipeline architecture](/assets/figure_dual_pipeline_architecture.svg)

## Collaboration footprint

![Collaboration network map](/assets/repository_figures/network_map_attic.png)

## Dossier contents

- [Scientific Case](scientific-case.md)
- [Work Packages](work-packages.md)
- [Governance and Sites](governance-and-sites.md)
- [Milestones, Deliverables, and Risk](milestones-deliverables-risk.md)

## Technical anchors

- [5-Minute System Model](../software/system-model.md)
- [Software Invariants](../software/invariants.md)
- [Change Impact Matrix](../software/change-impact-matrix.md)

