# Software

This section is intentionally structured as a collaboration-grade technical map, not a knowledge dump.

## The three software pillars

1. **Analysis (software)**
`MASTER` is the mother analysis code for both real and simulated inputs. `STATIONS` is where outputs and runtime state are materialized.

2. **Simulation (digital twin)**
`MINGO_DIGITAL_TWIN` generates synthetic station-compatible data with explicit step contracts and lineage.

3. **Dictionary-based inference (reconstruction)**
`MINGO_DICTIONARY_CREATION_AND_TEST` + `MASTER/common` form the reconstruction layer where real data and simulated knowledge meet.

## Start here (in order)

- [5-Minute System Model](system-model.md)
- [Software Invariants](invariants.md)
- [Analysis Software](operational-pipeline.md)
- [Simulation Digital Twin](simulation-pipeline.md)
- [Dictionary-Based Inference](inference-and-dictionary.md)

## Operationally critical traces

- [Real Data Trace](trace-real-data.md)
- [Simulated Data Trace](trace-simulated-data.md)
- [Change Impact Matrix](change-impact-matrix.md)

## Reference pages

- [Architecture Overview](architecture.md)
- [Code Structure Reference](code-structure.md)

