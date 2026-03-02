# Software

DATAFLOW_v3 software is organized around three domains:

1. **Analysis mother code and station outputs**: `MASTER/` provides the analysis logic for both real and simulated inputs; `STATIONS/` stores station-scoped runtime/output trees.
2. **Digital twin simulation** (synthetic detector data): `MINGO_DIGITAL_TWIN/`
3. **Dictionary and inference tooling** (flux/efficiency estimation): `MINGO_DICTIONARY_CREATION_AND_TEST/` plus `MASTER/common`

## Read this section in order

- [Architecture Overview](architecture.md)
- [Operational Pipeline](operational-pipeline.md)
- [Simulation Pipeline (Digital Twin)](simulation-pipeline.md)
- [Inference and Dictionary Workflow](inference-and-dictionary.md)
- [Code Structure Reference](code-structure.md)

The architecture and dataflow figures used across this section are designed for quick onboarding before diving into step-level contracts.

## Cross-system principle

The `MASTER` analysis code and the digital twin intentionally share geometry, timing conventions, and data-format assumptions so synthetic outputs can be injected and processed through the same analysis logic as real station data.
