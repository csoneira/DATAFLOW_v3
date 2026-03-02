# Software

DATAFLOW_v3 software is organized around three domains:

1. **Operational analysis pipeline** (real detector data): `MASTER/`, `STATIONS/`
2. **Digital twin simulation** (synthetic detector data): `MINGO_DIGITAL_TWIN/`
3. **Dictionary and inference tooling** (flux/efficiency estimation): `MINGO_DICTIONARY_CREATION_AND_TEST/` plus `MASTER/common`

## Read this section in order

- [Architecture Overview](architecture.md)
- [Operational Pipeline](operational-pipeline.md)
- [Simulation Pipeline (Digital Twin)](simulation-pipeline.md)
- [Inference and Dictionary Workflow](inference-and-dictionary.md)
- [Code Structure Reference](code-structure.md)

## Cross-system principle

The operational pipeline and the digital twin intentionally share geometry, timing conventions, and data-format assumptions so synthetic outputs can be injected into analysis and compared with real station behavior.

