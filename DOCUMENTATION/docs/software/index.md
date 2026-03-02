# Software

## Pillar map

| Pillar | Owns | Main inputs | Main outputs | Primary doc |
| --- | --- | --- | --- | --- |
| Analysis | `MASTER`, `STATIONS` | Real station data + simulated `.dat` | Station-level corrected/enriched outputs | [Analysis (Software)](operational-pipeline.md) |
| Simulation | `MINGO_DIGITAL_TWIN` | Physics config + runtime config | STEP_FINAL station-style `.dat` + lineage | [Simulation (Digital Twin)](simulation-pipeline.md) |
| Dictionary-based inference | `MINGO_DICTIONARY_CREATION_AND_TEST`, `MASTER/common` | Simulated truth-linked samples + measured rates | Reconstruction artifacts and estimates | [Dictionary-Based Inference](inference-and-dictionary.md) |

## Read in this order

1. [5-Minute System Model](system-model.md)
2. [Software Invariants](invariants.md)
3. [Real Data Trace](trace-real-data.md)
4. [Simulated Data Trace](trace-simulated-data.md)
5. [Change Impact Matrix](change-impact-matrix.md)

## Deep references

- [Architecture Overview](architecture.md)
- [Code Structure Reference](code-structure.md)
