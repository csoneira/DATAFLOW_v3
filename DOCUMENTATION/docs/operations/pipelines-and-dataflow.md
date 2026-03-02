# Pipelines and Dataflow

## End-to-end dataflow

DATAFLOW_v3 combines two upstream sources that converge into analysis workflows:

1. **Real station data** from hardware DAQ and slow-control logs
2. **Simulated station-like data** from the digital twin

## Real-data path

```text
Station DAQ (.hld/.dat, logs)
  -> MASTER/STAGE_0 ingestion/buffering
  -> STAGE_1 cleaning + alignment
  -> STAGE_2 corrections + merges
  -> STAGE_3 enriched analytics outputs
```

## Simulated-data path

```text
STEP_0 param mesh
  -> STEP_1..STEP_10 simulation chain
  -> STEP_FINAL .dat emission
  -> STAGE_0 simulation ingestion
  -> same downstream operational stages
```

## Data products and directories

| Product | Typical location |
| --- | --- |
| Simulation intersteps | `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_N_TO_N+1/` |
| Simulated `.dat` files | `MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES/` |
| Runtime cron logs | `OPERATIONS_RUNTIME/CRON_LOGS/` |
| Runtime locks | `OPERATIONS_RUNTIME/LOCKS/` |
| Operational station trees | `STATIONS/MINGO0X/` |

## Key formats

- `.hld`: raw DAQ payloads
- `.dat`: station-style event format (used by simulation and ingestion workflows)
- `.csv`/`.pkl`/chunk manifests: simulation intermediate data products
- `.meta.json`: sidecar metadata and lineage descriptors

## Data quality and provenance checkpoints

- Confirm queue movement between stages.
- Confirm logs advance at expected cadence.
- Confirm simulation hash/lineage integrity for generated `.dat`.
- Confirm no silent fallback behavior in correction/inference paths.

