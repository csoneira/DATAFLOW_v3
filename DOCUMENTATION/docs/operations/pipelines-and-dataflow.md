# Pipelines and Dataflow

## End-to-end dataflow

DATAFLOW_v3 combines two upstream sources that converge into analysis workflows:

1. **Real station data** from hardware DAQ and slow-control logs
2. **Simulated station-like data** from the digital twin

![Dataflow convergence](/assets/figures/architecture/figure_dataflow_convergence.svg)

*Figure 2. Real and simulated upstreams converge in STAGE_0 and share downstream operational stages.*

## Path summary

- Real data: station DAQ/log sources -> `MASTER` STAGE_0 -> STAGE_1 -> STAGE_2 -> STAGE_3 -> outputs in `STATIONS/`
- Simulated data: STEP_0 -> STEP_1..STEP_10 -> STEP_FINAL `.dat` -> `MASTER` STAGE_0 -> STAGE_1..STAGE_3 -> outputs in `STATIONS/`

## Data products and directories

| Product | Typical location |
| --- | --- |
| Simulation intersteps | `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_N_TO_N+1/` |
| Simulated `.dat` files | `MINGO_DIGITAL_TWIN/SIMULATED_DATA/FILES/` |
| Runtime cron logs | `OPERATIONS_RUNTIME/CRON_LOGS/` |
| Runtime locks | `OPERATIONS_RUNTIME/LOCKS/` |
| Operational station trees | `STATIONS/MINGO0X/` |
| Materialized analysis outputs | `STATIONS/MINGO0X/STAGE_*/...` |

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
