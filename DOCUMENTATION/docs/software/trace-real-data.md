# Real Data Trace

This is the concrete path for real station data through the software stack.

## End-to-end trace

```mermaid
flowchart LR
    A[Station DAQ and logs] --> B[MASTER STAGE_0 ingestion]
    B --> C[MASTER STAGE_1 cleaning/alignment]
    C --> D[MASTER STAGE_2 corrections/integration]
    D --> E[MASTER STAGE_3 enrichment]
    E --> F[STATIONS materialized outputs]
```

## Primary code ownership by segment

| Segment | Owner path |
| --- | --- |
| Ingestion and queueing | `MASTER/STAGES/STAGE_0/` |
| Event/lab-log transformations | `MASTER/STAGES/STAGE_1/` |
| Corrections and merges | `MASTER/STAGES/STAGE_2/` |
| Final analytics/enrichment | `MASTER/STAGES/STAGE_3/` |
| Output/state materialization | `STATIONS/MINGO0X/...` |

## Validation checkpoints

1. STAGE logs advance at expected cadence.
2. Queue movement is visible between stage boundaries.
3. No unexplained growth of error/reject directories.
4. Output files and metadata appear in expected station locations.

## Common failure boundaries

- Input acquisition/ingestion mismatch in STAGE_0.
- Partial transform completion in STAGE_1.
- Correction-source mismatch in STAGE_2.
- Final publication/enrichment gaps in STAGE_3.

For recovery commands and sequencing, use:
- [Operational Notes](../operations/index.md)
- [Troubleshooting](../troubleshooting/common-issues.md)

