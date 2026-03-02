# Contracts and Interfaces

## Simulation contracts

Primary contract document:

- <https://github.com/csoneira/DATAFLOW_v3/blob/main/MINGO_DIGITAL_TWIN/DOCS/contracts/STEP_CONTRACTS.md>

It defines for each step:

- required inputs
- output schema expectations
- failure modes
- metadata expectations

## Operational stage interfaces

Operational stages are split as STAGE_0..STAGE_3, implemented in `MASTER/` (mother code), with handoffs and materialized outputs in station trees under `STATIONS/`.

Key interface assumptions:

- Stage queue/metadata ownership is explicit per stage.
- Reprocessing and skip lists are tracked in station metadata CSVs.
- Ingestion of simulated `.dat` follows station-compatible conventions.

## Behavior change policy

Any interface change should include:

1. code change
2. updated docs/contracts
3. validation evidence

See standards:
- [Conventions and Governance](../standards/conventions-and-governance.md)
