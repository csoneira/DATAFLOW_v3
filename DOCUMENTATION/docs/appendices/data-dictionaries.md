# Data Dictionaries

This appendix summarizes key schemas used across operational and simulation workflows.

## Simulation outputs

Common simulation lineage fields:

- `event_id`
- `step_1_id` ... `step_10_id`
- `param_row_id`, `param_set_id`, `param_date`
- `config_hash`, `upstream_hash`

Common physics/electronics fields by stage include:

- Generation: `X_gen`, `Y_gen`, `Z_gen`, `Theta_gen`, `Phi_gen`
- Crossings: `X_gen_i`, `Y_gen_i`, `T_sum_i_ns`, `tt_crossing`
- Avalanche: `avalanche_*`, `tt_avalanche`
- Strip/electronics: `T_front_i_sj`, `T_back_i_sj`, `Q_front_i_sj`, `Q_back_i_sj`
- Trigger/DAQ: `tt_trigger`, `daq_jitter_ns`

## Station `.dat` payload format

STEP_FINAL writes event rows composed of:

1. Timestamp header (7 fields)
2. Payload with 64 channel values in plane/field/strip order

Reference details:
- <https://github.com/csoneira/DATAFLOW_v3/blob/main/MINGO_DIGITAL_TWIN/DOCS/OUTPUTS_METADATA_AND_VALIDATION.md>

## Operational data and logs

Typical operational artifacts:

- DAQ raw files (`.hld`)
- Derived/legacy files (`.mat`)
- Slow-control logs (`sensors_bus*`, `Flow*`, `hv*`, `rates*`)
- Stage outputs under `STATIONS/MINGO0X/`

Existing detailed pages:

- [Data Overview](../data/index.md)
- [Data Shape](../data/shape.md)
- [Dataflow](../data/dataflow.md)

