# Glossary

- **DAQ**: Data Acquisition system used at station level.
- **DCS**: Detector Control System for slow controls (HV, sensors, flow).
- **FEE**: Front-End Electronics handling strip signal conditioning.
- **SIM_RUN**: Immutable simulation run identifier chain across digital-twin steps.
- **STEP_FINAL**: Final simulation formatter that emits station-style `.dat` files.
- **Backpressure gate**: Orchestrator control that pauses new STEP_0 enqueue when downstream backlog is high.
- **Lineage hash**: Hash value used to preserve provenance across generated artifacts.
- **Resource gate**: Wrapper used to skip jobs under constrained system resources.
- **TT1/TT2**: Trigger categories used in station analysis context.
- **Streamer fraction**: Fraction of high-charge events used as a detector-health metric.

