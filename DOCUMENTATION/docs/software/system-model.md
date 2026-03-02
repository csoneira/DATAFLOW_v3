# 5-Minute System Model

This page is the fastest accurate mental model of DATAFLOW_v3 software.

## Three pillars

1. **Analysis software** (`MASTER` + `STATIONS`)
`MASTER` is the mother analysis code. `STATIONS` is where station-scoped runtime state and outputs are materialized.

2. **Simulation digital twin** (`MINGO_DIGITAL_TWIN`)
Deterministic multi-step RPC simulation producing station-style `.dat` files.

3. **Dictionary-based inference** (`MINGO_DICTIONARY_CREATION_AND_TEST` + `MASTER/common`)
Reconstruction layer where synthetic knowledge and measured data meet.

## System relationship

```mermaid
flowchart LR
    subgraph Analysis[Analysis Software]
        M[MASTER stages 0..3]
        S[STATIONS outputs]
        M --> S
    end

    subgraph Twin[Simulation Digital Twin]
        T0[STEP_0..STEP_10]
        TF[STEP_FINAL .dat]
        T0 --> TF
    end

    subgraph Inference[Dictionary-Based Inference]
        D1[Build dictionary from simulation]
        D2[Use dictionary during analysis]
        D1 --> D2
    end

    R[Real station inputs] --> M
    TF --> M
    T0 --> D1
    M --> D2
```

## What must be true

- Analysis behavior lives in `MASTER`, not scattered ad hoc in station folders.
- Real and simulated inputs are both processed through the same analysis mother code.
- Inference is versioned and traceable to simulation assumptions.
- Outputs are auditable at station scope under `STATIONS`.

## If you only read three pages next

- [Software Invariants](invariants.md)
- [Real Data Trace](trace-real-data.md)
- [Simulated Data Trace](trace-simulated-data.md)

