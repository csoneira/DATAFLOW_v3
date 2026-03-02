# Software Invariants

This is the non-negotiable contract for the software stack.

## Architecture invariants

1. `MASTER` is the mother analysis code for both real and simulated inputs.
2. `STATIONS` is the station-scoped runtime/output materialization tree.
3. `MINGO_DIGITAL_TWIN` produces simulation artifacts; it does not replace `MASTER` analysis stages.
4. Dictionary-based inference is the bridge layer and must remain traceable to simulation provenance.

## Data and provenance invariants

1. Raw source data is never modified/deleted by routine processing.
2. Simulation lineage (`config_hash`, step IDs, registries) must remain intact.
3. Cross-run mixing requires explicit provenance and documentation.
4. Output locations and stage ownership are explicit and stable.

## Operations invariants

1. Cron + locking semantics (`flock`, gates) are part of system correctness.
2. Overlap prevention is required for per-minute jobs.
3. Backpressure behavior is expected control, not an error condition.
4. Recovery actions must be verifiable by logs/state transitions.

## Engineering invariants

1. Behavior changes require docs updates in the same PR.
2. Configuration-driven behavior is mandatory; no hidden hardcoding.
3. Deterministic behavior and intentional non-determinism must be documented.
4. Interface-impacting changes require contract and impact-matrix updates.

## Violation response

If any invariant is violated:

1. Stop rollout.
2. Document symptom and scope.
3. Restore known-good behavior.
4. Add/repair tests and docs before resuming.

