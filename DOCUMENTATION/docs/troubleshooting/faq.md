# FAQ

## What is the difference between the operational pipeline and the digital twin?

The operational pipeline processes real station data. The digital twin creates synthetic detector data with stepwise physics/electronics modeling. Both are designed to share assumptions and formats where possible.

## Where should I look first during an incident?

Start with cron status, process presence, and the latest logs in `OPERATIONS_RUNTIME/CRON_LOGS/`, then move to queue/consistency checks.

## Can I rerun a simulation step safely?

Yes, but follow SIM_RUN immutability rules. Use `--force` only when overwrite is intentional and documented.

## How do I verify simulation provenance?

Check metadata sidecars/manifests, step IDs, and hash tools (for example `ensure_sim_hashes.py`).

## When should dictionary artifacts be regenerated?

Whenever simulation assumptions, parameter mesh coverage, or inference-relevant processing logic changes.

## Where are canonical engineering rules documented?

In `DOCS/REPO_DOCS/REPOSITORY_GOVERNANCE.md`.

## Is deleting raw data ever acceptable as part of routine maintenance?

No. Raw source data must not be modified or deleted by normal processing workflows.

