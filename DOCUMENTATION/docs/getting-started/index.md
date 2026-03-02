# Getting Started and Setup

This section provides a practical startup path for contributors working on DATAFLOW_v3.

## Environment prerequisites

- Linux environment with shell access
- Python 3.12
- `pip`
- `yq` (required by several Bash workflows)
- Repository checked out at `$HOME/DATAFLOW_v3`

## Install dependencies

```bash
cd $HOME/DATAFLOW_v3
python3 -m venv .venv
source .venv/bin/activate
pip install -r CONFIG/requirements.list
```

## Core repository layout

```text
DATAFLOW_v3/
├─ MASTER/                # Mother analysis code (real + simulated input processing)
├─ STATIONS/              # Per-station trees and output locations (MINGO00..MINGO04)
├─ MINGO_DIGITAL_TWIN/    # STEP_0..STEP_FINAL simulation pipeline
├─ OPERATIONS/            # Orchestration/observability utilities
├─ OPERATIONS_RUNTIME/    # Runtime locks, logs, status markers
├─ DOCS/                  # Repository governance and runbooks
└─ DOCUMENTATION/         # MkDocs site source
```

## Common commands

### Operational pipeline quick checks

```bash
# Cron and worker status
service cron status
crontab -l
pgrep -af "guide_raw_to_corrected.sh -s"
```

### Digital twin quickstart

```bash
cd $HOME/DATAFLOW_v3/MINGO_DIGITAL_TWIN

# STEP_0 mesh generation
python3 MASTER_STEPS/STEP_0/step_0_setup_to_blank.py \
  --config MASTER_STEPS/STEP_0/config_step_0_physics.yaml

# Run all simulation steps
./run_step.sh all

# Continue from a specific step
./run_step.sh --from 4 --no-plots

# Final formatting only
./run_step.sh final
```

### Main simulation orchestrator (cron target)

```bash
cd $HOME/DATAFLOW_v3/MINGO_DIGITAL_TWIN
./ORCHESTRATOR/core/run_main_simulation_cycle.sh
```

### Targeted simulation tests

```bash
cd $HOME/DATAFLOW_v3
pytest MINGO_DIGITAL_TWIN/PLOTTERS/EXECUTION/MESH/tests/
```

## Documentation local preview

```bash
cd $HOME/DATAFLOW_v3/DOCUMENTATION
mkdocs serve
```

Open `http://127.0.0.1:8000`.

## Next reading

- [Software](../software/index.md)
- [Operational Notes](../operations/index.md)
- [Conventions and Standards](../standards/index.md)
- [Reader Guide](reader-guide.md)
