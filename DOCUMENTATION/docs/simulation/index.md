# Digital twin simulation

*Last updated: March 2026*

This page introduces the **MINGO digital twin**, a parameterized software
model of the RPC detector and readout chain used throughout the
miniTRASGO/CASTRO collaboration. The simulation walks an individual muon
particle through eleven deterministic steps, producing outputs that are
indistinguishable from data collected by a real station.  Where randomness is
required (ray generation, ionisation locations, noise), the seed is explicitly
controlled so that runs are reproducible; the seed value is recorded in the
`*.meta.json` files.  The twin is used for validation, dictionary building, and
synthetic-data ingestion into the main analysis pipeline.

Sections below provide a high‑level overview; more detailed technical
information lives in the [digital twin documentation in the repository]
(../MINGO_DIGITAL_TWIN/DOCS/README.md) and in the per‑step contracts under
`MINGO_DIGITAL_TWIN/DOCS/contracts/`.

## Motivation and hardware context

The detector hardware is described on the
[Detector design → Hardware](../design/hardware.md) page.  The digital twin is
written to reproduce the same geometry, gas mixture, front‑end electronics,
channel mapping and timing conventions.  Having a faithful simulation allows us
to:

1. Generate large, labelled datasets for developing inference algorithms.
2. Test the impact of environmental changes (pressure, temperature, gas flow)
   without risking the physical chambers.
3. Build parameter/DC dictionaries that relate observable rates and
   efficiencies to physical flux, which form the core of the rapid
   reconstruction software.

## Pipeline overview

A single muon event is propagated through the following steps (each implemented
as a standalone Python script under `MINGO_DIGITAL_TWIN/MASTER_STEPS`).  Every
step implements a well‑defined interface contract; consult
`MINGO_DIGITAL_TWIN/DOCS/contracts/` for the detailed column definitions and
expected inputs/outputs.

A summary of the directories and their primary entry points follows:

```
MASTER_STEPS/
├─ STEP_0/                 # mesh generation
│   └─ step_0_setup_to_blank.py
├─ STEP_1/ STEP_2/         # ray generation & plane intersections
│   └─ step_1_to_2_generator.py
├─ STEP_3/                 # gas gap ionisation
│   └─ step_3_gas.py
├─ STEP_4/ STEP_5/ STEP_6/ # charge induction & endpoint extraction
│   └─ step_4_to_6_induction.py
├─ STEP_7/ STEP_8/ STEP_9/ # electronics modelling + trigger logic
│   └─ step_7_to_9_electronics.py
├─ STEP_10/                # TDC digitisation
│   └─ step_10_tdc.py
└─ STEP_FINAL/             # formatting to station .dat
    └─ step_final_daq_to_station_dat.py
```

For each step the corresponding `run_step.sh` option simply sources the
associated Python module; advanced users can invoke the Python file directly if
additional arguments are required (see the module docstring).  The `run_step.sh`
wrapper also handles establishing virtual environments, logging, and
sanitising the `SIM_RUN` directory structure.

Step directories contain a `README.md` and, for fan‑out steps, a
`sim_run_registry.json` that lists all produced SIM_RUNs along with their
parameter values.

1. **Geometry & kinematics** (`STEP_1`–`STEP_2`) – generate ray from a random
   position and incoming direction, compute intersections with RPC planes.
   These steps fan out over the mesh of angles, positions, and fluxes.
2. **Gas gap ionisation and avalanche modelling** (`STEP_3`) – simulate primary
   ionisation clusters and Townsend amplification.  STEP_3 also fans out,
   producing multiple gas-gap parameter variations per upstream run.
3. **Charge induction on strips and endpoint extraction**
   (`STEP_4`–`STEP_6`) – map charges to strip channels, apply diffusion and
   coupling models, then derive front/back timing endpoints.  These stages are
   1:1 with upstream runs.
4. **Electronics** (`STEP_7`–`STEP_9`) – apply cable length offsets,
   front‑end amplifier response, discriminator thresholds, and implement the
   trigger logic used by the real FEE boards.  Noise and cross-talk models are
   included.
5. **TDC digitisation and formatting** (`STEP_10`, `STEP_FINAL`) – digitise
   timestamps and pulse widths, then convert the combined data into the
   station-style `.dat` format used by the analysis pipeline.

_Notes on fan-out:_ Steps 1–3 are parametrised over multiple physics variables and
therefore produce many downstream SIM_RUNs for each upstream identifier.  A
SIM_RUN name encodes the chain of upstream ids, e.g. `SIM_RUN_001_002_003` for a
run that originated from STEP_1 id 001, STEP_2 id 002 and STEP_3 id 003.  The
hashes recorded in `*.meta.json` (`config_hash` and `upstream_hash`) guarantee
that two runs are identical only if both the configuration and the entire
upstream lineage match.

Steps 4–10 are always 1:1: their SIM_RUN names include a fixed `001` suffix and
do not expand the mesh further.  Upstream intermediates are automatically
pruned by the orchestration scripts (`ORCHESTRATOR/maintenance/sanitize_sim_runs.py`)
once they are consumed, which keeps the workspace size under control.  Users
can disable pruning by setting `--no-prune` when invoking `run_step.sh` for
debugging.

Each step writes a SIM_RUN directory and metadata; the orchestration scripts in
`MINGO_DIGITAL_TWIN/ORCHESTRATOR` manage dependencies, hashing, and
backpressure.  Steps 1–3 fan out to cover a mesh of physical parameters;
upstream intermediates are pruned once downstream stages have consumed them.

The simulation is configured via YAML files in `CONFIG_FILES/`.  See
[Digital twin configuration and parameter mesh](../MINGO_DIGITAL_TWIN/DOCS/CONFIGURATION_AND_PARAM_MESH.md)
for further details.

## Running the twin

A few common invocation patterns (run from within the
`MINGO_DIGITAL_TWIN` directory):

```bash
# prepare a blank parameter mesh (only needed once per machine)
python3 MASTER_STEPS/STEP_0/step_0_setup_to_blank.py \
    --config MASTER_STEPS/STEP_0/config_step_0_physics.yaml

# execute all steps in sequence
./run_step.sh all

# or run just a single step (e.g. STEP_5) on existing upstream data
./run_step.sh 5

# continue processing from a given step onward
./run_step.sh --from 4

# force overwrite of an existing SIM_RUN
./run_step.sh 3 --force

# run final formatting only (DAQ → station .dat)
./run_step.sh final
```

By default the scripts obey runtime configuration variables such as
`SIM_MAX_UNPROCESSED_FILES` (see the backpressure docs).  Environment
variables like `SIM_ALERT_TELEGRAM_CHAT_IDS` may be set to enable
telegram alerts for stuck executions; consult
`ORCHESTRATOR/core/run_main_simulation_cycle.sh` for details.

### Configuration files and parameter mesh

The behaviour of each step is controlled by YAML files in
`CONFIG_FILES/` (physics parameters, geometry, electronics models) and
`runtime_config` files stored alongside each SIM_RUN.  The most commonly
edited files are:

- `config_step_0_physics.yaml` – defines global physics constants and
  random-seed policy.
- `step_1_2_mesh.yaml` – parameter mesh for steps 1–3 (angles, fluxes,
  efficiencies, environmental conditions).
- `config_step_final_physics.yaml` – settings for the DAQ formatting.

To modify the mesh, edit `step_1_2_mesh.yaml` and regenerate via
`step_0_setup_to_blank.py`; subsequent `run_step.sh` invocations will
automatically pick up the new mesh if `--force` is supplied.  Reproducibility
is ensured by hashing the configuration; the `*.meta.json` file records
`config_hash` and `upstream_hash` so that results can be traced back to the
exact YAML that produced them.

#### Parameter mesh schema & lifecycle

The parameter mesh (`INTERSTEPS/STEP_0_TO_1/param_mesh.csv`) is a central
database that coordinates the fan‑out of STEPs 1–3.  Its core columns are:

```
done,step_1_id,...,step_10_id,cos_n,flux_cm2_min,eff_p1,...,eff_p4,
z_p1,...,z_p4,param_set_id,param_date
```

- `done` flags rows that have reached STEP_FINAL.
- `step_*_id` fields are three‑digit identifiers assigned sequentially by
  each step as it processes the mesh.
- `param_set_id` and `param_date` record the mesh version and creation date.

The lifecycle is:
1. STEP_0 appends new rows for each unique parameter combination.
2. Steps 1–3 sample randomised physics values from the mesh when configured to
   do so; they also assign their own `step_*_id` values.
3. Steps 4–10 carry IDs forward unchanged (always `001` suffix) and may use
   mesh values as defaults.
4. STEP_FINAL writes `param_set_id`, `param_date`, and sets `done=1` for the
   corresponding mesh row.
5. Optional housekeeping can prune `done=1` rows once downstream steps are
   complete; see `ORCHESTRATOR/maintenance/`.

#### Validation commands

Several helper scripts check mesh consistency and produced-output counts:

```bash
PYTHONPATH=. python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/helpers/check_param_mesh_consistency.py \
    --mesh INTERSTEPS/STEP_0_TO_1/param_mesh.csv \
    --intersteps INTERSTEPS --step 3

python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/helpers/size_and_expected_report.py
```

These are useful to run before and after large mesh changes to ensure no
parameter combinations are lost or duplicated.

See [Digital twin configuration and parameter mesh](../MINGO_DIGITAL_TWIN/DOCS/CONFIGURATION_AND_PARAM_MESH.md)
for full descriptions and supported fields.

### Output format

Each step produces a directory named `SIM_RUN_*` containing:

- Step-specific data files (CSV, HDF5, etc.)
- A `*.meta.json` sidecar with fields `step`, `config`, `runtime_config`,
  `sim_run`, `config_hash`, `upstream_hash` and `lineage` records.
- For fan‑out steps the directory contains multiple sub‑SIM_RUNs.

`STEP_FINAL` writes station‑format `.dat` files identical to those produced
by real hardware; these can be fed directly into the analysis software.

### Monitoring & troubleshooting

Backpressure is overseen by Python scripts and cron jobs located in
`MINGO_DIGITAL_TWIN/PLOTTERS/EXECUTION/BACKPRESSURE/`.  Run
`python3 plot_backpressure_monitor.py --help` to see the available plots
(showing pending files per step, processing lag, etc.).  The backpressure
thresholds themselves are configured in
`CONFIG_FILES/sim_main_pipeline_frequency.conf` (set any threshold to `0` to
disable that check).

The main orchestration cycle (`ORCHESTRATOR/core/run_main_simulation_cycle.sh`)
implements file-based locking (`sim_enqueue.lock`, `sim_processing.lock`,
`sim_final.lock`) to prevent concurrent executions and rotates logs under
`OPERATIONS_RUNTIME/CRON_LOGS/SIMULATION/`.  Look in the `*_cron.log` files for
recent failures or use `grep -i error` on the entire directory.

Each SIM_RUN has its own `run.log` file inside the directory; examine it when
an individual step fails.  The `ensure_sim_hashes.py` script can be run manually
or via cron to verify that outputs match their expected configuration hashes.

For common problems consult:

- [Digital twin troubleshooting runbook](../MINGO_DIGITAL_TWIN/DOCS/TROUBLESHOOTING/RUNBOOK.md)
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/` scripts (e.g.
  `sanitize_sim_runs.py`, `ensure_sim_hashes.py`)
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/prune_step_final_params.py` for
  housekeeping of final output catalog.


---

_View the full set of digital twin notes and troubleshooting guides in the
repository’s own `MINGO_DIGITAL_TWIN/DOCS` directory._

Outputs mirror the real station `.dat` format, so they can be ingested by the
main analysis pipeline (see [Analysis software](../analysis/index.md)).

---

_View the full set of digital twin notes and troubleshooting guides in the
repository’s own `MINGO_DIGITAL_TWIN/DOCS` directory._
