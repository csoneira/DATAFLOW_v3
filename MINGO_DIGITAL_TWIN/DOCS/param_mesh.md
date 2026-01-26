# Parameter Mesh Specification

The parameter mesh is the central table used to coordinate parameter scans and step ID
selection. It lives at `INTERSTEPS/STEP_0_TO_1/param_mesh.csv` and is generated/extended
by STEP 0.

## Core columns
- `done` (int): 0/1 flag; rows marked done are considered complete by STEP FINAL.
- `step_1_id`..`step_10_id` (string): 3-digit step IDs.
- `cos_n`: zenith exponent for muon generation.
- `flux_cm2_min`: muon flux (count/cm^2/min).
- `eff_p1..eff_p4`: per-plane efficiencies.
- `z_p1..z_p4`: plane z positions (mm).

Optional columns:
- `param_set_id`: unique integer assigned at STEP FINAL.
- `param_date`: ISO date assigned at STEP FINAL.
- Any additional scan variables (if added by future extensions).

## STEP 0 behavior
STEP 0 (`MASTER_STEPS/STEP_0/step_0_setup_to_blank.py`) does the following:
- Reads station configuration CSVs to enumerate valid `z_p1..z_p4` tuples.
- Samples `cos_n`, `flux_cm2_min`, and efficiencies from configured ranges.
- Optionally repeats samples (`repeat_samples`) and shares parameters across repeats
  (`shared_columns`).
- Optionally expands each sample across all z-position tuples (`expand_z_positions`).
- Appends rows to `param_mesh.csv` and assigns step IDs based on unique combinations.

## STEP 1-3 usage
- STEP 1 uses `cos_n` and `flux_cm2_min` if either is set to `random` in its config.
- STEP 2 uses `z_p1..z_p4` if `z_positions` is set to `random`.
- STEP 3 uses `eff_p1..eff_p4` if `efficiencies` is set to `random`.

Each step selects the first available row (in a seeded random order) that does not
already have a corresponding output SIM_RUN.

## STEP FINAL usage
STEP FINAL matches the current run parameters to a single mesh row and assigns:
- `param_set_id` (incremented integer).
- `param_date` (monotonic date sequence).
- `done = 1`.

## Metadata
`param_mesh_metadata.json` stores the mesh creation parameters, sample counts, and timestamps.
