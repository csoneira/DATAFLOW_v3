# Metadata and Reproducibility

This document describes how sim runs are tracked, how metadata is stored, and what affects
reproducibility in MINGO_DIGITAL_TWIN.

## Sim-run registries
Each INTERSTEPS directory contains `sim_run_registry.json` with entries like:
- `sim_run`: SIM_RUN identifier.
- `created_at`: ISO timestamp.
- `step`: step name (e.g., STEP_4).
- `config_path`: physics config path.
- `config_hash`: SHA256 hash of the physics config.
- `upstream_hash`: hash derived from upstream metadata.
- `config`: full physics config snapshot.

Sim runs are considered equivalent when both `config_hash` and `upstream_hash` match.

## Output metadata
Each output file contains metadata in one of two forms:
- Sidecar JSON for single files: `<output>.<ext>.meta.json`.
- Chunked manifest: `<output>_chunks.chunks.json` with a `metadata` block.

Common metadata fields:
- `created_at`, `step`, `config`, `runtime_config`, `sim_run`.
- `config_hash`, `upstream_hash`.
- `row_count` (when known).
- `upstream`: nested metadata from the previous step.

Step-specific metadata (examples):
- `step_1_id..step_10_id`: step ID chain.
- `param_set_id`, `param_date`, `param_row_id`: param mesh lineage.
- `param_mesh_path`: path to the mesh used for selection.
- `source_dataset`: upstream file path (steps 4+).
- `z_positions_mm` and `z_positions_raw_mm` (STEP 2).

## Chunked output conventions
When `chunk_rows` is set, outputs are stored as:
- `SIM_RUN_XXXX/<stem>_chunks/part_0000.(pkl|csv)` etc.
- `SIM_RUN_XXXX/<stem>_chunks.chunks.json` manifest with:
  - `version`, `chunks`, `row_count`, `metadata`.

Manifest `chunks` paths are stored as full paths to the chunk files.

## Step ID chain and sim_run names
Sim run names are constructed as:
```
SIM_RUN_<step_1_id>_<step_2_id>_..._<step_N_id>
```
Step IDs are 3-digit strings (e.g., 001). The chain length increases with each step.

For steps 1-3, IDs come from the parameter mesh. For steps 4-10, IDs are selected from the
mesh using `select_next_step_id`, with optional overrides in the physics config.

## Parameter mesh lineage
- `param_mesh.csv` records scan parameters and step IDs.
- `param_set_id` and `param_date` are assigned in STEP FINAL when outputs are emitted.
- `done=1` marks mesh rows that have completed a full pipeline to STEP FINAL.

See `DOCS/param_mesh.md` for details.

## Reproducibility notes
- Most stochastic operations use `np.random.default_rng(seed)` with `seed` from config.
- STEP 1 uses an unseeded RNG for thick-time generation; `T_thick_s` is therefore not
  strictly reproducible across runs unless modified upstream.
- `input_sim_run: random` uses the Python `random` module with optional seed.
- When chunked input is processed, ordering is determined by the chunk manifest and glob
  ordering; changes to file system state can alter selection order.
