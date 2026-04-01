---
title: Configuration and Parameter Mesh Reference
description: Runtime/physics configuration keys and parameter-mesh semantics across STEP_0 to STEP_FINAL.
last_updated: 2026-02-24
status: active
supersedes:
  - config_reference.md
  - param_mesh.md
---

# Configuration and Parameter Mesh Reference

## Table of contents
- [Configuration files](#configuration-files)
- [Common runtime keys](#common-runtime-keys)
- [Step-by-step key reference](#step-by-step-key-reference)
- [Parameter mesh schema](#parameter-mesh-schema)
- [Parameter mesh lifecycle](#parameter-mesh-lifecycle)
- [Validation and consistency checks](#validation-and-consistency-checks)

## Configuration files
Each step uses:
- physics config: `MASTER_STEPS/STEP_N/config_step_N_physics.yaml`
- runtime config: `MASTER_STEPS/STEP_N/config_step_N_runtime.yaml` (where applicable)

Canonical contracts are summarized in:
- `contracts/STEP_CONTRACTS.md`

## Common runtime keys
- `input_dir`: upstream directory root.
- `input_glob`: upstream file pattern.
- `input_sim_run`: `latest`, `random`, explicit `SIM_RUN_*`, or list (STEP_FINAL).
- `output_dir`: destination path.
- `output_format`: `pkl` or `csv`.
- `chunk_rows`: rows per chunk for chunked output.
- `plot_sample_rows`: sample count for plots.

## Step-by-step key reference

### STEP_0 (mesh generation)
Physics:
- `cos_n`, `flux_cm2_min`
- `efficiencies`, `efficiencies_identical`
- `efficiencies_max_spread` (number or `null` when `efficiencies_identical=false`)
- `repeat_samples`, `shared_columns`
- `expand_z_positions`

Runtime:
- `station_config_root`
- `output_dir`

### STEP_1 (generation)
Physics:
- `xlim_mm`, `ylim_mm`, `z_plane_mm`
- `seed`, `c_mm_per_ns`
- `cos_n`, `flux_cm2_min` (scalar or `random` from mesh)

Runtime:
- `n_tracks`, `output_basename`
- `param_mesh_dir`, `param_mesh_sim_run`

### STEP_2 (crossings)
Physics:
- `bounds_mm`
- `z_positions` (4 values or `random` from mesh)

Runtime:
- `input_basename`
- `normalize_to_first_plane`
- `param_mesh_dir`, `param_mesh_sim_run`

### STEP_3 (avalanche)
Physics:
- `avalanche_gap_mm`
- `townsend_alpha_per_mm`, `avalanche_electron_sigma`
- `efficiencies` (vector(s) or `random`)

Runtime:
- `param_mesh_dir`, `param_mesh_sim_run`

### STEP_4 (induction)
Physics:
- `x_noise_mm`, `time_sigma_ns`
- `lorentzian_gamma_mm`
- `induced_charge_fraction`

### STEP_5 (signal observables)
Physics:
- `qdiff_width` (required constant sigma for `q_diff`)
- `c_mm_per_ns` (optional override)

### STEP_6 (front/back)
- No required physics keys.

### STEP_7 (cable offsets)
Physics:
- `tfront_offsets` (4x4)
- `tback_offsets` (4x4)

### STEP_8 (FEE + threshold)
Physics:
- `t_fee_sigma_ns`
- `charge_threshold`
- `charge_conversion_model`
- `tot_to_charge_calibration_path`
- `q_to_time_factor` (legacy fallback for linear conversion mode)
- `qfront_offsets`, `qback_offsets`

### STEP_9 (trigger)
Physics:
- `trigger_combinations` (for example `"12"`, `"23"`)

### STEP_10 (DAQ jitter)
Physics:
- `jitter_width_ns`
- `tdc_sigma_ns`

### STEP_FINAL (event builder)
Runtime:
- `input_collect` (`matching` or `baseline_only`)
- `target_rows`
- `files_per_station_conf`
- `payload_sampling` (`random` or `sequential_random_start`)
- `rate_hz` (when `T_thick_s` absent)
- `param_mesh_dir`, `param_mesh_sim_run`

## Parameter mesh schema
Path:
- `INTERSTEPS/STEP_0_TO_1/param_mesh.csv`

Core columns:
- `done` (0/1)
- `step_1_id` to `step_10_id` (3-digit IDs)
- `cos_n`, `flux_cm2_min`
- `eff_p1` to `eff_p4`
- `z_p1` to `z_p4`

Optional lineage columns:
- `param_set_id`
- `param_date`
- future scan variables

Associated metadata file:
- `INTERSTEPS/STEP_0_TO_1/param_mesh_metadata.json`

## Parameter mesh lifecycle
1. STEP_0 appends rows and assigns step IDs from unique parameter combinations.
2. STEP_1 to STEP_3 read randomizable physics values from mesh when configured.
3. STEP_4 to STEP_10 carry IDs forward and use defaults/overrides.
4. STEP_FINAL writes `param_set_id`, `param_date`, and sets `done=1` for matched rows.
5. Housekeeping can prune `done=1` rows from mesh after downstream completion.

## Validation and consistency checks
Recommended checks:

```bash
# Ensure pending mesh rows have upstream runs
PYTHONPATH=. python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/helpers/check_param_mesh_consistency.py \
  --mesh MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv \
  --intersteps MINGO_DIGITAL_TWIN/INTERSTEPS --step 3

# Inspect expected-vs-produced summary
python3 MINGO_DIGITAL_TWIN/ORCHESTRATOR/helpers/size_and_expected_report.py
```

Open review points:
- If fixed-geometry operation is intended, verify mesh-generation policy and closure policy (`RUN_STEP_STRICT_LINE_CLOSURE`) remain compatible.
- If `input_sim_run=random` is used heavily, ensure selection bias is acceptable for the target analysis objective.
