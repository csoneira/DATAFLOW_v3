# MASTER_VS_SIMULATION Pipeline Reference

This document is an exhaustive reference for the active `STEP_*` scripts in `MASTER_VS_SIMULATION`:

- `STEP_1_DICTIONARY/build_dictionary.py`
- `STEP_2_SIM_VALIDATION/compute_relative_error.py`
- `STEP_3_SELF_CONSISTENCY/self_consistency_r2.py`

It is written as prompt context for future Codex sessions.

## End-to-End Purpose

1. Build a simulation dictionary CSV (Step 1).
2. Validate simulated efficiencies and filter reliable rows (Step 2).
3. Run self-consistency matching in `(flux, efficiency)` space (Step 3).

## Folder Layout (Active)

```text
MASTER_VS_SIMULATION/
  README.md
  STEP_1_DICTIONARY/
    build_dictionary.py
    output/
  STEP_2_SIM_VALIDATION/
    compute_relative_error.py
    config.json
    output/
  STEP_3_SELF_CONSISTENCY/
    self_consistency_r2.py
    config.json
    output/
```

## Runtime Dependencies

- Python 3
- `pandas`
- `matplotlib`
- `numpy` (Step 3)
- `scipy` optional (Step 3 smooth contour interpolation; code falls back if missing)

## CLI/Config Precedence Rule

For Steps 2 and 3, values are resolved as:

1. CLI argument, if provided.
2. Config JSON key, if present.
3. Hardcoded script default.

Step 1 takes CLI args directly and forwards to an external builder script.

## STEP 1: Dictionary Build Wrapper

Script: `STEP_1_DICTIONARY/build_dictionary.py`

### What It Does

- Creates `out_dir/task_XX/param_metadata_dictionary.csv` by delegating to:
  - `MASTER_VS_SIMULATION/STEP_1_DICTIONARY/STEP_1_BUILD/build_param_metadata_dictionary.py`
- Optionally creates quick sanity plots from output CSV.

### CLI

```bash
python3 MASTER_VS_SIMULATION/STEP_1_DICTIONARY/build_dictionary.py [options]
```

Arguments:

- `--config` (default: `/home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION/STEP_1_DICTIONARY/config/pipeline_config.json`)
- `--task-id` (default: `1`)
- `--out-dir` (default: `MASTER_VS_SIMULATION/STEP_1_DICTIONARY/output`)
- `--no-plots` (flag; skips histogram/scatter generation)

### Behavior Details

- Always creates `out_dir` and parent folders.
- Output file path is always:
  - `out_dir/task_{task_id:02d}/param_metadata_dictionary.csv`
- If plots are enabled:
  - Deletes existing `*.png` in `.../plots`
  - Writes:
    - `hist_flux_cm2_min.png`
    - `hist_cos_n.png`
    - `hist_z_plane_1.png`
    - `hist_z_plane_2.png`
    - `hist_z_plane_3.png`
    - `hist_z_plane_4.png`
    - `scatter_flux_vs_cos_n.png`
    - `scatter_flux_vs_z1.png`

### Data Produced

The dictionary CSV is wide and contains metadata, rates, counts, and derived rate/fraction features.  
Important fields used downstream include:

- IDs/join keys:
  - `file_name`
  - `filename_base`
  - `param_set_id`
- Physical params:
  - `flux_cm2_min`
  - `cos_n`
  - `z_plane_1..4`
  - `efficiencies` (stringified 4-element list)
- Count features used for efficiency estimation:
  - `raw_tt_1234_count`
  - `raw_tt_234_count`
  - `raw_tt_134_count`
  - `raw_tt_124_count`
  - `raw_tt_123_count`
- Rate features:
  - many `*_rate_hz`, including `raw_tt_*_rate_hz`
  - `events_per_second_global_rate`
- Event count fields:
  - `selected_rows`
  - `requested_rows`

## STEP 2: Simulation Validation and Filtering

Script: `STEP_2_SIM_VALIDATION/compute_relative_error.py`

### What It Does

1. Loads dictionary CSV.
2. Builds a validation table by calling:
   - `from validate_simulation_vs_parameters import build_validation_table`
   - local module path: `MASTER_VS_SIMULATION/STEP_2_SIM_VALIDATION/validate_simulation_vs_parameters.py`
3. Joins selected validation columns back to dictionary rows.
4. Filters rows by relative error thresholds and minimum event count.
5. Writes full and filtered outputs plus diagnostic plots.

### CLI

```bash
python3 MASTER_VS_SIMULATION/STEP_2_SIM_VALIDATION/compute_relative_error.py [options]
```

Arguments:

- `--config` (default: `MASTER_VS_SIMULATION/STEP_2_SIM_VALIDATION/config.json`)
- `--dictionary-csv`
- `--out-dir`
- `--prefix` (default fallback: `raw`)
- `--eff-method` choices:
  - `four_over_three_plus_four`
  - `one_minus_three_over_four`
- `--relerr-threshold` (absolute threshold, used for p2 and p3)
- `--min-events` (minimum generated events to keep row)
- `--no-plots`

### Config Keys (Step 2)

From `STEP_2_SIM_VALIDATION/config.json`:

- `dictionary_csv` (optional)
- `out_dir` (optional)
- `prefix` (optional)
- `eff_method` (optional)
- `relerr_threshold` (present in current file; currently `0.05`)
- `min_events` (present in current file; currently `20000`)

If missing, code defaults are:

- `dictionary_csv`: `STEP_1_DICTIONARY/output/task_01/param_metadata_dictionary.csv`
- `out_dir`: `STEP_2_SIM_VALIDATION/output`
- `prefix`: `raw`
- `eff_method`: `four_over_three_plus_four`
- `relerr_threshold`: `0.01`
- `min_events`: `50000`

### Validation Math

For each plane `i`, estimated efficiency is computed from topology counts:

- With `four_over_three_plus_four`:
  - `eff_est_pi = N1234 / (N1234 + N3of4_missing_i)`
- With `one_minus_three_over_four`:
  - `eff_est_pi = 1 - N3of4_missing_i / N1234`

Then:

- `eff_resid_pi = eff_est_pi - eff_sim_pi`
- `eff_rel_err_pi = eff_resid_pi / eff_sim_pi`
- `eff_abs_err_pi = abs(eff_resid_pi)`

`generated_events_count` is sourced from first existing field in:

- `generated_events_count`
- `total_events_generated`
- `n_events_generated`
- `num_events_generated`
- `event_count`
- `num_events`
- `selected_rows`

### Filtering Rule

Rows are marked usable when all conditions are true:

```text
abs(eff_rel_err_p2) <= relerr_threshold
abs(eff_rel_err_p3) <= relerr_threshold
generated_events_count >= min_events
```

A boolean column is added:

- `used_in_reference` (`True` for kept rows, `False` otherwise)

### Join Logic

Join key between dictionary and validation table:

- First available of `file_name` or `filename_base`.
- Raises `KeyError` if neither exists in both tables.

### Outputs

Written under `out_dir`:

- `validation_table.csv`
- `filtered_reference.csv`
- `used_dictionary_entries.csv`
- `unused_dictionary_entries.csv`

If plots enabled, under `out_dir/plots`:

- `hist_eff_rel_err_p2.png`
- `hist_eff_rel_err_p3.png`
- `scatter_relerr_p2_vs_p3.png`
- `scatter_eff2_sim_vs_est.png`
- `scatter_eff3_sim_vs_est.png`
- `hist_used_vs_unused_eff_rel_err_p2.png`
- `hist_used_vs_unused_eff_rel_err_p3.png`
- `scatter_used_vs_unused_relerr_p2_vs_p3.png`
- `hist_used_vs_unused_flux.png`
- `hist_used_vs_unused_cos_n.png`
- `counts_total_vs_filtered.png`
- `counts_used_vs_unused.png`

## STEP 3: Self-Consistency Search

Script: `STEP_3_SELF_CONSISTENCY/self_consistency_r2.py`

### What It Does

Given a reference CSV (normally Step 2 filtered output), it:

1. Chooses a sample row.
2. Restricts candidates to same z-plane geometry.
3. Scores candidate fingerprints against sample fingerprint.
4. Finds best candidate and parameter errors in `(flux, eff_1)`.
5. Exports candidate ranking, summary JSON, and many plots.

### CLI

```bash
python3 MASTER_VS_SIMULATION/STEP_3_SELF_CONSISTENCY/self_consistency_r2.py [options]
```

Arguments:

- `--config` (default: `STEP_3_SELF_CONSISTENCY/config.json`)
- `--reference-csv`
- `--dictionary-csv`
- `--out-dir`
- `--seed` (int or `"random"`)
- `--metric-suffix` (default fallback: `_rate_hz`)
- `--metric-prefix` (default fallback: `raw_tt_`)
- `--metric-mode` choices:
  - `raw_tt_rates` (default, recommended in docstring)
  - `eff_global`
  - `dict_eff_global`
- `--global-rate-col` (default fallback: `events_per_second_global_rate`)
- `--include-global-rate` (`true/false` parsing)
- `--eff-method` choices:
  - `four_over_three_plus_four`
  - `one_minus_three_over_four`
- `--rate-prefix` (default fallback: `raw`)
- `--metric-scale` choices: `none`, `zscore`
- `--score-metric` choices: `l2`, `chi2`, `poisson`, `r2`
- `--cos-n` (target for auto sample selection)
- `--eff-tol` (equal-eff constraint tolerance for auto sample selection)
- `--eff-match-tol` (truth rank matching tolerance)
- `--flux-tol` (truth rank matching tolerance)
- `--z-tol` (z-plane candidate matching tolerance)
- `--sample-index` (force exact row)
- `--max-sample-tries` (parsed from config/CLI; currently unused in logic)
- `--top-n`
- `--exclude-self` (remove sample row from candidate set)

### Config Keys (Current `STEP_3_SELF_CONSISTENCY/config.json`)

- `reference_csv`
- `dictionary_csv`
- `out_dir`
- `seed` (currently `"random"`)
- `metric_suffix`
- `metric_prefix`
- `metric_mode`
- `global_rate_col`
- `include_global_rate`
- `metric_scale`
- `score_metric`
- `eff_method`
- `rate_prefix`
- `cos_n`
- `eff_tol`
- `eff_match_tol`
- `flux_tol`
- `z_tol`
- `max_sample_tries`
- `top_n`
- `exclude_self`

### Metric Modes

1. `raw_tt_rates`
- Features: columns ending with `metric_suffix` and starting with `metric_prefix`.
- Optionally appends `global_rate_col` when `include_global_rate` is true.

2. `eff_global`
- Features: `eff_1`, `eff_2`, `eff_3`, `eff_4`, plus `global_rate_col`.
- `eff_1..4` are parsed from `efficiencies` column.

3. `dict_eff_global`
- Features: dictionary efficiencies (`dict_eff_1..4`) + `global_rate_col`.
- Sample vector uses empirical efficiencies estimated from raw topology counts.
- Script explicitly warns this can be unreliable for asymmetric geometries.

### Scoring Metrics

- `l2`: Euclidean norm of feature differences.
- `chi2`: sum of squared residuals weighted by `max(y_true, 1)`.
- `poisson`: Poisson deviance-like score.
- `r2`: coefficient of determination.

`l2`, `chi2`, `poisson` are "lower is better".  
`r2` is "higher is better".

### Sample Selection Logic

If `--sample-index` is provided, that row is used. Otherwise:

1. Prefer rows with:
   - `dict_cos_n` exactly near `cos_n` (`<= 1e-9`)
   - all `dict_eff_1..4` present
   - `dict_eff_2/3/4` each within `eff_tol` of `dict_eff_1`
2. If none found, relax to `|dict_cos_n - cos_n| <= 0.05` and `dict_eff_1` non-null.
3. Random choice among eligible rows (seeded or unseeded depending `seed`).

### Candidate Filtering

Candidates are rows in reference CSV with same geometry as sample:

- `z_plane_1..4` each matched with `np.isclose(..., atol=z_tol)`
- Optional sample removal if `exclude_self` is true.

### Exports

Written under `out_dir`:

- `r2_candidates.csv` (all z-matched candidates + dictionary fields + scores/errors)
- `top_candidates.csv` (top `N`, if `top_n > 0`)
- `r2_summary.json`

Summary JSON contains:

- sample identity/params
- feature/metric settings
- candidate counts
- best candidate identity/score
- absolute and relative parameter errors
- truth rank diagnostics

### Plot Outputs (11 files)

Filename tag is `{metric_mode}_{score_metric}`. Generated files:

- `contour_{tag}.png`
- `features_{tag}.png`
- `scatter_sample_vs_best_{tag}.png`
- `hist_score_{tag}.png`
- `score_vs_param_dist_{tag}.png`
- `top_n_param_space_{tag}.png`
- `residuals_{tag}.png`
- `l2_contribution_{tag}.png`
- `profiles_top_n_{tag}.png`
- `score_cdf_{tag}.png`
- `flux_eff_errors_{tag}.png`

## Step Interfaces (Data Contracts)

### Step 1 -> Step 2

Step 2 expects Step 1 dictionary to include at minimum:

- join column: `file_name` or `filename_base`
- `flux_cm2_min`, `cos_n`, `efficiencies`
- count fields:
  - `{prefix}_tt_1234_count`
  - `{prefix}_tt_234_count`
  - `{prefix}_tt_134_count`
  - `{prefix}_tt_124_count`
  - `{prefix}_tt_123_count`
- event count source (any one of):
  - `generated_events_count`, `total_events_generated`, `n_events_generated`,
    `num_events_generated`, `event_count`, `num_events`, or `selected_rows`

### Step 2 -> Step 3

Step 3 default input is `STEP_2_SIM_VALIDATION/output/filtered_reference.csv`.  
It additionally requires dictionary CSV (`STEP_1_DICTIONARY/output/task_01/param_metadata_dictionary.csv`) for true parameter joins and diagnostics.

## Embedded Code Policy

JUNK dependencies used by this pipeline are vendored into the active `STEP_*` folders:

1. Step 1 builder code:
   - `STEP_1_DICTIONARY/STEP_1_BUILD/build_param_metadata_dictionary.py`
   - config copy: `STEP_1_DICTIONARY/config/pipeline_config.json`

2. Step 2 validation code:
   - `STEP_2_SIM_VALIDATION/validate_simulation_vs_parameters.py`

So Step 1/Step 2 run without importing code from `MASTER_VS_SIMULATION/JUNK`.

## Typical Run Sequence

```bash
# Step 1
python3 MASTER_VS_SIMULATION/STEP_1_DICTIONARY/build_dictionary.py \
  --task-id 1 \
  --out-dir MASTER_VS_SIMULATION/STEP_1_DICTIONARY/output

# Step 2
python3 MASTER_VS_SIMULATION/STEP_2_SIM_VALIDATION/compute_relative_error.py \
  --config MASTER_VS_SIMULATION/STEP_2_SIM_VALIDATION/config.json

# Step 3
python3 MASTER_VS_SIMULATION/STEP_3_SELF_CONSISTENCY/self_consistency_r2.py \
  --config MASTER_VS_SIMULATION/STEP_3_SELF_CONSISTENCY/config.json
```

## Prompt-Ready Context Snippet

When asking Codex to modify this pipeline, include:

- step script path(s)
- metric mode (`raw_tt_rates`, `eff_global`, or `dict_eff_global`)
- score metric (`l2`, `chi2`, `poisson`, `r2`)
- filtering thresholds (`relerr_threshold`, `min_events`)
- whether path caveats above are already fixed in your branch
