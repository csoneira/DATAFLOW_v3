# STEP 2.2 Bias Corrections

## 2026-02-12: Off-dict leakage of dictionary-equivalent parameter sets

### Problem
`dict_vs_offdict_relerr_*.png` was using `true_is_dictionary_entry` only.  
That allowed rows to appear as `off-dict` even when their parameter set already existed in the dictionary (same physics, different `n_events`).

### Why this is a bias
Those rows are not truly out-of-dictionary in parameter space, so including them in `off-dict` narrows the apparent difference and biases the comparison.

### Correction implemented
- File: `INFERENCE_DICTIONARY_VALIDATION/STEP_2_INFERENCE/STEP_2_2_VALIDATION/validate_solution.py:246`
- Added parameter-set overlap detection with priority:
1. Exact key match by `param_hash_x` (mapped from `dataset_index` into `dataset.csv` and compared against dictionary keys).
2. Fallback match by rounded tuple of true parameters (`flux_cm2_min`, `cos_n`, `eff_sim_1..4`, `z_plane_1..4`).
- Strict off-dict definition:
1. `~true_is_dictionary_entry`
2. and `~same_parameter_set_as_dictionary`
- Plot labels/titles now use `off-dict strict`.
  - See `validate_solution.py:489` and `validate_solution.py:521`.

### Run verification (same date)
- `in-dict`: 74
- `off-dict raw`: 255
- `excluded overlap`: 7
- `off-dict strict`: 248

### Affected plot
- `INFERENCE_DICTIONARY_VALIDATION/STEP_2_INFERENCE/STEP_2_2_VALIDATION/OUTPUTS/PLOTS/dict_vs_offdict_relerr_eff_sim_1.png`
