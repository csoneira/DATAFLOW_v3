# Another Method

This small project builds a scale-factor LUT from simulation parameters plus
MINGO00 metadata (joined by `param_hash`), applies it to `synthetic_dataset.csv`,
and then applies it to a configurable slice of real station data.

## Workflow

1. `step_1_prepare_data.py`
  - Reads `step_final_simulation_params.csv`.
  - Reads MINGO00 `metadata_trigger_type.csv` from one or more Task IDs.
  - Joins simulation parameters and real observed rates by `param_hash`.
  - Reads only the configured trigger-type columns (minimal `usecols`) and can
    keep offender-related trigger-rate columns in the Step 1 output CSV.
   - Keeps the four empirical efficiencies, the z-position vector, the
     configured rate column, the simulated flux, and the number of events.
   - Filters on the configured z-position vector and minimum number of events.
   - If `step1.z_position_vector` is `null`, the most frequent z-vector is used.
   - Writes a compact parameter-space overview plot for `flux + the four efficiencies`.

2. `step_2_build_lut.py`
   - Bins the four efficiencies with `step2.efficiency_bin_width`.
   - Bins flux with `step2.flux_bin_count`.
   - Computes the median and IQR of `R` in each `(efficiency bin, flux bin)`.
   - Builds a flux-dependent reference curve from the `step2.reference_top_k_per_flux_bin`
     bins that are closest to `[1, 1, 1, 1]` in each flux bin.
   - Builds the scale factor as the median over flux bins of
     `R_reference_band(flux) / R_bin(flux)`.
   - Builds the diagnostic plots from a supported diagonal or near-diagonal
     band, summarized in bins of the mean efficiency, because exact 4D
     efficiency vectors in this dataset do not repeat across flux.
   - Writes an ASCII LUT whose first commented line stores the z-position
     vector, followed by `emp_eff_1 emp_eff_2 emp_eff_3 emp_eff_4 scale_factor`.

3. `step_3_apply_lut.py`
   - Reads the synthetic dataset.
   - Applies the same efficiency binning.
   - Matches each row to the LUT, keeping exact bin hits and then using either
     nearest-neighbour or inverse-distance interpolation over the raw empirical
     efficiencies.
   - Writes the corrected rate to the output CSV.
   - Plots the rate time series together with the flux time series and the four
     plane-efficiency time series.
   - Plots flux vs observed rate and flux vs corrected rate for a visual check
     of the correction.

4. `step_4_study_lut.py`
   - Reads the LUT diagnostic table from Step 2.
   - Extracts four single-plane slices through the LUT.
   - For each plane and each available efficiency value, keeps the row where the
     other three efficiencies are as close to `1` as the LUT allows.
   - Plots the LUT scale factor vs the varied efficiency for the four planes on
     the same figure.
   - Plots the reference-normalized rate `1 / scale_factor` vs the varied
     efficiency for the four planes on the same figure.
   - Extracts a configurable two-plane slice, by default `eff_2` vs `eff_3`,
     while keeping the other two planes as close to `1` as the LUT allows.
   - Plots a 2-D surface of the scale factor and the reference-normalized rate.
   - Plots a 2-D quality view showing the distance of the fixed planes to `1`
     and the support of the selected slice.

5. `step_5_apply_lut_to_real_data.py`
   - Reads the real station metadata directly from the station pipeline outputs.
   - Uses its own metadata collector (no dependency on the STEP 4.1 collector).
   - Filters the real rows by `step5.station`, `step5.date_from`, and
     `step5.date_to`.
   - Uses offender-limited rate and per-plane efficiencies from
     `task_<id>_metadata_noise_control.csv` (same source used by the
     noise-control reports), with the threshold controlled by
     `step5.max_selected_offenders`.
   - Applies the LUT with the same exact-match + fallback strategy used in
     Step 3.
   - Checks whether the real-data z positions in the requested window match the
     LUT z vector and writes a warning to the log and metadata if they do not.
   - Writes a query-coverage CSV that highlights which real-data efficiency
     bins are unsupported by the LUT and which simulated bins they fall back to.
   - Plots the observed and LUT-corrected real-data rate together with the four
     empirical efficiency time series.
   - Plots correction diagnostics: observed vs corrected rate scatter and LUT
     scale factor vs time.

## Run

```bash
cd /home/mingo/DATAFLOW_v3/MINGO_DICTIONARY_CREATION_AND_TEST/ANOTHER_METHOD
python3 run_all.py
```

`run_all.py` directly runs the local Step 1 to Step 5 workflow. It does not
call the upstream `collect_data.py` pre-refresh step.

To change the z-position vector, binning, or the Step 5 real-data station/date
window, edit `config.json`. To switch the LUT from the global rate to another
rate definition, change `columns.rate` in that file. The fallback behavior is
controlled with `step3.lut_match_mode` / `step5.lut_match_mode` (`exact`,
`nearest`, or `interpolate`) together with `lut_interpolation_k` and
`lut_interpolation_power`.

Step 1 source selection is controlled with:
- `step1.use_mingo00_param_hash_source` (set `false` to use legacy `collected_data_csv`)
- `step1.simulation_params_csv`
- `step1.mingo00_metadata_root`
- `step1.mingo00_task_ids`
- `step1.trigger_type_rate_column`
- `step1.trigger_type_rate_mode` (`column` or `sum_prefix`)
- `step1.trigger_type_rate_prefix` (used when `sum_prefix` is selected)
- `step1.trigger_type_columns_to_keep`
- `step1.trigger_type_offender_prefix`
- `step1.trigger_type_offender_counts`

For noise-control real-data inputs in Step 5, use:
- `step5.zero_offender_rate_source_task_id`
- `step5.zero_offender_efficiency_source_task_id`
- `step5.zero_offender_scope_preference` (`auto`, `plane_combination_filter`, or `strip_combination_filter`)
- `step5.max_selected_offenders`

When `TASK_3` is used as the Step 5 noise-control source, the selected-offender
thresholds come from `total_problematic_offender_count`, i.e. the cumulative
Task 1 + Task 2 + Task 3 offender count, not just the Task 3 local plane count.

## Main outputs

- `OUTPUTS/FILES/step1_filtered_data.csv`
- `OUTPUTS/PLOTS/step1_parameter_space_overview.png`
- `OUTPUTS/FILES/step2_flux_binned_cells.csv`
- `OUTPUTS/FILES/step2_lut_diagnostics.csv`
- `OUTPUTS/FILES/step2_scale_factor_lut.txt`
- `OUTPUTS/FILES/step3_synthetic_dataset_with_lut.csv`
- `OUTPUTS/FILES/step4_axis_slice_study.csv`
- `OUTPUTS/FILES/step4_pair_slice_study.csv`
- `OUTPUTS/FILES/step5_real_data_with_lut.csv`
- `OUTPUTS/FILES/step5_lut_query_coverage.csv`
- `OUTPUTS/PLOTS/step2_rate_vs_flux.png`
- `OUTPUTS/PLOTS/step2_scale_factor_vs_diagonal_eff.png`
- `OUTPUTS/PLOTS/step2_scale_factor_vs_flux.png`
- `OUTPUTS/PLOTS/step3_rate_correction.png`
- `OUTPUTS/PLOTS/step3_flux_rate_comparison.png`
- `OUTPUTS/PLOTS/step4_axis_slice_scale_factor.png`
- `OUTPUTS/PLOTS/step4_axis_slice_relative_rate.png`
- `OUTPUTS/PLOTS/step4_pair_slice_surface.png`
- `OUTPUTS/PLOTS/step4_pair_slice_quality.png`
- `OUTPUTS/PLOTS/step5_real_rate_correction.png`
- `OUTPUTS/PLOTS/step5_real_correction_diagnostics.png`
