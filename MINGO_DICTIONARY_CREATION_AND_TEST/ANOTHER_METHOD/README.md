# Another Method

This small project builds a scale-factor LUT from simulation parameters plus
MINGO00 metadata (joined by `param_hash`), applies it to `synthetic_dataset.csv`,
and then applies it to a configurable slice of real station data. The metadata
source can be either the existing `trigger_type` CSVs or the newer
`robust_efficiency` CSVs.

## Workflow

1. `step_1_prepare_data.py`
   - Reads `step_final_simulation_params.csv`.
   - Reads the selected MINGO00 metadata source from `STAGE_1/EVENT_DATA/STEP_1`.
   - Uses `task_<id>_metadata_trigger_type.csv` when
     `trigger_type_selection.metadata_source = "trigger_type"`.
   - Uses `TASK_4/task_4_metadata_robust_efficiency.csv` when
     `trigger_type_selection.metadata_source = "robust_efficiency"`.
   - Joins simulation parameters and real observed rates by `param_hash`.
   - Reads only the required metadata columns and keeps the selected source-rate
     columns in the Step 1 output CSV.
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
     vector, followed by the quantized efficiency-bin coordinates
     `emp_eff_1 emp_eff_2 emp_eff_3 emp_eff_4 scale_factor`.

3. `step_3_apply_lut.py`
   - By default reads the Step 1 filtered training table so the Step 3 input is
     coherent with the same metadata/rate selection used to build the LUT.
   - Can still read the legacy external `synthetic_dataset.csv` when
     `step3.input_source = "legacy_synthetic_dataset"`.
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
   - Uses the same metadata source configured in `trigger_type_selection`.
   - For `trigger_type`, task / stage / offender threshold are taken from the
     same config block.
   - For `robust_efficiency`, the source is fixed to
     `TASK_4/task_4_metadata_robust_efficiency.csv` and only the rate choice
     matters: `rate_1234_hz` or `rate_total_hz`.
   - Lets you choose how the 4-D efficiency query vector is built:
     `minimal_empirical` keeps the four plane efficiencies as-is, while
     `same_efficiency` averages a chosen plane set and queries the LUT with
     `[eff, eff, eff, eff]`.
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
window, edit `config.json`. The rate input is controlled through
`trigger_type_selection`, and the fallback behavior is controlled with
`step3.lut_match_mode` / `step5.lut_match_mode` (`exact`,
`nearest`, or `interpolate`). The interpolation hyperparameters
`step3.lut_interpolation_k` and `step3.lut_interpolation_power` are shared by
both Step 3 and Step 5.

Metadata source selection is controlled with:
- `trigger_type_selection.metadata_source` (`trigger_type` or `robust_efficiency`)
- `trigger_type_selection.rate_family`
  - `trigger_type`: supports `total`, `four_plane`, `three_plane`, `two_plane`,
    `three_and_four_plane`, and `two_and_three_plane`
  - `robust_efficiency`: supports `1234` / `four_plane`, `four_plane_robust_hz`, and `total`
  - when `metadata_source = "robust_efficiency"`, you can also choose
    `trigger_type_selection.robust_efficiency_variant`:
    - `default` -> `eff1`, `eff2`, `eff3`, `eff4`
    - `plateau` -> `eff1_plateau`, ...
    - `overall` -> `eff1_overall`, ...
    - `median_x` -> `eff1_median_x`, ...
- `trigger_type_selection.task_id`
- `trigger_type_selection.stage_prefix`
- `trigger_type_selection.offender_threshold`

When `metadata_source = "robust_efficiency"`, the task, stage, and offender
threshold are ignored. The code always reads
`TASK_4/METADATA/task_4_metadata_robust_efficiency.csv` and only switches
between `rate_1234_hz`, `four_plane_robust_hz`, and `rate_total_hz`.

Step 1 source paths are controlled with:
- `step1.use_mingo00_param_hash_source` (set `false` to use legacy `collected_data_csv`)
- `step1.simulation_params_csv`
- `step1.metadata_root` (or the older `step1.trigger_type_metadata_root`)

Step 5 real-data slicing is controlled with:
- `step5.station`
- `step5.date_from`
- `step5.date_to`
- `step5.min_events`
- `step5.metadata_agg`
- `step5.timestamp_column`
- `step5.selected_feature_columns_mode`
  - `minimal_empirical` -> use `[emp_eff_1, emp_eff_2, emp_eff_3, emp_eff_4]`
  - `same_efficiency` -> average `step5.same_efficiency_planes` and use
    `[eff, eff, eff, eff]`
- `step5.same_efficiency_planes`

Step 3 synthetic-input selection is controlled with:
- `step3.input_source` (`training_dataset` or `legacy_synthetic_dataset`)

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
