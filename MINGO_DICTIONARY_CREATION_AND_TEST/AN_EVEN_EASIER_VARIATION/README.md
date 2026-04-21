# AN_EVEN_EASIER_VARIATION

This variation is a simplified real-data collector.

## Purpose

Step 1 reads station metadata exactly as the `ANOTHER_METHOD` step 5 workflow does and writes a compact CSV containing:

- `selected_rate_hz`
- `eff_empirical_1`
- `eff_empirical_2`
- `eff_empirical_3`
- `eff_empirical_4`

It also preserves `filename_base`, `file_timestamp_utc`, and `execution_timestamp_utc` when available.
The metadata source can be either the existing `trigger_type` CSVs or the newer
`robust_efficiency` CSVs.

Step 2 reads that output, calculates the mean of the configured empirical-efficiency
planes, assumes that mean is the same efficiency for all four planes, applies it
to `selected_rate_hz`, and plots the original/corrected rate time series plus the
selected efficiency series and their mean reference.

## Run

```bash
cd /home/mingo/DATAFLOW_v3/MINGO_DICTIONARY_CREATION_AND_TEST/AN_EVEN_EASIER_VARIATION
python3 run_all.py
```

## Configuration

Edit `config.json` to change the station, date window, or selected rate definition.

`trigger_type_selection.metadata_source` controls which metadata file is read:

- `trigger_type`: reads `task_<id>_metadata_trigger_type.csv`
- `robust_efficiency`: reads `TASK_4/task_4_metadata_robust_efficiency.csv`

When `metadata_source = "robust_efficiency"`, the task/stage/offender settings
are ignored and only the rate choice matters:

- `rate_family = "four_plane"` or `rate_family = "1234"` uses `rate_1234_hz`
- `rate_family = "four_plane_robust_hz"` uses `four_plane_robust_hz`
- `rate_family = "total"` uses `rate_total_hz`

Step 2 scaling is controlled with:

- `step2.efficiency_reference_planes`

Example:

- `[3]` uses only `eff_empirical_3`
- `[2, 3]` uses the mean of `eff_empirical_2` and `eff_empirical_3`
