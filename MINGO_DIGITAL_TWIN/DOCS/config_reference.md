# Configuration Reference

This document lists the configuration keys used by each step. Keys are taken from
`config_step_X_physics.yaml` (physics parameters) and `config_step_X_runtime.yaml` (I/O and
execution parameters).

## Common runtime keys
- `input_dir`: base directory for upstream outputs.
- `input_glob`: glob pattern for input files (default varies by step).
- `input_sim_run`: `latest`, `random`, explicit `SIM_RUN_XXXX`, or list (Step FINAL).
- `output_dir`: directory for outputs.
- `output_format`: `pkl` or `csv`.
- `chunk_rows`: number of rows per chunk (enables chunked output).
- `plot_sample_rows`: sample size used for plotting (true = full last chunk).

## STEP 0 (param mesh)
Physics (`config_step_0_physics.yaml`):
- `cos_n`: [min, max] or scalar; zenith exponent sampling range.
- `flux_cm2_min`: [min, max] or scalar; flux sampling range.
- `efficiencies`: [min, max]; efficiency sampling range.
- `efficiencies_identical`: true to use the same efficiency for all planes.
- `repeat_samples`: number of rows added per invocation.
- `shared_columns`: list of columns to hold constant across repeats.
- `expand_z_positions`: true to use all geometries for each sample.

Runtime (`config_step_0_runtime.yaml`):
- `station_config_root`: directory of station config CSVs.
- `output_dir`: target directory for `param_mesh.csv`.

## STEP 1 (muon generation)
Physics:
- `xlim_mm`, `ylim_mm`: generation bounds (half-widths).
- `z_plane_mm`: generation plane z.
- `seed`: RNG seed.
- `c_mm_per_ns`: speed of light in mm/ns (used downstream).
- `cos_n`: angular exponent or `random` to read from param mesh.
- `flux_cm2_min`: flux or `random` to read from param mesh.

Runtime:
- `n_tracks`: number of generated events.
- `output_basename`: file stem (`muon_sample`).
- `param_mesh_dir`, `param_mesh_sim_run`: mesh location for random parameters.

## STEP 2 (plane crossings)
Physics:
- `bounds_mm`: {x_min, x_max, y_min, y_max} acceptance region.
- `z_positions`: [z1, z2, z3, z4] or `random` to read from param mesh.

Runtime:
- `input_basename`: overrides input file stem (optional).
- `normalize_to_first_plane`: subtract z1 from all planes.
- `param_mesh_dir`, `param_mesh_sim_run`: mesh location for random z positions.

## STEP 3 (avalanche)
Physics:
- `avalanche_gap_mm`: gas gap size.
- `avalanche_gain`: used for internal qsum (not persisted).
- `townsend_alpha_per_mm`: Townsend coefficient.
- `avalanche_electron_sigma`: lognormal sigma for avalanche size.
- `efficiencies`: [e1,e2,e3,e4], list of vectors, or `random` from mesh.

Runtime:
- `param_mesh_dir`, `param_mesh_sim_run`: mesh location for efficiencies.

## STEP 4 (induction)
Physics:
- `charge_share_points`: binomial samples per event.
- `x_noise_mm`: Gaussian noise on X.
- `time_sigma_ns`: Gaussian noise on `T_sum_meas`.
- `width_scale_exponent`: power-law for avalanche width scaling.
- `width_scale_max`: cap on width scale factor.
- `avalanche_width_mm`: base induction width.

## STEP 5 (signal observables)
Physics:
- `qdiff_frac`: fractional sigma for charge imbalance.
- `c_mm_per_ns`: optional override for T_diff conversion.

## STEP 6 (front/back)
Physics: none.

## STEP 7 (cable offsets)
Physics:
- `tfront_offsets`: 4x4 offsets for front times (ns).
- `tback_offsets`: 4x4 offsets for back times (ns).

## STEP 8 (FEE threshold)
Physics:
- `t_fee_sigma_ns`: per-channel Gaussian time jitter.
- `charge_threshold`: minimum Q to keep a channel active.
- `q_to_time_factor`: Q to time-walk conversion factor.
- `qfront_offsets`, `qback_offsets`: per-channel offsets (ns).

## STEP 9 (trigger)
Physics:
- `trigger_combinations`: list of plane strings (e.g., ["12", "23"]).

## STEP 10 (DAQ jitter)
Physics:
- `jitter_width_ns`: event-level uniform jitter width.
- `tdc_sigma_ns`: per-channel Gaussian smear.

## STEP FINAL (event builder)
Runtime:
- `input_collect`: `matching` (config/upstream hash match) or `baseline_only`.
- `target_rows`: number of events to sample.
- `files_per_station_conf`: number of .dat files to emit per param set.
- `payload_sampling`: `random` or `sequential_random_start`.
- `rate_hz`: Poisson inter-arrival rate when `T_thick_s` is absent.
- `param_mesh_dir`, `param_mesh_sim_run`: mesh location for param_set_id assignment.
