# Efficiency Vector Tuning

This study compares the Task 4 track-based efficiency vectors already written
to `task_4_metadata_efficiency.csv` for:

- `x`
- `y`
- `theta`

The goal is to decide whether the current efficiency modelling is already good
enough or whether it still needs tuning, with extra attention to bins where the
efficiency is low and mismodelling is usually most visible.

Run:

```bash
python3 MASTER/ANCILLARY/SIMULATION_TUNING/EFFICIENCY_VECTOR_TUNING/study_efficiency_vector_tuning.py
```

Outputs are written to `OUTPUTS/` in this directory.

The script compares the latest available Task 4 metadata rows per
`filename_base`, aggregates the simulation and real-data groups separately, and
reports:

- median and interquartile efficiency curves by plane and axis
- `REAL - SIM` discrepancy curves
- fiducial baseline matches in `x`, `y`, and `theta`
- outside-fiducial residuals after the best non-position-dependent baseline fit
- a plane/axis summary stating whether a uniform or global baseline is enough,
  or whether explicit position dependence is still required
- a per-file time series where each real-data curve is matched to the
  simulation dictionary in the fiducial region and then evaluated in the outer
  bins
- a text report stating whether the current modelling looks acceptable or still
  needs tuning
