# Projection Axis Mismatch

This study is the bridge between:

- Task 4 `xp/yp` ellipse asymmetry
- Task 2 calibrated `T_dif` widths
- A strip-balance sanity check on the strip-assignment side

The point is to distinguish whether the observed Task 4 `x/y` mismatch is more
consistent with:

- an `x` / `T_dif` / active-length scaling problem, or
- a `y` / strip-assignment / strip-population problem

Run:

```bash
python3 MASTER/ANCILLARY/SIMULATION_TUNING/PROJECTION_AXIS_MISMATCH/study_projection_axis_mismatch.py
```

Outputs are written to `OUTPUTS/` in this directory.

The study now also writes a per-file Task 4 time series, using the simulation
distribution as the reference band, so it is easy to check whether the mismatch
is systematic across files or dominated by a few runs.
