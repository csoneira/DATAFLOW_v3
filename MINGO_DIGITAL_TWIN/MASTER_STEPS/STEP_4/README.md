# STEP 4 (Avalanche -> Hit)

Purpose:
- Induce strip signals from avalanche centroids and sizes (readout coupling begins).

Inputs:
- Physics config: `config_step_4_physics.yaml`
- Runtime config: `config_step_4_runtime.yaml`
- Data: `INTERSTEPS/STEP_3_TO_4/SIM_RUN_<N>/step_3.(pkl|csv|chunks.json)`

Outputs:
- `INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/step_4.(pkl|csv|chunks.json)`
- `INTERSTEPS/STEP_4_TO_5/SIM_RUN_<N>/PLOTS/step_4_plots.pdf`

Algorithm highlights:
- Avalanche electrons are converted to gap charge in `fC`, then scaled to induced charge with `induced_charge_fraction`.
- An isotropic 2D Lorentzian is centered at the avalanche point.
- Strip charges are computed from exact rectangle integrals of that Lorentzian inside the detector x-range and each strip y-band.
- The physical width parameter is `lorentzian_gamma_mm`.
- `X_mea` and `T_sum_meas` receive Gaussian noise.

Run:
```
python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml
python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --runtime-config config_step_4_runtime.yaml
python3 step_4_hit_to_measured.py --config config_step_4_physics.yaml --plot-only
```

Notes:
- The step skips if the target SIM_RUN exists unless `--force` is provided.
- Output columns are per-plane per-strip: `Y_mea`, `X_mea`, `T_sum_meas`.
