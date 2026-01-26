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
- Avalanche size sets an effective induction width scaled by a power law.
- A uniform disk model is used to compute strip overlap fractions.
- Charge sharing is sampled with `charge_share_points` binomial draws.
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
