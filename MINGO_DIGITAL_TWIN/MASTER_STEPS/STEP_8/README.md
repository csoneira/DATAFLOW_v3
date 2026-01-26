# STEP 8 (Threshold / FEE)

Purpose:
- Apply front-end jitter, charge-to-time conversion, and thresholding.

Inputs:
- Physics config: `config_step_8_physics.yaml`
- Runtime config: `config_step_8_runtime.yaml`
- Data: `INTERSTEPS/STEP_7_TO_8/SIM_RUN_<N>/step_7.(pkl|csv|chunks.json)`

Outputs:
- `INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/step_8.(pkl|csv|chunks.json)`
- `INTERSTEPS/STEP_8_TO_9/SIM_RUN_<N>/PLOTS/step_8_plots.pdf`

Algorithm highlights:
- `T_front` and `T_back` receive Gaussian jitter (`t_fee_sigma_ns`).
- Q values are converted to time-walk units using `q_to_time_factor` and per-channel offsets.
- Channels below `charge_threshold` are set to 0.

Run:
```
python3 step_8_uncalibrated_to_threshold.py --config config_step_8_physics.yaml
python3 step_8_uncalibrated_to_threshold.py --config config_step_8_physics.yaml --runtime-config config_step_8_runtime.yaml
python3 step_8_uncalibrated_to_threshold.py --config config_step_8_physics.yaml --plot-only
```

Notes:
- The step skips if the target SIM_RUN exists unless `--force` is provided.
- After this step, Q values are no longer raw charge units.
