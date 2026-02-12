# STEP 7 — Simulated Demo

Generate presentation-ready evidence that the dictionary method:

1. recovers point estimates on simulated samples, and
2. assigns uncertainty intervals with measurable empirical coverage.
3. reports a calibrated uncertainty scale so 1σ coverage matches a target.

## Run

```bash
python3 INFERENCE_DICTIONARY_VALIDATION/STEP_7_SIMULATED_DEMO/simulated_uncertainty_demo.py
```

Optional:

```bash
python3 INFERENCE_DICTIONARY_VALIDATION/STEP_7_SIMULATED_DEMO/simulated_uncertainty_demo.py \
  --config INFERENCE_DICTIONARY_VALIDATION/STEP_7_SIMULATED_DEMO/config.json
```

## Main outputs

- `output/demo_report.md`
- `output/demo_summary.json`
- `output/demo_points_with_uncertainty.csv`
- `output/all_samples_with_lut_uncertainty.csv`
- `output/coverage_by_events.csv`
- `output/coverage_nsigma.png`
- `output/error_over_sigma_hist.png`
- `output/demo_points_true_vs_est.png`

`demo_summary.json` includes both:

- raw coverage from the LUT
- calibrated coverage after multiplying sigma by fitted scale factors
