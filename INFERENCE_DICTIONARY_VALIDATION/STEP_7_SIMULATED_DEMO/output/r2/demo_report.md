# Simulated Validation Demo

This report demonstrates dictionary-based point estimation with uncertainty
using simulated data only (self-consistency validation).

- Input all-samples results: `STEP_4_SELF_CONSISTENCY/output/r2/all_samples_results.csv`
- Input LUT directory: `STEP_6_UNCERTAINTY_LUT/output/r2/lut`
- Exact self-matches removed for conservative uncertainty check: `0` rows

## Coverage Summary (All Successful Samples)

- n samples: **182**
- Flux coverage @1σ: **50.0%**
- Efficiency coverage @1σ: **57.1%**
- Joint coverage @1σ: **35.7%**

## Coverage Summary (Conservative: Excluding Exact Self-Matches)

- n samples: **182**
- Flux coverage @1σ: **50.0%**
- Efficiency coverage @1σ: **57.1%**
- Joint coverage @1σ: **35.7%**
- Flux coverage @2σ: **76.4%**
- Efficiency coverage @2σ: **78.0%**

## Coverage After Sigma Calibration (Conservative Subset)

- Target 1σ coverage: **68.0%**
- Applied scale factors: flux **x1.449**, efficiency **x1.318**
- Flux coverage @1σ (calibrated): **68.1%**
- Efficiency coverage @1σ (calibrated): **68.1%**

## Demo Points

Representative points with estimates and uncertainty intervals are in: `STEP_7_SIMULATED_DEMO/output/r2/demo_points_with_uncertainty.csv`

Use these rows directly in slides as concrete examples of:
- point estimate (estimated flux / estimated efficiency)
- uncertainty interval (±σ from LUT)
- truth-in-interval check (`inside_*_1sigma`)
