# Simulated Validation Demo

This report demonstrates dictionary-based point estimation with uncertainty
using simulated data only (self-consistency validation).

- Input all-samples results: `STEP_4_SELF_CONSISTENCY/output/poisson/all_samples_results.csv`
- Input LUT directory: `STEP_6_UNCERTAINTY_LUT/output/poisson/lut`
- Exact self-matches removed for conservative uncertainty check: `0` rows

## Coverage Summary (All Successful Samples)

- n samples: **186**
- Flux coverage @1σ: **69.4%**
- Efficiency coverage @1σ: **65.1%**
- Joint coverage @1σ: **53.8%**

## Coverage Summary (Conservative: Excluding Exact Self-Matches)

- n samples: **186**
- Flux coverage @1σ: **69.4%**
- Efficiency coverage @1σ: **65.1%**
- Joint coverage @1σ: **53.8%**
- Flux coverage @2σ: **88.2%**
- Efficiency coverage @2σ: **89.8%**

## Coverage After Sigma Calibration (Conservative Subset)

- Target 1σ coverage: **68.0%**
- Applied scale factors: flux **x0.956**, efficiency **x1.061**
- Flux coverage @1σ (calibrated): **67.7%**
- Efficiency coverage @1σ (calibrated): **67.7%**

## Demo Points

Representative points with estimates and uncertainty intervals are in: `STEP_7_SIMULATED_DEMO/output/poisson/demo_points_with_uncertainty.csv`

Use these rows directly in slides as concrete examples of:
- point estimate (estimated flux / estimated efficiency)
- uncertainty interval (±σ from LUT)
- truth-in-interval check (`inside_*_1sigma`)
