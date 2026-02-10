# Simulated Validation Demo

This report demonstrates dictionary-based point estimation with uncertainty
using simulated data only (self-consistency validation).

- Input all-samples results: `STEP_4_SELF_CONSISTENCY/output/chi2/all_samples_results.csv`
- Input LUT directory: `STEP_6_UNCERTAINTY_LUT/output/chi2/lut`
- Exact self-matches removed for conservative uncertainty check: `0` rows

## Coverage Summary (All Successful Samples)

- n samples: **186**
- Flux coverage @1σ: **56.5%**
- Efficiency coverage @1σ: **63.4%**
- Joint coverage @1σ: **44.6%**

## Coverage Summary (Conservative: Excluding Exact Self-Matches)

- n samples: **186**
- Flux coverage @1σ: **56.5%**
- Efficiency coverage @1σ: **63.4%**
- Joint coverage @1σ: **44.6%**
- Flux coverage @2σ: **78.5%**
- Efficiency coverage @2σ: **82.8%**

## Coverage After Sigma Calibration (Conservative Subset)

- Target 1σ coverage: **68.0%**
- Applied scale factors: flux **x1.399**, efficiency **x1.187**
- Flux coverage @1σ (calibrated): **67.7%**
- Efficiency coverage @1σ (calibrated): **67.7%**

## Demo Points

Representative points with estimates and uncertainty intervals are in: `STEP_7_SIMULATED_DEMO/output/chi2/demo_points_with_uncertainty.csv`

Use these rows directly in slides as concrete examples of:
- point estimate (estimated flux / estimated efficiency)
- uncertainty interval (±σ from LUT)
- truth-in-interval check (`inside_*_1sigma`)
