# Simulated Validation Demo

This report demonstrates dictionary-based point estimation with uncertainty
using simulated data only (self-consistency validation).

- Input all-samples results: `/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_4_SELF_CONSISTENCY/output/all_samples_results.csv`
- Input LUT directory: `/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_6_UNCERTAINTY_LUT/output/lut`
- Exact self-matches removed for conservative uncertainty check: `0` rows

## Coverage Summary (All Successful Samples)

- n samples: **182**
- Flux coverage @1σ: **57.1%**
- Efficiency coverage @1σ: **53.8%**
- Joint coverage @1σ: **36.3%**

## Coverage Summary (Conservative: Excluding Exact Self-Matches)

- n samples: **182**
- Flux coverage @1σ: **57.1%**
- Efficiency coverage @1σ: **53.8%**
- Joint coverage @1σ: **36.3%**
- Flux coverage @2σ: **83.0%**
- Efficiency coverage @2σ: **78.0%**

## Coverage After Sigma Calibration (Conservative Subset)

- Target 1σ coverage: **68.0%**
- Applied scale factors: flux **x1.259**, efficiency **x1.546**
- Flux coverage @1σ (calibrated): **68.1%**
- Efficiency coverage @1σ (calibrated): **68.1%**

## Demo Points

Representative points with estimates and uncertainty intervals are in: `/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_7_SIMULATED_DEMO/output/demo_points_with_uncertainty.csv`

Use these rows directly in slides as concrete examples of:
- point estimate (estimated flux / estimated efficiency)
- uncertainty interval (±σ from LUT)
- truth-in-interval check (`inside_*_1sigma`)
