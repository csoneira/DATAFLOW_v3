# TDIF Width Comparison: Action Needed In The Simulation

This is the short memo with the updated numbers after adding the per-strip
shape plots, derivative plots, and histogram-fit scaling.

## Best Practical Recommendation

If you want one strip length to try in the simulation right now, use:

`270 mm`

That is the most practical value after combining:

- the robust width-based estimate
- the visual shape-based histogram fit

## The Two Important Scale Factors

1. Robust width factor from `q95(|T_dif|)`:

`0.902368`

If the current strip length is `300 mm`:

`300 mm * 0.902368 = 270.71 mm`

2. Shape-fit factor from fitting the real `|T_dif|` histogram with
`alpha * simulated`:

`0.889375`

If the current strip length is `300 mm`:

`300 mm * 0.889375 = 266.81 mm`

## Why `270 mm` Is The Right First Test

The two good methods now bracket the answer as:

- width-based: about `270.7 mm`
- shape-based: about `266.8 mm`

So the clean practical midpoint is:

`270 mm`

This is also closer to the visual “one square-like distribution is wider than
the other” interpretation than the previous single-number estimate.

## What Should Be Trusted And What Should Not

### Good diagnostics

- `q95(|T_dif|)` ratio
- histogram-fit `alpha`

### Not good as the final correction

- derivative-edge ratio alone

Why not:

- the derivative plot is useful to see the falling edge
- but in several strips it is visibly noisier and less stable
- it underestimates the scale too much in some cases

So:

- use `0.902368` if you want the robust summary
- use `0.889375` if you want the best visual shape match
- use `270 mm` if you want the single practical simulation value

## Main Results

- `std` ratio real/sim: `0.899467`
- `q90(|T_dif|)` ratio real/sim: `0.893729`
- `q95(|T_dif|)` ratio real/sim: `0.902368`
- `q97.5(|T_dif|)` ratio real/sim: `0.916328`
- `q99(|T_dif|)` ratio real/sim: `0.941354`
- derivative-edge ratio real/sim: `0.860411`
- histogram-fit alpha: `0.889375`

Stable conclusion:

- the simulated `T_dif` core is too wide by about `10%`
- the simulation should use a strip length about `9%` to `10%` smaller

## Files To Open

- Main report: [tdif_span_report.txt](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_span_report.txt)
- Main summary plot: [tdif_span_comparison.png](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_span_comparison.png)
- Scale-factor summary: [tdif_scale_factor_summary.png](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_scale_factor_summary.png)
- Per-strip signed overlays: [tdif_per_strip_signed_overlay.png](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_per_strip_signed_overlay.png)
- Per-strip abs overlays with scaled simulation: [tdif_per_strip_abs_overlay.png](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_per_strip_abs_overlay.png)
- Per-strip derivative plots: [tdif_per_strip_abs_derivative.png](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_per_strip_abs_derivative.png)
- Per-strip signed-derivative Gaussian fits: [tdif_per_strip_signed_derivative_gaussian_fit.png](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/tdif_per_strip_signed_derivative_gaussian_fit.png)
- Summary table: [recommended_scale_summary.csv](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/recommended_scale_summary.csv)
- Per-strip table: [per_strip_tdif_summary.csv](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/OUTPUTS/per_strip_tdif_summary.csv)
- Reproducible script: [compare_task2_tdif_span.py](/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/TDIF_SPAN_COMPARISON/compare_task2_tdif_span.py)

## Final Instruction

If you want one number only, put:

`270 mm`

If you want the factors explicitly:

- robust width factor: `0.907381`
- robust width factor: `0.902368`
- shape-fit factor: `0.889375`
