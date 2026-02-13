# PURELY_LINEAR Method Guide

## Purpose
`try_purely_linear_transform.py` calibrates a **simple affine inverse** to estimate flux from:
- `global_rate`
- `eff` (efficiency)

The intent is to keep the model explicit, stable, and easy to apply row-by-row.

---

## Intended Use
Use this script when you want a compact approximation of the inverse mapping:

`(global_rate, eff) -> (flux, eff)`

with one fixed matrix + offset.

This is useful for:
- fast correction of a time series where `global_rate` and `eff` are known,
- transparent validation against simulated truth,
- exporting a formula that can be applied outside Python.

---

## Method (what it does)
Given a training dictionary with `(flux, eff, global_rate)`:

1. Estimate local gradients of `global_rate` in `(flux, eff)` space:
   - `dRate/dFlux`
   - `dRate/dEff`

2. Build:
   - `tan(angle) = (dRate/dEff) / (dRate/dFlux)`

3. Keep only points near the median `tan(angle)`:
   - robust median + MAD threshold
   - this enforces the “locally near-parallel iso-rate lines” assumption

4. Fit one affine forward model on kept points:
   - `global_rate = a*flux + b*eff + c`

5. Invert analytically:
   - `flux = (global_rate - b*eff - c)/a`
   - equivalent:

```text
[flux]   [1/a   -b/a] [global_rate] + [-c/a]
[ eff] = [ 0      1 ] [   eff    ] + [  0 ]
```

No iterative solver is needed.

---

## Run
From anywhere:

```bash
python3 /home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/A_SMALL_SIDE_QUEST/TRYING_LINEAR_TRANSFORMATIONS/PURELY_LINEAR/try_purely_linear_transform.py
```

---

## Inputs
Configured inside script:
- Training dictionary candidates:
  - `.../TRYING_LINEAR_TRANSFORMATIONS/dictionary_test.csv`
  - `.../STEP_1_2_BUILD_DICTIONARY/OUTPUTS/FILES/dictionary.csv`
- Target series candidates:
  - `.../A_SMALL_SIDE_QUEST/the_simulated_file.csv`
  - `.../PURELY_LINEAR/dictionary_test.csv`

Column selection is automatic (with strict fallbacks):
- Flux: `flux` or `flux_cm2_min`
- Efficiency: prefer simulated efficiency (`eff_sim_1` / `eff`) for this workflow
- Global rate: `global_rate_hz` or fallback rate columns

---

## Outputs
Generated in `PURELY_LINEAR/PLOTS`:
- `00_linear_summary.txt`: fitted coefficients, matrix, formula, metrics
- `02_local_gradients_quiver.png`: iso-rate map + gradient vectors + kept/rejected points
- `04_tan_angle_histogram.png`: tan(angle) filtering diagnostic
- `05_validation_scatter_flux.png`: estimated vs true flux
- `06_flux_timeseries_validation.png`: time-series comparison
- `08_linearized_predictions.csv`: row-level predictions and errors

---

## How to Apply the Result
After running, read `00_linear_summary.txt` and use:

```text
flux_est = m11*global_rate + m12*eff + t1
eff_out  = eff
```

where:
- `m11 = 1/a`
- `m12 = -b/a`
- `t1  = -c/a`

Apply this for each row of your target data.
