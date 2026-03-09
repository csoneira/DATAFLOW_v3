# INFERENCE_DICTIONARY_VALIDATION

*We do simulations like this: we choose a point in the parameter space, generate a dataset simulated with those parameters, then we analyze the resulting datafile and that analysis outputs a point in the feature space. But since we know the origina parameter set, we have a bijective relation established from parameter space to feature space. So, for a file that we dont know the parameter set that generated it, the idea is that we calculate distances in the feature space, maybe with 1 or with n neighbours, or with all the neighbours, then we use that set of results to calcualte to go back to the parameter space and, in that space, calculate a representative point for the tested entry. It's the concept of a continous function: if we know where the points around a point go, we can estimate where that point goes. in this case the function has its domain in the feature space and goes to the parameter (of simulation space).*

*In other words. We construct a mapping between simulation parameter space and analysis feature space by generating simulated datasets at selected points in the parameter space and processing each dataset through the same analysis chain used for real data. Each simulation therefore produces a corresponding point in feature space, while its generating parameter vector is known by construction. In this way, the simulation set defines a sampled correspondence from parameters to features, and, equivalently, an empirical inverse relation from features back to parameters. For an input data file whose generating parameters are unknown, we first extract its feature-space coordinates and then compare this point with the simulated entries in feature space, for example using the single nearest neighbour, the n nearest neighbours, or a weighted combination of all neighbours. The associated parameter-space points of those neighbouring simulations are then used to infer the most representative parameter vector for the unknown file. The underlying idea is continuity: if nearby points in feature space arise from nearby points in parameter space, then the local structure of the simulated sample can be used to estimate the parameter-space origin of a previously unseen observation.*

Validation pipeline for the MINGO simulation-based inference method.
Given a dictionary of simulated detector responses, the pipeline tests
whether the matching procedure can reliably recover the physical
parameters (cosmic-ray flux and detection efficiency) from topology-count
observables alone.

## Pipeline overview

```
STEP 1  Build Dictionary         builds the param-metadata dictionary CSV
                                  from simulated data files

STEP 2  Validate Simulation      compares topology-estimated efficiencies
                                  against the simulation truth per plane

STEP 3  Relative Error           computes relative errors, applies quality
                                  cuts, produces a filtered reference CSV

STEP 4  Self-Consistency         matches every data sample against the
                                  dictionary and recovers (flux, eff)

STEP 5  Uncertainty Limits       calibrates uncertainty curves and
                                  dictionary-coverage diagnostics

STEP 6  Uncertainty LUT          builds a 3-D empirical uncertainty
                                  look-up table for inference

STEP 7  Simulated Demo           generates demo artifacts with point
                                  estimates and uncertainty coverage checks
```

### Shared libraries (not steps — imported by the scripts above)

| File | Purpose |
|---|---|
| `MODULES/simulation_validation_utils.py` | Common helpers: logging, config, parsing, scoring, plotting, geometry, uncertainty |
| `uncertainty_lut.py` | LUT loader + trilinear interpolation class (used by steps 6 & 7) |

## Directory layout

Each step lives in its own directory (`STEP_N_<NAME>/`) and writes outputs
into an `OUTPUTS/` subdirectory that is created automatically:

```
STEP_N_<NAME>/
  script.py
  config_method.json     (optional method config)
  config_plots.json      (optional plotting config)
  config_runtime.json    (optional runtime/path overrides)
  OUTPUTS/
    FILES/               CSVs, JSONs, markdown reports
    PLOTS/               PNG figures
```

## Running

### Full pipeline (steps 1–6)

```bash
./run_pipeline.sh              # all steps
./run_pipeline.sh 3 4 6        # only steps 3, 4, 6
./run_pipeline.sh --from 4     # steps 4, 5, 6
```

### Metric comparison (steps 1–3 + per-metric 4→6→7)

```bash
./run_metric_comparison.sh                 # all 4 metrics (l2, chi2, poisson, r2)
./run_metric_comparison.sh l2 chi2         # selected metrics
./run_metric_comparison.sh --skip-shared l2
```

Results are collected in `COMPARISON/metric_comparison.csv`.

## Configuration

Each step reads parameters in this priority order:

```
CLI argument  >  config_runtime.json override  >  config_plots.json key  >  config_method.json key  >  hardcoded default
```

Main configuration files at repo root:

- `config_method.json`: inference and processing method knobs
- `config_plots.json`: plotting/display knobs
- `config_runtime.json`: runtime path overrides
- `config_legacy.json`: deprecated keys kept only as reference

`config.json` has been retired.

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib
- scipy (optional — used for griddata interpolation and distance transforms)
