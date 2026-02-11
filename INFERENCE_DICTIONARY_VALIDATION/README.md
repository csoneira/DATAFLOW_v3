# INFERENCE_DICTIONARY_VALIDATION

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
| `msv_utils.py` | Common helpers: logging, config, parsing, scoring, plotting, geometry, uncertainty |
| `uncertainty_lut.py` | LUT loader + trilinear interpolation class (used by steps 6 & 7) |

## Directory layout

Each step lives in its own directory (`STEP_N_<NAME>/`) and writes outputs
into an `OUTPUTS/` subdirectory that is created automatically:

```
STEP_N_<NAME>/
  script.py
  config.json            (optional step-specific config)
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
CLI argument  >  config.json key  >  hardcoded default
```

Step-specific `config.json` files are in each step directory (step 1 uses
`config/pipeline_config.json`).  Paths in these files are absolute; update
them if you move the repository.

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib
- scipy (optional — used for griddata interpolation and distance transforms)
