# Inference and Dictionary Workflow

Dictionary-based inference links detector observables to physical parameters such as flux and efficiency.

## Purpose

- Use synthetic data from the digital twin to build lookup tables.
- Apply those lookup tables to measured station rates.
- Keep inference aligned with current simulation assumptions and hashes.

## Main workflow

1. Build dictionary from simulation outputs over a parameter mesh.
2. Validate interpolation quality on held-out synthetic or real samples.
3. Deploy dictionary artifacts for analysis-stage inference.
4. Regenerate dictionary whenever relevant simulation/config assumptions change.

```mermaid
flowchart LR
    A[Digital twin outputs] --> B[Dictionary build]
    B --> C[Validation and residual analysis]
    C --> D[Deploy dictionary artifact]
    D --> E[MASTER inference usage]
    E --> F[Monitor drift and refresh]
    F --> B
```

## Main code locations

- Build/test workflows: `MINGO_DICTIONARY_CREATION_AND_TEST/`
- Runtime usage in pipeline: `MASTER/common/simulated_data_utils.py`

## Operational notes

- Dictionary files can become large; versioning and checksums are required.
- Keep dictionary metadata tied to simulation hashes and mesh tags.
- Treat dictionary refresh as a pipeline-impacting update.

## Repository diagnostic figures

![Decorrelated dictionary timeseries diagnostic](/assets/repository_figures/decorrelation_original_timeseries.png)

![Decorrelated dictionary scatter diagnostic](/assets/repository_figures/decorrelation_original_scatters.png)

Source paths:
- `MINGO_DICTIONARY_CREATION_AND_TEST/A_SMALL_SIDE_QUEST/TRYING_DECORRELATION/PLOTS/01_original_timeseries.png`
- `MINGO_DICTIONARY_CREATION_AND_TEST/A_SMALL_SIDE_QUEST/TRYING_DECORRELATION/PLOTS/02_original_scatters.png`

## Reference pages

- Existing dictionary page in this site: [Legacy Dictionary Notes](../dictionary/index.md)
- Simulation source context: [Simulation Pipeline](simulation-pipeline.md)
