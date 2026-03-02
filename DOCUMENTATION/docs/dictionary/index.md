# Dictionary-based correction and inference

*Last updated: March 2026*

The inference dictionary is the heart of the rapid reconstruction used by the
analysis pipeline.  A large synthetic dataset produced by the digital twin is
binned in the space of detector observables (strip-level rates, coincidences,
efficiencies) and physical parameters (flux, incidence angle, environmental
variables).  The dictionary allows the software to look up the most-likely
flux given a set of measured rates and a previously estimated efficiency.

## Building and validating the dictionary

All work happens under the `MINGO_DICTIONARY_CREATION_AND_TEST/` tree; the
README there contains detailed command examples, but the overall workflow is
as follows:

1. **STEP 1: Build Dictionary**
   - Execute the digital twin over a mesh of physical parameters defined in
     `STEP_0/config_mesh.yaml`.
   - A helper script (`STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/run_build.sh`)
     iterates over the mesh, collects observables from each SIM_RUN, and writes
     a master CSV file (`dictionary.csv`).  Each row corresponds to a
     simulation run and contains fields such as `flux`, `efficiency`,
     `global_rate`, plus a vector of strip rates and coincidence counts.
   - Metadata including the source config hash and parameter ranges are written
     to `dictionary.meta.json`.

2. **STEP 2: Test inference**
   - Use `STEPS/STEP_2_TEST_INFERENCE/evaluate_dictionary.py` to load the
     CSV file and simulate a set of "measurement" rows (either from held‑out
     synthetic runs or from real station data).
   - The script interpolates the dictionary to predict `flux` and `eff`, then
     compares those values to the true inputs, writing out comparison tables
     and residual histograms.
   - A companion tool `STEPS/STEP_2_TEST_INFERENCE/compare_to_truth.py` produces
     summary statistics.

3. **Diagnostics**
   - Coverage plots (`1_2_6_dictionary_coverage.png`) show which regions of
     parameter space are well represented.
   - Efficiency scatter (`1_2_7_scatter_eff_sim_vs_estimated.png`) and other
     figures help identify biases; plot generation scripts live in the
     `OUTPUTS/PLOTS` directory of the corresponding step.
   - Advanced experiments (linear transforms, de-correlation) are recorded in
     the `A_SMALL_SIDE_QUEST/` subdirectories.

Build and validation steps are designed to be idempotent; re‑running them on
an existing dictionary will update it if the underlying simulation or mesh
definition has changed.

## Dictionary structure and sample data

The dictionary CSV is a wide table where each row corresponds to a single
simulation run.  The first columns record physical parameters, the next set of
columns record observable quantities, and trailing columns contain metadata
used for provenance.  A typical header looks like this:

```
flux,efficiency,global_rate,strip_rate_0,strip_rate_1,...,coincidence_01,
angle_x,angle_y,temperature,pressure,config_hash,mesh_tag,sim_run
```

The actual field order may vary depending on the mesh and the version of the
builder script; use `head` or `pandas.read_csv(..., nrows=1)` to inspect a
given file.

```python
import pandas as pd

df = pd.read_csv('dictionary.csv', nrows=5)
print(df.columns.tolist())
print(df.describe())
```

This provides a quick way to check whether the dictionary covers the expected
parameter ranges (see diagnostics below).

## Efficiency scatter plot

The figure below is an example diagnostic produced during STEP 1 that compares
the **simulated** efficiency (horizontal axis) with the **estimated** efficiency
obtained by interpolating the dictionary.  A 1:1 line (black) demonstrates good
agreement across the mesh; deviations highlight parameter combinations where
the dictionary may require refinement.

![Simulated vs estimated efficiency](/assets/dictionary_scatter.png)

_The original plot comes from the file
`MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/OUTPUTS/PLOTS/1_2_7_scatter_eff_sim_vs_estimated.png`.

> **Asset updates:** the set of figures that are synchronised into the
> documentation is controlled by a configuration list stored at
> `DOCUMENTATION/docs/assets/plot_list.txt`.  Add new paths there when you
> introduce additional plots to the site.

## Using and maintaining the dictionary

When a dictionary CSV is placed in the analysis pipeline (typically under
`MASTER/CONFIG_FILES/dictionary/` or copied alongside a processing run), the
helper `MASTER/common/simulated_data_utils.py` exposes the function
`get_flux_from_rates(rates, efficiency, config_hash=None)` which performs the
interpolation.  The same function is used for real station data and for
simulation-based validation runs.

```python
from MASTER.common.simulated_data_utils import get_flux_from_rates

rates = {...}          # strip-level rates dictionary
eff = 0.88             # estimated from coincidence data
flux, eff_est = get_flux_from_rates(rates, eff)
```

The dictionary CSV includes metadata columns (`config_hash`, `mesh_tag`) that
allow the software to automatically detect when the mesh or physics model has
changed.  When significant updates occur (new gas mixture, different FEE
response), generate a new dictionary and tag it with the corresponding hash;
this ensures reproducibility of past analyses.

> **Maintenance checklist:**
> 1. Regenerate dictionary whenever the digital twin configuration or mesh
>    changes.
> 2. Run validation (STEP 2) and inspect diagnostic plots for coverage gaps.
> 3. If any row has missing or NaN values, investigate upstream STEP_1 output.
> 4. Archive previous dictionaries alongside the new one (e.g. under
>    `ARCHIVE/`), since results depend critically on the file version.
>
> **Storage notes:** dictionary files tend to be large (several gigabytes) once
> the mesh contains millions of entries.  Compress them with `gzip` when not
> in active use and keep a checksum (`sha256sum`) in the `.meta.json` for
> sanity checks.  The analysis pipeline can read compressed CSVs directly via
> `pandas`.

---

For workflows involving real data, the analysis pipeline includes a cron entry
that triggers a monthly dictionary refresh and notifies analysts via telegram
if large discrepancies are observed.  See
`OPERATIONS/NOTIFICATIONS/TELEGRAM_BOT` for alert configuration.

> **Note:** keeping the dictionary up to date is critical for maintaining flux
> scale accuracy.  See the `MINGO_DICTIONARY_CREATION_AND_TEST/README.md` for
> additional instructions and example workflows.

---

> **Local preview:** run `mkdocs serve` from the `DOCUMENTATION/` root and
> open `http://127.0.0.1:8000` in your browser to see the rendered site.
