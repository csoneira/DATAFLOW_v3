Below is a refined, highly specific prompt you can paste directly to Codex/Claude. It enforces clear deliverables, constraints, and a validation architecture that is compatible with your pipeline design (step registries, metadata, chunked I/O, and row-event intersteps). It also includes an explicit outline for what `VALIDATION/validation.md` should contain and what scripts must implement.

---

## Prompt to Codex / Claude

You are working inside:

* `/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/`
* You must create a validation suite under:

  * `/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/VALIDATION/`

### Objective

Create **read-only** validation scripts that test the correctness, internal consistency, and basic physics/electronics plausibility of the **MINGO_DIGITAL_TWIN** pipeline outputs using the existing intermediate artifacts stored under:

* `/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/`

The validation must produce:

1. **Machine-readable CSV reports** for each validation run (for periodic regression checking).
2. Optional **plots** generated only when a plot flag is requested (`-p/--plot`).
3. A filled and well-structured `VALIDATION/validation.md` describing the validation strategy, what each validator checks, and how to interpret the reports.

Important: **INTERSTEPS contains row-event level outputs, not precomputed counts/rates**. The validators must compute derived quantities (counts, efficiencies, residuals, rates, distributions) from the row-event tables and metadata.

### Hard constraints

* **Do not modify** anything under `INTERSTEPS/` or other pipeline outputs. Validation is read-only.
* Use existing `sim_run_registry.json` and metadata sidecars/manifests to discover the latest run and upstream lineage.
* Must support both single-file outputs (`.pkl`, `.csv`) and chunked outputs (`*.chunks.json`).
* The validation scripts must be robust to missing optional fields and should fail gracefully with explicit “missing input” status in the report.
* Do not add heavy dependencies; use standard scientific stack already used elsewhere:

  * `python3`, `pandas`, `numpy`, `matplotlib` (plots optional).
* All plots must be suppressed unless `-p/--plot` is provided.

### Required deliverables (files to create)

Create (at minimum) the following structure:

```
MINGO_DIGITAL_TWIN/VALIDATION/
  validation.md                  # Must be filled with documentation and usage
  run_validation.py              # Main entrypoint CLI
  validators/
    __init__.py
    common_io.py                 # read artifacts, chunk manifests, metadata
    common_report.py             # report schema helpers, thresholds, pass/fail
    validate_step0_mesh.py
    validate_step1_muons.py
    validate_step2_crossings.py
    validate_step3_avalanche.py
    validate_step4_induction.py
    validate_step5_strip_obs.py
    validate_step6_endpoints.py
    validate_step7_offsets.py
    validate_step8_fee.py
    validate_step9_trigger.py
    validate_step10_tdc.py
    validate_final_dat.py
    validate_cross_step_lineage.py
  output/
    (created at runtime; one subfolder per validation run)
```

You may add more validators if justified, but do not remove the above.

### CLI behavior requirements

`run_validation.py` must:

* Discover which sim-run to validate from registries by default (latest run per step), but also allow explicit selection.
* Allow validating either:

  * all steps (default), or
  * a selected subset of steps.
* Always produce CSV output; plots only under `-p/--plot`.

Proposed CLI (implement equivalently):

* `python3 VALIDATION/run_validation.py`
* `python3 VALIDATION/run_validation.py --sim-run <ID>` (if sim-run is global)
* `python3 VALIDATION/run_validation.py --steps 1,2,3,9,final`
* `python3 VALIDATION/run_validation.py -p` or `--plot`

### Output requirements

Each run must create a unique output folder, for example:

* `VALIDATION/output/validate_<YYYYMMDD_HHMMSS>/`

Inside it, write:

1. `validation_results.csv` (long-form; one row per test metric)
2. `validation_summary.csv` (one row per test group/validator; pass/fail and key counts)
3. `validation_run_metadata.json` (high-level run context, including sim-run IDs, config hashes, upstream hashes)
4. `plots/` only if `--plot`

Also maintain or append to a persistent history file:

* `VALIDATION/output/validation_history.csv`

This file must accumulate one summary row per run, so we can track regressions over time.

### CSV schema (must be stable)

Design the long-form report `validation_results.csv` with at least these columns:

* `run_timestamp`
* `validator` (e.g., `validate_step2_crossings`)
* `test_id` (stable identifier string)
* `test_name` (human readable)
* `step` (integer or `final` / `cross`)
* `sim_run` (or per-step sim_run)
* `config_hash`, `upstream_hash` (when available)
* `n_rows_in`, `n_rows_out` (when meaningful)
* `metric_name`
* `metric_value`
* `expected_value` (nullable)
* `threshold_low`, `threshold_high` (nullable)
* `status` ∈ {`PASS`,`FAIL`,`WARN`,`SKIP`,`ERROR`}
* `notes` (short diagnostic)

The summary CSV should aggregate per validator:

* number of PASS/WARN/FAIL/SKIP/ERROR,
* the strict overall status rule (FAIL if any FAIL/ERROR unless explicitly justified).

### Validation philosophy and scope

Implement **three layers** of validation:

1. **Engineering / integrity checks**
   Schema, metadata, registries, chunk equivalence, deterministic invariants, ID conservation.

2. **Numerical/statistical checks**
   Distribution sanity, mean/variance checks, scaling with statistics where applicable, seed-independence at the level of aggregated statistics.

3. **Physics/electronics plausibility checks**
   Closure tests against configured parameters (efficiencies, jitter sigmas), geometry consistency, trigger logic truth table, and digitization behavior.

### Minimum set of tests to implement (high priority)

Implement tests that are robust and informative even without experimental data:

#### Cross-step and provenance (must implement)

* Registry integrity: every referenced file exists and metadata is readable.
* Lineage consistency: upstream hashes recorded and consistent across consecutive steps (when applicable).
* Event ID conservation: check that per-event keys (whatever identifier exists) are preserved across steps, except where selection occurs (trigger).
* Row count deltas: quantify n-rows change per step and flag unexpected losses/gains outside documented behavior.
* Chunk manifest consistency: total concatenated chunk rows equals manifest totals.

#### Step 0: parameter mesh

* Coverage/range checks for sampled parameters (flux, cos_n, efficiencies, z-planes).
* Duplicate detection or intentional duplicates documented.
* Mesh determinism check if IDs depend on ordering.

#### Step 1: muon generation

* Angular distribution check consistent with configured `cos_n` parameterization (verify expected shape qualitatively and via summary stats).
* Spatial sampling bounds and uniformity checks (coarse).
* Time sampling bounds sanity.

#### Step 2: plane crossings (geometry)

* Point-on-plane residuals: `|z - z_plane_i|` within tolerance.
* Monotonic ordering of crossings along track direction (time ordering consistent).
* Unphysical velocities or negative time increments flagged.
* Acceptance continuity: no pathological clustering at boundaries (basic diagnostics).

#### Step 3: avalanche / efficiency

* Efficiency closure per plane: observed detection fraction matches configured efficiency within statistical tolerance derived from binomial uncertainty (use WARN/FAIL levels).
* Avalanche size distribution sanity: check for nonphysical negatives, extreme outliers, and stable moments.

#### Steps 4–6: induction/readout observables/endpoints

* Charge sharing: non-negativity, locality (far strips negligible), symmetry for centered hits (when identifiable).
* Charge conservation: sum(strip charges) approximately equals modeled induced charge within tolerance.
* Endpoint ordering and consistency (`T_front ≤ T_back` if defined; analogous checks).
* Derived quantities (`T_diff`, `q_diff`) internal identity checks and sign conventions.

#### Step 7: cable offsets

* Offsets shift times without altering intra-channel ordering unexpectedly.
* If offsets are deterministic per channel: stationarity across events.

#### Step 8: FEE

* Jitter RMS consistent with configured sigma.
* Time-walk: monotonic dependence of leading-edge time on charge proxy (if model includes it); absence of unintended correlation if model claims none.
* Threshold behavior: sub-threshold hit suppression fraction behaves consistently.

#### Step 9: trigger

* Trigger truth table validation using observed hit patterns (verify exact logic).
* Trigger rate sanity: compare measured trigger acceptance to a simplified analytic expectation under independence (order-of-magnitude check, WARN not FAIL unless gross mismatch).
* Ensure selection behavior is the only intentional place where event counts drop substantially.

#### Step 10: TDC/DAQ smear

* Variance addition: confirm that time variance increases by approximately configured smear variance when independent Gaussian smear is assumed (use approximate tolerance).
* No unintended mean bias unless specified.

#### FINAL: station `.dat`

* Format/schema checks: parsable, correct column counts, event grouping, allowed numeric ranges.
* Round-trip test: parse `.dat` back and compare key fields to step-10 values within formatting precision.

### Plots (`--plot` only)

When `--plot` is enabled, produce a minimal but useful diagnostic set, saved under:

* `VALIDATION/output/<run>/plots/<validator_name>/...png`

Examples:

* histograms of timing residuals / jitter / smear,
* plane crossing residuals,
* efficiency closure plots (estimated vs configured),
* trigger acceptance vs expectation,
* scatter charge vs time (time-walk evidence),
* `.dat` field distributions.

### Thresholds and status logic

* Use conservative defaults with clear notes; distinguish:

  * `PASS`: within strict tolerance,
  * `WARN`: within relaxed tolerance or minor anomalies,
  * `FAIL`: violates physics/integrity constraints or large mismatch,
  * `SKIP`: missing required inputs,
  * `ERROR`: exception in validator.

Where possible, compute expected statistical tolerance from event count (binomial for efficiencies, Poisson-like for counts/rates). Document the rule in `validation.md`.

### Documentation requirement: fill `VALIDATION/validation.md`

You must create a complete `validation.md` that includes:

1. Purpose and validation layers (engineering/statistical/physics).
2. How to run validators (CLI examples).
3. Report formats and how to interpret PASS/WARN/FAIL.
4. Per-step checklist summarizing tests implemented.
5. Notes on common failure modes (selection bias at trigger, unit mismatches, geometry convention drift).
6. Guidance on adding new tests and updating thresholds responsibly.
7. A short section on limitations (what cannot be validated without real data).

Write it clearly as repository-quality documentation.

### Quality expectations

* The validation suite must be maintainable: small functions, clear naming, consistent report schema.
* It must be safe: read-only, no modifications to simulation artifacts.
* It must be informative: failures should include diagnostic notes pointing to likely causes and affected artifacts.

---

If you complete the above deliverables, the repository should gain a repeatable validation harness that can be executed periodically to detect regressions in physics, readout, electronics, and formatting behavior, using only the row-event intermediate artifacts in `INTERSTEPS/`.
