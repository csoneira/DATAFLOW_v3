- Check the L2, use normalized version, and check the Poisson-like score. Compare them to see if they give consistent results or if one is more stable than the others.
- Check the effect of z-score scaling on the scores and the inferred parameters. Does it improve the stability of the inference, especially in cases with varying event counts or efficiency regimes?
- Try to get the errors in the inference. For example a reasonable error would be the sqrt of counts since they come with Poisson statistics, so for example a chisq could be interesting to calculate.



---


# INFERENCE_DICTIONARY_VALIDATION — Pipeline Validation and Methodological Guidance (Non-Code)

This document provides high-level, non-code guidance for validating the simulation pipeline implemented in `INFERENCE_DICTIONARY_VALIDATION`, covering Steps 1 through 4. It is intended to be pasted into the repository and used as prompt context for LLM-assisted development (Codex, Claude) and for internal methodological traceability.

## Scope and Philosophy

This pipeline addresses two different validation problems that must be kept distinct:

1. **Simulation correctness (forward problem)**: whether simulated detector-response observables are internally consistent and physically plausible.
2. **Method correctness (inverse problem)**: whether the self-consistency matching procedure can reliably infer parameters (notably `(flux, eff_1)`) from observables under controlled conditions.

A pipeline can appear “validated” on in-dictionary points while failing to generalize off-dictionary. Therefore, validation must explicitly separate **interpolation performance** from **generalization/extrapolation performance** and must track selection effects introduced by filtering.

---

## 1. Clarify Validation Targets and Acceptance Criteria

Before tuning thresholds, define what “validated” means for the regime you actually care about. A minimally complete set of criteria typically includes:

### 1.1 Detector-response fidelity (simulation target)
- **Efficiency closure**: topology-based efficiency estimators reproduce the simulated (“truth”) plane efficiencies within a stated tolerance across the relevant parameter space.
- **Rate coherence**: topology rates and global rates scale consistently with flux and geometry, without unexplained discontinuities.
- **Unit and convention consistency**: time normalization, rate units, and count definitions are consistent across all steps.

### 1.2 Inverse-problem fidelity (method target)
- **Bias control**: inferred parameters show negligible systematic bias over the validated domain.
- **Variance scaling**: errors shrink with event count approximately as expected for counting statistics until limited by dictionary discreteness or degeneracy.
- **Domain-of-validity clarity**: Step 4 outputs provide defensible minimum-event thresholds and coverage/validity indicators that predict failure cases.

### 1.3 Common failure modes to track explicitly
Track these separately rather than collapsing everything into a single “passed/failed” label:
- Systematic bias in estimates.
- Variance larger than statistical expectations.
- Parameter non-identifiability (feature degeneracy, near-ties).
- Geometry-dependent deviations.
- Model mismatch between simulation and later observed data (if applicable).
- Selection bias introduced by filtering (Step 2).

---

## 2. Step 1 Validation: Dictionary Integrity and Internal Consistency

Step 1 generates the dictionary CSV that becomes the foundation for Steps 2–4. Validate it as a data product, not merely a file that exists.

### 2.1 Schema and join-key integrity
- Confirm presence and uniqueness behavior of `file_name` / `filename_base` and any other join keys used downstream.
- Verify that required columns for Step 2 and Step 3 exist and have expected dtypes (including parsable `efficiencies` list strings).
- Confirm that event-count fields used later (`selected_rows`, etc.) are populated and consistent with how “generated events” is defined.

### 2.2 Distribution and range sanity checks
- Check distributions of `flux_cm2_min`, `cos_n`, and `z_plane_1..4` for coverage gaps, unintended clustering, or missing regions.
- Confirm that simulated efficiencies lie in physically plausible ranges and that the sampling strategy matches intended experimental regimes.

### 2.3 Forward coherence checks (physics-level sanity)
These checks are conceptual expectations; failures require explanation, not immediate code changes:
- Topology rates and global rate should scale approximately linearly with flux (unless dead time, pileup, or saturation is intentionally modeled).
- Small geometry changes should not produce discontinuous rate jumps unless an acceptance edge is expected.
- Changes in a given plane efficiency should primarily affect topologies that depend on that plane.

---

## 3. Step 2 Validation: Efficiency Estimators as a Measurement Model

Step 2 is not only a filter. It defines an **empirical measurement model** for per-plane efficiencies derived from topology counts. Validate it accordingly.

### 3.1 Closure tests for efficiency reconstruction
For each plane `i`, evaluate:
- Residuals: `eff_resid_pi = eff_est_pi − eff_sim_pi`
- Relative errors: `eff_rel_err_pi = (eff_est_pi − eff_sim_pi)/eff_sim_pi`

Recommended diagnostic dependencies:
- **Event count**: error magnitude should generally shrink with increasing events, roughly consistent with `1/sqrt(N)` when dominated by counting statistics.
- **Efficiency regime**: near-unity efficiencies can amplify numerical sensitivity; confirm behavior is stable in the regime of interest.
- **Flux**: flux should change statistical precision, not introduce bias (unless the simulation includes flux-dependent effects).
- **Geometry (`z_plane_*`) and `cos_n`**: check for geometry-dependent biases.

### 3.2 Compare estimators as a robustness check
The pipeline supports:
- `four_over_three_plus_four`
- `one_minus_three_over_four`

Even if closely related under ideal assumptions, treat these as two independent “instruments”:
- If they disagree beyond statistical expectations in specific regions (especially asymmetric geometries), that flags estimator-model mismatch or topology classification issues.

### 3.3 Review filtering criteria and selection bias
The current filtering keeps rows when:
- `abs(eff_rel_err_p2) <= relerr_threshold`
- `abs(eff_rel_err_p3) <= relerr_threshold`
- `generated_events_count >= min_events`

Actions:
- Compare distributions of key parameters (flux, cos_n, geometry, efficiencies) for **used vs unused** rows.
- Check whether filtering on only p2/p3 preferentially retains a subset that looks artificially “well-behaved” while p1/p4 degrade.
- If downstream inference uses features influenced by all planes, consider extending quality criteria or reporting separate “validated plane sets.”

### 3.4 Make statistical assumptions explicit
Document assumptions underlying estimator interpretation, for example:
- Event samples are representative; topology classification is stable.
- Correlated inefficiencies are absent or negligible (or explicitly modeled).
- Topology counts follow approximate Poisson counting behavior at fixed parameters.

If assumptions are only approximate, interpret Step 2 filtering as empirical quality control rather than rigorous uncertainty control.

---

## 4. Step 3 Validation: Self-Consistency and Inverse Mapping Reliability

Step 3 estimates `(flux, eff_1)` by matching feature fingerprints in a restricted geometry subset. Validation must separate memorization-like interpolation from genuine generalization.

### 4.1 Distinguish in-dictionary vs off-dictionary performance
Always report errors stratified by:
- **Exact dictionary membership** (`sample_in_dictionary`, `best_in_dictionary`),
- **distance-to-dictionary** in normalized `(flux, eff_1)` space (Step 4 provides this),
- **event count**.

Interpretation guideline:
- Strong performance only on in-dictionary points suggests insufficient coverage or overreliance on near-duplicates.
- Performance degradation with distance is expected; quantify it to define a validity region.

### 4.2 Deliberate hold-out (leave-out) tests
To demonstrate real inferential behavior, perform controlled experiments:
- Remove a region/cell of `(flux, eff_1)` from the candidate dictionary and test samples from that region.
- Remove a geometry configuration and confirm that geometry constraints are essential and functioning as intended.

These tests distinguish interpolation from extrapolation and reveal degeneracies.

### 4.3 Detect feature degeneracy and non-identifiability
Nearest-neighbor methods fail when multiple parameter points yield similar features.
For each sample:
- Inspect top-N candidates and quantify spread in `(flux, eff_1)` and geometry.
- A wide spread among near-ties indicates ambiguity; do not report a single point estimate without a confidence qualifier.

If degeneracy is intrinsic:
- Consider augmenting the feature set (additional observables) or using a local regression/weighted approach rather than strict nearest-neighbor.

### 4.4 Evaluate scoring metrics and scaling strategies
The pipeline supports `l2`, `chi2`, `poisson`, `r2`, plus optional z-score scaling.

Guidance:
- **Poisson-like scoring** is often conceptually aligned with count-derived observables.
- **L2** is sensitive to feature scale; scaling choices can dominate outcomes.
- **R²** can behave poorly under heteroscedastic noise and non-Gaussian residuals.

Validation action:
- Compare at least two score metrics and at least two scaling modes across event bins and across dictionary-distance strata.
- A robust method should not collapse under reasonable choices.

### 4.5 Controlled perturbation runs (causal sensitivity)
Perform “single-variable sweeps” to confirm feature sensitivity is physically and methodologically consistent:
- Fix geometry and efficiencies; vary flux only.
- Fix geometry and flux; vary one efficiency at a time.
- Fix flux and efficiencies; vary geometry within a small neighborhood.

Confirm that inferred parameters respond monotonically and without unexplained discontinuities.

---

## 5. Step 4 Validation: Uncertainty Limits and Dictionary Coverage

Step 4 outputs are empirical calibrations conditioned on Step 2 filtering, Step 3 settings, and the sample distribution.

### 5.1 Conditional nature of uncertainty curves
Uncertainty bands and validity limits are not universal physical errors. They describe performance **given**:
- the reference set used in Step 3 all-mode,
- the filtering choices from Step 2,
- the dictionary density and coverage,
- the chosen features and scoring metric.

Therefore, always interpret Step 4 outputs with stratification:
- by in-dictionary vs off-dictionary,
- by distance-to-dictionary,
- by event count.

### 5.2 Sanity checks: monotonicity and error floors
Minimal expectations:
- Errors should generally decrease with event count.
- A persistent floor at high statistics suggests dictionary discreteness, feature degeneracy, or scoring limitations dominate (not counting noise).

Use this to separate:
- **statistical uncertainty** (improves with N),
- **method/coverage limitation** (does not improve with N).

### 5.3 Coverage metrics must correlate with errors
Coverage diagnostics (grid fill, convex hull area, radius-based coverage, nearest-neighbor spacing, sample-to-dictionary distance) should be predictive:
- If distance correlates strongly with error, the coverage model is meaningful.
- If correlation is weak, reconsider normalization choices or whether `(flux, eff_1)` alone captures the relevant notion of coverage.

### 5.4 Use Step 4 as a dictionary design tool
Treat Step 4 as feedback for simulation planning:
- Identify sparse regions in `(flux, eff_1)` and high-distance samples with large errors.
- Add targeted simulations to densify those regions.
- Re-run Steps 2–4 and quantify improvement.

This “simulate → validate → densify → revalidate” loop is the recommended development cycle.

---

## 6. Practical Validation Actions and Reporting

### 6.1 One-page closure reports per step
For each step, maintain a short summary including:
- key plots used,
- observed biases/trends,
- known failure regions,
- current configuration (thresholds, metric_mode, score_metric),
- domain-of-validity statements.

This prevents regressions and makes threshold changes auditable.

### 6.2 Cut sensitivity analysis
Vary `relerr_threshold` and `min_events` and observe:
- Step 2 bias/variance behavior,
- Step 3 inference bias/variance stratified by dictionary distance,
- Step 4 validity limits stability.

If small threshold changes cause large downstream instability, the method is brittle.

### 6.3 Define validity masks
Formalize a recommended “trust region,” for example:
- `events >= N_min(quantile, target_error_pct)` from Step 4, AND
- `sample_to_dict_dist_norm <= d_max`, AND
- optional Step 2 quality flags.

Use these masks both in development and in any later real-data application.

---

## 7. Criteria for Confidence in the Pipeline (Regime-Specific)

You can consider the pipeline validated for a specific regime if all conditions below hold in that regime:

- **Step 2**: residuals are approximately unbiased and variance scales with statistics as expected; estimator disagreements are understood.
- **Step 3**: bias is small; errors decrease with events until reaching a dictionary/degeneracy floor; top-N candidate spreads are acceptably narrow within the trust region.
- **Step 4**: predicted minimum-event thresholds reliably separate high-confidence from low-confidence samples; distance-to-dictionary meaningfully predicts error; coverage metrics highlight failure regions that improve when the dictionary is densified.
- **Traceability**: configuration changes produce explainable and documented shifts in performance.

Validation is always conditional on the configuration and domain; state both explicitly.

---

## 8. Prompt Context Template for LLM-Assisted Modifications (No Code)

When asking Codex/Claude to modify or extend this pipeline, include:

- Step script(s) to be modified and the intended behavior change.
- The target validation regime (event counts, flux range, geometry variability).
- Current settings: `metric_mode`, `score_metric`, `metric_scale`, `include_global_rate`, `relerr_threshold`, `min_events`, and Step 4 targets (quantiles, target errors).
- Whether the goal is improved interpolation, improved off-dictionary generalization, or better validity masks.
- What artifacts must be produced (tables/plots/summary JSON fields) to support the validation criteria in Sections 1 and 7.

This ensures method changes remain aligned with measurable validation outcomes rather than optimizing a single metric in isolation.
