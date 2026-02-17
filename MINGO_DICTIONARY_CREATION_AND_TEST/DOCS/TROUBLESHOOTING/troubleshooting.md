# Troubleshooting & Development Notes

This document collects the refactoring history, validation guidance, and
outstanding items for the `INFERENCE_DICTIONARY_VALIDATION` pipeline.

---

## 1. Refactoring history

### 1.1 Shared utilities extraction

The four original step scripts carried ~600 lines of duplicated helper
functions (parsing, scoring, efficiency computation, plotting, geometry).
All shared code was extracted into `MODULES/simulation_validation_utils.py`.

**Key changes per step:**

| Step | Lines removed | Main functions extracted |
|---|---|---|
| STEP 1 | ≈0 | `plot_histogram`, `plot_scatter` |
| STEP 2 (validate) | −33 | `parse_efficiencies`, `safe_numeric`, `compute_efficiency` |
| STEP 3 (relative error) | −54 | `plot_*` helpers, `resolve_param` |
| STEP 4 (self-consistency) | −179 | 15 symbols: scoring functions, config, parsing, efficiency, data helpers |
| STEP 5 (uncertainty limits) | −138 | `build_uncertainty_table`, geometry, `coerce_bool_series` |

### 1.2 Cross-cutting improvements

- **Structured logging**: all `print()` replaced with `logging` via `setup_logger()`.
- **Config resolution**: uniform `CLI > config.json > default` via `resolve_param()`.
- **Path normalisation**: all scripts use `STEP_DIR` / `REPO_ROOT` patterns.
- **Output layout**: `output/` directories replaced with `OUTPUTS/FILES/` and `OUTPUTS/PLOTS/`.
- **Plot style**: global `apply_clean_style()` sets clean, minimal matplotlib defaults.

---

## 2. Validation guidance

### 2.1 Validation targets

1. **Simulation correctness** (forward): simulated observables are internally
   consistent and physically plausible.
2. **Method correctness** (inverse): the matching procedure reliably infers
   `(flux, eff_1)` under controlled conditions.

### 2.2 Step-by-step checks

**Step 1 — Dictionary integrity**
- Confirm join-key uniqueness (`file_name` / `filename_base`).
- Check distributions of `flux_cm2_min`, `cos_n`, `z_plane_*` for coverage gaps.
- Rates should scale approximately linearly with flux.

**Step 2 — Efficiency estimators**
- Residuals `eff_est − eff_sim` should shrink with event count (~1/√N).
- Compare `four_over_three_plus_four` vs `one_minus_three_over_four`.
- Planes 1 & 4 carry an acceptance-factor bias; document it, don't suppress it.

**Step 3 — Relative error & filtering**
- Compare used vs unused distributions for selection bias.
- Verify filtering on planes 2 & 3 doesn't systematically exclude useful regimes.

**Step 4 — Self-consistency**
- Stratify errors by: in-dictionary vs off-dictionary, distance-to-dictionary, event count.
- Compare at least two score metrics and scaling modes.
- Consider hold-out tests to separate interpolation from extrapolation.

**Step 5 — Uncertainty calibration**
- Errors should decrease with event count until reaching a dictionary/degeneracy floor.
- Coverage diagnostics should correlate with prediction error.

**Steps 6–7 — LUT & demo**
- LUT provides 3-D uncertainty lookup (flux, eff, events).
- Demo step tests 1σ/2σ/3σ coverage against Gaussian expectations.
- Calibration scale factors close to 1.0 indicate well-calibrated raw uncertainties.

### 2.3 Criteria for confidence

The pipeline is considered validated for a specific regime when:
- Step 2 residuals are approximately unbiased, variance scales with statistics.
- Step 4 bias is small; errors decrease with events.
- Step 5 thresholds reliably separate high- from low-confidence samples.
- Configuration changes produce explainable, documented shifts.

---

## 3. Outstanding items

- [ ] Check L2 with normalized version; compare Poisson-like score for consistency.
- [ ] Check effect of z-score scaling on stability across event-count and efficiency regimes.
- [ ] Get inference errors (e.g. √counts Poisson-based χ²).
- [ ] Split STEP 4 `main()` into `_run_single_mode()` + `_run_all_mode()`.
- [ ] Add unit tests for `simulation_validation_utils` (scoring, `parse_efficiencies`).
- [ ] Type annotations on remaining helpers.
- [ ] Comparison with real data (next phase).
- [ ] Cut sensitivity analysis: vary `relerr_threshold` and `min_events` and measure downstream stability.
- [ ] Define formal validity masks: `events >= N_min`, `sample_to_dict_dist_norm <= d_max`.

---

## 4. Prompt context for LLM-assisted modifications

When asking an LLM to modify this pipeline, include:
- Step script(s) and intended behavior change.
- Target validation regime (event counts, flux range, geometry).
- Current settings: `metric_mode`, `score_metric`, `metric_scale`, thresholds.
- Goal: improved interpolation, off-dictionary generalization, or validity masks.
- Required artifacts (tables, plots, summary JSON fields).

---

## 5. Known expected behavior

### 5.1 STEP 1.1 (`collect_data.py`) can legitimately return far fewer rows

This is expected when these two filters are active:

1. **Single z-configuration filter**  
   `collect_data.py` keeps only rows matching one `(z_plane_1..4)` configuration
   from `config["z_position_config"]`.

2. **Inner join with task metadata**  
   It then keeps only `filename_base` values present in both:
   - `step_final_simulation_params.csv`
   - `task_*_metadata_specific.csv` for configured `task_ids`

Because of this, a large input table can reduce to a much smaller collected set.

**Recorded example (2026-02-12):**
- Simulation params rows: `17129`
- Task 1 metadata rows: `1649`
- Config z selection: `[0.0, 145.0, 290.0, 435.0]`
- Rows after z cut: `3024`
- Rows after inner join (task 1): `207`
- `z_config_selected.json` and `collected_data.csv` report this same `207`.

This behavior is **not a bug**.

If more rows are desired, change configuration intentionally:
- include more `task_ids`
- use a different `z_position_config` (or set to `null` to randomize)
- remove/relax the z filter logic (only if methodologically intended)
