## Problem Statement
STEP 1.2 builds an inverse dictionary (feature space -> physical parameter space). Inference quality depends on continuity of this inverse map: nearby feature points must map to nearby physical points, and dictionary support must be sufficiently connected and dense. If support is discontinuous, degenerate, or sparse, downstream estimation can fail silently while still producing outputs.

Observed failure mode: even when neighborhoods look compact in selected feature projections, local inverse neighborhoods can still span a large fraction of the flux range (`local_injectivity` FAIL), so flux estimates can become ambiguous.

## What Was Implemented
The following continuity controls were already integrated in STEP 1.2 and are preserved:

1. Continuity validation is executed inside STEP 1.2 (pre-filter and post-filter), not postponed to later steps.
2. Checks include parameter-space coverage, local continuity, bidirectional topology continuity, local injectivity, isotonic bounds, support adequacy, and density diagnostics.
3. Filtering is applied inside STEP 1.2 before final dictionary save (`continuity_filtering` report in `build_summary.json`).
4. Shared feature-space configuration is centralized in `config_method.json` via `common_feature_space`, with step-local overrides.
5. Feature-space dimensionality is reduced from unnecessary large default spaces to resolved/common selected features.
6. Continuity plots already existed: `continuity_validation.png` and neighborhood matrix outputs (`1_2_18`, `1_2_19`).
7. Test coverage exists in `tests/test_continuity_validation.py`.
8. Strict continuity policy is now enabled (`fail_on_error=true`), so dictionaries that fail continuity/injectivity no longer pass silently.

Additional implementation updates in this pass:

1. Fixed STEP 1.2 feature-selection bootstrapping when `common_feature_space.feature_columns="step12_selected"`:
   - During active STEP 1.2 auto-selection, the resolver now bypasses previous `selected_feature_columns.json` and recomputes from resolved common/derived config.
   - This removes the circular lock where one prior selection could force only one future candidate set.
2. Expanded default candidate-size exploration in STEP 1.2 auto-selection:
   - Deterministic extra-feature ladder now evaluates broader dimensionalities (still capped/configurable).
   - This allows selecting richer feature sets when low-dimensional sets remain non-injective.
3. Kept continuity neighborhood outputs backward-compatible and more stable:
   - Legacy filenames `1_2_18_continuity_neighborhood_example_param_to_feature_matrix.png` and `1_2_19_continuity_neighborhood_example_feature_to_param_matrix.png` are now written explicitly.
   - Unnumbered stable files are also written for both directional matrices plus the combined bidirectional matrix.

## Remaining Gaps / Why current plots are still insufficient
Current diagnostics are substantially better, but one essential gap remains:

1. The strict injectivity thresholds are still not met after capped filtering (`25%` max drop). With current support, local feature neighborhoods still map to broad parameter spans (especially flux), so strict mode aborts.
2. Two constraints are in direct tension:
   - continuity strictness (`injectivity_bad_fraction_max=0.05`, `injectivity_flux_span_fraction_p95_max=0.30`)
   - support retention (`filter_max_drop_fraction=0.25`)
3. Empirically, forcing injectivity pass under current thresholds requires very aggressive pruning (far beyond 25%), which risks coverage collapse.

## Proposed Improvements
Implemented in this update (without adding dependencies or changing downstream logic):

1. Upgraded `continuity_validation.png` into a 2x3 actionable dashboard with explicit threshold overlays, status badges, and panel-level metrics.
2. Added explicit topology panel (`overlap vs expansion`) with threshold guides and flagged-point highlighting.
3. Preserved existing directional neighborhood matrix outputs for compatibility.
4. Added a new combined bidirectional neighborhood figure:
   - `continuity_neighborhood_example_bidirectional.png`
   - Left half: parameter -> feature mapping
   - Right half: feature -> parameter mapping
   - Each half uses full lower-triangular matrices (no diagonal) with identical selected points/colors between source and mapped spaces.
5. Neighborhood point selection is deterministic and source-domain local (compact-anchor + nearest neighbors), so selected sets are genuinely close in the source topology.
6. Continuity plotting now prefers feature columns resolved by continuity checks (aligned with `common_feature_space` + overrides), ensuring consistent feature-space use between validation and visualization.
7. Added strict local injectivity validation in STEP 1.2: feature-neighborhood parameter-span limits (especially flux) are now checked, filtered, and can trigger build failure.
8. Added STEP 1.2 reselection guard and broader candidate-size search to avoid stale feature-lock and improve degeneracy handling.
9. Added stable legacy + unnumbered directional neighborhood plot filenames for compatibility and easier inspection.

## Validation Evidence
Validation was executed with the required commands after code changes:

1. `pytest -v MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/tests/test_continuity_validation.py`
2. `cd MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY && python3 build_dictionary.py`
3. Verified plot outputs and `build_summary.json` continuity blocks (`continuity_validation`, `continuity_validation_pre_filter`, `continuity_filtering`).

Observed outcomes (latest run, 2026-03-11):

- Tests: `14 passed`.
- Strict run (`fail_on_error=true`) still aborts as intended:
  - post-filter `local_injectivity=FAIL (flux_span_p95=0.819, bad_fraction=0.4761)`
  - post-filter topology improves but remains `WARN` (`bad_fraction=0.1215` vs threshold `0.08`)
- Non-strict run (same config except `fail_on_error=false`) confirms artifact generation and continuity blocks in `build_summary.json`:
  - rows before/after filtering: `1229 -> 922` (removed `307`, `24.98%`)
  - selected feature count for continuity: `45`
  - all continuity sections present:
    - `continuity_validation_pre_filter`
    - `continuity_filtering`
    - `continuity_validation`
