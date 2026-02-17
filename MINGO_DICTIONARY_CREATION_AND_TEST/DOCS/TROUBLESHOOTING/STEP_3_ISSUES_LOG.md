# STEP 3 Issues Log (Updated: 2026-02-12)

## Resolved Issues

1. Missing STEP 3 config structure.
   - Added `step_3_2` config block and maintained backward-compatible keys.

2. STEP 3.1 plot confusion (duplicate files and naming).
   - Removed redundant identical plot output.
   - Standardized naming to `complete` (instead of `smooth`).

3. STEP 3.1 start/end and overlay readability problems.
   - Fixed plotting overlays and axis handling.
   - Kept complete + discretized views in the same figure where requested.

4. STEP 3.2 highlight plot mismatch and duplication.
   - Added dictionary cloud to comparison.
   - Unified axis limits where needed.
   - Collapsed two-panel highlight into one combined panel.

5. STEP 3.2 highlight index was effectively deterministic.
   - Changed default behavior to truly random unless seed is explicitly set.

6. STEP 3.2 basis source for synthesis was wrong for your goal.
   - Switched basis to dataset (`STEP_1_2 .../dataset.csv`) by default.

7. Event-count conditioning logic was too rigid/incorrect.
   - Reworked to use per-target `n_events` from STEP 3.1 time series.
   - Switched tolerance control to percentage (`basis_n_events_tolerance_pct`).

8. Flux/eff assignment strategy produced unrealistic behavior.
   - Replaced direct/flat behavior with linear distance-weighted center assignment.
   - Kept synthetic global-rate comparison separately in overview plot.

9. Key bias you identified: uneven multiplicity per parameter set.
   - Fixed by enforcing **exactly one basis row per parameter set per target point**, selecting the closest `n_events` row.
   - This removes the 3-vs-1 row bias across parameter sets.

10. Contribution diagnostics looked inconsistent (“far” top contributors).
   - Added explicit visibility of excluded rows in highlight plot.
   - Added contribution metadata (`is_event_allowed`, `basis_parameter_set_id`).

## Current Behavior Guarantee

- In STEP 3.2 basis construction:
  - One row per parameter set is selected for each target point.
  - Selection criterion: closest `n_events` to that target point.
  - No parameter set can contribute multiple rows at the same target point.

## Note

- If `top_k` is set, it still truncates contributors after selection.
  - Set `top_k: null` to keep all selected parameter-set contributors.
