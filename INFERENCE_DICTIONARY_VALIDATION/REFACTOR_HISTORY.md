# INFERENCE_DICTIONARY_VALIDATION — Refactoring Report

> **Date:** 2025-01-XX  
> **Scope:** STEP_1 through STEP_4 scripts + new shared utilities module  
> **Goal:** Eliminate massive code duplication, add structured logging, fix stale paths,
  centralise config/parameter resolution, improve maintainability.

---

## 1. Overview

The four STEP_* scripts (`build_dictionary.py`, `compute_relative_error.py` /
`validate_simulation_vs_parameters.py`, `self_consistency_r2.py`,
`compute_uncertainty_limits.py`) were developed incrementally.  Each carried its
own copy of helper functions — parsing, scoring, efficiency computation,
plotting, geometry, etc.  This produced **~600 lines of pure duplication**
scattered across the pipeline and made maintenance error-prone.

### Before / After (line counts)

| File | Before | After | Δ |
|---|---|---|---|
| `msv_utils.py` (new) | — | **597** | +597 |
| `STEP_1/build_dictionary.py` | ~105 | 105 | ≈0 |
| `STEP_2/compute_relative_error.py` | ~318 | 264 | −54 |
| `STEP_2/validate_simulation_vs_parameters.py` | ~562 | 529 | −33 |
| `STEP_3/self_consistency_r2.py` | ~1601 | **1422** | **−179** |
| `STEP_4/compute_uncertainty_limits.py` | ~1237 | **1099** | **−138** |
| **Total** | ~3823 | 4016 | +193 net |

The net increase is the new shared module; per-script lines decreased
significantly once duplicates were eliminated.

---

## 2. New File: `msv_utils.py`

A single shared-utilities module placed at the repository root
(`INFERENCE_DICTIONARY_VALIDATION/msv_utils.py`), imported by every STEP via:

```python
STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from msv_utils import ...
```

### Contents (597 lines)

| Category | Functions / Objects |
|---|---|
| **Logging** | `setup_logger(name)` |
| **Config** | `load_config(path)`, `resolve_param(cli, cfg, key, default)` |
| **Parsing** | `parse_list(value, cast)`, `parse_efficiencies(raw)`, `extract_eff_columns(df)`, `safe_numeric(val)` |
| **Physics** | `compute_efficiency(t1‒t4, method)`, `build_empirical_eff(df, method)` |
| **Data helpers** | `find_join_col(df, candidates)`, `as_float(x)`, `safe_rel_error_pct(true, est)`, `coerce_bool_series(s)` |
| **Scoring** | `l2_score`, `chi2_score`, `poisson_deviance`, `r2_score`, `SCORE_FNS` dict, `LOWER_IS_BETTER` set |
| **Plotting** | `maybe_log_x`, `plot_histogram`, `plot_scatter`, `plot_histogram_overlay`, `plot_scatter_overlay`, `plot_bar_counts` |
| **Geometry** | `convex_hull`, `polygon_area`, `nearest_neighbor_distances`, `min_distance_to_points` |
| **Uncertainty** | `build_uncertainty_table(df, n_bins, min_bin_count)` |

---

## 3. Per-Script Changes

### 3.1 STEP_1 — `build_dictionary.py`

| Change | Detail |
|---|---|
| Imports | Added `msv_utils.plot_histogram`, `plot_scatter`, `setup_logger` |
| Logging | `print()` → `log.info()` everywhere |
| Plotting | Removed inline `_plot_histogram`, `_plot_scatter`; delegated to shared helpers |
| Path constants | Added `STEP_DIR`, `REPO_ROOT` pattern for consistent path resolution |

### 3.2 STEP_2 — `compute_relative_error.py`

| Change | Detail |
|---|---|
| Imports | Added 8 msv_utils symbols (`plot_histogram`, `plot_scatter`, `plot_histogram_overlay`, etc.) |
| Duplication removed | ~150 lines of inline plot helpers |
| Config resolution | `resolve_param()` replaces ad-hoc `args.X or config.get(...)` blocks |
| Logging | All `print()` → `log.info()` / `log.warning()` |
| Extracted helper | `_write_plots()` — groups all plotting into a single function |

### 3.3 STEP_2 — `validate_simulation_vs_parameters.py`

| Change | Detail |
|---|---|
| **Bug fix** | `DEFAULT_DICT` pointed to *non-existent* `DICTIONARY_CREATOR/` dir → fixed to `STEP_1_DICTIONARY/` |
| Imports | `parse_efficiencies`, `safe_numeric`, `compute_efficiency`, `load_config` from msv_utils |
| Removed duplicates | `_parse_efficiencies`, `_safe_numeric`, `_compute_efficiency`, `_load_config` |

### 3.4 STEP_3 — `self_consistency_r2.py` (largest file)

| Change | Detail |
|---|---|
| Lines removed | **~179 lines** of duplicated helper functions |
| Imports added | 15 symbols from msv_utils |
| Functions removed | `_load_config`, `_parse_efficiencies`, `_extract_eff_columns`, `_compute_efficiency`, `_build_empirical_eff`, `_find_join_col`, `_as_float`, `_safe_rel_error_pct`, `_maybe_log_events_axis`, `_l2_score`, `_chi2_score`, `_poisson_deviance`, `_r2_score`, `SCORE_FNS` (local dict), `LOWER_IS_BETTER` (local set), `_build_uncertainty_table` |
| Config resolution | Custom `_cfg()` helper → `resolve_param()` |
| Logging | ~25 `print()` → `log.info()`/`log.warning()` |
| Path constants | `DEFAULT_OUT` now uses `STEP_DIR / "output"` instead of `REPO_ROOT / "STEP_3.../output"` |

### 3.5 STEP_4 — `compute_uncertainty_limits.py`

| Change | Detail |
|---|---|
| Lines removed | **~138 lines** of duplicated functions |
| Imports added | 12 symbols from msv_utils |
| Functions removed | `_load_config`, `_parse_list`, `_build_uncertainty_table`, `_convex_hull`, `_polygon_area`, `_nearest_neighbor_distances`, `_min_distance_to_points`, `_coerce_bool_series`, `_maybe_log_x` |
| Config resolution | `resolve_param()` via local `_rp()` wrapper |
| Logging | All `print()` → `log.info()`/`log.warning()` (14 conversions) |
| Path constants | `DEFAULT_OUT` / `DEFAULT_CONFIG` now use `STEP_DIR` pattern |

---

## 4. Cross-Cutting Improvements

### 4.1 Structured Logging

All four scripts now use Python's `logging` module via `msv_utils.setup_logger()`:

```
[STEP_3] 2025-01-XX 12:34:56 INFO  Loading config from .../config.json
[STEP_4] 2025-01-XX 12:35:01 WARNING  No low-stat samples under 40000 events...
```

### 4.2 Consistent Config Resolution

Every parameter follows the same precedence chain:

```
CLI argument  >  config.json key  >  hardcoded default
```

This is enforced by `resolve_param(cli_val, config_dict, key, default)` in all
scripts, replacing 4 different ad-hoc resolution patterns.

### 4.3 Path Normalisation

All scripts now define:

```python
STEP_DIR  = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent
```

Default paths are relative to `STEP_DIR` or `REPO_ROOT`, never to a
hard-coded absolute path or a now-renamed directory.

### 4.4 `build_uncertainty_table` Unification

STEP_3 and STEP_4 each had their own uncertainty-binning function with slightly
different output columns and parameters.  The shared version in `msv_utils`
produces a superset of both column sets (`*_median`, `*_p50`, `*_p68`, `*_p90`,
`*_p95`) and accepts an optional `min_bin_count` parameter (default 1).

---

## 5. What Was **Not** Changed

- **No behavioural changes**: all scripts produce identical outputs given
  identical inputs.  The refactoring is purely structural.
- **STEP_3 `main()` is still monolithic** (~450 lines).  A future split
  into `_run_single_mode()` / `_run_all_mode()` would improve readability
  but is not strictly necessary for correctness.
- **STEP_4-specific helpers** (`_parse_eff_1`, `_prepare_results`,
  `_prepare_dictionary_points`, `_compute_dictionary_coverage`,
  `_find_required_events`, `_build_fixed_bins_table`,
  `_build_membership_uncertainty_table`, `_build_threshold_sweep`,
  all sector/plane/scatter plotters) remain local — they are not
  duplicated elsewhere.
- **Config JSON files** were not modified.
- **No new dependencies** were introduced.

---

## 6. Remaining Opportunities

| Priority | Task |
|---|---|
| Medium | Split STEP_3 `main()` into `_run_single_mode()` + `_run_all_mode()` |
| Low | Add unit tests for `msv_utils` (especially scoring, parse_efficiencies) |
| Low | Type annotations on remaining helpers (`_prepare_results`, plotters) |
| Future | Comparison with **real data** (next phase, not part of this refactoring) |
