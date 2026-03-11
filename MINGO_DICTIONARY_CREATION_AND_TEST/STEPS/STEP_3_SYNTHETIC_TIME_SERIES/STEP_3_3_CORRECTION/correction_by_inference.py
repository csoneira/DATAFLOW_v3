#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_3_CORRECTION/correction_by_inference.py
Purpose: STEP 3.3 — Correction by inference on synthetic dataset.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_3_SYNTHETIC_TIME_SERIES/STEP_3_3_CORRECTION/correction_by_inference.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
SYNTHETIC_DIR = STEP_DIR.parent
PIPELINE_DIR = SYNTHETIC_DIR.parent
PROJECT_DIR = PIPELINE_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "config_method.json"
CONFIG_COLUMNS_PATH = PROJECT_DIR / "config_columns.json"

DEFAULT_SYNTHETIC_DATASET = (
    SYNTHETIC_DIR / "STEP_3_2_SYNTHETIC_TIME_SERIES" / "OUTPUTS" / "FILES" / "synthetic_dataset.csv"
)
DEFAULT_STEP32_DIAGNOSTIC_CENTER = (
    SYNTHETIC_DIR / "STEP_3_2_SYNTHETIC_TIME_SERIES" / "OUTPUTS" / "FILES" / "diagnostic_center_series.csv"
)
DEFAULT_TIME_SERIES = (
    SYNTHETIC_DIR / "STEP_3_1_TIME_SERIES_CREATION" / "OUTPUTS" / "FILES" / "time_series.csv"
)
DEFAULT_COMPLETE_CURVE = (
    SYNTHETIC_DIR / "STEP_3_1_TIME_SERIES_CREATION" / "OUTPUTS" / "FILES" / "complete_curve_time_series.csv"
)
DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY" / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_DATASET_TEMPLATE = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY" / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY" / "OUTPUTS" / "FILES" / "selected_feature_columns.json"
)
DEFAULT_LUT = (
    PIPELINE_DIR / "STEP_2_INFERENCE" / "STEP_2_3_UNCERTAINTY" / "OUTPUTS" / "FILES" / "uncertainty_lut.csv"
)
DEFAULT_LUT_META = (
    PIPELINE_DIR / "STEP_2_INFERENCE" / "STEP_2_3_UNCERTAINTY" / "OUTPUTS" / "FILES" / "uncertainty_lut_meta.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "3_3"


def _save_figure(fig: plt.Figure, path: Path, **kwargs) -> None:
    """Save figure with a per-script sequential numeric prefix."""
    global _FIGURE_COUNTER
    _FIGURE_COUNTER += 1
    out_path = Path(path)
    out_path = out_path.with_name(f"{FIGURE_STEP_PREFIX}_{_FIGURE_COUNTER}_{out_path.name}")
    fig.savefig(out_path, **kwargs)


_PLOT_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".eps",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}


def _clear_plots_dir() -> None:
    """Remove previously generated plot files from the plots directory."""
    removed = 0
    for candidate in PLOTS_DIR.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in _PLOT_EXTENSIONS:
            try:
                candidate.unlink()
                removed += 1
            except OSError as exc:
                log.warning("Could not remove old plot file %s: %s", candidate, exc)
    log.info("Cleared %d plot file(s) from %s", removed, PLOTS_DIR)

logging.basicConfig(
    format="[%(levelname)s] STEP_3.3 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_3.3")

CANONICAL_FLUX_COLUMN = "flux_cm2_min"
CANONICAL_EFF_COLUMN = "eff_sim_1"

# Import estimation function from STEP 2 module.
INFERENCE_DIR = PIPELINE_DIR / "STEP_2_INFERENCE"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
try:
    from estimate_parameters import (  # noqa: E402
        _append_derived_physics_feature_columns,
        _append_derived_tt_global_rate_column,
        _auto_feature_columns as _shared_auto_feature_columns,
        _derived_feature_columns as _shared_derived_feature_columns,
        _normalize_derived_physics_features,
        estimate_from_dataframes,
        resolve_inverse_mapping_cfg,
    )
    from feature_columns_config import (  # noqa: E402
        parse_explicit_feature_columns,
        resolve_feature_columns_from_catalog,
        sync_feature_column_catalog,
    )
except Exception as exc:
    log.error("Failed to import estimate_parameters from %s: %s", INFERENCE_DIR, exc)
    raise


def _load_config(path: Path) -> dict:
    def _merge_dicts(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    cfg: dict = {}
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        log.warning("Config file not found: %s", path)

    plots_path = path.with_name("config_plots.json")
    if plots_path != path and plots_path.exists():
        plots_cfg = json.loads(plots_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, plots_cfg)
        log.info("Loaded plot config: %s", plots_path)

    runtime_path = path.with_name("config_runtime.json")
    if runtime_path.exists():
        runtime_cfg = json.loads(runtime_path.read_text(encoding="utf-8"))
        cfg = _merge_dicts(cfg, runtime_cfg)
        log.info("Loaded runtime overrides: %s", runtime_path)
    return cfg

def _safe_float(value: object, default: float) -> float:
    """Convert value to float with fallback."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_bool(value: object, default: bool) -> bool:
    """Convert common truthy/falsy representations to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


COMMON_FEATURE_SPACE_KEYS = (
    "feature_columns",
    "derived_features",
    "include_global_rate",
    "global_rate_col",
)


def _merge_common_feature_space_cfg(
    config: dict,
    cfg_21: dict | None = None,
) -> tuple[dict, dict]:
    """Merge top-level common feature-space config with STEP 2.1 overrides."""
    merged: dict = {}
    source_info: dict = {
        "common_keys": [],
        "step_2_1_override_keys": [],
        "has_common_feature_space": False,
    }
    common_cfg = config.get("common_feature_space", {}) if isinstance(config, dict) else {}
    if isinstance(common_cfg, dict):
        source_info["has_common_feature_space"] = True
        for key in COMMON_FEATURE_SPACE_KEYS:
            if key in common_cfg and common_cfg.get(key) is not None:
                merged[key] = common_cfg.get(key)
                source_info["common_keys"].append(key)
    cfg_21_local = cfg_21 if isinstance(cfg_21, dict) else {}
    for key in COMMON_FEATURE_SPACE_KEYS:
        if key in cfg_21_local and cfg_21_local.get(key) is not None:
            merged[key] = cfg_21_local.get(key)
            source_info["step_2_1_override_keys"].append(key)
    cfg_merged = dict(cfg_21_local)
    cfg_merged.update(merged)
    return cfg_merged, source_info


def _resolve_input_path(path_like: str | Path) -> Path:
    """Resolve path relative to pipeline when not absolute."""
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_pipeline


def _load_step12_selected_feature_columns(path: Path) -> tuple[list[str], dict]:
    """Load selected feature-column artifact generated by STEP 1.2."""
    info: dict = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        return [], info
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        info["error"] = str(exc)
        return [], info
    raw_cols = payload.get("selected_feature_columns", [])
    if not isinstance(raw_cols, list):
        info["error"] = "selected_feature_columns_not_list"
        return [], info
    selected = [str(c).strip() for c in raw_cols if str(c).strip()]
    info["selected_count"] = int(len(selected))
    info["selection_strategy"] = payload.get("selection_strategy", None)
    return selected, info


def _choose_eff_column(df: pd.DataFrame, preferred: str) -> str:
    """Select an efficiency column from dataframe with fallback candidates."""
    if preferred in df.columns:
        return preferred
    for candidate in (
        "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
        "eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4",
    ):
        if candidate in df.columns:
            return candidate
    raise KeyError("No efficiency column found in dataframe.")


def _resolve_inference_feature_columns(
    *,
    feature_cfg: object,
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    include_global_rate: bool,
    global_rate_col: str,
    step12_selected_path: Path | None = None,
    derived_feature_cfg: object = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Resolve feature columns for STEP 3.3 inference using STEP 2.1 criteria."""
    auto_feature_cols = sorted(
        set(_shared_auto_feature_columns(dict_df, include_global_rate, global_rate_col))
        & set(_shared_auto_feature_columns(data_df, include_global_rate, global_rate_col))
    )
    catalog = sync_feature_column_catalog(
        catalog_path=CONFIG_COLUMNS_PATH,
        dict_df=dict_df,
        default_enabled_columns=auto_feature_cols,
    )
    derived_cfg_raw = (
        derived_feature_cfg
        if isinstance(derived_feature_cfg, dict)
        else {}
    )
    categories = catalog.get("categories", {})
    if not isinstance(categories, dict):
        categories = {}
    trigger_cfg = categories.get("trigger_type", {})
    if not isinstance(trigger_cfg, dict):
        trigger_cfg = {}
    rate_hist_cfg = categories.get("rate_histogram", {})
    if not isinstance(rate_hist_cfg, dict):
        rate_hist_cfg = {}

    derived_tt_prefix = str(
        derived_cfg_raw.get("prefix", catalog.get("prefix", "last"))
    ).strip() or "last"
    derived_trigger_types = parse_explicit_feature_columns(
        derived_cfg_raw.get("trigger_types", trigger_cfg.get("trigger_types", []))
    )
    derived_include_to_tt = _safe_bool(
        derived_cfg_raw.get(
            "include_to_tt_rate_hz",
            derived_cfg_raw.get(
                "include_to_prefix_rates",
                catalog.get("*_to_*_tt_*_rate_hz", False),
            ),
        ),
        bool(catalog.get("*_to_*_tt_*_rate_hz", False)),
    )
    derived_include_trigger_rates = _safe_bool(
        derived_cfg_raw.get("include_trigger_type_rates", False),
        False,
    )
    derived_include_hist = _safe_bool(
        derived_cfg_raw.get(
            "include_rate_histogram",
            rate_hist_cfg.get("enabled", catalog.get("*_*_rate_hz", False)),
        ),
        bool(rate_hist_cfg.get("enabled", catalog.get("*_*_rate_hz", False))),
    )
    derived_physics_features = _normalize_derived_physics_features(
        derived_cfg_raw.get("physics_features", [])
    )

    def _materialize_derived_space(
        dict_in: pd.DataFrame,
        data_in: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str | None, list[str], list[str]]:
        (
            dict_derived,
            data_derived,
            derived_rate_col,
            derived_rate_sources,
        ) = _append_derived_tt_global_rate_column(
            dict_df=dict_in,
            data_df=data_in,
            prefix_selector=derived_tt_prefix,
            trigger_type_allowlist=derived_trigger_types,
            include_to_tt_rate_hz=bool(derived_include_to_tt),
        )
        if (
            derived_rate_col is None
            and global_rate_col in dict_in.columns
            and global_rate_col in data_in.columns
        ):
            derived_rate_col = str(global_rate_col)
        if derived_rate_col is None:
            return dict_derived, data_derived, None, [], []
        (
            dict_derived,
            data_derived,
            derived_physics_cols,
        ) = _append_derived_physics_feature_columns(
            dict_df=dict_derived,
            data_df=data_derived,
            rate_column=derived_rate_col,
            physics_features=derived_physics_features,
        )
        return (
            dict_derived,
            data_derived,
            str(derived_rate_col),
            [str(c) for c in derived_rate_sources],
            [str(c) for c in derived_physics_cols],
        )

    if isinstance(feature_cfg, str):
        mode = feature_cfg.strip().lower()
        selected_feature_modes = {
            "step12_selected",
            "step_1_2_selected",
            "selected_from_step12",
            "selected_from_step_1_2",
        }
        if mode in selected_feature_modes:
            selected_path = (
                Path(step12_selected_path).expanduser()
                if step12_selected_path is not None
                else DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS
            )
            selected_from_step12, selected_info = _load_step12_selected_feature_columns(selected_path)
            dict_selected, data_selected, _, _, _ = _materialize_derived_space(dict_df, data_df)
            selected = [
                c
                for c in selected_from_step12
                if c in dict_selected.columns and c in data_selected.columns
            ]
            if selected:
                log.info(
                    "Using STEP 1.2 selected features (%d) from %s.",
                    len(selected),
                    selected_path,
                )
                return dict_selected, data_selected, selected, "step12_selected"
            log.warning(
                "STEP 1.2 selected features unavailable/empty at %s (%s); falling back to derived mode.",
                selected_path,
                selected_info.get("error", "no_selected_columns"),
            )
            mode = "derived"
        if mode == "auto":
            return dict_df, data_df, auto_feature_cols, "auto"
        if mode == "derived":
            (
                dict_derived,
                data_derived,
                derived_rate_col,
                derived_rate_sources,
                derived_physics_cols,
            ) = _materialize_derived_space(dict_df, data_df)
            if derived_rate_col is None:
                raise ValueError(
                    "No derived feature global-rate source available. "
                    "TT trigger-type sum could not be built and fallback global-rate column is missing."
                )
            feat_dict = _shared_derived_feature_columns(
                dict_derived,
                rate_column=derived_rate_col,
                trigger_type_rate_columns=(
                    derived_rate_sources
                    if bool(derived_include_trigger_rates)
                    else None
                ),
                include_rate_histogram=bool(derived_include_hist),
                physics_feature_columns=derived_physics_cols,
            )
            feat_data = _shared_derived_feature_columns(
                data_derived,
                rate_column=derived_rate_col,
                trigger_type_rate_columns=(
                    derived_rate_sources
                    if bool(derived_include_trigger_rates)
                    else None
                ),
                include_rate_histogram=bool(derived_include_hist),
                physics_feature_columns=derived_physics_cols,
            )
            selected = sorted(set(feat_dict) & set(feat_data))
            if not selected:
                raise ValueError(
                    "No derived feature columns found in dictionary/dataset intersection. "
                    "Expected eff_empirical_1..4 and at least one global-rate feature."
                )
            log.info(
                "Derived features: prefix=%s trigger_types=%s include_hist=%s "
                "include_trigger_rates=%s physics=%s rate_feature=%s rate_sources=%s",
                derived_tt_prefix,
                derived_trigger_types,
                bool(derived_include_hist),
                bool(derived_include_trigger_rates),
                derived_physics_features,
                derived_rate_col,
                derived_rate_sources,
            )
            return dict_derived, data_derived, selected, "derived"
        if mode in {"config_columns", "catalog", "config_columns_json"}:
            selected, info = resolve_feature_columns_from_catalog(
                catalog=catalog,
                available_columns=sorted(set(dict_df.columns) & set(data_df.columns)),
            )
            if info.get("invalid_include_patterns"):
                log.warning(
                    "Ignoring invalid include pattern(s) in %s: %s",
                    CONFIG_COLUMNS_PATH,
                    info.get("invalid_include_patterns"),
                )
            if info.get("invalid_exclude_patterns"):
                log.warning(
                    "Ignoring invalid exclude pattern(s) in %s: %s",
                    CONFIG_COLUMNS_PATH,
                    info.get("invalid_exclude_patterns"),
                )
            if selected:
                return dict_df, data_df, selected, "config_columns"
            if auto_feature_cols:
                log.warning(
                    "No features selected by config_columns catalog; falling back to auto (%d columns).",
                    len(auto_feature_cols),
                )
                return dict_df, data_df, auto_feature_cols, "auto_fallback_from_config_columns"
            raise ValueError("No features selected by config_columns catalog and no auto fallback available.")

    requested = parse_explicit_feature_columns(feature_cfg)
    selected = [
        c for c in requested
        if c in dict_df.columns and c in data_df.columns
    ]
    if not selected:
        raise ValueError(
            "No explicit feature columns found in dictionary/dataset intersection. "
            f"Requested={requested!r}"
        )
    return dict_df, data_df, selected, "explicit"


def _load_lut(lut_path: Path) -> pd.DataFrame:
    """Load LUT CSV allowing comment-prefixed metadata header."""
    return pd.read_csv(lut_path, comment="#", low_memory=False)


def _lut_param_names(
    lut_df: pd.DataFrame,
    lut_meta_path: Path | None = None,
) -> list[str]:
    """Extract LUT parameter names from metadata JSON or LUT columns."""
    if lut_meta_path is not None and lut_meta_path.exists():
        try:
            meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
            params = meta.get("param_names", [])
            if isinstance(params, list):
                cleaned = [str(p) for p in params if str(p)]
                if cleaned:
                    return cleaned
        except Exception as exc:
            log.warning(
                "Could not parse LUT metadata at %s (%s). Falling back to LUT column scan.",
                lut_meta_path,
                exc,
            )

    params: list[str] = []
    for c in lut_df.columns:
        if not c.startswith("sigma_"):
            continue
        if "_p" in c:
            pname = c[len("sigma_"):].split("_p", 1)[0]
        elif c.endswith("_std"):
            pname = c[len("sigma_"):-len("_std")]
        else:
            continue
        if pname and pname not in params:
            params.append(pname)
    return params


def _interpolate_uncertainties(
    query_df: pd.DataFrame,
    lut_df: pd.DataFrame,
    param_names: list[str],
    quantile: float,
) -> pd.DataFrame:
    """Nearest-centre LUT interpolation with finite-value fallback per parameter.

    For each parameter and query row, select the nearest LUT row with finite
    sigma value for that parameter. If no finite sigma exists, return NaN.
    """
    if lut_df.empty or query_df.empty:
        return pd.DataFrame(index=query_df.index)

    q_label = str(int(round(float(quantile) * 100.0)))
    centre_cols = [c for c in lut_df.columns if c.endswith("_centre")]
    if not centre_cols:
        return pd.DataFrame(index=query_df.index)

    lut_centres_df = lut_df[centre_cols].apply(pd.to_numeric, errors="coerce")
    lut_centres = lut_centres_df.to_numpy(dtype=float)
    valid_centres = np.all(np.isfinite(lut_centres), axis=1)
    if not np.any(valid_centres):
        return pd.DataFrame(index=query_df.index)

    # Dimension scales for normalized distance.
    mins = np.nanmin(lut_centres[valid_centres], axis=0)
    maxs = np.nanmax(lut_centres[valid_centres], axis=0)
    ranges = maxs - mins
    ranges[~np.isfinite(ranges) | (ranges <= 0.0)] = 1.0
    dim_fallbacks = np.nanmedian(lut_centres[valid_centres], axis=0)

    n_rows = len(query_df)
    n_dims = len(centre_cols)
    query_vals = np.zeros((n_rows, n_dims), dtype=float)
    for j, cc in enumerate(centre_cols):
        dim = cc.replace("_centre", "")
        if dim in query_df.columns:
            qv = pd.to_numeric(query_df[dim], errors="coerce").to_numpy(dtype=float)
        elif dim == "n_events":
            if "n_events" in query_df.columns:
                qv = pd.to_numeric(query_df["n_events"], errors="coerce").to_numpy(dtype=float)
            elif "true_n_events" in query_df.columns:
                qv = pd.to_numeric(query_df["true_n_events"], errors="coerce").to_numpy(dtype=float)
            else:
                qv = np.full(n_rows, np.nan, dtype=float)
        else:
            qv = np.full(n_rows, np.nan, dtype=float)
        qv = np.where(np.isfinite(qv), qv, dim_fallbacks[j])
        query_vals[:, j] = qv

    d = (lut_centres[np.newaxis, :, :] - query_vals[:, np.newaxis, :]) / ranges[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    dist = np.where(valid_centres[np.newaxis, :], dist, np.inf)

    out = pd.DataFrame(index=query_df.index)
    for pname in param_names:
        pref_col = f"sigma_{pname}_p{q_label}"
        sigma_col = pref_col if pref_col in lut_df.columns else None
        if sigma_col is None:
            alt = f"sigma_{pname}_std"
            sigma_col = alt if alt in lut_df.columns else None
        if sigma_col is None:
            out[f"unc_{pname}_pct_raw"] = np.nan
            out[f"unc_{pname}_pct"] = np.nan
            continue

        sigma_vals = pd.to_numeric(lut_df[sigma_col], errors="coerce").to_numpy(dtype=float)
        valid_sigma = valid_centres & np.isfinite(sigma_vals)
        if not np.any(valid_sigma):
            out[f"unc_{pname}_pct_raw"] = np.nan
            out[f"unc_{pname}_pct"] = np.nan
            continue

        masked_dist = np.where(valid_sigma[np.newaxis, :], dist, np.inf)
        nearest_idx = np.argmin(masked_dist, axis=1)
        nearest_dist = masked_dist[np.arange(n_rows), nearest_idx]
        raw = sigma_vals[nearest_idx]
        raw = np.where(np.isfinite(nearest_dist), raw, np.nan)

        # Fallback to median finite sigma if a row has no finite nearest.
        sigma_median = float(np.nanmedian(sigma_vals[valid_sigma]))
        raw = np.where(np.isfinite(raw), raw, sigma_median)
        out[f"unc_{pname}_pct_raw"] = raw
        out[f"unc_{pname}_pct"] = np.abs(raw)
    return out


def _compute_density_center_series(
    *,
    config: dict,
    time_df: pd.DataFrame,
    flux_col: str,
    eff_pref: str,
    dictionary_path: Path,
    dataset_template_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Load STEP 3.2 diagnostic center series (no recomputation in STEP 3.3)."""
    _ = dictionary_path
    _ = dataset_template_path
    if time_df.empty:
        return None, None, None

    cfg_32 = config.get("step_3_2", {}) if isinstance(config, dict) else {}
    cfg_33 = config.get("step_3_3", {}) if isinstance(config, dict) else {}
    center_csv_cfg = (
        cfg_33.get("step32_diagnostic_center_csv", None)
        if isinstance(cfg_33, dict) else None
    )
    if center_csv_cfg in (None, "", "null", "None"):
        center_csv_cfg = (
            cfg_32.get("diagnostic_center_series_csv", None)
            if isinstance(cfg_32, dict) else None
        )
    center_path = _resolve_input_path(center_csv_cfg or DEFAULT_STEP32_DIAGNOSTIC_CENTER)
    if not center_path.exists():
        log.warning(
            "STEP 3.2 diagnostic center CSV not found: %s. "
            "STEP 3.3 overlay will omit the center curve.",
            center_path,
        )
        return None, None, None

    try:
        center_df = pd.read_csv(center_path, low_memory=False)
    except Exception as exc:
        log.warning("Could not read STEP 3.2 diagnostic center CSV %s: %s", center_path, exc)
        return None, None, None
    if center_df.empty:
        log.warning("STEP 3.2 diagnostic center CSV is empty: %s", center_path)
        return None, None, None

    flux_center_col = None
    for c in ("center_flux_cm2_min", "diagnostic_center_flux", "center_flux", flux_col):
        if c in center_df.columns:
            flux_center_col = c
            break
    if flux_center_col is None:
        log.warning("STEP 3.2 center CSV has no flux center column: %s", center_path)
        return None, None, None

    eff_col_time = eff_pref
    if "eff_column_used" in center_df.columns:
        eff_candidates = center_df["eff_column_used"].dropna().astype(str)
        if not eff_candidates.empty:
            eff_col_time = str(eff_candidates.iloc[0]).strip() or eff_pref
    else:
        try:
            eff_col_time = _choose_eff_column(time_df, eff_pref)
        except KeyError:
            eff_col_time = eff_pref

    eff_center_col = None
    for c in ("center_eff", "diagnostic_center_eff", eff_col_time, eff_pref):
        if c in center_df.columns:
            eff_center_col = c
            break
    if eff_center_col is None:
        log.warning("STEP 3.2 center CSV has no efficiency center column: %s", center_path)
        return None, None, eff_col_time

    n_rows = len(time_df)
    center_flux = np.full(n_rows, np.nan, dtype=float)
    center_eff = np.full(n_rows, np.nan, dtype=float)
    aligned_method = None

    if "file_index" in center_df.columns and "file_index" in time_df.columns:
        left = pd.DataFrame({
            "_row_idx": np.arange(n_rows, dtype=int),
            "file_index": pd.to_numeric(time_df["file_index"], errors="coerce"),
        })
        right = pd.DataFrame({
            "file_index": pd.to_numeric(center_df["file_index"], errors="coerce"),
            "_center_flux": pd.to_numeric(center_df[flux_center_col], errors="coerce"),
            "_center_eff": pd.to_numeric(center_df[eff_center_col], errors="coerce"),
        })
        aligned = left.merge(right, on="file_index", how="left", sort=False)
        center_flux = aligned["_center_flux"].to_numpy(dtype=float)
        center_eff = aligned["_center_eff"].to_numpy(dtype=float)
        if np.isfinite(center_flux).any() and np.isfinite(center_eff).any():
            aligned_method = "file_index"

    if aligned_method is None and "elapsed_hours" in center_df.columns and "elapsed_hours" in time_df.columns:
        left = pd.DataFrame({
            "_row_idx": np.arange(n_rows, dtype=int),
            "elapsed_hours_key": np.round(pd.to_numeric(time_df["elapsed_hours"], errors="coerce"), 10),
        })
        right = pd.DataFrame({
            "elapsed_hours_key": np.round(pd.to_numeric(center_df["elapsed_hours"], errors="coerce"), 10),
            "_center_flux": pd.to_numeric(center_df[flux_center_col], errors="coerce"),
            "_center_eff": pd.to_numeric(center_df[eff_center_col], errors="coerce"),
        })
        aligned = left.merge(right, on="elapsed_hours_key", how="left", sort=False)
        center_flux = aligned["_center_flux"].to_numpy(dtype=float)
        center_eff = aligned["_center_eff"].to_numpy(dtype=float)
        if np.isfinite(center_flux).any() and np.isfinite(center_eff).any():
            aligned_method = "elapsed_hours"

    if aligned_method is None:
        src_flux = pd.to_numeric(center_df[flux_center_col], errors="coerce").to_numpy(dtype=float)
        src_eff = pd.to_numeric(center_df[eff_center_col], errors="coerce").to_numpy(dtype=float)
        if len(src_flux) == n_rows:
            center_flux = src_flux
            center_eff = src_eff
            aligned_method = "row_order"
        else:
            take = min(n_rows, len(src_flux))
            center_flux[:take] = src_flux[:take]
            center_eff[:take] = src_eff[:take]
            aligned_method = "row_order_trim_pad"
            log.warning(
                "STEP 3.2 center-length mismatch (center=%d, time=%d). "
                "Applied positional trim/pad for first %d rows.",
                len(src_flux),
                n_rows,
                take,
            )

    finite_pairs = np.isfinite(center_flux) & np.isfinite(center_eff)
    if not np.any(finite_pairs):
        log.warning(
            "STEP 3.2 center CSV loaded but produced no finite aligned rows: %s",
            center_path,
        )
        return None, None, eff_col_time
    log.info(
        "Using STEP 3.2 center series from %s (alignment=%s, finite_pairs=%d/%d).",
        center_path,
        aligned_method,
        int(np.sum(finite_pairs)),
        int(n_rows),
    )
    return center_flux, center_eff, eff_col_time


def _time_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return numeric time axis values and axis label."""
    if "elapsed_hours" in df.columns:
        x = pd.to_numeric(df["elapsed_hours"], errors="coerce").to_numpy(dtype=float)
        return x, "Elapsed time [hours]"
    if "file_index" in df.columns:
        x = pd.to_numeric(df["file_index"], errors="coerce").to_numpy(dtype=float)
        return x, "File index"
    return np.arange(len(df), dtype=float), "Index"


def _apply_mean_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
    """Apply stripes at 1%-of-mean increments, uniformly across the y-axis."""
    y_min, y_max = ax.get_ylim()
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        return
    span = y_max - y_min
    if span <= 0.0:
        return

    y_arr = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(y_arr)
    if not np.any(valid):
        return
    mean_val = float(np.mean(y_arr[valid]))

    band = abs(mean_val) * 0.01
    if not np.isfinite(band) or band <= 0.0:
        band = span * 0.01
    if band <= 0.0:
        return

    ax.set_facecolor("#FFFFFF")
    idx = int(np.floor((y_min - mean_val) / band))
    y0 = mean_val + idx * band
    while y0 < y_max:
        y1 = y0 + band
        lo = max(y0, y_min)
        hi = min(y1, y_max)
        color = "#FFFFFF" if (idx % 2 == 0) else "#D8DDE4"
        if hi > lo:
            ax.axhspan(lo, hi, facecolor=color, alpha=1.0, linewidth=0.0, zorder=0)
        y0 = y1
        idx += 1
    ax.set_ylim(y_min, y_max)


def _plot_series_panel(
    ax: plt.Axes,
    x: np.ndarray,
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    unc_abs: np.ndarray | None,
    ylabel: str,
    title: str,
) -> None:
    """Plot one time-series comparison panel with optional uncertainty band."""
    m = np.isfinite(x) & np.isfinite(true_vals) & np.isfinite(est_vals)
    if not np.any(m):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return

    xo = x[m]
    yt = true_vals[m]
    ye = est_vals[m]
    order = np.argsort(xo)
    xo = xo[order]
    yt = yt[order]
    ye = ye[order]

    if unc_abs is not None and len(unc_abs) == len(x):
        yu = np.asarray(unc_abs, dtype=float)[m][order]
        yu = np.where(np.isfinite(yu), np.abs(yu), np.nan)
        valid_band = np.isfinite(yu)
        if np.any(valid_band):
            ax.fill_between(
                xo[valid_band],
                ye[valid_band] - yu[valid_band],
                ye[valid_band] + yu[valid_band],
                color="#FF7F0E",
                alpha=0.20,
                linewidth=0.0,
                label="Estimated ± uncertainty",
            )

    ax.scatter(xo, yt, s=18, facecolors="white", edgecolors="#1F77B4", linewidths=0.8, label="Simulated (true)")
    ax.scatter(xo, ye, s=18, color="#D62728", alpha=0.9, linewidths=0.0, label="Estimated")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _apply_mean_striped_background(ax, np.concatenate([yt, ye]))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _plot_diag_panel(
    ax: plt.Axes,
    true_vals: np.ndarray,
    est_vals: np.ndarray,
    unc_abs: np.ndarray | None,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot true-vs-estimated diagonal panel with optional vertical uncertainty."""
    m = np.isfinite(true_vals) & np.isfinite(est_vals)
    if not np.any(m):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        return

    xt = true_vals[m]
    ye = est_vals[m]
    if unc_abs is not None and len(unc_abs) == len(true_vals):
        yu = np.asarray(unc_abs, dtype=float)[m]
        yu = np.where(np.isfinite(yu), np.abs(yu), np.nan)
        valid_yu = np.isfinite(yu)
        if np.any(valid_yu):
            ax.errorbar(
                xt[valid_yu],
                ye[valid_yu],
                yerr=yu[valid_yu],
                fmt="none",
                ecolor="#D62728",
                alpha=0.20,
                elinewidth=0.8,
                capsize=0,
                zorder=1,
            )

    ax.scatter(xt, ye, s=22, color="#D62728", alpha=0.85, linewidths=0.0, zorder=2)
    lo = float(np.nanmin(np.concatenate([xt, ye])))
    hi = float(np.nanmax(np.concatenate([xt, ye])))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-6)
    else:
        pad = 0.03 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    _apply_mean_striped_background(ax, np.concatenate([xt, ye]))
    ax.grid(True, alpha=0.25)


def _plot_step32_style_overlay(
    *,
    complete_df: pd.DataFrame | None,
    time_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    flux_col: str,
    eff_col_time: str,
    est_flux_col: str,
    est_eff_col: str,
    flux_unc_abs_col: str | None,
    eff_unc_abs_col: str | None,
    center_flux: np.ndarray | None,
    center_eff: np.ndarray | None,
    path: Path,
) -> None:
    """Plot complete+discretized+STEP3.2-center+estimated without global rate."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 9.1),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0]},
    )

    # Discretized STEP 3.1 trajectory
    x_disc, x_label = _time_axis(time_df)
    y_flux_disc = pd.to_numeric(time_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
    y_eff_disc = pd.to_numeric(time_df.get(eff_col_time), errors="coerce").to_numpy(dtype=float)
    flux_stripe_vals = [y_flux_disc]
    eff_stripe_vals = [y_eff_disc]

    # Complete trajectory
    if complete_df is not None and not complete_df.empty:
        x_comp, _ = _time_axis(complete_df)
        y_flux_comp = pd.to_numeric(complete_df.get(flux_col), errors="coerce").to_numpy(dtype=float)
        y_eff_comp = pd.to_numeric(complete_df.get(eff_col_time), errors="coerce").to_numpy(dtype=float)
        flux_stripe_vals.append(y_flux_comp)
        eff_stripe_vals.append(y_eff_comp)
        m0c = np.isfinite(x_comp) & np.isfinite(y_flux_comp)
        m1c = np.isfinite(x_comp) & np.isfinite(y_eff_comp)
        if np.any(m0c):
            axes[1].scatter(
                x_comp[m0c], y_flux_comp[m0c], s=7, color="#1F77B4", alpha=0.55, linewidths=0.0, label="Complete"
            )
        if np.any(m1c):
            axes[2].scatter(
                x_comp[m1c], y_eff_comp[m1c], s=7, color="#FF7F0E", alpha=0.55, linewidths=0.0, label="Complete"
            )

    m0d = np.isfinite(x_disc) & np.isfinite(y_flux_disc)
    m1d = np.isfinite(x_disc) & np.isfinite(y_eff_disc)
    if np.any(m0d):
        axes[1].scatter(
            x_disc[m0d],
            y_flux_disc[m0d],
            s=18,
            facecolors="white",
            edgecolors="#1F77B4",
            linewidths=0.8,
            label="Discretized",
        )
    if np.any(m1d):
        axes[2].scatter(
            x_disc[m1d],
            y_eff_disc[m1d],
            s=18,
            facecolors="white",
            edgecolors="#FF7F0E",
            linewidths=0.8,
            label="Discretized",
        )

    # STEP 3.2 reference center (loaded from STEP 3.2 outputs, no recomputation here).
    if center_flux is not None and len(center_flux) == len(x_disc):
        c_flux = np.asarray(center_flux, dtype=float)
        flux_stripe_vals.append(c_flux)
        mc = np.isfinite(x_disc) & np.isfinite(c_flux)
        if np.any(mc):
            axes[1].plot(
                x_disc[mc], c_flux[mc],
                color="#17BECF", linewidth=1.0, linestyle="-.", marker="s", markersize=2.8,
                markerfacecolor="#17BECF", markeredgewidth=0.0, alpha=0.9,
                label="STEP 3.2 reference center",
            )
    if center_eff is not None and len(center_eff) == len(x_disc):
        c_eff = np.asarray(center_eff, dtype=float)
        eff_stripe_vals.append(c_eff)
        mc = np.isfinite(x_disc) & np.isfinite(c_eff)
        if np.any(mc):
            axes[2].plot(
                x_disc[mc], c_eff[mc],
                color="#BCBD22", linewidth=1.0, linestyle="-.", marker="s", markersize=2.8,
                markerfacecolor="#BCBD22", markeredgewidth=0.0, alpha=0.9,
                label="STEP 3.2 reference center",
            )

    # Estimated series (+ uncertainty)
    x_est, _ = _time_axis(merged_df)
    y_flux_est = pd.to_numeric(merged_df.get(est_flux_col), errors="coerce").to_numpy(dtype=float)
    y_eff_est = pd.to_numeric(merged_df.get(est_eff_col), errors="coerce").to_numpy(dtype=float)
    flux_stripe_vals.append(y_flux_est)
    eff_stripe_vals.append(y_eff_est)
    u_flux = (
        pd.to_numeric(merged_df.get(flux_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if flux_unc_abs_col and flux_unc_abs_col in merged_df.columns
        else None
    )
    u_eff = (
        pd.to_numeric(merged_df.get(eff_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if eff_unc_abs_col and eff_unc_abs_col in merged_df.columns
        else None
    )

    m0e = np.isfinite(x_est) & np.isfinite(y_flux_est)
    if np.any(m0e):
        order = np.argsort(x_est[m0e])
        xe = x_est[m0e][order]
        ye = y_flux_est[m0e][order]
        axes[1].plot(
            xe, ye,
            color="#D62728", linewidth=1.0, linestyle="-", marker="o", markersize=3.0,
            markerfacecolor="#D62728", markeredgewidth=0.0, alpha=0.9,
            label="Estimated",
        )
        if u_flux is not None and len(u_flux) == len(x_est):
            ue = np.abs(u_flux[m0e][order])
            valid_u = np.isfinite(ue)
            if np.any(valid_u):
                axes[1].fill_between(
                    xe[valid_u], ye[valid_u] - ue[valid_u], ye[valid_u] + ue[valid_u],
                    color="#D62728", alpha=0.16, linewidth=0.0, label="Estimated ± uncertainty",
                )

    m1e = np.isfinite(x_est) & np.isfinite(y_eff_est)
    if np.any(m1e):
        order = np.argsort(x_est[m1e])
        xe = x_est[m1e][order]
        ye = y_eff_est[m1e][order]
        axes[2].plot(
            xe, ye,
            color="#8C564B", linewidth=1.0, linestyle="-", marker="o", markersize=3.0,
            markerfacecolor="#8C564B", markeredgewidth=0.0, alpha=0.9,
            label="Estimated",
        )
        if u_eff is not None and len(u_eff) == len(x_est):
            ue = np.abs(u_eff[m1e][order])
            valid_u = np.isfinite(ue)
            if np.any(valid_u):
                axes[2].fill_between(
                    xe[valid_u], ye[valid_u] - ue[valid_u], ye[valid_u] + ue[valid_u],
                    color="#8C564B", alpha=0.16, linewidth=0.0, label="Estimated ± uncertainty",
                )

    # Best dictionary distance (same concept as STEP 4.2 distance-vs-time).
    y_dist = (
        pd.to_numeric(merged_df.get("best_distance"), errors="coerce").to_numpy(dtype=float)
        if "best_distance" in merged_df.columns
        else np.full(len(x_est), np.nan, dtype=float)
    )
    m_dist = np.isfinite(x_est) & np.isfinite(y_dist)
    if np.any(m_dist):
        order = np.argsort(x_est[m_dist])
        xd = x_est[m_dist][order]
        yd = y_dist[m_dist][order]
        axes[0].plot(
            xd,
            yd,
            color="#9467BD",
            linewidth=1.05,
            linestyle="-",
            marker="o",
            markersize=2.6,
            markerfacecolor="#9467BD",
            markeredgewidth=0.0,
            alpha=0.9,
            label="Best dictionary distance (best_distance)",
        )
        _apply_mean_striped_background(axes[0], yd)
        axes[0].legend(loc="best", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No finite best_distance values", ha="center", va="center")
    axes[0].set_ylabel("best_distance")
    axes[0].set_title("Best dictionary distance vs time")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_ylabel("flux_cm2_min")
    axes[1].set_title("Flux: complete + discretized + STEP 3.2 center + estimated")
    _apply_mean_striped_background(axes[1], np.concatenate(flux_stripe_vals))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    axes[2].set_ylabel("eff")
    axes[2].set_xlabel(x_label)
    axes[2].set_title("Efficiency: complete + discretized + STEP 3.2 center + estimated")
    _apply_mean_striped_background(axes[2], np.concatenate(eff_stripe_vals))
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    _save_figure(fig, path, dpi=160)
    plt.close(fig)


def _plot_flux_recovery_vs_global_rate(
    *,
    df: pd.DataFrame,
    complete_df: pd.DataFrame | None,
    flux_complete_col: str,
    eff_complete_col: str,
    flux_true_col: str,
    eff_true_col: str,
    flux_est_col: str,
    flux_unc_abs_col: str | None,
    global_rate_col: str,
    path: Path,
) -> None:
    """Plot a 4-panel story using complete curve as simulated reference when available."""
    def _apply_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
        """Apply stripes at 1%-of-mean increments, uniformly across the y-axis."""
        y_min, y_max = ax.get_ylim()
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        span = y_max - y_min
        if span <= 0.0:
            return

        valid = np.isfinite(y_vals)
        if not np.any(valid):
            return
        mean_val = float(np.mean(y_vals[valid]))

        band = abs(mean_val) * 0.01
        if not np.isfinite(band) or band <= 0.0:
            band = span * 0.01
        if band <= 0.0:
            return

        ax.set_facecolor("#FFFFFF")
        idx = int(np.floor((y_min - mean_val) / band))
        y0 = mean_val + idx * band
        while y0 < y_max:
            y1 = y0 + band
            lo = max(y0, y_min)
            hi = min(y1, y_max)
            color = "#FFFFFF" if (idx % 2 == 0) else "#D8DDE4"
            if hi > lo:
                ax.axhspan(lo, hi, facecolor=color, alpha=1.0, linewidth=0.0, zorder=0)
            y0 = y1
            idx += 1
        ax.set_ylim(y_min, y_max)

    x, x_label = _time_axis(df)
    true_flux = pd.to_numeric(df.get(flux_true_col), errors="coerce").to_numpy(dtype=float)
    true_eff = pd.to_numeric(df.get(eff_true_col), errors="coerce").to_numpy(dtype=float)
    est_flux = pd.to_numeric(df.get(flux_est_col), errors="coerce").to_numpy(dtype=float)
    rate_vals = pd.to_numeric(df.get(global_rate_col), errors="coerce").to_numpy(dtype=float)
    unc_flux = (
        pd.to_numeric(df.get(flux_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if flux_unc_abs_col and flux_unc_abs_col in df.columns
        else None
    )

    # Prefer complete-curve references for simulated flux/efficiency.
    x_ref_flux = x
    y_ref_flux = true_flux
    x_ref_eff = x
    y_ref_eff = true_eff
    ref_flux_label = "Simulated flux (discretized)"
    ref_eff_label = "Simulated efficiency (discretized)"
    if complete_df is not None and not complete_df.empty:
        if flux_complete_col in complete_df.columns:
            xc, _ = _time_axis(complete_df)
            yc = pd.to_numeric(complete_df.get(flux_complete_col), errors="coerce").to_numpy(dtype=float)
            if len(xc) == len(yc):
                x_ref_flux = xc
                y_ref_flux = yc
                ref_flux_label = "Simulated flux (complete)"
        if eff_complete_col in complete_df.columns:
            xc, _ = _time_axis(complete_df)
            yc = pd.to_numeric(complete_df.get(eff_complete_col), errors="coerce").to_numpy(dtype=float)
            if len(xc) == len(yc):
                x_ref_eff = xc
                y_ref_eff = yc
                ref_eff_label = "Simulated efficiency (complete)"

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.6, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.15]},
    )
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, alpha=0.24)

    # 1) Simulated flux.
    m_true_flux = np.isfinite(x_ref_flux) & np.isfinite(y_ref_flux)
    if np.any(m_true_flux):
        order = np.argsort(x_ref_flux[m_true_flux])
        xs = x_ref_flux[m_true_flux][order]
        ys = y_ref_flux[m_true_flux][order]
        axes[0].plot(
            xs,
            ys,
            color="#1F77B4",
            linewidth=1.2,
            marker="o",
            markersize=3.1,
            markerfacecolor="white",
            markeredgewidth=0.7,
            alpha=0.92,
            label=ref_flux_label,
        )
    axes[0].set_ylabel("Sim. flux")
    _apply_striped_background(axes[0], y_ref_flux)
    axes[0].legend(loc="best", fontsize=8)

    # 2) Simulated efficiency.
    m_true_eff = np.isfinite(x_ref_eff) & np.isfinite(y_ref_eff)
    if np.any(m_true_eff):
        order = np.argsort(x_ref_eff[m_true_eff])
        xs = x_ref_eff[m_true_eff][order]
        ys = y_ref_eff[m_true_eff][order]
        axes[1].plot(
            xs,
            ys,
            color="#FF7F0E",
            linewidth=1.15,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=0.65,
            alpha=0.90,
            label=ref_eff_label,
        )
    axes[1].set_ylabel("Sim. eff")
    _apply_striped_background(axes[1], y_ref_eff)
    axes[1].legend(loc="best", fontsize=8)

    # 3) Global rate.
    m_rate = np.isfinite(x) & np.isfinite(rate_vals)
    if np.any(m_rate):
        order = np.argsort(x[m_rate])
        xr = x[m_rate][order]
        yr = rate_vals[m_rate][order]
        axes[2].plot(
            xr,
            yr,
            color="#2E8B57",
            linewidth=2.4,
            alpha=0.46,
            solid_capstyle="round",
            label=f"Global rate ({global_rate_col})",
        )
    axes[2].set_ylabel("Global rate")
    _apply_striped_background(axes[2], rate_vals)
    axes[2].legend(loc="best", fontsize=8)

    # 4) Estimated reconstructed flux (+ uncertainty), with true flux reference.
    m_est_flux = np.isfinite(x) & np.isfinite(est_flux)
    if np.any(m_est_flux):
        order = np.argsort(x[m_est_flux])
        xe = x[m_est_flux][order]
        ye = est_flux[m_est_flux][order]
        axes[3].plot(
            xe,
            ye,
            color="#D62728",
            linewidth=1.3,
            marker="o",
            markersize=3.0,
            markerfacecolor="#D62728",
            markeredgewidth=0.0,
            alpha=0.88,
            label="Estimated reconstructed flux",
            zorder=3,
        )
        if unc_flux is not None and len(unc_flux) == len(x):
            ue = np.abs(np.asarray(unc_flux, dtype=float)[m_est_flux][order])
            valid_ue = np.isfinite(ue)
            if np.any(valid_ue):
                axes[3].fill_between(
                    xe[valid_ue],
                    ye[valid_ue] - ue[valid_ue],
                    ye[valid_ue] + ue[valid_ue],
                    color="#D62728",
                    alpha=0.16,
                    linewidth=0.0,
                    label="Estimated ± uncertainty",
                    zorder=2,
                )
    if np.any(m_true_flux):
        order = np.argsort(x_ref_flux[m_true_flux])
        xs = x_ref_flux[m_true_flux][order]
        ys = y_ref_flux[m_true_flux][order]
        axes[3].plot(
            xs,
            ys,
            color="#1F77B4",
            linewidth=1.0,
            linestyle="--",
            alpha=0.60,
            label=f"{ref_flux_label} reference",
            zorder=1,
        )
    axes[3].set_ylabel("Estimated flux")
    axes[3].set_xlabel(x_label)
    _apply_striped_background(axes[3], est_flux)
    axes[3].legend(loc="best", fontsize=8)

    # Harmonize number of 1%-bands across all four subplots (visual-only, preserve axis centers)
    # Determine per-axis 1% size using the same fallback logic as the striped background helper.
    y_sources = [y_ref_flux, y_ref_eff, rate_vals, est_flux]
    band_sizes = []
    current_bands = []
    centers = []
    for ax, y_arr in zip(axes, y_sources):
        y_min, y_max = ax.get_ylim()
        span = y_max - y_min
        valid = np.isfinite(y_arr)
        if np.any(valid):
            mean_val = float(np.mean(y_arr[valid]))
            band = abs(mean_val) * 0.01
        else:
            band = span * 0.01
        if not np.isfinite(band) or band <= 0.0:
            band = max(span * 0.01, 1e-12)
        band_sizes.append(band)
        centers.append(0.5 * (y_min + y_max))
        current_bands.append((span / band) if band > 0.0 else 0.0)

    # Target number of 1%-bands (take ceiling of the maximum observed)
    N = int(np.ceil(max(current_bands))) if current_bands else 0

    # Apply new y-limits (preserve center) and set 1%-spaced ticks (values in original units).
    # Then redraw the 1%-striped background so stripes cover the NEW ylim span (visual-only).
    if N > 0:
        for ax, center, band, y_arr in zip(axes, centers, band_sizes, y_sources):
            target_span = N * band
            new_min = center - 0.5 * target_span
            new_max = center + 0.5 * target_span
            ax.set_ylim(new_min, new_max)
            # set tick positions every 1% but limit tick count to avoid label overlap
            # keep ticks aligned to the 1%-band grid but sample them sparsely when N is large
            max_ticks = 9
            n_bands = max(1, int(round(target_span / band)))
            step_bands = max(1, int(np.ceil(n_bands / max_ticks)))
            # build symmetric indices around center on the band grid
            k_min = - (n_bands // 2)
            k_max = (n_bands // 2) + (1 if (n_bands % 2 == 0) else 0)
            ks = list(range(k_min, k_max + 1, step_bands))
            ticks = [center + k * band for k in ks]
            ax.set_yticks(ticks)
            # format tick labels to 2 decimal places to avoid excessive precision
            try:
                ax.set_yticklabels([f"{v:.2f}" for v in ticks])
            except Exception:
                # fallback: use a numeric formatter if direct set fails
                try:
                    from matplotlib.ticker import FormatStrFormatter

                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                except Exception:
                    pass
            # redraw 1%-striped background to fill the expanded ylim range
            try:
                _apply_striped_background(ax, y_arr)
            except Exception:
                # non-critical: leave background as-is if redraw fails
                pass

    fig.suptitle(
        "Flux-recovery story: simulated flux/efficiency -> global-rate response -> reconstructed flux",
        fontsize=11,
    )

    fig.tight_layout()
    _save_figure(fig, path, dpi=170)
    plt.close(fig)


def _make_plots(
    df: pd.DataFrame,
    *,
    flux_true_col: str,
    flux_est_col: str,
    flux_unc_abs_col: str | None,
    eff_true_col: str,
    eff_est_col: str,
    eff_unc_abs_col: str | None,
) -> None:
    """Generate requested 2x2 diagnostic plot (transposed layout)."""
    x, x_label = _time_axis(df)
    true_flux = pd.to_numeric(df.get(flux_true_col), errors="coerce").to_numpy(dtype=float)
    est_flux = pd.to_numeric(df.get(flux_est_col), errors="coerce").to_numpy(dtype=float)
    true_eff = pd.to_numeric(df.get(eff_true_col), errors="coerce").to_numpy(dtype=float)
    est_eff = pd.to_numeric(df.get(eff_est_col), errors="coerce").to_numpy(dtype=float)
    unc_flux_abs = (
        pd.to_numeric(df.get(flux_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if flux_unc_abs_col and flux_unc_abs_col in df.columns
        else None
    )
    unc_eff_abs = (
        pd.to_numeric(df.get(eff_unc_abs_col), errors="coerce").to_numpy(dtype=float)
        if eff_unc_abs_col and eff_unc_abs_col in df.columns
        else None
    )

    # Main 2x2 diagnostic plot (transposed w.r.t. previous layout).
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    _plot_series_panel(
        axes[0, 0],
        x=x,
        true_vals=true_flux,
        est_vals=est_flux,
        unc_abs=unc_flux_abs,
        ylabel="flux_cm2_min",
        title="Flux time series",
    )
    _plot_diag_panel(
        axes[0, 1],
        true_vals=true_flux,
        est_vals=est_flux,
        unc_abs=unc_flux_abs,
        xlabel="Simulated flux",
        ylabel="Estimated flux",
        title="Flux diagonal (y = x)",
    )
    _plot_series_panel(
        axes[1, 0],
        x=x,
        true_vals=true_eff,
        est_vals=est_eff,
        unc_abs=unc_eff_abs,
        ylabel="eff",
        title="Efficiency time series",
    )
    axes[1, 0].set_xlabel(x_label)
    _plot_diag_panel(
        axes[1, 1],
        true_vals=true_eff,
        est_vals=est_eff,
        unc_abs=unc_eff_abs,
        xlabel="Simulated efficiency",
        ylabel="Estimated efficiency",
        title="Efficiency diagonal (y = x)",
    )
    fig.suptitle("STEP 3.3 correction diagnostics", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "correction_overview_2x2.png", dpi=160)
    plt.close(fig)


def main() -> int:
    """Run STEP 3.3 correction workflow."""
    parser = argparse.ArgumentParser(
        description="Step 3.3: Apply dictionary inference + LUT uncertainty to synthetic dataset."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--synthetic-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--lut-csv", default=None)
    parser.add_argument("--lut-meta-json", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()
    cfg_21_raw = config.get("step_2_1", {})
    if not isinstance(cfg_21_raw, dict):
        cfg_21_raw = {}
    cfg_21, feature_cfg_sources = _merge_common_feature_space_cfg(config, cfg_21_raw)
    cfg_32 = config.get("step_3_2", {})
    cfg_33 = config.get("step_3_3", {})

    synthetic_csv_cfg = cfg_33.get("synthetic_dataset_csv", None)
    time_series_csv_cfg = cfg_33.get("time_series_csv", cfg_32.get("time_series_csv", None))
    complete_curve_csv_cfg = cfg_33.get("complete_curve_csv", cfg_32.get("complete_curve_csv", None))
    dictionary_csv_cfg = cfg_33.get("dictionary_csv", None)
    dataset_template_csv_cfg = cfg_33.get("dataset_template_csv", cfg_32.get("dataset_template_csv", None))
    lut_csv_cfg = cfg_33.get("uncertainty_lut_csv", None)
    lut_meta_cfg = cfg_33.get("uncertainty_lut_meta_json", None)
    step32_center_csv_cfg = cfg_33.get(
        "step32_diagnostic_center_csv",
        cfg_32.get("diagnostic_center_series_csv", None),
    )

    synthetic_path = _resolve_input_path(args.synthetic_csv or synthetic_csv_cfg or DEFAULT_SYNTHETIC_DATASET)
    time_series_path = _resolve_input_path(time_series_csv_cfg or DEFAULT_TIME_SERIES)
    complete_curve_path = _resolve_input_path(complete_curve_csv_cfg or DEFAULT_COMPLETE_CURVE)
    dictionary_path = _resolve_input_path(args.dictionary_csv or dictionary_csv_cfg or DEFAULT_DICTIONARY)
    dataset_template_path = _resolve_input_path(dataset_template_csv_cfg or DEFAULT_DATASET_TEMPLATE)
    lut_path = _resolve_input_path(args.lut_csv or lut_csv_cfg or DEFAULT_LUT)
    lut_meta_path = _resolve_input_path(args.lut_meta_json or lut_meta_cfg or DEFAULT_LUT_META)
    step32_center_path = _resolve_input_path(step32_center_csv_cfg or DEFAULT_STEP32_DIAGNOSTIC_CENTER)

    for label, p in (
        ("Synthetic dataset", synthetic_path),
        ("Dictionary", dictionary_path),
        ("Uncertainty LUT", lut_path),
    ):
        if not p.exists():
            log.error("%s CSV not found: %s", label, p)
            return 1

    feature_columns_cfg = cfg_21.get("feature_columns", "auto")
    selected_feature_columns_path_cfg = cfg_21.get("selected_feature_columns_path", None)
    step12_selected_path = (
        _resolve_input_path(selected_feature_columns_path_cfg)
        if selected_feature_columns_path_cfg
        else DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS
    )
    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    interpolation_k_raw = cfg_21.get("interpolation_k", None)
    if interpolation_k_raw in (None, "", "null", "None"):
        interpolation_k: int | None = None
    else:
        interpolation_k = int(interpolation_k_raw)
    inverse_mapping_cfg = resolve_inverse_mapping_cfg(
        inverse_mapping_cfg=cfg_21.get("inverse_mapping", {}),
        interpolation_k=interpolation_k,
        histogram_distance_weight=float(cfg_21.get("histogram_distance_weight", 1.0)),
        histogram_distance_blend_mode=str(cfg_21.get("histogram_distance_blend_mode", "normalized")),
    )
    include_global_rate = _safe_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    exclude_same_file = _safe_bool(cfg_33.get("exclude_same_file", True), True)
    uncertainty_quantile = _safe_float(cfg_33.get("uncertainty_quantile", 0.68), 0.68)
    uncertainty_quantile = float(np.clip(uncertainty_quantile, 0.0, 1.0))

    if (
        cfg_32.get("flux_column") is not None
        or cfg_32.get("eff_column") is not None
        or config.get("step_3_1", {}).get("flux_column") is not None
        or config.get("step_3_1", {}).get("eff_column") is not None
    ):
        log.warning(
            "Deprecated keys step_3_2/step_3_1 flux_column/eff_column detected; ignored. "
            "Using fixed columns %s/%s.",
            CANONICAL_FLUX_COLUMN,
            CANONICAL_EFF_COLUMN,
        )
    flux_col = CANONICAL_FLUX_COLUMN
    eff_pref = CANONICAL_EFF_COLUMN

    log.info("Synthetic dataset: %s", synthetic_path)
    log.info("Time series:      %s", time_series_path)
    log.info("Complete curve:   %s", complete_curve_path)
    log.info("Dictionary:       %s", dictionary_path)
    log.info("LUT:              %s", lut_path)
    log.info("Metric=%s, k=%s, uncertainty_quantile=%.3f", distance_metric, interpolation_k, uncertainty_quantile)
    log.info(
        "Inverse mapping: selection=%s k=%s weighting=%s aggregation=%s hist_weight=%.3g hist_blend=%s",
        inverse_mapping_cfg.get("neighbor_selection"),
        ("all" if inverse_mapping_cfg.get("neighbor_count") is None else str(inverse_mapping_cfg.get("neighbor_count"))),
        inverse_mapping_cfg.get("weighting"),
        inverse_mapping_cfg.get("aggregation"),
        float(inverse_mapping_cfg.get("histogram_distance_weight", 1.0)),
        inverse_mapping_cfg.get("histogram_distance_blend_mode"),
    )

    synthetic_df = pd.read_csv(synthetic_path, low_memory=False)
    if synthetic_df.empty:
        log.error("Synthetic dataset is empty: %s", synthetic_path)
        return 1
    dictionary_df = pd.read_csv(dictionary_path, low_memory=False)
    if dictionary_df.empty:
        log.error("Dictionary table is empty: %s", dictionary_path)
        return 1

    try:
        dict_work, synthetic_work, resolved_feature_columns, feature_strategy = _resolve_inference_feature_columns(
            feature_cfg=feature_columns_cfg,
            dict_df=dictionary_df,
            data_df=synthetic_df,
            include_global_rate=include_global_rate,
            global_rate_col=global_rate_col,
            step12_selected_path=step12_selected_path,
            derived_feature_cfg=cfg_21.get("derived_features", {}),
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    log.info(
        "Feature selection: strategy=%s selected=%d (config=%r, catalog=%s)",
        feature_strategy,
        len(resolved_feature_columns),
        feature_columns_cfg,
        CONFIG_COLUMNS_PATH,
    )
    if feature_cfg_sources.get("has_common_feature_space"):
        log.info(
            "Feature config source: common_feature_space keys=%s; step_2_1 overrides=%s",
            feature_cfg_sources.get("common_keys", []),
            feature_cfg_sources.get("step_2_1_override_keys", []),
        )
    if feature_strategy == "step12_selected":
        log.info("STEP 1.2 selected-feature artifact: %s", step12_selected_path)

    time_df = pd.DataFrame()
    if time_series_path.exists():
        time_df = pd.read_csv(time_series_path, low_memory=False)
    if time_df.empty:
        # Fallback to synthetic table for discretized trajectory.
        fallback_cols = [c for c in ("file_index", "elapsed_hours", "n_events") if c in synthetic_df.columns]
        time_df = synthetic_df[fallback_cols].copy()

    complete_df = None
    if complete_curve_path.exists():
        tmp_complete = pd.read_csv(complete_curve_path, low_memory=False)
        complete_df = tmp_complete if not tmp_complete.empty else None

    try:
        eff_col = _choose_eff_column(synthetic_df, eff_pref)
    except KeyError as exc:
        log.error("%s", exc)
        return 1
    if flux_col not in synthetic_df.columns:
        log.error("Flux column '%s' not found in synthetic dataset.", flux_col)
        return 1
    if flux_col not in time_df.columns:
        time_df[flux_col] = pd.to_numeric(synthetic_df.get(flux_col), errors="coerce")
    try:
        eff_col_time = _choose_eff_column(time_df, eff_pref)
    except KeyError:
        eff_col_time = eff_col
        time_df[eff_col_time] = pd.to_numeric(synthetic_df.get(eff_col), errors="coerce")

    # ── 1) Inference over synthetic dataset ─────────────────────────
    est_df = estimate_from_dataframes(
        dict_df=dict_work,
        data_df=synthetic_work,
        feature_columns=resolved_feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
        exclude_same_file=exclude_same_file,
        inverse_mapping_cfg=inverse_mapping_cfg,
    )

    # Merge with synthetic rows for time axes and true values.
    syn_with_idx = synthetic_df.copy()
    syn_with_idx["dataset_index"] = np.arange(len(syn_with_idx), dtype=int)
    merged = pd.merge(est_df, syn_with_idx, on="dataset_index", how="left", suffixes=("", "_synthetic"))

    # Ensure key true columns exist with explicit names.
    merged[f"true_{flux_col}"] = pd.to_numeric(merged.get(flux_col), errors="coerce")
    merged[f"true_{eff_col}"] = pd.to_numeric(merged.get(eff_col), errors="coerce")
    if "n_events" in merged.columns:
        merged["n_events"] = pd.to_numeric(merged["n_events"], errors="coerce")
    elif "true_n_events" in merged.columns:
        merged["n_events"] = pd.to_numeric(merged["true_n_events"], errors="coerce")

    # ── 2) LUT uncertainty interpolation ────────────────────────────
    lut_df = _load_lut(lut_path)
    lut_params = _lut_param_names(lut_df, lut_meta_path if lut_meta_path.exists() else None)
    lut_params = [p for p in lut_params if f"est_{p}" in merged.columns]
    if not lut_params:
        log.warning("No LUT parameter matches found in estimation output. Uncertainty columns will be NaN.")

    unc_df = _interpolate_uncertainties(
        query_df=merged,
        lut_df=lut_df,
        param_names=lut_params,
        quantile=uncertainty_quantile,
    )
    merged = pd.concat([merged, unc_df], axis=1)

    for pname in lut_params:
        est_col = f"est_{pname}"
        unc_pct_col = f"unc_{pname}_pct"
        unc_abs_col = f"unc_{pname}_abs"
        if est_col in merged.columns and unc_pct_col in merged.columns:
            # Convert percent uncertainty from LUT into absolute units.
            # Use |unc_pct| to keep uncertainty non-negative even if the LUT
            # was generated from a signed-error mode.
            est_v = pd.to_numeric(merged[est_col], errors="coerce").to_numpy(dtype=float)
            true_col = f"true_{pname}"
            true_v = (
                pd.to_numeric(merged.get(true_col), errors="coerce").to_numpy(dtype=float)
                if true_col in merged.columns else np.full(len(est_v), np.nan, dtype=float)
            )
            up = pd.to_numeric(merged[unc_pct_col], errors="coerce").to_numpy(dtype=float)

            # Prefer true-based conversion; fallback to estimate-based when true is
            # not available or too small to be meaningful (guard threshold matches validation denom).
            use_true = np.isfinite(true_v) & (np.abs(true_v) > 1e-9)
            abs_from_true = np.abs(true_v) * np.abs(up) / 100.0
            abs_from_est = np.abs(est_v) * np.abs(up) / 100.0
            abs_unc = np.where(use_true, abs_from_true, abs_from_est)

            # Ensure non-finite entries are explicit NaN
            abs_unc = np.where(np.isfinite(abs_unc), abs_unc, np.nan)
            merged[unc_abs_col] = abs_unc
        else:
            merged[unc_abs_col] = np.nan

    center_flux, center_eff, center_eff_col = _compute_density_center_series(
        config=config,
        time_df=time_df,
        flux_col=flux_col,
        eff_pref=eff_pref,
        dictionary_path=dictionary_path,
        dataset_template_path=dataset_template_path,
    )
    if center_eff_col is not None:
        eff_col_time = center_eff_col

    # Primary diagnostic parameter columns.
    est_flux_col = f"est_{flux_col}" if f"est_{flux_col}" in merged.columns else "est_flux_cm2_min"
    est_eff_col = f"est_{eff_col}" if f"est_{eff_col}" in merged.columns else f"est_{eff_pref}"
    if est_eff_col not in merged.columns:
        for c in ("est_eff_sim_1", "est_eff_sim_2", "est_eff_sim_3", "est_eff_sim_4"):
            if c in merged.columns:
                est_eff_col = c
                break

    bias_correction_summary = {
        "enabled": False,
        "removed": True,
        "reason": "posthoc_bias_correction_disabled_use_method_level_estimation",
    }

    true_flux_col = f"true_{flux_col}"
    true_eff_col = f"true_{eff_col}"
    flux_param = est_flux_col.replace("est_", "", 1) if est_flux_col.startswith("est_") else flux_col
    eff_param = est_eff_col.replace("est_", "", 1) if est_eff_col.startswith("est_") else eff_col
    flux_unc_abs_col = f"unc_{flux_param}_abs" if f"unc_{flux_param}_abs" in merged.columns else None
    eff_unc_abs_col = f"unc_{eff_param}_abs" if f"unc_{eff_param}_abs" in merged.columns else None

    # Error columns for primary diagnostics.
    t_flux = pd.to_numeric(merged.get(true_flux_col), errors="coerce")
    e_flux = pd.to_numeric(merged.get(est_flux_col), errors="coerce")
    t_eff = pd.to_numeric(merged.get(true_eff_col), errors="coerce")
    e_eff = pd.to_numeric(merged.get(est_eff_col), errors="coerce")
    merged["error_flux"] = e_flux - t_flux
    merged["error_eff"] = e_eff - t_eff
    merged["relerr_flux_pct"] = (e_flux - t_flux) / t_flux.replace({0.0: np.nan}) * 100.0
    merged["relerr_eff_pct"] = (e_eff - t_eff) / t_eff.replace({0.0: np.nan}) * 100.0
    merged["abs_relerr_flux_pct"] = merged["relerr_flux_pct"].abs()
    merged["abs_relerr_eff_pct"] = merged["relerr_eff_pct"].abs()

    # ── Save outputs ────────────────────────────────────────────────
    out_csv = FILES_DIR / "corrected_by_inference.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote corrected table: %s (%d rows)", out_csv, len(merged))

    summary = {
        "synthetic_dataset_csv": str(synthetic_path),
        "dictionary_csv": str(dictionary_path),
        "uncertainty_lut_csv": str(lut_path),
        "overlay_center_source": "step_3_2_diagnostic_center_csv",
        "step32_diagnostic_center_csv": str(step32_center_path),
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "inverse_mapping": inverse_mapping_cfg,
        "feature_columns_config": feature_columns_cfg,
        "feature_columns_strategy": feature_strategy,
        "feature_columns_resolved_count": int(len(resolved_feature_columns)),
        "feature_columns_resolved": resolved_feature_columns,
        "feature_columns_catalog": str(CONFIG_COLUMNS_PATH),
        "uncertainty_quantile": uncertainty_quantile,
        "n_rows": int(len(merged)),
        "n_successful_flux_estimates": int(pd.to_numeric(merged.get(est_flux_col), errors="coerce").notna().sum()),
        "n_successful_eff_estimates": int(pd.to_numeric(merged.get(est_eff_col), errors="coerce").notna().sum()),
        "flux_true_col": true_flux_col,
        "flux_est_col": est_flux_col,
        "eff_true_col": true_eff_col,
        "eff_est_col": est_eff_col,
        "flux_unc_abs_col": flux_unc_abs_col,
        "eff_unc_abs_col": eff_unc_abs_col,
        "median_abs_relerr_flux_pct": float(pd.to_numeric(merged["abs_relerr_flux_pct"], errors="coerce").median()),
        "median_abs_relerr_eff_pct": float(pd.to_numeric(merged["abs_relerr_eff_pct"], errors="coerce").median()),
        "median_unc_flux_pct": (
            float(pd.to_numeric(merged.get(f"unc_{flux_param}_pct"), errors="coerce").median())
            if f"unc_{flux_param}_pct" in merged.columns else None
        ),
        "median_unc_eff_pct": (
            float(pd.to_numeric(merged.get(f"unc_{eff_param}_pct"), errors="coerce").median())
            if f"unc_{eff_param}_pct" in merged.columns else None
        ),
        "lut_param_names_used": lut_params,
        "density_center_available": bool(center_flux is not None and center_eff is not None),
        "post_estimation_bias_correction": bias_correction_summary,
    }
    out_summary = FILES_DIR / "correction_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary: %s", out_summary)

    # ── Plots ───────────────────────────────────────────────────────
    for stale in (
        "diag_flux_true_vs_est.png",
        "diag_eff_true_vs_est.png",
        "estimated_vs_true_time_series.png",
    ):
        p = PLOTS_DIR / stale
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    _make_plots(
        merged,
        flux_true_col=true_flux_col,
        flux_est_col=est_flux_col,
        flux_unc_abs_col=flux_unc_abs_col,
        eff_true_col=true_eff_col,
        eff_est_col=est_eff_col,
        eff_unc_abs_col=eff_unc_abs_col,
    )
    out_overlay = PLOTS_DIR / "synthetic_time_series_overview_with_estimated.png"
    _plot_step32_style_overlay(
        complete_df=complete_df,
        time_df=time_df,
        merged_df=merged,
        flux_col=flux_col,
        eff_col_time=eff_col_time,
        est_flux_col=est_flux_col,
        est_eff_col=est_eff_col,
        flux_unc_abs_col=flux_unc_abs_col,
        eff_unc_abs_col=eff_unc_abs_col,
        center_flux=center_flux,
        center_eff=center_eff,
        path=out_overlay,
    )

    # Simulated flux + estimated flux + global-rate context.
    global_rate_plot_col = None
    for c in (
        global_rate_col,
        f"true_{global_rate_col}",
        "events_per_second_global_rate",
        "global_rate_hz_mean",
        "true_events_per_second_global_rate",
    ):
        if c in merged.columns:
            global_rate_plot_col = c
            break
    if global_rate_plot_col is None:
        log.warning(
            "Could not produce flux/global-rate plot: no global-rate column found (preferred: %s).",
            global_rate_col,
        )
    else:
        out_flux_rate = PLOTS_DIR / "flux_recovery_vs_global_rate.png"
        _plot_flux_recovery_vs_global_rate(
            df=merged,
            complete_df=complete_df,
            flux_complete_col=flux_col,
            eff_complete_col=eff_col_time,
            flux_true_col=true_flux_col,
            eff_true_col=true_eff_col,
            flux_est_col=est_flux_col,
            flux_unc_abs_col=flux_unc_abs_col,
            global_rate_col=global_rate_plot_col,
            path=out_flux_rate,
        )

    log.info("Wrote plots in: %s", PLOTS_DIR)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
