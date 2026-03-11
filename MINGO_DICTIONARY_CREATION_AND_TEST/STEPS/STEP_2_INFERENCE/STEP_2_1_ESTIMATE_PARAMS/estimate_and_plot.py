#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_1_ESTIMATE_PARAMS/estimate_and_plot.py
Purpose: STEP 2.1 — Solution to the inverse problem.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_2_INFERENCE/STEP_2_1_ESTIMATE_PARAMS/estimate_and_plot.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = STEP_DIR.parent           # STEP_2_INFERENCE
PIPELINE_DIR = INFERENCE_DIR.parent       # .../STEPS
PROJECT_DIR = PIPELINE_DIR.parent         # .../MINGO_DICTIONARY_CREATION_AND_TEST
DEFAULT_CONFIG = PROJECT_DIR / "config_method.json"

DEFAULT_DICTIONARY = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dictionary.csv"
)
DEFAULT_DATASET = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "dataset.csv"
)
DEFAULT_DATASET_ENLARGED = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_3_ENLARGE_DATASET"
    / "OUTPUTS" / "FILES" / "enlarged_dataset.csv"
)
DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS = (
    PIPELINE_DIR / "STEP_1_SETUP" / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS" / "FILES" / "selected_feature_columns.json"
)
CONFIG_COLUMNS_PATH = PROJECT_DIR / "config_columns.json"

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "2_1"


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

# Import the self-contained estimation module
sys.path.insert(0, str(INFERENCE_DIR))
from estimate_parameters import (  # noqa: E402
    DISTANCE_FNS,
    _append_derived_physics_feature_columns,
    _append_derived_tt_global_rate_column,
    _append_tt_global_sum_feature,
    _auto_feature_columns as _shared_auto_feature_columns,
    _derived_feature_columns as _shared_derived_feature_columns,
    _normalize_derived_physics_features,
    _resolve_shared_parameter_exclusion_mode,
    build_shared_parameter_exclusion_mask,
    compute_candidate_distances,
    estimate_from_dataframes,
    resolve_inverse_mapping_cfg,
)
from feature_columns_config import (  # noqa: E402
    parse_explicit_feature_columns,
    resolve_feature_columns_from_catalog,
    sync_feature_column_catalog,
)

logging.basicConfig(
    format="[%(levelname)s] STEP_2.1 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_2.1")

RATE_HISTOGRAM_BIN_RE = re.compile(r"^events_per_second_(?P<bin>\d+)_rate_hz$")


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


STEP21_DENSITY_KEYS = {
    "density_correction_enabled",
    "density_correction_k_neighbors",
}

STEP21_DENSITY_DEFAULTS = {
    "density_correction_enabled": True,
    "density_correction_k_neighbors": 10,
}


def _merge_step21_density_cfg(
    config: dict,
    cfg_21: dict | None = None,
) -> tuple[dict, list[str]]:
    """Resolve STEP 2.1 density-correction config from centralized STEP 1.3 block.

    Priority:
    1. defaults (STEP 2.1 feature-space density behavior),
    2. step_1_3.step_3_2_weighting (legacy alias),
    3. step_1_3.weighting_shared (central source),
    4. step_1_3.weighting_overrides (final explicit overrides).
    """
    cfg_13 = config.get("step_1_3", {}) if isinstance(config, dict) else {}
    merged = dict(STEP21_DENSITY_DEFAULTS)
    applied: set[str] = set()
    warned_legacy_keys: set[str] = set()

    for source_key in ("step_3_2_weighting", "weighting_shared", "weighting_overrides"):
        source = cfg_13.get(source_key, {}) if isinstance(cfg_13, dict) else {}
        if not isinstance(source, dict):
            continue
        for legacy_key in (
            "flux_column",
            "eff_column",
            "density_correction_space",
            "density_correction_exponent",
            "density_correction_clip_min",
            "density_correction_clip_max",
        ):
            if source.get(legacy_key) is not None and legacy_key not in warned_legacy_keys:
                warned_legacy_keys.add(legacy_key)
                log.warning(
                    "Deprecated config key step_1_3.%s.%s detected; ignored. "
                    "STEP 2.1 density correction now uses fixed feature-space behavior.",
                    source_key,
                    legacy_key,
                )
        for key in STEP21_DENSITY_KEYS:
            if key in source and source.get(key) is not None:
                merged[key] = source.get(key)
                applied.add(key)

    # Warn when deprecated and current keys conflict
    legacy_source = cfg_13.get("step_3_2_weighting", {})
    current_source = cfg_13.get("weighting_shared", {})
    if isinstance(legacy_source, dict) and isinstance(current_source, dict):
        for key in STEP21_DENSITY_KEYS:
            lv, cv = legacy_source.get(key), current_source.get(key)
            if lv is not None and cv is not None and lv != cv:
                log.warning(
                    "Config conflict: step_1_3.step_3_2_weighting.%s=%r vs "
                    "step_1_3.weighting_shared.%s=%r; using weighting_shared.",
                    key, lv, key, cv,
                )

    # Final STEP 2.1-local override (highest priority).
    cfg_21_local = cfg_21 if isinstance(cfg_21, dict) else {}
    for key in STEP21_DENSITY_KEYS:
        if key in cfg_21_local and cfg_21_local.get(key) is not None:
            merged[key] = cfg_21_local.get(key)
            applied.add(f"step_2_1.{key}")

    return merged, sorted(applied)


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    candidate_project = PROJECT_DIR / p
    if candidate_project.exists():
        return candidate_project
    candidate_pipeline = PIPELINE_DIR / p
    if candidate_pipeline.exists():
        return candidate_pipeline
    candidate_step = STEP_DIR / p
    if candidate_step.exists():
        return candidate_step
    return candidate_project


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


def _materialize_derived_tt_global_feature_if_needed(
    *,
    cfg_21: dict,
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_mode = str(cfg_21.get("feature_columns", "auto")).strip().lower()
    selected_feature_modes = {
        "step12_selected",
        "step_1_2_selected",
        "selected_from_step12",
        "selected_from_step_1_2",
    }
    if feature_mode not in {"derived", *selected_feature_modes}:
        return dict_df, data_df

    derived_cfg = cfg_21.get("derived_features", {})
    if not isinstance(derived_cfg, dict):
        derived_cfg = {}
    tt_prefix = str(derived_cfg.get("prefix", "last")).strip() or "last"
    trigger_types = parse_explicit_feature_columns(derived_cfg.get("trigger_types", []))
    include_to_tt = _as_bool(derived_cfg.get("include_to_tt_rate_hz", False), False)
    physics_features = _normalize_derived_physics_features(
        derived_cfg.get("physics_features", [])
    )

    dict_out, data_out, derived_rate_col, _ = _append_derived_tt_global_rate_column(
        dict_df=dict_df,
        data_df=data_df,
        prefix_selector=tt_prefix,
        trigger_type_allowlist=trigger_types,
        include_to_tt_rate_hz=include_to_tt,
    )
    if (
        derived_rate_col is None
        and "events_per_second_global_rate" in dict_out.columns
        and "events_per_second_global_rate" in data_out.columns
    ):
        derived_rate_col = "events_per_second_global_rate"
    if derived_rate_col is not None and physics_features:
        dict_out, data_out, _ = _append_derived_physics_feature_columns(
            dict_df=dict_out,
            data_df=data_out,
            rate_column=derived_rate_col,
            physics_features=physics_features,
        )
    return dict_out, data_out


def _select_default_dataset_path(config: dict) -> Path:
    """Choose STEP 2 dataset source from STEP 1.3 enable state."""
    cfg_13 = config.get("step_1_3", {})
    enabled_13 = _as_bool(cfg_13.get("enabled", False), False)
    if not enabled_13:
        return DEFAULT_DATASET

    enlarged_cfg = cfg_13.get("enlarged_dataset_csv", None)
    enlarged_path = _resolve_input_path(enlarged_cfg) if enlarged_cfg else DEFAULT_DATASET_ENLARGED
    if enlarged_path.exists():
        log.info("STEP 1.3 selection: using enlarged dataset for STEP 2 (%s).", enlarged_path)
        return enlarged_path

    log.warning(
        "STEP 1.3 is enabled but enlarged dataset file is missing: %s. Falling back to STEP 1.2 dataset.",
        enlarged_path,
    )
    return DEFAULT_DATASET


def _resolve_step21_feature_columns(
    *,
    feature_cfg: object,
    dict_df: pd.DataFrame,
    data_df: pd.DataFrame,
    include_global_rate: bool,
    global_rate_col: str,
    catalog_path: Path,
    step12_selected_path: Path | None = None,
    derived_feature_cfg: object = None,
) -> tuple[list[str], dict]:
    """Resolve STEP 2.1 feature columns from auto/explicit/catalog config."""
    auto_feature_cols = sorted(
        set(_auto_feature_columns(dict_df, include_global_rate, global_rate_col))
        & set(_auto_feature_columns(data_df, include_global_rate, global_rate_col))
    )

    catalog = sync_feature_column_catalog(
        catalog_path=catalog_path,
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
    derived_include_to_tt = _as_bool(
        derived_cfg_raw.get(
            "include_to_tt_rate_hz",
            derived_cfg_raw.get(
                "include_to_prefix_rates",
                catalog.get("*_to_*_tt_*_rate_hz", False),
            ),
        ),
        bool(catalog.get("*_to_*_tt_*_rate_hz", False)),
    )
    derived_include_trigger_rates = _as_bool(
        # Raw trigger-type absolute rates are opt-in; they can dominate
        # distance-space and re-couple flux with efficiency.
        derived_cfg_raw.get("include_trigger_type_rates", False),
        False,
    )
    derived_include_hist = _as_bool(
        derived_cfg_raw.get(
            "include_rate_histogram",
            rate_hist_cfg.get("enabled", catalog.get("*_*_rate_hz", False)),
        ),
        bool(rate_hist_cfg.get("enabled", catalog.get("*_*_rate_hz", False))),
    )
    derived_physics_features = _normalize_derived_physics_features(
        derived_cfg_raw.get("physics_features", [])
    )
    derived_options = {
        "tt_prefix": derived_tt_prefix,
        "trigger_types": derived_trigger_types,
        "include_to_tt_rate_hz": bool(derived_include_to_tt),
        "include_trigger_type_rates": bool(derived_include_trigger_rates),
        "include_rate_histogram": bool(derived_include_hist),
        "physics_features": derived_physics_features,
    }

    strategy = "explicit"
    selection_info: dict = {
        "catalog_path": str(catalog_path),
        "catalog_selection_mode": str(catalog.get("selection_mode", "enabled")),
    }

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
            selected = [
                c for c in selected_from_step12
                if c in dict_df.columns and c in data_df.columns
            ]
            if selected:
                strategy = "step12_selected"
                selection_info["source"] = "step_1_2.selected_feature_columns"
                selection_info["selected_feature_columns_path"] = str(selected_path)
                selection_info["selected_feature_columns_info"] = selected_info
                selection_info["selected_missing_count"] = int(max(0, len(selected_from_step12) - len(selected)))
                return selected, {"strategy": strategy, **selection_info}
            log.warning(
                "STEP 1.2 selected features unavailable/empty at %s; falling back to derived mode.",
                selected_path,
            )
            mode = "derived"
        if mode == "auto":
            strategy = "auto"
            selected = auto_feature_cols
            selection_info["source"] = "step_2_1.feature_columns=auto"
            return selected, {"strategy": strategy, **selection_info}
        if mode == "derived":
            strategy = "derived"
            (
                dict_derived,
                data_derived,
                derived_rate_col,
                derived_rate_sources,
            ) = _append_derived_tt_global_rate_column(
                dict_df=dict_df,
                data_df=data_df,
                prefix_selector=derived_options["tt_prefix"],
                trigger_type_allowlist=derived_options["trigger_types"],
                include_to_tt_rate_hz=bool(derived_options["include_to_tt_rate_hz"]),
            )
            if derived_rate_col is None and global_rate_col in dict_df.columns and global_rate_col in data_df.columns:
                derived_rate_col = str(global_rate_col)
            if derived_rate_col is None:
                raise ValueError(
                    "No derived feature global-rate source available. "
                    "TT trigger-type sum could not be built and fallback global-rate column is missing."
                )
            (
                dict_derived,
                data_derived,
                derived_physics_cols,
            ) = _append_derived_physics_feature_columns(
                dict_df=dict_derived,
                data_df=data_derived,
                rate_column=derived_rate_col,
                physics_features=derived_options.get("physics_features", []),
            )
            feat_dict = _shared_derived_feature_columns(
                dict_derived,
                rate_column=derived_rate_col,
                trigger_type_rate_columns=(
                    derived_rate_sources
                    if bool(derived_options["include_trigger_type_rates"])
                    else None
                ),
                include_rate_histogram=bool(derived_options["include_rate_histogram"]),
                physics_feature_columns=derived_physics_cols,
            )
            feat_data = _shared_derived_feature_columns(
                data_derived,
                rate_column=derived_rate_col,
                trigger_type_rate_columns=(
                    derived_rate_sources
                    if bool(derived_options["include_trigger_type_rates"])
                    else None
                ),
                include_rate_histogram=bool(derived_options["include_rate_histogram"]),
                physics_feature_columns=derived_physics_cols,
            )
            selected = sorted(set(feat_dict) & set(feat_data))
            if not selected:
                raise ValueError(
                    "No derived feature columns found in dictionary/dataset intersection. "
                    "Expected eff_empirical_1..4 and at least one global-rate feature."
                )
            selection_info["source"] = "step_2_1.feature_columns=derived"
            selection_info["derived_options"] = {
                **derived_options,
                "resolved_global_rate_feature": derived_rate_col,
                "resolved_global_rate_sources": derived_rate_sources,
                "resolved_physics_features": derived_physics_cols,
            }
            return selected, {"strategy": strategy, **selection_info}
        if mode in {"config_columns", "catalog", "config_columns_json"}:
            strategy = "config_columns"
            available = sorted(set(dict_df.columns) & set(data_df.columns))
            selected, catalog_info = resolve_feature_columns_from_catalog(
                catalog=catalog,
                available_columns=available,
            )
            selection_info.update(catalog_info)
            selection_info["source"] = "config_columns.json"
            if catalog_info.get("invalid_include_patterns"):
                log.warning(
                    "Ignoring invalid include pattern(s) in %s: %s",
                    catalog_path,
                    catalog_info.get("invalid_include_patterns"),
                )
            if catalog_info.get("invalid_exclude_patterns"):
                log.warning(
                    "Ignoring invalid exclude pattern(s) in %s: %s",
                    catalog_path,
                    catalog_info.get("invalid_exclude_patterns"),
                )
            if not selected:
                if auto_feature_cols:
                    log.warning(
                        "No features selected by config_columns catalog; falling back to auto (%d columns).",
                        len(auto_feature_cols),
                    )
                    strategy = "auto_fallback_from_config_columns"
                    selected = auto_feature_cols
                    selection_info["source"] = "auto_fallback"
                else:
                    raise ValueError(
                        "No features selected by config_columns catalog and no auto fallback available."
                    )
            return selected, {"strategy": strategy, **selection_info}

    requested = parse_explicit_feature_columns(feature_cfg)
    selected = [
        c for c in requested
        if c in dict_df.columns and c in data_df.columns
    ]
    missing = [c for c in requested if c not in selected]
    if missing:
        log.warning(
            "Ignoring %d explicit feature column(s) missing in dictionary/dataset intersection: %s",
            len(missing),
            missing,
        )
    if not selected:
        raise ValueError(
            "No explicit feature columns found in dictionary/dataset intersection. "
            f"Requested={requested!r}"
        )
    selection_info["source"] = "step_2_1.feature_columns explicit list"
    selection_info["explicit_requested_count"] = int(len(requested))
    selection_info["explicit_missing_count"] = int(len(missing))
    return selected, {"strategy": strategy, **selection_info}


def _resolve_step21_inverse_mapping_cfg(
    cfg_21: dict,
    interpolation_k: int | None,
) -> dict:
    inverse_mapping_cfg = cfg_21.get("inverse_mapping", {})
    if not isinstance(inverse_mapping_cfg, dict):
        inverse_mapping_cfg = {}
    legacy_hist_weight = cfg_21.get("histogram_distance_weight", 1.0)
    legacy_hist_blend = cfg_21.get("histogram_distance_blend_mode", "normalized")
    return resolve_inverse_mapping_cfg(
        inverse_mapping_cfg=inverse_mapping_cfg,
        interpolation_k=interpolation_k,
        histogram_distance_weight=float(legacy_hist_weight),
        histogram_distance_blend_mode=str(legacy_hist_blend),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 2.1: Estimate parameters using dictionary matching."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--dataset-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()
    cfg_21_raw = config.get("step_2_1", {})
    if not isinstance(cfg_21_raw, dict):
        cfg_21_raw = {}
    cfg_21, feature_cfg_sources = _merge_common_feature_space_cfg(config, cfg_21_raw)
    density_cfg, density_cfg_keys = _merge_step21_density_cfg(config, cfg_21)

    dict_path = _resolve_input_path(args.dictionary_csv) if args.dictionary_csv else DEFAULT_DICTIONARY
    data_path = _resolve_input_path(args.dataset_csv) if args.dataset_csv else _select_default_dataset_path(config)
    cfg_13 = config.get("step_1_3", {})
    if args.dataset_csv:
        dataset_mode = "cli_dataset_override"
    elif (
        _as_bool(cfg_13.get("enabled", False), False)
        and data_path.resolve() != DEFAULT_DATASET.resolve()
    ):
        dataset_mode = "step_1_3_enlarged"
    else:
        dataset_mode = "step_1_2_original"

    feature_columns_cfg = cfg_21.get("feature_columns", "auto")
    selected_feature_columns_path_cfg = cfg_21.get("selected_feature_columns_path", None)
    step12_selected_path = (
        _resolve_input_path(selected_feature_columns_path_cfg)
        if selected_feature_columns_path_cfg
        else DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS
    )
    distance_metric = cfg_21.get("distance_metric", "l2_zscore")
    interpolation_k_cfg = cfg_21.get("interpolation_k", 5)
    interpolation_k = None if interpolation_k_cfg is None else int(interpolation_k_cfg)
    inverse_mapping_cfg = _resolve_step21_inverse_mapping_cfg(cfg_21, interpolation_k)
    include_global_rate = _as_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = cfg_21.get("global_rate_col", "events_per_second_global_rate")
    shared_exclusion_mode = _resolve_shared_parameter_exclusion_mode(
        cfg_21.get("shared_parameter_exclusion_mode", None),
        legacy_flag=_as_bool(cfg_21.get("exclude_candidates_sharing_parameter_values", False), False),
    )
    shared_param_cols_cfg = cfg_21.get("shared_parameter_exclusion_columns", "auto")
    shared_param_ignore_cfg = cfg_21.get("shared_parameter_exclusion_ignore", ["cos_n"])
    shared_param_match_atol = float(cfg_21.get("shared_parameter_match_atol", 1e-12))
    # Shared plot-parameter source (global): top-level plot_parameters.
    # Keep step-local keys only as deprecated compatibility fallback.
    plot_params = config.get("plot_parameters", None)
    if plot_params is None:
        legacy_plot_params_21 = cfg_21.get("plot_parameters", None)
        if legacy_plot_params_21 is not None:
            log.warning(
                "Deprecated config key step_2_1.plot_parameters detected; use top-level plot_parameters."
            )
            plot_params = legacy_plot_params_21
    if plot_params is None:
        legacy_plot_params_22 = config.get("step_2_2", {}).get("plot_parameters", None)
        if legacy_plot_params_22 is not None:
            log.warning(
                "Deprecated config key step_2_2.plot_parameters detected; use top-level plot_parameters."
            )
            plot_params = legacy_plot_params_22
    if plot_params is None:
        legacy_plot_params_12 = config.get("step_1_2", {}).get("plot_parameters", None)
        if legacy_plot_params_12 is not None:
            log.warning(
                "Deprecated config key step_1_2.plot_parameters detected; use top-level plot_parameters."
            )
            plot_params = legacy_plot_params_12

    if not dict_path.exists():
        log.error("Dictionary CSV not found: %s", dict_path)
        return 1
    if not data_path.exists():
        log.error("Dataset CSV not found: %s", data_path)
        return 1

    dict_df = pd.read_csv(dict_path, low_memory=False)
    data_df = pd.read_csv(data_path, low_memory=False)
    dict_df, data_df = _materialize_derived_tt_global_feature_if_needed(
        cfg_21=cfg_21,
        dict_df=dict_df,
        data_df=data_df,
    )
    if dict_df.empty:
        log.error("Dictionary table is empty: %s", dict_path)
        return 1
    if data_df.empty:
        log.error("Dataset table is empty: %s", data_path)
        return 1

    try:
        resolved_feature_columns, feature_resolution = _resolve_step21_feature_columns(
            feature_cfg=feature_columns_cfg,
            dict_df=dict_df,
            data_df=data_df,
            include_global_rate=include_global_rate,
            global_rate_col=str(global_rate_col),
            catalog_path=CONFIG_COLUMNS_PATH,
            step12_selected_path=step12_selected_path,
            derived_feature_cfg=cfg_21.get("derived_features", {}),
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("Dictionary: %s", dict_path)
    log.info("Dataset:    %s", data_path)
    log.info("Metric:     %s", distance_metric)
    log.info(
        "K:          %s",
        "all dictionary candidates" if interpolation_k is None else str(interpolation_k),
    )
    log.info(
        "Inverse mapping: selection=%s k=%s weighting=%s aggregation=%s hist_weight=%.3g hist_blend=%s",
        inverse_mapping_cfg.get("neighbor_selection"),
        ("all" if inverse_mapping_cfg.get("neighbor_count") is None else str(inverse_mapping_cfg.get("neighbor_count"))),
        inverse_mapping_cfg.get("weighting"),
        inverse_mapping_cfg.get("aggregation"),
        float(inverse_mapping_cfg.get("histogram_distance_weight", 1.0)),
        inverse_mapping_cfg.get("histogram_distance_blend_mode"),
    )
    log.info(
        "Density cfg: enabled=%s k=%s "
        "[feature-space only; fixed exponent=1 and clip=[0.25,4.0]]",
        density_cfg.get("density_correction_enabled"),
        density_cfg.get("density_correction_k_neighbors"),
    )
    log.info(
        "Shared-parameter candidate exclusion: mode=%s cols=%s ignore=%s atol=%.3g",
        shared_exclusion_mode,
        shared_param_cols_cfg,
        shared_param_ignore_cfg,
        shared_param_match_atol,
    )
    if density_cfg_keys:
        log.info(
            "Density cfg source keys: %s",
            ", ".join(density_cfg_keys),
        )
    log.info(
        "Feature selection: strategy=%s selected=%d (config=%r, catalog=%s)",
        feature_resolution.get("strategy", "unknown"),
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
    if feature_resolution.get("strategy") == "derived":
        log.info(
            "Derived feature options: %s",
            feature_resolution.get("derived_options", {}),
        )
    if feature_resolution.get("strategy") == "step12_selected":
        log.info(
            "STEP 1.2 selected-feature artifact: %s",
            feature_resolution.get("selected_feature_columns_path", step12_selected_path),
        )

    # ── Run estimation ───────────────────────────────────────────────
    derived_options = feature_resolution.get("derived_options", {})
    if not isinstance(derived_options, dict):
        derived_options = {}
    result_df = estimate_from_dataframes(
        dict_df=dict_df,
        data_df=data_df,
        feature_columns=resolved_feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
        exclude_same_file=True,
        shared_parameter_exclusion_mode=shared_exclusion_mode,
        shared_parameter_exclusion_columns=shared_param_cols_cfg,
        shared_parameter_exclusion_ignore=shared_param_ignore_cfg,
        shared_parameter_match_atol=shared_param_match_atol,
        density_weighting_cfg=density_cfg,
        inverse_mapping_cfg=inverse_mapping_cfg,
        derived_tt_prefix=str(derived_options.get("tt_prefix", "last")),
        derived_trigger_types=derived_options.get("trigger_types", []),
        derived_include_to_tt_rate_hz=bool(derived_options.get("include_to_tt_rate_hz", False)),
        derived_include_trigger_type_rates=bool(derived_options.get("include_trigger_type_rates", False)),
        derived_include_rate_histogram=bool(derived_options.get("include_rate_histogram", False)),
        derived_physics_features=derived_options.get("physics_features", []),
    )

    # Attach truth columns needed for validation
    truth_cols = ["flux_cm2_min", "cos_n",
                  "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
                  "n_events", "is_dictionary_entry",
                  "param_hash_x", "param_set_id"]
    for col in truth_cols:
        if col in data_df.columns:
            result_df[f"true_{col}"] = data_df[col].values[:len(result_df)]

    # ── Save ─────────────────────────────────────────────────────────
    out_path = FILES_DIR / "estimated_params.csv"
    result_df.to_csv(out_path, index=False)
    log.info("Wrote estimated params: %s (%d rows)", out_path, len(result_df))

    n_ok = result_df["best_distance"].notna().sum()
    n_fail = result_df["best_distance"].isna().sum()
    distance_group_summary: dict[str, dict[str, float] | None] = {}
    for label, col in (
        ("eff_empirical", "best_distance_base_share_eff_empirical"),
        ("tt_rates", "best_distance_base_share_tt_rates"),
        ("other", "best_distance_base_share_other"),
    ):
        if col not in result_df.columns:
            distance_group_summary[label] = None
            continue
        vals = pd.to_numeric(result_df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            distance_group_summary[label] = None
            continue
        distance_group_summary[label] = {
            "median": float(np.nanmedian(vals)),
            "p90": float(np.nanpercentile(vals, 90)),
        }
    eff_oos_masking_summary = result_df.attrs.get(
        "efficiency_feature_out_of_support_masking"
    )

    summary = {
        "dictionary": str(dict_path),
        "dataset": str(data_path),
        "dataset_source_mode": dataset_mode,
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "inverse_mapping": inverse_mapping_cfg,
        "feature_columns_config": feature_columns_cfg,
        "feature_columns_strategy": feature_resolution.get("strategy", "unknown"),
        "feature_columns_resolved_count": int(len(resolved_feature_columns)),
        "feature_columns_resolved": resolved_feature_columns,
        "feature_columns_catalog": str(CONFIG_COLUMNS_PATH),
        "feature_columns_catalog_resolution": feature_resolution,
        "density_weighting": {
            "config": density_cfg,
            "source_keys": density_cfg_keys,
            "source_keys_from_step_1_3": density_cfg_keys,
        },
        "shared_parameter_candidate_exclusion": {
            "mode": shared_exclusion_mode,
            "columns": shared_param_cols_cfg,
            "ignore_columns": shared_param_ignore_cfg,
            "match_atol": shared_param_match_atol,
        },
        "best_match_non_hist_distance_share": distance_group_summary,
        "total_points": len(result_df),
        "successful_estimates": int(n_ok),
        "failed_estimates": int(n_fail),
    }
    if isinstance(eff_oos_masking_summary, dict):
        summary["efficiency_feature_out_of_support_masking"] = eff_oos_masking_summary
    with open(FILES_DIR / "estimation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostic plots ─────────────────────────────────────────────
    _make_plots(
        result_df=result_df,
        data_df=data_df,
        plot_params=plot_params,
        dict_path=dict_path,
        cfg_21=cfg_21,
        resolved_feature_columns=resolved_feature_columns,
    )

    log.info("Done.")
    return 0


def _auto_feature_columns(
    df: pd.DataFrame,
    include_global_rate: bool = True,
    global_rate_col: str = "events_per_second_global_rate",
) -> list[str]:
    return _shared_auto_feature_columns(
        df=df,
        include_global_rate=include_global_rate,
        global_rate_col=global_rate_col,
    )


def _l2_distances(sample_vec: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    valid = np.isfinite(candidates) & np.isfinite(sample_vec[np.newaxis, :])
    n_valid = valid.sum(axis=1)
    diff = np.where(valid, candidates - sample_vec[np.newaxis, :], 0.0)
    d = np.sqrt(np.sum(diff * diff, axis=1))
    d[n_valid < 2] = np.nan
    return d


def _axis_label_for_param(param_name: str) -> str:
    if param_name == "flux_cm2_min":
        return "Flux [cm⁻² min⁻¹]"
    if param_name.startswith("eff_"):
        return f"Efficiency ({param_name})"
    return param_name


def _axis_label_for_feature(feature_name: str) -> str:
    text = str(feature_name)
    if text.startswith("__derived_tt_global_rate_hz"):
        return "Derived TT global-rate sum [Hz]"
    match = re.match(r"^(?P<prefix>.+?)_tt_(?P<label>[^_]+)_rate_hz$", text)
    if match is not None:
        prefix = str(match.group("prefix")).strip()
        label = str(match.group("label")).strip()
        try:
            label_float = float(label)
            if np.isfinite(label_float) and label_float.is_integer():
                label = str(int(label_float))
        except (TypeError, ValueError):
            pass
        return f"{prefix}:tt_{label} [Hz]"
    if text.endswith("_rate_hz"):
        return text[:-len("_rate_hz")] + " [Hz]"
    return text


def _sanitize_plot_token(token: str) -> str:
    out = []
    for char in str(token):
        if char.isalnum() or char in {"_", "-"}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "param"


def _split_showcase_feature_groups(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split feature columns into non-histogram and histogram-bin groups."""
    hist_cols: list[str] = []
    other_cols: list[str] = []
    for col in feature_cols:
        if RATE_HISTOGRAM_BIN_RE.match(str(col)) is not None:
            hist_cols.append(str(col))
        else:
            other_cols.append(str(col))
    hist_cols.sort(key=lambda c: int(RATE_HISTOGRAM_BIN_RE.match(str(c)).group("bin")))
    return (other_cols, hist_cols)


def _histogram_feature_indices_for_distance(feature_cols: list[str]) -> list[int]:
    indexed: list[tuple[int, int]] = []
    for idx, col in enumerate(feature_cols):
        match = RATE_HISTOGRAM_BIN_RE.match(str(col))
        if match is None:
            continue
        indexed.append((int(match.group("bin")), idx))
    indexed.sort(key=lambda x: x[0])
    return [idx for _, idx in indexed]


def _normalize_histogram_profile(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        return np.asarray([], dtype=float)
    vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)
    finite = np.isfinite(vals)
    out = np.full(vals.shape, np.nan, dtype=float)
    if int(np.count_nonzero(finite)) < 2:
        return out
    total = float(np.nansum(vals[finite]))
    if not np.isfinite(total) or total <= 0.0:
        return out
    out[finite] = vals[finite] / total
    return out


def _cdf_from_histogram_profile(profile: np.ndarray) -> np.ndarray:
    vals = np.asarray(profile, dtype=float)
    if vals.ndim != 1 or vals.size == 0:
        return np.asarray([], dtype=float)
    vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), 0.0)
    total = float(np.sum(vals))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(vals.shape, np.nan, dtype=float)
    vals = vals / total
    return np.cumsum(vals)


def _make_random_showcase_feature_histogram_plot(
    *,
    cand_feat: pd.DataFrame,
    cand_distance: np.ndarray,
    sample_feat_values: pd.Series,
    best_feat_values: pd.Series,
    hist_feature_cols: list[str],
    cfg_21: dict,
    ds_idx: int,
    distance_metric: str,
    showcase_seed_used: int,
) -> None:
    """Compact showcase for correlated histogram-bin features."""
    if not hist_feature_cols:
        return

    hist_cols = [c for c in hist_feature_cols if c in cand_feat.columns]
    if not hist_cols:
        return
    hist_cols.sort(key=lambda c: int(RATE_HISTOGRAM_BIN_RE.match(str(c)).group("bin")))

    bins = np.asarray([int(RATE_HISTOGRAM_BIN_RE.match(str(c)).group("bin")) for c in hist_cols], dtype=float)
    cand_hist = cand_feat[hist_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sample_hist = pd.to_numeric(sample_feat_values.reindex(hist_cols), errors="coerce").to_numpy(dtype=float)
    best_hist = pd.to_numeric(best_feat_values.reindex(hist_cols), errors="coerce").to_numpy(dtype=float)

    sample_norm = _normalize_histogram_profile(sample_hist)
    best_norm = _normalize_histogram_profile(best_hist)

    finite_dist = np.isfinite(cand_distance)
    if not np.any(finite_dist):
        return
    ranked_idx = np.where(finite_dist)[0][np.argsort(cand_distance[finite_dist])]

    top_k_raw = cfg_21.get("showcase_histogram_top_candidates", 50)
    try:
        top_k = max(5, int(top_k_raw))
    except (TypeError, ValueError):
        top_k = 50
    min_candidates_raw = cfg_21.get("showcase_histogram_min_candidates", 5)
    try:
        min_candidates = max(1, int(min_candidates_raw))
    except (TypeError, ValueError):
        min_candidates = 5
    tol_pct_raw = cfg_21.get("showcase_histogram_distance_tolerance_pct", 10.0)
    try:
        tol_pct = max(0.0, float(tol_pct_raw))
    except (TypeError, ValueError):
        tol_pct = 10.0

    best_dist = float(cand_distance[ranked_idx[0]])
    if np.isfinite(best_dist) and best_dist > 0.0:
        tol_limit = best_dist * (1.0 + tol_pct / 100.0)
        tol_mask = cand_distance[ranked_idx] <= tol_limit
    elif np.isfinite(best_dist):
        abs_tol_raw = cfg_21.get("showcase_histogram_distance_tolerance_abs", 1e-12)
        try:
            abs_tol = max(0.0, float(abs_tol_raw))
        except (TypeError, ValueError):
            abs_tol = 1e-12
        tol_limit = best_dist + abs_tol
        tol_mask = cand_distance[ranked_idx] <= tol_limit
    else:
        tol_limit = np.nan
        tol_mask = np.ones(len(ranked_idx), dtype=bool)

    tol_idx = ranked_idx[tol_mask]
    if len(tol_idx) == 0:
        tol_idx = ranked_idx[:1]

    target_cap = min(top_k, len(ranked_idx))
    forced_floor = min(min_candidates, target_cap)
    top_idx = tol_idx[: min(target_cap, len(tol_idx))]
    used_forced_floor = False
    if len(top_idx) < forced_floor:
        top_idx = ranked_idx[:forced_floor]
        used_forced_floor = True

    norm_rows: list[np.ndarray] = []
    used_idx: set[int] = set()
    for idx in top_idx:
        used_idx.add(int(idx))
        norm_curve = _normalize_histogram_profile(cand_hist[idx])
        if int(np.isfinite(norm_curve).sum()) >= 2:
            norm_rows.append(norm_curve)

    # Keep extending in rank order to satisfy the minimum requested curves when possible.
    if len(norm_rows) < forced_floor:
        for idx in ranked_idx:
            idx_int = int(idx)
            if idx_int in used_idx:
                continue
            used_idx.add(idx_int)
            norm_curve = _normalize_histogram_profile(cand_hist[idx_int])
            if int(np.isfinite(norm_curve).sum()) >= 2:
                norm_rows.append(norm_curve)
            if len(norm_rows) >= forced_floor:
                break

    if not norm_rows and int(np.isfinite(sample_norm).sum()) < 2 and int(np.isfinite(best_norm).sum()) < 2:
        return

    top_mat = np.vstack(norm_rows) if norm_rows else np.empty((0, len(hist_cols)), dtype=float)
    if top_mat.size > 0:
        p10 = np.nanpercentile(top_mat, 10.0, axis=0)
        p50 = np.nanpercentile(top_mat, 50.0, axis=0)
        p90 = np.nanpercentile(top_mat, 90.0, axis=0)
    else:
        p10 = np.full(len(hist_cols), np.nan, dtype=float)
        p50 = np.full(len(hist_cols), np.nan, dtype=float)
        p90 = np.full(len(hist_cols), np.nan, dtype=float)

    log.info(
        (
            "Showcase histogram candidates: best_distance=%.6g, tolerance=+%.3g%%, "
            "in_tolerance=%d, selected=%d (cap=%d, min=%d, forced=%s, tol_limit=%s)."
        ),
        best_dist,
        tol_pct,
        len(tol_idx),
        len(norm_rows),
        top_k,
        min_candidates,
        "yes" if used_forced_floor else "no",
        ("nan" if not np.isfinite(tol_limit) else f"{tol_limit:.6g}"),
    )

    sample_cdf = _cdf_from_histogram_profile(sample_norm)
    best_cdf = _cdf_from_histogram_profile(best_norm)
    p50_cdf = _cdf_from_histogram_profile(p50)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    ax_top, ax_cdf = axes

    band_mask = np.isfinite(p10) & np.isfinite(p90)
    if np.any(band_mask):
        ax_top.fill_between(
            bins[band_mask],
            p10[band_mask],
            p90[band_mask],
            color="#A0A0A0",
            alpha=0.35,
            linewidth=0.0,
            label=f"Top-{len(norm_rows)} candidates p10-p90",
        )
    if np.isfinite(p50).any():
        ax_top.plot(bins, p50, color="#555555", linewidth=1.5, alpha=0.95, label="Top candidates median")
    if np.isfinite(sample_norm).any():
        ax_top.plot(bins, sample_norm, color="#E45756", linewidth=2.0, linestyle="--", label="Sample histogram")
    if np.isfinite(best_norm).any():
        ax_top.plot(bins, best_norm, color="#F58518", linewidth=1.9, linestyle="-.", label="Best-match histogram")
    ax_top.set_ylabel("Normalized bin weight")
    ax_top.grid(True, alpha=0.22)
    ax_top.legend(loc="upper right", fontsize=8)

    if np.isfinite(sample_cdf).any():
        ax_cdf.plot(bins, sample_cdf, color="#E45756", linewidth=2.0, linestyle="--", label="Sample CDF")
    if np.isfinite(best_cdf).any():
        ax_cdf.plot(bins, best_cdf, color="#F58518", linewidth=1.9, linestyle="-.", label="Best-match CDF")
    if np.isfinite(p50_cdf).any():
        ax_cdf.plot(bins, p50_cdf, color="#555555", linewidth=1.4, label="Top candidates median CDF")
    ax_cdf.set_xlabel("Rate-histogram bin")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_ylim(-0.01, 1.01)
    ax_cdf.grid(True, alpha=0.22)
    ax_cdf.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Random showcase rate-histogram profile\n"
        f"(dataset_index={ds_idx}, metric={distance_metric}, seed={showcase_seed_used}, "
        f"bins={len(hist_cols)}, tolerance=+{tol_pct:g}%)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    _save_figure(fig, PLOTS_DIR / "random_showcase_feature_rate_histogram.png")
    plt.close(fig)


def _normalize_plot_parameters(plot_params: object) -> list[str]:
    """Normalize plot parameter config into an ordered unique list."""
    if isinstance(plot_params, str):
        requested = [x.strip() for x in plot_params.split(",") if x.strip()]
    elif isinstance(plot_params, (list, tuple, set)):
        requested = [str(x).strip() for x in plot_params if str(x).strip()]
    else:
        requested = []

    normalized: list[str] = []
    seen: set[str] = set()
    for pname in requested:
        if pname in seen:
            continue
        normalized.append(pname)
        seen.add(pname)
    return normalized


def _exclude_candidates_sharing_plot_parameters(
    cand_df: pd.DataFrame,
    sample_row: pd.Series,
    plot_params: object,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Exclude candidate rows that match the sample on ANY configured plot parameter.

    This is only active when plot_parameters are explicitly configured.
    """
    requested = _normalize_plot_parameters(plot_params)
    if not requested:
        # Fallback leakage guard for enlarged synthetic datasets.
        if "param_set_id" in cand_df.columns and "param_set_id" in sample_row.index:
            requested = ["param_set_id"]
    if not requested:
        keep_mask = np.ones(len(cand_df), dtype=bool)
        return (cand_df.copy(), keep_mask, [])

    keep_mask = np.ones(len(cand_df), dtype=bool)
    used_params: list[str] = []
    for pname in requested:
        if pname not in cand_df.columns or pname not in sample_row.index:
            continue

        target = float(pd.to_numeric(pd.Series([sample_row.get(pname)]), errors="coerce").iloc[0])
        if not np.isfinite(target):
            continue

        vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(vals)
        if not np.any(finite_mask):
            continue

        atol = max(1e-12, 1e-12 * max(1.0, abs(target)))
        same_mask = finite_mask & np.isclose(vals, target, rtol=0.0, atol=atol)
        if np.any(same_mask):
            keep_mask &= ~same_mask
            used_params.append(pname)

    filtered = cand_df.loc[keep_mask].copy()
    return (filtered, keep_mask, used_params)


def _select_showcase_param_pairs(
    plot_params: object,
    result_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    requested = _normalize_plot_parameters(plot_params)

    if not requested:
        requested = []
        for col in result_df.columns:
            if not col.startswith("est_"):
                continue
            pname = col[4:]
            if f"true_{pname}" not in result_df.columns:
                continue
            if pname not in cand_df.columns:
                continue
            vals = pd.to_numeric(cand_df[pname], errors="coerce")
            if vals.notna().sum() == 0:
                continue
            requested.append(pname)

    selected_params: list[str] = []
    seen: set[str] = set()
    for pname in requested:
        if pname in seen:
            continue
        if pname not in cand_df.columns:
            log.warning("Showcase parameter '%s' not found in dictionary; skipping.", pname)
            continue
        if f"est_{pname}" not in result_df.columns:
            log.warning("Showcase parameter '%s' has no est_%s column; skipping.", pname, pname)
            continue
        if f"true_{pname}" not in result_df.columns:
            log.warning("Showcase parameter '%s' has no true_%s column; skipping.", pname, pname)
            continue
        vals = pd.to_numeric(cand_df[pname], errors="coerce")
        if vals.notna().sum() == 0:
            log.warning("Showcase parameter '%s' has no finite dictionary values; skipping.", pname)
            continue
        selected_params.append(pname)
        seen.add(pname)

    return list(combinations(selected_params, 2))


def _select_showcase_matrix_parameters(
    cfg_21: dict,
    plot_params: object,
    result_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> list[str]:
    """Resolve ordered parameter list for showcase matrix cells."""
    requested: list[str] = _normalize_plot_parameters(plot_params)

    # Backward compatibility with previous showcase-only knobs.
    if not requested:
        matrix_cfg = cfg_21.get("showcase_matrix_parameters", None)
        if isinstance(matrix_cfg, str):
            requested = [x.strip() for x in matrix_cfg.split(",") if x.strip()]
        elif isinstance(matrix_cfg, (list, tuple, set)):
            requested = [str(x) for x in matrix_cfg]
        if requested:
            log.warning(
                "Deprecated key step_2_1.showcase_matrix_parameters detected; use top-level plot_parameters."
            )

    if not requested:
        for col in result_df.columns:
            if not col.startswith("est_"):
                continue
            pname = col[4:]
            if f"true_{pname}" in result_df.columns:
                requested.append(pname)

    selected: list[str] = []
    seen: set[str] = set()
    for pname in requested:
        if pname in seen:
            continue
        if pname not in cand_df.columns:
            log.warning("Showcase matrix parameter '%s' not found in dictionary candidates; skipping.", pname)
            continue
        if f"est_{pname}" not in result_df.columns:
            log.warning("Showcase matrix parameter '%s' has no est_%s column; skipping.", pname, pname)
            continue
        if f"true_{pname}" not in result_df.columns:
            log.warning("Showcase matrix parameter '%s' has no true_%s column; skipping.", pname, pname)
            continue
        vals = pd.to_numeric(cand_df[pname], errors="coerce")
        if vals.notna().sum() == 0:
            log.warning("Showcase matrix parameter '%s' has no finite candidate values; skipping.", pname)
            continue
        selected.append(pname)
        seen.add(pname)
    return selected


def _resolve_showcase_fixed_tolerance_pct(cfg_21: dict) -> float:
    """Global tolerance (%) used to relax fixed-parameter matching in showcase slices."""
    raw = (cfg_21 or {}).get("showcase_fixed_tolerance_pct", 5.0)
    if raw in (None, "", "null", "None"):
        return 5.0
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "Invalid step_2_1.showcase_fixed_tolerance_pct=%r; using default 5.0%%.",
            raw,
        )
        return 5.0
    if not np.isfinite(pct) or pct < 0.0:
        log.warning(
            "Invalid step_2_1.showcase_fixed_tolerance_pct=%r; using default 5.0%%.",
            raw,
        )
        return 5.0
    return pct


def _tolerance_band_from_unique_values(
    uniq_values: np.ndarray,
    target: float,
    nearest: float,
    tolerance_pct: float,
) -> tuple[float, float]:
    """Return (band, eps) for a percent-based tolerance around target."""
    pct = max(0.0, float(tolerance_pct))
    if pct <= 0.0:
        return (0.0, 0.0)
    span = float(np.max(uniq_values) - np.min(uniq_values)) if uniq_values.size >= 2 else 0.0
    scale = span if span > 0.0 else max(1.0, abs(target), abs(nearest))
    band = (pct / 100.0) * scale
    eps = max(1e-12, 1e-12 * scale)
    return (band, eps)


def _snap_fixed_value(
    values: np.ndarray,
    target: float,
    tolerance_pct: float,
) -> tuple[float | None, float | None, str | None]:
    uniq = np.unique(values[np.isfinite(values)])
    if uniq.size == 0:
        return (None, None, "no_finite_values")
    nearest = float(uniq[int(np.argmin(np.abs(uniq - target)))])
    if uniq.size >= 2:
        diffs = np.diff(np.sort(uniq))
        diffs = diffs[diffs > 0.0]
        min_step = float(diffs.min()) if diffs.size else np.nan
    else:
        min_step = np.nan
    if np.isfinite(min_step):
        atol = max(1e-9, 0.10 * min_step)
    else:
        atol = max(1e-9, 1e-6 * max(1.0, abs(nearest)))

    pct = max(0.0, float(tolerance_pct))
    if pct > 0.0:
        band, eps = _tolerance_band_from_unique_values(
            uniq_values=uniq,
            target=target,
            nearest=nearest,
            tolerance_pct=pct,
        )
        diff = abs(nearest - target)
        if diff > band + eps:
            return (
                None,
                None,
                f"outside_tolerance(diff={diff:.4g}>band={band:.4g},pct={pct:.4g})",
            )

    return nearest, atol, None


def _mask_within_fixed_tolerance(
    values: np.ndarray,
    target: float,
    nearest: float,
    atol: float,
    tolerance_pct: float,
) -> np.ndarray:
    """Mask values compatible with the fixed-parameter tolerance criterion."""
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values, dtype=bool)

    pct = max(0.0, float(tolerance_pct))
    if pct <= 0.0:
        return finite_mask & np.isclose(values, nearest, rtol=0.0, atol=atol)

    uniq = np.unique(values[finite_mask])
    band, eps = _tolerance_band_from_unique_values(
        uniq_values=uniq,
        target=target,
        nearest=nearest,
        tolerance_pct=pct,
    )
    mask = finite_mask & (np.abs(values - target) <= (band + eps))
    if np.any(mask):
        return mask

    # Numerical fallback: always retain the snapped closest value.
    return finite_mask & np.isclose(values, nearest, rtol=0.0, atol=atol)


def _format_fixed_params_note(fixed_params: dict[str, float]) -> str:
    if not fixed_params:
        return "fixed: (none)"
    chunks: list[str] = []
    items = [f"{k}={v:.4g}" for k, v in fixed_params.items()]
    for i in range(0, len(items), 3):
        chunks.append(", ".join(items[i:i + 3]))
    return "fixed: " + "\n       ".join(chunks)


def _is_axis_alias_param(
    cand_df: pd.DataFrame,
    fixed_param: str,
    axis_param: str,
) -> bool:
    if fixed_param not in cand_df.columns or axis_param not in cand_df.columns:
        return False
    a = pd.to_numeric(cand_df[fixed_param], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(cand_df[axis_param], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return False
    delta = np.abs(a[mask] - b[mask])
    scale = max(
        1.0,
        float(np.nanmax(np.abs(a[mask]))),
        float(np.nanmax(np.abs(b[mask]))),
    )
    return bool(np.nanmax(delta) <= 1e-9 * scale)


def _resolve_showcase_seed(
    cfg_21: dict | None,
    *,
    context_label: str,
) -> int:
    """Resolve showcase seed from config (fixed or auto-random)."""
    seed_raw = (cfg_21 or {}).get("showcase_seed", None)
    auto_seed = seed_raw in (None, "", "null", "None", "auto", "random")
    if auto_seed:
        showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        log.info("%s seed (auto): %d", context_label, showcase_seed)
        return showcase_seed
    try:
        showcase_seed = int(seed_raw)
    except (TypeError, ValueError):
        showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        log.warning(
            "Invalid step_2_1.showcase_seed=%r; using auto seed %d for %s.",
            seed_raw,
            showcase_seed,
            context_label.lower(),
        )
        return showcase_seed
    log.info("%s seed (fixed): %d", context_label, showcase_seed)
    return showcase_seed


def _select_showcase_result_row(
    result_df: pd.DataFrame,
    cfg_21: dict | None,
    *,
    context_label: str,
) -> tuple[int, int, int] | None:
    """Pick one showcase row index and its dataset index using resolved seed."""
    required = ["dataset_index", "best_distance"]
    for col in required:
        if col not in result_df.columns:
            return None

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return None

    showcase_seed = _resolve_showcase_seed(cfg_21, context_label=context_label)
    rng = np.random.default_rng(showcase_seed)
    valid_indices = result_df.index[valid_mask].to_numpy()
    chosen_idx = int(rng.choice(valid_indices))
    ds_raw = pd.to_numeric(pd.Series([result_df.loc[chosen_idx, "dataset_index"]]), errors="coerce").iloc[0]
    if not np.isfinite(ds_raw):
        return None
    ds_idx = int(ds_raw)
    log.info(
        "%s selected dataset_index=%d (result row=%d).",
        context_label,
        ds_idx,
        chosen_idx,
    )
    return (chosen_idx, ds_idx, showcase_seed)


def _parse_true_is_dictionary_entry_mask(df: pd.DataFrame) -> pd.Series:
    if "true_is_dictionary_entry" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["true_is_dictionary_entry"].astype(str).str.lower().isin(
        ("true", "1", "yes", "y")
    )


def _rows_with_dictionary_parameter_set(
    val: pd.DataFrame,
    dict_df: pd.DataFrame,
) -> pd.Series:
    mask = pd.Series(False, index=val.index, dtype=bool)

    # Preferred: exact parameter hash copied from source dataset.
    if "true_param_hash_x" in val.columns and "param_hash_x" in dict_df.columns:
        row_keys = val["true_param_hash_x"].astype(str)
        dict_keys = set(dict_df["param_hash_x"].astype(str).dropna().tolist())
        return row_keys.isin(dict_keys).fillna(False)

    # Fallback: tuple over true physical params (and z planes when available).
    base_cols = [
        "flux_cm2_min", "cos_n",
        "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
        "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4",
    ]
    dict_cols = [c for c in base_cols if c in dict_df.columns and f"true_{c}" in val.columns]
    if not dict_cols:
        return mask

    dict_num = dict_df[dict_cols].apply(pd.to_numeric, errors="coerce")
    val_num = val[[f"true_{c}" for c in dict_cols]].apply(pd.to_numeric, errors="coerce")
    dict_keys = {
        tuple(np.round(r, 12))
        for r in dict_num.to_numpy(dtype=float)
        if np.all(np.isfinite(r))
    }
    if not dict_keys:
        return mask

    val_keys = [
        tuple(np.round(r, 12)) if np.all(np.isfinite(r)) else None
        for r in val_num.to_numpy(dtype=float)
    ]
    return pd.Series([k in dict_keys if k is not None else False for k in val_keys], index=val.index)


def _make_random_showcase_l2_contour(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    dict_path: Path,
    cfg_21: dict,
    plot_params: object = None,
    resolved_feature_columns: list[str] | None = None,
) -> None:
    if not dict_path.exists():
        return

    required = ["dataset_index", "best_distance"]
    for col in required:
        if col not in result_df.columns:
            return

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return

    seed_raw = (cfg_21 or {}).get("showcase_seed", None)
    auto_seed = seed_raw in (None, "", "null", "None", "auto", "random")
    if auto_seed:
        showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        log.info("Random showcase seed (auto): %d", showcase_seed)
    else:
        try:
            showcase_seed = int(seed_raw)
        except (TypeError, ValueError):
            showcase_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
            log.warning(
                "Invalid step_2_1.showcase_seed=%r; using auto seed %d.",
                seed_raw,
                showcase_seed,
            )
        else:
            log.info("Random showcase seed (fixed): %d", showcase_seed)
    rng = np.random.default_rng(showcase_seed)
    valid_indices = result_df.index[valid_mask].to_numpy()
    chosen_idx = int(rng.choice(valid_indices))
    row = result_df.loc[chosen_idx]
    ds_idx = int(pd.to_numeric(pd.Series([row["dataset_index"]]), errors="coerce").iloc[0])
    log.info(
        "Random showcase selected dataset_index=%d (result row=%d).",
        ds_idx,
        chosen_idx,
    )
    if ds_idx < 0 or ds_idx >= len(data_df):
        return
    row = result_df.loc[chosen_idx]

    dict_df = pd.read_csv(dict_path, low_memory=False)
    dict_df, data_df = _materialize_derived_tt_global_feature_if_needed(
        cfg_21=cfg_21,
        dict_df=dict_df,
        data_df=data_df,
    )

    include_global_rate = _as_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    selected_path_cfg = cfg_21.get("selected_feature_columns_path", None)
    step12_selected_path = (
        _resolve_input_path(selected_path_cfg)
        if selected_path_cfg
        else DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS
    )
    if resolved_feature_columns is None:
        try:
            feature_cols, _ = _resolve_step21_feature_columns(
                feature_cfg=cfg_21.get("feature_columns", "auto"),
                dict_df=dict_df,
                data_df=data_df,
                include_global_rate=include_global_rate,
                global_rate_col=global_rate_col,
                catalog_path=CONFIG_COLUMNS_PATH,
                step12_selected_path=step12_selected_path,
            )
        except ValueError:
            return
    else:
        feature_cols = [
            str(c)
            for c in resolved_feature_columns
            if str(c) in dict_df.columns and str(c) in data_df.columns
        ]
    if not feature_cols:
        return

    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    sample_feat_df = pd.DataFrame(
        [pd.to_numeric(data_df.loc[ds_idx, feature_cols], errors="coerce")],
        columns=feature_cols,
    )
    dict_feat, sample_feat_df, feature_cols, _, _ = _append_tt_global_sum_feature(
        dict_features=dict_feat,
        data_features=sample_feat_df,
        feature_cols=feature_cols,
        include_global_rate=include_global_rate,
    )
    sample_feat = sample_feat_df.iloc[0].to_numpy(dtype=float)

    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    # short token for labels/filenames (e.g. 'l2' from 'l2_zscore', 'chi2' from 'chi2')
    metric_short = distance_metric.split("_")[0]
    metric_label = "L2" if metric_short == "l2" else metric_short

    if distance_metric == "l2_zscore":
        means = dict_feat.mean(axis=0, skipna=True)
        stds = dict_feat.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_mat = ((dict_feat - means) / stds).to_numpy(dtype=float)
        sample_vec = ((sample_feat - means.to_numpy(dtype=float)) / stds.to_numpy(dtype=float))
    else:
        dict_mat = dict_feat.to_numpy(dtype=float)
        sample_vec = sample_feat

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"] if c in dict_df.columns and c in data_df.columns]
    if z_cols:
        z_tol = float(cfg_21.get("z_tol", 1e-6))
        dict_z = dict_df[z_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sample_z = pd.to_numeric(data_df.loc[ds_idx, z_cols], errors="coerce").to_numpy(dtype=float)
        z_mask = np.all(np.abs(dict_z - sample_z[np.newaxis, :]) <= z_tol, axis=1)
    else:
        z_mask = np.ones(len(dict_df), dtype=bool)

    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    if join_col is not None:
        sample_id = str(data_df.loc[ds_idx, join_col])
        z_mask &= (dict_df[join_col].astype(str).to_numpy() != sample_id)

    cand_df = dict_df.loc[z_mask].copy()
    if cand_df.empty:
        return

    cand_mat = dict_mat[z_mask]
    sample_row = data_df.loc[ds_idx]
    n_before_exclusion = len(cand_df)
    cand_df, keep_mask, exclusion_params = _exclude_candidates_sharing_plot_parameters(
        cand_df=cand_df,
        sample_row=sample_row,
        plot_params=plot_params,
    )
    cand_mat = cand_mat[keep_mask]
    if exclusion_params:
        n_removed = int(n_before_exclusion - len(cand_df))
        log.info(
            "Random showcase: removed %d candidate rows sharing any of %s with dataset_index=%d (remaining=%d).",
            n_removed,
            exclusion_params,
            ds_idx,
            len(cand_df),
        )
    if cand_df.empty:
        log.warning(
            "Random showcase: no candidates remain after plot-parameter exclusion for dataset_index=%d.",
            ds_idx,
        )
        return

    param_pairs = _select_showcase_param_pairs(
        plot_params=plot_params,
        result_df=result_df,
        cand_df=cand_df,
    )
    showcase_pair_cfg = (cfg_21 or {}).get("showcase_param_pair", None)
    if isinstance(showcase_pair_cfg, (list, tuple)) and len(showcase_pair_cfg) == 2:
        pair_x = str(showcase_pair_cfg[0])
        pair_y = str(showcase_pair_cfg[1])
        if (pair_x, pair_y) in param_pairs:
            param_pairs = [(pair_x, pair_y)]
        elif (pair_y, pair_x) in param_pairs:
            param_pairs = [(pair_y, pair_x)]
        else:
            log.warning(
                "Configured showcase_param_pair=(%s, %s) is not available; using auto-selected pairs.",
                pair_x,
                pair_y,
            )
    if not param_pairs:
        log.warning("Random showcase: no valid parameter pairs selected for contour plotting.")
        return

    showcase_max_plots_raw = (cfg_21 or {}).get("showcase_max_plots", 1)
    try:
        showcase_max_plots = max(1, int(showcase_max_plots_raw))
    except (TypeError, ValueError):
        showcase_max_plots = 1
    fixed_tolerance_pct = _resolve_showcase_fixed_tolerance_pct(cfg_21)

    all_estimated_params: list[str] = []
    for col in result_df.columns:
        if not col.startswith("est_"):
            continue
        pname = col[4:]
        if pname not in cand_df.columns:
            continue
        if f"true_{pname}" not in result_df.columns:
            continue
        all_estimated_params.append(pname)
    all_estimated_params = sorted(set(all_estimated_params))

    # Use the same distance function used by the estimator so the plotted
    # quantity matches the reported `best_distance` (e.g. chi2, l2_zscore).
    dist_fn = DISTANCE_FNS.get(distance_metric, DISTANCE_FNS.get(metric_short, None))

    generated = 0
    for x_param, y_param in param_pairs:
        if generated >= showcase_max_plots:
            break
        true_x = float(pd.to_numeric(pd.Series([row.get(f"true_{x_param}")]), errors="coerce").iloc[0])
        true_y = float(pd.to_numeric(pd.Series([row.get(f"true_{y_param}")]), errors="coerce").iloc[0])
        est_x = float(pd.to_numeric(pd.Series([row.get(f"est_{x_param}")]), errors="coerce").iloc[0])
        est_y = float(pd.to_numeric(pd.Series([row.get(f"est_{y_param}")]), errors="coerce").iloc[0])
        if not (np.isfinite(true_x) and np.isfinite(true_y) and np.isfinite(est_x) and np.isfinite(est_y)):
            log.warning(
                "Random showcase pair (%s, %s): missing true/estimated values in chosen row; skipping.",
                x_param,
                y_param,
            )
            continue

        raw_fixed_params = [p for p in all_estimated_params if p not in {x_param, y_param}]
        fixed_params: list[str] = []
        for pname in raw_fixed_params:
            if _is_axis_alias_param(cand_df, pname, x_param) or _is_axis_alias_param(cand_df, pname, y_param):
                log.info(
                    "Random showcase pair (%s, %s): not fixing '%s' because it mirrors a plotted axis.",
                    x_param,
                    y_param,
                    pname,
                )
                continue
            fixed_params.append(pname)
        fixed_mask = np.ones(len(cand_df), dtype=bool)
        fixed_values: dict[str, float] = {}

        failed_fixed = False
        for pname in fixed_params:
            est_col = f"est_{pname}"
            target = float(pd.to_numeric(pd.Series([row.get(est_col)]), errors="coerce").iloc[0])
            if not np.isfinite(target):
                log.warning(
                    "Random showcase pair (%s, %s): fixed parameter %s has no finite estimate; skipping pair.",
                    x_param,
                    y_param,
                    pname,
                )
                failed_fixed = True
                break

            vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size == 0:
                log.warning(
                    "Random showcase pair (%s, %s): fixed parameter %s has no finite dictionary values; skipping pair.",
                    x_param,
                    y_param,
                    pname,
                )
                failed_fixed = True
                break

            snapped, atol, snap_reason = _snap_fixed_value(
                finite_vals,
                target,
                tolerance_pct=fixed_tolerance_pct,
            )
            if snapped is None or atol is None:
                log.warning(
                    "Random showcase pair (%s, %s): fixed parameter %s target %.6g not matched within tolerance %.4g%% (%s); skipping pair.",
                    x_param,
                    y_param,
                    pname,
                    target,
                    fixed_tolerance_pct,
                    snap_reason or "snap_failed",
                )
                failed_fixed = True
                break
            fixed_values[pname] = snapped
            fixed_mask &= np.isfinite(vals) & np.isclose(vals, snapped, rtol=0.0, atol=atol)

            if fixed_mask.sum() == 0:
                log.warning(
                    "Random showcase pair (%s, %s): no candidates remain after fixing %s=%.6g; skipping pair.",
                    x_param,
                    y_param,
                    pname,
                    snapped,
                )
                failed_fixed = True
                break

        pair_df: pd.DataFrame
        pair_mat: np.ndarray
        fallback_reason: str | None = None
        if failed_fixed:
            fallback_reason = "fixed_constraints_rejected"
            pair_df = cand_df.copy()
            pair_mat = cand_mat
            fixed_values = {}
        else:
            pair_df = cand_df.loc[fixed_mask].copy()
            pair_mat = cand_mat[fixed_mask]
            if pair_df.empty:
                fallback_reason = "fixed_constraints_empty_slice"
                pair_df = cand_df.copy()
                pair_mat = cand_mat
                fixed_values = {}

        if fallback_reason is not None:
            log.info(
                "Random showcase pair (%s, %s): fallback to unconstrained slice (%s).",
                x_param,
                y_param,
                fallback_reason,
            )

        def _build_pair_frame(work_df: pd.DataFrame, work_mat: np.ndarray) -> pd.DataFrame:
            tmp = work_df.copy()
            if dist_fn is None:
                z_vals = _l2_distances(sample_vec, work_mat)
                tmp["distance_value"] = z_vals
            else:
                z_list = [dist_fn(sample_vec, work_mat[i]) for i in range(work_mat.shape[0])]
                tmp["distance_value"] = np.array(z_list, dtype=float)
            tmp["x_for_plot"] = pd.to_numeric(tmp[x_param], errors="coerce")
            tmp["y_for_plot"] = pd.to_numeric(tmp[y_param], errors="coerce")
            tmp = tmp.dropna(subset=["x_for_plot", "y_for_plot", "distance_value"])
            if tmp.empty:
                return tmp
            return (
                tmp.groupby(["x_for_plot", "y_for_plot"], as_index=False, sort=True)["distance_value"]
                .min()
            )

        pair_df = _build_pair_frame(pair_df, pair_mat)
        if pair_df.empty:
            log.warning(
                "Random showcase pair (%s, %s): no finite points after numeric cleanup; skipping.",
                x_param,
                y_param,
            )
            continue

        if len(pair_df) < 3:
            if pair_mat.shape[0] < cand_mat.shape[0]:
                log.info(
                    "Random showcase pair (%s, %s): only %d unique points after fixed slicing; "
                    "fallback to unconstrained slice.",
                    x_param,
                    y_param,
                    len(pair_df),
                )
                fixed_values = {}
                pair_df = _build_pair_frame(cand_df, cand_mat)
            if pair_df.empty or len(pair_df) < 3:
                log.warning(
                    "Random showcase pair (%s, %s): only %d unique points; skipping.",
                    x_param,
                    y_param,
                    len(pair_df),
                )
                continue

        x = pair_df["x_for_plot"].to_numpy(dtype=float)
        y = pair_df["y_for_plot"].to_numpy(dtype=float)
        z = pair_df["distance_value"].to_numpy(dtype=float)
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min, z_max = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(8, 5.5))
        contour_ok = False
        try:
            tri = mtri.Triangulation(x, y)
            ctf = ax.tricontourf(
                tri, z, levels=24, cmap="viridis_r", alpha=0.75, vmin=z_min, vmax=z_max
            )
            ax.tricontour(
                tri, z, levels=12, colors="white", linewidths=0.35, alpha=0.30
            )
            contour_ok = True
        except Exception:
            contour_ok = False

        sc = ax.scatter(
            x, y, c=z, cmap="viridis_r", vmin=z_min, vmax=z_max,
            s=36, marker="o", alpha=0.93,
            edgecolors=(1.0, 1.0, 1.0, 0.75), linewidths=0.35, zorder=4
        )
        cb = fig.colorbar(ctf if contour_ok else sc, ax=ax, shrink=0.88)
        cb.set_label(f"{metric_label} distance in feature space")

        ax.scatter(
            [true_x], [true_y], s=170, marker="*", color="#E45756",
            edgecolors="black", linewidths=0.6, zorder=6, label="True point"
        )
        ax.scatter(
            [est_x], [est_y], s=140, marker="X", color="#F58518",
            edgecolors="black", linewidths=0.6, zorder=6, label="Estimated point"
        )

        ax.set_xlabel(_axis_label_for_param(x_param))
        ax.set_ylabel(_axis_label_for_param(y_param))
        ax.set_title(
            "Random showcase "
            f"{metric_label} distance map ({x_param} vs {y_param}, dataset_index={ds_idx}, "
            f"candidates={len(pair_df)})"
        )
        ax.legend(loc="best", fontsize=8)

        note = (
            f"true: {x_param}={true_x:.4g}, {y_param}={true_y:.4g}\n"
            f"est:  {x_param}={est_x:.4g}, {y_param}={est_y:.4g}\n"
            f"best_distance={float(row['best_distance']):.4g}\n"
            f"{_format_fixed_params_note(fixed_values)}"
        )
        ax.text(
            0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85)
        )

        fig.tight_layout()
        safe_x = _sanitize_plot_token(x_param)
        safe_y = _sanitize_plot_token(y_param)
        _save_figure(
            fig,
            PLOTS_DIR / f"random_showcase_distance_contour_{safe_x}__{safe_y}.png",
        )
        plt.close(fig)
        generated += 1

    if generated == 0:
        log.warning("Random showcase: no contour figure generated for any selected parameter pair.")


def _make_random_showcase_distance_matrix(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    dict_path: Path,
    cfg_21: dict,
    plot_params: object = None,
    resolved_feature_columns: list[str] | None = None,
    showcase_result_row_index: int | None = None,
    showcase_dataset_index: int | None = None,
    showcase_seed: int | None = None,
) -> None:
    """Build one n x n showcase matrix with pair maps + diagonal projections."""
    if not dict_path.exists():
        return

    required = ["dataset_index", "best_distance"]
    for col in required:
        if col not in result_df.columns:
            return

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return

    if showcase_result_row_index is None:
        selected = _select_showcase_result_row(
            result_df=result_df,
            cfg_21=cfg_21,
            context_label="Showcase matrix",
        )
        if selected is None:
            return
        chosen_idx, ds_idx, showcase_seed_used = selected
    else:
        chosen_idx = int(showcase_result_row_index)
        if chosen_idx not in result_df.index:
            log.warning(
                "Showcase matrix: requested shared result row %d is not present; skipping.",
                chosen_idx,
            )
            return
        row_ds_raw = pd.to_numeric(
            pd.Series([result_df.loc[chosen_idx, "dataset_index"]]),
            errors="coerce",
        ).iloc[0]
        if showcase_dataset_index is None:
            ds_raw = row_ds_raw
        else:
            ds_raw = pd.to_numeric(pd.Series([showcase_dataset_index]), errors="coerce").iloc[0]
        if not np.isfinite(ds_raw):
            log.warning(
                "Showcase matrix: shared dataset index is non-finite for row %d; skipping.",
                chosen_idx,
            )
            return
        ds_idx = int(ds_raw)
        showcase_seed_used = int(showcase_seed) if showcase_seed is not None else _resolve_showcase_seed(
            cfg_21,
            context_label="Showcase matrix",
        )
        log.info(
            "Showcase matrix using shared dataset_index=%d (result row=%d, seed=%d).",
            ds_idx,
            chosen_idx,
            showcase_seed_used,
        )
    row = result_df.loc[chosen_idx]
    if ds_idx < 0 or ds_idx >= len(data_df):
        return
    row = result_df.loc[chosen_idx]

    dict_df = pd.read_csv(dict_path, low_memory=False)
    dict_df, data_df = _materialize_derived_tt_global_feature_if_needed(
        cfg_21=cfg_21,
        dict_df=dict_df,
        data_df=data_df,
    )

    include_global_rate = _as_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    selected_path_cfg = cfg_21.get("selected_feature_columns_path", None)
    step12_selected_path = (
        _resolve_input_path(selected_path_cfg)
        if selected_path_cfg
        else DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS
    )
    if resolved_feature_columns is None:
        try:
            feature_cols, _ = _resolve_step21_feature_columns(
                feature_cfg=cfg_21.get("feature_columns", "auto"),
                dict_df=dict_df,
                data_df=data_df,
                include_global_rate=include_global_rate,
                global_rate_col=global_rate_col,
                catalog_path=CONFIG_COLUMNS_PATH,
                step12_selected_path=step12_selected_path,
            )
        except ValueError:
            return
    else:
        feature_cols = [
            str(c)
            for c in resolved_feature_columns
            if str(c) in dict_df.columns and str(c) in data_df.columns
        ]
    if not feature_cols:
        return

    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    sample_feat_df = pd.DataFrame(
        [pd.to_numeric(data_df.loc[ds_idx, feature_cols], errors="coerce")],
        columns=feature_cols,
    )
    dict_feat, sample_feat_df, feature_cols, _, _ = _append_tt_global_sum_feature(
        dict_features=dict_feat,
        data_features=sample_feat_df,
        feature_cols=feature_cols,
        include_global_rate=include_global_rate,
    )
    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    metric_short = distance_metric.split("_")[0]
    metric_label = "L2" if metric_short == "l2" else metric_short
    interpolation_k_cfg = cfg_21.get("interpolation_k", 5)
    interpolation_k = None if interpolation_k_cfg in (None, "", "null", "None") else int(interpolation_k_cfg)
    inverse_mapping_cfg = _resolve_step21_inverse_mapping_cfg(cfg_21, interpolation_k)

    dict_raw_mat = dict_feat.to_numpy(dtype=float)
    sample_raw_vec = sample_feat_df.iloc[0].to_numpy(dtype=float)
    if distance_metric == "l2_zscore":
        means = dict_feat.mean(axis=0, skipna=True)
        stds = dict_feat.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_scaled_full = ((dict_feat - means) / stds).to_numpy(dtype=float)
        sample_scaled_full = ((sample_feat_df.iloc[0] - means) / stds).to_numpy(dtype=float)
    else:
        dict_scaled_full = dict_raw_mat
        sample_scaled_full = sample_raw_vec

    hist_feature_idx = _histogram_feature_indices_for_distance(feature_cols)
    hist_feature_set = set(hist_feature_idx)
    non_hist_feature_idx = [idx for idx in range(len(feature_cols)) if idx not in hist_feature_set]

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"] if c in dict_df.columns and c in data_df.columns]
    if z_cols:
        z_tol = float(cfg_21.get("z_tol", 1e-6))
        dict_z = dict_df[z_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sample_z = pd.to_numeric(data_df.loc[ds_idx, z_cols], errors="coerce").to_numpy(dtype=float)
        z_mask = np.all(np.abs(dict_z - sample_z[np.newaxis, :]) <= z_tol, axis=1)
    else:
        z_mask = np.ones(len(dict_df), dtype=bool)

    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    if join_col is not None:
        sample_id = str(data_df.loc[ds_idx, join_col])
        z_mask &= (dict_df[join_col].astype(str).to_numpy() != sample_id)

    cand_indices = np.where(z_mask)[0]
    if len(cand_indices) == 0:
        return

    cand_df = dict_df.iloc[cand_indices].copy()
    sample_row = data_df.loc[ds_idx]
    shared_keep_mask, shared_info = build_shared_parameter_exclusion_mask(
        dict_df=cand_df,
        sample_row=sample_row,
        initial_mask=np.ones(len(cand_df), dtype=bool),
        param_columns=[
            "flux_cm2_min",
            "cos_n",
            "eff_sim_1",
            "eff_sim_2",
            "eff_sim_3",
            "eff_sim_4",
            "eff_empirical_1",
            "eff_empirical_2",
            "eff_empirical_3",
            "eff_empirical_4",
        ],
        shared_parameter_exclusion_mode=cfg_21.get("shared_parameter_exclusion_mode", None),
        shared_parameter_exclusion_columns=cfg_21.get("shared_parameter_exclusion_columns", "auto"),
        shared_parameter_exclusion_ignore=cfg_21.get("shared_parameter_exclusion_ignore", ["cos_n"]),
        shared_parameter_match_atol=float(cfg_21.get("shared_parameter_match_atol", 1e-12)),
    )
    if int(shared_info.get("n_removed", 0)) > 0:
        log.info(
            "Showcase matrix: removed %d candidates by shared-parameter exclusion (mode=%s, remaining=%d).",
            int(shared_info.get("n_removed", 0)),
            shared_info.get("mode", "off"),
            int(shared_info.get("n_after", int(np.sum(shared_keep_mask)))),
        )

    keep_mask = np.asarray(shared_keep_mask, dtype=bool)
    apply_plot_exclusion = _as_bool(
        cfg_21.get("showcase_exclude_shared_plot_parameters", False),
        False,
    )
    if apply_plot_exclusion and np.any(keep_mask):
        cand_after_shared = cand_df.loc[keep_mask].copy()
        cand_after_shared, keep_plot_mask, exclusion_params = _exclude_candidates_sharing_plot_parameters(
            cand_df=cand_after_shared,
            sample_row=sample_row,
            plot_params=plot_params,
        )
        shared_positions = np.where(keep_mask)[0]
        keep_mask_after_plot = np.zeros_like(keep_mask, dtype=bool)
        keep_mask_after_plot[shared_positions[keep_plot_mask]] = True
        keep_mask = keep_mask_after_plot
        if exclusion_params:
            log.info(
                "Showcase matrix: additionally removed %d candidates by plot-parameter exclusion (%s).",
                int(np.sum(shared_keep_mask)) - int(np.sum(keep_mask)),
                exclusion_params,
            )

    cand_indices = cand_indices[keep_mask]
    cand_df = cand_df.loc[keep_mask].copy()
    if cand_df.empty:
        log.warning(
            "Showcase matrix: no candidates remain after exclusion rules for dataset_index=%d.",
            ds_idx,
        )
        return

    matrix_params = _select_showcase_matrix_parameters(
        cfg_21=cfg_21,
        plot_params=plot_params,
        result_df=result_df,
        cand_df=cand_df,
    )
    if len(matrix_params) == 0:
        log.warning("Showcase matrix: no valid parameters selected.")
        return

    max_params_raw = cfg_21.get("showcase_matrix_max_params", None)
    if max_params_raw not in (None, "", "null", "None"):
        try:
            max_params = max(1, int(max_params_raw))
            if len(matrix_params) > max_params:
                matrix_params = matrix_params[:max_params]
        except (TypeError, ValueError):
            log.warning("Invalid step_2_1.showcase_matrix_max_params=%r; ignoring.", max_params_raw)
    fixed_tolerance_pct = _resolve_showcase_fixed_tolerance_pct(cfg_21)

    n_params = len(matrix_params)
    if n_params == 0:
        return

    cand_scaled_non_hist = (
        dict_scaled_full[cand_indices][:, non_hist_feature_idx] if non_hist_feature_idx else None
    )
    sample_scaled_non_hist = (
        sample_scaled_full[non_hist_feature_idx] if non_hist_feature_idx else None
    )
    cand_hist_raw = (
        dict_raw_mat[cand_indices][:, hist_feature_idx] if hist_feature_idx else None
    )
    sample_hist_raw = sample_raw_vec[hist_feature_idx] if hist_feature_idx else None
    cand_distance = compute_candidate_distances(
        distance_metric=distance_metric,
        sample_scaled_non_hist=sample_scaled_non_hist,
        candidates_scaled_non_hist=cand_scaled_non_hist,
        sample_hist_raw=sample_hist_raw,
        candidates_hist_raw=cand_hist_raw,
        histogram_distance_weight=float(inverse_mapping_cfg.get("histogram_distance_weight", 1.0)),
        histogram_distance_blend_mode=str(inverse_mapping_cfg.get("histogram_distance_blend_mode", "normalized")),
    )

    best_candidate_values: dict[str, float] = {}
    anchor_values: dict[str, float] = {}
    finite_dist = np.isfinite(cand_distance)
    row_best_distance = float(pd.to_numeric(pd.Series([row.get("best_distance")]), errors="coerce").iloc[0])
    if np.any(finite_dist) and np.isfinite(row_best_distance):
        recomputed_best = float(np.nanmin(cand_distance[finite_dist]))
        if abs(recomputed_best - row_best_distance) > 1e-8 * max(1.0, abs(row_best_distance)):
            log.info(
                "Showcase matrix: recomputed best_distance differs from result row (recomputed=%.6g, row=%.6g).",
                recomputed_best,
                row_best_distance,
            )
    if np.any(finite_dist):
        best_local_idx = int(np.nanargmin(cand_distance))
        best_row = cand_df.iloc[best_local_idx]
        for pname in matrix_params:
            v = float(pd.to_numeric(pd.Series([best_row.get(pname)]), errors="coerce").iloc[0])
            if np.isfinite(v):
                best_candidate_values[pname] = v

    # Anchor fixed-parameter slices on the inferred solution itself.
    # Fallback to nearest-candidate values only when the estimate is unavailable.
    for pname in matrix_params:
        est_v = float(pd.to_numeric(pd.Series([row.get(f"est_{pname}")]), errors="coerce").iloc[0])
        if np.isfinite(est_v):
            anchor_values[pname] = est_v
            continue
        best_v = best_candidate_values.get(pname, np.nan)
        if np.isfinite(best_v):
            anchor_values[pname] = float(best_v)

    def _limits_with_pad(values: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (0.0, 1.0)
        if np.isclose(lo, hi):
            pad = max(1e-6, 0.02 * max(1.0, abs(lo)))
            return (lo - pad, hi + pad)
        pad = (hi - lo) * pad_frac
        return (lo - pad, hi + pad)

    param_limits: dict[str, tuple[float, float]] = {}
    for pname in matrix_params:
        vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
        true_val = float(pd.to_numeric(pd.Series([row.get(f"true_{pname}")]), errors="coerce").iloc[0])
        est_val = float(pd.to_numeric(pd.Series([row.get(f"est_{pname}")]), errors="coerce").iloc[0])
        ext = vals
        if np.isfinite(true_val):
            ext = np.append(ext, true_val)
        if np.isfinite(est_val):
            ext = np.append(ext, est_val)
        param_limits[pname] = _limits_with_pad(ext)

    def _slice_with_fixed(active_params: set[str]) -> tuple[pd.DataFrame, dict[str, float], str | None]:
        fixed_params = [p for p in matrix_params if p not in active_params]
        active_axis = next(iter(active_params)) if len(active_params) == 1 else None

        trial_tolerances = [float(fixed_tolerance_pct)]
        if fixed_tolerance_pct > 0.0:
            while trial_tolerances[-1] < 100.0:
                next_tol = min(100.0, trial_tolerances[-1] * 2.0)
                if np.isclose(next_tol, trial_tolerances[-1]):
                    break
                trial_tolerances.append(next_tol)

        best_out = pd.DataFrame()
        best_fixed_values: dict[str, float] = {}
        best_unique = -1
        best_tol = trial_tolerances[0]
        last_reason: str | None = None

        for tol_pct in trial_tolerances:
            fixed_mask = np.ones(len(cand_df), dtype=bool)
            fixed_values: dict[str, float] = {}
            failed_reason: str | None = None

            for pname in fixed_params:
                est_col = f"est_{pname}"
                target = anchor_values.get(pname, np.nan)
                if not np.isfinite(target):
                    target = float(pd.to_numeric(pd.Series([row.get(est_col)]), errors="coerce").iloc[0])
                if not np.isfinite(target):
                    failed_reason = f"missing_est_{pname}"
                    break
                vals = pd.to_numeric(cand_df[pname], errors="coerce").to_numpy(dtype=float)
                finite_vals = vals[np.isfinite(vals)]
                if finite_vals.size == 0:
                    failed_reason = f"no_values_{pname}"
                    break
                snapped, atol, snap_reason = _snap_fixed_value(
                    finite_vals,
                    target,
                    tolerance_pct=tol_pct,
                )
                if snapped is None or atol is None:
                    failed_reason = snap_reason or f"no_match_{pname}"
                    break
                fixed_values[pname] = snapped
                fixed_mask &= _mask_within_fixed_tolerance(
                    values=vals,
                    target=target,
                    nearest=snapped,
                    atol=atol,
                    tolerance_pct=tol_pct,
                )
                if fixed_mask.sum() == 0:
                    failed_reason = f"empty_after_fix_{pname}"
                    break

            if failed_reason is not None:
                last_reason = failed_reason
                continue

            out = cand_df.loc[fixed_mask].copy()
            out["distance_value"] = cand_distance[fixed_mask]
            out = out.replace([np.inf, -np.inf], np.nan)
            if out.empty:
                last_reason = "empty_after_all_fixes"
                continue

            unique_count = 1
            if active_axis is not None and active_axis in out.columns:
                unique_count = int(pd.to_numeric(out[active_axis], errors="coerce").dropna().nunique())

            if unique_count > best_unique:
                best_out = out
                best_fixed_values = fixed_values
                best_unique = unique_count
                best_tol = tol_pct

            if active_axis is None or unique_count >= 2:
                if np.isclose(tol_pct, fixed_tolerance_pct):
                    return (out, fixed_values, None)
                return (out, fixed_values, f"expanded_tol_pct={tol_pct:.4g}")

        if best_unique >= 1 and not best_out.empty:
            if np.isclose(best_tol, fixed_tolerance_pct):
                return (best_out, best_fixed_values, "single_point_slice")
            return (best_out, best_fixed_values, f"expanded_tol_pct={best_tol:.4g}")

        return (pd.DataFrame(), {}, last_reason or "empty_slice")

    slice_cache: dict[tuple[str, ...], tuple[pd.DataFrame, dict[str, float], str | None]] = {}

    def _slice_with_fixed_cached(active_params: set[str]) -> tuple[pd.DataFrame, dict[str, float], str | None]:
        key = tuple(sorted(active_params))
        cached = slice_cache.get(key, None)
        if cached is None:
            cached = _slice_with_fixed(active_params)
            slice_cache[key] = cached
        df, fixed_vals, reason = cached
        return (df.copy(), dict(fixed_vals), reason)

    def _prepare_pair_df(x_param: str, y_param: str) -> tuple[pd.DataFrame, str | None]:
        pair_slice_df, _, pair_reason = _slice_with_fixed_cached({x_param, y_param})
        pair_df = (
            pair_slice_df[[x_param, y_param, "distance_value"]].copy()
            if not pair_slice_df.empty
            else pd.DataFrame(columns=[x_param, y_param, "distance_value"])
        )
        pair_df[x_param] = pd.to_numeric(pair_df[x_param], errors="coerce")
        pair_df[y_param] = pd.to_numeric(pair_df[y_param], errors="coerce")
        pair_df["distance_value"] = pd.to_numeric(pair_df["distance_value"], errors="coerce")
        pair_df = pair_df.dropna(subset=[x_param, y_param, "distance_value"])
        if not pair_df.empty:
            pair_df = (
                pair_df.groupby([x_param, y_param], as_index=False, sort=True)["distance_value"]
                .min()
            )
        return (pair_df, pair_reason)

    pair_data_cache: dict[tuple[str, str], tuple[pd.DataFrame, str | None]] = {}
    pair_distance_arrays: list[np.ndarray] = []
    for i, y_param in enumerate(matrix_params):
        for j, x_param in enumerate(matrix_params):
            if j >= i:
                continue
            pair_df, pair_reason = _prepare_pair_df(x_param, y_param)
            pair_data_cache[(x_param, y_param)] = (pair_df, pair_reason)
            if not pair_df.empty:
                z_vals = pair_df["distance_value"].to_numpy(dtype=float)
                z_vals = z_vals[np.isfinite(z_vals)]
                if z_vals.size > 0:
                    pair_distance_arrays.append(z_vals)

    if pair_distance_arrays:
        dist_limits = _limits_with_pad(np.concatenate(pair_distance_arrays))
    else:
        dist_limits = _limits_with_pad(cand_distance)

    n_color_levels = 14
    dist_levels = np.linspace(dist_limits[0], dist_limits[1], n_color_levels + 1)
    base_colors = plt.get_cmap("viridis_r")(np.linspace(0.06, 0.94, n_color_levels))
    pastel_mix = 0.32
    pastel_rgb = (1.0 - pastel_mix) * base_colors[:, :3] + pastel_mix * 1.0
    pastel_rgba = np.column_stack([pastel_rgb, np.full(n_color_levels, 0.96)])
    dist_cmap = mcolors.ListedColormap(pastel_rgba, name="viridis_r_pastel")
    dist_norm = mcolors.BoundaryNorm(dist_levels, dist_cmap.N, clip=True)

    fig_w = max(5.5, 3.2 * n_params)
    fig_h = max(5.0, 3.0 * n_params)
    fig, axes = plt.subplots(n_params, n_params, figsize=(fig_w, fig_h), squeeze=False)
    diag_share_y = _as_bool(cfg_21.get("showcase_matrix_diag_share_y", True), True)

    plotted_lower_any = False
    for i, y_param in enumerate(matrix_params):
        for j, x_param in enumerate(matrix_params):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            if i == j:
                diag_df, fixed_vals, reason = _slice_with_fixed_cached({x_param})
                diag_df[x_param] = pd.to_numeric(diag_df.get(x_param), errors="coerce")
                diag_df["distance_value"] = pd.to_numeric(diag_df.get("distance_value"), errors="coerce")
                diag_df = diag_df.dropna(subset=[x_param, "distance_value"])
                if not diag_df.empty:
                    curve = (
                        diag_df.groupby(x_param, as_index=False, sort=True)["distance_value"]
                        .min()
                    )
                else:
                    curve = pd.DataFrame(columns=[x_param, "distance_value"])

                if len(curve) >= 1:
                    xv = curve[x_param].to_numpy(dtype=float)
                    dv = curve["distance_value"].to_numpy(dtype=float)
                    if len(curve) >= 2:
                        ax.plot(xv, dv, color="#4C78A8", linewidth=1.5)
                    ax.scatter(xv, dv, s=14, color="#4C78A8", alpha=0.9)
                    true_x = float(pd.to_numeric(pd.Series([row.get(f"true_{x_param}")]), errors="coerce").iloc[0])
                    est_x = float(pd.to_numeric(pd.Series([row.get(f"est_{x_param}")]), errors="coerce").iloc[0])
                    if np.isfinite(true_x):
                        ax.axvline(true_x, color="#E45756", linestyle="--", linewidth=1.0)
                    if np.isfinite(est_x):
                        ax.axvline(est_x, color="#F58518", linestyle="-.", linewidth=1.0)
                    if reason in {"single_point_slice"} or (isinstance(reason, str) and reason.startswith("expanded_tol_pct=")):
                        ax.text(
                            0.03,
                            0.97,
                            reason,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=6.3,
                            color="#555555",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", alpha=0.8),
                        )
                else:
                    message = "N/A"
                    if reason is not None:
                        message = f"N/A\n({reason})"
                    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=7, transform=ax.transAxes)

                ax.set_xlim(*param_limits[x_param])
                if diag_share_y:
                    ax.set_ylim(*dist_limits)
                else:
                    diag_y = (
                        pd.to_numeric(curve.get("distance_value"), errors="coerce").to_numpy(dtype=float)
                        if not curve.empty else np.asarray([], dtype=float)
                    )
                    best_distance_row = float(
                        pd.to_numeric(pd.Series([row.get("best_distance")]), errors="coerce").iloc[0]
                    )
                    if np.isfinite(best_distance_row):
                        diag_y = np.append(diag_y, best_distance_row)
                    ax.set_ylim(*_limits_with_pad(diag_y))
                ax.set_title(f"{x_param}", fontsize=8)
            else:
                # Lower-triangle cells vary this pair while fixing all other
                # parameters with the same tolerance policy used on diagonals.
                pair_df, pair_reason = pair_data_cache.get(
                    (x_param, y_param),
                    (pd.DataFrame(columns=[x_param, y_param, "distance_value"]), "empty_slice"),
                )

                plotted = False
                if len(pair_df) >= 3:
                    x = pair_df[x_param].to_numpy(dtype=float)
                    y = pair_df[y_param].to_numpy(dtype=float)
                    z = pair_df["distance_value"].to_numpy(dtype=float)
                    try:
                        tri = mtri.Triangulation(x, y)
                        ax.tricontourf(
                            tri,
                            z,
                            levels=dist_levels,
                            cmap=dist_cmap,
                            norm=dist_norm,
                            alpha=0.96,
                        )
                        ax.tricontour(
                            tri,
                            z,
                            levels=dist_levels[1:-1:2],
                            colors="white",
                            linewidths=0.22,
                            alpha=0.20,
                        )
                        plotted = True
                    except Exception:
                        plotted = False
                elif len(pair_df) > 0:
                    x = pair_df[x_param].to_numpy(dtype=float)
                    y = pair_df[y_param].to_numpy(dtype=float)
                    z = pair_df["distance_value"].to_numpy(dtype=float)
                    ax.scatter(
                        x,
                        y,
                        c=z,
                        s=18,
                        cmap=dist_cmap,
                        norm=dist_norm,
                        alpha=0.95,
                        linewidths=0.0,
                        zorder=2,
                    )
                    plotted = True

                if plotted:
                    plotted_lower_any = True
                    true_x = float(pd.to_numeric(pd.Series([row.get(f"true_{x_param}")]), errors="coerce").iloc[0])
                    true_y = float(pd.to_numeric(pd.Series([row.get(f"true_{y_param}")]), errors="coerce").iloc[0])
                    est_x = float(pd.to_numeric(pd.Series([row.get(f"est_{x_param}")]), errors="coerce").iloc[0])
                    est_y = float(pd.to_numeric(pd.Series([row.get(f"est_{y_param}")]), errors="coerce").iloc[0])
                    if np.isfinite(true_x) and np.isfinite(true_y):
                        ax.scatter([true_x], [true_y], s=46, marker="*", color="#E45756", edgecolors="black", linewidths=0.45, zorder=4)
                    if np.isfinite(est_x) and np.isfinite(est_y):
                        ax.scatter([est_x], [est_y], s=40, marker="X", color="#F58518", edgecolors="black", linewidths=0.45, zorder=4)
                    if isinstance(pair_reason, str) and pair_reason.startswith("expanded_tol_pct="):
                        ax.text(
                            0.03,
                            0.97,
                            pair_reason,
                            transform=ax.transAxes,
                            va="top",
                            ha="left",
                            fontsize=6.3,
                            color="#555555",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.85", alpha=0.8),
                        )
                else:
                    message = "N/A"
                    if pair_reason is not None:
                        message = f"N/A\n({pair_reason})"
                    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=7, transform=ax.transAxes)

                ax.set_xlim(*param_limits[x_param])
                ax.set_ylim(*param_limits[y_param])

            if i < n_params - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(_axis_label_for_param(x_param))
            if j > 0:
                ax.set_yticklabels([])
            else:
                if i == j:
                    ax.set_ylabel(f"{metric_label} distance")
                else:
                    ax.set_ylabel(_axis_label_for_param(y_param))

    fig.suptitle(
        "Random showcase distance matrix\n"
        f"(dataset_index={ds_idx}, metric={distance_metric}, seed={showcase_seed_used}, "
        f"fixed_anchor=estimated, fixed_tol={fixed_tolerance_pct:.4g}%)",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.88 if plotted_lower_any else 0.97,
        bottom=0.06,
        top=0.93,
        wspace=0.08,
        hspace=0.08,
    )
    if plotted_lower_any:
        sm = plt.cm.ScalarMappable(norm=dist_norm, cmap=dist_cmap)
        sm.set_array([])
        cax = fig.add_axes([0.905, 0.10, 0.018, 0.78])
        tick_step = max(1, int(np.ceil(n_color_levels / 6)))
        tick_values = dist_levels[::tick_step]
        if not np.isclose(tick_values[-1], dist_levels[-1]):
            tick_values = np.append(tick_values, dist_levels[-1])
        cbar = fig.colorbar(
            sm,
            cax=cax,
            boundaries=dist_levels,
            ticks=tick_values,
            spacing="proportional",
        )
        cbar.set_label(f"{metric_label} distance")

    _save_figure(fig, PLOTS_DIR / "random_showcase_distance_matrix.png")
    plt.close(fig)


def _make_random_showcase_feature_space_matrix(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    dict_path: Path,
    cfg_21: dict,
    plot_params: object = None,
    resolved_feature_columns: list[str] | None = None,
    showcase_result_row_index: int | None = None,
    showcase_dataset_index: int | None = None,
    showcase_seed: int | None = None,
) -> None:
    """Showcase pairwise feature-space projections for one random dataset point."""
    if not dict_path.exists():
        return

    required = ["dataset_index", "best_distance"]
    for col in required:
        if col not in result_df.columns:
            return

    valid_mask = (
        pd.to_numeric(result_df["dataset_index"], errors="coerce").notna()
        & result_df["best_distance"].notna()
    )
    if valid_mask.sum() == 0:
        return

    if showcase_result_row_index is None:
        selected = _select_showcase_result_row(
            result_df=result_df,
            cfg_21=cfg_21,
            context_label="Showcase feature matrix",
        )
        if selected is None:
            return
        chosen_idx, ds_idx, showcase_seed_used = selected
    else:
        chosen_idx = int(showcase_result_row_index)
        if chosen_idx not in result_df.index:
            log.warning(
                "Showcase feature matrix: requested shared result row %d is not present; skipping.",
                chosen_idx,
            )
            return
        row_ds_raw = pd.to_numeric(
            pd.Series([result_df.loc[chosen_idx, "dataset_index"]]),
            errors="coerce",
        ).iloc[0]
        if showcase_dataset_index is None:
            ds_raw = row_ds_raw
        else:
            ds_raw = pd.to_numeric(pd.Series([showcase_dataset_index]), errors="coerce").iloc[0]
        if not np.isfinite(ds_raw):
            log.warning(
                "Showcase feature matrix: shared dataset index is non-finite for row %d; skipping.",
                chosen_idx,
            )
            return
        ds_idx = int(ds_raw)
        showcase_seed_used = int(showcase_seed) if showcase_seed is not None else _resolve_showcase_seed(
            cfg_21,
            context_label="Showcase feature matrix",
        )
        log.info(
            "Showcase feature matrix using shared dataset_index=%d (result row=%d, seed=%d).",
            ds_idx,
            chosen_idx,
            showcase_seed_used,
        )
    if ds_idx < 0 or ds_idx >= len(data_df):
        return
    row = result_df.loc[chosen_idx]

    dict_df = pd.read_csv(dict_path, low_memory=False)
    dict_df, data_df = _materialize_derived_tt_global_feature_if_needed(
        cfg_21=cfg_21,
        dict_df=dict_df,
        data_df=data_df,
    )
    include_global_rate = _as_bool(cfg_21.get("include_global_rate", True), True)
    global_rate_col = str(cfg_21.get("global_rate_col", "events_per_second_global_rate"))
    selected_path_cfg = cfg_21.get("selected_feature_columns_path", None)
    step12_selected_path = (
        _resolve_input_path(selected_path_cfg)
        if selected_path_cfg
        else DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS
    )

    if resolved_feature_columns is None:
        try:
            feature_cols, _ = _resolve_step21_feature_columns(
                feature_cfg=cfg_21.get("feature_columns", "auto"),
                dict_df=dict_df,
                data_df=data_df,
                include_global_rate=include_global_rate,
                global_rate_col=global_rate_col,
                catalog_path=CONFIG_COLUMNS_PATH,
                step12_selected_path=step12_selected_path,
            )
        except ValueError:
            return
    else:
        feature_cols = [
            str(c)
            for c in resolved_feature_columns
            if str(c) in dict_df.columns and str(c) in data_df.columns
        ]
    if not feature_cols:
        return

    dict_feat = dict_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    sample_feat_df = pd.DataFrame(
        [pd.to_numeric(data_df.loc[ds_idx, feature_cols], errors="coerce")],
        columns=feature_cols,
    )
    dict_feat, sample_feat_df, feature_cols, _, _ = _append_tt_global_sum_feature(
        dict_features=dict_feat,
        data_features=sample_feat_df,
        feature_cols=feature_cols,
        include_global_rate=include_global_rate,
    )
    if not feature_cols:
        return

    distance_metric = str(cfg_21.get("distance_metric", "l2_zscore"))
    metric_short = distance_metric.split("_")[0]
    metric_label = "L2" if metric_short == "l2" else metric_short
    interpolation_k_cfg = cfg_21.get("interpolation_k", 5)
    interpolation_k = None if interpolation_k_cfg in (None, "", "null", "None") else int(interpolation_k_cfg)
    inverse_mapping_cfg = _resolve_step21_inverse_mapping_cfg(cfg_21, interpolation_k)

    dict_raw_mat = dict_feat.to_numpy(dtype=float)
    sample_raw_vec = sample_feat_df.iloc[0].to_numpy(dtype=float)
    if distance_metric == "l2_zscore":
        means = dict_feat.mean(axis=0, skipna=True)
        stds = dict_feat.std(axis=0, skipna=True).replace({0.0: np.nan})
        dict_scaled_full = ((dict_feat - means) / stds).to_numpy(dtype=float)
        sample_scaled_full = ((sample_feat_df.iloc[0] - means) / stds).to_numpy(dtype=float)
    else:
        dict_scaled_full = dict_raw_mat
        sample_scaled_full = sample_raw_vec

    hist_feature_idx = _histogram_feature_indices_for_distance(feature_cols)
    hist_feature_set = set(hist_feature_idx)
    non_hist_feature_idx = [idx for idx in range(len(feature_cols)) if idx not in hist_feature_set]

    z_cols = [c for c in ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"] if c in dict_df.columns and c in data_df.columns]
    if z_cols:
        z_tol = float(cfg_21.get("z_tol", 1e-6))
        dict_z = dict_df[z_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sample_z = pd.to_numeric(data_df.loc[ds_idx, z_cols], errors="coerce").to_numpy(dtype=float)
        z_mask = np.all(np.abs(dict_z - sample_z[np.newaxis, :]) <= z_tol, axis=1)
    else:
        z_mask = np.ones(len(dict_df), dtype=bool)

    join_col = None
    for candidate in ("filename_base", "file_name"):
        if candidate in dict_df.columns and candidate in data_df.columns:
            join_col = candidate
            break
    if join_col is not None:
        sample_id = str(data_df.loc[ds_idx, join_col])
        z_mask &= (dict_df[join_col].astype(str).to_numpy() != sample_id)

    cand_indices = np.where(z_mask)[0]
    if len(cand_indices) == 0:
        log.warning(
            "Showcase feature matrix: no candidates after z/self exclusion for dataset_index=%d.",
            ds_idx,
        )
        return

    cand_df = dict_df.iloc[cand_indices].copy()
    sample_row = data_df.loc[ds_idx]
    shared_keep_mask, shared_info = build_shared_parameter_exclusion_mask(
        dict_df=cand_df,
        sample_row=sample_row,
        initial_mask=np.ones(len(cand_df), dtype=bool),
        param_columns=[
            "flux_cm2_min",
            "cos_n",
            "eff_sim_1",
            "eff_sim_2",
            "eff_sim_3",
            "eff_sim_4",
            "eff_empirical_1",
            "eff_empirical_2",
            "eff_empirical_3",
            "eff_empirical_4",
        ],
        shared_parameter_exclusion_mode=cfg_21.get("shared_parameter_exclusion_mode", None),
        shared_parameter_exclusion_columns=cfg_21.get("shared_parameter_exclusion_columns", "auto"),
        shared_parameter_exclusion_ignore=cfg_21.get("shared_parameter_exclusion_ignore", ["cos_n"]),
        shared_parameter_match_atol=float(cfg_21.get("shared_parameter_match_atol", 1e-12)),
    )
    if int(shared_info.get("n_removed", 0)) > 0:
        log.info(
            "Showcase feature matrix: removed %d candidates by shared-parameter exclusion (mode=%s, remaining=%d).",
            int(shared_info.get("n_removed", 0)),
            shared_info.get("mode", "off"),
            int(shared_info.get("n_after", int(np.sum(shared_keep_mask)))),
        )

    keep_mask = np.asarray(shared_keep_mask, dtype=bool)
    apply_plot_exclusion = _as_bool(
        cfg_21.get("showcase_exclude_shared_plot_parameters", False),
        False,
    )
    if apply_plot_exclusion and np.any(keep_mask):
        cand_after_shared = cand_df.loc[keep_mask].copy()
        cand_after_shared, keep_plot_mask, exclusion_params = _exclude_candidates_sharing_plot_parameters(
            cand_df=cand_after_shared,
            sample_row=sample_row,
            plot_params=plot_params,
        )
        shared_positions = np.where(keep_mask)[0]
        keep_mask_after_plot = np.zeros_like(keep_mask, dtype=bool)
        keep_mask_after_plot[shared_positions[keep_plot_mask]] = True
        keep_mask = keep_mask_after_plot
        if exclusion_params:
            log.info(
                "Showcase feature matrix: additionally removed %d candidates by plot-parameter exclusion (%s).",
                int(np.sum(shared_keep_mask)) - int(np.sum(keep_mask)),
                exclusion_params,
            )

    cand_indices = cand_indices[keep_mask]
    cand_df = cand_df.loc[keep_mask].copy()
    if len(cand_indices) == 0:
        log.warning(
            "Showcase feature matrix: no candidates remain after exclusion rules for dataset_index=%d.",
            ds_idx,
        )
        return

    cand_feat = dict_feat.iloc[cand_indices].copy()

    cand_scaled_non_hist = (
        dict_scaled_full[cand_indices][:, non_hist_feature_idx] if non_hist_feature_idx else None
    )
    sample_scaled_non_hist = (
        sample_scaled_full[non_hist_feature_idx] if non_hist_feature_idx else None
    )
    cand_hist_raw = (
        dict_raw_mat[cand_indices][:, hist_feature_idx] if hist_feature_idx else None
    )
    sample_hist_raw = sample_raw_vec[hist_feature_idx] if hist_feature_idx else None
    cand_distance = compute_candidate_distances(
        distance_metric=distance_metric,
        sample_scaled_non_hist=sample_scaled_non_hist,
        candidates_scaled_non_hist=cand_scaled_non_hist,
        sample_hist_raw=sample_hist_raw,
        candidates_hist_raw=cand_hist_raw,
        histogram_distance_weight=float(inverse_mapping_cfg.get("histogram_distance_weight", 1.0)),
        histogram_distance_blend_mode=str(inverse_mapping_cfg.get("histogram_distance_blend_mode", "normalized")),
    )

    finite_dist = np.isfinite(cand_distance)
    row_best_distance = float(pd.to_numeric(pd.Series([row.get("best_distance")]), errors="coerce").iloc[0])
    if np.any(finite_dist) and np.isfinite(row_best_distance):
        recomputed_best = float(np.nanmin(cand_distance[finite_dist]))
        if abs(recomputed_best - row_best_distance) > 1e-8 * max(1.0, abs(row_best_distance)):
            log.info(
                "Showcase feature matrix: recomputed best_distance differs from result row (recomputed=%.6g, row=%.6g).",
                recomputed_best,
                row_best_distance,
            )
    if not np.any(finite_dist):
        log.warning(
            "Showcase feature matrix: all distances are non-finite for dataset_index=%d.",
            ds_idx,
        )
        return

    best_local_idx = int(np.nanargmin(cand_distance))
    best_feat_values = cand_feat.iloc[best_local_idx]
    sample_feat_values = sample_feat_df.iloc[0]

    feature_plot_cols = [c for c in feature_cols if c in cand_feat.columns]
    matrix_feature_cols, hist_feature_cols = _split_showcase_feature_groups(feature_plot_cols)
    log.info(
        "Showcase feature matrix split: %d non-hist features, %d histogram-bin features.",
        len(matrix_feature_cols),
        len(hist_feature_cols),
    )
    _make_random_showcase_feature_histogram_plot(
        cand_feat=cand_feat,
        cand_distance=cand_distance,
        sample_feat_values=sample_feat_values,
        best_feat_values=best_feat_values,
        hist_feature_cols=hist_feature_cols,
        cfg_21=cfg_21,
        ds_idx=ds_idx,
        distance_metric=distance_metric,
        showcase_seed_used=showcase_seed_used,
    )
    max_features_raw = cfg_21.get("showcase_feature_matrix_max_features", None)
    if max_features_raw not in (None, "", "null", "None"):
        try:
            max_features = max(1, int(max_features_raw))
        except (TypeError, ValueError):
            max_features = None
            log.warning(
                "Invalid step_2_1.showcase_feature_matrix_max_features=%r; using all feature dimensions.",
                max_features_raw,
            )
        if max_features is not None and len(matrix_feature_cols) > max_features:
            matrix_feature_cols = matrix_feature_cols[:max_features]

    n_features = len(matrix_feature_cols)
    if n_features == 0:
        log.info(
            "Showcase feature matrix skipped: no non-histogram features available for dataset_index=%d.",
            ds_idx,
        )
        return

    def _limits_with_pad(values: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (0.0, 1.0)
        if np.isclose(lo, hi):
            pad = max(1e-6, 0.02 * max(1.0, abs(lo)))
            return (lo - pad, hi + pad)
        pad = (hi - lo) * pad_frac
        return (lo - pad, hi + pad)

    feature_limits: dict[str, tuple[float, float]] = {}
    for feature_name in matrix_feature_cols:
        vals = pd.to_numeric(cand_feat[feature_name], errors="coerce").to_numpy(dtype=float)
        sample_v = float(pd.to_numeric(pd.Series([sample_feat_values.get(feature_name)]), errors="coerce").iloc[0])
        best_v = float(pd.to_numeric(pd.Series([best_feat_values.get(feature_name)]), errors="coerce").iloc[0])
        ext = vals
        if np.isfinite(sample_v):
            ext = np.append(ext, sample_v)
        if np.isfinite(best_v):
            ext = np.append(ext, best_v)
        feature_limits[feature_name] = _limits_with_pad(ext)

    dist_limits = _limits_with_pad(cand_distance[finite_dist], pad_frac=0.02)
    n_color_levels = 14
    dist_levels = np.linspace(dist_limits[0], dist_limits[1], n_color_levels + 1)
    base_colors = plt.get_cmap("viridis_r")(np.linspace(0.06, 0.94, n_color_levels))
    pastel_mix = 0.32
    pastel_rgb = (1.0 - pastel_mix) * base_colors[:, :3] + pastel_mix * 1.0
    pastel_rgba = np.column_stack([pastel_rgb, np.full(n_color_levels, 0.96)])
    dist_cmap = mcolors.ListedColormap(pastel_rgba, name="viridis_r_pastel_feature")
    dist_norm = mcolors.BoundaryNorm(dist_levels, dist_cmap.N, clip=True)

    fig_w = max(5.5, 3.0 * n_features)
    fig_h = max(5.0, 2.8 * n_features)
    fig, axes = plt.subplots(n_features, n_features, figsize=(fig_w, fig_h), squeeze=False)

    plotted_lower_any = False
    for i, y_feature in enumerate(matrix_feature_cols):
        for j, x_feature in enumerate(matrix_feature_cols):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            if i == j:
                x_vals = pd.to_numeric(cand_feat[x_feature], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(x_vals)
                if np.any(finite):
                    ax.hist(x_vals[finite], bins=24, color="#4C78A8", alpha=0.78, edgecolor="white")
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7, transform=ax.transAxes)

                sample_x = float(pd.to_numeric(pd.Series([sample_feat_values.get(x_feature)]), errors="coerce").iloc[0])
                best_x = float(pd.to_numeric(pd.Series([best_feat_values.get(x_feature)]), errors="coerce").iloc[0])
                if np.isfinite(sample_x):
                    ax.axvline(sample_x, color="#E45756", linestyle="--", linewidth=1.0)
                if np.isfinite(best_x):
                    ax.axvline(best_x, color="#F58518", linestyle="-.", linewidth=1.0)

                ax.set_xlim(*feature_limits[x_feature])
                ax.set_title(x_feature, fontsize=8)
            else:
                x_vals = pd.to_numeric(cand_feat[x_feature], errors="coerce").to_numpy(dtype=float)
                y_vals = pd.to_numeric(cand_feat[y_feature], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x_vals) & np.isfinite(y_vals) & finite_dist
                if np.any(mask):
                    ax.scatter(
                        x_vals[mask],
                        y_vals[mask],
                        c=cand_distance[mask],
                        s=14,
                        cmap=dist_cmap,
                        norm=dist_norm,
                        alpha=0.90,
                        linewidths=0.0,
                        zorder=2,
                    )

                    sample_x = float(pd.to_numeric(pd.Series([sample_feat_values.get(x_feature)]), errors="coerce").iloc[0])
                    sample_y = float(pd.to_numeric(pd.Series([sample_feat_values.get(y_feature)]), errors="coerce").iloc[0])
                    best_x = float(pd.to_numeric(pd.Series([best_feat_values.get(x_feature)]), errors="coerce").iloc[0])
                    best_y = float(pd.to_numeric(pd.Series([best_feat_values.get(y_feature)]), errors="coerce").iloc[0])
                    if np.isfinite(sample_x) and np.isfinite(sample_y):
                        ax.scatter([sample_x], [sample_y], s=44, marker="*", color="#E45756", edgecolors="black", linewidths=0.45, zorder=4)
                    if np.isfinite(best_x) and np.isfinite(best_y):
                        ax.scatter([best_x], [best_y], s=38, marker="X", color="#F58518", edgecolors="black", linewidths=0.45, zorder=4)
                    plotted_lower_any = True
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7, transform=ax.transAxes)

                ax.set_xlim(*feature_limits[x_feature])
                ax.set_ylim(*feature_limits[y_feature])

            if i < n_features - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(_axis_label_for_feature(x_feature))
            if j > 0:
                ax.set_yticklabels([])
            else:
                if i == j:
                    ax.set_ylabel("Count")
                else:
                    ax.set_ylabel(_axis_label_for_feature(y_feature))

    fig.suptitle(
        "Random showcase feature-space matrix\n"
        f"(dataset_index={ds_idx}, metric={distance_metric}, seed={showcase_seed_used}, "
        f"candidates={len(cand_feat)}, features={n_features})",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.88 if plotted_lower_any else 0.97,
        bottom=0.06,
        top=0.93,
        wspace=0.08,
        hspace=0.08,
    )
    if plotted_lower_any:
        sm = plt.cm.ScalarMappable(norm=dist_norm, cmap=dist_cmap)
        sm.set_array([])
        cax = fig.add_axes([0.905, 0.10, 0.018, 0.78])
        tick_step = max(1, int(np.ceil(n_color_levels / 6)))
        tick_values = dist_levels[::tick_step]
        if not np.isclose(tick_values[-1], dist_levels[-1]):
            tick_values = np.append(tick_values, dist_levels[-1])
        cbar = fig.colorbar(
            sm,
            cax=cax,
            boundaries=dist_levels,
            ticks=tick_values,
            spacing="proportional",
        )
        cbar.set_label(f"{metric_label} distance")

    _save_figure(fig, PLOTS_DIR / "random_showcase_feature_space_matrix.png")
    plt.close(fig)


def _make_plots(
    result_df: pd.DataFrame,
    data_df: pd.DataFrame,
    plot_params=None,
    dict_path: Path | None = None,
    cfg_21: dict | None = None,
    resolved_feature_columns: list[str] | None = None,
) -> None:
    """Quick diagnostic plots for the estimation step."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    metric = str((cfg_21 or {}).get("distance_metric", "unknown"))
    k_cfg = (cfg_21 or {}).get("interpolation_k", None)
    k_label = "all" if k_cfg is None else str(k_cfg)
    exclude_dict_rows = _as_bool(
        (cfg_21 or {}).get("exclude_dictionary_entries_from_plots", True),
        True,
    )
    exclude_dict_paramset = _as_bool(
        (cfg_21 or {}).get("exclude_dictionary_paramset_matches_from_plots", True),
        True,
    )

    plot_df = result_df.copy()
    exclusion_mask = pd.Series(False, index=plot_df.index, dtype=bool)
    if exclude_dict_rows:
        exclusion_mask |= _parse_true_is_dictionary_entry_mask(plot_df)
    if exclude_dict_paramset and dict_path is not None and dict_path.exists():
        try:
            dict_df_for_plots = pd.read_csv(dict_path, low_memory=False)
            exclusion_mask |= _rows_with_dictionary_parameter_set(
                plot_df,
                dict_df_for_plots,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning(
                "Could not resolve dictionary-parameter-set exclusion for STEP 2.1 plots: %s",
                exc,
            )

    n_excluded = int(exclusion_mask.sum())
    if n_excluded > 0:
        plot_df = plot_df.loc[~exclusion_mask].copy()
        log.info(
            "STEP 2.1 plots: excluded %d dictionary-like row(s) from diagnostics "
            "(dict_entries=%s, dict_paramset_matches=%s). Remaining=%d.",
            n_excluded,
            exclude_dict_rows,
            exclude_dict_paramset,
            len(plot_df),
        )
    if plot_df.empty:
        log.warning(
            "STEP 2.1 plots: all rows were excluded by dictionary-like filtering; "
            "falling back to full result set."
        )
        plot_df = result_df.copy()

    # ── 1. Distance diagnostics (distribution + method relevance) ───
    distances = pd.to_numeric(plot_df.get("best_distance"), errors="coerce").dropna()
    if not distances.empty:
        q1 = float(distances.quantile(0.25))
        q3 = float(distances.quantile(0.75))
        iqr = q3 - q1
        upper_fence = float(q3 + 1.5 * iqr) if np.isfinite(iqr) else float(distances.max())
        inlier_mask = distances <= upper_fence
        inliers = distances[inlier_mask]
        n_outliers = int((~inlier_mask).sum())

        q50 = float(distances.quantile(0.50))
        q90 = float(distances.quantile(0.90))
        q95 = float(distances.quantile(0.95))

        fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_cdf = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 0])
        ax_oper = fig.add_subplot(gs[1, 1])

        # 1A) Core histogram (IQR-clipped) with robust quantiles
        hist_values = inliers if len(inliers) >= 5 else distances
        ax_hist.hist(hist_values, bins=45, color="#4C78A8", alpha=0.82, edgecolor="white")
        ax_hist.axvline(q50, color="#E45756", linestyle="--", linewidth=1.6, label=f"p50 = {q50:.4g}")
        ax_hist.axvline(q90, color="#F58518", linestyle="-.", linewidth=1.4, label=f"p90 = {q90:.4g}")
        ax_hist.axvline(q95, color="#72B7B2", linestyle=":", linewidth=1.6, label=f"p95 = {q95:.4g}")
        if n_outliers:
            ax_hist.axvline(
                upper_fence, color="#B279A2", linestyle="-", linewidth=1.1,
                label=f"IQR upper fence = {upper_fence:.4g}",
            )
        ax_hist.set_xlabel("Best distance")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Core distance density (IQR-clipped)")
        ax_hist.legend(fontsize=7.5, loc="upper right")
        ax_hist.text(
            0.02,
            0.98,
            (
                f"N={len(distances)} | outliers={n_outliers} "
                f"({(100.0 * n_outliers / len(distances)):.1f}%)\n"
                f"median={q50:.3g}, p90={q90:.3g}, p95={q95:.3g}"
            ),
            transform=ax_hist.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
        )

        # 1B) Coverage view: fraction of rows below distance threshold
        d_sorted = np.sort(distances.to_numpy(dtype=float))
        cdf_y = np.arange(1, len(d_sorted) + 1, dtype=float) / len(d_sorted)
        ax_cdf.plot(d_sorted, cdf_y, color="#54A24B", linewidth=1.8)
        for value, color, label in [
            (q50, "#E45756", "p50"),
            (q90, "#F58518", "p90"),
            (q95, "#72B7B2", "p95"),
        ]:
            ax_cdf.axvline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.8, label=label)
        ax_cdf.set_xlabel("Best distance threshold")
        ax_cdf.set_ylabel("Fraction with best_distance <= threshold")
        ax_cdf.set_ylim(0.0, 1.02)
        ax_cdf.set_title("Coverage curve (all rows)")
        ax_cdf.legend(fontsize=7.5, loc="lower right")
        # Keep the CDF readable when very large outliers exist.
        cdf_xmax = float(distances.quantile(0.995))
        if np.isfinite(cdf_xmax) and cdf_xmax > 0 and distances.max() > 1.15 * cdf_xmax:
            ax_cdf.set_xlim(0.0, cdf_xmax)
            ax_cdf.text(
                0.02,
                0.03,
                f"Zoomed to 99.5% (max={float(distances.max()):.3g})",
                transform=ax_cdf.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="0.35",
            )

        # 1C) Distance-vs-error calibration: does distance track inference quality?
        selected_params: list[str] = []
        if isinstance(plot_params, (list, tuple, set)):
            selected_params = [
                str(p) for p in plot_params
                if f"true_{p}" in plot_df.columns and f"est_{p}" in plot_df.columns
            ]
        if not selected_params:
            for pname in ["flux_cm2_min", "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4", "cos_n"]:
                if f"true_{pname}" in plot_df.columns and f"est_{pname}" in plot_df.columns:
                    selected_params.append(pname)

        relerr_cols = []
        for pname in selected_params:
            t = pd.to_numeric(plot_df[f"true_{pname}"], errors="coerce")
            e = pd.to_numeric(plot_df[f"est_{pname}"], errors="coerce")
            denom = np.maximum(np.abs(t), 1e-9)
            relerr_cols.append((((e - t).abs() / denom) * 100.0).rename(pname))

        if relerr_cols:
            relerr_df = pd.concat(relerr_cols, axis=1)
            row_relerr = relerr_df.median(axis=1, skipna=True)
            eval_df = pd.DataFrame({
                "distance": pd.to_numeric(plot_df["best_distance"], errors="coerce"),
                "agg_relerr_pct": row_relerr,
            }).dropna()
        else:
            eval_df = pd.DataFrame(columns=["distance", "agg_relerr_pct"])

        if len(eval_df) >= 3:
            ax_err.scatter(
                eval_df["distance"], eval_df["agg_relerr_pct"],
                s=15, alpha=0.35, color="#72B7B2", edgecolors="none",
            )
            if len(eval_df) >= 20 and eval_df["distance"].nunique() >= 6:
                q_edges = np.unique(np.quantile(eval_df["distance"], np.linspace(0.0, 1.0, 9)))
                if len(q_edges) >= 3:
                    dist_bins = pd.cut(eval_df["distance"], bins=q_edges, include_lowest=True, duplicates="drop")
                    trend = (
                        eval_df.assign(dist_bin=dist_bins)
                        .groupby("dist_bin", observed=True)
                        .agg(
                            distance_mid=("distance", "median"),
                            relerr_median=("agg_relerr_pct", "median"),
                        )
                        .dropna()
                    )
                    if not trend.empty:
                        ax_err.plot(
                            trend["distance_mid"], trend["relerr_median"],
                            color="#E45756", linewidth=1.7, marker="o",
                            label="Median |rel.err| across distance quantiles",
                        )

            pearson = float(eval_df["distance"].corr(eval_df["agg_relerr_pct"], method="pearson"))
            spearman = float(eval_df["distance"].corr(eval_df["agg_relerr_pct"], method="spearman"))
            ptxt = f"{pearson:.2f}" if np.isfinite(pearson) else "nan"
            stxt = f"{spearman:.2f}" if np.isfinite(spearman) else "nan"
            shown_params = ", ".join(selected_params[:3]) + ("..." if len(selected_params) > 3 else "")
            ax_err.set_title("Distance vs estimation error")
            ax_err.set_xlabel("Best distance")
            ax_err.set_ylabel("Median |relative error| [%]")
            ax_err.text(
                0.02,
                0.98,
                f"Params: {shown_params}\nPearson={ptxt}, Spearman={stxt}",
                transform=ax_err.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
            )
            y_hi = float(eval_df["agg_relerr_pct"].quantile(0.99))
            if np.isfinite(y_hi) and y_hi > 0:
                ax_err.set_ylim(0.0, max(0.5, 1.15 * y_hi))
            x_hi = float(eval_df["distance"].quantile(0.99))
            if np.isfinite(x_hi) and x_hi > 0 and eval_df["distance"].max() > 1.15 * x_hi:
                ax_err.set_xlim(0.0, x_hi)
                ax_err.text(
                    0.98,
                    0.03,
                    f"Zoomed to p99 (max={float(eval_df['distance'].max()):.3g})",
                    transform=ax_err.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )
            if ax_err.get_legend_handles_labels()[0]:
                ax_err.legend(fontsize=7.5, loc="upper right")
        else:
            ax_err.text(
                0.5, 0.5,
                "Not enough true/estimated\nparameter overlap for\nerror-calibration panel",
                transform=ax_err.transAxes,
                ha="center", va="center", fontsize=9,
            )
            ax_err.set_title("Distance vs estimation error")
            ax_err.set_xlabel("Best distance")
            ax_err.set_ylabel("Median |relative error| [%]")

        # 1D) Distance operating curve: threshold trade-off for coverage vs quality
        if len(eval_df) >= 20 and eval_df["distance"].nunique() >= 8:
            thr = np.unique(np.quantile(eval_df["distance"], np.linspace(0.05, 0.95, 15)))
            if len(thr) >= 3:
                coverage_pct = []
                med_relerr = []
                p90_relerr = []
                n_eval = float(len(eval_df))
                for tval in thr:
                    subset = eval_df[eval_df["distance"] <= tval]
                    if subset.empty:
                        coverage_pct.append(np.nan)
                        med_relerr.append(np.nan)
                        p90_relerr.append(np.nan)
                    else:
                        coverage_pct.append(100.0 * len(subset) / n_eval)
                        med_relerr.append(float(subset["agg_relerr_pct"].median()))
                        p90_relerr.append(float(subset["agg_relerr_pct"].quantile(0.90)))

                ax_oper.plot(
                    thr, coverage_pct, color="#54A24B", linewidth=1.8, marker="o",
                    markersize=3.2, label="Coverage retained [%]",
                )
                ax_oper.set_xlabel("Distance threshold")
                ax_oper.set_ylabel("Coverage retained [%]", color="#2F6B2D")
                ax_oper.tick_params(axis="y", labelcolor="#2F6B2D")
                ax_oper.set_ylim(0.0, 101.0)

                ax_err2 = ax_oper.twinx()
                ax_err2.plot(
                    thr, med_relerr, color="#E45756", linewidth=1.6, marker="s",
                    markersize=3.0, label="Median |rel.err| [%]",
                )
                ax_err2.plot(
                    thr, p90_relerr, color="#F58518", linewidth=1.3, linestyle="--",
                    label="p90 |rel.err| [%]",
                )
                ax_err2.set_ylabel("Error among retained rows [%]", color="#A94D00")
                ax_err2.tick_params(axis="y", labelcolor="#A94D00")

                star_thr = q90
                star_subset = eval_df[eval_df["distance"] <= star_thr]
                if len(star_subset) > 0:
                    star_cov = 100.0 * len(star_subset) / len(eval_df)
                    star_med = float(star_subset["agg_relerr_pct"].median())
                    ax_oper.axvline(star_thr, color="0.45", linestyle=":", linewidth=1.0)
                    ax_oper.text(
                        0.02,
                        0.98,
                        (
                            f"At p90 threshold ({star_thr:.3g}):\n"
                            f"coverage={star_cov:.1f}%, median err={star_med:.2f}%"
                        ),
                        transform=ax_oper.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
                    )

                ax_oper.set_title("Operating curve: threshold trade-off")
                h1, l1 = ax_oper.get_legend_handles_labels()
                h2, l2 = ax_err2.get_legend_handles_labels()
                ax_oper.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")
            else:
                ax_oper.text(
                    0.5, 0.5, "Insufficient distance spread\nfor operating-curve panel",
                    transform=ax_oper.transAxes, ha="center", va="center", fontsize=9,
                )
                ax_oper.set_title("Operating curve: threshold trade-off")
                ax_oper.set_xlabel("Distance threshold")
                ax_oper.set_ylabel("Coverage retained [%]")
        else:
            ax_oper.text(
                0.5, 0.5, "Not enough rows with error estimates\nfor operating-curve panel",
                transform=ax_oper.transAxes, ha="center", va="center", fontsize=9,
            )
            ax_oper.set_title("Operating curve: threshold trade-off")
            ax_oper.set_xlabel("Distance threshold")
            ax_oper.set_ylabel("Coverage retained [%]")

        fig.suptitle(
            f"Best-match distance diagnostics (metric={metric}, IDW K={k_label})",
            fontsize=11,
        )
        _save_figure(fig, PLOTS_DIR / "distance_distribution.png")
        plt.close(fig)

    # ── 2. True vs estimated scatter for available params ────────────
    # Build all possible pairs, then filter by plot_parameters if set
    all_param_pairs = []
    for col in plot_df.columns:
        if col.startswith("est_"):
            pname = col[4:]  # strip "est_"
            true_col = f"true_{pname}"
            if true_col in plot_df.columns:
                all_param_pairs.append((true_col, col, pname))
    if plot_params:
        all_param_pairs = [(t, e, l) for t, e, l in all_param_pairs
                           if l in plot_params]
    valid_pairs = all_param_pairs

    if valid_pairs:
        n_p = len(valid_pairs)
        fig, axes = plt.subplots(1, n_p, figsize=(5 * n_p, 5.4))
        if n_p == 1:
            axes = [axes]
        for ax, (true_col, est_col, label) in zip(axes, valid_pairs):
            t = pd.to_numeric(plot_df[true_col], errors="coerce")
            e = pd.to_numeric(plot_df[est_col], errors="coerce")
            m = t.notna() & e.notna()
            if m.sum() > 0:
                ax.scatter(t[m], e[m], s=12, alpha=0.5, color="#F58518")
                lo = min(t[m].min(), e[m].min())
                hi = max(t[m].max(), e[m].max())
                pad = 0.02 * (hi - lo) if hi > lo else 0.01
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        "k--", linewidth=1)
            ax.set_xlabel(f"True {label}")
            ax.set_ylabel(f"Estimated {label}")
            ax.set_title(f"True vs Est: {label}")
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(f"Parameter estimation: true vs estimated (metric={metric})", fontsize=11, y=0.98)
        # Keep enough bottom padding for long x-labels and ensure nothing is clipped on save.
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.88, wspace=0.26)
        _save_figure(
            fig,
            PLOTS_DIR / "true_vs_estimated.png",
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(fig)

    # ── 3. Random showcase matrix: lower-triangle 2D maps + diagonal projections ──
    if dict_path is not None:
        shared_showcase = _select_showcase_result_row(
            result_df=plot_df,
            cfg_21=cfg_21 or {},
            context_label="Shared showcase",
        )
        shared_row_idx: int | None = None
        shared_ds_idx: int | None = None
        shared_seed: int | None = None
        if shared_showcase is not None:
            shared_row_idx, shared_ds_idx, shared_seed = shared_showcase
        _make_random_showcase_distance_matrix(
            result_df=result_df,
            data_df=data_df,
            dict_path=dict_path,
            cfg_21=cfg_21 or {},
            plot_params=plot_params,
            resolved_feature_columns=resolved_feature_columns,
            showcase_result_row_index=shared_row_idx,
            showcase_dataset_index=shared_ds_idx,
            showcase_seed=shared_seed,
        )
        _make_random_showcase_feature_space_matrix(
            result_df=result_df,
            data_df=data_df,
            dict_path=dict_path,
            cfg_21=cfg_21 or {},
            plot_params=plot_params,
            resolved_feature_columns=resolved_feature_columns,
            showcase_result_row_index=shared_row_idx,
            showcase_dataset_index=shared_ds_idx,
            showcase_seed=shared_seed,
        )

    # ── 4. Dictionary continuity validation summary ───────────────
    if dict_path is not None:
        _plot_dictionary_continuity_summary(
            result_df=plot_df,
            dict_path=dict_path,
        )


def _plot_dictionary_continuity_summary(
    result_df: pd.DataFrame,
    dict_path: Path,
) -> None:
    """Load STEP 1.2 continuity validation and plot a summary alongside estimation quality."""
    build_summary_path = dict_path.parent / "build_summary.json"
    if not build_summary_path.exists():
        log.info("No build_summary.json found at %s — skipping continuity plot.", build_summary_path)
        return

    try:
        with open(build_summary_path, encoding="utf-8") as f:
            build_summary = json.load(f)
    except Exception as exc:
        log.warning("Could not load build_summary.json: %s", exc)
        return

    cv = build_summary.get("continuity_validation")
    if not cv or not cv.get("enabled", False):
        log.info("Continuity validation not enabled in build_summary — skipping plot.")
        return

    checks = cv.get("checks", {})
    messages = cv.get("messages", [])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── (0,0) Estimation error vs distance ────────────────────────
    ax = axes[0, 0]
    dist_col = "best_distance"
    # Try flux first, then eff
    flux_true = "true_flux_cm2_min"
    flux_est = "est_flux_cm2_min"
    if dist_col in result_df.columns and flux_true in result_df.columns and flux_est in result_df.columns:
        d = pd.to_numeric(result_df[dist_col], errors="coerce")
        t = pd.to_numeric(result_df[flux_true], errors="coerce")
        e = pd.to_numeric(result_df[flux_est], errors="coerce")
        m = d.notna() & t.notna() & e.notna() & (t.abs() > 1e-15)
        if m.sum() > 0:
            relerr = ((e[m] - t[m]) / t[m] * 100.0).to_numpy()
            dists = d[m].to_numpy()
            sc = ax.scatter(dists, relerr, s=8, alpha=0.5, c="#4C78A8", edgecolors="none")
            ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Best distance")
            ax.set_ylabel("Flux relative error [%]")
            q90 = float(np.nanpercentile(np.abs(relerr), 90))
            ax.set_title(f"Error vs Distance (|relerr| P90={q90:.1f}%)")
    else:
        ax.text(0.5, 0.5, "No distance/flux data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Error vs Distance")

    # ── (0,1) Per-plane support adequacy from STEP 1.2 ────────────
    ax = axes[0, 1]
    c4 = checks.get("support_adequacy", {})
    dict_ranges = c4.get("dict_range_by_plane", {})
    ds_ranges = c4.get("dataset_range_by_plane", {})
    oos_fracs = c4.get("out_of_support_fraction_by_plane", {})
    planes = sorted(dict_ranges.keys())
    if planes:
        y_pos = np.arange(len(planes))
        for i, plane_key in enumerate(planes):
            dr = dict_ranges[plane_key]
            dsr = ds_ranges.get(plane_key, dr)
            ax.barh(i, dsr["max"] - dsr["min"], left=dsr["min"], height=0.5,
                    color="#CCCCCC", alpha=0.6, label="Dataset" if i == 0 else "")
            ax.barh(i, dr["max"] - dr["min"], left=dr["min"], height=0.5,
                    color="#4C78A8", alpha=0.8, label="Dictionary" if i == 0 else "")
            oos = oos_fracs.get(plane_key, 0)
            ax.text(max(dr["max"], dsr["max"]) + 0.005, i, f"OOS: {oos:.0%}", fontsize=8, va="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([p.replace("eff_sim_", "Plane ") for p in planes])
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlabel("Efficiency range")
    ax.set_title(f"Support Adequacy ({c4.get('status', 'N/A')})")

    # ── (1,0) Local continuity CV by parameter ────────────────────
    ax = axes[1, 0]
    c2 = checks.get("local_continuity", {})
    cv_p95 = c2.get("cv_p95_by_param", {})
    if cv_p95:
        params = list(cv_p95.keys())
        vals = [cv_p95[p] for p in params]
        colors = ["#E45756" if v > 0.50 else "#F58518" if v > 0.30 else "#4C78A8" for v in vals]
        y_pos = np.arange(len(params))
        ax.barh(y_pos, vals, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([p.replace("_", " ") for p in params], fontsize=8)
        ax.axvline(0.50, color="#E45756", linewidth=1, linestyle="--", alpha=0.7, label="Threshold (0.50)")
        ax.legend(fontsize=7)
        ax.set_xlabel("CV (P95)")
    ax.set_title(f"Local Continuity by Param ({c2.get('status', 'N/A')})")

    # ── (1,1) Overall validation summary ──────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    color_map = {"PASS": "#2A9D8F", "WARN": "#E9C46A", "FAIL": "#E45756", "SKIPPED": "#999999"}
    lines = [f"Overall: {cv.get('status', 'N/A')}"]
    for name, chk in checks.items():
        st = chk.get("status", "?")
        lines.append(f"  {name}: {st}")
    if messages:
        lines.append("")
        for m in messages[:6]:
            lines.append(f"  * {m}")

    # Add estimation quality metrics
    lines.append("")
    if flux_true in result_df.columns and flux_est in result_df.columns:
        t = pd.to_numeric(result_df[flux_true], errors="coerce")
        e = pd.to_numeric(result_df[flux_est], errors="coerce")
        m = t.notna() & e.notna() & (t.abs() > 1e-15)
        if m.sum() > 0:
            relerr = np.abs((e[m] - t[m]) / t[m] * 100.0)
            lines.append(f"  Flux |relerr| median: {float(np.median(relerr)):.2f}%")
            lines.append(f"  Flux |relerr| P90:    {float(np.percentile(relerr, 90)):.2f}%")

    y_start = 0.95
    for i, line in enumerate(lines):
        st_word = line.strip().split(":")[-1].strip().split()[0] if ":" in line else ""
        color = color_map.get(st_word, "#333333")
        ax.text(0.05, y_start - i * 0.058, line, fontsize=9, family="monospace",
                color=color, transform=ax.transAxes, va="top")
    ax.set_title("Dictionary Quality + Estimation Summary")

    fig.suptitle("STEP 2.1: Dictionary Continuity & Estimation Quality", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "dictionary_continuity_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
