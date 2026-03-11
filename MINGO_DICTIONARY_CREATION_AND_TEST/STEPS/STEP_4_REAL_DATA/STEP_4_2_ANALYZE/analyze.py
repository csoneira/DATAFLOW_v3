#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_4_REAL_DATA/STEP_4_2_ANALYZE/analyze.py
Purpose: STEP 4.2 - Run inference on real data and attach LUT uncertainties.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_4_REAL_DATA/STEP_4_2_ANALYZE/analyze.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np
import pandas as pd

# -- Paths --------------------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
# Support both layouts:
#   - <pipeline>/STEP_4_REAL_DATA/STEP_4_2_ANALYZE
#   - <pipeline>/STEPS/STEP_4_REAL_DATA/STEP_4_2_ANALYZE
if STEP_DIR.parents[2].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[3]
else:
    PIPELINE_DIR = STEP_DIR.parents[2]
DEFAULT_CONFIG = PIPELINE_DIR / "config_method.json"
CONFIG_COLUMNS_PATH = PIPELINE_DIR / "config_columns.json"

if (PIPELINE_DIR / "STEP_1_SETUP").exists() and (PIPELINE_DIR / "STEP_2_INFERENCE").exists():
    STEP_ROOT = PIPELINE_DIR
else:
    STEP_ROOT = PIPELINE_DIR / "STEPS"

INFERENCE_DIR = STEP_ROOT / "STEP_2_INFERENCE"

DEFAULT_REAL_COLLECTED = (
    STEP_DIR.parent
    / "STEP_4_1_COLLECT_REAL_DATA"
    / "OUTPUTS"
    / "FILES"
    / "real_collected_data.csv"
)
DEFAULT_DICTIONARY = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "dictionary.csv"
)
DEFAULT_LUT = (
    STEP_ROOT
    / "STEP_2_INFERENCE"
    / "STEP_2_3_UNCERTAINTY"
    / "OUTPUTS"
    / "FILES"
    / "uncertainty_lut.csv"
)
DEFAULT_LUT_META = (
    STEP_ROOT
    / "STEP_2_INFERENCE"
    / "STEP_2_3_UNCERTAINTY"
    / "OUTPUTS"
    / "FILES"
    / "uncertainty_lut_meta.json"
)
DEFAULT_BUILD_SUMMARY = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "build_summary.json"
)
DEFAULT_STEP12_SELECTED_FEATURE_COLUMNS = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_2_BUILD_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "selected_feature_columns.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_EFF = PLOTS_DIR / "STEP_4_2_2_efficiency_vs_time.png"
PLOT_EST_CURVE = PLOTS_DIR / "STEP_4_2_6_estimated_curve_flux_vs_eff.png"
PLOT_RECOVERY_STORY = PLOTS_DIR / "STEP_4_2_7_flux_recovery_vs_global_rate.png"

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

logging.basicConfig(format="[%(levelname)s] STEP_4.2 - %(message)s", level=logging.INFO)
log = logging.getLogger("STEP_4.2")

# Import estimator directly from STEP_2_INFERENCE.
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
try:
    from estimate_parameters import (  # noqa: E402
        DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL,
        _append_derived_physics_feature_columns,
        _append_derived_tt_global_rate_column,
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
    log.error("Could not import estimate_from_dataframes from %s: %s", INFERENCE_DIR, exc)
    raise


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
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isfinite(out):
        return out
    return float(default)


def _safe_bool(value: object, default: bool) -> bool:
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


def _safe_int(value: object, default: int, *, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    if minimum is not None:
        out = max(int(minimum), out)
    return out


def _safe_task_ids(raw: object) -> list[int]:
    if raw is None:
        return [1]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return [1]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = [x.strip() for x in stripped.split(",") if x.strip()]
        raw = parsed
    out: list[int] = []
    if isinstance(raw, (list, tuple)):
        for value in raw:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
    return sorted(set(out)) or [1]


def _preferred_tt_prefixes_for_task_ids(task_ids: list[int]) -> list[str]:
    """Preferred TT-rate prefixes for efficiency extraction by most advanced task."""
    max_task_id = max(task_ids) if task_ids else 1
    if max_task_id <= 1:
        return ["raw"]
    if max_task_id == 2:
        return ["clean", "raw_to_clean", "raw"]
    if max_task_id == 3:
        return ["cal", "clean", "raw_to_clean", "raw"]
    if max_task_id == 4:
        return ["list", "list_to_fit", "cal", "clean", "raw_to_clean", "raw"]
    return [
        "corr",
        "task5_to_corr",
        "fit_to_corr",
        "definitive",
        "fit",
        "list_to_fit",
        "list",
        "cal",
        "clean",
        "raw_to_clean",
        "raw",
    ]


def _preferred_feature_prefixes_for_task_ids(task_ids: list[int]) -> list[str]:
    """Preferred TT-rate prefixes for STEP 4.2 auto feature selection."""
    max_task_id = max(task_ids) if task_ids else 1
    if max_task_id <= 1:
        return ["raw", "clean"]
    if max_task_id == 2:
        return ["clean", "raw_to_clean", "raw"]
    if max_task_id == 3:
        return ["cal", "clean", "raw_to_clean", "raw"]
    if max_task_id == 4:
        return ["fit", "list_to_fit", "list", "cal", "clean", "raw_to_clean", "raw"]
    return [
        "post",
        "fit_to_post",
        "fit",
        "list_to_fit",
        "list",
        "cal",
        "clean",
        "raw_to_clean",
        "raw",
        "corr",
        "task5_to_corr",
        "fit_to_corr",
        "definitive",
    ]


def _coalesce(primary: object, fallback: object) -> object:
    if primary in (None, "", "null", "None"):
        return fallback
    return primary


def _resolve_input_path(path_like: str | Path) -> Path:
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


def _parse_ts(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    parsed = pd.to_datetime(s, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed_fallback = pd.to_datetime(s[missing], errors="coerce", utc=True)
        parsed.loc[missing] = parsed_fallback
    return parsed


def _extract_tt_parts(col: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<prefix>.+?)_tt_(?P<rest>.+)_rate_hz$", col)
    if match is None:
        return None
    rest = match.group("rest")
    # Some task outputs use names like tt_1234.0_rate_hz; normalize to tt_1234_rate_hz.
    rest = re.sub(r"\.0$", "", rest)
    return (match.group("prefix"), f"tt_{rest}_rate_hz")


def _tt_rate_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if re.search(r"_tt_.+_rate_hz$", c)])


def _prefix_rank(prefix: str) -> int:
    order = [
        "raw",
        "clean",
        "cal",
        "list",
        "fit",
        "corr",
        "definitive",
        "raw_to_clean",
        "list_to_fit",
        "fit_to_corr",
        "task5_to_corr",
    ]
    try:
        return order.index(prefix)
    except ValueError:
        return len(order)


def _choose_best_col(columns: list[str], df: pd.DataFrame | None = None) -> str:
    scored: list[tuple[int, int, str]] = []
    for col in columns:
        parts = _extract_tt_parts(col)
        rank = _prefix_rank(parts[0]) if parts is not None else 999
        finite_rank = 0
        if df is not None and col in df.columns:
            finite_count = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            finite_rank = -finite_count
        scored.append((finite_rank, rank, col))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][2]


def _count_rows_with_min_finite_features(df: pd.DataFrame, features: list[str], min_finite: int = 2) -> int:
    if not features:
        return 0
    usable_min = min(max(1, int(min_finite)), len(features))
    vals = df[features].apply(pd.to_numeric, errors="coerce")
    return int((vals.notna().sum(axis=1) >= usable_min).sum())


def _resolve_feature_columns_auto(
    dict_df: pd.DataFrame,
    real_df: pd.DataFrame,
    include_global_rate: bool,
    global_rate_col: str,
    preferred_prefixes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str, list[dict[str, str]]]:
    """Build a robust feature set even when task prefixes differ."""
    dict_tt = _tt_rate_columns(dict_df)
    real_tt = _tt_rate_columns(real_df)

    feature_mapping: list[dict[str, str]] = []

    # 1) Prefer same-prefix direct intersections (mirrors STEP 2.1 behavior).
    prefixes = [str(p).strip() for p in (preferred_prefixes or []) if str(p).strip()]
    if not prefixes:
        prefixes = ["raw", "clean", "cal", "list", "fit", "corr", "definitive"]
    for prefix in prefixes:
        common = sorted(
            [c for c in dict_tt if c.startswith(f"{prefix}_tt_") and c in set(real_tt)]
        )
        if common:
            features = common.copy()
            if (
                include_global_rate
                and global_rate_col in dict_df.columns
                and global_rate_col in real_df.columns
                and global_rate_col not in features
            ):
                features.append(global_rate_col)
            min_finite = 2 if len(features) >= 2 else 1
            dict_rows_usable = _count_rows_with_min_finite_features(
                dict_df, features, min_finite=min_finite
            )
            real_rows_usable = _count_rows_with_min_finite_features(
                real_df, features, min_finite=min_finite
            )
            if dict_rows_usable <= 0 or real_rows_usable <= 0:
                log.info(
                    "Skipping direct_prefix:%s (usable rows with >=%d finite features: dict=%d, real=%d).",
                    prefix,
                    min_finite,
                    dict_rows_usable,
                    real_rows_usable,
                )
                continue
            log.info(
                "Selected direct_prefix:%s with %d features (usable rows: dict=%d, real=%d).",
                prefix,
                len(features),
                dict_rows_usable,
                real_rows_usable,
            )
            return (
                dict_df,
                real_df,
                features,
                f"direct_prefix:{prefix}",
                feature_mapping,
            )

    # 2) Any exact common tt-rate columns.
    exact_common = sorted(set(dict_tt) & set(real_tt))
    if exact_common:
        features = exact_common.copy()
        if (
            include_global_rate
            and global_rate_col in dict_df.columns
            and global_rate_col in real_df.columns
            and global_rate_col not in features
        ):
            features.append(global_rate_col)
        min_finite = 2 if len(features) >= 2 else 1
        dict_rows_usable = _count_rows_with_min_finite_features(
            dict_df, features, min_finite=min_finite
        )
        real_rows_usable = _count_rows_with_min_finite_features(
            real_df, features, min_finite=min_finite
        )
        if dict_rows_usable > 0 and real_rows_usable > 0:
            log.info(
                "Selected direct_exact with %d features (usable rows: dict=%d, real=%d).",
                len(features),
                dict_rows_usable,
                real_rows_usable,
            )
            return (dict_df, real_df, features, "direct_exact", feature_mapping)
        log.info(
            "Skipping direct_exact (usable rows with >=%d finite features: dict=%d, real=%d).",
            min_finite,
            dict_rows_usable,
            real_rows_usable,
        )

    # 3) Align by tt topology key and create temporary aliases.
    dict_key_to_cols: dict[str, list[str]] = {}
    real_key_to_cols: dict[str, list[str]] = {}

    for col in dict_tt:
        parts = _extract_tt_parts(col)
        if parts is None:
            continue
        key = parts[1]
        dict_key_to_cols.setdefault(key, []).append(col)

    for col in real_tt:
        parts = _extract_tt_parts(col)
        if parts is None:
            continue
        key = parts[1]
        real_key_to_cols.setdefault(key, []).append(col)

    common_keys = sorted(set(dict_key_to_cols) & set(real_key_to_cols))
    if not common_keys:
        raise ValueError(
            "No compatible *_tt_*_rate_hz features found between dictionary and real data."
        )

    dict_work = dict_df.copy()
    real_work = real_df.copy()
    features: list[str] = []
    for idx, key in enumerate(common_keys):
        dcol = _choose_best_col(dict_key_to_cols[key], dict_df)
        rcol = _choose_best_col(real_key_to_cols[key], real_df)
        dvals = pd.to_numeric(dict_df[dcol], errors="coerce")
        rvals = pd.to_numeric(real_df[rcol], errors="coerce")
        if int(dvals.notna().sum()) <= 0 or int(rvals.notna().sum()) <= 0:
            continue
        alias = f"tt_feature_{idx:03d}_rate_hz"
        dict_work[alias] = dvals
        real_work[alias] = rvals
        features.append(alias)
        feature_mapping.append(
            {
                "feature_alias": alias,
                "dictionary_column": dcol,
                "real_column": rcol,
                "tt_key": key,
            }
        )

    if include_global_rate and global_rate_col in dict_df.columns and global_rate_col in real_df.columns:
        alias = "global_rate_feature_hz"
        dict_work[alias] = pd.to_numeric(dict_work[global_rate_col], errors="coerce")
        real_work[alias] = pd.to_numeric(real_work[global_rate_col], errors="coerce")
        features.append(alias)
        feature_mapping.append(
            {
                "feature_alias": alias,
                "dictionary_column": global_rate_col,
                "real_column": global_rate_col,
                "tt_key": "global_rate",
            }
        )

    if not features:
        raise ValueError(
            "No usable aligned tt-rate features found between dictionary and real data."
        )

    min_finite = 2 if len(features) >= 2 else 1
    dict_rows_usable = _count_rows_with_min_finite_features(
        dict_work, features, min_finite=min_finite
    )
    real_rows_usable = _count_rows_with_min_finite_features(
        real_work, features, min_finite=min_finite
    )
    if dict_rows_usable <= 0 or real_rows_usable <= 0:
        raise ValueError(
            "Aligned tt-rate features are not usable: "
            f"dict rows with >={min_finite} finite features={dict_rows_usable}, "
            f"real rows with >={min_finite} finite features={real_rows_usable}."
        )
    log.info(
        "Selected aligned_by_tt_key with %d features (usable rows: dict=%d, real=%d).",
        len(features),
        dict_rows_usable,
        real_rows_usable,
    )

    return (dict_work, real_work, features, "aligned_by_tt_key", feature_mapping)


def _pick_n_events_column(df: pd.DataFrame) -> str | None:
    priority = [
        "n_events",
        "selected_rows",
        "requested_rows",
        "raw_tt_1234_count",
        "clean_tt_1234_count",
        "list_tt_1234_count",
        "fit_tt_1234_count",
        "corr_tt_1234_count",
        "definitive_tt_1234_count",
        "events_per_second_total_seconds",
    ]
    for col in priority:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col

    patt = re.compile(r"_tt_1234(?:\.0)?_count$")
    for col in sorted([c for c in df.columns if patt.search(c)]):
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return col
    return None


def _normalize_tt_label(label: object) -> str:
    text = str(label).strip()
    if not text:
        return ""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return text


def _derive_global_rate_from_tt_sum(
    df: pd.DataFrame,
    *,
    target_col: str = "events_per_second_global_rate",
) -> str | None:
    canonical_labels = {
        "0",
        "1",
        "2",
        "3",
        "4",
        "12",
        "13",
        "14",
        "23",
        "24",
        "34",
        "123",
        "124",
        "134",
        "234",
        "1234",
    }
    by_prefix: dict[str, list[str]] = {}
    for col in df.columns:
        match = re.match(r"^(?P<prefix>.+_tt)_(?P<label>[^_]+)_rate_hz$", str(col))
        if match is None:
            continue
        prefix = str(match.group("prefix")).strip()
        label = _normalize_tt_label(match.group("label"))
        if label not in canonical_labels:
            continue
        by_prefix.setdefault(prefix, []).append(col)

    if not by_prefix:
        return None

    selected_prefix = min(
        by_prefix.keys(),
        key=lambda p: (-len(by_prefix[p]), _prefix_rank(p.replace("_tt", "")), p),
    )
    cols = sorted(set(by_prefix[selected_prefix]))
    if not cols:
        return None

    summed = pd.Series(0.0, index=df.index, dtype=float)
    valid_any = pd.Series(False, index=df.index)
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        summed = summed + numeric.fillna(0.0)
        valid_any = valid_any | numeric.notna()

    df[target_col] = summed.where(valid_any, np.nan)
    return target_col


def _pick_global_rate_column(df: pd.DataFrame, preferred: str = "events_per_second_global_rate") -> str | None:
    if preferred in df.columns:
        vals = pd.to_numeric(df[preferred], errors="coerce")
        if vals.notna().any():
            return preferred

    candidates = []
    for c in df.columns:
        cl = c.lower()
        if "global_rate" in cl and ("hz" in cl or cl.endswith("_rate")):
            candidates.append(c)
    for c in sorted(candidates):
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            return c
    return _derive_global_rate_from_tt_sum(df, target_col="events_per_second_global_rate")


def _find_tt_rate_column(
    df: pd.DataFrame,
    tt_code: str,
    preferred_prefixes: list[str] | None = None,
) -> str | None:
    pattern = re.compile(rf"_tt_{re.escape(tt_code)}(?:\.0)?_rate_hz$")
    candidates = [c for c in df.columns if pattern.search(c)]
    if not candidates:
        return None
    preferred_order: dict[str, int] = {}
    if preferred_prefixes:
        preferred_order = {str(p): i for i, p in enumerate(preferred_prefixes)}

    def _sort_key(col: str) -> tuple[int, int, str]:
        parts = _extract_tt_parts(col)
        prefix = parts[0] if parts is not None else ""
        pref_rank = preferred_order.get(prefix, len(preferred_order) + 100)
        base_rank = _prefix_rank(prefix) if parts is not None else 999
        return (pref_rank, base_rank, col)

    candidates = sorted(candidates, key=_sort_key)
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            return c
    return candidates[0]


def _compute_eff_from_rates(
    df: pd.DataFrame,
    *,
    col_missing_rate: str | None,
    col_1234_rate: str | None,
) -> tuple[pd.Series, str]:
    if col_missing_rate is None or col_1234_rate is None:
        return (pd.Series(np.nan, index=df.index), "missing_rate_columns")

    r_miss = pd.to_numeric(df[col_missing_rate], errors="coerce")
    r_1234 = pd.to_numeric(df[col_1234_rate], errors="coerce")
    # Empirical efficiency definition consistent with STEP 1.2 dictionary:
    #   eff = N(1234) / ( N(1234) + N(three-plane-missing-i) )
    # This keeps efficiencies bounded in [0, 1] for non-negative rates.
    denom = (r_1234 + r_miss).replace({0.0: np.nan})
    eff = r_1234 / denom
    eff = eff.where(np.isfinite(eff), np.nan)
    return (eff, f"{col_1234_rate}/({col_1234_rate} + {col_missing_rate})")


def _compute_empirical_efficiencies_from_rates(
    df: pd.DataFrame,
) -> tuple[dict[int, pd.Series], dict[int, str], dict[int, dict[str, str | None]], str | None]:
    """Compute plane efficiencies from four/(four+three-missing) using TT rates."""
    preferred_prefixes: list[str] | None = None
    if isinstance(df.attrs.get("preferred_tt_prefixes"), list):
        preferred_prefixes = [str(v) for v in df.attrs.get("preferred_tt_prefixes", [])]

    four_col = _find_tt_rate_column(df, "1234", preferred_prefixes=preferred_prefixes)
    miss_by_plane = {1: "234", 2: "134", 3: "124", 4: "123"}

    selected_prefix: str | None = None
    four_parts = _extract_tt_parts(four_col) if four_col is not None else None
    if four_parts is not None:
        selected_prefix = four_parts[0]

    eff_by_plane: dict[int, pd.Series] = {}
    formula_by_plane: dict[int, str] = {}
    cols_by_plane: dict[int, dict[str, str | None]] = {}
    for plane, miss_code in miss_by_plane.items():
        miss_col = _find_tt_rate_column(df, miss_code, preferred_prefixes=preferred_prefixes)
        eff, formula = _compute_eff_from_rates(
            df,
            col_missing_rate=miss_col,
            col_1234_rate=four_col,
        )
        eff_by_plane[plane] = eff
        formula_by_plane[plane] = formula
        cols_by_plane[plane] = {
            "three_plane_col": miss_col,
            "four_plane_col": four_col,
        }
    return (eff_by_plane, formula_by_plane, cols_by_plane, selected_prefix)


def _format_polynomial_expr(
    coeffs: np.ndarray | list[float] | tuple[float, ...],
    *,
    variable: str = "x",
    precision: int = 6,
) -> str:
    """Return compact polynomial expression from highest to lowest degree."""
    arr = np.asarray(coeffs, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return "invalid"
    degree = arr.size - 1
    parts: list[tuple[str, str]] = []
    for idx, coef in enumerate(arr):
        if abs(float(coef)) < 1e-14:
            continue
        power = degree - idx
        mag = f"{abs(float(coef)):.{precision}g}"
        if power == 0:
            term = mag
        elif power == 1:
            term = variable if np.isclose(abs(float(coef)), 1.0) else f"{mag}{variable}"
        else:
            term = f"{variable}^{power}" if np.isclose(abs(float(coef)), 1.0) else f"{mag}{variable}^{power}"
        sign = "-" if coef < 0 else "+"
        parts.append((sign, term))
    if not parts:
        return "0"
    first_sign, first_term = parts[0]
    expr = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in parts[1:]:
        expr += f" {sign} {term}"
    return expr


def _invert_polynomial_values(
    y_values: pd.Series,
    coeffs: np.ndarray,
) -> pd.Series:
    """Solve P(x)=y row-wise with physical-root preference and smooth fallback.

    Prefer real roots within [0, 1] (efficiency domain). When no physical root
    exists, fall back to nearby real roots instead of hard clipping to bounds.
    """
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    degree = int(len(coeffs) - 1)

    x_grid = np.linspace(-0.5, 1.5, 8001, dtype=float)
    y_grid = np.polyval(coeffs, x_grid)

    def _distance_to_unit_interval(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        return np.where(vals < 0.0, -vals, np.where(vals > 1.0, vals - 1.0, 0.0))

    if degree == 1:
        a, b = float(coeffs[0]), float(coeffs[1])
        if np.isfinite(a) and abs(a) >= 1e-12:
            out = (y - b) / a
            out[~np.isfinite(out)] = np.nan
        return pd.Series(out, index=y_values.index)

    for idx, target in enumerate(y):
        if not np.isfinite(target):
            continue
        coeff_eq = coeffs.copy()
        coeff_eq[-1] -= float(target)
        try:
            roots = np.roots(coeff_eq)
        except Exception:
            continue
        if roots.size == 0:
            continue
        real_mask = np.isfinite(roots.real) & np.isfinite(roots.imag) & (np.abs(roots.imag) < 1e-8)
        real_roots = roots.real[real_mask]
        if real_roots.size == 0:
            idx_best = int(np.argmin(np.abs(y_grid - target)))
            out[idx] = float(x_grid[idx_best])
            continue
        physical_roots = real_roots[(real_roots >= 0.0) & (real_roots <= 1.0)]
        if physical_roots.size > 0:
            clipped_target = float(np.clip(target, 0.0, 1.0))
            order = np.lexsort(
                (
                    np.abs(physical_roots - 0.5),
                    np.abs(physical_roots - clipped_target),
                )
            )
            out[idx] = float(physical_roots[int(order[0])])
            continue
        # No physical root: keep transformation behavior via nearest real root
        # to the physical interval, instead of forcing boundary clipping.
        dist = _distance_to_unit_interval(real_roots)
        order = np.lexsort((np.abs(real_roots), np.abs(real_roots - 0.5), dist))
        out[idx] = float(real_roots[int(order[0])])
    return pd.Series(out, index=y_values.index)


def _load_eff_fit_lines(summary_path: Path) -> tuple[dict[int, list[float]], str, dict]:
    """Load fit coefficients from STEP 1.2 build_summary.json (fit_line_eff_i = coeff list)."""
    if not summary_path.exists():
        return ({}, f"missing:{summary_path}", {})
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ({}, f"invalid_json:{exc}", {})

    out: dict[int, list[float]] = {}
    for plane in (1, 2, 3, 4):
        raw = payload.get(f"fit_line_eff_{plane}")
        if raw is None:
            raw = payload.get(f"fit_poly_eff_{plane}")
        if isinstance(raw, dict):
            if "coefficients" in raw:
                raw = raw.get("coefficients")
            elif "a" in raw and "b" in raw:
                raw = [raw.get("a"), raw.get("b")]
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        coeffs: list[float] = []
        valid = True
        for value in raw:
            c = _safe_float(value, np.nan)
            if not np.isfinite(c):
                valid = False
                break
            coeffs.append(float(c))
        if valid and len(coeffs) >= 2:
            out[plane] = coeffs
    return (out, "ok", payload)


def _load_isotonic_calibration(
    summary_path: Path,
) -> tuple[dict[int, dict], str]:
    """Load isotonic calibration knots from STEP 1.2 build_summary.json."""
    if not summary_path.exists():
        return ({}, f"missing:{summary_path}")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ({}, f"invalid_json:{exc}")

    out: dict[int, dict] = {}
    for plane in (1, 2, 3, 4):
        raw = payload.get(f"isotonic_calibration_eff_{plane}")
        if not isinstance(raw, dict):
            continue
        x_knots = raw.get("x_knots")
        y_knots = raw.get("y_knots")
        if (
            not isinstance(x_knots, list)
            or not isinstance(y_knots, list)
            or len(x_knots) < 2
            or len(x_knots) != len(y_knots)
        ):
            continue
        try:
            xk = np.array(x_knots, dtype=float)
            yk = np.array(y_knots, dtype=float)
        except (TypeError, ValueError):
            continue
        if not (np.all(np.isfinite(xk)) and np.all(np.isfinite(yk))):
            continue
        out[plane] = {
            "x_knots": xk,
            "y_knots": yk,
            "slope_lo": _safe_float(raw.get("slope_lo"), 1.0),
            "slope_hi": _safe_float(raw.get("slope_hi"), 1.0),
            "x_min": _safe_float(raw.get("x_min"), float(xk[0])),
            "x_max": _safe_float(raw.get("x_max"), float(xk[-1])),
        }
    status = "ok" if out else "no_isotonic_data"
    return (out, status)


def _transform_efficiencies_isotonic(
    eff_by_plane: dict[int, pd.Series],
    isotonic_by_plane: dict[int, dict],
) -> tuple[dict[int, pd.Series], dict[int, str]]:
    """Apply isotonic (piecewise-linear monotonic) calibration per plane.

    For values within the calibration support, uses np.interp on the stored knots.
    For values outside the support, uses monotonic asymptotic extrapolation with
    boundary-matched derivatives, which avoids hard boundary pile-up artifacts.
    """
    transformed: dict[int, pd.Series] = {}
    formula: dict[int, str] = {}
    for plane in (1, 2, 3, 4):
        raw = pd.to_numeric(eff_by_plane.get(plane), errors="coerce")
        if plane not in isotonic_by_plane:
            transformed[plane] = pd.Series(np.nan, index=raw.index)
            formula[plane] = "missing_isotonic_calibration"
            continue
        cal = isotonic_by_plane[plane]
        xk = cal["x_knots"]
        yk = cal["y_knots"]
        x_min = cal["x_min"]
        x_max = cal["x_max"]
        slope_lo = cal["slope_lo"]
        slope_hi = cal["slope_hi"]

        x = raw.to_numpy(dtype=float)
        y = np.full_like(x, np.nan, dtype=float)
        finite = np.isfinite(x)

        # Interior: piecewise-linear interpolation
        interior = finite & (x >= x_min) & (x <= x_max)
        if np.any(interior):
            y[interior] = np.interp(x[interior], xk, yk)

        y_lo = float(yk[0])
        y_hi = float(yk[-1])

        # Extrapolation below support: monotonic asymptote toward 0 with
        # derivative matched to boundary slope.
        below = finite & (x < x_min)
        if np.any(below):
            if slope_lo > 0.0 and y_lo > 1e-8:
                k_lo = slope_lo / max(y_lo, 1e-8)
                y[below] = y_lo * np.exp(k_lo * (x[below] - x_min))
            else:
                y[below] = y_lo

        # Extrapolation above support: monotonic asymptote toward 1 with
        # derivative matched to boundary slope.
        above = finite & (x > x_max)
        if np.any(above):
            headroom = max(1.0 - y_hi, 1e-8)
            if slope_hi > 0.0 and headroom > 1e-8:
                k_hi = slope_hi / headroom
                y[above] = 1.0 - headroom * np.exp(-k_hi * (x[above] - x_max))
            else:
                y[above] = y_hi

        # Soft physical bound: clip to [0, 1]
        y = np.clip(y, 0.0, 1.0)

        transformed[plane] = pd.Series(y, index=raw.index)
        n_knots = len(xk)
        formula[plane] = (
            f"isotonic_piecewise_linear({n_knots} knots), "
            f"domain=[{x_min:.6g},{x_max:.6g}], "
            f"range=[{float(yk[0]):.6g},{float(yk[-1]):.6g}], "
            f"extrap_slope=[{slope_lo:.4g},{slope_hi:.4g}], "
            "extrap=asymptotic_monotonic"
        )
    return (transformed, formula)


def _resolve_eff_transform_mode(
    requested_mode: str,
    fit_summary_payload: dict,
) -> tuple[str, str]:
    """Resolve transform mode from config request and STEP 1.2 fit relation metadata."""
    mode = str(requested_mode).strip().lower()
    if mode in {"inverse", "forward"}:
        return (mode, f"config:{mode}")
    if mode != "auto":
        mode = "auto"

    relation_raw = str(fit_summary_payload.get("fit_polynomial_relation", "")).strip()
    relation_norm = re.sub(r"\s+", "", relation_raw.lower())
    x_var = str(fit_summary_payload.get("fit_polynomial_x_variable", "")).strip().lower()
    y_var = str(fit_summary_payload.get("fit_polynomial_y_variable", "")).strip().lower()

    if "simulated=p(empirical)" in relation_norm:
        return ("forward", "summary_relation:simulated=P(empirical)")
    if "empirical=p(simulated)" in relation_norm:
        return ("inverse", "summary_relation:empirical=P(simulated)")
    if "empirical" in x_var and "simulated" in y_var:
        return ("forward", "summary_axes:x=empirical,y=simulated")
    if "simulated" in x_var and "empirical" in y_var:
        return ("inverse", "summary_axes:x=simulated,y=empirical")

    # Legacy STEP 1.2 summaries did not store relation metadata.
    return ("inverse", "fallback_legacy_inverse(no_relation_metadata)")


def _read_fit_order_info(summary_path: Path) -> tuple[int | None, dict[str, int]]:
    if not summary_path.exists():
        return (None, {})
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return (None, {})
    requested_raw = payload.get("fit_polynomial_order_requested")
    requested: int | None = None
    try:
        if requested_raw is not None:
            requested = int(requested_raw)
    except (TypeError, ValueError):
        requested = None

    used_by_plane: dict[str, int] = {}
    raw_used = payload.get("fit_polynomial_order_by_plane", {})
    if isinstance(raw_used, dict):
        for k, v in raw_used.items():
            try:
                used_by_plane[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    return (requested, used_by_plane)


def _transform_efficiencies_with_fits(
    eff_by_plane: dict[int, pd.Series],
    fit_by_plane: dict[int, list[float]],
    *,
    mode: str = "forward",
    input_domain_by_plane: dict[int, tuple[float, float]] | None = None,
    clip_output_to_unit_interval: bool = False,
) -> tuple[dict[int, pd.Series], dict[int, str]]:
    """Apply polynomial fit transform per plane.

    Fits are from STEP 1.2 and can represent either:
    - empirical = P(simulated)  -> use mode='inverse'
    - simulated = P(empirical)  -> use mode='forward'
    """
    transformed: dict[int, pd.Series] = {}
    formula: dict[int, str] = {}
    use_inverse = str(mode).strip().lower() != "forward"
    for plane in (1, 2, 3, 4):
        raw = pd.to_numeric(eff_by_plane.get(plane), errors="coerce")
        raw_eval = raw.copy()
        domain_note = ""
        if input_domain_by_plane is not None and plane in input_domain_by_plane:
            dom_raw = input_domain_by_plane.get(plane)
            if isinstance(dom_raw, (list, tuple)) and len(dom_raw) == 2:
                lo = _safe_float(dom_raw[0], np.nan)
                hi = _safe_float(dom_raw[1], np.nan)
                if np.isfinite(lo) and np.isfinite(hi):
                    if hi < lo:
                        lo, hi = hi, lo
                    raw_eval = raw_eval.clip(lower=float(lo), upper=float(hi))
                    domain_note = f"; x_clipped_to_[{float(lo):.6g},{float(hi):.6g}]"
        if plane not in fit_by_plane:
            transformed[plane] = pd.Series(np.nan, index=raw.index)
            formula[plane] = "missing_fit_polynomial"
            continue
        coeffs = np.asarray(fit_by_plane[plane], dtype=float)
        if coeffs.ndim != 1 or coeffs.size < 2 or not np.isfinite(coeffs).all():
            transformed[plane] = pd.Series(np.nan, index=raw.index)
            formula[plane] = "invalid_fit_polynomial"
            continue
        degree = int(coeffs.size - 1)
        poly_expr = _format_polynomial_expr(coeffs, variable="x", precision=8)
        if use_inverse:
            corr = _invert_polynomial_values(raw_eval, coeffs)
            formula[plane] = (
                f"inverse_root(P(x)=eff_raw_empirical), deg={degree}, P(x)={poly_expr}{domain_note}"
            )
        else:
            corr = pd.Series(np.polyval(coeffs, raw_eval.to_numpy(dtype=float)), index=raw.index)
            formula[plane] = f"P(eff_raw_empirical), deg={degree}, P(x)={poly_expr}{domain_note}"
        corr = corr.where(np.isfinite(corr), np.nan)
        if clip_output_to_unit_interval:
            corr = corr.clip(lower=0.0, upper=1.0)
            formula[plane] = f"{formula[plane]}; y_clipped_to_[0,1]"
        transformed[plane] = corr
    return (transformed, formula)


def _fraction_in_closed_interval(series: pd.Series, lo: float, hi: float) -> float:
    """Fraction of finite values inside [lo, hi]. Returns 0 when no finite values."""
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(s)
    if not np.any(m):
        return 0.0
    vals = s[m]
    return float(np.mean((vals >= float(lo)) & (vals <= float(hi))))


def _boundary_fraction_metrics(
    series: pd.Series,
    *,
    near_tol: float = 0.01,
    atol: float = 1e-10,
) -> dict[str, float]:
    """Compact boundary/saturation diagnostics for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(s)
    if not np.any(m):
        return {
            "near_low_fraction": 0.0,
            "near_high_fraction": 0.0,
            "exact_min_fraction": 0.0,
            "exact_max_fraction": 0.0,
            "n_unique_rounded_1e6": 0.0,
        }
    vals = s[m]
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    return {
        "near_low_fraction": float(np.mean(vals <= float(near_tol))),
        "near_high_fraction": float(np.mean(vals >= 1.0 - float(near_tol))),
        "exact_min_fraction": float(np.mean(np.isclose(vals, vmin, atol=atol))),
        "exact_max_fraction": float(np.mean(np.isclose(vals, vmax, atol=atol))),
        "n_unique_rounded_1e6": float(len(np.unique(np.round(vals, 6)))),
    }


def _build_rate_model(
    *,
    flux: pd.Series,
    eff: pd.Series,
    rate: pd.Series,
) -> dict | None:
    x = pd.to_numeric(flux, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(eff, errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(rate, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(mask.sum()) < 1:
        return None
    x = x[mask]
    y = y[mask]
    z = z[mask]
    tri = None
    interp = None
    try:
        tri = Triangulation(x, y)
        interp = LinearTriInterpolator(tri, z)
    except Exception:
        tri = None
        interp = None
    return {
        "x": x,
        "y": y,
        "z": z,
        "tri": tri,
        "interp": interp,
        "flux_min": float(np.nanmin(x)),
        "flux_max": float(np.nanmax(x)),
        "eff_min": float(np.nanmin(y)),
        "eff_max": float(np.nanmax(y)),
    }


def _predict_rate(
    model: dict,
    flux_values: np.ndarray,
    eff_values: np.ndarray,
) -> np.ndarray:
    xq = np.asarray(flux_values, dtype=float)
    yq = np.asarray(eff_values, dtype=float)
    qx = xq.ravel()
    qy = yq.ravel()
    zq = np.full_like(qx, np.nan, dtype=float)

    interp = model.get("interp")
    if interp is not None:
        zi = interp(qx, qy)
        zq = np.asarray(np.ma.filled(zi, np.nan), dtype=float)

    missing = ~np.isfinite(zq)
    if missing.any():
        x = np.asarray(model["x"], dtype=float)
        y = np.asarray(model["y"], dtype=float)
        z = np.asarray(model["z"], dtype=float)
        qx_m = qx[missing]
        qy_m = qy[missing]
        out = np.empty(len(qx_m), dtype=float)
        chunk = 4096
        for s in range(0, len(qx_m), chunk):
            e = min(len(qx_m), s + chunk)
            dx = qx_m[s:e, None] - x[None, :]
            dy = qy_m[s:e, None] - y[None, :]
            idx = np.argmin(dx * dx + dy * dy, axis=1)
            out[s:e] = z[idx]
        zq[missing] = out
    return zq.reshape(xq.shape)


def _ordered_row_indices(df: pd.DataFrame, valid_mask: pd.Series) -> np.ndarray:
    valid_idx = np.where(valid_mask.to_numpy(dtype=bool))[0]
    if len(valid_idx) == 0:
        return valid_idx
    if "file_timestamp_utc" in df.columns:
        ts = pd.to_datetime(df.loc[valid_mask, "file_timestamp_utc"], errors="coerce", utc=True)
        if ts.notna().any():
            ts_ns = ts.astype("int64").to_numpy(dtype=np.int64, copy=False)
            ts_ns = np.where(ts.notna().to_numpy(), ts_ns, np.iinfo(np.int64).max)
            return valid_idx[np.argsort(ts_ns)]
    if "execution_timestamp_utc" in df.columns:
        ts = pd.to_datetime(df.loc[valid_mask, "execution_timestamp_utc"], errors="coerce", utc=True)
        if ts.notna().any():
            ts_ns = ts.astype("int64").to_numpy(dtype=np.int64, copy=False)
            ts_ns = np.where(ts.notna().to_numpy(), ts_ns, np.iinfo(np.int64).max)
            return valid_idx[np.argsort(ts_ns)]
    return valid_idx


def _pick_estimated_eff_col_for_plane(df: pd.DataFrame, plane: int = 2) -> str | None:
    preferred = f"est_eff_sim_{int(plane)}"
    if preferred in df.columns and pd.to_numeric(df[preferred], errors="coerce").notna().any():
        return preferred
    return _choose_primary_eff_est_col(df)


def _pick_dictionary_eff_col_for_plane(df: pd.DataFrame, plane: int = 2) -> str | None:
    preferred = f"eff_sim_{int(plane)}"
    if preferred in df.columns and pd.to_numeric(df[preferred], errors="coerce").notna().any():
        return preferred
    for c in ("eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"):
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _mask_sim_eff_within_tolerance_band(
    df: pd.DataFrame,
    tolerance_pct: float,
) -> np.ndarray:
    """Rows where eff_sim_1..4 are finite and inside one tolerance band."""
    n_rows = len(df)
    if n_rows == 0:
        return np.zeros(0, dtype=bool)
    eff_cols = [f"eff_sim_{i}" for i in range(1, 5)]
    if not all(col in df.columns for col in eff_cols):
        return np.zeros(n_rows, dtype=bool)

    tol_pct = float(tolerance_pct)
    if not np.isfinite(tol_pct):
        tol_pct = 10.0
    tol_pct = max(0.0, tol_pct)
    tol_abs = tol_pct / 100.0

    eff_mat = np.column_stack(
        [pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in eff_cols]
    )
    finite = np.isfinite(eff_mat).all(axis=1)
    if not np.any(finite):
        return np.zeros(n_rows, dtype=bool)
    span = np.full(n_rows, np.nan, dtype=float)
    eff_finite = eff_mat[finite]
    span[finite] = np.max(eff_finite, axis=1) - np.min(eff_finite, axis=1)
    return finite & (span <= (tol_abs + 1e-12))


def _load_lut(lut_path: Path) -> pd.DataFrame:
    return pd.read_csv(lut_path, comment="#", low_memory=False)


def _lut_param_names(lut_df: pd.DataFrame, lut_meta_path: Path | None = None) -> list[str]:
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
                "Could not parse LUT metadata at %s (%s). Falling back to column scan.",
                lut_meta_path,
                exc,
            )

    params: list[str] = []
    for c in lut_df.columns:
        if not c.startswith("sigma_"):
            continue
        if "_p" in c:
            pname = c[len("sigma_") :].split("_p", 1)[0]
        elif c.endswith("_std"):
            pname = c[len("sigma_") : -len("_std")]
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
            qv = pd.to_numeric(query_df.get("n_events"), errors="coerce").to_numpy(dtype=float)
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

        sigma_median = float(np.nanmedian(sigma_vals[valid_sigma]))
        raw = np.where(np.isfinite(raw), raw, sigma_median)
        out[f"unc_{pname}_pct_raw"] = raw
        out[f"unc_{pname}_pct"] = np.abs(raw)
    return out


def _choose_primary_eff_est_col(df: pd.DataFrame) -> str | None:
    for candidate in ("est_eff_sim_1", "est_eff_sim_2", "est_eff_sim_3", "est_eff_sim_4"):
        if candidate in df.columns:
            return candidate
    generic = [c for c in df.columns if c.startswith("est_eff_")]
    return sorted(generic)[0] if generic else None


def _time_axis(df: pd.DataFrame) -> tuple[pd.Series, str, bool]:
    if "file_timestamp_utc" in df.columns:
        ts = _parse_ts(df["file_timestamp_utc"])
        if ts.notna().any():
            return (ts.dt.tz_convert(None), "Data time from filename_base [UTC]", True)
    if "execution_timestamp_utc" in df.columns:
        ts = _parse_ts(df["execution_timestamp_utc"])
        if ts.notna().any():
            return (ts.dt.tz_convert(None), "Execution time [UTC]", True)
    if "execution_timestamp" in df.columns:
        ts = _parse_ts(df["execution_timestamp"])
        if ts.notna().any():
            return (ts.dt.tz_convert(None), "Execution time [UTC]", True)
    return (pd.Series(np.arange(len(df), dtype=float), index=df.index), "Row index", False)


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.7))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_series_with_uncertainty(
    *,
    x: pd.Series,
    has_time_axis: bool,
    y: pd.Series,
    y_unc: pd.Series | None,
    title: str,
    ylabel: str,
    xlabel: str,
    out_path: Path,
) -> None:
    yv = pd.to_numeric(y, errors="coerce")
    if yv.notna().sum() == 0:
        _plot_placeholder(out_path, title, f"No finite values found for '{ylabel}'.")
        return

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    x_values = x.to_numpy()
    y_values = yv.to_numpy(dtype=float)

    ax.plot(x_values, y_values, color="#1F77B4", linewidth=1.2, alpha=0.9, marker="o", markersize=2.4)

    if y_unc is not None:
        uv = pd.to_numeric(y_unc, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y_values) & np.isfinite(uv)
        if finite.any():
            lower = y_values[finite] - np.abs(uv[finite])
            upper = y_values[finite] + np.abs(uv[finite])
            ax.fill_between(
                x_values[finite],
                lower,
                upper,
                color="#1F77B4",
                alpha=0.16,
                linewidth=0.0,
                label="Estimate +/- uncertainty",
            )
            ax.legend(loc="best", frameon=True, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    if not has_time_axis:
        ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)) if len(x_values) > 1 else 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pre_estimation_efficiencies(
    *,
    x: pd.Series,
    has_time_axis: bool,
    xlabel: str,
    global_rate: pd.Series | None,
    global_rate_label: str,
    raw_eff_by_plane: dict[int, pd.Series],
    raw_eff_source_prefix: str | None,
    transformed_eff_by_plane: dict[int, pd.Series],
    transform_mode: str,
    out_path: Path,
) -> None:
    """Four-panel diagnostic:
    top: global-rate only,
    middle: raw eff(1..4) from 1 - three/four,
    third: transformed efficiencies using STEP 1.2 fit polynomials,
    bottom: same transformed efficiencies zoomed to y in [0.75, 1.0].
    """
    raw_valid = any(pd.to_numeric(raw_eff_by_plane.get(p), errors="coerce").notna().any() for p in (1, 2, 3, 4))
    tr_valid = any(
        pd.to_numeric(transformed_eff_by_plane.get(p), errors="coerce").notna().any() for p in (1, 2, 3, 4)
    )
    gr_valid = False
    gr = pd.Series(np.nan, index=x.index, dtype=float)
    if global_rate is not None:
        gr = pd.to_numeric(global_rate, errors="coerce")
        gr_valid = gr.notna().any()

    if not (raw_valid or tr_valid or gr_valid):
        _plot_placeholder(
            out_path,
            "Pre-estimation efficiency diagnostics",
            "No finite global-rate or source/transformed efficiency values available.",
        )
        return

    source_prefix_label = (
        f"{str(raw_eff_source_prefix)}_tt"
        if raw_eff_source_prefix not in (None, "", "None")
        else "selected_tt"
    )

    def _eff_ylim(vmin: float, vmax: float) -> tuple[float, float]:
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return (-0.02, 1.02)
        lo = min(-0.02, float(vmin))
        hi = max(1.02, float(vmax))
        if hi <= lo:
            hi = lo + 0.05
        pad = 0.03 * (hi - lo)
        return (lo - pad, hi + pad)

    xv = x.to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(12.2, 14.0), sharex=True)

    # Top panel: global rate only
    ax_rate = axes[0]
    color_by_plane = {1: "#1F77B4", 2: "#FF7F0E", 3: "#2CA02C", 4: "#9467BD"}
    if gr_valid:
        ax_rate.scatter(
            xv,
            gr.to_numpy(dtype=float),
            color="#111111",
            s=9,
            alpha=0.85,
            label=f"{global_rate_label} [Hz]",
        )
        ax_rate.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax_rate.text(0.5, 0.5, "No finite global-rate values", ha="center", va="center")
    ax_rate.set_ylabel("Global rate [Hz]")
    ax_rate.set_title("Before dictionary estimation: global_rate_hz")
    ax_rate.grid(True, alpha=0.22)

    # Middle panel: efficiencies from selected TT prefix
    ax_raw = axes[1]
    n_raw = 0
    raw_y_min = np.inf
    raw_y_max = -np.inf
    for plane in (1, 2, 3, 4):
        eff = pd.to_numeric(raw_eff_by_plane.get(plane), errors="coerce")
        valid = eff.notna()
        if valid.any():
            yvals = eff[valid].to_numpy(dtype=float)
            if yvals.size:
                raw_y_min = min(raw_y_min, float(np.nanmin(yvals)))
                raw_y_max = max(raw_y_max, float(np.nanmax(yvals)))
            ax_raw.scatter(
                xv[valid.to_numpy()],
                yvals,
                s=10,
                alpha=0.88,
                color=color_by_plane[plane],
                label=f"eff_{plane} ({source_prefix_label}) = 1 - three/four",
            )
            n_raw += 1
    if n_raw == 0:
        ax_raw.text(0.5, 0.5, "No raw efficiency values available", ha="center", va="center")
    else:
        ax_raw.legend(loc="best", fontsize=8, frameon=True, ncol=2)
    ax_raw.set_ylim(*_eff_ylim(raw_y_min, raw_y_max))
    ax_raw.set_ylabel("Source efficiencies")
    ax_raw.set_title(f"Efficiencies from {source_prefix_label} rates (1 - threeplane/fourplane)")
    ax_raw.grid(True, alpha=0.22)

    # Third and bottom panels: transformed efficiencies (full + zoomed).
    ax_bot = axes[2]
    ax_bot_zoom = axes[3]
    n_drawn = 0
    tr_y_min = np.inf
    tr_y_max = -np.inf
    for plane in (1, 2, 3, 4):
        eff_t = pd.to_numeric(transformed_eff_by_plane.get(plane), errors="coerce")
        valid = eff_t.notna()
        if valid.any():
            yvals_t = eff_t[valid].to_numpy(dtype=float)
            xvals_t = xv[valid.to_numpy()]
            if yvals_t.size:
                tr_y_min = min(tr_y_min, float(np.nanmin(yvals_t)))
                tr_y_max = max(tr_y_max, float(np.nanmax(yvals_t)))
            label = f"transformed_eff_{plane}"
            ax_bot.scatter(
                xvals_t,
                yvals_t,
                s=10,
                alpha=0.9,
                color=color_by_plane[plane],
                label=label,
            )
            ax_bot_zoom.scatter(
                xvals_t,
                yvals_t,
                s=10,
                alpha=0.9,
                color=color_by_plane[plane],
                label=label,
            )
            n_drawn += 1
    if n_drawn == 0:
        ax_bot.text(0.5, 0.5, "No transformed efficiency values available", ha="center", va="center")
        ax_bot_zoom.text(0.5, 0.5, "No transformed efficiency values available", ha="center", va="center")

    ax_bot.set_ylim(*_eff_ylim(tr_y_min, tr_y_max))
    ax_bot.set_ylabel("Transformed efficiencies")
    ax_bot.set_title(
        "Transformed efficiency (using STEP 1.2 fit polynomials; "
        f"mode={transform_mode})"
    )
    ax_bot.grid(True, alpha=0.22)
    if n_drawn > 0:
        ax_bot.legend(loc="best", fontsize=8, frameon=True, ncol=2)

    ax_bot_zoom.set_ylim(0.75, 1.0)
    ax_bot_zoom.set_ylabel("Transf. eff (zoom)")
    ax_bot_zoom.set_xlabel(xlabel)
    ax_bot_zoom.set_title(
        "Transformed efficiency (zoomed y-range: [0.75, 1.0])"
    )
    ax_bot_zoom.grid(True, alpha=0.22)
    if n_drawn > 0:
        ax_bot_zoom.legend(loc="best", fontsize=8, frameon=True, ncol=2)

    if not has_time_axis and len(xv) > 0:
        xmin = float(np.nanmin(xv))
        xmax = float(np.nanmax(xv)) if len(xv) > 1 else xmin + 1.0
        ax_rate.set_xlim(xmin, xmax)
        ax_raw.set_xlim(xmin, xmax)
        ax_bot.set_xlim(xmin, xmax)
        ax_bot_zoom.set_xlim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_flux_recovery_story_real(
    *,
    x: pd.Series,
    has_time_axis: bool,
    xlabel: str,
    distance_series: pd.Series | None,
    distance_label: str,
    eff_series: pd.Series,
    eff_label: str,
    eff_series_map: dict[str, pd.Series] | None,
    global_rate_series: pd.Series,
    global_rate_label: str,
    flux_est_series: pd.Series,
    flux_unc_series: pd.Series | None,
    flux_reference_series: pd.Series | None,
    flux_reference_label: str,
    out_path: Path,
) -> None:
    """STEP 3.3-like story using real-data inferred quantities, prefixed by best distance."""

    def _apply_striped_background(ax: plt.Axes, y_vals: np.ndarray) -> None:
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

    xv = x.to_numpy()
    distance = (
        pd.to_numeric(distance_series, errors="coerce").to_numpy(dtype=float)
        if distance_series is not None
        else np.full(len(xv), np.nan, dtype=float)
    )
    eff_curves: list[tuple[str, np.ndarray]] = []
    if eff_series_map:
        for label, series in eff_series_map.items():
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            if len(vals) == len(xv):
                eff_curves.append((str(label), vals))
    if not eff_curves:
        eff_curves = [(str(eff_label), pd.to_numeric(eff_series, errors="coerce").to_numpy(dtype=float))]

    eff_any_finite = any(np.isfinite(vals).any() for _, vals in eff_curves)
    eff_all_finite = [vals[np.isfinite(vals)] for _, vals in eff_curves if np.isfinite(vals).any()]
    eff_bg = np.concatenate(eff_all_finite) if eff_all_finite else np.array([], dtype=float)
    rate = pd.to_numeric(global_rate_series, errors="coerce").to_numpy(dtype=float)
    flux_est = pd.to_numeric(flux_est_series, errors="coerce").to_numpy(dtype=float)
    flux_unc = (
        pd.to_numeric(flux_unc_series, errors="coerce").to_numpy(dtype=float)
        if flux_unc_series is not None
        else None
    )
    flux_ref = (
        pd.to_numeric(flux_reference_series, errors="coerce").to_numpy(dtype=float)
        if flux_reference_series is not None
        else None
    )

    valid_any = (
        np.isfinite(distance).any()
        or eff_any_finite
        or np.isfinite(rate).any()
        or np.isfinite(flux_est).any()
        or (flux_ref is not None and np.isfinite(flux_ref).any())
    )
    if not valid_any:
        _plot_placeholder(
            out_path,
            "Flux-recovery style real-data story",
            "No finite series available for best_distance / estimated efficiency / global rate / estimated flux.",
        )
        return

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.6, 9.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 1.15]},
    )
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, alpha=0.24)

    # 1) Best dictionary distance.
    m_dist = np.isfinite(xv) & np.isfinite(distance)
    if np.any(m_dist):
        order = np.argsort(xv[m_dist])
        xd = xv[m_dist][order]
        yd = distance[m_dist][order]
        axes[0].plot(
            xd,
            yd,
            color="#9467BD",
            linewidth=1.15,
            marker="o",
            markersize=2.8,
            markerfacecolor="#9467BD",
            markeredgewidth=0.0,
            alpha=0.88,
            label=f"Best dictionary distance ({distance_label})",
        )
    else:
        axes[0].text(0.5, 0.5, f"No finite values for {distance_label}", ha="center", va="center")
    axes[0].set_ylabel("Best distance")
    _apply_striped_background(axes[0], distance)
    _legend_if_labeled(axes[0], loc="best", fontsize=8)

    # 2) Estimated efficiencies (all available planes).
    eff_colors = ["#FF7F0E", "#1F77B4", "#2CA02C", "#9467BD", "#8C564B", "#17BECF"]
    n_eff_drawn = 0
    for i, (label, eff_vals) in enumerate(eff_curves):
        m_eff = np.isfinite(xv) & np.isfinite(eff_vals)
        if not np.any(m_eff):
            continue
        order = np.argsort(xv[m_eff])
        xe = xv[m_eff][order]
        ye = eff_vals[m_eff][order]
        axes[1].plot(
            xe,
            ye,
            color=eff_colors[i % len(eff_colors)],
            linewidth=1.05,
            marker="o",
            markersize=2.7,
            markerfacecolor="white",
            markeredgewidth=0.60,
            alpha=0.90,
            label=f"Estimated efficiency ({label})",
        )
        n_eff_drawn += 1
    if n_eff_drawn == 0:
        axes[1].text(0.5, 0.5, "No finite estimated efficiency values", ha="center", va="center")
    axes[1].set_ylabel("Estimated eff")
    _apply_striped_background(axes[1], eff_bg)
    _legend_if_labeled(axes[1], loc="best", fontsize=8, ncol=2)

    # 3) Global rate.
    m_rate = np.isfinite(xv) & np.isfinite(rate)
    if np.any(m_rate):
        order = np.argsort(xv[m_rate])
        xr = xv[m_rate][order]
        yr = rate[m_rate][order]
        axes[2].plot(
            xr,
            yr,
            color="#2E8B57",
            linewidth=2.4,
            alpha=0.46,
            solid_capstyle="round",
            label=f"Global rate ({global_rate_label})",
        )
    else:
        axes[2].text(0.5, 0.5, f"No finite values for {global_rate_label}", ha="center", va="center")
    axes[2].set_ylabel("Global rate")
    _apply_striped_background(axes[2], rate)
    _legend_if_labeled(axes[2], loc="best", fontsize=8)

    # 4) Estimated flux (+ uncertainty), with optional real-data-derived reference.
    m_flux = np.isfinite(xv) & np.isfinite(flux_est)
    if np.any(m_flux):
        order = np.argsort(xv[m_flux])
        xf = xv[m_flux][order]
        yf = flux_est[m_flux][order]
        axes[3].plot(
            xf,
            yf,
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
        if flux_unc is not None and len(flux_unc) == len(xv):
            uf = np.abs(np.asarray(flux_unc, dtype=float)[m_flux][order])
            valid_uf = np.isfinite(uf)
            if np.any(valid_uf):
                axes[3].fill_between(
                    xf[valid_uf],
                    yf[valid_uf] - uf[valid_uf],
                    yf[valid_uf] + uf[valid_uf],
                    color="#D62728",
                    alpha=0.16,
                    linewidth=0.0,
                    label="Estimated ± uncertainty",
                    zorder=2,
                )
    else:
        axes[3].text(0.5, 0.5, "No finite estimated flux values", ha="center", va="center")

    if flux_ref is not None and len(flux_ref) == len(xv):
        m_ref = np.isfinite(xv) & np.isfinite(flux_ref)
        if np.any(m_ref):
            order = np.argsort(xv[m_ref])
            xr = xv[m_ref][order]
            yr = flux_ref[m_ref][order]
            axes[3].plot(
                xr,
                yr,
                color="#1F77B4",
                linewidth=1.0,
                linestyle="--",
                alpha=0.62,
                label=flux_reference_label,
                zorder=1,
            )

    axes[3].set_ylabel("Estimated flux")
    axes[3].set_xlabel(xlabel)
    _apply_striped_background(
        axes[3],
        flux_est if np.isfinite(flux_est).any() else (flux_ref if flux_ref is not None else np.array([])),
    )
    _legend_if_labeled(axes[3], loc="best", fontsize=8)

    if not has_time_axis and len(xv) > 0:
        xmin = float(np.nanmin(xv))
        xmax = float(np.nanmax(xv)) if len(xv) > 1 else xmin + 1.0
        for ax in axes:
            ax.set_xlim(xmin, xmax)

    fig.suptitle(
        "Real-data story: best distance -> estimated efficiency -> global-rate response -> reconstructed flux",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _legend_if_labeled(ax: plt.Axes, **legend_kwargs: object) -> None:
    handles, labels = ax.get_legend_handles_labels()
    has_labeled = any(str(lbl).strip() and not str(lbl).startswith("_") for lbl in labels)
    if has_labeled and len(handles) > 0:
        ax.legend(**legend_kwargs)


def _plot_eff2_vs_global_rate(
    *,
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    real_eff2_col: str,
    real_rate_col: str,
    dict_eff2_col: str,
    dict_rate_col: str,
    out_path: Path,
) -> tuple[int, int]:
    """Plot real-data eff2 trajectory directly in the (global_rate, eff2) plane."""
    real_eff = pd.to_numeric(real_df[real_eff2_col], errors="coerce")
    real_rate = pd.to_numeric(real_df[real_rate_col], errors="coerce")
    dict_eff = pd.to_numeric(dict_df[dict_eff2_col], errors="coerce")
    dict_rate = pd.to_numeric(dict_df[dict_rate_col], errors="coerce")

    dict_valid = dict_eff.notna() & dict_rate.notna()
    real_valid = real_eff.notna() & real_rate.notna()
    n_dict = int(dict_valid.sum())
    n_real = int(real_valid.sum())

    if n_real == 0:
        _plot_placeholder(
            out_path,
            "Eff2 vs global rate",
            "No finite real points available for eff2/global-rate trajectory.",
        )
        return (n_real, n_dict)

    x_real = real_rate[real_valid].to_numpy(dtype=float)
    y_real = real_eff[real_valid].to_numpy(dtype=float)
    x_dict = dict_rate[dict_valid].to_numpy(dtype=float)
    y_dict = dict_eff[dict_valid].to_numpy(dtype=float)

    x_all = np.concatenate([x_real, x_dict]) if x_dict.size else x_real
    y_all = np.concatenate([y_real, y_dict]) if y_dict.size else y_real
    x_lo = float(np.nanmin(x_all))
    x_hi = float(np.nanmax(x_all))
    y_lo = float(np.nanmin(y_all))
    y_hi = float(np.nanmax(y_all))
    x_span = max(x_hi - x_lo, 1e-9)
    y_span = max(y_hi - y_lo, 1e-9)
    x_pad = max(0.03 * x_span, 1e-6)
    y_pad = max(0.03 * y_span, 1e-6)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad

    fig, ax = plt.subplots(figsize=(9.2, 7.1))
    if n_dict > 0:
        ax.scatter(
            x_dict,
            y_dict,
            s=11,
            alpha=0.20,
            color="#606060",
            edgecolors="none",
            zorder=1,
            label="Dictionary points",
        )

    ordered_idx = _ordered_row_indices(real_df, real_valid)
    x_ord = real_df.iloc[ordered_idx][real_rate_col].to_numpy(dtype=float)
    y_ord = real_df.iloc[ordered_idx][real_eff2_col].to_numpy(dtype=float)
    finite_ord = np.isfinite(x_ord) & np.isfinite(y_ord)
    x_ord = x_ord[finite_ord]
    y_ord = y_ord[finite_ord]

    ax.plot(
        x_ord,
        y_ord,
        linewidth=1.8,
        color="#1F77B4",
        alpha=0.92,
        zorder=3,
        label="Real-data trajectory",
    )
    ax.scatter(
        x_ord,
        y_ord,
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Real-data points",
    )

    ax.scatter([x_ord[0]], [y_ord[0]], color="#2CA02C", marker="o", s=82, edgecolor="black", linewidth=0.8, zorder=5, label="Start")
    ax.scatter([x_ord[-1]], [y_ord[-1]], color="#D62728", marker="X", s=95, edgecolor="black", linewidth=0.8, zorder=5, label="End")

    if len(x_ord) >= 3:
        i = min(len(x_ord) - 2, max(0, int(0.85 * len(x_ord))))
        ax.annotate(
            "",
            xy=(x_ord[i + 1], y_ord[i + 1]),
            xytext=(x_ord[i], y_ord[i]),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            zorder=5,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Global rate [Hz]")
    ax.set_ylabel("Efficiency (eff2)")
    ax.set_title(
        "Real-data eff2 trajectory vs global rate\n"
        f"real_y={real_eff2_col} | dict_y={dict_eff2_col}"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return (n_real, n_dict)


def _plot_estimated_curve_flux_vs_eff(
    *,
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    est_flux_col: str,
    est_eff_col: str,
    dict_eff_col: str,
    dict_rate_col: str,
    out_path: Path,
) -> tuple[int, int]:
    """Plot estimated (flux, eff) trajectory over dictionary global-rate contours."""
    flux_col = "flux_cm2_min"
    model = _build_rate_model(
        flux=dict_df[flux_col],
        eff=dict_df[dict_eff_col],
        rate=dict_df[dict_rate_col],
    )
    if model is None:
        _plot_placeholder(
            out_path,
            "Estimated curve in flux-eff plane",
            "Not enough finite dictionary points to build global-rate contours.",
        )
        return (0, 0)

    dict_flux = pd.to_numeric(dict_df[flux_col], errors="coerce")
    dict_eff = pd.to_numeric(dict_df[dict_eff_col], errors="coerce")
    dict_rate = pd.to_numeric(dict_df[dict_rate_col], errors="coerce")
    dict_valid = dict_flux.notna() & dict_eff.notna() & dict_rate.notna()
    n_dict = int(dict_valid.sum())

    real_flux = pd.to_numeric(real_df[est_flux_col], errors="coerce")
    real_eff = pd.to_numeric(real_df[est_eff_col], errors="coerce")
    n_real_flux = int(real_flux.notna().sum())
    n_real_eff = int(real_eff.notna().sum())
    real_valid = real_flux.notna() & real_eff.notna()
    n_real = int(real_valid.sum())
    if n_real == 0:
        log.warning(
            "STEP_4.2.6 no finite estimated curve points: %s finite=%d/%d, %s finite=%d/%d.",
            est_flux_col,
            n_real_flux,
            len(real_df),
            est_eff_col,
            n_real_eff,
            len(real_df),
        )
        _plot_placeholder(
            out_path,
            "Estimated curve in flux-eff plane",
            "No finite estimated (flux, eff) points available.\n"
            f"{est_flux_col}: {n_real_flux}/{len(real_df)} finite, "
            f"{est_eff_col}: {n_real_eff}/{len(real_df)} finite.",
        )
        return (n_real, n_dict)

    x_ref = np.asarray(model["x"], dtype=float)
    y_ref = np.asarray(model["y"], dtype=float)
    x_real = real_flux[real_valid].to_numpy(dtype=float)
    y_real = real_eff[real_valid].to_numpy(dtype=float)
    x_all = np.concatenate([x_ref[np.isfinite(x_ref)], x_real[np.isfinite(x_real)]])
    y_all = np.concatenate([y_ref[np.isfinite(y_ref)], y_real[np.isfinite(y_real)]])
    x_lo = float(np.nanmin(x_all))
    x_hi = float(np.nanmax(x_all))
    y_lo = float(np.nanmin(y_all))
    y_hi = float(np.nanmax(y_all))
    x_span = max(x_hi - x_lo, 1e-9)
    y_span = max(y_hi - y_lo, 1e-9)
    x_lo -= 0.03 * x_span
    x_hi += 0.03 * x_span
    y_lo -= 0.03 * y_span
    y_hi += 0.03 * y_span

    xi = np.linspace(x_lo, x_hi, 230, dtype=float)
    yi = np.linspace(y_lo, y_hi, 230, dtype=float)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _predict_rate(model, Xi, Yi)
    finite_z = Zi[np.isfinite(Zi)]

    fig, ax = plt.subplots(figsize=(9.2, 7.1))
    z_min = float(np.nanmin(finite_z)) if finite_z.size else np.nan
    z_max = float(np.nanmax(finite_z)) if finite_z.size else np.nan
    if finite_z.size >= 10 and np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        levels = np.linspace(z_min, z_max, 16)
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="viridis", alpha=0.35, zorder=0)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")
        ax.contour(Xi, Yi, Zi, levels=levels[::2], colors="k", linewidths=0.35, alpha=0.28, zorder=1)
    else:
        sc_fallback = ax.scatter(
            dict_flux[dict_valid],
            dict_eff[dict_valid],
            c=dict_rate[dict_valid],
            cmap="viridis",
            s=12,
            alpha=0.5,
            edgecolors="none",
            zorder=0,
        )
        cbar = fig.colorbar(sc_fallback, ax=ax, pad=0.02, fraction=0.048)
        cbar.set_label("Global rate [Hz]")

    # Reference dictionary points.
    ax.scatter(
        dict_flux[dict_valid],
        dict_eff[dict_valid],
        s=10,
        alpha=0.18,
        color="#606060",
        zorder=1,
        label="Dictionary points",
    )

    order_idx = _ordered_row_indices(real_df, real_valid)
    x_ord = real_df.iloc[order_idx][est_flux_col].to_numpy(dtype=float)
    y_ord = real_df.iloc[order_idx][est_eff_col].to_numpy(dtype=float)

    ax.plot(
        x_ord,
        y_ord,
        linewidth=1.8,
        color="#1F77B4",
        alpha=0.92,
        zorder=3,
        label="Estimated trajectory",
    )
    ax.scatter(
        x_ord,
        y_ord,
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Estimated points",
    )

    ax.scatter([x_ord[0]], [y_ord[0]], color="#2CA02C", marker="o", s=82, edgecolor="black", linewidth=0.8, zorder=5)
    ax.scatter([x_ord[-1]], [y_ord[-1]], color="#D62728", marker="X", s=95, edgecolor="black", linewidth=0.8, zorder=5)

    if len(x_ord) >= 3:
        i = min(len(x_ord) - 2, max(0, int(0.85 * len(x_ord))))
        ax.annotate(
            "",
            xy=(x_ord[i + 1], y_ord[i + 1]),
            xytext=(x_ord[i], y_ord[i]),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            zorder=5,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Flux [cm^-2 min^-1]")
    ax.set_ylabel(f"Estimated efficiency ({est_eff_col})")
    ax.set_title("Estimated real-data curve in flux-eff plane with global-rate contours")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return (n_real, n_dict)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4.2: Infer real-data parameters and attach uncertainty LUT."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--real-csv", default=None)
    parser.add_argument("--dictionary-csv", default=None)
    parser.add_argument("--lut-csv", default=None)
    parser.add_argument("--lut-meta-json", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_21_raw = config.get("step_2_1", {})
    if not isinstance(cfg_21_raw, dict):
        cfg_21_raw = {}
    cfg_21, feature_cfg_sources = _merge_common_feature_space_cfg(config, cfg_21_raw)
    cfg_41 = config.get("step_4_1", {})
    cfg_42 = config.get("step_4_2", {})
    _clear_plots_dir()

    real_csv_cfg = cfg_42.get("real_collected_csv", None)
    dictionary_csv_cfg = cfg_42.get("dictionary_csv", None)
    lut_csv_cfg = cfg_42.get("uncertainty_lut_csv", None)
    lut_meta_cfg = cfg_42.get("uncertainty_lut_meta_json", None)
    build_summary_cfg = cfg_42.get("build_summary_json", None)

    real_path = _resolve_input_path(args.real_csv or real_csv_cfg or DEFAULT_REAL_COLLECTED)
    dict_path = _resolve_input_path(args.dictionary_csv or dictionary_csv_cfg or DEFAULT_DICTIONARY)
    lut_path = _resolve_input_path(args.lut_csv or lut_csv_cfg or DEFAULT_LUT)
    lut_meta_path = _resolve_input_path(args.lut_meta_json or lut_meta_cfg or DEFAULT_LUT_META)
    build_summary_path = _resolve_input_path(build_summary_cfg or DEFAULT_BUILD_SUMMARY)

    for label, path in (
        ("Real collected CSV", real_path),
        ("Dictionary CSV", dict_path),
        ("Uncertainty LUT CSV", lut_path),
    ):
        if not path.exists():
            log.error("%s not found: %s", label, path)
            return 1

    # STEP 4.2 always inherits matching criteria from STEP 2.1 to avoid duplicate/overriding knobs.
    ignored_step42_criteria_keys = [
        "feature_columns",
        "distance_metric",
        "interpolation_k",
        "inverse_mapping",
        "histogram_distance_weight",
        "histogram_distance_blend_mode",
        "include_global_rate",
        "global_rate_col",
    ]
    for key in ignored_step42_criteria_keys:
        if cfg_42.get(key, None) not in (None, "", "null", "None"):
            log.info("Ignoring step_4_2.%s; using step_2_1.%s instead.", key, key)

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
    exclude_same_file = _safe_bool(cfg_42.get("exclude_same_file", False), False)
    uncertainty_quantile = _safe_float(cfg_42.get("uncertainty_quantile", 0.68), 0.68)
    uncertainty_quantile = float(np.clip(uncertainty_quantile, 0.0, 1.0))
    contour_eff_band_tolerance_pct = max(
        0.0,
        _safe_float(
            cfg_42.get(
                "iso_rate_efficiency_band_tolerance_pct",
                config.get("iso_rate_efficiency_band_tolerance_pct", 10.0),
            ),
            10.0,
        ),
    )
    if feature_cfg_sources.get("has_common_feature_space"):
        log.info(
            "Feature config source: common_feature_space keys=%s; step_2_1 overrides=%s",
            feature_cfg_sources.get("common_keys", []),
            feature_cfg_sources.get("step_2_1_override_keys", []),
        )
    inherit_step_2_1_method = _safe_bool(
        cfg_42.get("inherit_step_2_1_method", True),
        True,
    )
    n_events_column_cfg = cfg_42.get("n_events_column", "auto")
    derived_feature_subset_mode_requested = str(
        cfg_42.get("derived_feature_subset", "full")
    ).strip().lower()
    if derived_feature_subset_mode_requested not in {
        "full",
        "tt_only",
        "tt_plus_global",
        "tt_plus_global_log",
        "tt_plus_global_eff",
    }:
        log.warning(
            "Invalid step_4_2.derived_feature_subset=%r; using 'full'.",
            derived_feature_subset_mode_requested,
        )
        derived_feature_subset_mode_requested = "full"
    derived_feature_subset_mode = derived_feature_subset_mode_requested
    derived_tt_only_neighbor_count = _safe_int(
        cfg_42.get("derived_tt_only_neighbor_count", 120),
        120,
        minimum=1,
    )
    derived_tt_only_neighbor_selection = str(
        cfg_42.get("derived_tt_only_neighbor_selection", "knn")
    ).strip().lower()
    if derived_tt_only_neighbor_selection not in {"nearest", "knn", "all"}:
        log.warning(
            "Invalid step_4_2.derived_tt_only_neighbor_selection=%r; using 'knn'.",
            derived_tt_only_neighbor_selection,
        )
        derived_tt_only_neighbor_selection = "knn"
    derived_tt_only_weighting = str(
        cfg_42.get("derived_tt_only_weighting", "uniform")
    ).strip().lower()
    if derived_tt_only_weighting not in {"uniform", "inverse_distance", "softmax"}:
        log.warning(
            "Invalid step_4_2.derived_tt_only_weighting=%r; using 'uniform'.",
            derived_tt_only_weighting,
        )
        derived_tt_only_weighting = "uniform"
    derived_tt_only_aggregation = str(
        cfg_42.get("derived_tt_only_aggregation", "local_linear")
    ).strip().lower()
    if derived_tt_only_aggregation not in {"weighted_mean", "weighted_median", "local_linear"}:
        log.warning(
            "Invalid step_4_2.derived_tt_only_aggregation=%r; using 'local_linear'.",
            derived_tt_only_aggregation,
        )
        derived_tt_only_aggregation = "local_linear"
    efficiency_source_prefix_mode = str(
        cfg_42.get("efficiency_source_prefix_mode", "feature_consistent")
    ).strip().lower()
    if efficiency_source_prefix_mode not in {"feature_consistent", "task_chain"}:
        log.warning(
            "Invalid step_4_2.efficiency_source_prefix_mode=%r; using 'feature_consistent'.",
            efficiency_source_prefix_mode,
        )
        efficiency_source_prefix_mode = "feature_consistent"
    task_ids_cfg = cfg_41.get("task_ids", config.get("task_ids", [1]))
    selected_task_ids = _safe_task_ids(task_ids_cfg)
    preferred_tt_prefixes = _preferred_tt_prefixes_for_task_ids(selected_task_ids)
    preferred_feature_prefixes = _preferred_feature_prefixes_for_task_ids(selected_task_ids)
    eff_transform_mode_requested = str(cfg_42.get("eff_transform_mode", "auto")).strip().lower()
    if eff_transform_mode_requested not in {"auto", "inverse", "forward"}:
        eff_transform_mode_requested = "auto"
    eff_transform_clip_input_to_dictionary_domain = _safe_bool(
        cfg_42.get("eff_transform_clip_input_to_dictionary_domain", True),
        True,
    )
    eff_transform_clip_output_to_unit_interval = _safe_bool(
        cfg_42.get("eff_transform_clip_output_to_unit_interval", True),
        True,
    )

    log.info("Real collected: %s", real_path)
    log.info("Dictionary:     %s", dict_path)
    log.info("LUT:            %s", lut_path)
    log.info("Fit summary:    %s", build_summary_path)
    log.info("Task IDs used for efficiency source: %s", selected_task_ids)
    log.info("Preferred TT prefix order for efficiencies: %s", preferred_tt_prefixes)
    log.info("Preferred TT prefix order for auto features: %s", preferred_feature_prefixes)
    log.info(
        "Method inheritance from STEP 2.1: %s",
        "enabled" if inherit_step_2_1_method else "disabled (legacy STEP 4.2 overrides allowed)",
    )
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

    real_df = pd.read_csv(real_path, low_memory=False)
    dict_df = pd.read_csv(dict_path, low_memory=False)
    if real_df.empty:
        log.error("Real collected table is empty: %s", real_path)
        return 1
    if dict_df.empty:
        log.error("Dictionary table is empty: %s", dict_path)
        return 1

    # Prepare per-plane empirical efficiencies before feature resolution so
    # derived mode can use efficiency coordinates (same method domain as STEP 2.1).
    # Use the same prefix preference as feature matching (post/fit/...) to keep
    # rate-space and efficiency-space internally consistent.
    feature_eff_source_prefix: dict[str, str | None] = {}
    feature_eff_source_formula: dict[str, dict[str, str]] = {}
    feature_eff_finite_count: dict[str, dict[str, int]] = {}
    for label, source_frame in (("real", real_df), ("dictionary", dict_df)):
        frame = source_frame.copy()
        frame.attrs["preferred_tt_prefixes"] = preferred_feature_prefixes
        eff_feature_by_plane, eff_feature_formula_by_plane, _, source_prefix = _compute_empirical_efficiencies_from_rates(frame)
        feature_eff_source_prefix[label] = source_prefix
        feature_eff_source_formula[label] = {
            f"eff{plane}": str(eff_feature_formula_by_plane.get(plane, "missing_rate_columns"))
            for plane in (1, 2, 3, 4)
        }
        counts_by_plane: dict[str, int] = {}
        injected_cols: dict[str, pd.Series] = {}
        for plane in (1, 2, 3, 4):
            col = f"eff_empirical_{plane}"
            derived_vals = pd.to_numeric(eff_feature_by_plane.get(plane), errors="coerce")
            if col in frame.columns:
                existing = pd.to_numeric(frame[col], errors="coerce")
                injected_cols[col] = existing.where(existing.notna(), derived_vals)
            else:
                injected_cols[col] = derived_vals
            counts_by_plane[f"eff{plane}"] = int(pd.to_numeric(injected_cols[col], errors="coerce").notna().sum())
        if injected_cols:
            frame[list(injected_cols.keys())] = pd.DataFrame(injected_cols, index=frame.index)
        feature_eff_finite_count[label] = counts_by_plane
        if label == "real":
            real_df = frame
        else:
            dict_df = frame
    log.info(
        "Feature empirical efficiencies injected: real_prefix=%s dict_prefix=%s (finite counts real=%s dict=%s)",
        feature_eff_source_prefix.get("real"),
        feature_eff_source_prefix.get("dictionary"),
        feature_eff_finite_count.get("real"),
        feature_eff_finite_count.get("dictionary"),
    )

    feature_mode = str(feature_columns_cfg).strip().lower() if isinstance(feature_columns_cfg, str) else ""
    selected_feature_modes = {
        "step12_selected",
        "step_1_2_selected",
        "selected_from_step12",
        "selected_from_step_1_2",
    }
    auto_resolution_error: str | None = None
    if feature_mode in {"derived", *selected_feature_modes}:
        dict_auto, real_auto = dict_df, real_df
        auto_feature_columns = []
        auto_feature_strategy = "auto_skipped_for_derived"
        auto_feature_mapping = []
    else:
        try:
            dict_auto, real_auto, auto_feature_columns, auto_feature_strategy, auto_feature_mapping = _resolve_feature_columns_auto(
                dict_df=dict_df,
                real_df=real_df,
                include_global_rate=include_global_rate,
                global_rate_col=global_rate_col,
                preferred_prefixes=preferred_feature_prefixes,
            )
        except ValueError as exc:
            auto_resolution_error = str(exc)
            dict_auto, real_auto = dict_df, real_df
            auto_feature_columns = []
            auto_feature_strategy = "auto_unavailable"
            auto_feature_mapping = []
    catalog = sync_feature_column_catalog(
        catalog_path=CONFIG_COLUMNS_PATH,
        dict_df=dict_df,
        default_enabled_columns=auto_feature_columns,
    )
    derived_cfg_raw = cfg_21.get("derived_features", {})
    if not isinstance(derived_cfg_raw, dict):
        derived_cfg_raw = {}
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
    derived_trigger_types_override_requested = parse_explicit_feature_columns(
        cfg_42.get("derived_trigger_types_override", None)
    )
    derived_trigger_types_override_applied = False
    if inherit_step_2_1_method:
        if (
            derived_feature_subset_mode_requested != "full"
            or bool(derived_trigger_types_override_requested)
            or any(
                key in cfg_42
                for key in (
                    "derived_tt_only_neighbor_selection",
                    "derived_tt_only_neighbor_count",
                    "derived_tt_only_weighting",
                    "derived_tt_only_aggregation",
                )
            )
        ):
            log.info(
                "STEP 4.2 inherit_step_2_1_method=true: ignoring step_4_2 method overrides "
                "(derived_feature_subset, derived_trigger_types_override, derived_tt_only_*)."
            )
        derived_feature_subset_mode = "full"
    elif derived_trigger_types_override_requested:
        derived_trigger_types = derived_trigger_types_override_requested
        derived_trigger_types_override_applied = True
        log.info(
            "Applied STEP 4.2 trigger-type override for derived features: %s",
            derived_trigger_types,
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

    def _materialize_derived_feature_space(
        dict_in: pd.DataFrame,
        real_in: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str | None, list[str], list[str]]:
        (
            dict_out,
            real_out,
            derived_rate_col_local,
            derived_rate_sources_local,
        ) = _append_derived_tt_global_rate_column(
            dict_df=dict_in,
            data_df=real_in,
            prefix_selector=derived_tt_prefix,
            trigger_type_allowlist=derived_trigger_types,
            include_to_tt_rate_hz=bool(derived_include_to_tt),
        )
        if (
            derived_rate_col_local is None
            and global_rate_col in dict_in.columns
            and global_rate_col in real_in.columns
        ):
            derived_rate_col_local = str(global_rate_col)
        if derived_rate_col_local is None:
            return dict_out, real_out, None, [], []
        (
            dict_out,
            real_out,
            derived_physics_cols_local,
        ) = _append_derived_physics_feature_columns(
            dict_df=dict_out,
            data_df=real_out,
            rate_column=derived_rate_col_local,
            physics_features=derived_physics_features,
        )
        return (
            dict_out,
            real_out,
            str(derived_rate_col_local),
            [str(c) for c in derived_rate_sources_local],
            [str(c) for c in derived_physics_cols_local],
        )
    use_step12_selected = False
    if feature_mode in selected_feature_modes:
        selected_from_step12, selected_info = _load_step12_selected_feature_columns(step12_selected_path)
        dict_selected, real_selected, _, _, _ = _materialize_derived_feature_space(dict_df, real_df)
        selected = [
            c for c in selected_from_step12
            if c in dict_selected.columns and c in real_selected.columns
        ]
        if selected:
            dict_work, real_work = dict_selected, real_selected
            feature_columns = selected
            feature_strategy = "step12_selected"
            feature_mapping = []
            use_step12_selected = True
            log.info(
                "Using STEP 1.2 selected features (%d) from %s.",
                len(feature_columns),
                step12_selected_path,
            )
        else:
            log.warning(
                "STEP 1.2 selected features unavailable/empty at %s (%s); falling back to derived mode.",
                step12_selected_path,
                selected_info.get("error", "no_selected_columns"),
            )
            feature_mode = "derived"

    if use_step12_selected:
        pass
    elif feature_mode == "auto":
        if auto_resolution_error is not None:
            log.error("%s", auto_resolution_error)
            return 1
        dict_work, real_work = dict_auto, real_auto
        feature_columns = auto_feature_columns
        feature_strategy = auto_feature_strategy
        feature_mapping = auto_feature_mapping
    elif feature_mode == "derived":
        (
            dict_work,
            real_work,
            derived_rate_col,
            derived_rate_sources,
            derived_physics_cols,
        ) = _materialize_derived_feature_space(dict_df, real_df)
        if derived_rate_col is None:
            log.error(
                "No derived feature global-rate source available. "
                "TT trigger-type sum could not be built and fallback global-rate column is missing."
            )
            return 1
        feat_dict = _shared_derived_feature_columns(
            dict_work,
            rate_column=derived_rate_col,
            trigger_type_rate_columns=(
                derived_rate_sources
                if bool(derived_include_trigger_rates)
                else None
            ),
            include_rate_histogram=bool(derived_include_hist),
            physics_feature_columns=derived_physics_cols,
        )
        feat_real = _shared_derived_feature_columns(
            real_work,
            rate_column=derived_rate_col,
            trigger_type_rate_columns=(
                derived_rate_sources
                if bool(derived_include_trigger_rates)
                else None
            ),
            include_rate_histogram=bool(derived_include_hist),
            physics_feature_columns=derived_physics_cols,
        )
        feature_columns = sorted(set(feat_dict) & set(feat_real))
        if derived_feature_subset_mode != "full":
            subset_cols: list[str] = []
            if derived_feature_subset_mode in {"tt_only", "tt_plus_global", "tt_plus_global_log", "tt_plus_global_eff"}:
                subset_cols.extend([c for c in derived_rate_sources if c in feature_columns])
            if derived_feature_subset_mode in {"tt_plus_global", "tt_plus_global_log", "tt_plus_global_eff"}:
                if derived_rate_col in feature_columns:
                    subset_cols.append(derived_rate_col)
            if derived_feature_subset_mode in {"tt_plus_global_log"}:
                if DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL in feature_columns:
                    subset_cols.append(DERIVED_LOG_RATE_OVER_EFF_PRODUCT_COL)
            if derived_feature_subset_mode in {"tt_plus_global_eff"}:
                subset_cols.extend(
                    [
                        c
                        for c in ("eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4")
                        if c in feature_columns
                    ]
                )
            feature_columns = list(dict.fromkeys(subset_cols))
        if not feature_columns:
            log.error(
                "No derived feature columns found in dictionary/real-data intersection."
            )
            return 1
        feature_strategy = "derived"
        feature_mapping = []
        log.info(
            "Derived features: prefix=%s trigger_types=%s include_hist=%s "
            "include_trigger_rates=%s physics=%s rate_feature=%s rate_sources=%s subset=%s",
            derived_tt_prefix,
            derived_trigger_types,
            bool(derived_include_hist),
            bool(derived_include_trigger_rates),
            derived_physics_features,
            derived_rate_col,
            derived_rate_sources,
            derived_feature_subset_mode,
        )
    elif feature_mode in {"config_columns", "catalog", "config_columns_json"}:
        selected, catalog_info = resolve_feature_columns_from_catalog(
            catalog=catalog,
            available_columns=sorted(set(dict_df.columns) & set(real_df.columns)),
        )
        if catalog_info.get("invalid_include_patterns"):
            log.warning(
                "Ignoring invalid include pattern(s) in %s: %s",
                CONFIG_COLUMNS_PATH,
                catalog_info.get("invalid_include_patterns"),
            )
        if catalog_info.get("invalid_exclude_patterns"):
            log.warning(
                "Ignoring invalid exclude pattern(s) in %s: %s",
                CONFIG_COLUMNS_PATH,
                catalog_info.get("invalid_exclude_patterns"),
            )
        if not selected:
            if auto_feature_columns:
                log.warning(
                    "No features selected by config_columns catalog; falling back to auto (%d columns).",
                    len(auto_feature_columns),
                )
                dict_work, real_work = dict_auto, real_auto
                feature_columns = auto_feature_columns
                feature_strategy = "auto_fallback_from_config_columns"
                feature_mapping = auto_feature_mapping
            else:
                log.error("No features selected by config_columns catalog and no auto fallback available.")
                return 1
        else:
            dict_work, real_work = dict_df, real_df
            feature_columns = selected
            feature_strategy = "config_columns"
            feature_mapping = []
    else:
        explicit_features = parse_explicit_feature_columns(feature_columns_cfg)
        feature_columns = [c for c in explicit_features if c in dict_df.columns and c in real_df.columns]
        if not feature_columns:
            log.error("No explicit feature columns found in both dictionary and real data.")
            return 1
        if include_global_rate and global_rate_col in dict_df.columns and global_rate_col in real_df.columns:
            if global_rate_col not in feature_columns:
                feature_columns.append(global_rate_col)
        dict_work, real_work = dict_df, real_df
        feature_strategy = "explicit"
        feature_mapping = []

    if not feature_columns:
        log.error("Feature column set is empty after resolution.")
        return 1
    log.info("Using %d features (%s).", len(feature_columns), feature_strategy)
    if feature_strategy == "step12_selected":
        log.info("STEP 1.2 selected-feature artifact: %s", step12_selected_path)

    inverse_mapping_cfg_runtime = dict(inverse_mapping_cfg)
    if (
        not inherit_step_2_1_method
        and feature_mode == "derived"
        and derived_feature_subset_mode == "tt_only"
    ):
        inverse_mapping_cfg_runtime["neighbor_selection"] = str(derived_tt_only_neighbor_selection)
        if derived_tt_only_neighbor_selection == "knn":
            inverse_mapping_cfg_runtime["neighbor_count"] = int(derived_tt_only_neighbor_count)
        else:
            inverse_mapping_cfg_runtime["neighbor_count"] = None
        inverse_mapping_cfg_runtime["weighting"] = str(derived_tt_only_weighting)
        inverse_mapping_cfg_runtime["aggregation"] = str(derived_tt_only_aggregation)
        k_disp = (
            int(derived_tt_only_neighbor_count)
            if derived_tt_only_neighbor_selection == "knn"
            else "all"
        )
        log.info(
            "Applied STEP 4.2 tt_only runtime override: selection=%s k=%s weighting=%s aggregation=%s.",
            str(derived_tt_only_neighbor_selection),
            str(k_disp),
            str(derived_tt_only_weighting),
            str(derived_tt_only_aggregation),
        )

    est_df = estimate_from_dataframes(
        dict_df=dict_work,
        data_df=real_work,
        feature_columns=feature_columns,
        distance_metric=distance_metric,
        interpolation_k=interpolation_k,
        include_global_rate=False,
        global_rate_col=global_rate_col,
        exclude_same_file=exclude_same_file,
        inverse_mapping_cfg=inverse_mapping_cfg_runtime,
    )
    eff_oos_masking_summary = est_df.attrs.get(
        "efficiency_feature_out_of_support_masking"
    )

    real_with_idx = real_df.copy()
    real_with_idx["dataset_index"] = np.arange(len(real_with_idx), dtype=int)
    merged = pd.merge(est_df, real_with_idx, on="dataset_index", how="left", suffixes=("", "_real"))

    if n_events_column_cfg == "auto":
        n_events_col_used = _pick_n_events_column(merged)
    else:
        n_events_col_used = str(n_events_column_cfg) if str(n_events_column_cfg) in merged.columns else None
    if n_events_col_used is not None:
        merged["n_events"] = pd.to_numeric(merged[n_events_col_used], errors="coerce")
    elif "n_events" not in merged.columns:
        merged["n_events"] = np.nan

    # Real-data efficiencies for diagnostics/plots:
    # by default, keep prefix basis consistent with inference features.
    efficiency_source_prefix_order = (
        preferred_feature_prefixes
        if efficiency_source_prefix_mode == "feature_consistent"
        else preferred_tt_prefixes
    )
    merged.attrs["preferred_tt_prefixes"] = efficiency_source_prefix_order
    raw_eff_by_plane, raw_eff_formula_by_plane, raw_eff_cols_by_plane, raw_eff_selected_prefix = _compute_empirical_efficiencies_from_rates(merged)
    for plane in (1, 2, 3, 4):
        merged[f"eff{plane}_raw_from_data"] = raw_eff_by_plane[plane]
    merged["eff2_from_data"] = merged["eff2_raw_from_data"]
    eff2_formula = raw_eff_formula_by_plane.get(2, "missing_rate_columns")

    # Dictionary-side efficiencies from rates with the same definition.
    dict_df_plot = dict_df.copy()
    dict_df_plot.attrs["preferred_tt_prefixes"] = efficiency_source_prefix_order
    dict_eff_by_plane, _, dict_eff_cols_by_plane, dict_eff_selected_prefix = _compute_empirical_efficiencies_from_rates(dict_df_plot)
    for plane in (1, 2, 3, 4):
        dict_df_plot[f"dict_eff{plane}_raw_from_rates"] = dict_eff_by_plane[plane]
    dict_eff2_col = "dict_eff2_raw_from_rates"

    fit_input_domain_by_plane: dict[int, tuple[float, float]] = {}
    for plane in (1, 2, 3, 4):
        vals = pd.to_numeric(dict_eff_by_plane.get(plane), errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(vals)
        if np.any(m):
            lo = float(np.nanmin(vals[m]))
            hi = float(np.nanmax(vals[m]))
            if np.isfinite(lo) and np.isfinite(hi):
                if hi < lo:
                    lo, hi = hi, lo
                fit_input_domain_by_plane[plane] = (lo, hi)

    real_raw_eff_outside_domain_fraction: dict[str, float] = {}
    for plane in (1, 2, 3, 4):
        vals = pd.to_numeric(raw_eff_by_plane.get(plane), errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(vals)
        frac = np.nan
        dom = fit_input_domain_by_plane.get(plane)
        if dom is not None and np.any(m):
            lo, hi = dom
            frac = float(np.mean((vals[m] < lo) | (vals[m] > hi)))
            if eff_transform_clip_input_to_dictionary_domain and frac > 0.0:
                log.warning(
                    "Plane %d: %.1f%% of raw empirical efficiencies lie outside dictionary calibration support [%.4f, %.4f].",
                    plane,
                    100.0 * frac,
                    lo,
                    hi,
                )
        real_raw_eff_outside_domain_fraction[f"eff{plane}"] = frac

    # Preferred global-rate columns for real and dictionary data.
    real_global_rate_col = _pick_global_rate_column(merged, preferred=global_rate_col)
    dict_global_rate_col = _pick_global_rate_column(dict_df_plot, preferred=global_rate_col)

    # Efficiency calibration: load both polynomial and isotonic from STEP 1.2 summary.
    fit_models_by_plane, fit_status, fit_summary_payload = _load_eff_fit_lines(build_summary_path)
    isotonic_models_by_plane, isotonic_status = _load_isotonic_calibration(build_summary_path)
    eff_calibration_method = str(cfg_42.get("eff_calibration_method", "isotonic")).strip().lower()
    if eff_calibration_method not in {"isotonic", "polynomial"}:
        eff_calibration_method = "isotonic"
    eff_transform_mode, eff_transform_mode_reason = _resolve_eff_transform_mode(
        eff_transform_mode_requested,
        fit_summary_payload,
    )
    fit_order_requested, fit_order_by_plane_from_summary = _read_fit_order_info(build_summary_path)

    # Choose calibration method: prefer isotonic, fall back to polynomial.
    use_isotonic = (
        eff_calibration_method == "isotonic"
        and isotonic_status == "ok"
        and len(isotonic_models_by_plane) > 0
    )
    if use_isotonic:
        log.info(
            "Efficiency calibration: isotonic (piecewise-linear monotonic) for %d plane(s).",
            len(isotonic_models_by_plane),
        )
        transformed_eff_by_plane, transformed_eff_formula_by_plane = _transform_efficiencies_isotonic(
            raw_eff_by_plane,
            isotonic_models_by_plane,
        )
        dict_transformed_eff_by_plane, dict_transformed_eff_formula_by_plane = _transform_efficiencies_isotonic(
            dict_eff_by_plane,
            isotonic_models_by_plane,
        )
    else:
        if eff_calibration_method == "isotonic" and isotonic_status != "ok":
            log.warning(
                "Isotonic calibration unavailable (%s); falling back to polynomial transform.",
                isotonic_status,
            )
        log.info(
            "Efficiency transform mode: request=%s -> using=%s (%s)",
            eff_transform_mode_requested,
            eff_transform_mode,
            eff_transform_mode_reason,
        )
        transformed_eff_by_plane, transformed_eff_formula_by_plane = _transform_efficiencies_with_fits(
            raw_eff_by_plane,
            fit_models_by_plane,
            mode=eff_transform_mode,
            input_domain_by_plane=(
                fit_input_domain_by_plane if eff_transform_clip_input_to_dictionary_domain else None
            ),
            clip_output_to_unit_interval=eff_transform_clip_output_to_unit_interval,
        )
        dict_transformed_eff_by_plane, dict_transformed_eff_formula_by_plane = _transform_efficiencies_with_fits(
            dict_eff_by_plane,
            fit_models_by_plane,
            mode=eff_transform_mode,
            input_domain_by_plane=(
                fit_input_domain_by_plane if eff_transform_clip_input_to_dictionary_domain else None
            ),
            clip_output_to_unit_interval=eff_transform_clip_output_to_unit_interval,
        )
    for plane in (1, 2, 3, 4):
        merged[f"eff{plane}_transformed"] = transformed_eff_by_plane[plane]
    for plane in (1, 2, 3, 4):
        dict_df_plot[f"dict_eff{plane}_transformed_from_rates"] = dict_transformed_eff_by_plane[plane]

    dict_rows_total = int(len(dict_df_plot))
    dict_contour_mask = _mask_sim_eff_within_tolerance_band(
        dict_df_plot,
        contour_eff_band_tolerance_pct,
    )
    dict_rows_for_contours = int(np.count_nonzero(dict_contour_mask))
    dict_df_contours = dict_df_plot.loc[dict_contour_mask].copy()
    if dict_rows_for_contours == 0:
        log.warning(
            "No dictionary rows satisfy 4-eff band tolerance (<= %.3f%%); "
            "contour backgrounds will be unavailable.",
            contour_eff_band_tolerance_pct,
        )
    else:
        log.info(
            "Contour-background dictionary rows: %d/%d (4-eff band <= %.3f%%).",
            dict_rows_for_contours,
            dict_rows_total,
            contour_eff_band_tolerance_pct,
        )

    lut_df = _load_lut(lut_path)
    lut_params = _lut_param_names(lut_df, lut_meta_path if lut_meta_path.exists() else None)
    lut_params = [p for p in lut_params if f"est_{p}" in merged.columns]

    if not lut_params:
        log.warning("No matching LUT parameters found in inference output. Uncertainty columns will be NaN.")

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
            est_v = pd.to_numeric(merged[est_col], errors="coerce").to_numpy(dtype=float)
            up = pd.to_numeric(merged[unc_pct_col], errors="coerce").to_numpy(dtype=float)
            abs_unc = np.abs(est_v) * np.abs(up) / 100.0
            merged[unc_abs_col] = np.where(np.isfinite(abs_unc), abs_unc, np.nan)
        else:
            merged[unc_abs_col] = np.nan

    flux_est_col = "est_flux_cm2_min" if "est_flux_cm2_min" in merged.columns else None
    eff_est_col = _choose_primary_eff_est_col(merged)
    distance_col = "best_distance" if "best_distance" in merged.columns else None

    if flux_est_col is None:
        fallback_flux = [c for c in merged.columns if c.startswith("est_") and "flux" in c]
        if fallback_flux:
            flux_est_col = sorted(fallback_flux)[0]
    if distance_col is None:
        log.error("Inference output has no 'best_distance' column.")
        return 1

    success = pd.to_numeric(merged[distance_col], errors="coerce").notna()
    if flux_est_col is not None:
        success &= pd.to_numeric(merged[flux_est_col], errors="coerce").notna()
    if eff_est_col is not None:
        success &= pd.to_numeric(merged[eff_est_col], errors="coerce").notna()
    merged["inference_success"] = success.astype(int)
    n_success = int(merged["inference_success"].sum())
    if n_success == 0:
        n_total = int(len(merged))
        finite_best_distance = int(pd.to_numeric(merged[distance_col], errors="coerce").notna().sum())
        finite_flux = (
            int(pd.to_numeric(merged[flux_est_col], errors="coerce").notna().sum())
            if flux_est_col is not None
            else 0
        )
        finite_eff = (
            int(pd.to_numeric(merged[eff_est_col], errors="coerce").notna().sum())
            if eff_est_col is not None
            else 0
        )
        no_candidate_rows = 0
        if "n_candidates" in merged.columns:
            no_candidate_rows = int(
                (pd.to_numeric(merged["n_candidates"], errors="coerce").fillna(0.0) <= 0.0).sum()
            )
        dict_feature_counts = ", ".join(
            f"{col}={int(pd.to_numeric(dict_work[col], errors='coerce').notna().sum())}"
            for col in feature_columns[:6]
            if col in dict_work.columns
        )
        real_feature_counts = ", ".join(
            f"{col}={int(pd.to_numeric(real_work[col], errors='coerce').notna().sum())}"
            for col in feature_columns[:6]
            if col in real_work.columns
        )
        if len(feature_columns) > 6:
            dict_feature_counts += ", ..."
            real_feature_counts += ", ..."
        log.warning(
            "Inference produced 0/%d successful rows (best_distance finite=%d, flux finite=%d, eff finite=%d, no-candidate rows=%d). "
            "Feature finite counts (dict): %s | (real): %s",
            n_total,
            finite_best_distance,
            finite_flux,
            finite_eff,
            no_candidate_rows,
            dict_feature_counts or "n/a",
            real_feature_counts or "n/a",
        )

    x, x_label, has_time_axis = _time_axis(merged)
    if has_time_axis:
        merged = merged.assign(execution_time_for_plot=x)

    out_csv = FILES_DIR / "real_results.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote real results: %s (%d rows)", out_csv, len(merged))

    flux_unc_abs_col = None
    eff_unc_abs_col = None
    if flux_est_col is not None:
        flux_param = flux_est_col.replace("est_", "", 1)
        candidate = f"unc_{flux_param}_abs"
        if candidate in merged.columns:
            flux_unc_abs_col = candidate
    if eff_est_col is not None:
        eff_param = eff_est_col.replace("est_", "", 1)
        candidate = f"unc_{eff_param}_abs"
        if candidate in merged.columns:
            eff_unc_abs_col = candidate

    _plot_pre_estimation_efficiencies(
        x=x,
        has_time_axis=has_time_axis,
        xlabel=x_label,
        global_rate=merged[real_global_rate_col] if real_global_rate_col is not None else None,
        global_rate_label=real_global_rate_col if real_global_rate_col is not None else "global_rate_hz",
        raw_eff_by_plane=raw_eff_by_plane,
        raw_eff_source_prefix=raw_eff_selected_prefix,
        transformed_eff_by_plane=transformed_eff_by_plane,
        transform_mode=eff_transform_mode,
        out_path=PLOT_EFF,
    )

    eff2_plot_source_cfg = str(cfg_42.get("eff2_global_rate_eff_source", "auto")).strip().lower()
    if eff2_plot_source_cfg not in {"auto", "transformed", "raw"}:
        log.warning(
            "Invalid step_4_2.eff2_global_rate_eff_source=%r; using 'auto'.",
            eff2_plot_source_cfg,
        )
        eff2_plot_source_cfg = "auto"

    has_real_eff2_trans = (
        "eff2_transformed" in merged.columns
        and pd.to_numeric(merged["eff2_transformed"], errors="coerce").notna().any()
    )
    has_dict_eff2_trans = (
        "dict_eff2_transformed_from_rates" in dict_df_plot.columns
        and pd.to_numeric(dict_df_plot["dict_eff2_transformed_from_rates"], errors="coerce").notna().any()
    )
    transformed_available = has_real_eff2_trans and has_dict_eff2_trans
    real_eff2_trans_frac_physical = (
        _fraction_in_closed_interval(merged["eff2_transformed"], 0.0, 1.0)
        if has_real_eff2_trans
        else 0.0
    )
    dict_eff2_trans_frac_physical = (
        _fraction_in_closed_interval(dict_df_plot["dict_eff2_transformed_from_rates"], 0.0, 1.0)
        if has_dict_eff2_trans
        else 0.0
    )
    transformed_eff_frac_physical_real: dict[str, float] = {}
    transformed_eff_frac_physical_dict: dict[str, float] = {}
    transformed_eff_boundary_metrics_real: dict[str, dict[str, float]] = {}
    transformed_eff_boundary_metrics_dict: dict[str, dict[str, float]] = {}
    for plane in (1, 2, 3, 4):
        real_col = f"eff{plane}_transformed"
        dict_col = f"dict_eff{plane}_transformed_from_rates"
        transformed_eff_frac_physical_real[f"eff{plane}"] = (
            _fraction_in_closed_interval(merged[real_col], 0.0, 1.0)
            if real_col in merged.columns
            else 0.0
        )
        transformed_eff_frac_physical_dict[f"eff{plane}"] = (
            _fraction_in_closed_interval(dict_df_plot[dict_col], 0.0, 1.0)
            if dict_col in dict_df_plot.columns
            else 0.0
        )
        transformed_eff_boundary_metrics_real[f"eff{plane}"] = (
            _boundary_fraction_metrics(merged[real_col])
            if real_col in merged.columns
            else _boundary_fraction_metrics(pd.Series(dtype=float))
        )
        transformed_eff_boundary_metrics_dict[f"eff{plane}"] = (
            _boundary_fraction_metrics(dict_df_plot[dict_col])
            if dict_col in dict_df_plot.columns
            else _boundary_fraction_metrics(pd.Series(dtype=float))
        )

    if eff2_plot_source_cfg == "raw":
        use_transformed_eff2_for_plane_plot = False
    elif eff2_plot_source_cfg == "transformed":
        use_transformed_eff2_for_plane_plot = transformed_available
    else:
        # Auto: only trust transformed eff2 when both real and dictionary are
        # predominantly within physical bounds.
        use_transformed_eff2_for_plane_plot = (
            transformed_available
            and real_eff2_trans_frac_physical >= 0.95
            and dict_eff2_trans_frac_physical >= 0.95
        )
        if transformed_available and not use_transformed_eff2_for_plane_plot:
            log.warning(
                "Fallback to raw eff2: transformed physical fractions "
                "(real=%.3f, dict=%.3f) below threshold 0.95.",
                real_eff2_trans_frac_physical,
                dict_eff2_trans_frac_physical,
            )

    eff2_real_col_for_plane_plot = (
        "eff2_transformed" if use_transformed_eff2_for_plane_plot else "eff2_from_data"
    )
    dict_eff2_col_for_plane_plot = (
        "dict_eff2_transformed_from_rates" if use_transformed_eff2_for_plane_plot else dict_eff2_col
    )

    n_est_curve_real = 0
    n_est_curve_dict = 0
    est_curve_eff_col = _pick_estimated_eff_col_for_plane(merged, plane=2)
    dict_curve_eff_col = _pick_dictionary_eff_col_for_plane(dict_df_plot, plane=2)
    if (
        flux_est_col is not None
        and est_curve_eff_col is not None
        and dict_curve_eff_col is not None
        and dict_global_rate_col is not None
    ):
        n_est_curve_real, n_est_curve_dict = _plot_estimated_curve_flux_vs_eff(
            real_df=merged,
            dict_df=dict_df_contours,
            est_flux_col=flux_est_col,
            est_eff_col=est_curve_eff_col,
            dict_eff_col=dict_curve_eff_col,
            dict_rate_col=dict_global_rate_col,
            out_path=PLOT_EST_CURVE,
        )
    else:
        missing = []
        if flux_est_col is None:
            missing.append("estimated flux column")
        if est_curve_eff_col is None:
            missing.append("estimated efficiency column")
        if dict_curve_eff_col is None:
            missing.append("dictionary efficiency column")
        if dict_global_rate_col is None:
            missing.append("dictionary global_rate column")
        _plot_placeholder(
            PLOT_EST_CURVE,
            "Estimated curve in flux-eff plane",
            "Cannot build plot: missing " + ", ".join(missing) + ".",
        )

    story_eff_col = eff_est_col if eff_est_col is not None else est_curve_eff_col
    story_eff_series_map: dict[str, pd.Series] = {}
    for plane in (1, 2, 3, 4):
        col = f"est_eff_sim_{plane}"
        if col in merged.columns and pd.to_numeric(merged[col], errors="coerce").notna().any():
            story_eff_series_map[col] = merged[col]
    if not story_eff_series_map and story_eff_col is not None and story_eff_col in merged.columns:
        story_eff_series_map[story_eff_col] = merged[story_eff_col]

    if flux_est_col is not None and story_eff_col is not None and real_global_rate_col is not None:
        _plot_flux_recovery_story_real(
            x=x,
            has_time_axis=has_time_axis,
            xlabel=x_label,
            distance_series=merged[distance_col],
            distance_label=distance_col,
            eff_series=merged[story_eff_col],
            eff_label=story_eff_col,
            eff_series_map=story_eff_series_map,
            global_rate_series=merged[real_global_rate_col],
            global_rate_label=real_global_rate_col,
            flux_est_series=merged[flux_est_col],
            flux_unc_series=merged[flux_unc_abs_col] if flux_unc_abs_col is not None else None,
            flux_reference_series=None,
            flux_reference_label="Reference flux",
            out_path=PLOT_RECOVERY_STORY,
        )
    else:
        missing = []
        if flux_est_col is None:
            missing.append("estimated flux column")
        if story_eff_col is None:
            missing.append("estimated efficiency column")
        if real_global_rate_col is None:
            missing.append("real global_rate column")
        _plot_placeholder(
            PLOT_RECOVERY_STORY,
            "Real-data recovery story",
            "Cannot build plot: missing " + ", ".join(missing) + ".",
        )

    estimated_eff_boundary_metrics: dict[str, dict[str, float]] = {}
    for plane in (1, 2, 3, 4):
        est_col = f"est_eff_sim_{plane}"
        if est_col in merged.columns:
            estimated_eff_boundary_metrics[f"eff{plane}"] = _boundary_fraction_metrics(merged[est_col])

    distance_group_summary: dict[str, dict[str, float] | None] = {}
    for label, col in (
        ("eff_empirical", "best_distance_base_share_eff_empirical"),
        ("tt_rates", "best_distance_base_share_tt_rates"),
        ("other", "best_distance_base_share_other"),
    ):
        if col not in merged.columns:
            distance_group_summary[label] = None
            continue
        vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            distance_group_summary[label] = None
            continue
        distance_group_summary[label] = {
            "median": float(np.nanmedian(vals)),
            "p90": float(np.nanpercentile(vals, 90)),
        }

    ts_valid = pd.Series([], dtype="datetime64[ns]")
    if has_time_axis:
        ts_valid = pd.to_datetime(x, errors="coerce").dropna()

    summary = {
        "real_collected_csv": str(real_path),
        "dictionary_csv": str(dict_path),
        "uncertainty_lut_csv": str(lut_path),
        "matching_criteria_source": "step_2_1",
        "distance_metric": distance_metric,
        "interpolation_k": interpolation_k,
        "inverse_mapping": inverse_mapping_cfg,
        "inverse_mapping_runtime_applied": inverse_mapping_cfg_runtime,
        "inherit_step_2_1_method": bool(inherit_step_2_1_method),
        "feature_strategy": feature_strategy,
        "derived_feature_subset_mode_requested": derived_feature_subset_mode_requested,
        "derived_feature_subset_mode": derived_feature_subset_mode,
        "derived_trigger_types_used": derived_trigger_types,
        "derived_trigger_types_override_applied": bool(derived_trigger_types_override_applied),
        "n_features_used": int(len(feature_columns)),
        "feature_columns_used": feature_columns,
        "n_rows": int(len(merged)),
        "n_successful_rows": int(merged["inference_success"].sum()),
        "coverage_fraction": float(merged["inference_success"].mean()),
        "flux_estimate_column": flux_est_col,
        "eff_estimate_column": eff_est_col,
        "distance_column": distance_col,
        "n_events_column_used": n_events_col_used,
        "real_global_rate_column_used": real_global_rate_col,
        "dictionary_global_rate_column_used": dict_global_rate_col,
        "iso_rate_efficiency_band_tolerance_pct": float(contour_eff_band_tolerance_pct),
        "dictionary_rows_total_for_iso_rate_contours": int(dict_rows_total),
        "dictionary_rows_for_iso_rate_contours": int(dict_rows_for_contours),
        "dictionary_rows_excluded_iso_rate_eff_band": int(dict_rows_total - dict_rows_for_contours),
        "build_summary_json_used": str(build_summary_path),
        "fit_lines_load_status": fit_status,
        "fit_polynomial_relation_from_summary": fit_summary_payload.get("fit_polynomial_relation"),
        "fit_polynomial_x_variable_from_summary": fit_summary_payload.get("fit_polynomial_x_variable"),
        "fit_polynomial_y_variable_from_summary": fit_summary_payload.get("fit_polynomial_y_variable"),
        "fit_polynomial_order_requested": fit_order_requested,
        "fit_polynomial_order_by_plane": fit_order_by_plane_from_summary,
        "fit_lines_by_plane": {
            str(k): {"a": float(v[0]), "b": float(v[1])}
            for k, v in fit_models_by_plane.items()
            if len(v) == 2
        },
        "fit_polynomials_by_plane": {
            str(k): {
                "order": int(len(v) - 1),
                "coefficients": [float(c) for c in v],
            }
            for k, v in fit_models_by_plane.items()
        },
        "eff_calibration_method_requested": eff_calibration_method,
        "eff_calibration_method_used": "isotonic" if use_isotonic else "polynomial",
        "isotonic_calibration_status": isotonic_status,
        "eff_transform_mode_requested": eff_transform_mode_requested,
        "eff_transform_mode": eff_transform_mode,
        "eff_transform_mode_resolution_reason": eff_transform_mode_reason,
        "eff_transform_clip_input_to_dictionary_domain": bool(
            eff_transform_clip_input_to_dictionary_domain
        ),
        "eff_transform_clip_output_to_unit_interval": bool(
            eff_transform_clip_output_to_unit_interval
        ),
        "eff_transform_input_domain_by_plane": {
            f"eff{p}": {
                "min": (
                    float(fit_input_domain_by_plane[p][0])
                    if p in fit_input_domain_by_plane
                    else None
                ),
                "max": (
                    float(fit_input_domain_by_plane[p][1])
                    if p in fit_input_domain_by_plane
                    else None
                ),
            }
            for p in (1, 2, 3, 4)
        },
        "real_raw_efficiency_fraction_outside_fit_domain_by_plane": {
            f"eff{p}": (
                float(real_raw_eff_outside_domain_fraction.get(f"eff{p}"))
                if np.isfinite(real_raw_eff_outside_domain_fraction.get(f"eff{p}", np.nan))
                else None
            )
            for p in (1, 2, 3, 4)
        },
        "raw_efficiency_columns": {
            "eff1": "eff1_raw_from_data",
            "eff2": "eff2_raw_from_data",
            "eff3": "eff3_raw_from_data",
            "eff4": "eff4_raw_from_data",
        },
        "efficiency_source_task_ids": selected_task_ids,
        "efficiency_source_most_advanced_task_id": int(max(selected_task_ids)) if selected_task_ids else 1,
        "efficiency_source_prefix_mode": efficiency_source_prefix_mode,
        "efficiency_source_preferred_prefix_order": efficiency_source_prefix_order,
        "efficiency_source_prefix_used_real": raw_eff_selected_prefix,
        "efficiency_source_prefix_used_dictionary": dict_eff_selected_prefix,
        "feature_empirical_efficiency_preferred_prefix_order": preferred_feature_prefixes,
        "feature_empirical_efficiency_prefix_used_real": feature_eff_source_prefix.get("real"),
        "feature_empirical_efficiency_prefix_used_dictionary": feature_eff_source_prefix.get("dictionary"),
        "feature_empirical_efficiency_finite_counts": feature_eff_finite_count,
        "feature_empirical_efficiency_formulas": feature_eff_source_formula,
        "raw_efficiency_formulas": {f"eff{p}": raw_eff_formula_by_plane.get(p) for p in (1, 2, 3, 4)},
        "raw_efficiency_rate_columns": {
            f"eff{p}": raw_eff_cols_by_plane.get(p, {})
            for p in (1, 2, 3, 4)
        },
        "transformed_efficiency_columns": {
            "eff1": "eff1_transformed",
            "eff2": "eff2_transformed",
            "eff3": "eff3_transformed",
            "eff4": "eff4_transformed",
        },
        "transformed_efficiency_formulas": {
            f"eff{p}": transformed_eff_formula_by_plane.get(p)
            for p in (1, 2, 3, 4)
        },
        "eff2_real_column": eff2_real_col_for_plane_plot,
        "eff2_global_rate_eff_source_requested": eff2_plot_source_cfg,
        "eff2_global_rate_eff_source_used": (
            "transformed" if use_transformed_eff2_for_plane_plot else "raw"
        ),
        "eff2_global_rate_transformed_fraction_in_0_1": {
            "real": float(real_eff2_trans_frac_physical),
            "dictionary": float(dict_eff2_trans_frac_physical),
        },
        "transformed_efficiency_fraction_in_0_1_by_plane": {
            "real": transformed_eff_frac_physical_real,
            "dictionary": transformed_eff_frac_physical_dict,
        },
        "transformed_efficiency_boundary_metrics_by_plane": {
            "real": transformed_eff_boundary_metrics_real,
            "dictionary": transformed_eff_boundary_metrics_dict,
        },
        "estimated_efficiency_boundary_metrics_by_plane": estimated_eff_boundary_metrics,
        "best_match_non_hist_distance_share": distance_group_summary,
        "efficiency_feature_out_of_support_masking": (
            eff_oos_masking_summary if isinstance(eff_oos_masking_summary, dict) else None
        ),
        "eff2_formula": (
            transformed_eff_formula_by_plane.get(2)
            if eff2_real_col_for_plane_plot == "eff2_transformed"
            else eff2_formula
        ),
        "eff2_raw_formula": eff2_formula,
        "eff2_transformed_formula": transformed_eff_formula_by_plane.get(2),
        "eff2_real_rate_columns": raw_eff_cols_by_plane.get(2, {}),
        "eff2_dictionary_eff_column": dict_eff2_col_for_plane_plot,
        "eff2_dictionary_eff_formula": (
            dict_transformed_eff_formula_by_plane.get(2)
            if dict_eff2_col_for_plane_plot == "dict_eff2_transformed_from_rates"
            else "raw_from_rates"
        ),
        "eff2_dictionary_rate_columns": {
            "three_plane_col": dict_eff_cols_by_plane.get(2, {}).get("three_plane_col"),
            "four_plane_col": dict_eff_cols_by_plane.get(2, {}).get("four_plane_col"),
        },
        "pre_estimation_efficiency_plot": str(PLOT_EFF),
        "estimated_curve_eff_column": est_curve_eff_col,
        "dictionary_curve_eff_column": dict_curve_eff_col,
        "n_estimated_curve_points": int(n_est_curve_real),
        "n_dictionary_curve_background_points": int(n_est_curve_dict),
        "estimated_curve_plot": str(PLOT_EST_CURVE),
        "recovery_story_plot": str(PLOT_RECOVERY_STORY),
        "recovery_story_eff_column": story_eff_col,
        "recovery_story_global_rate_column": real_global_rate_col,
        "recovery_story_flux_column": flux_est_col,
        "lut_param_names_used": lut_params,
        "uncertainty_quantile": uncertainty_quantile,
        "has_time_axis": bool(has_time_axis),
        "time_min_utc": str(ts_valid.min()) if len(ts_valid) else None,
        "time_max_utc": str(ts_valid.max()) if len(ts_valid) else None,
        "feature_mapping_preview": feature_mapping[:25],
    }
    out_summary = FILES_DIR / "real_analysis_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote summary: %s", out_summary)
    log.info("Wrote plots in: %s", PLOTS_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
