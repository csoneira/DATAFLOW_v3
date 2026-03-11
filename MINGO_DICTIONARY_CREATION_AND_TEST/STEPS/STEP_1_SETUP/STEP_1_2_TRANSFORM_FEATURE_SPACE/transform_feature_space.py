#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_TRANSFORM_FEATURE_SPACE/transform_feature_space.py
Purpose: STEP 1.2 — Transform expanded feature space from STEP 1.1.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-11
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_1_SETUP/STEP_1_2_TRANSFORM_FEATURE_SPACE/transform_feature_space.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────
STEP_DIR = Path(__file__).resolve().parent
if STEP_DIR.parents[1].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[2]
else:
    PIPELINE_DIR = STEP_DIR.parents[1]
DEFAULT_CONFIG = PIPELINE_DIR / "config_method.json"
DEFAULT_INPUT = (
    STEP_DIR.parent / "STEP_1_1_COLLECT_DATA" / "OUTPUTS" / "FILES" / "collected_data.csv"
)
DEFAULT_STEP13_SELECTED_FEATURES = (
    STEP_DIR.parent / "STEP_1_3_BUILD_DICTIONARY" / "OUTPUTS" / "FILES" / "selected_feature_columns.json"
)
DEFAULT_STEP14_SELECTED_FEATURES = (
    STEP_DIR.parent / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY" / "OUTPUTS" / "FILES" / "selected_feature_columns.json"
)
DEFAULT_STEP11_PARAMETER_SPACE = (
    STEP_DIR.parent / "STEP_1_1_COLLECT_DATA" / "OUTPUTS" / "FILES" / "parameter_space_columns.json"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_FIGURE_COUNTER = 0
FIGURE_STEP_PREFIX = "1_2"


def _save_figure(fig: plt.Figure, path: Path, **kwargs) -> None:
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

CANONICAL_TT_LABELS = frozenset(
    {
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
)
TT_RATE_COLUMN_RE = re.compile(r"^(?P<prefix>.+_tt)_(?P<label>[^_]+)_rate_hz$")
HIST_RATE_COLUMN_RE = re.compile(r"^events_per_second_(?P<bin>\d+)_rate_hz$")
STANDARD_TASK_PREFIXES = ("raw_tt", "clean_tt", "cal_tt", "list_tt", "fit_tt", "post_tt")
CANONICAL_PREFIX_PRIORITY = (
    "post_tt",
    "fit_to_post_tt",
    "fit_tt",
    "list_to_fit_tt",
    "list_tt",
    "cal_tt",
    "clean_tt",
    "raw_to_clean_tt",
    "raw_tt",
    "corr_tt",
    "task5_to_corr_tt",
    "fit_to_corr_tt",
    "definitive_tt",
)
RATE_HIST_PLACEHOLDER_COL = "__RATE_HISTOGRAM_SUPPRESSED__"
RATE_HIST_PLACEHOLDER_LABEL = "events_per_second_<bin>_rate_hz [suppressed]"


def _clear_plots_dir() -> None:
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
    format="[%(levelname)s] STEP_1.2 — %(message)s", level=logging.INFO
)
log = logging.getLogger("STEP_1.2")


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


def _find_eff_source_columns(df: pd.DataFrame) -> list[str] | None:
    direct = ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]
    if all(col in df.columns for col in direct):
        return direct
    parsed = ["eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4"]
    if all(col in df.columns for col in parsed):
        return parsed
    return None


def _parse_efficiencies(raw: object) -> tuple[float, float, float, float] | None:
    value = raw
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: list[float] = []
    for item in value:
        try:
            num = float(item)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(num):
            return None
        out.append(num)
    return (out[0], out[1], out[2], out[3])


def _ensure_efficiency_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_cols = _find_eff_source_columns(out)
    if source_cols is not None and source_cols != ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]:
        for idx, src in enumerate(source_cols, start=1):
            out[f"eff_p{idx}"] = pd.to_numeric(out[src], errors="coerce")
    elif source_cols is None and "efficiencies" in out.columns:
        parsed = out["efficiencies"].map(_parse_efficiencies)
        for idx in range(4):
            out[f"eff_p{idx + 1}"] = parsed.map(lambda v, j=idx: np.nan if v is None else float(v[j]))

    for idx in range(4):
        ep = f"eff_p{idx + 1}"
        es = f"eff_sim_{idx + 1}"
        if ep in out.columns and es not in out.columns:
            out[es] = pd.to_numeric(out[ep], errors="coerce")
    return out


def _build_prefix_global_rate_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    by_prefix: dict[str, list[str]] = {}
    for col in df.columns:
        match = TT_RATE_COLUMN_RE.match(str(col))
        if match is None:
            continue
        prefix = str(match.group("prefix")).strip()
        label = _normalize_tt_label(match.group("label"))
        if label not in CANONICAL_TT_LABELS:
            continue
        by_prefix.setdefault(prefix, []).append(str(col))

    out = df.copy()
    rate_col_by_prefix: dict[str, str] = {}
    tt_cols_by_prefix: dict[str, list[str]] = {}
    for prefix in sorted(by_prefix):
        cols = sorted(set(by_prefix[prefix]))
        if not cols:
            continue
        sum_col = f"events_per_second_global_rate_{prefix}"
        summed = pd.Series(0.0, index=out.index, dtype=float)
        valid_any = pd.Series(False, index=out.index, dtype=bool)
        for col in cols:
            numeric = pd.to_numeric(out[col], errors="coerce")
            summed = summed + numeric.fillna(0.0)
            valid_any = valid_any | numeric.notna()
        out[sum_col] = summed.where(valid_any, np.nan)
        rate_col_by_prefix[prefix] = sum_col
        tt_cols_by_prefix[prefix] = cols
    return out, rate_col_by_prefix, tt_cols_by_prefix


def _select_canonical_global_rate(
    df: pd.DataFrame,
    *,
    rate_col_by_prefix: dict[str, str],
    preferred_prefixes: tuple[str, ...],
    fallback_existing_col: str = "events_per_second_global_rate",
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    canonical = pd.Series(np.nan, index=out.index, dtype=float)
    source_counts: dict[str, int] = {}

    all_prefixes: list[str] = []
    seen: set[str] = set()
    for prefix in preferred_prefixes:
        if prefix in seen:
            continue
        all_prefixes.append(prefix)
        seen.add(prefix)
    for prefix in sorted(rate_col_by_prefix.keys()):
        if prefix in seen:
            continue
        all_prefixes.append(prefix)
        seen.add(prefix)

    for prefix in all_prefixes:
        candidate = rate_col_by_prefix.get(prefix)
        if candidate is None or candidate not in out.columns:
            continue
        vals = pd.to_numeric(out[candidate], errors="coerce")
        fill_mask = canonical.isna() & vals.notna()
        n_fill = int(fill_mask.sum())
        if n_fill > 0:
            canonical = canonical.where(~fill_mask, vals)
            source_counts[prefix] = source_counts.get(prefix, 0) + n_fill

    if fallback_existing_col in out.columns:
        vals = pd.to_numeric(out[fallback_existing_col], errors="coerce")
        fill_mask = canonical.isna() & vals.notna()
        n_fill = int(fill_mask.sum())
        if n_fill > 0:
            canonical = canonical.where(~fill_mask, vals)
            source_counts["fallback_existing_global_rate"] = (
                source_counts.get("fallback_existing_global_rate", 0) + n_fill
            )

    out["events_per_second_global_rate"] = canonical
    return out, source_counts


def _ensure_standard_task_prefix_rate_columns(
    df: pd.DataFrame,
    *,
    rate_col_by_prefix: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added_cols: list[str] = []
    for prefix in STANDARD_TASK_PREFIXES:
        col = f"events_per_second_global_rate_{prefix}"
        if col in out.columns:
            rate_col_by_prefix.setdefault(prefix, col)
            continue
        out[col] = np.nan
        rate_col_by_prefix[prefix] = col
        added_cols.append(col)
    return out, added_cols


def _compute_eff(n_four: pd.Series, n_three_missing: pd.Series) -> pd.Series:
    denom = n_four + n_three_missing
    return n_four / denom.replace({0: np.nan})


def _compute_empirical_efficiencies(
    df: pd.DataFrame,
    *,
    preferred_prefixes: tuple[str, ...],
) -> tuple[pd.DataFrame, str | None, list[str]]:
    out = df.copy()
    all_prefixes: list[str] = []
    seen: set[str] = set()
    for p in preferred_prefixes:
        if p not in seen:
            all_prefixes.append(p)
            seen.add(p)
    for p in sorted(set(str(m.group("prefix")).strip() for c in out.columns for m in [TT_RATE_COLUMN_RE.match(str(c))] if m)):
        if p not in seen:
            all_prefixes.append(p)
            seen.add(p)

    required_labels = ("1234", "234", "134", "124", "123")
    per_plane_missing = {
        1: "234",
        2: "134",
        3: "124",
        4: "123",
    }

    eff_series = {
        1: pd.Series(np.nan, index=out.index, dtype=float),
        2: pd.Series(np.nan, index=out.index, dtype=float),
        3: pd.Series(np.nan, index=out.index, dtype=float),
        4: pd.Series(np.nan, index=out.index, dtype=float),
    }
    source_prefix = pd.Series("", index=out.index, dtype=object)
    used_prefixes: list[str] = []

    for prefix in all_prefixes:
        four_col = f"{prefix}_1234_rate_hz"
        needed = [f"{prefix}_{lab}_rate_hz" for lab in required_labels]
        if not all(col in out.columns for col in needed):
            continue

        n_four = pd.to_numeric(out[four_col], errors="coerce")
        any_plane_used = False
        for plane in (1, 2, 3, 4):
            miss_label = per_plane_missing[plane]
            miss_col = f"{prefix}_{miss_label}_rate_hz"
            n_miss = pd.to_numeric(out[miss_col], errors="coerce")
            eff_candidate = _compute_eff(n_four, n_miss)
            fill_mask = eff_series[plane].isna() & eff_candidate.notna()
            if bool(fill_mask.any()):
                eff_series[plane] = eff_series[plane].where(~fill_mask, eff_candidate)
                source_prefix = source_prefix.where(~fill_mask, prefix)
                any_plane_used = True
        if any_plane_used:
            used_prefixes.append(prefix)

    for plane in (1, 2, 3, 4):
        out[f"eff_empirical_{plane}"] = eff_series[plane]

    valid_source = source_prefix.astype(str).str.len() > 0
    out["eff_empirical_source_prefix"] = source_prefix.where(valid_source, np.nan)
    selected_prefix = used_prefixes[0] if used_prefixes else None
    return out, selected_prefix, used_prefixes


def _add_derived_physics_helper_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    out = _ensure_efficiency_columns(df)
    if not all(col in out.columns for col in ("eff_p1", "eff_p2", "eff_p3", "eff_p4")):
        return out, 0

    eff_cols = ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]
    eff_df = out[eff_cols].apply(pd.to_numeric, errors="coerce")
    valid_eff = eff_df.notna().all(axis=1)
    eff_prod = eff_df.prod(axis=1, min_count=4).where(valid_eff, np.nan)
    out["efficiency_product_4planes"] = eff_prod

    out["efficiency_product_123"] = (
        pd.to_numeric(out["eff_p1"], errors="coerce")
        * pd.to_numeric(out["eff_p2"], errors="coerce")
        * pd.to_numeric(out["eff_p3"], errors="coerce")
    )
    out["efficiency_product_234"] = (
        pd.to_numeric(out["eff_p2"], errors="coerce")
        * pd.to_numeric(out["eff_p3"], errors="coerce")
        * pd.to_numeric(out["eff_p4"], errors="coerce")
    )
    out["efficiency_product_12"] = (
        pd.to_numeric(out["eff_p1"], errors="coerce")
        * pd.to_numeric(out["eff_p2"], errors="coerce")
    )
    out["efficiency_product_34"] = (
        pd.to_numeric(out["eff_p3"], errors="coerce")
        * pd.to_numeric(out["eff_p4"], errors="coerce")
    )

    helper_count = 5
    rate = pd.to_numeric(out.get("events_per_second_global_rate"), errors="coerce")
    product_to_proxy = {
        "efficiency_product_4planes": "flux_proxy_rate_div_effprod",
        "efficiency_product_123": "flux_proxy_rate_div_effprod_123",
        "efficiency_product_234": "flux_proxy_rate_div_effprod_234",
        "efficiency_product_12": "flux_proxy_rate_div_effprod_12",
        "efficiency_product_34": "flux_proxy_rate_div_effprod_34",
    }
    for prod_col, proxy_col in product_to_proxy.items():
        prod = pd.to_numeric(out.get(prod_col), errors="coerce")
        proxy = pd.Series(np.nan, index=out.index, dtype=float)
        valid = prod.notna() & (prod > 0.0) & rate.notna()
        proxy.loc[valid] = rate.loc[valid] / prod.loc[valid]
        out[proxy_col] = proxy
        helper_count += 1
    return out, helper_count


def _normalize_requested_columns(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                out.append(text)
        return out
    return []


def _resolve_configured_keep_columns(
    df: pd.DataFrame,
    *,
    requested_patterns: list[str],
) -> tuple[list[str], list[str]]:
    available = list(df.columns)
    available_set = set(available)
    resolved: list[str] = []
    seen: set[str] = set()
    unmatched: list[str] = []

    for pattern in requested_patterns:
        if any(ch in pattern for ch in ("*", "?", "[")):
            matches = [c for c in available if fnmatch.fnmatchcase(c, pattern)]
        else:
            matches = [pattern] if pattern in available_set else []
        if not matches:
            unmatched.append(pattern)
            continue
        for col in matches:
            if col in seen:
                continue
            resolved.append(col)
            seen.add(col)
    return resolved, unmatched


def _normalize_plot_pairs(raw: object) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        x, y = item
        if not isinstance(x, str) or not isinstance(y, str):
            continue
        xs = x.strip()
        ys = y.strip()
        if not xs or not ys:
            continue
        out.append((xs, ys))
    return out


def _load_selected_feature_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    cols = payload.get("selected_feature_columns", [])
    if not isinstance(cols, list):
        return []
    out: list[str] = []
    for col in cols:
        if not isinstance(col, str):
            continue
        text = col.strip()
        if text:
            out.append(text)
    return out


def _load_parameter_space_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []

    candidates: list[object] = [
        payload.get("parameter_space_columns_downstream_preferred"),
        payload.get("selected_parameter_space_columns_downstream_preferred"),
        payload.get("selected_parameter_space_columns"),
        payload.get("parameter_space_columns"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, list):
            continue
        for col in raw:
            if not isinstance(col, str):
                continue
            text = col.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        if out:
            break
    return out


def _select_best_tt_prefix(
    available_prefixes: set[str],
    *,
    priority: tuple[str, ...],
) -> str | None:
    """Return the highest-priority tt prefix that is present in the data."""
    for p in priority:
        if p in available_prefixes:
            return p
    if available_prefixes:
        return sorted(available_prefixes)[0]
    return None


def _drop_non_best_tt_columns(
    df: pd.DataFrame,
    *,
    best_prefix: str | None,
    tt_cols_by_prefix: dict[str, list[str]],
    rate_col_by_prefix: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop individual tt-rate columns and per-prefix global-rate summaries for all
    prefixes except *best_prefix*.

    The two dictionaries are iterated independently so that per-prefix global-rate
    columns (events_per_second_global_rate_{prefix}) are always cleaned up even when
    no individual breakdown columns exist (tt_cols_by_prefix is empty).
    """
    if best_prefix is None:
        return df, []
    drop_set: set[str] = set()
    # Individual tt-rate breakdown columns (prefix_tt_{label}_rate_hz)
    for prefix, cols in tt_cols_by_prefix.items():
        if prefix == best_prefix:
            continue
        drop_set.update(c for c in cols if c in df.columns)
    # Per-prefix global-rate summary columns (events_per_second_global_rate_{prefix})
    # are all redundant: the canonical events_per_second_global_rate already holds
    # the value from the best prefix, so drop every per-prefix variant.
    for prefix, rate_col in rate_col_by_prefix.items():
        if rate_col in df.columns:
            drop_set.add(rate_col)
    drop_cols = list(drop_set)
    if not drop_cols:
        return df, []
    return df.drop(columns=drop_cols), drop_cols


def _is_rate_histogram_feature(col: str) -> bool:
    return HIST_RATE_COLUMN_RE.match(str(col)) is not None


def _resolve_feature_matrix_plot_columns(
    feature_cols: list[str],
    *,
    include_rate_histogram: bool,
) -> tuple[list[str], list[str], bool]:
    base = [c for c in feature_cols if isinstance(c, str) and c.strip()]
    hist_cols = [c for c in base if _is_rate_histogram_feature(c)]
    non_hist_cols = [c for c in base if c not in set(hist_cols)]

    if include_rate_histogram or not hist_cols:
        return non_hist_cols + hist_cols, hist_cols, False
    return non_hist_cols + [RATE_HIST_PLACEHOLDER_COL], hist_cols, True


def _default_step13_like_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude: set[str] = {
        "filename_base",
        "task_id",
        "param_hash_x",
        "param_set_id",
        "cos_n",
        "flux_cm2_min",
        "z_plane_1",
        "z_plane_2",
        "z_plane_3",
        "z_plane_4",
        "eff_p1",
        "eff_p2",
        "eff_p3",
        "eff_p4",
        "eff_sim_1",
        "eff_sim_2",
        "eff_sim_3",
        "eff_sim_4",
        "selected_rows",
        "requested_rows",
        "generated_events_count",
        "num_events",
        "event_count",
        "n_events",
    }
    return [c for c in df.columns if c not in exclude]


def _resolve_parameter_matrix_columns(
    df: pd.DataFrame,
    *,
    cfg_12: dict,
    cfg_14: dict,
    parameter_space_path: Path,
) -> tuple[list[str], list[str], str]:
    raw = cfg_12.get("parameter_matrix_columns", "auto")
    requested: list[str]
    source = "step_1_2.parameter_matrix_columns"
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        requested = _load_parameter_space_columns(parameter_space_path)
        if requested:
            source = f"artifact:{parameter_space_path.name}"
        else:
            requested = _normalize_requested_columns(cfg_14.get("parameter_columns"))
            if requested:
                source = "step_1_4.parameter_columns"
            else:
                requested = [
                    "flux_cm2_min",
                    "eff_sim_1",
                    "eff_sim_2",
                    "eff_sim_3",
                    "eff_sim_4",
                    "cos_n",
                ]
                source = "internal_fallback"
    else:
        requested = _normalize_requested_columns(raw)
        source = "step_1_2.parameter_matrix_columns"

    resolved, unmatched = _resolve_configured_keep_columns(
        df,
        requested_patterns=requested,
    )
    return resolved, unmatched, source


def _resolve_feature_matrix_columns(
    df: pd.DataFrame,
    *,
    cfg_12: dict,
    config: dict,
    parameter_cols: list[str],
) -> tuple[list[str], list[str], str]:
    raw = cfg_12.get("feature_matrix_columns", "auto")
    parameter_set = set(parameter_cols)

    if isinstance(raw, str) and raw.strip().lower() == "auto":
        for candidate_path in (DEFAULT_STEP14_SELECTED_FEATURES, DEFAULT_STEP13_SELECTED_FEATURES):
            selected = _load_selected_feature_columns(candidate_path)
            if selected:
                cols = [c for c in selected if c in df.columns and c not in parameter_set]
                unmatched = [c for c in selected if c not in df.columns]
                if cols:
                    return cols, unmatched, f"artifact:{candidate_path.name}"

        common_cfg = config.get("common_feature_space", {}) if isinstance(config, dict) else {}
        common_raw = common_cfg.get("feature_columns")
        if not (
            isinstance(common_raw, str)
            and common_raw.strip().lower() in {"", "auto", "step12_selected", "selected", "selected_json"}
        ):
            patterns = _normalize_requested_columns(common_raw)
            resolved, unmatched = _resolve_configured_keep_columns(df, requested_patterns=patterns)
            cols = [c for c in resolved if c not in parameter_set]
            if cols:
                return cols, unmatched, "common_feature_space.feature_columns"

        cols = [c for c in _default_step13_like_feature_columns(df) if c not in parameter_set]
        return cols, [], "step13_style_fallback"

    requested = _normalize_requested_columns(raw)
    resolved, unmatched = _resolve_configured_keep_columns(df, requested_patterns=requested)
    cols = [c for c in resolved if c not in parameter_set]
    return cols, unmatched, "step_1_2.feature_matrix_columns"


def _plot_parameter_scatter_matrix(
    df: pd.DataFrame,
    *,
    parameter_cols: list[str],
    sample_max_rows: int,
    random_seed: int,
) -> None:
    cols = [c for c in parameter_cols if c in df.columns]
    if len(cols) < 2:
        return

    plot_df = df
    if sample_max_rows > 0 and len(plot_df) > sample_max_rows:
        plot_df = plot_df.sample(n=sample_max_rows, random_state=random_seed)

    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n), squeeze=False)
    for i, cy in enumerate(cols):
        for j, cx in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                vals = pd.to_numeric(plot_df[cx], errors="coerce").dropna()
                if vals.empty:
                    ax.set_visible(False)
                    continue
                ax.hist(vals.to_numpy(dtype=float), bins=35, color="#4C78A8", alpha=0.75)
                ax.set_xlabel(cx, fontsize=7)
                ax.set_ylabel("Rows", fontsize=7)
                ax.grid(True, alpha=0.2)
            elif i > j:
                x = pd.to_numeric(plot_df[cx], errors="coerce")
                y = pd.to_numeric(plot_df[cy], errors="coerce")
                m = x.notna() & y.notna()
                if bool(m.any()):
                    ax.scatter(
                        x[m].to_numpy(dtype=float),
                        y[m].to_numpy(dtype=float),
                        s=7,
                        alpha=0.28,
                        color="#4C78A8",
                        edgecolors="none",
                    )
                ax.set_xlabel(cx, fontsize=7)
                ax.set_ylabel(cy, fontsize=7)
                ax.grid(True, alpha=0.2)
            else:
                ax.axis("off")
            ax.tick_params(labelsize=6)

    fig.suptitle("STEP 1.2 parameter scatter matrix (lower-triangle)", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save_figure(fig, PLOTS_DIR / "parameter_scatter_matrix.png", dpi=150)
    plt.close(fig)


def _plot_feature_scatter_matrix(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    sample_max_rows: int,
    random_seed: int,
    rate_hist_suppressed_count: int = 0,
) -> None:
    cols = [c for c in feature_cols if c == RATE_HIST_PLACEHOLDER_COL or c in df.columns]
    if len(cols) < 2:
        return

    plot_df = df
    if sample_max_rows > 0 and len(plot_df) > sample_max_rows:
        plot_df = plot_df.sample(n=sample_max_rows, random_state=random_seed)

    n = len(cols)
    if n > 40 and len(plot_df) > 700:
        plot_df = plot_df.sample(n=700, random_state=random_seed)
    elif n > 24 and len(plot_df) > 1200:
        plot_df = plot_df.sample(n=1200, random_state=random_seed)

    if n <= 12:
        cell = 2.4
    elif n <= 24:
        cell = 1.5
    elif n <= 40:
        cell = 1.0
    else:
        cell = 0.78

    fig, axes = plt.subplots(n, n, figsize=(cell * n, cell * n), squeeze=False)
    diag_bins = 28 if n <= 24 else 16
    point_size = 7 if n <= 24 else 3

    for i, cy in enumerate(cols):
        for j, cx in enumerate(cols):
            ax = axes[i, j]
            x_is_placeholder = (cx == RATE_HIST_PLACEHOLDER_COL)
            y_is_placeholder = (cy == RATE_HIST_PLACEHOLDER_COL)
            if i == j:
                if x_is_placeholder:
                    ax.set_facecolor("#F2F2F2")
                    msg = "Rate-histogram block\nsuppressed for display"
                    if rate_hist_suppressed_count > 0:
                        msg = (
                            f"Rate-histogram block suppressed\n"
                            f"({rate_hist_suppressed_count} columns)"
                        )
                    ax.text(
                        0.5,
                        0.5,
                        msg,
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        transform=ax.transAxes,
                    )
                    if i == n - 1:
                        ax.set_xlabel(RATE_HIST_PLACEHOLDER_LABEL, fontsize=5.5)
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel("Rows", fontsize=5.5)
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=4.8, length=1.5)
                    ax.grid(True, alpha=0.12)
                    continue

                vals = pd.to_numeric(plot_df[cx], errors="coerce").dropna()
                if vals.empty:
                    ax.set_visible(False)
                    continue
                ax.hist(vals.to_numpy(dtype=float), bins=diag_bins, color="#4C78A8", alpha=0.70)
                if i == n - 1:
                    ax.set_xlabel(cx, fontsize=5.5)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel("Rows", fontsize=5.5)
                else:
                    ax.set_yticklabels([])
                ax.grid(True, alpha=0.15)
            elif i > j:
                if x_is_placeholder or y_is_placeholder:
                    ax.set_facecolor("#F7F7F7")
                    if i == n - 1:
                        ax.set_xlabel(
                            RATE_HIST_PLACEHOLDER_LABEL if x_is_placeholder else cx,
                            fontsize=5.5,
                        )
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel(
                            RATE_HIST_PLACEHOLDER_LABEL if y_is_placeholder else cy,
                            fontsize=5.5,
                        )
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=4.8, length=1.5)
                    ax.grid(True, alpha=0.10)
                    continue

                x = pd.to_numeric(plot_df[cx], errors="coerce")
                y = pd.to_numeric(plot_df[cy], errors="coerce")
                m = x.notna() & y.notna()
                if bool(m.any()):
                    ax.scatter(
                        x[m].to_numpy(dtype=float),
                        y[m].to_numpy(dtype=float),
                        s=point_size,
                        alpha=0.22,
                        color="#4C78A8",
                        edgecolors="none",
                        rasterized=True,
                    )
                if i == n - 1:
                    ax.set_xlabel(cx, fontsize=5.5)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(cy, fontsize=5.5)
                else:
                    ax.set_yticklabels([])
                ax.grid(True, alpha=0.15)
            else:
                ax.axis("off")
            ax.tick_params(labelsize=4.8, length=1.5)

    fig.suptitle("STEP 1.2 feature-space scatter matrix (lower-triangle)", y=0.998, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    _save_figure(fig, PLOTS_DIR / "feature_scatter_matrix.png", dpi=140)
    plt.close(fig)


def _plot_feature_space_regions(
    df: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]],
    zoom_quantiles: tuple[float, float],
    sample_max_rows: int,
    random_seed: int,
) -> None:
    valid_pairs = [(x, y) for x, y in pairs if x in df.columns and y in df.columns]
    if not valid_pairs:
        return

    plot_df = df
    if sample_max_rows > 0 and len(plot_df) > sample_max_rows:
        plot_df = plot_df.sample(n=sample_max_rows, random_state=random_seed)

    n_rows = len(valid_pairs)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(12.0, 3.8 * n_rows),
        squeeze=False,
    )
    q_low, q_high = zoom_quantiles
    for r, (x_col, y_col) in enumerate(valid_pairs):
        x = pd.to_numeric(plot_df[x_col], errors="coerce")
        y = pd.to_numeric(plot_df[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        if not bool(mask.any()):
            axes[r, 0].set_visible(False)
            axes[r, 1].set_visible(False)
            continue
        xv = x.loc[mask].to_numpy(dtype=float)
        yv = y.loc[mask].to_numpy(dtype=float)

        ax_full = axes[r, 0]
        ax_full.hexbin(xv, yv, gridsize=55, mincnt=1, cmap="viridis")
        ax_full.set_xlabel(x_col)
        ax_full.set_ylabel(y_col)
        ax_full.set_title("Full range")
        ax_full.grid(True, alpha=0.2)

        ax_zoom = axes[r, 1]
        x_q0 = float(np.nanquantile(xv, q_low))
        x_q1 = float(np.nanquantile(xv, q_high))
        y_q0 = float(np.nanquantile(yv, q_low))
        y_q1 = float(np.nanquantile(yv, q_high))
        zmask = (xv >= x_q0) & (xv <= x_q1) & (yv >= y_q0) & (yv <= y_q1)
        if bool(np.any(zmask)):
            zx = xv[zmask]
            zy = yv[zmask]
        else:
            zx = xv
            zy = yv
        ax_zoom.hexbin(zx, zy, gridsize=45, mincnt=1, cmap="viridis")
        ax_zoom.set_xlabel(x_col)
        ax_zoom.set_ylabel(y_col)
        ax_zoom.set_title(f"Central region [{q_low:.0%}, {q_high:.0%}]")
        ax_zoom.grid(True, alpha=0.2)

    fig.suptitle("STEP 1.2 feature-space regions", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save_figure(fig, PLOTS_DIR / "feature_space_regions.png", dpi=150)
    plt.close(fig)


def _plot_global_rate_prefix_overview(
    df: pd.DataFrame,
    *,
    rate_col_by_prefix: dict[str, str],
) -> None:
    if not rate_col_by_prefix:
        return
    cols = [rate_col_by_prefix[p] for p in sorted(rate_col_by_prefix.keys()) if rate_col_by_prefix[p] in df.columns]
    if not cols:
        return
    sample_cols = cols[:8]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    values = [pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float) for c in sample_cols]
    labels = [c.replace("events_per_second_global_rate_", "") for c in sample_cols]
    non_empty_vals = [v for v in values if v.size > 0]
    if not non_empty_vals:
        plt.close(fig)
        return
    ax.boxplot(
        [v if v.size > 0 else np.asarray([np.nan]) for v in values],
        tick_labels=labels,
        showfliers=False,
    )
    ax.set_ylabel("Global rate [Hz]")
    ax.set_title("Per-prefix global-rate overview (STEP 1.2 transform)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, PLOTS_DIR / "global_rate_prefix_overview.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 1.2: Transform expanded feature space."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-csv", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    _clear_plots_dir()
    cfg_12 = config.get("step_1_2", {}) if isinstance(config, dict) else {}
    cfg_14 = config.get("step_1_4", {}) if isinstance(config, dict) else {}
    preferred_prefixes_t = tuple(CANONICAL_PREFIX_PRIORITY)
    requested_keep_patterns = _normalize_requested_columns(
        cfg_12.get("transform_keep_columns")
    )
    if not requested_keep_patterns:
        log.error(
            "STEP 1.2 requires step_1_2.transform_keep_columns in config (list of column names/patterns)."
        )
        return 1

    try:
        sample_max_rows = int(
            cfg_12.get(
                "parameter_matrix_plot_sample_max_rows",
                cfg_12.get("feature_space_plot_sample_max_rows", 25000),
            )
        )
    except (TypeError, ValueError):
        sample_max_rows = 25000
    sample_max_rows = max(0, sample_max_rows)
    try:
        plot_seed = int(
            cfg_12.get(
                "parameter_matrix_plot_random_seed",
                cfg_12.get("feature_space_plot_random_seed", 1234),
            )
        )
    except (TypeError, ValueError):
        plot_seed = 1234
    include_rate_histogram_in_feature_matrix = str(
        cfg_12.get("feature_matrix_plot_include_rate_histogram", True)
    ).strip().lower() not in {"0", "false", "no", "off"}

    input_path = Path(args.input_csv).expanduser() if args.input_csv else DEFAULT_INPUT
    if not input_path.exists():
        log.error("Input CSV not found: %s", input_path)
        return 1

    log.info("Loading expanded feature-space data: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    log.info("  Rows loaded: %d, columns=%d", len(df), len(df.columns))

    try:
        min_eff_sim = float(cfg_12.get("min_simulated_efficiency", 0.3))
    except (TypeError, ValueError):
        min_eff_sim = 0.3

    out, rate_col_by_prefix, tt_cols_by_prefix = _build_prefix_global_rate_columns(df)
    out, added_standard_rate_cols = _ensure_standard_task_prefix_rate_columns(
        out,
        rate_col_by_prefix=rate_col_by_prefix,
    )
    out, canonical_source_counts = _select_canonical_global_rate(
        out,
        rate_col_by_prefix=rate_col_by_prefix,
        preferred_prefixes=preferred_prefixes_t,
    )
    out, empirical_selected_prefix, empirical_used_prefixes = _compute_empirical_efficiencies(
        out,
        preferred_prefixes=preferred_prefixes_t,
    )
    out, helper_count = _add_derived_physics_helper_columns(
        out,
    )

    # Filter: keep only rows where all simulated efficiencies exceed a threshold.
    eff_sim_cols = [c for c in ("eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4") if c in out.columns]
    if eff_sim_cols and min_eff_sim > 0.0:
        eff_vals = out[eff_sim_cols].apply(pd.to_numeric, errors="coerce")
        keep_mask = (eff_vals >= min_eff_sim).all(axis=1)
        n_before = len(out)
        out = out.loc[keep_mask].reset_index(drop=True)
        log.info(
            "Simulated-efficiency filter (>= %.2f on %s): kept %d / %d rows (removed %d).",
            min_eff_sim, eff_sim_cols, len(out), n_before, n_before - len(out),
        )

    curated_cols, unmatched_keep_patterns = _resolve_configured_keep_columns(
        out,
        requested_patterns=requested_keep_patterns,
    )
    if unmatched_keep_patterns:
        log.warning(
            "Configured transform_keep_columns patterns without matches: %s",
            unmatched_keep_patterns,
        )
    if not curated_cols:
        log.error("No columns selected by step_1_2.transform_keep_columns.")
        return 1
    out = out[curated_cols].copy()

    # Keep only the highest-priority tt prefix that is present after curation.
    # Consider both individual breakdown columns and per-prefix global-rate summaries.
    available_tt_prefixes: set[str] = {
        p for p, cols in tt_cols_by_prefix.items()
        if any(c in out.columns for c in cols)
    }
    for _prefix, _rate_col in rate_col_by_prefix.items():
        if _rate_col in out.columns:
            vals = pd.to_numeric(out[_rate_col], errors="coerce")
            if vals.notna().any():
                available_tt_prefixes.add(_prefix)
    best_tt_prefix = _select_best_tt_prefix(
        available_tt_prefixes, priority=preferred_prefixes_t
    )
    out, dropped_tt_cols = _drop_non_best_tt_columns(
        out,
        best_prefix=best_tt_prefix,
        tt_cols_by_prefix=tt_cols_by_prefix,
        rate_col_by_prefix=rate_col_by_prefix,
    )
    if dropped_tt_cols:
        log.info(
            "Kept tt prefix '%s'; dropped %d tt-rate columns from other prefixes: %s%s",
            best_tt_prefix,
            len(dropped_tt_cols),
            dropped_tt_cols[:8],
            " ..." if len(dropped_tt_cols) > 8 else "",
        )

    parameter_matrix_cols, unmatched_matrix_patterns, parameter_matrix_source = _resolve_parameter_matrix_columns(
        out,
        cfg_12=cfg_12,
        cfg_14=cfg_14,
        parameter_space_path=DEFAULT_STEP11_PARAMETER_SPACE,
    )
    if unmatched_matrix_patterns:
        log.warning(
            "Configured parameter_matrix_columns patterns without matches: %s",
            unmatched_matrix_patterns,
        )
    feature_matrix_cols, unmatched_feature_matrix_patterns, feature_matrix_source = _resolve_feature_matrix_columns(
        out,
        cfg_12=cfg_12,
        config=config,
        parameter_cols=parameter_matrix_cols,
    )
    (
        feature_matrix_cols_plot,
        feature_matrix_rate_hist_cols,
        feature_matrix_rate_hist_placeholder_added,
    ) = _resolve_feature_matrix_plot_columns(
        feature_matrix_cols,
        include_rate_histogram=include_rate_histogram_in_feature_matrix,
    )
    if unmatched_feature_matrix_patterns:
        log.warning(
            "Configured feature_matrix_columns entries without matches: %s",
            unmatched_feature_matrix_patterns,
        )

    _plot_parameter_scatter_matrix(
        out,
        parameter_cols=parameter_matrix_cols,
        sample_max_rows=sample_max_rows,
        random_seed=plot_seed,
    )
    _plot_feature_scatter_matrix(
        out,
        feature_cols=feature_matrix_cols_plot,
        sample_max_rows=sample_max_rows,
        random_seed=plot_seed,
        rate_hist_suppressed_count=(
            0 if include_rate_histogram_in_feature_matrix else len(feature_matrix_rate_hist_cols)
        ),
    )

    out_csv = FILES_DIR / "transformed_feature_space.csv"
    out.to_csv(out_csv, index=False)
    log.info("Wrote transformed feature space: %s (%d rows, %d columns)", out_csv, len(out), len(out.columns))

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(out_csv),
        "n_rows": int(len(out)),
        "n_columns": int(len(out.columns)),
        "min_simulated_efficiency": float(min_eff_sim),
        "preferred_prefix_order": list(preferred_prefixes_t),
        "empirical_efficiency_selected_prefix": empirical_selected_prefix,
        "empirical_efficiency_used_prefixes": empirical_used_prefixes,
        "global_rate_prefix_columns": rate_col_by_prefix,
        "standard_task_rate_columns_forced": added_standard_rate_cols,
        "tt_rate_columns_by_prefix": tt_cols_by_prefix,
        "canonical_global_rate_source_counts": canonical_source_counts,
        "best_tt_prefix_selected": best_tt_prefix,
        "dropped_tt_columns_count": int(len(dropped_tt_cols)),
        "dropped_tt_columns": dropped_tt_cols,
        "has_canonical_global_rate": bool("events_per_second_global_rate" in out.columns),
        "derived_helper_column_count_added": int(helper_count),
        "transform_keep_columns_requested": requested_keep_patterns,
        "transform_keep_columns_unmatched": unmatched_keep_patterns,
        "parameter_matrix_columns_used": parameter_matrix_cols,
        "parameter_matrix_columns_unmatched": unmatched_matrix_patterns,
        "parameter_matrix_columns_source": parameter_matrix_source,
        "feature_matrix_columns_used": feature_matrix_cols,
        "feature_matrix_columns_plotted": feature_matrix_cols_plot,
        "feature_matrix_columns_unmatched": unmatched_feature_matrix_patterns,
        "feature_matrix_columns_source": feature_matrix_source,
        "feature_matrix_plot_include_rate_histogram": bool(include_rate_histogram_in_feature_matrix),
        "feature_matrix_rate_histogram_columns_total": int(len(feature_matrix_rate_hist_cols)),
        "feature_matrix_rate_histogram_columns_suppressed": int(
            0 if include_rate_histogram_in_feature_matrix else len(feature_matrix_rate_hist_cols)
        ),
        "feature_matrix_rate_histogram_placeholder_added": bool(feature_matrix_rate_hist_placeholder_added),
        "parameter_matrix_plot_sample_max_rows": int(sample_max_rows),
        "parameter_matrix_plot_random_seed": int(plot_seed),
        "curated_feature_columns": curated_cols,
        "curated_feature_count": int(len(curated_cols)),
    }
    summary_path = FILES_DIR / "transform_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote transform summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
