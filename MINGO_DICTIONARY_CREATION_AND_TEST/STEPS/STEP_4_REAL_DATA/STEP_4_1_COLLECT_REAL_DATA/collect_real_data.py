#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_4_REAL_DATA/STEP_4_1_COLLECT_REAL_DATA/collect_real_data.py
Purpose: STEP 4.1 - Collect real metadata for inference.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MINGO_DICTIONARY_CREATION_AND_TEST/STEPS/STEP_4_REAL_DATA/STEP_4_1_COLLECT_REAL_DATA/collect_real_data.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# -- Paths --------------------------------------------------------------------
STEP_DIR = Path(__file__).resolve().parent
# Support both layouts:
#   - <pipeline>/STEP_4_REAL_DATA/STEP_4_1_COLLECT_REAL_DATA
#   - <pipeline>/STEPS/STEP_4_REAL_DATA/STEP_4_1_COLLECT_REAL_DATA
if STEP_DIR.parents[2].name == "STEPS":
    PIPELINE_DIR = STEP_DIR.parents[3]
else:
    PIPELINE_DIR = STEP_DIR.parents[2]
REPO_ROOT = PIPELINE_DIR.parent
DEFAULT_CONFIG = PIPELINE_DIR / "config_method.json"
if (PIPELINE_DIR / "STEP_1_SETUP").exists():
    STEP_ROOT = PIPELINE_DIR
else:
    STEP_ROOT = PIPELINE_DIR / "STEPS"
DEFAULT_DICTIONARY = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "dictionary.csv"
)
DEFAULT_STEP14_SELECTED_FEATURES = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_4_ENSURE_CONTINUITY_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "selected_feature_columns.json"
)
DEFAULT_BUILD_SUMMARY = (
    STEP_ROOT
    / "STEP_1_SETUP"
    / "STEP_1_3_BUILD_DICTIONARY"
    / "OUTPUTS"
    / "FILES"
    / "build_summary.json"
)
ONLINE_RUN_DICTIONARY_ROOT = (
    REPO_ROOT
    / "MASTER"
    / "CONFIG_FILES"
    / "STAGE_0"
    / "NEW_FILES"
    / "ONLINE_RUN_DICTIONARY"
)

FILES_DIR = STEP_DIR / "OUTPUTS" / "FILES"
PLOTS_DIR = STEP_DIR / "OUTPUTS" / "PLOTS"
FILES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_EVENT_HIST = PLOTS_DIR / "STEP_4_1_1_event_count_histogram.png"
PLOT_FEATURE_MATRIX = PLOTS_DIR / "STEP_4_1_2_feature_scatter_matrix_real_vs_dictionary.png"
PLOT_COMPARE_COLS = PLOTS_DIR / "STEP_4_1_3_comparison_columns_and_global_rate.png"
PLOT_TT_BREAKDOWN = PLOTS_DIR / "STEP_4_1_4_tt_rate_breakdown.png"

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
_FILE_TS_RE = re.compile(r"(\d{11})$")
RATE_HIST_BIN_RE = re.compile(r"^events_per_second_(?P<bin>\d+)_rate_hz")

logging.basicConfig(format="[%(levelname)s] STEP_4.1 - %(message)s", level=logging.INFO)
log = logging.getLogger("STEP_4.1")

MODULES_DIR = STEP_ROOT / "MODULES"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))
try:
    from efficiency_fit_utils import (  # noqa: E402
        POLY_CORRECTED_EFFPROD_COL_TEMPLATE,
        append_polynomial_corrected_efficiency_columns,
        load_efficiency_fit_models,
    )
except Exception as exc:
    log.error("Could not import efficiency_fit_utils from %s: %s", MODULES_DIR, exc)
    raise

STEP12_TRANSFORM_DIR = STEP_ROOT / "STEP_1_SETUP" / "STEP_1_2_TRANSFORM_FEATURE_SPACE"
if str(STEP12_TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(STEP12_TRANSFORM_DIR))
try:
    from transform_feature_space import (  # noqa: E402
        CANONICAL_PREFIX_PRIORITY as STEP12_CANONICAL_PREFIX_PRIORITY,
        RATE_HIST_PLACEHOLDER_COL as STEP12_RATE_HIST_PLACEHOLDER_COL,
        RATE_HIST_PLACEHOLDER_LABEL as STEP12_RATE_HIST_PLACEHOLDER_LABEL,
        _add_derived_physics_helper_columns as _step12_add_derived_physics_helper_columns,
        _build_prefix_global_rate_columns as _step12_build_prefix_global_rate_columns,
        _compute_empirical_efficiencies as _step12_compute_empirical_efficiencies,
        _drop_non_best_tt_columns as _step12_drop_non_best_tt_columns,
        _ensure_standard_task_prefix_rate_columns as _step12_ensure_standard_task_prefix_rate_columns,
        _normalize_requested_columns as _step12_normalize_requested_columns,
        _resolve_feature_matrix_plot_columns as _step12_resolve_feature_matrix_plot_columns,
        _resolve_configured_keep_columns as _step12_resolve_configured_keep_columns,
        _select_best_tt_prefix as _step12_select_best_tt_prefix,
        _select_canonical_global_rate as _step12_select_canonical_global_rate,
    )
except Exception as exc:
    log.error(
        "Could not import STEP 1.2 transform helpers from %s: %s",
        STEP12_TRANSFORM_DIR,
        exc,
    )
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


def _preferred_feature_prefixes_for_task_ids(task_ids: list[int]) -> list[str]:
    """Preferred TT-rate prefixes for auto feature/comparison selection."""
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


def _safe_station_id(raw: object, default: int = 0) -> int:
    """Parse station selector from int-like or MINGOxx string."""
    if raw in (None, "", "null", "None"):
        return int(default)
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return int(default)
        m = re.fullmatch(r"(?i)MINGO(\d{1,2})", stripped)
        if m is not None:
            return int(m.group(1))
        try:
            return int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Invalid station selector '{raw}'. Use int (e.g. 1) or string like 'MINGO01'."
            ) from exc
    raise ValueError(
        f"Invalid station selector type: {type(raw).__name__}. "
        "Use int (e.g. 1) or string like 'MINGO01'."
    )


def _task_metadata_dir(station_id: int, task_id: int) -> Path:
    station = f"MINGO{station_id:02d}"
    return (
        REPO_ROOT
        / "STATIONS"
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
    )


def _task_specific_metadata_path(station_id: int, task_id: int) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_specific.csv"


def _task_trigger_type_metadata_path(station_id: int, task_id: int) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_trigger_type.csv"


def _task_rate_histogram_metadata_path(station_id: int, task_id: int) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_rate_histogram.csv"


def _normalize_basename_for_time(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    stem = Path(text).stem.strip().lower()
    # Historical real-data prefix variant.
    if stem.startswith("mini"):
        return "mi01" + stem[4:]
    return stem


def _parse_filename_base_ts(value: object) -> pd.Timestamp:
    """Parse file time encoded in filename_base (mi0XYYDDDHHMMSS)."""
    base = _normalize_basename_for_time(value)
    match = _FILE_TS_RE.search(base)
    if match is None:
        return pd.NaT

    stamp = match.group(1)
    try:
        yy = int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
    except ValueError:
        return pd.NaT

    if not (1 <= day_of_year <= 366):
        return pd.NaT
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return pd.NaT

    try:
        dt = datetime(2000 + yy, 1, 1) + timedelta(
            days=day_of_year - 1,
            hours=hour,
            minutes=minute,
            seconds=second,
        )
    except ValueError:
        return pd.NaT
    return pd.Timestamp(dt, tz="UTC")


def _parse_ts(series: pd.Series) -> pd.Series:
    """Parse execution timestamps to timezone-aware UTC datetimes."""
    s = series.astype("string")
    parsed = pd.to_datetime(s, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed_fallback = pd.to_datetime(s[missing], errors="coerce", utc=True)
        parsed.loc[missing] = parsed_fallback
    return parsed


def _parse_bound(value: object) -> pd.Timestamp | None:
    if value in (None, "", "null", "None"):
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _coalesce(primary: object, fallback: object) -> object:
    if primary in (None, "", "null", "None"):
        return fallback
    return primary


def _safe_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            return default
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


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


def _load_step13_efficiency_fit(path: Path) -> tuple[dict[int, dict[str, object]], dict]:
    """Load STEP 1.3 empirical-efficiency polynomial fit metadata."""
    fit_models, fit_status, payload = load_efficiency_fit_models(path)
    efficiency_fit = payload.get("efficiency_fit", {}) if isinstance(payload, dict) else {}
    info: dict[str, object] = {
        "path": str(path),
        "exists": bool(path.exists()),
        "status": fit_status,
        "planes_loaded": sorted(int(plane) for plane in fit_models.keys()),
        "polynomial_order_requested": (
            efficiency_fit.get("polynomial_order_requested")
            if isinstance(efficiency_fit, dict)
            else None
        ),
    }
    return fit_models, info


def _append_step13_polynomial_efficiency_products(
    df: pd.DataFrame,
    *,
    build_summary_path: Path,
) -> dict[str, object]:
    """Append STEP 1.3 polynomial-corrected efficiency columns for TT diagnostics."""
    fit_models, load_info = _load_step13_efficiency_fit(build_summary_path)
    append_info = append_polynomial_corrected_efficiency_columns(df, fit_models)
    info = dict(load_info)
    info["application"] = append_info
    if append_info.get("status") == "ok":
        log.info(
            "Applied STEP 1.3 polynomial efficiency correction from %s: planes=%s, products=%s",
            build_summary_path,
            append_info.get("planes_applied", []),
            append_info.get("efficiency_product_columns_created", []),
        )
    else:
        log.warning(
            "STEP 1.3 polynomial efficiency correction unavailable for STEP 4.1 TT plot "
            "(summary=%s, load_status=%s, apply_status=%s).",
            build_summary_path,
            load_info.get("status"),
            append_info.get("status"),
        )
    return info


def _canonical_z_tuple(values: object) -> tuple[float, float, float, float] | None:
    try:
        seq = list(values)
    except TypeError:
        return None
    if len(seq) != 4:
        return None
    out: list[float] = []
    for v in seq:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        out.append(float(np.round(f, 6)))
    return (out[0], out[1], out[2], out[3])


def _extract_dictionary_z_tuples(dict_df: pd.DataFrame) -> set[tuple[float, float, float, float]]:
    z_cols = ["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]
    if any(c not in dict_df.columns for c in z_cols):
        return set()
    z_numeric = dict_df[z_cols].apply(pd.to_numeric, errors="coerce")
    out: set[tuple[float, float, float, float]] = set()
    for row in z_numeric.itertuples(index=False, name=None):
        zt = _canonical_z_tuple(row)
        if zt is not None:
            out.add(zt)
    return out


def _online_run_dictionary_path(station_id: int) -> Path:
    suffix = f"{int(station_id):02d}"
    candidates = [
        ONLINE_RUN_DICTIONARY_ROOT / f"STATION_{int(station_id)}" / f"input_file_mingo{suffix}.csv",
        ONLINE_RUN_DICTIONARY_ROOT / f"STATION_{suffix}" / f"input_file_mingo{suffix}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    globbed = sorted(ONLINE_RUN_DICTIONARY_ROOT.glob(f"STATION_*/input_file_mingo{suffix}.csv"))
    if globbed:
        return globbed[0]
    raise FileNotFoundError(
        f"ONLINE_RUN_DICTIONARY CSV not found for station {station_id} "
        f"under {ONLINE_RUN_DICTIONARY_ROOT}"
    )


def _load_online_run_schedule(station_id: int) -> tuple[pd.DataFrame, Path]:
    """Load run-configuration schedule with z positions from ONLINE_RUN_DICTIONARY."""
    path = _online_run_dictionary_path(station_id)
    raw = pd.read_csv(path, header=[0, 1], low_memory=False)

    if isinstance(raw.columns, pd.MultiIndex):
        flat_cols: list[str] = []
        for col in raw.columns:
            top = str(col[0]).strip()
            sub = str(col[1]).strip()
            if sub and not sub.lower().startswith("unnamed"):
                flat_cols.append(sub)
            else:
                flat_cols.append(top)
        df = raw.copy()
        df.columns = flat_cols
    else:
        df = raw.copy()

    col_by_lower = {str(c).strip().lower(): c for c in df.columns}

    def _pick(*names: str) -> str | None:
        for name in names:
            c = col_by_lower.get(name.lower())
            if c is not None:
                return str(c)
        return None

    station_col = _pick("station", "detector")
    start_col = _pick("start", "date_start")
    end_col = _pick("end", "date_end")
    p1_col = _pick("p1")
    p2_col = _pick("p2")
    p3_col = _pick("p3")
    p4_col = _pick("p4")
    conf_col = _pick("conf", "conf number")

    required = [start_col, p1_col, p2_col, p3_col, p4_col]
    if any(c is None for c in required):
        raise ValueError(
            f"Could not parse ONLINE_RUN_DICTIONARY schema in {path}. "
            f"Columns found: {list(df.columns)}"
        )

    work = pd.DataFrame(index=df.index)
    if station_col is not None:
        station_series = pd.to_numeric(df[station_col], errors="coerce")
        work["station"] = station_series
        work = work.loc[station_series == int(station_id)].copy()
    else:
        work["station"] = int(station_id)

    work["start_utc"] = pd.to_datetime(df[start_col], errors="coerce", utc=True)
    if end_col is not None:
        work["end_utc"] = pd.to_datetime(df[end_col], errors="coerce", utc=True)
    else:
        work["end_utc"] = pd.NaT

    work["z_plane_1"] = pd.to_numeric(df[p1_col], errors="coerce")
    work["z_plane_2"] = pd.to_numeric(df[p2_col], errors="coerce")
    work["z_plane_3"] = pd.to_numeric(df[p3_col], errors="coerce")
    work["z_plane_4"] = pd.to_numeric(df[p4_col], errors="coerce")
    if conf_col is not None:
        work["conf"] = pd.to_numeric(df[conf_col], errors="coerce")
    else:
        work["conf"] = np.nan

    work = work.dropna(subset=["start_utc", "z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]).copy()
    work["z_tuple"] = work[["z_plane_1", "z_plane_2", "z_plane_3", "z_plane_4"]].apply(
        lambda r: _canonical_z_tuple(r.values),
        axis=1,
    )
    work = work.loc[work["z_tuple"].notna()].copy()
    work = work.sort_values(["start_utc", "end_utc"], kind="mergesort").reset_index(drop=True)
    return (work, path)


def _select_schedule_rows_for_window(
    schedule_df: pd.DataFrame,
    *,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df.copy()
    keep = pd.Series(True, index=schedule_df.index)
    end_fallback = pd.Timestamp("2100-01-01", tz="UTC")
    if date_from is not None:
        keep &= schedule_df["end_utc"].fillna(end_fallback) >= date_from
    if date_to is not None:
        keep &= schedule_df["start_utc"] <= date_to
    return schedule_df.loc[keep].copy()


def _online_z_tuple_for_timestamp(
    ts: pd.Timestamp,
    schedule_df: pd.DataFrame,
) -> tuple[float, float, float, float] | None:
    if pd.isna(ts) or schedule_df.empty:
        return None
    # Primary rule: [start, end)
    keep = (schedule_df["start_utc"] <= ts) & (
        schedule_df["end_utc"].isna() | (ts < schedule_df["end_utc"])
    )
    candidates = schedule_df.loc[keep]
    # Boundary fallback: allow exact equality at end.
    if candidates.empty:
        keep2 = (schedule_df["start_utc"] <= ts) & (
            schedule_df["end_utc"].notna() & (ts <= schedule_df["end_utc"])
        )
        candidates = schedule_df.loc[keep2]
    if candidates.empty:
        return None
    row = candidates.sort_values("start_utc", kind="mergesort").iloc[-1]
    return row["z_tuple"]


def _extract_tt_parts(col: str) -> tuple[str, str] | None:
    match = re.match(r"^(?P<prefix>.+?)_tt_(?P<rest>.+)_rate_hz$", col)
    if match is None:
        return None
    rest = re.sub(r"\.0$", "", match.group("rest"))
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


def _choose_best_col(columns: list[str]) -> str:
    scored: list[tuple[int, str]] = []
    for col in columns:
        parts = _extract_tt_parts(col)
        rank = _prefix_rank(parts[0]) if parts is not None else 999
        scored.append((rank, col))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][1]


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
    candidates: list[str] = []
    preferred_clean = str(preferred).strip()
    if preferred_clean:
        candidates.append(preferred_clean)
    candidates.extend(
        [
            "events_per_second_global_rate",
            "global_rate_hz",
            "global_rate_hz_mean",
        ]
    )
    seen = set(candidates)
    for c in df.columns:
        cl = c.lower()
        if (
            "global_rate" in cl and ("hz" in cl or cl.endswith("_rate"))
        ) and c not in seen:
            candidates.append(c)
            seen.add(c)
    for c in candidates:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            return c
    for c in candidates:
        if c in df.columns:
            return c
    return _derive_global_rate_from_tt_sum(df, target_col="events_per_second_global_rate")


def _resolve_comparison_preview(
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame | None,
    preferred_prefixes: list[str] | None = None,
) -> tuple[str, list[str], list[dict[str, str]]]:
    """Preview columns likely to be used in STEP 4.2 comparison."""
    real_tt = _tt_rate_columns(real_df)
    mapping_preview: list[dict[str, str]] = []
    if not real_tt:
        return ("no_tt_rate_columns", [], mapping_preview)

    if dict_df is not None and not dict_df.empty:
        dict_tt = _tt_rate_columns(dict_df)
        prefixes = [str(p).strip() for p in (preferred_prefixes or []) if str(p).strip()]
        if not prefixes:
            prefixes = ["raw", "clean", "cal", "list", "fit", "corr", "definitive"]
        for prefix in prefixes:
            common = sorted(
                [c for c in dict_tt if c.startswith(f"{prefix}_tt_") and c in set(real_tt)]
            )
            if common:
                return (f"direct_prefix:{prefix}", common, mapping_preview)

        exact_common = sorted(set(dict_tt) & set(real_tt))
        if exact_common:
            return ("direct_exact", exact_common, mapping_preview)

        dict_key_to_cols: dict[str, list[str]] = {}
        real_key_to_cols: dict[str, list[str]] = {}
        for col in dict_tt:
            parts = _extract_tt_parts(col)
            if parts is None:
                continue
            dict_key_to_cols.setdefault(parts[1], []).append(col)
        for col in real_tt:
            parts = _extract_tt_parts(col)
            if parts is None:
                continue
            real_key_to_cols.setdefault(parts[1], []).append(col)

        common_keys = sorted(set(dict_key_to_cols) & set(real_key_to_cols))
        if common_keys:
            selected_real: list[str] = []
            for key in common_keys:
                dcol = _choose_best_col(dict_key_to_cols[key])
                rcol = _choose_best_col(real_key_to_cols[key])
                selected_real.append(rcol)
                mapping_preview.append(
                    {
                        "dictionary_column": dcol,
                        "real_column": rcol,
                        "tt_key": key,
                    }
                )
            selected_real = sorted(set(selected_real), key=lambda c: (_prefix_rank(_extract_tt_parts(c)[0]) if _extract_tt_parts(c) else 999, c))
            return ("aligned_by_tt_key", selected_real, mapping_preview)

    selected_real = sorted(
        real_tt,
        key=lambda c: (_prefix_rank(_extract_tt_parts(c)[0]) if _extract_tt_parts(c) else 999, c),
    )
    return ("real_only_fallback", selected_real, mapping_preview)


def _aggregate_latest(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Keep only the latest execution per filename_base."""
    work = df.copy()
    if timestamp_col in work.columns:
        ts = _parse_ts(work[timestamp_col])
        work = work.assign(_exec_dt=ts)
        work = work.sort_values(
            ["filename_base", "_exec_dt"], na_position="last", kind="mergesort"
        )
        work = work.groupby("filename_base").tail(1)
        return work.drop(columns=["_exec_dt"])
    return work.groupby("filename_base").tail(1)


def _load_task_metadata_source_csv(
    *,
    task_id: int,
    source_name: str,
    path: Path,
    metadata_agg: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Load one task metadata source using the same equal-level pattern as STEP 1.1."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {source_name} metadata for task {task_id}: {path}"
        )
    log.info("Loading task %d %s metadata: %s", task_id, source_name, path)
    meta_df = pd.read_csv(path, low_memory=False)
    if "filename_base" not in meta_df.columns:
        raise KeyError(
            f"Task {task_id} {source_name} metadata has no 'filename_base' column: {path}"
        )
    if metadata_agg == "latest":
        meta_df = _aggregate_latest(meta_df, timestamp_col=timestamp_col)
    meta_df = meta_df.groupby("filename_base", sort=False).tail(1).reset_index(drop=True)
    log.info(
        "  %s rows (after aggregation): %d, columns=%d",
        source_name,
        len(meta_df),
        len(meta_df.columns),
    )
    return meta_df


def _merge_metadata_sources_equal_level(
    *,
    task_id: int,
    ordered_sources: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict]:
    """Merge task metadata sources at equal level (same architecture as STEP 1.1)."""
    if not ordered_sources:
        raise ValueError(f"No metadata sources were provided for task {task_id}.")
    merged = ordered_sources[0][1].copy()
    info: dict = {
        "task_id": int(task_id),
        "source_names": [name for name, _ in ordered_sources],
        "rows_per_source_before_merge": {
            name: int(len(df)) for name, df in ordered_sources
        },
        "renamed_overlap_columns": {},
    }

    for source_name, source_df in ordered_sources[1:]:
        overlap = sorted(
            set(merged.columns).intersection(set(source_df.columns)) - {"filename_base"}
        )
        renamed: dict[str, str] = {}
        if overlap:
            renamed = {col: f"{source_name}__{col}" for col in overlap}
            source_df = source_df.rename(columns=renamed)
        merged = merged.merge(source_df, on="filename_base", how="inner")
        info["renamed_overlap_columns"][source_name] = renamed
        info[f"rows_after_merge_with_{source_name}"] = int(len(merged))
    return merged, info


def _load_rate_histogram_bins(
    *,
    station_id: int,
    task_id: int,
    metadata_agg: str,
    timestamp_col: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load per-file histogram-bin columns from metadata_rate_histogram."""
    rate_hist_path = _task_rate_histogram_metadata_path(station_id, task_id)
    if not rate_hist_path.exists():
        log.info("Task %d rate_histogram metadata not found: %s", task_id, rate_hist_path)
        return (None, [])

    log.info("Loading task %d rate_histogram metadata: %s", task_id, rate_hist_path)
    rate_hist_df = pd.read_csv(rate_hist_path, low_memory=False)
    if "filename_base" not in rate_hist_df.columns:
        log.warning(
            "Task %d rate_histogram metadata has no 'filename_base'; skipping histogram merge.",
            task_id,
        )
        return (None, [])
    if metadata_agg == "latest":
        rate_hist_df = _aggregate_latest(rate_hist_df, timestamp_col=timestamp_col)

    hist_cols = [
        c for c in rate_hist_df.columns
        if RATE_HIST_BIN_RE.match(str(c)) is not None
    ]
    hist_cols.sort(key=lambda c: int(RATE_HIST_BIN_RE.match(str(c)).group("bin")))
    if not hist_cols:
        log.warning(
            "Task %d rate_histogram metadata has no events_per_second_<bin>_rate_hz columns.",
            task_id,
        )
        return (None, [])

    out = rate_hist_df[["filename_base", *hist_cols]].copy()
    out = out.groupby("filename_base", sort=False).tail(1)
    log.info(
        "Task %d rate_histogram rows (after aggregation): %d with %d bin columns.",
        task_id,
        len(out),
        len(hist_cols),
    )
    return (out, hist_cols)


def _pick_event_count_column(df: pd.DataFrame) -> str | None:
    # Prefer explicit event-count columns for TT=1234; do not use duration columns.
    priority = [
        "clean_tt_1234_count",
        "clean_tt_1234.0_count",
        "raw_tt_1234_count",
        "raw_tt_1234.0_count",
        "cal_tt_1234_count",
        "cal_tt_1234.0_count",
        "list_tt_1234_count",
        "list_tt_1234.0_count",
        "fit_tt_1234_count",
        "fit_tt_1234.0_count",
        "corr_tt_1234_count",
        "corr_tt_1234.0_count",
        "definitive_tt_1234_count",
        "definitive_tt_1234.0_count",
        "n_events",
    ]
    for col in priority:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col

    tt_count_re = re.compile(r"^(raw|clean|cal|list|fit|corr|definitive)_tt_1234(?:\.0)?_count$")
    fallback = sorted([c for c in df.columns if tt_count_re.match(c)])
    for col in fallback:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return col

    generic_tt_count_re = re.compile(r"_tt_1234(?:\.0)?_count$")
    generic = sorted([c for c in df.columns if generic_tt_count_re.search(c)])
    for col in generic:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return col

    for col in ("selected_rows", "requested_rows"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col
    for source in ("trigger_type", "rate_histogram", "specific"):
        for base_col in (
            "selected_rows",
            "requested_rows",
            "generated_events_count",
            "num_events",
            "event_count",
        ):
            col = f"{source}__{base_col}"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.notna().any():
                    return col

    return None


def _pick_duration_seconds_column(df: pd.DataFrame) -> str | None:
    """Pick an acquisition-duration column (seconds) for count derivation."""
    priority = [
        "events_per_second_total_seconds",
        "count_rate_denominator_seconds",
        "duration_seconds",
        "acquisition_seconds",
    ]
    for col in priority:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return col

    pattern = re.compile(r"(duration|denominator|acquisition).*(second|sec)|second|sec", re.IGNORECASE)
    for col in sorted(df.columns):
        if not pattern.search(str(col)):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            return col
    return None


def _resolve_event_count_values(
    df: pd.DataFrame,
    *,
    event_count_column: str = "auto",
) -> tuple[pd.Series, str | None, str]:
    """Resolve per-row event counts from explicit count columns or derived fallback."""
    event_col_cfg = str(event_count_column).strip()
    event_col: str | None
    if event_col_cfg and event_col_cfg.lower() != "auto":
        event_col = event_col_cfg if event_col_cfg in df.columns else None
    else:
        event_col = _pick_event_count_column(df)

    if event_col is not None:
        values = pd.to_numeric(df[event_col], errors="coerce")
        if values.notna().any():
            return (values, event_col, "column")

    # Last-resort fallback: estimate counts from global rate and acquisition duration.
    duration_col = _pick_duration_seconds_column(df)
    global_rate_col = _pick_global_rate_column(df, preferred="events_per_second_global_rate")
    if duration_col is not None and global_rate_col is not None:
        duration = pd.to_numeric(df[duration_col], errors="coerce")
        global_rate = pd.to_numeric(df[global_rate_col], errors="coerce")
        values = duration * global_rate
        if values.notna().any():
            return (values, f"{global_rate_col} * {duration_col}", "derived_rate_x_duration")

    return (
        pd.Series(np.nan, index=df.index, dtype=float),
        (event_col_cfg if event_col_cfg and event_col_cfg.lower() != "auto" else None),
        "none",
    )


def _backfill_efficiency_columns_from_empirical(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Backfill eff_p*/eff_sim_* from eff_empirical_* when sim efficiencies are absent."""
    out = df.copy()
    created = 0
    for idx in (1, 2, 3, 4):
        emp_col = f"eff_empirical_{idx}"
        ep_col = f"eff_p{idx}"
        es_col = f"eff_sim_{idx}"
        if emp_col not in out.columns:
            continue

        emp_vals = pd.to_numeric(out[emp_col], errors="coerce")

        if ep_col not in out.columns:
            out[ep_col] = emp_vals
            created += 1
        else:
            ep_vals = pd.to_numeric(out[ep_col], errors="coerce")
            fill_ep = ep_vals.isna() & emp_vals.notna()
            if bool(fill_ep.any()):
                out.loc[fill_ep, ep_col] = emp_vals.loc[fill_ep]

        ep_vals_now = pd.to_numeric(out[ep_col], errors="coerce")
        if es_col not in out.columns:
            out[es_col] = ep_vals_now
            created += 1
        else:
            es_vals = pd.to_numeric(out[es_col], errors="coerce")
            fill_es = es_vals.isna() & ep_vals_now.notna()
            if bool(fill_es.any()):
                out.loc[fill_es, es_col] = ep_vals_now.loc[fill_es]

    return out, created


def _apply_step12_feature_space_transform(
    df: pd.DataFrame,
    *,
    cfg_12: dict,
    identity_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply STEP 1.2 feature-space transformation using STEP 1.2 definitions."""
    requested_keep_patterns = _step12_normalize_requested_columns(
        cfg_12.get("transform_keep_columns")
    )
    requested_keep_patterns_cfg = list(requested_keep_patterns)
    if (
        identity_column
        and isinstance(identity_column, str)
        and identity_column in df.columns
        and identity_column not in requested_keep_patterns
    ):
        requested_keep_patterns = [*requested_keep_patterns, identity_column]
    if not requested_keep_patterns:
        raise ValueError(
            "STEP 4.1 cannot apply STEP 1.2 transform: "
            "step_1_2.transform_keep_columns is missing/empty."
        )

    preferred_prefixes_t = tuple(STEP12_CANONICAL_PREFIX_PRIORITY)
    out, rate_col_by_prefix, tt_cols_by_prefix = _step12_build_prefix_global_rate_columns(df)
    out, added_standard_rate_cols = _step12_ensure_standard_task_prefix_rate_columns(
        out,
        rate_col_by_prefix=rate_col_by_prefix,
    )
    out, canonical_source_counts = _step12_select_canonical_global_rate(
        out,
        rate_col_by_prefix=rate_col_by_prefix,
        preferred_prefixes=preferred_prefixes_t,
    )
    out, empirical_selected_prefix, empirical_used_prefixes = _step12_compute_empirical_efficiencies(
        out,
        preferred_prefixes=preferred_prefixes_t,
    )
    out, helper_count = _step12_add_derived_physics_helper_columns(out)

    try:
        min_eff_sim = float(cfg_12.get("min_simulated_efficiency", 0.5))
    except (TypeError, ValueError):
        min_eff_sim = 0.5
    try:
        max_eff_spread = float(cfg_12.get("max_simulated_efficiency_spread", 0.15))
    except (TypeError, ValueError):
        max_eff_spread = 0.15

    eff_sim_cols = [c for c in ("eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4") if c in out.columns]
    rows_before_min_eff_filter = int(len(out))
    rows_removed_min_eff_filter = 0
    if eff_sim_cols and min_eff_sim > 0.0:
        eff_vals = out[eff_sim_cols].apply(pd.to_numeric, errors="coerce")
        keep_mask = (eff_vals >= min_eff_sim).all(axis=1)
        rows_removed_min_eff_filter = int(np.count_nonzero(~keep_mask))
        out = out.loc[keep_mask].reset_index(drop=True)

    rows_before_spread_filter = int(len(out))
    rows_removed_spread_filter = 0
    if eff_sim_cols and max_eff_spread > 0.0:
        eff_vals = out[eff_sim_cols].apply(pd.to_numeric, errors="coerce")
        spread = eff_vals.max(axis=1) - eff_vals.min(axis=1)
        keep_mask = spread <= max_eff_spread
        rows_removed_spread_filter = int(np.count_nonzero(~keep_mask))
        out = out.loc[keep_mask].reset_index(drop=True)

    # Real-data path: materialize STEP 1 helper columns from empirical efficiencies
    # when eff_p*/eff_sim_* are absent, so feature-space dimensionality stays aligned.
    out, backfilled_eff_cols = _backfill_efficiency_columns_from_empirical(out)
    out, helper_count_post = _step12_add_derived_physics_helper_columns(out)
    helper_count += int(helper_count_post)
    # Real data must not expose unknown simulation-efficiency coordinates.
    nonobservable_eff_cols = [
        c for c in (
            "eff_p1", "eff_p2", "eff_p3", "eff_p4",
            "eff_sim_1", "eff_sim_2", "eff_sim_3", "eff_sim_4",
        )
        if c in out.columns
    ]
    if nonobservable_eff_cols:
        out = out.drop(columns=nonobservable_eff_cols, errors="ignore")

    curated_cols, unmatched_keep_patterns = _step12_resolve_configured_keep_columns(
        out,
        requested_patterns=requested_keep_patterns,
    )
    if unmatched_keep_patterns:
        log.warning(
            "STEP 4.1: STEP 1.2 keep patterns without matches on real data: %s",
            unmatched_keep_patterns,
        )
    if not curated_cols:
        raise ValueError(
            "STEP 4.1 cannot apply STEP 1.2 transform: "
            "no columns matched step_1_2.transform_keep_columns."
        )
    out = out[curated_cols].copy()
    curated_cols_summary = [
        c for c in curated_cols
        if not (identity_column and isinstance(identity_column, str) and c == identity_column)
    ]

    available_tt_prefixes: set[str] = {
        p for p, cols in tt_cols_by_prefix.items()
        if any(c in out.columns for c in cols)
    }
    for prefix, rate_col in rate_col_by_prefix.items():
        if rate_col in out.columns:
            vals = pd.to_numeric(out[rate_col], errors="coerce")
            if vals.notna().any():
                available_tt_prefixes.add(prefix)

    best_tt_prefix = _step12_select_best_tt_prefix(
        available_tt_prefixes,
        priority=preferred_prefixes_t,
    )
    out, dropped_tt_cols = _step12_drop_non_best_tt_columns(
        out,
        best_prefix=best_tt_prefix,
        tt_cols_by_prefix=tt_cols_by_prefix,
        rate_col_by_prefix=rate_col_by_prefix,
    )

    info = {
        "step12_transform_keep_columns_requested": requested_keep_patterns_cfg,
        "step12_transform_keep_columns_unmatched": unmatched_keep_patterns,
        "step12_curated_columns": curated_cols_summary,
        "step12_curated_column_count": int(len(curated_cols_summary)),
        "step12_min_simulated_efficiency": float(min_eff_sim),
        "step12_max_simulated_efficiency_spread": float(max_eff_spread),
        "step12_eff_sim_columns_seen": eff_sim_cols,
        "step12_rows_before_min_eff_filter": rows_before_min_eff_filter,
        "step12_rows_removed_min_eff_filter": rows_removed_min_eff_filter,
        "step12_rows_before_spread_filter": rows_before_spread_filter,
        "step12_rows_removed_spread_filter": rows_removed_spread_filter,
        "step12_global_rate_prefix_columns": rate_col_by_prefix,
        "step12_standard_task_rate_columns_forced": added_standard_rate_cols,
        "step12_tt_rate_columns_by_prefix": tt_cols_by_prefix,
        "step12_canonical_global_rate_source_counts": canonical_source_counts,
        "step12_empirical_efficiency_selected_prefix": empirical_selected_prefix,
        "step12_empirical_efficiency_used_prefixes": empirical_used_prefixes,
        "step12_backfilled_efficiency_columns_count": int(backfilled_eff_cols),
        "step12_removed_nonobservable_eff_columns": nonobservable_eff_cols,
        "step12_derived_helper_column_count_added": int(helper_count),
        "step12_best_tt_prefix_selected": best_tt_prefix,
        "step12_dropped_tt_columns_count": int(len(dropped_tt_cols)),
        "step12_dropped_tt_columns": dropped_tt_cols,
    }
    return out, info


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_event_histogram(
    df: pd.DataFrame,
    *,
    event_count_column: str = "auto",
    min_n_events: float | None = None,
) -> str | None:
    values_full, source_label, _source_mode = _resolve_event_count_values(
        df,
        event_count_column=event_count_column,
    )
    values = values_full.dropna()
    if values.empty:
        _plot_placeholder(
            PLOT_EVENT_HIST,
            "Event count histogram",
            "No valid event-count values found in collected table.",
        )
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=40, alpha=0.5, color="#4C78A8", edgecolor="white", label=f"All rows (n={len(values)})")
    title = f"Event count distribution — {len(values)} rows"
    if min_n_events is not None and np.isfinite(min_n_events):
        cut = float(min_n_events)
        kept = int((values > cut).sum())
        removed = int((values <= cut).sum())
        cut_label = f"{cut:.0f}" if np.isclose(cut, np.round(cut)) else f"{cut:g}"
        ax.axvline(
            cut,
            color="#D62728",
            linestyle="--",
            linewidth=1.8,
            alpha=0.95,
            label=f"Cut: n_events > {cut_label}",
        )
        title = (
            f"Event count distribution — all rows: {len(values)} "
            f"(kept: {kept}, cut: {removed})"
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.92, facecolor="white")

    ax.set_xlabel(f"Event count ({source_label})")
    ax.set_ylabel("Number of files")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_EVENT_HIST, dpi=150)
    plt.close(fig)
    return source_label


def _load_selected_feature_columns(path: Path) -> list[str]:
    """Load ordered feature list (STEP 1.4 selected_feature_columns.json format)."""
    if not path.exists():
        log.warning("Selected feature-columns JSON not found: %s", path)
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse selected feature-columns JSON %s: %s", path, exc)
        return []

    if isinstance(raw, dict):
        values = raw.get("selected_feature_columns", [])
    elif isinstance(raw, list):
        values = raw
    else:
        values = []

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _resolve_feature_matrix_columns(
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    *,
    preferred_cols: list[str],
    include_rate_histogram: bool,
) -> tuple[list[str], list[str], bool]:
    common_cols = set(real_df.columns) & set(dict_df.columns)
    admin_exact = {
        "filename_base",
        "task_id",
        "station_id",
        "n_events",
        "dataset_index",
    }
    admin_contains = (
        "timestamp",
        "execution_",
        "online_z_",
    )
    feature_common = sorted(
        c for c in common_cols
        if (
            c not in admin_exact
            and not any(token in str(c).lower() for token in admin_contains)
        )
    )
    if preferred_cols:
        preferred = [c for c in preferred_cols if c in feature_common]
        preferred_set = set(preferred)
        extras = [c for c in feature_common if c not in preferred_set]
        candidates = preferred + extras
    else:
        candidates = feature_common

    out: list[str] = []
    for col in candidates:
        real_vals = pd.to_numeric(real_df[col], errors="coerce")
        dict_vals = pd.to_numeric(dict_df[col], errors="coerce")
        if not real_vals.notna().any():
            continue
        if not dict_vals.notna().any():
            continue
        out.append(col)
    resolved_cols, hist_cols, suppressed = _step12_resolve_feature_matrix_plot_columns(
        out,
        include_rate_histogram=include_rate_histogram,
    )
    return resolved_cols, hist_cols, suppressed


def _make_feature_scatter_matrix_real_vs_dictionary(
    real_df: pd.DataFrame,
    dict_df: pd.DataFrame,
    *,
    selected_features_path: Path,
    sample_max_rows: int,
    random_seed: int,
    include_rate_histogram: bool,
) -> dict:
    selected_feature_cols = _load_selected_feature_columns(selected_features_path)
    matrix_cols, rate_hist_cols, rate_hist_suppressed = _resolve_feature_matrix_columns(
        real_df,
        dict_df,
        preferred_cols=selected_feature_cols,
        include_rate_histogram=include_rate_histogram,
    )
    source_label = (
        f"selected_feature_columns.json ({selected_features_path})"
        if selected_feature_cols
        else "numeric intersection fallback"
    )

    summary = {
        "path": str(PLOT_FEATURE_MATRIX),
        "selected_features_source": source_label,
        "selected_features_requested_count": int(len(selected_feature_cols)),
        "columns_used_count": int(len(matrix_cols)),
        "columns_used": matrix_cols[:200],
        "feature_matrix_plot_include_rate_histogram": bool(include_rate_histogram),
        "rate_hist_columns_detected_count": int(len(rate_hist_cols)),
        "rate_hist_columns_detected": rate_hist_cols[:200],
        "rate_hist_block_suppressed": bool(rate_hist_suppressed),
        "rows_real_total": int(len(real_df)),
        "rows_dictionary_total": int(len(dict_df)),
        "sample_max_rows": int(sample_max_rows),
        "random_seed": int(random_seed),
        "status": "ok",
    }

    if len(matrix_cols) < 2:
        _plot_placeholder(
            PLOT_FEATURE_MATRIX,
            "Feature-space matrix: real vs dictionary",
            "Not enough common numeric feature columns to draw the matrix.",
        )
        summary["status"] = "insufficient_columns"
        summary["rows_real_plotted"] = 0
        summary["rows_dictionary_plotted"] = 0
        return summary

    plot_data_cols = [c for c in matrix_cols if c != STEP12_RATE_HIST_PLACEHOLDER_COL]
    real_plot = real_df[plot_data_cols].copy()
    dict_plot = dict_df[plot_data_cols].copy()
    if sample_max_rows > 0:
        if len(real_plot) > sample_max_rows:
            real_plot = real_plot.sample(n=sample_max_rows, random_state=random_seed)
        if len(dict_plot) > sample_max_rows:
            dict_plot = dict_plot.sample(n=sample_max_rows, random_state=random_seed)

    n = len(matrix_cols)
    if n > 40:
        cap = 700
    elif n > 24:
        cap = 1200
    else:
        cap = 0
    if cap > 0:
        if len(real_plot) > cap:
            real_plot = real_plot.sample(n=cap, random_state=random_seed)
        if len(dict_plot) > cap:
            dict_plot = dict_plot.sample(n=cap, random_state=random_seed)

    summary["rows_real_plotted"] = int(len(real_plot))
    summary["rows_dictionary_plotted"] = int(len(dict_plot))

    if n <= 12:
        cell = 2.4
    elif n <= 24:
        cell = 1.5
    elif n <= 40:
        cell = 1.0
    else:
        cell = 0.78

    real_numeric = {
        col: pd.to_numeric(real_plot[col], errors="coerce")
        for col in plot_data_cols
    }
    dict_numeric = {
        col: pd.to_numeric(dict_plot[col], errors="coerce")
        for col in plot_data_cols
    }
    feature_axis_limits: dict[str, tuple[float, float]] = {}
    for col in plot_data_cols:
        combined = pd.concat([real_numeric[col], dict_numeric[col]], ignore_index=True).to_numpy(dtype=float)
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            continue
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        if np.isclose(lower, upper):
            pad = max(abs(lower) * 0.05, 1e-6)
            feature_axis_limits[col] = (lower - pad, upper + pad)
            continue
        # Keep exact plotted span (no margin) so real/dictionary overlays
        # are visually comparable without artificial extension to nearby ticks.
        pad = 0.0
        feature_axis_limits[col] = (lower - pad, upper + pad)

    fig, axes = plt.subplots(n, n, figsize=(cell * n, cell * n), squeeze=False)
    diag_bins = 28 if n <= 24 else 16
    dict_color = "#7A7A7A"
    real_color = "#1F77B4"
    point_size = 7 if n <= 24 else 3
    dict_alpha = 0.20
    real_alpha = 0.30

    for i, cy in enumerate(matrix_cols):
        for j, cx in enumerate(matrix_cols):
            ax = axes[i, j]
            x_is_placeholder = (cx == STEP12_RATE_HIST_PLACEHOLDER_COL)
            y_is_placeholder = (cy == STEP12_RATE_HIST_PLACEHOLDER_COL)
            if i == j:
                if x_is_placeholder:
                    ax.set_facecolor("#F2F2F2")
                    msg = "Rate-histogram block\nsuppressed for display"
                    if len(rate_hist_cols) > 0:
                        msg = f"Rate-histogram block suppressed\n({len(rate_hist_cols)} columns)"
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
                        ax.set_xlabel(STEP12_RATE_HIST_PLACEHOLDER_LABEL, fontsize=5.3)
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel("Rows", fontsize=5.3)
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=4.8, length=1.5)
                    ax.grid(True, alpha=0.12)
                    continue
                dvals = dict_numeric[cx].dropna()
                rvals = real_numeric[cx].dropna()
                if dvals.empty and rvals.empty:
                    ax.set_visible(False)
                    continue
                if not dvals.empty:
                    ax.hist(
                        dvals.to_numpy(dtype=float),
                        bins=diag_bins,
                        color=dict_color,
                        alpha=0.45,
                    )
                if not rvals.empty:
                    ax.hist(
                        rvals.to_numpy(dtype=float),
                        bins=diag_bins,
                        color=real_color,
                        alpha=0.55,
                    )
            elif i > j:
                if x_is_placeholder or y_is_placeholder:
                    ax.set_facecolor("#F7F7F7")
                    # Keep per-feature axis consistency even in the suppressed
                    # rate-histogram placeholder row/column panels.
                    if not x_is_placeholder:
                        x_limits = feature_axis_limits.get(cx)
                        if x_limits is not None:
                            ax.set_xlim(x_limits)
                    if not y_is_placeholder:
                        y_limits = feature_axis_limits.get(cy)
                        if y_limits is not None:
                            ax.set_ylim(y_limits)
                    if i == n - 1:
                        ax.set_xlabel(
                            STEP12_RATE_HIST_PLACEHOLDER_LABEL if x_is_placeholder else cx,
                            fontsize=5.3,
                        )
                    else:
                        ax.set_xticklabels([])
                    if j == 0:
                        ax.set_ylabel(
                            STEP12_RATE_HIST_PLACEHOLDER_LABEL if y_is_placeholder else cy,
                            fontsize=5.3,
                        )
                    else:
                        ax.set_yticklabels([])
                    ax.tick_params(labelsize=4.8, length=1.5)
                    ax.grid(True, alpha=0.10)
                    continue
                xd = dict_numeric[cx]
                yd = dict_numeric[cy]
                md = xd.notna() & yd.notna()
                if md.any():
                    ax.scatter(
                        xd[md].to_numpy(dtype=float),
                        yd[md].to_numpy(dtype=float),
                        s=point_size,
                        alpha=dict_alpha,
                        color=dict_color,
                        edgecolors="none",
                        rasterized=True,
                    )
                xr = real_numeric[cx]
                yr = real_numeric[cy]
                mr = xr.notna() & yr.notna()
                if mr.any():
                    ax.scatter(
                        xr[mr].to_numpy(dtype=float),
                        yr[mr].to_numpy(dtype=float),
                        s=point_size,
                        alpha=real_alpha,
                        color=real_color,
                        edgecolors="none",
                        rasterized=True,
                    )
            else:
                ax.axis("off")
                continue

            if not x_is_placeholder:
                x_limits = feature_axis_limits.get(cx)
                if x_limits is not None:
                    ax.set_xlim(x_limits)
            if i > j and not y_is_placeholder:
                y_limits = feature_axis_limits.get(cy)
                if y_limits is not None:
                    ax.set_ylim(y_limits)

            if i == n - 1:
                ax.set_xlabel(
                    STEP12_RATE_HIST_PLACEHOLDER_LABEL if x_is_placeholder else cx,
                    fontsize=5.3,
                )
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(
                    (STEP12_RATE_HIST_PLACEHOLDER_LABEL if y_is_placeholder else (cy if i > 0 else "Rows")),
                    fontsize=5.3,
                )
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=4.8, length=1.5)
            ax.grid(True, alpha=0.12)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color=dict_color,
            markerfacecolor=dict_color,
            markeredgecolor="none",
            markersize=5.0,
            alpha=0.7,
            label=f"Dictionary (n={len(dict_plot)})",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color=real_color,
            markerfacecolor=real_color,
            markeredgecolor="none",
            markersize=5.0,
            alpha=0.8,
            label=f"Real data (n={len(real_plot)})",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        framealpha=0.92,
        borderaxespad=0.6,
    )
    fig.suptitle(
        "STEP 4.1 feature-space scatter matrix (lower-triangle): real vs dictionary",
        y=0.998,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(PLOT_FEATURE_MATRIX, dpi=140)
    plt.close(fig)
    return summary


def _choose_timeline_column(df: pd.DataFrame) -> str | None:
    for candidate in ("file_timestamp_utc", "execution_timestamp_utc"):
        if candidate in df.columns and pd.to_datetime(df[candidate], errors="coerce", utc=True).notna().any():
            return candidate
    return None


def _pick_feature_space_preview_columns(df: pd.DataFrame, *, max_lines: int = 8) -> list[str]:
    """Pick a compact, readable set of feature-space columns for timeline preview."""
    preferred = [
        "eff_empirical_1",
        "eff_empirical_2",
        "eff_empirical_3",
        "eff_empirical_4",
        "efficiency_product_4planes",
        "flux_proxy_rate_div_effprod",
        "flux_proxy_rate_div_effprod_123",
        "flux_proxy_rate_div_effprod_234",
    ]
    out: list[str] = []
    seen: set[str] = set()

    def _maybe_add(col: str) -> None:
        if col in seen or col not in df.columns:
            return
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() < 3:
            return
        out.append(col)
        seen.add(col)

    for col in preferred:
        _maybe_add(col)

    hist_cols: list[tuple[int, str]] = []
    for col in df.columns:
        m = RATE_HIST_BIN_RE.match(str(col))
        if m is None:
            continue
        hist_cols.append((int(m.group("bin")), str(col)))
    hist_cols.sort(key=lambda x: x[0])
    if hist_cols:
        pick_idx = sorted({0, len(hist_cols) // 2, len(hist_cols) - 1})
        for i in pick_idx:
            _maybe_add(hist_cols[i][1])

    if len(out) < max_lines:
        for col in df.columns:
            if col in seen:
                continue
            cl = str(col).lower()
            if col == "events_per_second_global_rate":
                continue
            if "timestamp" in cl or "filename" in cl or cl.startswith("online_z_"):
                continue
            if col in {"task_id", "station_id", "n_events"}:
                continue
            _maybe_add(str(col))
            if len(out) >= max_lines:
                break

    return out[:max_lines]


def _make_comparison_columns_plot(
    df: pd.DataFrame,
    *,
    comparison_cols: list[str],
    comparison_strategy: str,
    global_rate_col: str | None,
) -> int:
    """Plot global rate, efficiencies, rate-histogram heatmap, and physics transforms."""
    if global_rate_col is None and not any(c in df.columns for c in ("eff_empirical_1",)):
        _plot_placeholder(
            PLOT_COMPARE_COLS,
            "Comparison columns preview",
            "No global-rate, feature-space, or comparison columns available.",
        )
        return 0

    # --- resolve time axis -------------------------------------------------
    tcol = _choose_timeline_column(df)
    if tcol is None:
        x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        x_label = "Row index"
        x_is_time = False
    else:
        ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(None)
        if ts.notna().any():
            x = ts
            x_label = (
                "Data time from filename_base [UTC]"
                if tcol == "file_timestamp_utc"
                else "Execution time [UTC]"
            )
            x_is_time = True
        else:
            x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
            x_label = "Row index"
            x_is_time = False

    # --- identify feature groups -------------------------------------------
    eff_cols = [c for c in ("eff_empirical_1", "eff_empirical_2", "eff_empirical_3", "eff_empirical_4") if c in df.columns]
    hist_indexed: list[tuple[int, str]] = []
    for col in df.columns:
        m = RATE_HIST_BIN_RE.match(str(col))
        if m is not None:
            hist_indexed.append((int(m.group("bin")), str(col)))
    hist_indexed.sort(key=lambda t: t[0])
    hist_cols = [c for _, c in hist_indexed]
    hist_bins = [b for b, _ in hist_indexed]

    # --- decide panel layout (2 panels: global_rate+histogram, efficiencies) --
    panels: list[str] = ["global_rate"]
    if eff_cols:
        panels.append("efficiencies")
    n_panels = len(panels)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13.5, 3.6 * n_panels),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # --- Panel: global rate with histogram heatmap as background -----------
    ax = axes[ax_idx]; ax_idx += 1

    # Draw histogram heatmap as background (shared y-axis: bin = rate Hz)
    if hist_cols:
        hist_matrix = df[hist_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T  # (n_bins, n_rows)
        xvals = x.to_numpy()
        if x_is_time:
            import matplotlib.dates as mdates
            xvals_num = mdates.date2num(pd.to_datetime(xvals))
        else:
            xvals_num = xvals.astype(float)

        if len(xvals_num) > 1:
            dx = np.diff(xvals_num)
            x_edges = np.concatenate([
                [xvals_num[0] - dx[0] / 2],
                (xvals_num[:-1] + xvals_num[1:]) / 2,
                [xvals_num[-1] + dx[-1] / 2],
            ])
        else:
            x_edges = np.array([xvals_num[0] - 0.5, xvals_num[0] + 0.5])
        hist_bins_arr = np.array(hist_bins, dtype=float)
        y_edges = np.concatenate([
            [hist_bins_arr[0] - 0.5],
            (hist_bins_arr[:-1] + hist_bins_arr[1:]) / 2,
            [hist_bins_arr[-1] + 0.5],
        ])

        finite_vals = hist_matrix[np.isfinite(hist_matrix)]
        vmin = float(np.nanpercentile(finite_vals, 2)) if len(finite_vals) else 0.0
        vmax = float(np.nanpercentile(finite_vals, 98)) if len(finite_vals) else 1.0

        pcm = ax.pcolormesh(
            x_edges, y_edges, hist_matrix,
            cmap="turbo", shading="flat",
            vmin=vmin, vmax=vmax, alpha=0.35,
            zorder=0,
        )
        cb = fig.colorbar(pcm, ax=ax, pad=0.012, aspect=30, fraction=0.04)
        cb.set_label("Hist. rate [Hz]", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # Draw global rate on the same axis (above the heatmap)
    if global_rate_col is not None and global_rate_col in df.columns:
        gv = pd.to_numeric(df[global_rate_col], errors="coerce")
        valid = gv.notna()
        if valid.any():
            xv = x[valid].to_numpy()
            yv = gv[valid].to_numpy(dtype=float)
            ax.scatter(xv, yv, s=14, color="#4C72B0", alpha=0.6, zorder=3, edgecolors="none")
            ax.set_ylabel("Rate [Hz]", fontsize=9)
            ax.set_title(f"Global rate ({global_rate_col})", fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"{global_rate_col}: no finite values", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("Rate [Hz]", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No global-rate column found", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Rate [Hz]", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # --- Panel: empirical efficiencies -------------------------------------
    if "efficiencies" in panels:
        ax = axes[ax_idx]; ax_idx += 1
        eff_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        for i, col in enumerate(eff_cols):
            v = pd.to_numeric(df[col], errors="coerce")
            ok = v.notna()
            if ok.sum() < 2:
                continue
            label = col.replace("eff_empirical_", "Plane ")
            ax.scatter(
                x[ok].to_numpy(), v[ok].to_numpy(dtype=float),
                color=eff_colors[i % len(eff_colors)],
                s=12, alpha=0.8, edgecolors="none",
                label=label,
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Efficiency", fontsize=9)
        ax.set_title("Empirical efficiencies per plane", fontsize=10, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8, ncol=4, frameon=True, framealpha=0.85)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # --- finalize ----------------------------------------------------------
    axes[-1].set_xlabel(x_label, fontsize=9)
    if x_is_time:
        fig.autofmt_xdate(rotation=25, ha="right")
    fig.savefig(PLOT_COMPARE_COLS, dpi=170)
    plt.close(fig)
    return len(eff_cols)


_EFFPROD_SUFFIX_TO_TT_LABEL = {
    "4planes": "1234",
    "123": "123",
    "234": "234",
    "12": "12",
    "23": "23",
    "34": "34",
}


def _make_tt_rate_breakdown_plot(
    df: pd.DataFrame,
) -> int:
    """Plot per-TT rate, efficiency product, and rate/eff_product in a 3×N grid."""
    tcol = _choose_timeline_column(df)
    if tcol is None:
        x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        x_label = "Row index"
        x_is_time = False
    else:
        ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(None)
        if ts.notna().any():
            x = ts
            x_label = (
                "Data time from filename_base [UTC]"
                if tcol == "file_timestamp_utc"
                else "Execution time [UTC]"
            )
            x_is_time = True
        else:
            x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
            x_label = "Row index"
            x_is_time = False

    # Find efficiency_product columns and matching TT rate columns
    tt_entries: list[tuple[str, str, str, str]] = []  # (label, tt_rate_col, effprod_col, source)
    for suffix, tt_label in _EFFPROD_SUFFIX_TO_TT_LABEL.items():
        corrected_col = POLY_CORRECTED_EFFPROD_COL_TEMPLATE.format(suffix=suffix)
        raw_col = f"efficiency_product_{suffix}"
        effprod_col = None
        effprod_source = "feature_space"
        if corrected_col in df.columns and pd.to_numeric(df[corrected_col], errors="coerce").notna().any():
            effprod_col = corrected_col
            effprod_source = "poly_corrected"
        elif raw_col in df.columns:
            effprod_col = raw_col
        if effprod_col is None:
            continue
        pattern = re.compile(rf"^.+_tt_{re.escape(tt_label)}_rate_hz$")
        candidates = [c for c in df.columns if pattern.match(c)]
        if not candidates:
            continue
        tt_rate_col = _choose_best_col(candidates)
        tt_entries.append((tt_label, tt_rate_col, effprod_col, effprod_source))

    if not tt_entries:
        _plot_placeholder(
            PLOT_TT_BREAKDOWN,
            "TT rate breakdown",
            "No matching TT rate + efficiency product columns found.",
        )
        return 0

    n_tt = len(tt_entries)
    fig, axes = plt.subplots(
        3, n_tt,
        figsize=(4.0 * n_tt, 9.0),
        squeeze=False,
    )

    # Share y-axis within each row and x-axis within each column
    for row in range(3):
        for col in range(1, n_tt):
            axes[row, col].sharey(axes[row, 0])
    for col in range(n_tt):
        for row in range(2):
            axes[row, col].sharex(axes[2, col])

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    use_poly_corrected = any(source == "poly_corrected" for _, _, _, source in tt_entries)
    row_labels = [
        "TT rate [Hz]",
        ("Poly-corrected eff. product" if use_poly_corrected else "Eff. product"),
        ("Rate / poly-corrected eff. prod." if use_poly_corrected else "Rate / eff. prod."),
    ]

    for j, (tt_label, tt_rate_col, effprod_col, effprod_source) in enumerate(tt_entries):
        rate = pd.to_numeric(df[tt_rate_col], errors="coerce")
        effprod = pd.to_numeric(df[effprod_col], errors="coerce")
        color = colors[j % len(colors)]

        # Row 0: TT rate
        ax = axes[0, j]
        ok = rate.notna()
        if ok.any():
            ax.scatter(
                x[ok].to_numpy(), rate[ok].to_numpy(dtype=float),
                s=8, color=color, alpha=0.6, edgecolors="none",
            )
        title = f"TT {tt_label}"
        if effprod_source == "poly_corrected":
            title += " (poly)"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # Row 1: efficiency product
        ax = axes[1, j]
        ok = effprod.notna()
        if ok.any():
            ax.scatter(
                x[ok].to_numpy(), effprod[ok].to_numpy(dtype=float),
                s=8, color=color, alpha=0.6, edgecolors="none",
            )
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # Row 2: rate / eff_product
        ax = axes[2, j]
        both_ok = rate.notna() & effprod.notna() & (effprod > 0)
        if both_ok.any():
            ratio = rate[both_ok].to_numpy(dtype=float) / effprod[both_ok].to_numpy(dtype=float)
            ax.scatter(
                x[both_ok].to_numpy(), ratio,
                s=8, color=color, alpha=0.6, edgecolors="none",
            )
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Y-axis labels only on leftmost column
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=8)
        for col in range(1, n_tt):
            plt.setp(axes[row, col].get_yticklabels(), visible=False)

    # X-axis labels only on bottom row
    for col in range(n_tt):
        axes[2, col].set_xlabel(x_label, fontsize=7)
        for row in range(2):
            plt.setp(axes[row, col].get_xticklabels(), visible=False)

    if x_is_time:
        fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_TT_BREAKDOWN, dpi=170)
    plt.close(fig)
    return n_tt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Step 4.1: Collect real metadata rows for selected station/task IDs."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--station-id", default=None)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    cfg_12 = config.get("step_1_2", {})
    if not isinstance(cfg_12, dict):
        cfg_12 = {}
    cfg_41 = config.get("step_4_1", {})
    _clear_plots_dir()

    station_cfg = cfg_41.get("station_id", config.get("station_id", 0))
    station_raw = args.station_id if args.station_id is not None else station_cfg
    station_id = _safe_station_id(station_raw, default=0)
    station_name = f"MINGO{station_id:02d}"

    task_ids_cfg = (
        args.task_ids
        if args.task_ids is not None
        else cfg_41.get("task_ids", config.get("task_ids", [1]))
    )
    task_ids = _safe_task_ids(task_ids_cfg)
    preferred_feature_prefixes = _preferred_feature_prefixes_for_task_ids(task_ids)

    timestamp_col = str(cfg_41.get("timestamp_column", "execution_timestamp"))
    metadata_agg = str(config.get("metadata_agg", "latest")).strip().lower()
    dictionary_csv_cfg = cfg_41.get("dictionary_csv", None)
    build_summary_cfg = cfg_41.get("build_summary_json", None)
    global_rate_col_pref = str(cfg_41.get("global_rate_col", "events_per_second_global_rate"))
    event_count_col_cfg = str(cfg_41.get("event_count_column", "auto")).strip()
    selected_features_json_cfg = cfg_41.get("selected_feature_columns_json", None)
    selected_features_path = _resolve_input_path(
        selected_features_json_cfg or DEFAULT_STEP14_SELECTED_FEATURES
    )
    matrix_sample_max_raw = _coalesce(
        cfg_41.get("feature_matrix_plot_sample_max_rows", None),
        cfg_12.get("feature_space_plot_sample_max_rows", 25000),
    )
    try:
        matrix_sample_max_rows = int(matrix_sample_max_raw)
    except (TypeError, ValueError):
        matrix_sample_max_rows = 25000
    if matrix_sample_max_rows < 0:
        matrix_sample_max_rows = 0
    matrix_seed_raw = _coalesce(
        cfg_41.get("feature_matrix_plot_random_seed", None),
        cfg_12.get("feature_space_plot_random_seed", 1234),
    )
    try:
        matrix_random_seed = int(matrix_seed_raw)
    except (TypeError, ValueError):
        matrix_random_seed = 1234
    matrix_include_rate_hist_raw = _coalesce(
        cfg_41.get("feature_matrix_plot_include_rate_histogram", None),
        cfg_12.get("feature_matrix_plot_include_rate_histogram", True),
    )
    matrix_include_rate_histogram = _safe_bool(
        matrix_include_rate_hist_raw,
        default=True,
    )
    min_n_events_raw = cfg_41.get("min_n_events", 30000)
    if min_n_events_raw in (None, "", "null", "None"):
        min_n_events: float | None = None
    else:
        try:
            min_n_events = float(min_n_events_raw)
        except (TypeError, ValueError):
            min_n_events = 30000.0
            log.warning(
                "Invalid step_4_1.min_n_events=%r; using default %.0f.",
                min_n_events_raw,
                min_n_events,
            )
        if not np.isfinite(min_n_events):
            min_n_events = 30000.0
            log.warning(
                "Non-finite step_4_1.min_n_events=%r; using default %.0f.",
                min_n_events_raw,
                min_n_events,
            )
        if min_n_events < 0.0:
            log.warning(
                "Negative step_4_1.min_n_events=%r; clipping to 0.",
                min_n_events_raw,
            )
            min_n_events = 0.0

    date_from_cfg = args.date_from if args.date_from is not None else cfg_41.get("date_from", config.get("date_from", None))
    date_to_cfg = args.date_to if args.date_to is not None else cfg_41.get("date_to", config.get("date_to", None))
    date_from = _parse_bound(date_from_cfg)
    date_to = _parse_bound(date_to_cfg)

    log.info("Station: %s", station_name)
    log.info("Task IDs: %s", task_ids)
    log.info("Preferred TT prefix order for auto comparison features: %s", preferred_feature_prefixes)
    log.info(
        "Date range (applied to filename_base data-time): %s .. %s",
        date_from if date_from is not None else "None",
        date_to if date_to is not None else "None",
    )
    log.info(
        "Event-count filter: event_count_column=%s min_n_events=%s (strictly greater-than).",
        event_count_col_cfg if event_count_col_cfg else "auto",
        ("disabled" if min_n_events is None else f"{min_n_events:.0f}"),
    )
    build_summary_path = _resolve_input_path(build_summary_cfg or DEFAULT_BUILD_SUMMARY)
    log.info("STEP 1.3 fit summary: %s", build_summary_path)

    dict_path = _resolve_input_path(dictionary_csv_cfg or DEFAULT_DICTIONARY)
    if not dict_path.exists():
        log.error(
            "Dictionary CSV not found: %s. STEP 4.1 requires dictionary z positions for z-matching.",
            dict_path,
        )
        return 1
    try:
        dict_df = pd.read_csv(dict_path, low_memory=False)
    except Exception as exc:
        log.error("Could not load dictionary CSV at %s: %s", dict_path, exc)
        return 1
    if dict_df.empty:
        log.error("Dictionary CSV is empty: %s", dict_path)
        return 1

    dictionary_z_tuples = _extract_dictionary_z_tuples(dict_df)
    if not dictionary_z_tuples:
        log.error(
            "Dictionary has no usable z-plane columns (z_plane_1..4): %s",
            dict_path,
        )
        return 1
    log.info("Dictionary z tuples: %s", [list(z) for z in sorted(dictionary_z_tuples)])

    try:
        online_schedule_all, online_schedule_path = _load_online_run_schedule(station_id)
    except Exception as exc:
        log.error("Could not load ONLINE_RUN_DICTIONARY for station %s: %s", station_name, exc)
        return 1
    if online_schedule_all.empty:
        log.error(
            "ONLINE_RUN_DICTIONARY has no valid schedule rows for station %s: %s",
            station_name,
            online_schedule_path,
        )
        return 1
    online_schedule_window = _select_schedule_rows_for_window(
        online_schedule_all,
        date_from=date_from,
        date_to=date_to,
    )
    if online_schedule_window.empty:
        log.error(
            "No ONLINE_RUN_DICTIONARY schedule rows overlap requested date range for station %s: %s",
            station_name,
            online_schedule_path,
        )
        return 1

    online_z_tuples_window = set(online_schedule_window["z_tuple"].tolist())
    common_z_tuples = set(dictionary_z_tuples) & set(online_z_tuples_window)
    if not common_z_tuples:
        log.error(
            "No common z tuples between dictionary and ONLINE_RUN_DICTIONARY in requested window. "
            "dictionary=%s | online_window=%s",
            [list(z) for z in sorted(dictionary_z_tuples)],
            [list(z) for z in sorted(online_z_tuples_window)],
        )
        return 1
    log.info(
        "ONLINE_RUN_DICTIONARY rows in requested window: %d (from %s). Common z tuples with dictionary: %s",
        len(online_schedule_window),
        online_schedule_path,
        [list(z) for z in sorted(common_z_tuples)],
    )

    all_frames: list[pd.DataFrame] = []
    task_stats: dict[str, dict[str, int]] = {}

    for task_id in task_ids:
        specific_path = _task_specific_metadata_path(station_id, task_id)
        trigger_path = _task_trigger_type_metadata_path(station_id, task_id)
        rate_hist_path = _task_rate_histogram_metadata_path(station_id, task_id)
        key = str(task_id)
        task_stats[key] = {
            "rows_read": 0,
            "rows_read_specific": 0,
            "rows_read_trigger_type": 0,
            "rows_read_rate_histogram": 0,
            "rows_after_latest_agg": 0,
            "rows_after_latest_agg_specific": 0,
            "rows_after_latest_agg_trigger_type": 0,
            "rows_after_latest_agg_rate_histogram": 0,
            "rows_after_metadata_merge": 0,
            "rows_with_rate_histogram_after_merge": 0,
            "rows_after_date_filter": 0,
            "rows_with_online_z_mapped": 0,
            "rows_after_z_filter": 0,
            "rows_with_valid_file_timestamp": 0,
            "rows_with_valid_timestamp": 0,
        }

        try:
            trigger_df = _load_task_metadata_source_csv(
                task_id=task_id,
                source_name="trigger_type",
                path=trigger_path,
                metadata_agg=metadata_agg,
                timestamp_col=timestamp_col,
            )
            rate_hist_df = _load_task_metadata_source_csv(
                task_id=task_id,
                source_name="rate_histogram",
                path=rate_hist_path,
                metadata_agg=metadata_agg,
                timestamp_col=timestamp_col,
            )
            specific_df = _load_task_metadata_source_csv(
                task_id=task_id,
                source_name="specific",
                path=specific_path,
                metadata_agg=metadata_agg,
                timestamp_col=timestamp_col,
            )
        except (FileNotFoundError, KeyError) as exc:
            log.error("%s", exc)
            return 1

        task_stats[key]["rows_read_trigger_type"] = int(len(trigger_df))
        task_stats[key]["rows_read_rate_histogram"] = int(len(rate_hist_df))
        task_stats[key]["rows_read_specific"] = int(len(specific_df))
        task_stats[key]["rows_after_latest_agg_trigger_type"] = int(len(trigger_df))
        task_stats[key]["rows_after_latest_agg_rate_histogram"] = int(len(rate_hist_df))
        task_stats[key]["rows_after_latest_agg_specific"] = int(len(specific_df))
        task_stats[key]["rows_read"] = int(
            task_stats[key]["rows_read_trigger_type"]
            + task_stats[key]["rows_read_rate_histogram"]
            + task_stats[key]["rows_read_specific"]
        )

        meta_df, merge_info = _merge_metadata_sources_equal_level(
            task_id=task_id,
            ordered_sources=[
                ("trigger_type", trigger_df),
                ("rate_histogram", rate_hist_df),
                ("specific", specific_df),
            ],
        )
        log.info(
            "  Metadata merged at equal level for task %d: rows=%d, columns=%d.",
            task_id,
            len(meta_df),
            len(meta_df.columns),
        )
        if merge_info.get("renamed_overlap_columns"):
            overlap_counts = {
                src: int(len(cols))
                for src, cols in merge_info["renamed_overlap_columns"].items()
                if isinstance(cols, dict)
            }
            log.info("  Metadata overlap columns renamed by source: %s", overlap_counts)

        hist_cols_merged = [
            c
            for c in meta_df.columns
            if (
                RATE_HIST_BIN_RE.match(str(c)) is not None
                or str(c).startswith("rate_histogram__events_per_second_")
            )
        ]
        if hist_cols_merged:
            task_stats[key]["rows_with_rate_histogram_after_merge"] = int(
                meta_df[hist_cols_merged].notna().any(axis=1).sum()
            )
        else:
            task_stats[key]["rows_with_rate_histogram_after_merge"] = 0

        task_stats[key]["rows_after_latest_agg"] = int(len(meta_df))
        task_stats[key]["rows_after_metadata_merge"] = int(len(meta_df))

        file_ts = meta_df["filename_base"].map(_parse_filename_base_ts)
        meta_df["file_timestamp_utc"] = file_ts
        task_stats[key]["rows_with_valid_file_timestamp"] = int(file_ts.notna().sum())

        if timestamp_col in meta_df.columns:
            ts = _parse_ts(meta_df[timestamp_col])
            meta_df["execution_timestamp_utc"] = ts
            task_stats[key]["rows_with_valid_timestamp"] = int(ts.notna().sum())
        else:
            meta_df["execution_timestamp_utc"] = pd.NaT

        if date_from is not None or date_to is not None:
            keep = file_ts.notna()
            if date_from is not None:
                keep &= file_ts >= date_from
            if date_to is not None:
                keep &= file_ts <= date_to
            meta_df = meta_df.loc[keep].copy()

        task_stats[key]["rows_after_date_filter"] = int(len(meta_df))

        if meta_df.empty:
            continue

        online_z = meta_df["file_timestamp_utc"].map(
            lambda ts: _online_z_tuple_for_timestamp(ts, online_schedule_window)
        )
        task_stats[key]["rows_with_online_z_mapped"] = int(online_z.notna().sum())
        keep_z = online_z.isin(common_z_tuples)
        task_stats[key]["rows_after_z_filter"] = int(keep_z.sum())
        meta_df = meta_df.loc[keep_z].copy()
        if meta_df.empty:
            continue

        online_z = online_z.loc[keep_z]
        meta_df["online_z_tuple"] = list(online_z.values)
        z_split = pd.DataFrame(
            meta_df["online_z_tuple"].tolist(),
            columns=["online_z_plane_1", "online_z_plane_2", "online_z_plane_3", "online_z_plane_4"],
            index=meta_df.index,
        )
        meta_df = pd.concat([meta_df, z_split], axis=1)

        meta_df["task_id"] = int(task_id)
        meta_df["station_id"] = int(station_id)
        all_frames.append(meta_df)

    if not all_frames:
        log.error("No real rows were collected. Check station/task/date settings.")
        return 1

    collected = pd.concat(all_frames, ignore_index=True)
    collected_before_event_cut = collected.copy()
    rows_before_event_cut = int(len(collected))
    event_values, event_source_for_filter, event_source_mode_for_filter = _resolve_event_count_values(
        collected,
        event_count_column=event_count_col_cfg or "auto",
    )
    if event_values.notna().any():
        collected["n_events"] = event_values.to_numpy(dtype=float)

    event_cut_applied = False
    rows_after_event_cut = rows_before_event_cut
    if min_n_events is not None:
        if event_values.notna().any():
            keep_events = event_values > float(min_n_events)
            rows_after_event_cut = int(keep_events.sum())
            collected = collected.loc[keep_events].copy()
            event_cut_applied = True
            log.info(
                "Applied event-count cut: %s > %.0f (kept %d/%d rows).",
                (event_source_for_filter or "n_events"),
                min_n_events,
                rows_after_event_cut,
                rows_before_event_cut,
            )
        else:
            log.warning(
                "Requested event-count cut (min_n_events=%.0f) but no finite event-count values were found. "
                "Cut skipped.",
                min_n_events,
            )
    if collected.empty:
        log.error("No real rows left after STEP 4.1 event-count filtering.")
        return 1

    if "file_timestamp_utc" in collected.columns:
        collected = collected.sort_values(
            by="file_timestamp_utc", kind="mergesort", na_position="last"
        ).reset_index(drop=True)
    elif "execution_timestamp_utc" in collected.columns:
        collected = collected.sort_values(
            by="execution_timestamp_utc", kind="mergesort", na_position="last"
        ).reset_index(drop=True)

    step41_row_id_col = "__step41_row_id"
    collected[step41_row_id_col] = np.arange(len(collected), dtype=int)
    # Save TT rate columns before the step12 transform drops them
    tt_rate_cols_pre = _tt_rate_columns(collected)
    if tt_rate_cols_pre:
        tt_rate_pre_df = collected[[step41_row_id_col, *tt_rate_cols_pre]].copy()
    else:
        tt_rate_pre_df = None
    passthrough_cols = [
        c for c in collected.columns
        if (
            c in {"filename_base", "task_id", "station_id", "n_events"}
            or str(c).startswith("online_z_")
            or ("timestamp" in str(c).lower())
        )
    ]
    passthrough_cols = [c for c in passthrough_cols if c != step41_row_id_col]
    passthrough_df = collected[[step41_row_id_col, *passthrough_cols]].copy()

    rows_before_step12_transform = int(len(collected))
    try:
        collected, step12_transform_info = _apply_step12_feature_space_transform(
            collected,
            cfg_12=cfg_12,
            identity_column=step41_row_id_col,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    restore_cols = [c for c in passthrough_cols if c not in collected.columns]
    if restore_cols:
        collected = collected.merge(
            passthrough_df[[step41_row_id_col, *restore_cols]],
            on=step41_row_id_col,
            how="left",
        )
    # Restore TT rate columns for the breakdown plot (not saved to CSV)
    if tt_rate_pre_df is not None:
        tt_restore = [c for c in tt_rate_cols_pre if c not in collected.columns]
        if tt_restore:
            collected = collected.merge(
                tt_rate_pre_df[[step41_row_id_col, *tt_restore]],
                on=step41_row_id_col,
                how="left",
            )
    collected = collected.drop(columns=[step41_row_id_col], errors="ignore")
    rows_after_step12_transform = int(len(collected))
    if collected.empty:
        log.error("No real rows left after STEP 1.2 feature-space transformation in STEP 4.1.")
        return 1
    log.info(
        "Applied STEP 1.2 feature-space transform in STEP 4.1: rows %d -> %d, columns=%d.",
        rows_before_step12_transform,
        rows_after_step12_transform,
        int(len(collected.columns)),
    )
    step13_poly_correction_info = _append_step13_polynomial_efficiency_products(
        collected,
        build_summary_path=build_summary_path,
    )

    out_csv = FILES_DIR / "real_collected_data.csv"
    csv_cols = [c for c in collected.columns if not re.search(r"_tt_.+_rate_hz$", str(c))]
    collected[csv_cols].to_csv(out_csv, index=False)
    log.info("Wrote collected real data (STEP 1.2 transformed): %s (%d rows)", out_csv, len(collected))

    event_col = _make_event_histogram(
        collected_before_event_cut,
        event_count_column=event_count_col_cfg or "auto",
        min_n_events=min_n_events,
    )
    feature_matrix_info = _make_feature_scatter_matrix_real_vs_dictionary(
        collected,
        dict_df,
        selected_features_path=selected_features_path,
        sample_max_rows=matrix_sample_max_rows,
        random_seed=matrix_random_seed,
        include_rate_histogram=matrix_include_rate_histogram,
    )
    timeline_col = _choose_timeline_column(collected)

    comparison_strategy, comparison_cols, comparison_map_preview = _resolve_comparison_preview(
        real_df=collected,
        dict_df=dict_df if not dict_df.empty else None,
        preferred_prefixes=preferred_feature_prefixes,
    )
    global_rate_col_used = _pick_global_rate_column(collected, preferred=global_rate_col_pref)
    n_compare_shown = _make_comparison_columns_plot(
        collected,
        comparison_cols=comparison_cols,
        comparison_strategy=comparison_strategy,
        global_rate_col=global_rate_col_used,
    )
    n_tt_breakdown = _make_tt_rate_breakdown_plot(collected)
    log.info("Wrote plots in: %s", PLOTS_DIR)

    summary = {
        "station_id": station_id,
        "station_name": station_name,
        "task_ids_requested": task_ids,
        "source_path_template": str(
            _task_metadata_dir(station_id, "<id>")
        ),
        "source_metadata_files": [
            "task_<id>_metadata_specific.csv",
            "task_<id>_metadata_trigger_type.csv",
            "task_<id>_metadata_rate_histogram.csv",
        ],
        "task_stats": task_stats,
        "metadata_aggregation": metadata_agg,
        "timestamp_column": timestamp_col,
        "date_from": str(date_from) if date_from is not None else None,
        "date_to": str(date_to) if date_to is not None else None,
        "date_filter_source": "file_timestamp_utc_from_filename_base",
        "z_filter_source": "ONLINE_RUN_DICTIONARY x dictionary_z_tuple_match",
        "online_run_dictionary_csv": str(online_schedule_path),
        "online_schedule_rows_total": int(len(online_schedule_all)),
        "online_schedule_rows_in_requested_window": int(len(online_schedule_window)),
        "online_schedule_z_tuples_in_requested_window": [list(z) for z in sorted(online_z_tuples_window)],
        "dictionary_z_tuples": [list(z) for z in sorted(dictionary_z_tuples)],
        "common_z_tuples_used_for_filter": [list(z) for z in sorted(common_z_tuples)],
        "timeline_column": timeline_col,
        "comparison_preview_strategy": comparison_strategy,
        "comparison_preview_preferred_prefix_order": preferred_feature_prefixes,
        "comparison_preview_columns_total": int(len(comparison_cols)),
        "comparison_preview_columns_shown": int(n_compare_shown),
        "comparison_preview_columns": comparison_cols[:100],
        "comparison_preview_mapping": comparison_map_preview[:100],
        "global_rate_column_used": global_rate_col_used,
        "global_rate_column_preference": global_rate_col_pref,
        "dictionary_for_comparison_preview": str(dict_path),
        "selected_feature_columns_json": str(selected_features_path),
        "feature_matrix_plot": feature_matrix_info,
        "step13_polynomial_efficiency_correction": step13_poly_correction_info,
        "rows_total": int(len(collected)),
        "rows_before_event_cut": rows_before_event_cut,
        "rows_after_event_cut": rows_after_event_cut,
        "event_cut_applied": bool(event_cut_applied),
        "min_n_events": (float(min_n_events) if min_n_events is not None else None),
        "event_count_column_config": (event_count_col_cfg if event_count_col_cfg else "auto"),
        "event_count_source_for_filter": event_source_for_filter,
        "event_count_source_mode_for_filter": event_source_mode_for_filter,
        "rows_before_step12_transform": rows_before_step12_transform,
        "rows_after_step12_transform": rows_after_step12_transform,
        "step12_transform_applied": True,
        "step12_transform": step12_transform_info,
        "rows_with_valid_file_timestamp": int(
            pd.to_datetime(collected.get("file_timestamp_utc"), errors="coerce", utc=True).notna().sum()
        )
        if "file_timestamp_utc" in collected.columns
        else 0,
        "rows_with_valid_timestamp": int(
            pd.to_datetime(collected.get("execution_timestamp_utc"), errors="coerce", utc=True).notna().sum()
        )
        if "execution_timestamp_utc" in collected.columns
        else 0,
        "event_count_column": event_col,
        "output_csv": str(out_csv),
    }
    out_summary = FILES_DIR / "real_collection_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote summary: %s", out_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
