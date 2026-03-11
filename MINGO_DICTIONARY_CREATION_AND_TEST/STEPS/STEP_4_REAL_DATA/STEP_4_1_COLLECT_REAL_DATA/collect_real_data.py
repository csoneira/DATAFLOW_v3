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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
PLOT_COMPARE_COLS = PLOTS_DIR / "STEP_4_1_3_comparison_columns_and_global_rate.png"

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


def _plot_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_event_histogram(df: pd.DataFrame) -> str | None:
    event_col = _pick_event_count_column(df)
    source_label = event_col
    source_mode = "column"
    values = pd.Series(dtype=float)

    if event_col is not None:
        values = pd.to_numeric(df[event_col], errors="coerce").dropna()

    # Last-resort fallback: estimate counts from global rate and acquisition duration.
    if values.empty:
        duration_col = _pick_duration_seconds_column(df)
        global_rate_col = _pick_global_rate_column(df, preferred="events_per_second_global_rate")
        if duration_col is not None and global_rate_col is not None:
            duration = pd.to_numeric(df[duration_col], errors="coerce")
            global_rate = pd.to_numeric(df[global_rate_col], errors="coerce")
            values = (duration * global_rate).dropna()
            if not values.empty:
                source_label = f"{global_rate_col} * {duration_col}"
                source_mode = "derived_rate_x_duration"

    if values.empty:
        _plot_placeholder(
            PLOT_EVENT_HIST,
            "Event count histogram",
            "No valid event-count values found in collected table.",
        )
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=40, alpha=0.5, color="#4C78A8", edgecolor="white")
    ax.set_xlabel(f"Event count ({source_label})")
    ax.set_ylabel("Number of files")
    ax.set_title(f"Event count distribution — {len(values)} collected files")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_EVENT_HIST, dpi=150)
    plt.close(fig)
    return source_label


def _choose_timeline_column(df: pd.DataFrame) -> str | None:
    for candidate in ("file_timestamp_utc", "execution_timestamp_utc"):
        if candidate in df.columns and pd.to_datetime(df[candidate], errors="coerce", utc=True).notna().any():
            return candidate
    return None


def _make_comparison_columns_plot(
    df: pd.DataFrame,
    *,
    comparison_cols: list[str],
    comparison_strategy: str,
    global_rate_col: str | None,
) -> int:
    """Plot global rate and selected comparison columns over data time."""
    if not comparison_cols and global_rate_col is None:
        _plot_placeholder(
            PLOT_COMPARE_COLS,
            "Comparison columns preview",
            "No comparison rate columns or global-rate column available.",
        )
        return 0

    tcol = _choose_timeline_column(df)
    if tcol is None:
        x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        x_label = "Row index"
    else:
        ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(None)
        if ts.notna().any():
            x = ts
            x_label = "Data time from filename_base [UTC]" if tcol == "file_timestamp_utc" else "Execution time [UTC]"
        else:
            x = pd.Series(np.arange(len(df), dtype=float), index=df.index)
            x_label = "Row index"

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4), sharex=True, height_ratios=[1.2, 1.0])

    # Panel 1: global rate
    ax0 = axes[0]
    if global_rate_col is not None and global_rate_col in df.columns:
        gv = pd.to_numeric(df[global_rate_col], errors="coerce")
        valid = gv.notna()
        if valid.any():
            ax0.plot(
                x[valid].to_numpy(),
                gv[valid].to_numpy(dtype=float),
                color="#1F77B4",
                linewidth=1.2,
                marker="o",
                markersize=2.2,
                alpha=0.85,
            )
            ax0.set_ylabel(f"{global_rate_col} [Hz]")
            ax0.set_title("Global rate over selected real-data window")
        else:
            ax0.text(0.5, 0.5, f"{global_rate_col}: no finite values", ha="center", va="center")
            ax0.set_ylabel("Global rate [Hz]")
    else:
        ax0.text(0.5, 0.5, "No global-rate column found", ha="center", va="center")
        ax0.set_ylabel("Global rate [Hz]")
    ax0.grid(True, alpha=0.22)

    # Panel 2: standardized comparison feature traces
    ax1 = axes[1]
    shown = 0
    max_lines = 8
    for col in comparison_cols[:max_lines]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.notna()
        if valid.sum() < 3:
            continue
        v = values[valid].to_numpy(dtype=float)
        center = float(np.nanmedian(v))
        scale = float(np.nanstd(v))
        if not np.isfinite(scale) or scale <= 0.0:
            z = np.zeros_like(v, dtype=float)
        else:
            z = (v - center) / scale
        ax1.plot(
            x[valid].to_numpy(),
            z,
            linewidth=1.0,
            alpha=0.9,
            label=col,
        )
        shown += 1

    if shown == 0:
        ax1.text(0.5, 0.5, "No finite comparison feature traces available", ha="center", va="center")
    else:
        ax1.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax1.set_ylabel("Standardized rate (z-score)")
    ax1.set_xlabel(x_label)
    ax1.set_title(
        "Columns preview for STEP 4.2 matching "
        f"({comparison_strategy}; showing {shown}/{len(comparison_cols)})"
    )
    ax1.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(PLOT_COMPARE_COLS, dpi=150)
    plt.close(fig)
    return shown


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
    global_rate_col_pref = str(cfg_41.get("global_rate_col", "events_per_second_global_rate"))

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

        meta_df: pd.DataFrame | None = None

        # Primary source: trigger_type metadata.
        if trigger_path.exists():
            log.info("Loading task %d trigger_type metadata: %s", task_id, trigger_path)
            trigger_df = pd.read_csv(trigger_path, low_memory=False)
            task_stats[key]["rows_read_trigger_type"] = int(len(trigger_df))
            if "filename_base" not in trigger_df.columns:
                log.warning("Task %d trigger_type metadata has no 'filename_base'; skipping trigger_type.", task_id)
            else:
                if metadata_agg == "latest":
                    trigger_df = _aggregate_latest(trigger_df, timestamp_col=timestamp_col)
                task_stats[key]["rows_after_latest_agg_trigger_type"] = int(len(trigger_df))
                meta_df = trigger_df
        else:
            log.info("Task %d trigger_type metadata not found: %s", task_id, trigger_path)

        # Fallback source: specific metadata only when trigger_type is unavailable.
        if meta_df is None:
            if specific_path.exists():
                log.warning(
                    "Using fallback specific metadata for task %d because trigger_type is unavailable: %s",
                    task_id,
                    specific_path,
                )
                specific_df = pd.read_csv(specific_path, low_memory=False)
                task_stats[key]["rows_read_specific"] = int(len(specific_df))
                if "filename_base" not in specific_df.columns:
                    log.warning("Task %d specific metadata has no 'filename_base'; skipping specific.", task_id)
                else:
                    if metadata_agg == "latest":
                        specific_df = _aggregate_latest(specific_df, timestamp_col=timestamp_col)
                    task_stats[key]["rows_after_latest_agg_specific"] = int(len(specific_df))
                    meta_df = specific_df
            else:
                log.warning("Task %d specific metadata not found: %s", task_id, specific_path)

        rate_hist_df, rate_hist_cols = _load_rate_histogram_bins(
            station_id=station_id,
            task_id=task_id,
            metadata_agg=metadata_agg,
            timestamp_col=timestamp_col,
        )
        if rate_hist_df is not None:
            task_stats[key]["rows_read_rate_histogram"] = int(len(rate_hist_df))
            task_stats[key]["rows_after_latest_agg_rate_histogram"] = int(len(rate_hist_df))

        task_stats[key]["rows_read"] = int(
            task_stats[key]["rows_read_specific"]
            + task_stats[key]["rows_read_trigger_type"]
            + task_stats[key]["rows_read_rate_histogram"]
        )

        if meta_df is None:
            log.warning("Task %d has no usable trigger_type/specific metadata; skipping.", task_id)
            continue

        if rate_hist_df is not None and rate_hist_cols:
            new_hist_cols = [c for c in rate_hist_cols if c not in meta_df.columns]
            if new_hist_cols:
                meta_df = meta_df.merge(
                    rate_hist_df[["filename_base", *new_hist_cols]],
                    on="filename_base",
                    how="left",
                )
                task_stats[key]["rows_with_rate_histogram_after_merge"] = int(
                    meta_df[new_hist_cols].notna().any(axis=1).sum()
                )
                log.info(
                    "Task %d joined rate_histogram bins: %d columns, %d/%d rows with histogram values.",
                    task_id,
                    len(new_hist_cols),
                    task_stats[key]["rows_with_rate_histogram_after_merge"],
                    len(meta_df),
                )
            else:
                task_stats[key]["rows_with_rate_histogram_after_merge"] = int(len(meta_df))
                log.info("Task %d rate_histogram bins already present; merge skipped.", task_id)

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
    if "file_timestamp_utc" in collected.columns:
        collected = collected.sort_values(
            by="file_timestamp_utc", kind="mergesort", na_position="last"
        ).reset_index(drop=True)
    elif "execution_timestamp_utc" in collected.columns:
        collected = collected.sort_values(
            by="execution_timestamp_utc", kind="mergesort", na_position="last"
        ).reset_index(drop=True)

    out_csv = FILES_DIR / "real_collected_data.csv"
    collected.to_csv(out_csv, index=False)
    log.info("Wrote collected real data: %s (%d rows)", out_csv, len(collected))

    event_col = _make_event_histogram(collected)
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
        "rows_total": int(len(collected)),
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
