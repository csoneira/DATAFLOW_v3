#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
import re
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.selection_config import load_master_event_markers

from common import (
    CANONICAL_EFF_COLUMNS,
    CANONICAL_Z_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    apply_lut_fallback_matches,
    cfg_path,
    derive_trigger_rate_features,
    ensure_output_dirs,
    get_rate_column_name,
    get_trigger_type_selection,
    load_config,
    quantize_efficiency_series,
    read_ascii_lut,
    write_json,
)
from multi_z_support import add_z_config_columns, apply_rate_to_flux_lines, build_rate_to_flux_lines, load_rate_to_flux_lines, load_reference_curve_table, unique_z_vectors, z_mask_for_vector

log = logging.getLogger("another_method.step5")

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILE_TS_RE = re.compile(r"(\d{11})$")
_OFFENDER_RATE_RE = re.compile(
    r"^(?P<scope>plane_combination_filter|strip_combination_filter)_rows_with_(?P<count>\d+)_selected_offenders_rate_hz$"
)

STATIONS_ROOT = REPO_ROOT / "STATIONS"
ONLINE_RUN_DICTIONARY_ROOT = (
    REPO_ROOT
    / "MASTER"
    / "CONFIG_FILES"
    / "STAGE_0"
    / "ONLINE_RUN_DICTIONARY"
)


def _configure_logging() -> None:
    logging.basicConfig(format="[%(levelname)s] STEP_5 - %(message)s", level=logging.INFO, force=True)


def _parse_station_id(raw: object) -> int:
    if raw in (None, "", "null", "None"):
        raise ValueError("step5.station must not be empty.")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    text = str(raw).strip()
    match = re.fullmatch(r"(?i)MINGO(\d{1,2})", text)
    if match is not None:
        return int(match.group(1))
    return int(text)


def _normalize_task_ids(raw: object, fallback: list[int]) -> list[int]:
    if raw in (None, "", "null", "None"):
        return list(fallback)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [int(raw)]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return list(fallback)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = [piece.strip() for piece in text.split(",") if piece.strip()]
        raw = decoded
    if isinstance(raw, (list, tuple)):
        parsed = []
        for value in raw:
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                continue
        return sorted(set(parsed)) or list(fallback)
    return list(fallback)


def _resolve_efficiency_plot_ylim(config: dict[str, Any]) -> tuple[float | None, float | None]:
    step5_config = config.get("step5", {})
    if not isinstance(step5_config, dict):
        step5_config = {}

    raw_ylim = step5_config.get("efficiency_plot_ylim", [0.0, 1.05])
    if isinstance(raw_ylim, str):
        text = raw_ylim.strip()
        if not text:
            raw_ylim = [0.0, 1.05]
        else:
            raw_ylim = json.loads(text)

    if not isinstance(raw_ylim, (list, tuple)) or len(raw_ylim) != 2:
        raise ValueError("step5.efficiency_plot_ylim must be a two-element list like [null, 1.0].")

    limits: list[float | None] = []
    for value in raw_ylim:
        if value in (None, "", "null", "None"):
            limits.append(None)
        else:
            limits.append(float(value))

    bottom, top = limits
    if bottom is not None and top is not None and bottom >= top:
        raise ValueError("step5.efficiency_plot_ylim must satisfy bottom < top when both limits are set.")
    return bottom, top


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", "null", "None"):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_plot_moving_average(config: dict[str, Any]) -> tuple[bool, int]:
    step5_config = config.get("step5", {})
    if not isinstance(step5_config, dict):
        step5_config = {}

    enabled = _normalize_bool(
        step5_config.get(
            "plot_apply_moving_average",
            step5_config.get("efficiency_plot_apply_moving_average", False),
        )
    )
    kernel = int(
        step5_config.get(
            "plot_moving_average_kernel",
            step5_config.get("efficiency_plot_moving_average_kernel", 5),
        )
    )
    if kernel < 1:
        raise ValueError("step5.plot_moving_average_kernel must be >= 1.")
    return enabled, kernel


def _resolve_step5_feature_vector_config(config: dict[str, Any]) -> dict[str, Any]:
    step5_config = config.get("step5", {})
    if not isinstance(step5_config, dict):
        step5_config = {}

    raw_mode = step5_config.get("selected_feature_columns_mode", "minimal_empirical")
    mode_text = "" if raw_mode in (None, "", "null", "None") else str(raw_mode).strip().lower()
    mode_aliases = {
        "default": "minimal_empirical",
        "minimal": "minimal_empirical",
        "per_plane": "minimal_empirical",
        "same_eff": "same_efficiency",
    }
    mode = mode_aliases.get(mode_text, mode_text or "minimal_empirical")
    if mode not in {"minimal_empirical", "same_efficiency"}:
        raise ValueError(
            "step5.selected_feature_columns_mode must be 'minimal_empirical' or 'same_efficiency'."
        )

    if mode != "same_efficiency":
        return {
            "mode": mode,
            "source_columns": list(CANONICAL_EFF_COLUMNS),
            "same_efficiency_planes": None,
        }

    raw_planes = step5_config.get("same_efficiency_planes", step5_config.get("efficiency_reference_planes", [2, 3]))
    if isinstance(raw_planes, str):
        text = raw_planes.strip()
        if not text:
            raw_planes = [2, 3]
        else:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = [piece.strip() for piece in text.split(",") if piece.strip()]
            raw_planes = decoded

    if isinstance(raw_planes, (int, float)) and not isinstance(raw_planes, bool):
        raw_planes = [int(raw_planes)]

    if not isinstance(raw_planes, (list, tuple)) or not raw_planes:
        raise ValueError("step5.same_efficiency_planes must be a non-empty list of plane numbers between 1 and 4.")

    planes: list[int] = []
    for value in raw_planes:
        plane = int(value)
        if plane < 1 or plane > 4:
            raise ValueError(
                "step5.same_efficiency_planes contains an invalid plane index "
                f"{plane}. Valid values are 1, 2, 3, 4."
            )
        if plane not in planes:
            planes.append(plane)

    return {
        "mode": mode,
        "source_columns": [f"emp_eff_{plane}" for plane in planes],
        "same_efficiency_planes": planes,
    }


def _build_step5_query_columns(
    dataframe: pd.DataFrame,
    *,
    efficiency_bin_width: float,
    feature_vector_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = dataframe.copy()
    mode = str(feature_vector_config.get("mode", "minimal_empirical"))
    query_columns: list[str] = []

    if mode == "same_efficiency":
        same_eff_source_columns = list(feature_vector_config["source_columns"])
        same_eff_series = work[same_eff_source_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        raw_columns: list[str] = []
        for column in CANONICAL_EFF_COLUMNS:
            raw_column = f"query_raw_{column}"
            query_column = f"query_{column}"
            work[raw_column] = same_eff_series
            work[query_column] = quantize_efficiency_series(work[raw_column], efficiency_bin_width)
            raw_columns.append(raw_column)
            query_columns.append(query_column)
        return work, query_columns, raw_columns

    raw_columns = list(CANONICAL_EFF_COLUMNS)
    for column in CANONICAL_EFF_COLUMNS:
        query_column = f"query_{column}"
        work[query_column] = quantize_efficiency_series(work[column], efficiency_bin_width)
        query_columns.append(query_column)
    return work, query_columns, raw_columns


def _parse_time_bound(value: object, *, end_of_day: bool) -> pd.Timestamp | None:
    if value in (None, "", "null", "None"):
        return None
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Could not parse datetime bound: {value!r}")
    if end_of_day and _DATE_ONLY_RE.fullmatch(text):
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return pd.Timestamp(parsed)


def _parse_filename_base_ts(value: object) -> pd.Timestamp:
    text = str(value).strip().lower()
    if text.startswith("mini"):
        text = "mi01" + text[4:]
    match = _FILE_TS_RE.search(text)
    if match is None:
        return pd.NaT
    stamp = match.group(1)
    try:
        year = 2000 + int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
        dt = datetime(year, 1, 1) + timedelta(
            days=day_of_year - 1,
            hours=hour,
            minutes=minute,
            seconds=second,
        )
    except ValueError:
        return pd.NaT
    return pd.Timestamp(dt, tz="UTC")


def _parse_execution_timestamp(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    parsed = pd.to_datetime(text, format="%Y-%m-%d_%H.%M.%S", errors="coerce", utc=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], errors="coerce", utc=True)
    return parsed


def _task_metadata_dir(station_id: int, task_id: int) -> Path:
    station = f"MINGO{station_id:02d}"
    return (
        STATIONS_ROOT
        / station
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
    )


def _task_metadata_path(station_id: int, task_id: int, source_name: str) -> Path:
    return _task_metadata_dir(station_id, task_id) / f"task_{task_id}_metadata_{source_name}.csv"


def _aggregate_latest_per_file(dataframe: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    work = dataframe.copy()
    if timestamp_column in work.columns:
        work["_exec_dt"] = _parse_execution_timestamp(work[timestamp_column])
        work = work.sort_values(["filename_base", "_exec_dt"], na_position="last", kind="mergesort")
        work = work.groupby("filename_base").tail(1).drop(columns=["_exec_dt"])
    return work.groupby("filename_base", sort=False).tail(1).reset_index(drop=True)


def _load_task_metadata_source_csv(
    *,
    station_id: int,
    task_id: int,
    source_name: str,
    metadata_agg: str,
    timestamp_column: str,
) -> pd.DataFrame:
    source_path = _task_metadata_path(station_id, task_id, source_name)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing required {source_name} metadata for task {task_id}: {source_path}")
    dataframe = pd.read_csv(source_path, low_memory=False)
    if "filename_base" not in dataframe.columns:
        raise KeyError(f"Task {task_id} {source_name} metadata has no 'filename_base' column: {source_path}")
    if str(metadata_agg).strip().lower() == "latest":
        dataframe = _aggregate_latest_per_file(dataframe, timestamp_column)
    return dataframe


def _merge_sources(
    base_source: tuple[str, pd.DataFrame],
    extra_sources: list[tuple[str, pd.DataFrame]],
    *,
    how: str,
) -> pd.DataFrame:
    merged = base_source[1].copy()
    for source_name, source_df in extra_sources:
        overlap = sorted(set(merged.columns).intersection(set(source_df.columns)) - {"filename_base"})
        renamed = source_df.rename(columns={column: f"{source_name}__{column}" for column in overlap})
        merged = merged.merge(renamed, on="filename_base", how=how)
    return merged


def _online_run_dictionary_path(station_id: int) -> Path:
    suffix = f"{int(station_id):02d}"
    candidates = [
        ONLINE_RUN_DICTIONARY_ROOT / f"STATION_{int(station_id)}" / f"input_file_mingo{suffix}.csv",
        ONLINE_RUN_DICTIONARY_ROOT / f"STATION_{suffix}" / f"input_file_mingo{suffix}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = sorted(ONLINE_RUN_DICTIONARY_ROOT.glob(f"STATION_*/input_file_mingo{suffix}.csv"))
    if found:
        return found[0]
    raise FileNotFoundError(f"ONLINE_RUN_DICTIONARY CSV not found for station {station_id}")


def _load_online_schedule(station_id: int) -> tuple[pd.DataFrame, Path]:
    path = _online_run_dictionary_path(station_id)
    raw = pd.read_csv(path, header=[0, 1], low_memory=False)
    if isinstance(raw.columns, pd.MultiIndex):
        columns = []
        for col in raw.columns:
            top = str(col[0]).strip()
            sub = str(col[1]).strip()
            columns.append(sub if sub and not sub.lower().startswith("unnamed") else top)
        dataframe = raw.copy()
        dataframe.columns = columns
    else:
        dataframe = raw.copy()

    col_by_lower = {str(column).strip().lower(): column for column in dataframe.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            column = col_by_lower.get(name.lower())
            if column is not None:
                return str(column)
        return None

    station_col = pick("station", "detector")
    start_col = pick("start", "date_start")
    end_col = pick("end", "date_end")
    p1_col = pick("p1")
    p2_col = pick("p2")
    p3_col = pick("p3")
    p4_col = pick("p4")

    required = [start_col, p1_col, p2_col, p3_col, p4_col]
    if any(column is None for column in required):
        raise ValueError(f"Could not parse ONLINE_RUN_DICTIONARY schema in {path}")

    work = pd.DataFrame(index=dataframe.index)
    if station_col is not None:
        station_series = pd.to_numeric(dataframe[station_col], errors="coerce")
        work = work.loc[station_series == int(station_id)].copy()

    work["start_utc"] = pd.to_datetime(dataframe[start_col], errors="coerce", utc=True)
    work["end_utc"] = pd.to_datetime(dataframe[end_col], errors="coerce", utc=True) if end_col is not None else pd.NaT
    work["z_tuple"] = dataframe[[p1_col, p2_col, p3_col, p4_col]].apply(
        lambda row: tuple(float(value) for value in row.values),
        axis=1,
    )
    work = work.dropna(subset=["start_utc"]).sort_values(["start_utc", "end_utc"], kind="mergesort")
    return work.reset_index(drop=True), path


def _select_schedule_rows_for_window(
    schedule_df: pd.DataFrame,
    *,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
) -> pd.DataFrame:
    if schedule_df.empty:
        return schedule_df.copy()
    keep = pd.Series(True, index=schedule_df.index)
    far_future = pd.Timestamp("2100-01-01", tz="UTC")
    if date_from is not None:
        keep &= schedule_df["end_utc"].fillna(far_future) >= date_from
    if date_to is not None:
        keep &= schedule_df["start_utc"] <= date_to
    return schedule_df.loc[keep].copy()


def _online_z_tuple_for_timestamp(ts: pd.Timestamp, schedule_df: pd.DataFrame) -> tuple[float, float, float, float] | None:
    if pd.isna(ts) or schedule_df.empty:
        return None
    keep = (schedule_df["start_utc"] <= ts) & (
        schedule_df["end_utc"].isna() | (ts < schedule_df["end_utc"])
    )
    candidates = schedule_df.loc[keep]
    if candidates.empty:
        return None
    row = candidates.sort_values("start_utc", kind="mergesort").iloc[-1]
    return tuple(float(value) for value in row["z_tuple"])


def _resolve_same_or_int(raw: object, *, default_value: int) -> int:
    if raw in (None, "", "null", "None", "same"):
        return int(default_value)
    return int(raw)


def _offender_rate_columns(columns: list[str], scope_preference: str, max_selected_offenders: int) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    for column in columns:
        match = _OFFENDER_RATE_RE.fullmatch(str(column))
        if match is None:
            continue
        scope = match.group("scope")
        count = int(match.group("count"))
        if count > int(max_selected_offenders):
            continue
        priority = 1
        if scope_preference == scope:
            priority = 0
        elif scope_preference == "auto":
            priority = 0 if scope == "plane_combination_filter" else 1
        candidates.append((priority, count, str(column)))
    if not candidates:
        return []
    candidates.sort()
    best_priority = candidates[0][0]
    selected = [column for priority, _count, column in candidates if priority == best_priority]
    selected.sort()
    return selected


def _offender_eff_column(
    columns: list[str],
    plane_idx: int,
    scope_preference: str,
    max_selected_offenders: int,
) -> str | None:
    patterns = []
    offender_limit = int(max_selected_offenders)
    if scope_preference in {"plane_combination_filter", "strip_combination_filter"}:
        patterns.append(
            rf"^{scope_preference}_eff_p{plane_idx}_selected_offenders_le_{offender_limit}$"
        )
    else:
        patterns.extend(
            [
                rf"^plane_combination_filter_eff_p{plane_idx}_selected_offenders_le_{offender_limit}$",
                rf"^strip_combination_filter_eff_p{plane_idx}_selected_offenders_le_{offender_limit}$",
            ]
        )
    for pattern in patterns:
        regex = re.compile(pattern)
        for column in columns:
            if regex.fullmatch(str(column)):
                return str(column)
    return None


def _fill_empirical_eff_from_trigger_ratios(dataframe: pd.DataFrame) -> str | None:
    # Standard empirical efficiency definition: eps_i = R(3-fold without plane i) / R(4-fold).
    base_prefixes = ["clean_to_cal_tt", "raw_to_clean_tt", "cal_tt", "clean_tt", "raw_tt"]
    source_prefixes = ["trigger_type__", ""]

    for source_prefix in source_prefixes:
        for base in base_prefixes:
            prefix = f"{source_prefix}{base}"
            denominator_col = f"{prefix}_1234_rate_hz"
            numerator_cols = {
                1: f"{prefix}_234_rate_hz",
                2: f"{prefix}_134_rate_hz",
                3: f"{prefix}_124_rate_hz",
                4: f"{prefix}_123_rate_hz",
            }
            required = [denominator_col] + [numerator_cols[i] for i in (1, 2, 3, 4)]
            if not all(column in dataframe.columns for column in required):
                continue

            denominator = pd.to_numeric(dataframe[denominator_col], errors="coerce")
            valid_denominator = denominator.where(denominator > 0.0)
            for plane_idx in range(1, 5):
                numerator = pd.to_numeric(dataframe[numerator_cols[plane_idx]], errors="coerce")
                dataframe[f"eff_empirical_{plane_idx}"] = numerator / valid_denominator
            return prefix

    return None


def _resolve_lut_match_settings(
    step5_config: dict[str, Any],
    step3_config: dict[str, Any],
) -> tuple[str, int | None, float]:
    raw_mode = step5_config.get("lut_match_mode", step3_config.get("lut_match_mode"))
    if raw_mode in (None, "", "null", "None"):
        match_mode = "nearest"
    else:
        normalized = str(raw_mode).strip().lower()
        match_mode = {
            "idw": "interpolate",
            "interpolated": "interpolate",
            "interpolation": "interpolate",
        }.get(normalized, normalized)

    interpolation_k_raw = step3_config.get("lut_interpolation_k", 8)
    interpolation_k = None if interpolation_k_raw in (None, "", "null", "None") else int(interpolation_k_raw)

    interpolation_power_raw = step3_config.get("lut_interpolation_power", 2.0)
    interpolation_power = (
        2.0 if interpolation_power_raw in (None, "", "null", "None") else float(interpolation_power_raw)
    )
    return match_mode, interpolation_k, interpolation_power


def _build_query_coverage_diagnostics(
    dataframe: pd.DataFrame,
    lut_diagnostics: pd.DataFrame,
    query_columns: list[str],
) -> pd.DataFrame:
    group_columns = [
        column
        for column in CANONICAL_Z_COLUMNS
        if column in dataframe.columns and column in lut_diagnostics.columns
    ]
    summary_group_columns = [*group_columns, *query_columns]
    summary = (
        dataframe.groupby(summary_group_columns, dropna=False)
        .agg(
            real_row_count=("rate_hz", "size"),
            real_rate_median=("rate_hz", "median"),
            applied_scale_factor_median=("lut_scale_factor", "median"),
            applied_scale_factor_std=("lut_scale_factor", "std"),
            real_emp_eff_1_median=(CANONICAL_EFF_COLUMNS[0], "median"),
            real_emp_eff_2_median=(CANONICAL_EFF_COLUMNS[1], "median"),
            real_emp_eff_3_median=(CANONICAL_EFF_COLUMNS[2], "median"),
            real_emp_eff_4_median=(CANONICAL_EFF_COLUMNS[3], "median"),
        )
        .reset_index()
        .sort_values(
            ["real_row_count", *summary_group_columns],
            ascending=[False] + [True] * len(summary_group_columns),
        )
        .reset_index(drop=True)
    )

    exact_lookup = lut_diagnostics.rename(
        columns={column: f"query_{column}" for column in CANONICAL_EFF_COLUMNS}
    )
    exact_columns = summary_group_columns + ["scale_factor"]
    rename_map = {"scale_factor": "exact_scale_factor"}
    if "n_flux_bins" in exact_lookup.columns:
        exact_columns.append("n_flux_bins")
        rename_map["n_flux_bins"] = "exact_n_flux_bins"
    if "support_rows" in exact_lookup.columns:
        exact_columns.append("support_rows")
        rename_map["support_rows"] = "exact_support_rows"
    summary = summary.merge(
        exact_lookup[exact_columns].rename(columns=rename_map),
        how="left",
        on=summary_group_columns,
    )
    summary["exact_lut_support"] = summary["exact_scale_factor"].notna().astype(int)

    for column in CANONICAL_EFF_COLUMNS:
        summary[f"nearest_lut_{column}"] = np.nan
    summary["nearest_scale_factor"] = np.nan
    summary["nearest_distance"] = np.nan
    if "n_flux_bins" in lut_diagnostics.columns:
        summary["nearest_n_flux_bins"] = np.nan
    if "support_rows" in lut_diagnostics.columns:
        summary["nearest_support_rows"] = np.nan
    summary["lut_rows_within_distance_0_10"] = 0
    summary["lut_rows_within_distance_0_15"] = 0

    if not group_columns:
        lut_matrix = lut_diagnostics[CANONICAL_EFF_COLUMNS].to_numpy(dtype=float)
        query_matrix = summary[query_columns].to_numpy(dtype=float)
        distances = np.sqrt(((query_matrix[:, None, :] - lut_matrix[None, :, :]) ** 2).sum(axis=2))
        best_indices = np.argmin(distances, axis=1)
        best_distances = distances[np.arange(len(best_indices)), best_indices]
        nearest_rows = lut_diagnostics.iloc[best_indices].reset_index(drop=True)

        for column in CANONICAL_EFF_COLUMNS:
            summary[f"nearest_lut_{column}"] = nearest_rows[column].to_numpy(dtype=float)
        summary["nearest_scale_factor"] = nearest_rows["scale_factor"].to_numpy(dtype=float)
        summary["nearest_distance"] = best_distances
        if "n_flux_bins" in nearest_rows.columns:
            summary["nearest_n_flux_bins"] = nearest_rows["n_flux_bins"].to_numpy(dtype=float)
        if "support_rows" in nearest_rows.columns:
            summary["nearest_support_rows"] = nearest_rows["support_rows"].to_numpy(dtype=float)
        summary["lut_rows_within_distance_0_10"] = (distances <= 0.10 + 1e-9).sum(axis=1)
        summary["lut_rows_within_distance_0_15"] = (distances <= 0.15 + 1e-9).sum(axis=1)
        return summary

    for z_vector in unique_z_vectors(summary, z_columns=group_columns):
        summary_mask = z_mask_for_vector(summary, z_vector, z_columns=group_columns)
        lut_mask = z_mask_for_vector(lut_diagnostics, z_vector, z_columns=group_columns)
        lut_subset = lut_diagnostics.loc[lut_mask].reset_index(drop=True)
        if lut_subset.empty:
            continue
        query_matrix = summary.loc[summary_mask, query_columns].to_numpy(dtype=float)
        lut_matrix = lut_subset[CANONICAL_EFF_COLUMNS].to_numpy(dtype=float)
        distances = np.sqrt(((query_matrix[:, None, :] - lut_matrix[None, :, :]) ** 2).sum(axis=2))
        best_indices = np.argmin(distances, axis=1)
        best_distances = distances[np.arange(len(best_indices)), best_indices]
        nearest_rows = lut_subset.iloc[best_indices].reset_index(drop=True)
        summary.loc[summary_mask, "nearest_scale_factor"] = nearest_rows["scale_factor"].to_numpy(dtype=float)
        summary.loc[summary_mask, "nearest_distance"] = best_distances
        summary.loc[summary_mask, "lut_rows_within_distance_0_10"] = (distances <= 0.10 + 1e-9).sum(axis=1)
        summary.loc[summary_mask, "lut_rows_within_distance_0_15"] = (distances <= 0.15 + 1e-9).sum(axis=1)
        for column in CANONICAL_EFF_COLUMNS:
            summary.loc[summary_mask, f"nearest_lut_{column}"] = nearest_rows[column].to_numpy(dtype=float)
        if "n_flux_bins" in nearest_rows.columns:
            summary.loc[summary_mask, "nearest_n_flux_bins"] = nearest_rows["n_flux_bins"].to_numpy(dtype=float)
        if "support_rows" in nearest_rows.columns:
            summary.loc[summary_mask, "nearest_support_rows"] = nearest_rows["support_rows"].to_numpy(dtype=float)
    return summary


def _extract_lut_z_vector(lut_meta: dict[str, Any], lut_comments: list[str]) -> tuple[float, float, float, float] | None:
    selected = lut_meta.get("selected_z_positions")
    if isinstance(selected, list) and len(selected) == 4:
        return tuple(float(value) for value in selected)

    for comment in lut_comments:
        if not comment.startswith("# z_positions:"):
            continue
        payload = comment.split(":", 1)[1].strip()
        values = [piece.strip() for piece in payload.split(",")]
        if len(values) != 4:
            continue
        try:
            return tuple(float(value) for value in values)
        except ValueError:
            continue
    return None


def _prepare_z_quality_columns(
    dataframe: pd.DataFrame,
    lut_z_vector: tuple[float, float, float, float] | None,
) -> tuple[pd.DataFrame, list[list[float]], str]:
    work = dataframe.copy()

    normalized_cols = [f"z_pos_{idx}" for idx in range(1, 5)]
    online_cols = [f"online_z_plane_{idx}" for idx in range(1, 5)]
    specific_cols = [f"z_P{idx}" for idx in range(1, 5)]

    if all(column in work.columns for column in normalized_cols):
        z_columns = normalized_cols
        z_source = "normalized_real_slice"
    elif all(column in work.columns for column in online_cols):
        z_columns = online_cols
        z_source = "online_run_dictionary"
    elif all(column in work.columns for column in specific_cols):
        z_columns = specific_cols
        z_source = "task_specific_metadata"
    else:
        work["selected_z_vector_match"] = pd.NA
        return work, [], "unavailable"

    z_numeric = work[z_columns].apply(pd.to_numeric, errors="coerce")
    unique_real_z: list[list[float]] = []
    if z_numeric.notna().all(axis=1).any():
        unique_real_z = [
            [float(value) for value in row]
            for row in z_numeric.dropna().drop_duplicates().itertuples(index=False, name=None)
        ]

    if lut_z_vector is None:
        work["selected_z_vector_match"] = pd.NA
        return work, unique_real_z, z_source

    matches = np.ones(len(work), dtype=bool)
    valid_mask = z_numeric.notna().all(axis=1).to_numpy(dtype=bool)
    for idx, value in enumerate(lut_z_vector):
        matches &= np.isclose(z_numeric.iloc[:, idx].to_numpy(dtype=float), float(value), equal_nan=False)
    work["selected_z_vector_match"] = np.where(valid_mask, matches, np.nan)
    return work, unique_real_z, z_source


def _resolve_time_axis(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    for candidate in ("file_timestamp_utc", "execution_timestamp_utc", "time_utc"):
        if candidate not in dataframe.columns:
            continue
        parsed = pd.to_datetime(dataframe[candidate], errors="coerce", utc=True)
        if parsed.notna().any():
            order = np.argsort(parsed.fillna(parsed.min()))
            ordered = dataframe.iloc[order].reset_index(drop=True)
            ordered_time = parsed.iloc[order].dt.tz_convert(None).reset_index(drop=True)
            return ordered, ordered_time, candidate

    ordered = dataframe.reset_index(drop=True).copy()
    return ordered, pd.Series(np.arange(len(ordered)), dtype=float), "row_index"


def _prepare_plot_frame(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    *,
    apply_moving_average: bool,
    moving_average_kernel: int,
) -> pd.DataFrame:
    plot_frame = dataframe.copy()
    available_columns = [column for column in numeric_columns if column in plot_frame.columns]
    if available_columns:
        plot_frame[available_columns] = plot_frame[available_columns].apply(pd.to_numeric, errors="coerce")
    if not apply_moving_average or moving_average_kernel <= 1 or not available_columns:
        return plot_frame

    plot_frame[available_columns] = plot_frame[available_columns].rolling(
        window=moving_average_kernel,
        min_periods=1,
        center=True,
    ).mean()
    return plot_frame


def _load_event_markers_for_station(station_id: int) -> list[dict[str, object]]:
    return [
        {"time": marker.time, "label": marker.label}
        for marker in load_master_event_markers(station=station_id)
    ]


def _add_event_markers(
    axes: list[plt.Axes] | tuple[plt.Axes, ...],
    x_values: pd.Series,
    event_markers: list[dict[str, object]],
) -> None:
    if not event_markers or not pd.api.types.is_datetime64_any_dtype(x_values):
        return

    valid_times = pd.to_datetime(x_values, errors="coerce", utc=True).dropna()
    if valid_times.empty:
        return
    valid_times = valid_times.dt.tz_convert(None)
    min_time = valid_times.min()
    max_time = valid_times.max()

    label_levels = [0.98, 0.86, 0.74]
    top_axis = axes[0]
    for index, marker in enumerate(event_markers):
        event_time = pd.to_datetime(marker["time"], errors="coerce", utc=True)
        if pd.isna(event_time):
            continue
        event_time = event_time.tz_convert(None)
        if event_time < min_time or event_time > max_time:
            continue
        level = label_levels[index % len(label_levels)]
        for axis in axes:
            axis.axvline(
                event_time,
                color="black",
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
            )
        top_axis.annotate(
            str(marker["label"]),
            xy=(event_time, level),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )


def _plot_real_rate_correction(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
    event_markers: list[dict[str, object]],
    efficiency_plot_ylim: tuple[float | None, float | None],
    apply_moving_average: bool,
    moving_average_kernel: int,
) -> str:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    ordered = _prepare_plot_frame(
        ordered,
        ["rate_hz", "corrected_rate_to_perfect_hz", *CANONICAL_EFF_COLUMNS],
        apply_moving_average=apply_moving_average,
        moving_average_kernel=moving_average_kernel,
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        height_ratios=[2.0, 1.2],
    )

    axes[0].plot(x_values,
        ordered["rate_hz"],
        marker="o",
        linewidth=0.35,
        markersize=3.0,
        label="Observed rate")
    axes[0].plot(
        x_values,
        ordered["corrected_rate_to_perfect_hz"],
        marker="o",
        linewidth=0.35,
        markersize=3.0,
        label="LUT-corrected rate",
    )
    axes[0].set_ylabel("Rate [Hz]")
    axes[0].set_title(
        "Observed and corrected real-data rate\n"
        f"rate column: {rate_column_name}"
    )
    if apply_moving_average and moving_average_kernel > 1:
        axes[0].set_title(axes[0].get_title() + f" | moving average = {moving_average_kernel}")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    plane_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(CANONICAL_EFF_COLUMNS):
        axes[1].plot(
            x_values,
            ordered[column],
            marker="o",
            linewidth=0.45,
            markersize=2.8,
            color=plane_colors[idx],
            label=f"Plane {idx + 1} eff",
        )
    axes[1].set_xlabel(x_label.replace("_", " "))
    axes[1].set_ylabel("Empirical efficiency")
    eff_y_min, eff_y_max = efficiency_plot_ylim
    axes[1].set_ylim(bottom=eff_y_min, top=eff_y_max)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)

    _add_event_markers(list(axes), x_values, event_markers)

    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return x_label


def _plot_real_correction_diagnostics(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
    event_markers: list[dict[str, object]],
    apply_moving_average: bool,
    moving_average_kernel: int,
) -> str:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    sequence = np.arange(len(ordered))
    plot_frame = _prepare_plot_frame(
        ordered,
        ["lut_scale_factor"],
        apply_moving_average=apply_moving_average,
        moving_average_kernel=moving_average_kernel,
    )

    # Euclidean distance in 4D efficiency space: 0 means perfect LUT-coordinate agreement.
    eff_values = ordered[CANONICAL_EFF_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lut_eff_values = ordered[[f"lut_{column}" for column in CANONICAL_EFF_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid_eff = np.isfinite(eff_values).all(axis=1) & np.isfinite(lut_eff_values).all(axis=1)
    lut_eff_distance = np.full(len(ordered), np.nan, dtype=float)
    if valid_eff.any():
        lut_eff_distance[valid_eff] = np.linalg.norm(eff_values[valid_eff] - lut_eff_values[valid_eff], axis=1)
    # Max distance is 2.0 because each of 4 efficiencies lives in [0,1].
    trust_score = np.clip(1.0 - (lut_eff_distance / 2.0), 0.0, 1.0)
    if apply_moving_average and moving_average_kernel > 1:
        diagnostic_frame = pd.DataFrame(
            {
                "lut_eff_distance": lut_eff_distance,
                "trust_score": trust_score,
            }
        ).rolling(window=moving_average_kernel, min_periods=1, center=True).mean()
        lut_eff_distance = diagnostic_frame["lut_eff_distance"].to_numpy(dtype=float)
        trust_score = diagnostic_frame["trust_score"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    observed_axis = fig.add_subplot(grid[0, 0])
    proximity_axis = fig.add_subplot(grid[0, 1])
    scale_axis = fig.add_subplot(grid[1, :])

    scatter = observed_axis.scatter(
        ordered["rate_hz"],
        ordered["corrected_rate_to_perfect_hz"],
        c=sequence,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    finite_rates = pd.concat(
        [ordered["rate_hz"], ordered["corrected_rate_to_perfect_hz"]],
        ignore_index=True,
    ).to_numpy(dtype=float)
    finite_rates = finite_rates[np.isfinite(finite_rates)]
    if finite_rates.size:
        low = float(np.min(finite_rates))
        high = float(np.max(finite_rates))
        observed_axis.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="black", alpha=0.7)
    observed_axis.set_title(f"Observed vs corrected rate\nrate column: {rate_column_name}")
    observed_axis.set_xlabel(f"Observed rate [Hz]\n({rate_column_name})")
    observed_axis.set_ylabel(f"Corrected rate [Hz]\n(from {rate_column_name})")
    observed_axis.grid(alpha=0.25)

    scale_axis.plot(
        x_values,
        plot_frame["lut_scale_factor"],
        marker="o",
        linewidth=1.3,
        markersize=3.0,
        color="#8B1E3F",
        label="LUT scale factor",
    )
    scale_axis.set_title(
        "Scale factor vs time"
        + (f" | moving average = {moving_average_kernel}" if apply_moving_average and moving_average_kernel > 1 else "")
    )
    scale_axis.set_xlabel(x_label.replace("_", " "))
    scale_axis.set_ylabel("Scale factor")
    scale_axis.grid(alpha=0.25)
    scale_axis.legend()

    proximity_axis.plot(
        x_values,
        lut_eff_distance,
        marker="o",
        linewidth=1.4,
        markersize=3.0,
        color="#1F6FEB",
        label="LUT efficiency distance",
    )
    proximity_axis.set_title(
        "LUT proximity / trust vs time"
        + (f" | moving average = {moving_average_kernel}" if apply_moving_average and moving_average_kernel > 1 else "")
    )
    proximity_axis.set_xlabel(x_label.replace("_", " "))
    proximity_axis.set_ylabel("Distance in eff space")
    proximity_axis.grid(alpha=0.25)

    trust_axis = proximity_axis.twinx()
    trust_axis.plot(
        x_values,
        trust_score,
        marker="s",
        linewidth=1.3,
        markersize=2.8,
        color="#E67E22",
        alpha=0.8,
        label="Trust score",
    )
    trust_axis.set_ylim(0.0, 1.05)
    trust_axis.set_ylabel("Trust [0-1]")

    distance_lines, distance_labels = proximity_axis.get_legend_handles_labels()
    trust_lines, trust_labels = trust_axis.get_legend_handles_labels()
    proximity_axis.legend(distance_lines + trust_lines, distance_labels + trust_labels, loc="upper right")

    _add_event_markers([scale_axis, proximity_axis], x_values, event_markers)

    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()

    cbar = fig.colorbar(scatter, ax=[observed_axis, proximity_axis, scale_axis], shrink=0.92)
    cbar.set_label("Time-series order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return x_label


def _plot_query_coverage_match_distance_by_efficiencies(
    query_coverage: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    query_columns = [f"query_{column}" for column in CANONICAL_EFF_COLUMNS]
    plane_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (column, ax) in enumerate(zip(query_columns, axes.flat)):
        grouped = (
            query_coverage.groupby(column, dropna=False, sort=True)
            .agg(
                median_nearest_distance=("nearest_distance", "median"),
                exact_support_fraction=("exact_lut_support", "mean"),
            )
            .reset_index()
        )

        if grouped.empty:
            ax.set_title(f"No data for {column}")
            ax.axis("off")
            continue

        x = grouped[column].astype(float).to_numpy()
        y = grouped["median_nearest_distance"].astype(float).to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.6, color=plane_colors[idx], label="Median nearest distance")
        ax.set_title(f"Nearest LUT distance vs {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Median nearest distance")
        ax.grid(alpha=0.25)
        if y.size:
            y_max = float(np.nanmax(y))
            ax.set_ylim(0.0, max(0.15, y_max * 1.05))

        ax2 = ax.twinx()
        ax2.plot(
            x,
            grouped["exact_support_fraction"].to_numpy(dtype=float),
            color="#444444",
            linestyle="--",
            linewidth=1.3,
            alpha=0.9,
            label="Exact support fraction",
        )
        ax2.set_ylabel("Exact support fraction", color="#444444")
        ax2.set_ylim(0.0, 1.05)
        if idx == 0:
            combined_lines, combined_labels = ax.get_legend_handles_labels()
            extra_lines, extra_labels = ax2.get_legend_handles_labels()
            ax.legend(combined_lines + extra_lines, combined_labels + extra_labels, loc="upper right", fontsize=8)

    fig.suptitle(
        "Query coverage diagnostics by efficiency plane\n"
        f"rate column: {rate_column_name}",
        y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_lut_empirical_support_for_plot(lut_support: pd.DataFrame) -> pd.DataFrame:
    lut_empirical_columns = [f"eff_empirical_{idx}" for idx in range(1, 5)]
    if all(column in lut_support.columns for column in lut_empirical_columns):
        return (
            lut_support[lut_empirical_columns]
            .rename(columns=dict(zip(lut_empirical_columns, CANONICAL_EFF_COLUMNS)))
            .apply(pd.to_numeric, errors="coerce")
        )
    return lut_support[CANONICAL_EFF_COLUMNS].apply(pd.to_numeric, errors="coerce")


def _plot_lut_vs_real_efficiency_coverage(
    merged: pd.DataFrame,
    lut_diagnostics: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), constrained_layout=True)
    eff_columns = CANONICAL_EFF_COLUMNS
    lut_data = _select_lut_empirical_support_for_plot(lut_diagnostics)
    real_data = merged[eff_columns].apply(pd.to_numeric, errors="coerce")

    pair_layout = [
        [(0, 1), None, None],
        [(0, 2), (1, 2), None],
        [(0, 3), (1, 3), (2, 3)],
    ]

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            pair = pair_layout[row][col]
            if pair is None:
                ax.axis("off")
                continue

            x_idx, y_idx = pair
            x_col = eff_columns[x_idx]
            y_col = eff_columns[y_idx]
            ax.scatter(
                lut_data[x_col],
                lut_data[y_col],
                s=30,
                alpha=0.55,
                color="#2ca02c",
                label="LUT rows",
                edgecolors="none",
            )
            ax.scatter(
                real_data[x_col],
                real_data[y_col],
                s=22,
                alpha=0.65,
                color="#1f77b4",
                label="Real rows",
                edgecolors="none",
            )
            x_values = np.concatenate(
                [lut_data[x_col].to_numpy(dtype=float), real_data[x_col].to_numpy(dtype=float)]
            )
            y_values = np.concatenate(
                [lut_data[y_col].to_numpy(dtype=float), real_data[y_col].to_numpy(dtype=float)]
            )
            finite_x = x_values[np.isfinite(x_values)]
            finite_y = y_values[np.isfinite(y_values)]
            if finite_x.size:
                x_min = float(np.min(finite_x))
                x_max = float(np.max(finite_x))
                x_pad = max((x_max - x_min) * 0.03, 0.01)
                ax.set_xlim(x_min - x_pad, x_max + x_pad)
            if finite_y.size:
                y_min = float(np.min(finite_y))
                y_max = float(np.max(finite_y))
                y_pad = max((y_max - y_min) * 0.03, 0.01)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(alpha=0.2)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "LUT coverage vs real-data empirical efficiencies\n"
        f"rate column: {rate_column_name}",
        y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_real_rate_vs_efficiencies_2x2(
    merged: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    eff_columns = CANONICAL_EFF_COLUMNS
    rate_original = pd.to_numeric(merged["rate_hz"], errors="coerce")
    rate_corrected = pd.to_numeric(merged["corrected_rate_to_perfect_hz"], errors="coerce")
    x_line_end = 1.01

    def _add_linear_trend_line(ax: plt.Axes, x_values: pd.Series, y_values: pd.Series, color: str, label: str) -> None:
        valid = x_values.notna() & y_values.notna()
        if int(valid.sum()) < 2:
            return
        x = x_values.loc[valid].to_numpy(dtype=float)
        y = y_values.loc[valid].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        x_start = float(np.nanmin(x))
        x_line = np.linspace(x_start, x_line_end, 120)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linestyle="--", linewidth=1.6, color=color, alpha=0.8, label=label)

    all_y = pd.concat([rate_original, rate_corrected], ignore_index=True)
    finite_y = all_y[np.isfinite(all_y)]
    if finite_y.size:
        y_min = float(np.nanmin(finite_y))
        y_max = float(np.nanmax(finite_y))
    else:
        y_min, y_max = 0.0, 1.0
    y_pad = max((y_max - y_min) * 0.05, 0.1)

    for idx, ax in enumerate(axes.flat):
        eff_col = eff_columns[idx]
        x_values = pd.to_numeric(merged[eff_col], errors="coerce")

        ax.scatter(
            x_values,
            rate_original,
            s=24,
            alpha=0.65,
            color="#1f77b4",
            label="Original rate",
            edgecolors="none",
        )
        ax.scatter(
            x_values,
            rate_corrected,
            s=24,
            alpha=0.65,
            color="#ff7f0e",
            label="Corrected rate",
            edgecolors="none",
        )

        _add_linear_trend_line(ax, x_values, rate_original, "#1f77b4", "Original trend")
        _add_linear_trend_line(ax, x_values, rate_corrected, "#ff7f0e", "Corrected trend")
        ax.axvline(1.0, linestyle=":", linewidth=1.5, color="black", alpha=0.8)

        ax.set_title(f"Rate vs {eff_col}")
        ax.set_xlabel(eff_col)
        ax.set_ylabel("Rate [Hz]")
        ax.grid(alpha=0.25)
        ax.set_xlim(0.0, x_line_end)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if idx == 0:
            ax.legend()

    fig.suptitle(
        "Original and corrected rate vs empirical efficiencies\n"
        f"rate column: {rate_column_name}",
        y=1.02,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_corrected_flux_from_rate(
    dataframe: pd.DataFrame,
    line_table: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
    event_markers: list[dict[str, object]],
    apply_moving_average: bool,
    moving_average_kernel: int,
) -> None:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    ordered = _prepare_plot_frame(
        ordered,
        ["corrected_flux_cm2_min"],
        apply_moving_average=apply_moving_average,
        moving_average_kernel=moving_average_kernel,
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    color_map = plt.get_cmap("tab10")

    if "z_config_id" in ordered.columns and ordered["z_config_id"].notna().any():
        group_values = [value for value in ordered["z_config_id"].dropna().unique().tolist()]
    else:
        group_values = ["all"]
        ordered = ordered.copy()
        ordered["z_config_id"] = "all"

    for idx, group_value in enumerate(group_values):
        subset = ordered.loc[ordered["z_config_id"] == group_value].copy()
        if subset.empty:
            continue
        color = color_map(idx % 10)
        axes[0].plot(
            x_values.loc[subset.index],
            subset["corrected_flux_cm2_min"],
            marker="o",
            linewidth=0.8,
            markersize=2.8,
            color=color,
            label=str(group_value),
        )
        axes[1].scatter(
            subset["corrected_rate_to_perfect_hz"],
            subset["corrected_flux_cm2_min"],
            s=22,
            alpha=0.75,
            color=color,
            label=str(group_value),
            edgecolors="none",
        )
        if "z_config_id" in line_table.columns:
            line_subset = line_table.loc[line_table["z_config_id"] == group_value].copy()
        else:
            line_subset = line_table.copy()
        if line_subset.empty and len(line_table) == 1:
            line_subset = line_table.copy()
        if line_subset.empty:
            continue
        line = line_subset.iloc[0]
        x_candidates = [
            float(value)
            for value in [
                pd.to_numeric(subset["corrected_rate_to_perfect_hz"], errors="coerce").min(),
                pd.to_numeric(subset["corrected_rate_to_perfect_hz"], errors="coerce").max(),
                line.get("reference_rate_min"),
                line.get("reference_rate_max"),
            ]
            if pd.notna(value)
        ]
        if len(x_candidates) < 2:
            continue
        x_line = np.linspace(min(x_candidates), max(x_candidates), 120)
        y_line = float(line["slope"]) * x_line + float(line["intercept"])
        axes[1].plot(
            x_line,
            y_line,
            linewidth=1.8,
            color=color,
            alpha=0.95,
        )

    axes[0].set_title(
        "Corrected flux inferred from Step 2 rate-to-flux reference"
        + (f" | moving average = {moving_average_kernel}" if apply_moving_average and moving_average_kernel > 1 else "")
    )
    axes[0].set_xlabel(x_label.replace("_", " "))
    axes[0].set_ylabel("Corrected flux [cm^-2 min^-1]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(title="z config", fontsize=8)

    axes[1].set_title(
        "Corrected rate to inferred flux mapping (linear rate-to-flux fit)\n"
        f"rate column: {rate_column_name}"
    )
    axes[1].set_xlabel("Corrected rate [Hz]")
    axes[1].set_ylabel("Corrected flux [cm^-2 min^-1]")
    axes[1].grid(alpha=0.25)

    _add_event_markers([axes[0]], x_values, event_markers)
    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _collect_real_data_slice(
    *,
    config: dict[str, Any],
    station_id: int,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
    min_events: float | None,
    metadata_agg: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    online_schedule_all, online_schedule_path = _load_online_schedule(station_id)
    online_schedule_window = _select_schedule_rows_for_window(
        online_schedule_all,
        date_from=date_from,
        date_to=date_to,
    )
    trigger_selection = get_trigger_type_selection(config)
    task_id = int(trigger_selection.get("metadata_task_id", trigger_selection["task_id"]))
    source_name = str(trigger_selection.get("source_name", "trigger_type"))
    task_stats: dict[str, dict[str, int]] = {
        str(task_id): {
            "rows_after_metadata_merge": 0,
            "rows_after_date_filter": 0,
            "rows_with_online_z_mapped": 0,
        }
    }

    trigger_df = _load_task_metadata_source_csv(
        station_id=station_id,
        task_id=task_id,
        source_name=source_name,
        metadata_agg=metadata_agg,
        timestamp_column=timestamp_column,
    )
    merged, trigger_info = derive_trigger_rate_features(
        trigger_df,
        config,
        allow_plain_fallback=False,
    )
    task_stats[str(task_id)]["rows_after_metadata_merge"] = int(len(merged))

    merged["file_timestamp_utc"] = merged["filename_base"].map(_parse_filename_base_ts)
    if timestamp_column in merged.columns:
        merged["execution_timestamp_utc"] = _parse_execution_timestamp(merged[timestamp_column])
    else:
        merged["execution_timestamp_utc"] = pd.NaT

    keep = merged["file_timestamp_utc"].notna()
    if date_from is not None:
        keep &= merged["file_timestamp_utc"] >= date_from
    if date_to is not None:
        keep &= merged["file_timestamp_utc"] <= date_to
    collected = merged.loc[keep].copy()
    task_stats[str(task_id)]["rows_after_date_filter"] = int(len(collected))
    if collected.empty:
        raise ValueError("No real rows were collected for the requested station/date window.")

    rate_values = pd.to_numeric(collected["rate_hz"], errors="coerce")
    four_plane_values = pd.to_numeric(collected.get("four_plane_rate_hz"), errors="coerce")
    eff_frame = collected[[f"eff_empirical_{idx}" for idx in range(1, 5)]].apply(pd.to_numeric, errors="coerce")
    valid_trigger_mask = np.isfinite(rate_values) & (rate_values > 0.0)
    if "four_plane_rate_hz" in collected.columns:
        valid_trigger_mask &= np.isfinite(four_plane_values) & (four_plane_values > 0.0)
    valid_trigger_mask &= np.isfinite(eff_frame.to_numpy()).all(axis=1)
    collected = collected.loc[valid_trigger_mask].copy()
    if collected.empty:
        selected_rate_name = str(trigger_info.get("selected_source_rate_column", trigger_info["rate_family_column"]))
        raise ValueError(
            "No real rows remain after deriving rate-source features for "
            f"{selected_rate_name}. The selected metadata columns currently yield "
            "no positive four-plane support with finite empirical efficiencies."
        )

    online_z = collected["file_timestamp_utc"].map(
        lambda ts: _online_z_tuple_for_timestamp(ts, online_schedule_window)
    )
    task_stats[str(task_id)]["rows_with_online_z_mapped"] = int(online_z.notna().sum())
    z_rows = [
        [np.nan, np.nan, np.nan, np.nan]
        if value is None
        else [float(item) for item in value]
        for value in online_z
    ]
    z_split = pd.DataFrame(
        z_rows,
        columns=["online_z_plane_1", "online_z_plane_2", "online_z_plane_3", "online_z_plane_4"],
        index=collected.index,
    )
    collected = pd.concat([collected, z_split], axis=1)
    collected["task_id"] = int(task_id)
    collected["station_id"] = int(station_id)

    rows_before_event_cut = int(len(collected))
    event_values = pd.to_numeric(collected.get("selected_rate_count"), errors="coerce")
    event_source = "selected_rate_count" if event_values.notna().any() else None

    if event_source is not None:
        collected["n_events"] = event_values
    if min_events is not None and event_source is not None:
        collected = collected.loc[event_values >= float(min_events)].copy()
    rows_after_event_cut = int(len(collected))
    if collected.empty:
        raise ValueError("No real rows remain after Step 5 event-count filtering.")

    sort_column = "file_timestamp_utc" if "file_timestamp_utc" in collected.columns else "execution_timestamp_utc"
    collected = collected.sort_values(sort_column, kind="mergesort").reset_index(drop=True)

    metadata = {
        "online_run_dictionary_csv": str(online_schedule_path),
        "online_schedule_rows_total": int(len(online_schedule_all)),
        "online_schedule_rows_in_requested_window": int(len(online_schedule_window)),
        "online_schedule_z_tuples_in_requested_window": [
            list(z_tuple)
            for z_tuple in sorted(set(online_schedule_window["z_tuple"].dropna().tolist()))
        ] if not online_schedule_window.empty else [],
        "rows_before_event_cut": rows_before_event_cut,
        "rows_after_event_cut": rows_after_event_cut,
        "event_count_source_for_filter": event_source,
        "task_ids_used": [task_id],
        "task_stats": task_stats,
        "trigger_rate_selection": trigger_info,
        "trigger_type_selection": trigger_info,
        "offender_count_semantics": (
            "cumulative_total_problematic_offender_count_from_trigger_type"
            if str(trigger_info.get("metadata_source", "trigger_type")) == "trigger_type"
            else "not_applicable_for_robust_efficiency_metadata"
        ),
    }
    return collected, metadata


def _rename_real_columns(dataframe: pd.DataFrame, rate_column_name: str) -> pd.DataFrame:
    rename_map = {
        "eff_empirical_1": CANONICAL_EFF_COLUMNS[0],
        "eff_empirical_2": CANONICAL_EFF_COLUMNS[1],
        "eff_empirical_3": CANONICAL_EFF_COLUMNS[2],
        "eff_empirical_4": CANONICAL_EFF_COLUMNS[3],
    }
    if "rate_hz" not in dataframe.columns and rate_column_name in dataframe.columns:
        rename_map[rate_column_name] = "rate_hz"
    for idx in range(1, 5):
        online_column = f"online_z_plane_{idx}"
        specific_column = f"z_P{idx}"
        target_column = f"z_pos_{idx}"
        if online_column in dataframe.columns:
            rename_map[online_column] = target_column
        elif specific_column in dataframe.columns:
            rename_map[specific_column] = target_column
    return dataframe.rename(columns=rename_map)


def run(config_path: str | Path | None = None) -> Path:
    _configure_logging()
    ensure_output_dirs()
    config = load_config(config_path)

    step5_config = config.get("step5", {})
    if not isinstance(step5_config, dict):
        step5_config = {}
    step3_config = config.get("step3", {})
    if not isinstance(step3_config, dict):
        step3_config = {}

    station_id = _parse_station_id(step5_config.get("station", "MINGO01"))
    station_name = f"MINGO{station_id:02d}"
    date_from = _parse_time_bound(step5_config.get("date_from"), end_of_day=False)
    date_to = _parse_time_bound(step5_config.get("date_to"), end_of_day=True)
    min_events_raw = step5_config.get("min_events")
    min_events = None if min_events_raw in (None, "", "null", "None") else float(min_events_raw)
    metadata_agg = str(step5_config.get("metadata_agg", "latest")).strip().lower()
    timestamp_column = str(step5_config.get("timestamp_column", "execution_timestamp"))
    efficiency_plot_ylim = _resolve_efficiency_plot_ylim(config)
    apply_plot_moving_average, plot_moving_average_kernel = _resolve_plot_moving_average(config)

    trigger_selection = get_trigger_type_selection(config)
    task_ids = [int(trigger_selection.get("metadata_task_id", trigger_selection["task_id"]))]
    rate_column_name = str(trigger_selection["selected_source_rate_column"])

    lut_path = cfg_path(config, "paths", "step2_lut_ascii")
    lut_meta_path = cfg_path(config, "paths", "step2_meta_json")
    flux_cells_path = cfg_path(config, "paths", "step2_flux_cells_csv")
    rate_to_flux_lines_path = cfg_path(config, "paths", "step2_rate_to_flux_lines_csv")
    lut_diag_path = cfg_path(config, "paths", "step2_lut_diagnostics_csv")
    output_path = cfg_path(config, "paths", "step5_output_csv")
    meta_path = cfg_path(config, "paths", "step5_meta_json")

    real_dataframe, collection_meta = _collect_real_data_slice(
        config=config,
        station_id=station_id,
        date_from=date_from,
        date_to=date_to,
        min_events=min_events,
        metadata_agg=metadata_agg,
        timestamp_column=timestamp_column,
    )

    if "rate_hz" not in real_dataframe.columns and rate_column_name not in real_dataframe.columns:
        raise ValueError(
            f"Configured rate column '{rate_column_name}' is not present in the collected real-data slice."
        )

    work = _rename_real_columns(real_dataframe.copy(), rate_column_name)
    required_columns = set(CANONICAL_EFF_COLUMNS + ["rate_hz"])
    missing_required = sorted(column for column in required_columns if column not in work.columns)
    if missing_required:
        raise ValueError(
            "Step 5 real-data slice is missing required columns after transformation: "
            + ", ".join(missing_required)
        )
    work = add_z_config_columns(work, z_columns=CANONICAL_Z_COLUMNS)

    lut_dataframe, lut_comments = read_ascii_lut(lut_path)
    lut_meta = {}
    if lut_meta_path.exists():
        lut_meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
    efficiency_bin_width = float(
        lut_meta.get("efficiency_bin_width", config.get("step2", {}).get("efficiency_bin_width", 0.02))
    )
    feature_vector_config = _resolve_step5_feature_vector_config(config)
    work, query_columns, raw_query_columns = _build_step5_query_columns(
        work,
        efficiency_bin_width=efficiency_bin_width,
        feature_vector_config=feature_vector_config,
    )

    lut_has_z = all(column in lut_dataframe.columns for column in CANONICAL_Z_COLUMNS)
    if lut_has_z:
        missing_z_columns = [column for column in CANONICAL_Z_COLUMNS if column not in work.columns]
        if missing_z_columns:
            raise ValueError(
                "The combined LUT requires z-position columns in the real dataframe, but these are missing: "
                + ", ".join(missing_z_columns)
            )

    lut_lookup = lut_dataframe.rename(columns={column: f"lut_{column}" for column in CANONICAL_EFF_COLUMNS})
    left_on = list(query_columns)
    right_on = [f"lut_{column}" for column in CANONICAL_EFF_COLUMNS]
    if lut_has_z:
        left_on = [*CANONICAL_Z_COLUMNS, *left_on]
        right_on = [*CANONICAL_Z_COLUMNS, *right_on]
    merged = work.merge(
        lut_lookup,
        how="left",
        left_on=left_on,
        right_on=right_on,
    )
    merged = merged.rename(columns={"scale_factor": "lut_scale_factor"})
    merged["lut_match_method"] = np.where(merged["lut_scale_factor"].notna(), "exact", pd.NA)
    merged["lut_match_distance"] = np.where(merged["lut_scale_factor"].notna(), 0.0, np.nan)

    match_mode, interpolation_k, interpolation_power = _resolve_lut_match_settings(step5_config, step3_config)
    merged = apply_lut_fallback_matches(
        merged,
        lut_dataframe,
        query_columns=query_columns,
        raw_columns=raw_query_columns,
        match_mode=match_mode,
        interpolation_k=interpolation_k,
        interpolation_power=interpolation_power,
        group_columns=CANONICAL_Z_COLUMNS if lut_has_z else None,
    )

    merged["corrected_rate_to_perfect_hz"] = merged["rate_hz"] * merged["lut_scale_factor"]
    try:
        rate_to_flux_lines = load_rate_to_flux_lines(rate_to_flux_lines_path)
    except FileNotFoundError:
        reference_table = load_reference_curve_table(flux_cells_path)
        rate_to_flux_lines = build_rate_to_flux_lines(reference_table)
    (
        merged["corrected_flux_cm2_min"],
        merged["corrected_flux_assignment_method"],
    ) = apply_rate_to_flux_lines(
        merged["corrected_rate_to_perfect_hz"],
        row_z_frame=(merged[CANONICAL_Z_COLUMNS] if all(column in rate_to_flux_lines.columns for column in CANONICAL_Z_COLUMNS) else None),
        line_table=rate_to_flux_lines,
    )

    unique_real_z = unique_z_vectors(merged, z_columns=CANONICAL_Z_COLUMNS)
    available_lut_z = unique_z_vectors(lut_dataframe, z_columns=CANONICAL_Z_COLUMNS) if lut_has_z else []
    z_source = "normalized_real_slice"
    merged["selected_z_vector_match"] = True
    if lut_has_z:
        merged["selected_z_vector_match"] = False
        for z_vector in available_lut_z:
            merged.loc[z_mask_for_vector(merged, z_vector, z_columns=CANONICAL_Z_COLUMNS), "selected_z_vector_match"] = True

    missing_real_z = [
        z_vector
        for z_vector in unique_real_z
        if not any(np.allclose(np.asarray(z_vector, dtype=float), np.asarray(candidate, dtype=float)) for candidate in available_lut_z)
    ] if lut_has_z else []
    z_warning_message = None
    if lut_has_z and missing_real_z:
        z_warning_message = (
            "Some real-data z configurations do not have a matching LUT in the requested window."
        )
        log.warning("%s real=%s | lut=%s", z_warning_message, unique_real_z, available_lut_z)

    output_dataframe = real_dataframe.copy()
    output_dataframe["z_config_id"] = merged["z_config_id"]
    output_dataframe["z_config_label"] = merged["z_config_label"]
    for column in query_columns:
        output_dataframe[column] = merged[column]
    for column in CANONICAL_EFF_COLUMNS:
        output_dataframe[f"lut_{column}"] = merged[f"lut_{column}"]
    output_dataframe["lut_scale_factor"] = merged["lut_scale_factor"]
    output_dataframe["lut_match_method"] = merged["lut_match_method"]
    output_dataframe["lut_match_distance"] = merged["lut_match_distance"]
    output_dataframe["lut_neighbor_count"] = merged["lut_neighbor_count"]
    output_dataframe["lut_neighbor_min_distance"] = merged["lut_neighbor_min_distance"]
    output_dataframe["lut_neighbor_max_distance"] = merged["lut_neighbor_max_distance"]
    output_dataframe["corrected_rate_to_perfect_hz"] = merged["corrected_rate_to_perfect_hz"]
    output_dataframe["corrected_flux_cm2_min"] = merged["corrected_flux_cm2_min"]
    output_dataframe["corrected_flux_assignment_method"] = merged["corrected_flux_assignment_method"]
    output_dataframe["selected_z_vector_match"] = merged["selected_z_vector_match"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, index=False)

    if lut_diag_path.exists():
        lut_diagnostics = pd.read_csv(lut_diag_path)
    else:
        lut_diagnostics = lut_dataframe.copy()
    coverage_path = output_path.with_name("step5_lut_query_coverage.csv")
    query_coverage = _build_query_coverage_diagnostics(merged, lut_diagnostics, query_columns)
    query_coverage.to_csv(coverage_path, index=False)
    event_markers = _load_event_markers_for_station(station_id)

    time_axis_column_used = _plot_real_rate_correction(
        merged,
        PLOTS_DIR / "step5_real_rate_correction.png",
        rate_column_name=rate_column_name,
        event_markers=event_markers,
        efficiency_plot_ylim=efficiency_plot_ylim,
        apply_moving_average=apply_plot_moving_average,
        moving_average_kernel=plot_moving_average_kernel,
    )
    _plot_real_correction_diagnostics(
        merged,
        PLOTS_DIR / "step5_real_correction_diagnostics.png",
        rate_column_name=rate_column_name,
        event_markers=event_markers,
        apply_moving_average=apply_plot_moving_average,
        moving_average_kernel=plot_moving_average_kernel,
    )
    _plot_lut_vs_real_efficiency_coverage(
        merged,
        lut_diagnostics,
        PLOTS_DIR / "step5_lut_real_efficiency_coverage.png",
        rate_column_name=rate_column_name,
    )
    _plot_real_rate_vs_efficiencies_2x2(
        merged,
        PLOTS_DIR / "step5_real_rate_vs_efficiencies_2x2.png",
        rate_column_name=rate_column_name,
    )
    _plot_corrected_flux_from_rate(
        merged,
        rate_to_flux_lines,
        PLOTS_DIR / "step5_corrected_flux_from_rate.png",
        rate_column_name=rate_column_name,
        event_markers=event_markers,
        apply_moving_average=apply_plot_moving_average,
        moving_average_kernel=plot_moving_average_kernel,
    )

    metadata = {
        "source_station_metadata_root": str(_task_metadata_dir(station_id, task_ids[0]).parent.parent.parent),
        "station_id": int(station_id),
        "station_name": station_name,
        "task_ids": task_ids,
        "date_from": str(date_from) if date_from is not None else None,
        "date_to": str(date_to) if date_to is not None else None,
        "min_events": min_events,
        "metadata_agg": metadata_agg,
        "timestamp_column": timestamp_column,
        "rate_input_column": rate_column_name,
        "trigger_rate_selection": trigger_selection,
        "trigger_type_selection": trigger_selection,
        "lut_file": str(lut_path),
        "rate_to_flux_lines_file": str(rate_to_flux_lines_path),
        "lut_comments": lut_comments,
        "efficiency_bin_width": efficiency_bin_width,
        "plot_apply_moving_average": apply_plot_moving_average,
        "plot_moving_average_kernel": plot_moving_average_kernel,
        "selected_feature_columns_mode": feature_vector_config["mode"],
        "same_efficiency_planes": feature_vector_config.get("same_efficiency_planes"),
        "selected_feature_source_columns": feature_vector_config["source_columns"],
        "lut_match_mode_requested": match_mode,
        "lut_interpolation_k": interpolation_k,
        "lut_interpolation_power": interpolation_power,
        "row_count": int(len(output_dataframe)),
        "exact_matches": int((output_dataframe["lut_match_method"] == "exact").sum()),
        "nearest_matches": int((output_dataframe["lut_match_method"] == "nearest").sum()),
        "interpolated_matches": int((output_dataframe["lut_match_method"] == "interpolated").sum()),
        "unmatched_rows": int(output_dataframe["lut_scale_factor"].isna().sum()),
        "lut_selected_z_positions": [
            [float(value) for value in z_vector]
            for z_vector in available_lut_z
        ] if lut_has_z else None,
        "lut_z_configuration_count": int(len(available_lut_z)) if lut_has_z else 1,
        "real_z_positions_source": z_source,
        "real_z_positions_in_window": unique_real_z,
        "rows_matching_selected_z_vector": int(pd.to_numeric(output_dataframe["selected_z_vector_match"], errors="coerce").fillna(0).sum()),
        "corrected_flux_assignment_method_counts": {
            str(key): int(value)
            for key, value in output_dataframe["corrected_flux_assignment_method"].value_counts(dropna=False).to_dict().items()
        },
        "rate_to_flux_lines": rate_to_flux_lines.to_dict(orient="records"),
        "z_warning_message": z_warning_message,
        "time_axis_column_used_for_plots": time_axis_column_used,
        "query_coverage_csv": str(coverage_path),
        "query_bins_total": int(len(query_coverage)),
        "query_bins_with_exact_lut_support": int(query_coverage["exact_lut_support"].sum()),
        "top_missing_query_bins": query_coverage.loc[query_coverage["exact_lut_support"] == 0]
        .head(5)
        .apply(
            lambda row: {
                "z_positions": [float(row[column]) for column in CANONICAL_Z_COLUMNS if column in row.index],
                "query_bin": [float(row[column]) for column in query_columns],
                "real_row_count": int(row["real_row_count"]),
                "nearest_distance": float(row["nearest_distance"]),
                "nearest_scale_factor": float(row["nearest_scale_factor"]),
            },
            axis=1,
        )
        .tolist(),
        "collection": collection_meta,
    }
    write_json(meta_path, metadata)

    log.info("Wrote Step 5 real-data LUT application to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the scale-factor LUT to a real-data station/date slice."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON file.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
