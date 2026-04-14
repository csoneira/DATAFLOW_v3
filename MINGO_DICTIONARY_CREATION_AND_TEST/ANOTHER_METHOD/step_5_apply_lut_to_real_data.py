#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CANONICAL_EFF_COLUMNS,
    DEFAULT_CONFIG_PATH,
    PLOTS_DIR,
    apply_lut_fallback_matches,
    cfg_path,
    derive_trigger_rate_features,
    ensure_output_dirs,
    format_selected_rate_name,
    get_rate_column_name,
    get_trigger_type_selection,
    load_config,
    quantize_efficiency_series,
    read_ascii_lut,
    write_json,
)

log = logging.getLogger("another_method.step5")

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILE_TS_RE = re.compile(r"(\d{11})$")
_OFFENDER_RATE_RE = re.compile(
    r"^(?P<scope>plane_combination_filter|strip_combination_filter)_rows_with_(?P<count>\d+)_selected_offenders_rate_hz$"
)

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parents[1]
STATIONS_ROOT = REPO_ROOT / "STATIONS"
ONLINE_RUN_DICTIONARY_ROOT = (
    REPO_ROOT
    / "MASTER"
    / "CONFIG_FILES"
    / "STAGE_0"
    / "NEW_FILES"
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
        allow_nearest = bool(
            step5_config.get(
                "allow_nearest_lut_match",
                step3_config.get("allow_nearest_lut_match", True),
            )
        )
        match_mode = "nearest" if allow_nearest else "exact"
    else:
        normalized = str(raw_mode).strip().lower()
        match_mode = {
            "idw": "interpolate",
            "interpolated": "interpolate",
            "interpolation": "interpolate",
        }.get(normalized, normalized)

    interpolation_k_raw = step5_config.get("lut_interpolation_k", step3_config.get("lut_interpolation_k", 8))
    interpolation_k = None if interpolation_k_raw in (None, "", "null", "None") else int(interpolation_k_raw)

    interpolation_power_raw = step5_config.get(
        "lut_interpolation_power",
        step3_config.get("lut_interpolation_power", 2.0),
    )
    interpolation_power = (
        2.0 if interpolation_power_raw in (None, "", "null", "None") else float(interpolation_power_raw)
    )
    return match_mode, interpolation_k, interpolation_power


def _build_query_coverage_diagnostics(
    dataframe: pd.DataFrame,
    lut_diagnostics: pd.DataFrame,
    query_columns: list[str],
) -> pd.DataFrame:
    summary = (
        dataframe.groupby(query_columns, dropna=False)
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
            ["real_row_count", *query_columns],
            ascending=[False, True, True, True, True],
        )
        .reset_index(drop=True)
    )

    exact_lookup = lut_diagnostics.rename(
        columns={column: f"query_{column}" for column in CANONICAL_EFF_COLUMNS}
    )
    exact_columns = query_columns + ["scale_factor"]
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
        on=query_columns,
    )
    summary["exact_lut_support"] = summary["exact_scale_factor"].notna().astype(int)

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


def _plot_real_rate_correction(
    dataframe: pd.DataFrame,
    output_path: Path,
    *,
    rate_column_name: str,
) -> str:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        height_ratios=[2.0, 1.2],
    )

    axes[0].plot(x_values, ordered["rate_hz"], marker="o", linewidth=1.6, label="Observed rate")
    axes[0].plot(
        x_values,
        ordered["corrected_rate_to_perfect_hz"],
        marker="o",
        linewidth=1.6,
        label="LUT-corrected rate",
    )
    axes[0].set_ylabel("Rate [Hz]")
    axes[0].set_title(
        "Observed and corrected real-data rate\n"
        f"rate column: {rate_column_name}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    plane_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(CANONICAL_EFF_COLUMNS):
        axes[1].plot(
            x_values,
            ordered[column],
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=plane_colors[idx],
            label=f"Plane {idx + 1} eff",
        )
    axes[1].set_xlabel(x_label.replace("_", " "))
    axes[1].set_ylabel("Empirical efficiency")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)

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
) -> str:
    ordered, x_values, x_label = _resolve_time_axis(dataframe)
    sequence = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    scatter = axes[0].scatter(
        ordered["rate_hz"],
        ordered["corrected_rate_to_perfect_hz"],
        c=sequence,
        cmap="viridis",
        s=42,
        alpha=0.85,
    )
    finite_rates = pd.concat(
        [ordered["rate_hz"], ordered["corrected_rate_to_perfect_hz"]],
        ignore_index=True,
    ).to_numpy(dtype=float)
    finite_rates = finite_rates[np.isfinite(finite_rates)]
    if finite_rates.size:
        low = float(np.min(finite_rates))
        high = float(np.max(finite_rates))
        axes[0].plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="black", alpha=0.7)
    axes[0].set_title(f"Observed vs corrected rate\nrate column: {rate_column_name}")
    axes[0].set_xlabel(f"Observed rate [Hz]\n({rate_column_name})")
    axes[0].set_ylabel(f"Corrected rate [Hz]\n(from {rate_column_name})")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x_values,
        ordered["lut_scale_factor"],
        marker="o",
        linewidth=1.6,
        color="#8B1E3F",
        label="LUT scale factor",
    )
    axes[1].set_title("Scale factor vs time")
    axes[1].set_xlabel(x_label.replace("_", " "))
    axes[1].set_ylabel("Scale factor")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.92)
    cbar.set_label("Time-series order")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return x_label


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
    task_id = int(trigger_selection["task_id"])
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
        source_name="trigger_type",
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
        selected_rate_name = format_selected_rate_name(
            stage_prefix=str(trigger_info["stage_prefix"]),
            rate_family_column=str(trigger_info["rate_family_column"]),
            offender_threshold=trigger_info.get("used_offender_threshold"),
        )
        raise ValueError(
            "No real rows remain after deriving trigger-type features for "
            f"{selected_rate_name}. The selected task/offender/rate combination currently yields "
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
    event_source = "selected_rate_count"

    if event_values.notna().any():
        collected["n_events"] = event_values
    if min_events is not None and event_values.notna().any():
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
        "trigger_type_selection": trigger_info,
        "offender_count_semantics": "cumulative_total_problematic_offender_count_from_trigger_type",
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

    trigger_selection = get_trigger_type_selection(config)
    task_ids = [int(trigger_selection["task_id"])]
    rate_column_name = format_selected_rate_name(
        stage_prefix=str(trigger_selection["stage_prefix"]),
        rate_family_column=str(trigger_selection["rate_family_column"]),
        offender_threshold=trigger_selection["offender_threshold"],
    )

    lut_path = cfg_path(config, "paths", "step2_lut_ascii")
    lut_meta_path = cfg_path(config, "paths", "step2_meta_json")
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

    lut_dataframe, lut_comments = read_ascii_lut(lut_path)
    lut_meta = {}
    if lut_meta_path.exists():
        lut_meta = json.loads(lut_meta_path.read_text(encoding="utf-8"))
    efficiency_bin_width = float(
        lut_meta.get("efficiency_bin_width", config.get("step2", {}).get("efficiency_bin_width", 0.02))
    )

    query_columns: list[str] = []
    for column in CANONICAL_EFF_COLUMNS:
        query_column = f"query_{column}"
        work[query_column] = quantize_efficiency_series(work[column], efficiency_bin_width)
        query_columns.append(query_column)

    lut_lookup = lut_dataframe.rename(columns={column: f"lut_{column}" for column in CANONICAL_EFF_COLUMNS})
    merged = work.merge(
        lut_lookup,
        how="left",
        left_on=query_columns,
        right_on=[f"lut_{column}" for column in CANONICAL_EFF_COLUMNS],
    )
    merged = merged.rename(columns={"scale_factor": "lut_scale_factor"})
    merged["lut_match_method"] = np.where(merged["lut_scale_factor"].notna(), "exact", pd.NA)
    merged["lut_match_distance"] = np.where(merged["lut_scale_factor"].notna(), 0.0, np.nan)

    match_mode, interpolation_k, interpolation_power = _resolve_lut_match_settings(step5_config, step3_config)
    merged = apply_lut_fallback_matches(
        merged,
        lut_dataframe,
        query_columns=query_columns,
        raw_columns=CANONICAL_EFF_COLUMNS,
        match_mode=match_mode,
        interpolation_k=interpolation_k,
        interpolation_power=interpolation_power,
    )

    merged["corrected_rate_to_perfect_hz"] = merged["rate_hz"] * merged["lut_scale_factor"]

    lut_z_vector = _extract_lut_z_vector(lut_meta, lut_comments)
    merged, unique_real_z, z_source = _prepare_z_quality_columns(merged, lut_z_vector)

    z_warning_message = None
    if lut_z_vector is None:
        z_warning_message = "Could not determine a z-position vector from the LUT metadata."
        log.warning("%s", z_warning_message)
    elif not unique_real_z:
        z_warning_message = "Real-data z positions could not be resolved for the requested window."
        log.warning("%s", z_warning_message)
    else:
        matching_tuples = [
            z_vector
            for z_vector in unique_real_z
            if np.allclose(np.asarray(z_vector, dtype=float), np.asarray(lut_z_vector, dtype=float))
        ]
        if not matching_tuples:
            z_warning_message = (
                "LUT z-position vector does not match the real-data z positions in the requested window."
            )
            log.warning(
                "%s LUT=%s | real=%s",
                z_warning_message,
                [float(value) for value in lut_z_vector],
                unique_real_z,
            )

    output_dataframe = real_dataframe.copy()
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

    time_axis_column_used = _plot_real_rate_correction(
        merged,
        PLOTS_DIR / "step5_real_rate_correction.png",
        rate_column_name=rate_column_name,
    )
    _plot_real_correction_diagnostics(
        merged,
        PLOTS_DIR / "step5_real_correction_diagnostics.png",
        rate_column_name=rate_column_name,
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
        "trigger_type_selection": trigger_selection,
        "lut_file": str(lut_path),
        "lut_comments": lut_comments,
        "efficiency_bin_width": efficiency_bin_width,
        "lut_match_mode_requested": match_mode,
        "lut_interpolation_k": interpolation_k,
        "lut_interpolation_power": interpolation_power,
        "row_count": int(len(output_dataframe)),
        "exact_matches": int((output_dataframe["lut_match_method"] == "exact").sum()),
        "nearest_matches": int((output_dataframe["lut_match_method"] == "nearest").sum()),
        "interpolated_matches": int((output_dataframe["lut_match_method"] == "interpolated").sum()),
        "unmatched_rows": int(output_dataframe["lut_scale_factor"].isna().sum()),
        "lut_selected_z_positions": (
            [float(value) for value in lut_z_vector] if lut_z_vector is not None else None
        ),
        "real_z_positions_source": z_source,
        "real_z_positions_in_window": unique_real_z,
        "rows_matching_selected_z_vector": int(pd.to_numeric(output_dataframe["selected_z_vector_match"], errors="coerce").fillna(0).sum()),
        "z_warning_message": z_warning_message,
        "time_axis_column_used_for_plots": time_axis_column_used,
        "query_coverage_csv": str(coverage_path),
        "query_bins_total": int(len(query_coverage)),
        "query_bins_with_exact_lut_support": int(query_coverage["exact_lut_support"].sum()),
        "top_missing_query_bins": query_coverage.loc[query_coverage["exact_lut_support"] == 0]
        .head(5)
        .apply(
            lambda row: {
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
