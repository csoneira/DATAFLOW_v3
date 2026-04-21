"""Shared config/bootstrap/time-axis helpers for QUALITY_ASSURANCE tasks."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from MASTER.common.file_selection import extract_run_datetime_from_name

DEFAULT_DATE_RANGE_EXCLUDED_STATIONS = ["MINGO00"]
DEFAULT_TIME_COLUMNS_PRIORITY = ["execution_timestamp", "datetime"]

__all__ = [
    "DEFAULT_DATE_RANGE_EXCLUDED_STATIONS",
    "DEFAULT_TIME_COLUMNS_PRIORITY",
    "apply_date_filter",
    "bootstrap_task",
    "deep_merge_dicts",
    "ensure_task_station_tree",
    "get_date_range_by_station",
    "get_station_date_range",
    "load_step_configs",
    "load_task_config",
    "load_task_configs",
    "load_yaml_mapping",
    "normalize_station_name",
    "parse_boundary",
    "parse_filename_timestamp_series",
    "parse_timestamp_series",
    "pick_time_column",
    "resolve_filter_timestamp",
    "resolve_plot_x_axis",
    "validate_task_config",
    "x_axis_config",
]


def normalize_station_name(station: Any) -> str:
    """Return normalized station folder names like MINGO00."""
    if isinstance(station, int):
        return f"MINGO{station:02d}"

    text = str(station).strip()
    if not text:
        raise ValueError("Empty station value in config.")

    if text.isdigit():
        return f"MINGO{int(text):02d}"

    upper = text.upper()
    if upper.startswith("MINGO"):
        suffix = upper.removeprefix("MINGO")
        if suffix.isdigit():
            return f"MINGO{int(suffix):02d}"
        raise ValueError(f"Invalid station value '{station}'. Expected digits after MINGO.")

    raise ValueError(f"Invalid station value '{station}'.")


def load_yaml_mapping(config_path: Path, *, required: bool = False) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    if not config_path.exists():
        if required:
            raise FileNotFoundError(f"Config not found: {config_path}")
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return data


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _validate_date_range(config: dict[str, Any]) -> None:
    date_range = config.get("date_range")
    if date_range is None:
        return
    if not isinstance(date_range, dict):
        raise ValueError("'date_range' must be a mapping with optional 'start' and 'end'.")

    for key in ("start", "end"):
        value = date_range.get(key)
        if value is None:
            continue
        if isinstance(value, (date, datetime)):
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"'date_range.{key}' must be a non-empty string or null.")

    excluded_stations = config.get("date_range_excluded_stations")
    if excluded_stations is not None:
        if not isinstance(excluded_stations, list):
            raise ValueError("'date_range_excluded_stations' must be a list when provided.")
        for station in excluded_stations:
            normalize_station_name(station)

    priority = config.get("time_columns_priority")
    if priority is None:
        return
    if not isinstance(priority, list):
        raise ValueError("'time_columns_priority' must be a list when provided.")
    for entry in priority:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("'time_columns_priority' entries must be non-empty strings.")


def _validate_column_patterns(config: dict[str, Any]) -> None:
    patterns = config.get("column_patterns")
    if patterns is None:
        return
    if not isinstance(patterns, list):
        raise ValueError("'column_patterns' must be a list when provided.")
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("All 'column_patterns' entries must be non-empty strings.")


def validate_task_config(config: dict[str, Any]) -> None:
    """Validate task-level configuration schema."""
    _validate_date_range(config)
    _validate_column_patterns(config)


def load_task_config(config_path: Path) -> dict[str, Any]:
    """Backward-compatible single-file config loader."""
    return load_yaml_mapping(config_path, required=True)


def load_task_configs(task_dir: Path) -> dict[str, Any]:
    """Load merged config from QA root + runtime + step/task layers, including optional quality overrides."""
    step_dir = task_dir.parent
    qa_root = step_dir.parent
    qa_config = load_yaml_mapping(qa_root / "config.yaml")
    qa_runtime_config = load_yaml_mapping(qa_root / "config_runtime.yaml")
    qa_quality_config = load_yaml_mapping(qa_root / "config_quality.yaml")
    step_common_config = load_yaml_mapping(step_dir / "common" / "config.yaml")
    step_common_quality_config = load_yaml_mapping(step_dir / "common" / "config_quality.yaml")
    step_config = load_yaml_mapping(step_dir / "config.yaml")
    step_quality_config = load_yaml_mapping(step_dir / "config_quality.yaml")
    task_config = load_yaml_mapping(task_dir / "config.yaml", required=True)
    task_quality_config = load_yaml_mapping(task_dir / "config_quality.yaml")

    merged: dict[str, Any] = {}
    for layer in (
        qa_config,
        qa_runtime_config,
        qa_quality_config,
        step_common_config,
        step_common_quality_config,
        step_config,
        step_quality_config,
        task_config,
        task_quality_config,
    ):
        merged = deep_merge_dicts(merged, layer)

    merged.setdefault("date_range_excluded_stations", list(DEFAULT_DATE_RANGE_EXCLUDED_STATIONS))
    validate_task_config(merged)
    return merged


def load_step_configs(step_dir: Path) -> dict[str, Any]:
    """Load merged config from QA root + runtime + step layers for step-level runners."""
    qa_root = step_dir.parent
    qa_config = load_yaml_mapping(qa_root / "config.yaml")
    qa_runtime_config = load_yaml_mapping(qa_root / "config_runtime.yaml")
    qa_quality_config = load_yaml_mapping(qa_root / "config_quality.yaml")
    step_common_config = load_yaml_mapping(step_dir / "common" / "config.yaml")
    step_common_quality_config = load_yaml_mapping(step_dir / "common" / "config_quality.yaml")
    step_config = load_yaml_mapping(step_dir / "config.yaml", required=True)
    step_quality_config = load_yaml_mapping(step_dir / "config_quality.yaml")

    merged: dict[str, Any] = {}
    for layer in (
        qa_config,
        qa_runtime_config,
        qa_quality_config,
        step_common_config,
        step_common_quality_config,
        step_config,
        step_quality_config,
    ):
        merged = deep_merge_dicts(merged, layer)

    merged.setdefault("date_range_excluded_stations", list(DEFAULT_DATE_RANGE_EXCLUDED_STATIONS))
    validate_task_config(merged)
    return merged


def _is_empty_date_boundary(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def get_station_date_range(config: dict[str, Any], station: Any) -> dict[str, Any] | None:
    """Return effective date range for a station, or None when disabled."""
    date_range = config.get("date_range")
    if not isinstance(date_range, dict):
        return None

    start = date_range.get("start")
    end = date_range.get("end")
    if _is_empty_date_boundary(start) and _is_empty_date_boundary(end):
        return None

    excluded_stations = config.get("date_range_excluded_stations", DEFAULT_DATE_RANGE_EXCLUDED_STATIONS)
    if not isinstance(excluded_stations, list):
        raise ValueError("'date_range_excluded_stations' must be a list.")

    excluded_normalized = {normalize_station_name(item) for item in excluded_stations}
    station_name = normalize_station_name(station)
    if station_name in excluded_normalized:
        return None

    return {"start": start, "end": end}


def get_date_range_by_station(config: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """Return effective date-range map keyed by normalized station name."""
    stations = config.get("stations", [])
    if not isinstance(stations, list):
        raise ValueError("'stations' must be a list.")

    result: dict[str, dict[str, Any] | None] = {}
    for station in stations:
        station_name = normalize_station_name(station)
        result[station_name] = get_station_date_range(config=config, station=station)
    return result


def ensure_task_station_tree(task_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Create STATIONS/MINGOXX folders and optional configured subdirectories."""
    stations = config.get("stations", [])
    if not isinstance(stations, list):
        raise ValueError("'stations' must be a list.")

    stations_root = config.get("stations_root", "STATIONS")
    if not isinstance(stations_root, str) or not stations_root.strip():
        raise ValueError("'stations_root' must be a non-empty string.")

    station_subdirs = config.get("station_subdirectories", [])
    if station_subdirs is None:
        station_subdirs = []
    if not isinstance(station_subdirs, list):
        raise ValueError("'station_subdirectories' must be a list.")

    created: list[Path] = []
    base_dir = task_dir / stations_root
    base_dir.mkdir(parents=True, exist_ok=True)

    for station in stations:
        station_dir = base_dir / normalize_station_name(station)
        station_dir.mkdir(parents=True, exist_ok=True)
        created.append(station_dir)

        for subdir in station_subdirs:
            if not isinstance(subdir, str):
                raise ValueError("All entries in 'station_subdirectories' must be strings.")
            rel = Path(subdir.strip())
            if not str(rel):
                continue
            if rel.is_absolute():
                raise ValueError("Subdirectories in 'station_subdirectories' must be relative paths.")
            target = station_dir / rel
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)

    return created


def bootstrap_task(task_dir: Path) -> list[Path]:
    """Load task config stack and ensure configured directory tree exists."""
    config = load_task_configs(task_dir=task_dir)
    return ensure_task_station_tree(task_dir=task_dir, config=config)


def parse_boundary(value: Any, *, end_of_day_if_date_only: bool) -> pd.Timestamp | None:
    """Parse a date/datetime boundary into a pandas timestamp."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        boundary = pd.to_datetime(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        boundary = pd.to_datetime(text, errors="coerce")

    if pd.isna(boundary):
        raise ValueError(f"Invalid date boundary '{value}'.")

    if end_of_day_if_date_only and isinstance(value, str) and len(value.strip()) <= 10:
        boundary = boundary + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return boundary


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse a timestamp series using the formats used in Stage 1 metadata."""
    as_text = series.astype("string").fillna("").str.strip()
    parsed = pd.to_datetime(as_text, format="%Y-%m-%d_%H.%M.%S", errors="coerce")
    if parsed.notna().all():
        return parsed

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        remaining = parsed.isna()
        if not remaining.any():
            break
        parsed.loc[remaining] = pd.to_datetime(as_text.loc[remaining], format=fmt, errors="coerce")
    return parsed


def parse_filename_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse run datetimes from filename_base-like values."""
    as_text = series.astype("string").fillna("").str.strip()
    parsed = as_text.map(lambda value: extract_run_datetime_from_name(value) if value else None)
    return pd.to_datetime(parsed, errors="coerce")


def x_axis_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("x_axis")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "mode": str(raw.get("mode", "filename_timestamp")).strip().lower(),
        "filename_column": str(raw.get("filename_column", "filename_base")).strip() or "filename_base",
        "column": raw.get("column"),
    }


def pick_time_column(df: pd.DataFrame, config: dict[str, Any]) -> str:
    priority = config.get("time_columns_priority") or list(DEFAULT_TIME_COLUMNS_PRIORITY)
    for candidate in priority:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "No preferred timestamp column found. "
        f"Looked for: {priority}. Available columns: {list(df.columns)}"
    )


def resolve_filter_timestamp(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, str]:
    axis_cfg = x_axis_config(config)
    filename_column = axis_cfg["filename_column"]
    if filename_column in df.columns:
        from_filename = parse_filename_timestamp_series(df[filename_column])
        if from_filename.notna().any():
            return from_filename, f"{filename_column} (miXXYYDDDHHMMSS)"

    time_col = pick_time_column(df, config)
    parsed_time = parse_timestamp_series(df[time_col])
    if parsed_time.notna().any():
        return parsed_time, time_col

    raise KeyError(
        "Could not resolve timestamp values for filtering. "
        f"Tried filename column '{filename_column}' and priority columns "
        f"{config.get('time_columns_priority') or list(DEFAULT_TIME_COLUMNS_PRIORITY)}."
    )


def resolve_plot_x_axis(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, str, bool]:
    axis_cfg = x_axis_config(config)
    mode = axis_cfg["mode"]

    if mode == "filename_timestamp":
        filename_column = axis_cfg["filename_column"]
        if filename_column in df.columns:
            parsed = parse_filename_timestamp_series(df[filename_column])
            if parsed.notna().any():
                return parsed, filename_column, True
        return df["__timestamp__"], "__timestamp__", True

    if mode == "column":
        column = axis_cfg.get("column")
        if column is None:
            return df["__timestamp__"], "__timestamp__", True
        column_name = str(column).strip()
        if not column_name:
            return df["__timestamp__"], "__timestamp__", True
        if column_name not in df.columns:
            raise KeyError(f"x_axis.column='{column_name}' not present in metadata.")

        numeric = pd.to_numeric(df[column_name], errors="coerce")
        if numeric.notna().any():
            return numeric, column_name, False

        parsed = parse_timestamp_series(df[column_name])
        if parsed.notna().any():
            return parsed, column_name, True

        raise ValueError(f"x_axis.column='{column_name}' could not be parsed as numeric or datetime.")

    raise ValueError("Invalid x_axis.mode. Use one of: filename_timestamp, column.")


def apply_date_filter(df: pd.DataFrame, date_range: dict[str, Any] | None) -> pd.DataFrame:
    """Apply optional date filtering using the resolved __timestamp__ column."""
    if date_range is None:
        return df

    start = parse_boundary(date_range.get("start"), end_of_day_if_date_only=False)
    end = parse_boundary(date_range.get("end"), end_of_day_if_date_only=True)

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["__timestamp__"] >= start
    if end is not None:
        mask &= df["__timestamp__"] <= end
    return df.loc[mask].copy()
