"""Config bootstrap for STEP_1 calibration QA task folders."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


def _normalize_station_name(station: Any) -> str:
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


def _load_yaml_mapping(config_path: Path, *, required: bool) -> dict[str, Any]:
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


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _validate_date_range(config: dict[str, Any]) -> None:
    """Validate optional date-range configuration."""
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
            _normalize_station_name(station)

    priority = config.get("time_columns_priority")
    if priority is None:
        return
    if not isinstance(priority, list):
        raise ValueError("'time_columns_priority' must be a list when provided.")
    for entry in priority:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("'time_columns_priority' entries must be non-empty strings.")


def _validate_column_patterns(config: dict[str, Any]) -> None:
    """Validate plotting column pattern configuration."""
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
    return _load_yaml_mapping(config_path, required=True)


def load_task_configs(task_dir: Path) -> dict[str, Any]:
    """Load merged config from QA root + step + task layers."""
    step_dir = task_dir.parent
    qa_root = step_dir.parent
    qa_config = _load_yaml_mapping(qa_root / "config.yaml", required=False)
    qa_runtime_config = _load_yaml_mapping(qa_root / "config_runtime.yaml", required=False)
    common_config = _load_yaml_mapping(Path(__file__).resolve().parent / "config.yaml", required=False)
    step_config = _load_yaml_mapping(step_dir / "config.yaml", required=False)
    task_config = _load_yaml_mapping(task_dir / "config.yaml", required=True)

    merged: dict[str, Any] = {}
    for layer in (qa_config, qa_runtime_config, common_config, step_config, task_config):
        merged = _deep_merge_dicts(merged, layer)

    merged.setdefault("date_range_excluded_stations", ["MINGO00"])
    validate_task_config(merged)
    return merged


def _is_empty_date_boundary(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def get_station_date_range(config: dict[str, Any], station: Any) -> dict[str, Any] | None:
    """
    Return effective date range for a station.

    Returns None when date filtering is disabled for that station.
    """
    date_range = config.get("date_range")
    if not isinstance(date_range, dict):
        return None

    start = date_range.get("start")
    end = date_range.get("end")
    if _is_empty_date_boundary(start) and _is_empty_date_boundary(end):
        return None

    excluded_stations = config.get("date_range_excluded_stations", ["MINGO00"])
    if not isinstance(excluded_stations, list):
        raise ValueError("'date_range_excluded_stations' must be a list.")

    excluded_normalized = {_normalize_station_name(item) for item in excluded_stations}
    station_name = _normalize_station_name(station)
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
        station_name = _normalize_station_name(station)
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
        station_dir = base_dir / _normalize_station_name(station)
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
