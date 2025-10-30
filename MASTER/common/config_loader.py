from __future__ import annotations

import csv
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict


def _parse_value(raw_value: str) -> Any:
    """Return *raw_value* converted to Python types when possible."""
    value = raw_value.strip()
    if value == "":
        return value
    lower_value = value.lower()
    if lower_value in {"true", "false"}:
        return lower_value == "true"
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_parameter_overrides(
    csv_path: str | Path,
    station: str,
) -> Dict[str, Any]:
    """Load parameter overrides for *station* from the CSV file."""
    parameter_csv_path = Path(csv_path)
    if not parameter_csv_path.exists():
        raise FileNotFoundError(f"Configuration parameters file not found: {csv_path}")

    overrides: Dict[str, Any] = {}
    station_column = f"station_{station}"

    with parameter_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Configuration parameters file has no header: {csv_path}")
        headers = set(reader.fieldnames)
        use_default_only = station_column not in headers

        for row in reader:
            parameter_name = (row.get("parameter") or "").strip()
            if not parameter_name:
                continue

            value_str = ""
            if not use_default_only:
                value_str = (row.get(station_column) or "").strip()
            if value_str == "":
                value_str = (row.get("default") or "").strip()
            if value_str == "":
                # Skip empty overrides entirely.
                continue

            overrides[parameter_name] = _parse_value(value_str)

    return overrides


def update_config_with_parameters(
    config: Dict[str, Any],
    csv_path: str | Path,
    station: str,
) -> Dict[str, Any]:
    """Merge station-specific parameter overrides into *config*."""
    overrides = load_parameter_overrides(csv_path, station)
    config.update(overrides)
    return config

