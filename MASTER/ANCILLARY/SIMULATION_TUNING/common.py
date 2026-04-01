from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


DEFAULT_SIMULATION_STATIONS = ["MINGO00"]
DEFAULT_REAL_STATIONS = ["MINGO01", "MINGO02", "MINGO03", "MINGO04"]


@dataclass(frozen=True)
class TuningSelection:
    simulation_stations: list[str]
    real_stations: list[str]
    simulation_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None
    real_date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def tuning_root() -> Path:
    return Path(__file__).resolve().parent


def default_config_path() -> Path:
    return tuning_root() / "config_simulation_tuning.yaml"


def _normalize_station_label(raw: object) -> str:
    if isinstance(raw, int):
        return f"MINGO{int(raw):02d}"
    text = str(raw).strip().upper()
    if not text:
        raise ValueError("Empty station token in tuning config.")
    if text.isdigit():
        return f"MINGO{int(text):02d}"
    if text.startswith("MINGO"):
        suffix = text[5:]
        if suffix.isdigit():
            return f"MINGO{int(suffix):02d}"
    raise ValueError(f"Unsupported station token in tuning config: {raw!r}")


def _normalize_station_list(raw: object, *, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Station selection must be null or a list.")
    normalized = [_normalize_station_label(item) for item in raw]
    seen: set[str] = set()
    ordered: list[str] = []
    for label in normalized:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _parse_date_value(raw: object, *, is_end: bool) -> pd.Timestamp:
    text = str(raw).strip()
    ts = pd.Timestamp(text)
    if text and len(text) <= 10 and "T" not in text and " " not in text and is_end:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _normalize_date_ranges(raw: object) -> list[tuple[pd.Timestamp, pd.Timestamp]] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Date ranges must be null or a list of [start, end] pairs.")
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each date range must contain exactly two values: [start, end].")
        start = _parse_date_value(item[0], is_end=False)
        end = _parse_date_value(item[1], is_end=True)
        if end < start:
            raise ValueError(f"Invalid date range with end before start: {item!r}")
        ranges.append((start, end))
    return ranges or None


def load_tuning_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else default_config_path()
    if not path.is_absolute():
        path = tuning_root() / path
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(path)
    return data


def station_task3_input_dirs(station_label: str) -> list[Path]:
    base = (
        repo_root()
        / "STATIONS"
        / station_label
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / "TASK_3"
        / "INPUT_FILES"
    )
    return [
        base / "COMPLETED_DIRECTORY",
        base / "UNPROCESSED_DIRECTORY",
        base / "PROCESSING_DIRECTORY",
    ]


def station_has_calibrated_data(station_label: str) -> bool:
    for directory in station_task3_input_dirs(station_label):
        if directory.exists() and any(directory.glob("calibrated_*.parquet")):
            return True
    return False


def available_real_stations() -> list[str]:
    return [label for label in DEFAULT_REAL_STATIONS if station_has_calibrated_data(label)]


def resolve_selection(config: dict) -> TuningSelection:
    selection_cfg = config.get("selection", {})
    simulation_stations = _normalize_station_list(
        selection_cfg.get("simulation_stations"),
        default=DEFAULT_SIMULATION_STATIONS,
    )
    if not simulation_stations:
        raise ValueError("At least one simulation station is required.")

    default_real = available_real_stations() or DEFAULT_REAL_STATIONS
    real_stations = _normalize_station_list(
        selection_cfg.get("real_stations"),
        default=default_real,
    )
    if not real_stations:
        raise ValueError("At least one real-data station is required.")

    return TuningSelection(
        simulation_stations=simulation_stations,
        real_stations=real_stations,
        simulation_date_ranges=_normalize_date_ranges(selection_cfg.get("simulation_date_ranges")),
        real_date_ranges=_normalize_date_ranges(selection_cfg.get("real_date_ranges")),
    )


def collect_calibrated_file_entries(station_labels: Iterable[str]) -> list[tuple[str, Path]]:
    files_by_key: dict[tuple[str, str], Path] = {}
    for station_label in station_labels:
        for input_dir in station_task3_input_dirs(station_label):
            if not input_dir.exists():
                continue
            for parquet_path in sorted(input_dir.glob("calibrated_*.parquet")):
                files_by_key[(station_label, parquet_path.name)] = parquet_path
    return [(station_label, files_by_key[(station_label, name)]) for station_label, name in sorted(files_by_key)]


def date_range_mask(
    datetimes: pd.Series,
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.Series:
    if not date_ranges:
        return pd.Series(True, index=datetimes.index)
    parsed = pd.to_datetime(datetimes, errors="coerce")
    mask = pd.Series(False, index=datetimes.index)
    for start, end in date_ranges:
        mask |= (parsed >= start) & (parsed <= end)
    return mask.fillna(False)


def filter_frame_by_datetime(
    frame: pd.DataFrame,
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> pd.DataFrame:
    if not date_ranges:
        return frame
    if "datetime" not in frame.columns:
        return frame.iloc[0:0].copy()
    mask = date_range_mask(frame["datetime"], date_ranges)
    return frame.loc[mask].copy()


def group_label(stations: list[str], *, fallback: str) -> str:
    if len(stations) == 1:
        return stations[0]
    return fallback
