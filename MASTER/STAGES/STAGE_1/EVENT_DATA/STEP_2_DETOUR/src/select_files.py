from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import calendar
import re

import pandas as pd
import yaml


TIMESTAMP_PATTERN = re.compile(r"mi0(?P<digits>\d{11,})", re.IGNORECASE)
SUPPORTED_SUFFIXES = {".parquet", ".csv"}
PLOT_RATE_MODES = {"rates", "zscores"}


@dataclass(frozen=True)
class RateOutputConfig:
    enabled: bool = False
    time_column: str = ""
    bin_size: str = "1min"
    output_file: Path = Path("rates/gate_rates_per_minute.parquet")
    moving_average_minutes: int | None = None


@dataclass(frozen=True)
class PlottingConfig:
    enabled: bool = False
    gate_columns: list[str] = field(default_factory=list)
    output_file: Path = Path("plots/gate_rates.png")
    rate_mode: str = "rates"


@dataclass(frozen=True)
class SelectionConfig:
    station_base_dir: Path
    stations: list[str]
    input_subdir: Path
    start_datetime: datetime
    end_datetime: datetime
    output_dir: Path
    rate_output: RateOutputConfig = field(default_factory=RateOutputConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    def resolve_output_path(self, path_value: Path) -> Path:
        if path_value.is_absolute():
            return path_value
        return self.output_dir / path_value


@dataclass(frozen=True)
class SelectedFile:
    station: str
    path: Path
    timestamp: datetime


@dataclass(frozen=True)
class FileSelectionResult:
    selected_files: list[SelectedFile]
    invalid_files: list[Path]


def parse_filename_timestamp(filename: str) -> datetime:
    match = TIMESTAMP_PATTERN.search(Path(filename).stem)
    if match is None:
        raise ValueError(f"No mi0*YYDDDHHMMSS timestamp found in '{filename}'.")

    digits = match.group("digits")[-11:]
    yy = int(digits[0:2])
    day_of_year = int(digits[2:5])
    hour = int(digits[5:7])
    minute = int(digits[7:9])
    second = int(digits[9:11])

    year = 2000 + yy
    max_day = 366 if calendar.isleap(year) else 365
    if not 1 <= day_of_year <= max_day:
        raise ValueError(f"Invalid day-of-year {day_of_year} in '{filename}'.")
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"Invalid clock time in '{filename}'.")

    return datetime(year, 1, 1) + timedelta(
        days=day_of_year - 1,
        hours=hour,
        minutes=minute,
        seconds=second,
    )


def load_selection_config(config_path: Path) -> SelectionConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selection_section = raw_config.get("selection")
    if not isinstance(selection_section, dict):
        raise ValueError("Selection config must define a top-level 'selection' mapping.")

    stations = selection_section.get("stations") or []
    if isinstance(stations, str) or not isinstance(stations, list):
        raise ValueError("Selection config field 'stations' must be a list of station names.")
    if not stations:
        raise ValueError("Selection config must include at least one station.")

    start_datetime = _parse_datetime(selection_section.get("start_datetime"), "start_datetime")
    end_datetime = _parse_datetime(selection_section.get("end_datetime"), "end_datetime")
    if start_datetime >= end_datetime:
        raise ValueError("Selection config has an invalid date range: start_datetime must be earlier than end_datetime.")

    rate_output_section = raw_config.get("rate_output") or {}
    rate_output = RateOutputConfig(
        enabled=bool(rate_output_section.get("enabled", False)),
        time_column=str(rate_output_section.get("time_column", "") or ""),
        bin_size=str(rate_output_section.get("bin_size", "1min") or "1min"),
        output_file=Path(rate_output_section.get("output_file") or "rates/gate_rates_per_minute.parquet"),
        moving_average_minutes=_parse_moving_average_minutes(rate_output_section.get("moving_average_minutes")),
    )
    if rate_output.enabled and not rate_output.time_column:
        raise ValueError("rate_output.enabled is true but rate_output.time_column is missing.")

    plotting_section = raw_config.get("plotting") or {}
    plotting = PlottingConfig(
        enabled=bool(plotting_section.get("enabled", False)),
        gate_columns=list(plotting_section.get("gate_columns") or []),
        output_file=Path(plotting_section.get("output_file") or "plots/gate_rates.png"),
        rate_mode=_parse_plotting_rate_mode(plotting_section.get("rate_mode")),
    )

    return SelectionConfig(
        station_base_dir=Path(_require_selection_value(selection_section, "station_base_dir")),
        stations=[str(station) for station in stations],
        input_subdir=Path(_require_selection_value(selection_section, "input_subdir")),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        output_dir=Path(_require_selection_value(selection_section, "output_dir")),
        rate_output=rate_output,
        plotting=plotting,
    )


def discover_selected_files(config: SelectionConfig) -> FileSelectionResult:
    if not config.station_base_dir.exists():
        raise FileNotFoundError(f"Station base directory does not exist: {config.station_base_dir}")

    selected_files: list[SelectedFile] = []
    invalid_files: list[Path] = []

    for station in config.stations:
        station_dir = config.station_base_dir / station
        if not station_dir.exists():
            raise FileNotFoundError(f"Station directory does not exist: {station_dir}")

        input_dir = station_dir / config.input_subdir
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist for station {station}: {input_dir}")

        for path in sorted(input_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            try:
                timestamp = parse_filename_timestamp(path.name)
            except ValueError:
                invalid_files.append(path)
                continue

            if config.start_datetime <= timestamp < config.end_datetime:
                selected_files.append(SelectedFile(station=station, path=path, timestamp=timestamp))

    selected_files.sort(key=lambda item: (item.timestamp, item.station, item.path.name))
    return FileSelectionResult(selected_files=selected_files, invalid_files=invalid_files)


def _parse_datetime(value: object, field_name: str) -> datetime:
    if value in (None, ""):
        raise ValueError(f"Selection config is missing {field_name}.")

    parsed = pd.to_datetime(value, errors="raise")
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    raise ValueError(f"Selection config field {field_name} could not be parsed as a datetime.")


def _require_selection_value(selection_section: dict[str, object], field_name: str) -> object:
    value = selection_section.get(field_name)
    if value in (None, ""):
        raise ValueError(f"Selection config is missing {field_name}.")
    return value


def _parse_moving_average_minutes(value: object) -> int | None:
    if value in (None, ""):
        return None

    moving_average = int(value)
    if moving_average <= 1:
        return None
    return moving_average


def _parse_plotting_rate_mode(value: object) -> str:
    if value in (None, ""):
        return "rates"

    rate_mode = str(value).strip().lower()
    if rate_mode not in PLOT_RATE_MODES:
        valid_modes = ", ".join(sorted(PLOT_RATE_MODES))
        raise ValueError(f"plotting.rate_mode must be one of: {valid_modes}.")
    return rate_mode
