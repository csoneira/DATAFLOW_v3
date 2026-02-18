"""Shared parsing for station/date selection config with master overrides."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
import re
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from MASTER.common.path_config import get_master_config_root


DateRange = Tuple[Optional[date], Optional[date]]
MASTER_SELECTION_FILENAME = "config_selection.yaml"


@dataclass(frozen=True)
class SelectionConfig:
    stations: Optional[Tuple[int, ...]]
    date_ranges: Tuple[DateRange, ...]


def load_yaml_mapping(path: str | Path) -> Mapping[str, object]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return {}
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_date_value(value: object) -> Optional[date]:
    if value in ("", None):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z") and "T" in text:
        text = text[:-1] + "+00:00"

    parsed: Optional[datetime] = None
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            if fmt is None:
                parsed = datetime.fromisoformat(text)
            else:
                parsed = datetime.strptime(text, fmt)
            break
        except Exception:
            parsed = None
    if parsed is None:
        return None
    return parsed.date()


def _normalize_station_list(value: object) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in {"all", "*"}:
            return None
        tokens = [tok for tok in re.split(r"[,;\s]+", text) if tok]
        parsed = []
        for token in tokens:
            try:
                parsed.append(int(token))
            except Exception:
                continue
        if not parsed:
            return None
        return tuple(sorted(set(parsed)))
    if isinstance(value, (list, tuple, set)):
        parsed = []
        for item in value:
            if isinstance(item, str) and item.strip().lower() in {"all", "*"}:
                return None
            try:
                parsed.append(int(item))
            except Exception:
                continue
        if not parsed:
            return None
        return tuple(sorted(set(parsed)))
    if isinstance(value, (int, float)):
        return (int(value),)
    return None


def _normalize_date_range(start_day: Optional[date], end_day: Optional[date]) -> Optional[DateRange]:
    if start_day is None and end_day is None:
        return None
    if start_day is not None and end_day is not None and start_day > end_day:
        start_day, end_day = end_day, start_day
    return (start_day, end_day)


def _collect_ranges_from_node(node: object, out_ranges: list[DateRange]) -> None:
    if not isinstance(node, Mapping):
        return

    selection_node = node.get("selection")
    if isinstance(selection_node, Mapping):
        range_list = selection_node.get("date_ranges")
        if isinstance(range_list, list):
            for item in range_list:
                if not isinstance(item, Mapping):
                    continue
                normalized = _normalize_date_range(
                    _parse_date_value(item.get("start")),
                    _parse_date_value(item.get("end")),
                )
                if normalized is not None:
                    out_ranges.append(normalized)

    legacy_range = node.get("date_range")
    if isinstance(legacy_range, Mapping):
        normalized = _normalize_date_range(
            _parse_date_value(legacy_range.get("start")),
            _parse_date_value(legacy_range.get("end")),
        )
        if normalized is not None:
            out_ranges.append(normalized)

    legacy_ranges = node.get("date_ranges")
    if isinstance(legacy_ranges, list):
        for item in legacy_ranges:
            if not isinstance(item, Mapping):
                continue
            normalized = _normalize_date_range(
                _parse_date_value(item.get("start")),
                _parse_date_value(item.get("end")),
            )
            if normalized is not None:
                out_ranges.append(normalized)


def extract_selection(config: Mapping[str, object]) -> SelectionConfig:
    ranges: list[DateRange] = []
    _collect_ranges_from_node(config, ranges)

    deduped_ranges: list[DateRange] = []
    seen_ranges: set[DateRange] = set()
    for item in ranges:
        if item in seen_ranges:
            continue
        seen_ranges.add(item)
        deduped_ranges.append(item)

    stations: Optional[Tuple[int, ...]] = None
    selection_node = config.get("selection")
    if isinstance(selection_node, Mapping):
        stations = _normalize_station_list(selection_node.get("stations"))
    if stations is None:
        stations = _normalize_station_list(config.get("stations"))

    return SelectionConfig(stations=stations, date_ranges=tuple(deduped_ranges))


def combine_local_selections(configs: Sequence[Mapping[str, object]]) -> SelectionConfig:
    combined_ranges: list[DateRange] = []
    stations: Optional[Tuple[int, ...]] = None
    for config in configs:
        selection = extract_selection(config)
        if selection.stations is not None:
            stations = selection.stations
        combined_ranges.extend(selection.date_ranges)

    deduped_ranges: list[DateRange] = []
    seen_ranges: set[DateRange] = set()
    for item in combined_ranges:
        if item in seen_ranges:
            continue
        seen_ranges.add(item)
        deduped_ranges.append(item)

    return SelectionConfig(stations=stations, date_ranges=tuple(deduped_ranges))


def load_master_selection(master_config_root: str | Path | None = None) -> SelectionConfig:
    root = Path(master_config_root).expanduser() if master_config_root is not None else get_master_config_root()
    config = load_yaml_mapping(root / MASTER_SELECTION_FILENAME)
    return extract_selection(config)


def resolve_selection_from_configs(
    configs: Sequence[Mapping[str, object]],
    *,
    master_config_root: str | Path | None = None,
) -> SelectionConfig:
    local_selection = combine_local_selections(configs)
    master_selection = load_master_selection(master_config_root)

    stations = (
        master_selection.stations
        if master_selection.stations is not None
        else local_selection.stations
    )
    date_ranges = (
        master_selection.date_ranges
        if master_selection.date_ranges
        else local_selection.date_ranges
    )
    return SelectionConfig(stations=stations, date_ranges=date_ranges)


def load_selection_for_paths(
    config_paths: Iterable[str | Path],
    *,
    master_config_root: str | Path | None = None,
) -> SelectionConfig:
    configs = [load_yaml_mapping(path) for path in config_paths]
    return resolve_selection_from_configs(configs, master_config_root=master_config_root)


def station_is_selected(station: int | str, stations: Optional[Sequence[int]]) -> bool:
    station_id = parse_station_id(station)
    if station_id is None:
        return False
    if stations is None:
        return True
    return station_id in set(int(item) for item in stations)


def parse_station_id(station: int | str) -> Optional[int]:
    try:
        return int(str(station))
    except Exception:
        return None


def date_filter_allowed_for_station(station: int | str) -> bool:
    station_id = parse_station_id(station)
    if station_id is None:
        return True
    return station_id != 0


def effective_date_ranges_for_station(
    station: int | str,
    ranges: Sequence[DateRange],
) -> Tuple[DateRange, ...]:
    if not date_filter_allowed_for_station(station):
        return ()
    return tuple(ranges)


def date_in_ranges(day_value: date, ranges: Sequence[DateRange]) -> bool:
    if not ranges:
        return True
    for start_day, end_day in ranges:
        if start_day is not None and day_value < start_day:
            continue
        if end_day is not None and day_value > end_day:
            continue
        return True
    return False


def datetime_in_ranges(value: datetime, ranges: Sequence[DateRange]) -> bool:
    return date_in_ranges(value.date(), ranges)


def serialize_date_ranges_for_shell(ranges: Sequence[DateRange]) -> tuple[str, str]:
    epoch_chunks: list[str] = []
    label_chunks: list[str] = []

    for start_day, end_day in ranges:
        start_epoch = ""
        end_epoch = ""
        if start_day is not None:
            start_dt = datetime.combine(start_day, time.min).replace(tzinfo=timezone.utc)
            start_epoch = str(int(start_dt.timestamp()))
        if end_day is not None:
            end_dt = datetime.combine(end_day, time.max).replace(tzinfo=timezone.utc)
            end_epoch = str(int(end_dt.timestamp()))
        epoch_chunks.append(f"{start_epoch},{end_epoch}")
        label_chunks.append(
            f"{'' if start_day is None else start_day.isoformat()}|{'' if end_day is None else end_day.isoformat()}"
        )

    return ";".join(epoch_chunks), ";".join(label_chunks)
