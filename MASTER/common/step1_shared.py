"""Shared STEP_1 helpers for TASK_1..TASK_5 scripts."""

from __future__ import annotations

import argparse
import builtins
import csv
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_STATION_CHOICES: Tuple[str, ...] = ("0", "1", "2", "3", "4")
DEFAULT_IMPORTANT_KEYWORDS: Tuple[str, ...] = (
    "error",
    "warning",
    "failed",
    "exception",
    "traceback",
    "usage",
)

EVENTS_PER_SECOND_MAX = 100
EVENTS_PER_SECOND_COLUMNS = [
    *(f"events_per_second_{idx}_count" for idx in range(EVENTS_PER_SECOND_MAX + 1)),
    "events_per_second_total_seconds",
    "events_per_second_global_rate",
]


def build_step1_cli_parser(
    description: str,
    station_choices: Sequence[str] = DEFAULT_STATION_CHOICES,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "station",
        nargs="?",
        choices=tuple(station_choices),
        help="Station identifier (0, 1, 2, 3, or 4).",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional input file path to process instead of auto-selecting.",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file_flag",
        help="Optional input file path (named form).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose execution logging.",
    )
    return parser


def validate_step1_input_file_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.input_file and args.input_file_flag:
        parser.error("Use either positional input_file or --input-file, not both.")


def build_step1_filtered_print(
    verbose: bool,
    debug_mode_getter: Callable[[], bool],
    raw_print: Callable[..., None] = builtins.print,
    important_keywords: Sequence[str] = DEFAULT_IMPORTANT_KEYWORDS,
) -> Callable[..., None]:
    important = tuple(keyword.lower() for keyword in important_keywords)

    def is_important_message(message: str) -> bool:
        lowered = message.lower()
        if any(keyword in lowered for keyword in important):
            return True
        return "total execution time" in lowered or "data purity" in lowered

    def filtered_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or debug_mode_getter() or verbose:
            raw_print(*args, **kwargs)
            return
        message = " ".join(str(arg) for arg in args)
        if is_important_message(message):
            raw_print(*args, **kwargs)

    return filtered_print


def resolve_step1_station(
    cli_station: str | None,
    run_jupyter_notebook: bool,
    default_station: str,
    parser: argparse.ArgumentParser,
    station_choices: Sequence[str] = DEFAULT_STATION_CHOICES,
    config_hint: str = "TASK_<N>/config_task_<N>.yaml",
) -> str:
    if cli_station is not None:
        station = str(cli_station)
    elif run_jupyter_notebook:
        station = str(default_station)
    else:
        parser.error(
            f"No station provided. Pass <station> or enable run_jupyter_notebook in {config_hint}."
        )

    if station not in station_choices:
        parser.error("Invalid station. Choose one of: 0, 1, 2, 3, 4.")
    return station


def apply_step1_task_parameter_overrides(
    config_obj: dict,
    station_id: str,
    task_parameter_path: str | Path,
    fallback_parameter_path: str | Path,
    task_number: int,
    update_fn: Callable[[dict, Path | str, str], dict],
    log_fn: Callable[..., None] = builtins.print,
) -> dict:
    task_path = Path(task_parameter_path)
    if task_path.exists():
        config_obj = update_fn(config_obj, task_path, station_id)
        log_fn(f"Warning: Loaded task parameters from {task_path}")
        return config_obj

    fallback_path = Path(fallback_parameter_path)
    if fallback_path.exists():
        log_fn(f"Warning: Task parameters file not found; falling back to {fallback_path}")
        config_obj = update_fn(config_obj, fallback_path, station_id)
    else:
        log_fn(f"Warning: No parameters file found for task {task_number}")
    return config_obj


def normalize_analysis_mode_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "0", "1"}:
        return text
    if text in {"0.0", "1.0"}:
        return text[0]
    if "1" in text:
        return "1"
    if "0" in text:
        return "0"
    return ""


def sanitize_analysis_mode_rows(rows: List[Dict[str, object]]) -> int:
    fixed = 0
    for row in rows:
        if "analysis_mode" not in row:
            continue
        clean_value = normalize_analysis_mode_value(row.get("analysis_mode"))
        if row.get("analysis_mode") != clean_value:
            row["analysis_mode"] = clean_value
            fixed += 1
    return fixed


def repair_metadata_file(metadata_path: Path) -> int:
    original_limit = csv.field_size_limit()
    csv.field_size_limit(2**31 - 1)
    try:
        with metadata_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(existing) for existing in reader]
    finally:
        csv.field_size_limit(original_limit)

    if not fieldnames:
        return 0

    fixed = sanitize_analysis_mode_rows(rows)
    if fixed:
        with metadata_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return fixed


def save_metadata(
    metadata_path: str | Path,
    row: Dict[str, object],
    preferred_fieldnames: Iterable[str] | None = None,
    log_fn: Callable[..., None] = builtins.print,
) -> Path:
    metadata_path = Path(metadata_path)
    rows: List[Dict[str, object]] = []
    fieldnames: List[str] = []

    def normalize_row(raw: Dict[str, object]) -> Dict[str, object]:
        return {key: value for key, value in raw.items() if key is not None}

    def load_existing_rows() -> Tuple[List[str], List[Dict[str, object]]]:
        with metadata_path.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [normalize_row(existing) for existing in reader]
            return existing_fields, existing_rows

    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        try:
            fieldnames, existing_rows = load_existing_rows()
        except csv.Error as exc:
            if "field larger than field limit" in str(exc).lower():
                fixed = repair_metadata_file(metadata_path)
                log_fn(
                    f"Detected oversized analysis_mode entries in {metadata_path}; normalized {fixed} row(s)."
                )
                try:
                    fieldnames, existing_rows = load_existing_rows()
                except csv.Error as err:
                    raise RuntimeError(
                        f"Failed to repair metadata file {metadata_path} after detecting oversized fields."
                    ) from err
            else:
                raise
        rows.extend(existing_rows)

    rows.append(normalize_row(dict(row)))

    fixed_during_append = sanitize_analysis_mode_rows(rows)
    if fixed_during_append:
        log_fn(f"Clamped analysis_mode to 0/1 in {fixed_during_append} metadata row(s).")

    seen = set(fieldnames)
    for item in rows:
        for key in item.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    if preferred_fieldnames:
        preferred = [name for name in preferred_fieldnames if name in seen]
        remainder = [name for name in fieldnames if name not in preferred]
        fieldnames = preferred + remainder
        if "data_purity_percentage" in fieldnames:
            fieldnames = [name for name in fieldnames if name != "data_purity_percentage"] + [
                "data_purity_percentage"
            ]

    with metadata_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in rows:
            formatted = {}
            for key in fieldnames:
                if key in EVENTS_PER_SECOND_COLUMNS:
                    value = item.get(key, 0)
                    if value in ("", None) or (isinstance(value, float) and math.isnan(value)):
                        formatted[key] = 0
                    else:
                        formatted[key] = value
                    continue
                value = item.get(key, "")
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    formatted[key] = ""
                elif isinstance(value, (list, dict, np.ndarray)):
                    formatted[key] = str(value)
                else:
                    formatted[key] = value
            writer.writerow(formatted)

    return metadata_path


def build_events_per_second_metadata(
    df: pd.DataFrame,
    time_columns: Tuple[str, ...] = ("datetime", "Time"),
) -> Dict[str, object]:
    metadata = {column: 0 for column in EVENTS_PER_SECOND_COLUMNS}
    if df is None or df.empty:
        return metadata

    time_col = next((col for col in time_columns if col in df.columns), None)
    if time_col is not None:
        times = pd.to_datetime(df[time_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        times = pd.Series(df.index)
    else:
        return metadata

    times = pd.Series(times).dropna()
    if times.empty:
        return metadata

    times = times.dt.floor("s")
    start_time = times.min()
    end_time = times.max()
    if pd.isna(start_time) or pd.isna(end_time):
        return metadata

    full_range = pd.date_range(start=start_time, end=end_time, freq="s")
    events_per_second = times.value_counts().reindex(full_range, fill_value=0).sort_index()

    total_seconds = int(events_per_second.size)
    total_events = int(events_per_second.sum())
    metadata["events_per_second_total_seconds"] = total_seconds
    metadata["events_per_second_global_rate"] = round(total_events / total_seconds, 6) if total_seconds > 0 else 0

    hist_counts = events_per_second.value_counts()
    for events_count, seconds_count in hist_counts.items():
        events_count_int = int(events_count)
        if 0 <= events_count_int <= EVENTS_PER_SECOND_MAX:
            metadata[f"events_per_second_{events_count_int}_count"] = int(seconds_count)

    return metadata


def add_normalized_count_metadata(
    metadata: Dict[str, object],
    denominator_seconds: object,
    log_fn: Callable[..., None] = builtins.print,
) -> None:
    try:
        denom = float(denominator_seconds) if denominator_seconds is not None else 0.0
    except (TypeError, ValueError):
        denom = 0.0

    metadata["count_rate_denominator_seconds"] = int(denom) if denom > 0 else 0

    if denom <= 0:
        log_fn("[count-rates] Denominator seconds is 0; skipping normalized count columns.")
        return

    for key, value in list(metadata.items()):
        if not isinstance(key, str):
            continue

        is_count = key.endswith("_count")
        is_entries = key.endswith(("_entries", "_entries_final", "_entries_initial"))
        if not (is_count or is_entries):
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(num):
            continue

        if is_count:
            if key.startswith("events_per_second_"):
                out_key = key[: -len("_count")] + "_fraction"
            else:
                out_key = key[: -len("_count")] + "_rate_hz"
        else:
            out_key = key + "_rate_hz"

        metadata[out_key] = round(num / denom, 6)


def load_itineraries_from_file(
    file_path: Path,
    required: bool = True,
    log_fn: Callable[..., None] = builtins.print,
    echo_entries: bool = True,
    header_suffix: str = ":",
) -> list[list[str]]:
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        if log_fn is not None:
            log_fn(f"Loading itineraries from {file_path}{header_suffix}")
        for raw_line in itinerary_file:
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                if log_fn is not None and echo_entries:
                    log_fn(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")
    return itineraries


def write_itineraries_to_file(file_path: Path, itineraries: Iterable[Iterable[str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_itineraries: dict[tuple[str, ...], None] = {}

    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if not itinerary_tuple:
            continue
        unique_itineraries.setdefault(itinerary_tuple, None)

    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique_itineraries:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")


def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2


def collect_columns(columns: Iterable[str], pattern) -> list[str]:
    return [column for column in columns if pattern.match(column)]
