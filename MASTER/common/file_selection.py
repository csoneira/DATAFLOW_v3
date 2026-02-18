"""Utilities for selecting the newest DATAFLOW artifacts by basename."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import os
import re
import shutil
from typing import Callable, Iterable, Mapping, Optional

_ORDER_SUFFIXES: tuple[str, ...] = (
    ".hld.tar.gz",
    ".hld-tar-gz",
    ".tar.gz",
    ".hld",
    ".parquet",
    ".dat",
    ".root",
    ".list",
    ".lis",
    ".fit",
    ".corr",
)


def _strip_order_suffixes(name: str) -> str:
    """Remove known multi-part extensions so comparisons focus on the basename."""
    lowered = name.lower()
    for suffix in _ORDER_SUFFIXES:
        if lowered.endswith(suffix):
            return _strip_order_suffixes(name[: -len(suffix)])
    return name


def _normalize_prefix(name: str) -> str:
    """Normalize minI* prefixes back to mi01* for comparisons."""
    lowered = name.lower()
    if lowered.startswith("mini"):
        return "mi01" + lowered[4:]
    return lowered


def _station_prefix(station: str) -> str:
    try:
        station_int = int(str(station))
    except (TypeError, ValueError):
        return f"mi0{station}".lower()
    return f"mi0{station_int}".lower()


def newest_order_key(file_name: str, station: str) -> str:
    """Compute a comparison key that ignores the mi0X prefix when possible."""
    base = _strip_order_suffixes(file_name)
    normalized = _normalize_prefix(base)
    prefix = _station_prefix(station)
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    match = re.search(r"(\d{11})$", normalized)
    if match:
        return match.group(1)
    return normalized


def select_latest_candidate(files: Iterable[str], station: str) -> Optional[str]:
    """Return the lexicographically-last artifact after normalizing its prefix."""
    candidates = [name for name in files if name]
    if not candidates:
        return None
    return max(candidates, key=lambda name: newest_order_key(name, station))


def _coerce_date(raw_value: object) -> Optional[date]:
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value

    text = str(raw_value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None

    token = text.split("T", 1)[0].split(" ", 1)[0]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


def load_date_range_from_config(config: Mapping[str, object]) -> tuple[Optional[date], Optional[date]]:
    """Load an inclusive [start, end] date range from config."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    raw_range = config.get("date_range")
    if isinstance(raw_range, Mapping):
        start_date = _coerce_date(
            raw_range.get("start")
            or raw_range.get("from")
            or raw_range.get("start_date")
        )
        end_date = _coerce_date(
            raw_range.get("end")
            or raw_range.get("to")
            or raw_range.get("end_date")
        )
    elif isinstance(raw_range, (list, tuple)):
        if len(raw_range) >= 1:
            start_date = _coerce_date(raw_range[0])
        if len(raw_range) >= 2:
            end_date = _coerce_date(raw_range[1])
    elif isinstance(raw_range, str):
        text = raw_range.strip()
        if text:
            separator = "," if "," in text else ".." if ".." in text else None
            if separator:
                left, right = text.split(separator, 1)
                start_date = _coerce_date(left.strip())
                end_date = _coerce_date(right.strip())
            else:
                single = _coerce_date(text)
                start_date = single
                end_date = single

    if start_date is None and end_date is None:
        legacy_ranges = config.get("date_ranges")
        if isinstance(legacy_ranges, list) and legacy_ranges:
            first_range = legacy_ranges[0]
            if isinstance(first_range, Mapping):
                start_date = _coerce_date(
                    first_range.get("start")
                    or first_range.get("from")
                    or first_range.get("start_date")
                )
                end_date = _coerce_date(
                    first_range.get("end")
                    or first_range.get("to")
                    or first_range.get("end_date")
                )

    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date

    return start_date, end_date


def extract_run_datetime_from_name(file_name: str) -> Optional[datetime]:
    """Extract YYDDDHHMMSS timestamp from DATAFLOW artifact names."""
    base = _strip_order_suffixes(file_name)
    normalized = _normalize_prefix(base)
    match = re.search(r"(\d{11})$", normalized)
    if not match:
        return None

    stamp = match.group(1)
    try:
        yy = int(stamp[0:2])
        day_of_year = int(stamp[2:5])
        hour = int(stamp[5:7])
        minute = int(stamp[7:9])
        second = int(stamp[9:11])
    except ValueError:
        return None

    if not (1 <= day_of_year <= 366):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None

    year = 2000 + yy
    try:
        return datetime(year, 1, 1) + timedelta(
            days=day_of_year - 1,
            hours=hour,
            minutes=minute,
            seconds=second,
        )
    except ValueError:
        return None


def file_name_in_date_range(
    file_name: str,
    start_date: Optional[date],
    end_date: Optional[date],
) -> bool:
    """Return True when file timestamp is inside configured date range."""
    if start_date is None and end_date is None:
        return True

    file_datetime = extract_run_datetime_from_name(file_name)
    if file_datetime is None:
        # Keep files with unparseable names to avoid accidental data loss.
        return True

    day_value = file_datetime.date()
    if start_date and day_value < start_date:
        return False
    if end_date and day_value > end_date:
        return False
    return True


def sync_unprocessed_with_date_range(
    *,
    config: Mapping[str, object],
    unprocessed_directory: str,
    out_of_date_directory: str,
    log_fn: Callable[[str], None] = print,
) -> tuple[Optional[date], Optional[date]]:
    """
    Move out-of-range files out of UNPROCESSED and restore in-range files when needed.
    """
    start_date, end_date = load_date_range_from_config(config)

    os.makedirs(unprocessed_directory, exist_ok=True)
    os.makedirs(out_of_date_directory, exist_ok=True)

    if start_date is None and end_date is None:
        return start_date, end_date

    moved_out = 0
    for file_name in sorted(os.listdir(unprocessed_directory)):
        src_path = os.path.join(unprocessed_directory, file_name)
        if not os.path.isfile(src_path):
            continue
        if file_name_in_date_range(file_name, start_date, end_date):
            continue

        dst_path = os.path.join(out_of_date_directory, file_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.move(src_path, dst_path)
        moved_out += 1
        log_fn(
            f"[DATE_RANGE] Moved out-of-range file to OUT_OF_DATE_DIRECTORY: {file_name}"
        )

    remaining = [
        name
        for name in os.listdir(unprocessed_directory)
        if os.path.isfile(os.path.join(unprocessed_directory, name))
    ]
    if remaining:
        if moved_out:
            log_fn(
                f"[DATE_RANGE] Date filtering complete; kept {len(remaining)} file(s) in UNPROCESSED_DIRECTORY."
            )
        return start_date, end_date

    log_fn(
        "[DATE_RANGE] UNPROCESSED_DIRECTORY is empty; checking OUT_OF_DATE_DIRECTORY for files in range."
    )
    restored = 0
    for file_name in sorted(os.listdir(out_of_date_directory)):
        src_path = os.path.join(out_of_date_directory, file_name)
        if not os.path.isfile(src_path):
            continue
        if not file_name_in_date_range(file_name, start_date, end_date):
            continue

        dst_path = os.path.join(unprocessed_directory, file_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.move(src_path, dst_path)
        restored += 1
        log_fn(
            f"[DATE_RANGE] Restored in-range file to UNPROCESSED_DIRECTORY: {file_name}"
        )

    if restored == 0:
        log_fn("[DATE_RANGE] No in-range files found in OUT_OF_DATE_DIRECTORY.")

    return start_date, end_date
