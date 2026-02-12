#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "MASTER").is_dir()),
    Path.home() / "DATAFLOW_v3",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.common.plot_utils import pdf_save_rasterized_page

PLOTTER_ROOT = REPO_ROOT / "MASTER" / "ANCILLARY" / "PLOTTERS" / "RATES"
OUTPUT_DIR = PLOTTER_ROOT / "PLOTS"
DEFAULT_OUTPUT_NAME = "events_per_second_report.pdf"

RATE_COLUMN_PREFIX = "events_per_second_"
DEFAULT_MAX_RATE = 100

DEFAULT_STATIONS = ["0", "1", "2", "3", "4"]
DEFAULT_TASKS = ["1", "2", "3", "4", "5"]

START_END_COLUMN_CANDIDATES: Sequence[Tuple[str, str]] = (
    ("start_time", "end_time"),
    ("Start_Time", "End_Time"),
    ("start_datetime", "end_datetime"),
    ("start", "end"),
)

SINGLE_TIME_COLUMNS = (
    "center_time",
    "start_time",
    "Start_Time",
    "start_datetime",
    "start",
    "datetime",
    "Time",
    "execution_timestamp",
)

FILENAME_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = (
    (re.compile(r"(?<!\d)(?P<year>\d{4})[-_]?((?P<month>\d{2})[-_]?)(?P<day>\d{2})[T_\-.]?(?P<hour>\d{2})[\.:_-]?(?P<minute>\d{2})[\.:_-]?(?P<second>\d{2})(?!\d)"), "ymd_hms"),
    (re.compile(r"(?<!\d)(?P<year>\d{4})[-_]?((?P<doy>\d{3}))[T_\-.]?(?P<hour>\d{2})[\.:_-]?(?P<minute>\d{2})[\.:_-]?(?P<second>\d{2})(?!\d)"), "ydoy_hms"),
    (re.compile(r"(?<!\d)(?P<year>\d{2})[-_]?((?P<month>\d{2})[-_]?)(?P<day>\d{2})[T_\-.]?(?P<hour>\d{2})[\.:_-]?(?P<minute>\d{2})[\.:_-]?(?P<second>\d{2})(?!\d)"), "y2md_hms"),
    (re.compile(r"(?<!\d)(?P<year>\d{4})[-_]?((?P<month>\d{2})[-_]?)(?P<day>\d{2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(?P<year>\d{2})[-_]?((?P<month>\d{2})[-_]?)(?P<day>\d{2})(?!\d)"), "y2md"),
)

# MINGO-style: mi0<station><YY><DDD><HH><MM><SS>
# Example: mi0005265000001 -> YY=05, DDD=265, HH=00, MM=00, SS=01
MI_FILENAME_PATTERN = re.compile(r"mi0\d(?P<digits>\d{11})$", re.IGNORECASE)


@dataclass(frozen=True)
class MetadataSource:
    path: Path
    station: Optional[str]
    task: Optional[str]


def rate_columns(max_rate: int) -> List[str]:
    return [f"{RATE_COLUMN_PREFIX}{idx}_count" for idx in range(max_rate + 1)]


def read_header_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def build_usecols(
    header_cols: Sequence[str],
    last_columns: int,
    max_rate: int,
    preferred_time_columns: Sequence[str],
) -> Optional[List[str]]:
    if last_columns <= 0:
        return None

    tail_cols = (
        list(header_cols[-last_columns:])
        if last_columns < len(header_cols)
        else list(header_cols)
    )

    required: set[str] = set(tail_cols)
    required.update(rate_columns(max_rate))
    required.update(
        [
            "events_per_second_total_seconds",
            "events_per_second_global_rate",
            "filename_base",
            "execution_timestamp",
        ]
    )
    required.update(preferred_time_columns)
    required.update(list(SINGLE_TIME_COLUMNS))
    for start_col, end_col in START_END_COLUMN_CANDIDATES:
        required.add(start_col)
        required.add(end_col)

    return [col for col in header_cols if col in required]


def read_csv_tail(path: Path, n_rows: int, usecols: Optional[Sequence[str]]) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.read_csv(path, low_memory=False, usecols=usecols)

    with path.open("rb") as handle:
        header_line = handle.readline()
        if not header_line:
            return pd.DataFrame()

        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        block_size = 8192
        buffer = b""
        remaining = file_size

        while remaining > 0 and buffer.count(b"\n") <= n_rows:
            read_size = min(block_size, remaining)
            remaining -= read_size
            handle.seek(remaining)
            buffer = handle.read(read_size) + buffer

    header_str = header_line.decode("utf-8", errors="replace").strip()
    lines = buffer.splitlines()
    if lines and lines[0].decode("utf-8", errors="replace").strip() == header_str:
        lines = lines[1:]
    tail_lines = lines[-n_rows:] if n_rows < len(lines) else lines

    text = header_str + "\n" + "\n".join(
        line.decode("utf-8", errors="replace") for line in tail_lines
    )

    return pd.read_csv(io.StringIO(text), low_memory=False, usecols=usecols)


def ensure_rate_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure all *columns* exist, adding missing ones in a non-fragmenting way."""
    missing = [col for col in columns if col not in df.columns]
    if not missing:
        return df
    missing_df = pd.DataFrame(0, index=df.index, columns=missing)
    # pd.concat avoids the repeated frame.insert() pattern that fragments DataFrames.
    return pd.concat([df, missing_df], axis=1)


def metadata_sources(
    stations: Sequence[str], tasks: Sequence[str], explicit_paths: Sequence[Path]
) -> List[MetadataSource]:
    sources: List[MetadataSource] = []

    for path in explicit_paths:
        sources.append(MetadataSource(path=path, station=None, task=None))

    for station in stations:
        for task in tasks:
            path = (
                REPO_ROOT
                / "STATIONS"
                / f"MINGO0{station}"
                / "STAGE_1"
                / "EVENT_DATA"
                / "STEP_1"
                / f"TASK_{task}"
                / "METADATA"
                / f"task_{task}_metadata_specific.csv"
            )
            if path.exists():
                sources.append(MetadataSource(path=path, station=station, task=task))

    unique: Dict[Path, MetadataSource] = {}
    for source in sources:
        unique[source.path] = source

    return list(unique.values())


def read_metadata_csv(
    source: MetadataSource,
    tail_rows: int,
    last_columns: int,
    max_rate: int,
    preferred_time_columns: Sequence[str],
) -> pd.DataFrame:
    header_cols = read_header_columns(source.path)
    usecols = build_usecols(header_cols, last_columns, max_rate, preferred_time_columns)
    df = read_csv_tail(source.path, tail_rows, usecols=usecols)
    df["_source_csv"] = str(source.path)
    if source.station is not None:
        df["station"] = source.station
    if source.task is not None:
        df["task"] = source.task
    return df


def parse_time_value(value: object) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        for fmt in (
            "%Y-%m-%d_%H.%M.%S",
            "%Y-%m-%d_%H.%M.%S.%f",
            "%Y-%m-%d %H.%M.%S",
            "%Y-%m-%d %H.%M.%S.%f",
        ):
            try:
                return pd.Timestamp(datetime.strptime(cleaned, fmt))
            except ValueError:
                pass
        if "_" in cleaned or "." in cleaned:
            alt = cleaned.replace("_", " ").replace(".", ":")
            parsed_alt = pd.to_datetime(alt, errors="coerce")
            if not pd.isna(parsed_alt):
                return parsed_alt

    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        if value > 1e12:
            return pd.to_datetime(int(value), unit="ms", errors="coerce")
        if value > 1e9:
            return pd.to_datetime(int(value), unit="s", errors="coerce")

    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return None

    if pd.isna(parsed):
        return None
    return parsed


def parse_time_from_filename(filename: str) -> Optional[pd.Timestamp]:
    if not filename:
        return None

    # Prefer explicit MINGO basename decoding (YY + day-of-year + time).
    match = MI_FILENAME_PATTERN.search(filename)
    if match:
        digits = match.group("digits")
        try:
            year = 2000 + int(digits[0:2])
            doy = int(digits[2:5])
            hour = int(digits[5:7])
            minute = int(digits[7:9])
            second = int(digits[9:11])
            base = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return pd.Timestamp(base.replace(hour=hour, minute=minute, second=second))
        except (ValueError, OverflowError):
            pass

    for pattern, kind in FILENAME_PATTERNS:
        match = pattern.search(filename)
        if not match:
            continue
        parts = match.groupdict()
        try:
            if kind == "ydoy_hms":
                year = int(parts["year"])
                doy = int(parts["doy"])
                hour = int(parts["hour"])
                minute = int(parts["minute"])
                second = int(parts["second"])
                return pd.Timestamp(datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second))
            if kind in ("ymd_hms", "y2md_hms"):
                year = int(parts["year"])
                if year < 100:
                    year = 2000 + year if year < 80 else 1900 + year
                month = int(parts["month"])
                day = int(parts["day"])
                hour = int(parts["hour"])
                minute = int(parts["minute"])
                second = int(parts["second"])
                return pd.Timestamp(datetime(year, month, day, hour, minute, second))
            if kind in ("ymd", "y2md"):
                year = int(parts["year"])
                if year < 100:
                    year = 2000 + year if year < 80 else 1900 + year
                month = int(parts["month"])
                day = int(parts["day"])
                return pd.Timestamp(datetime(year, month, day, 12, 0, 0))
        except (ValueError, KeyError):
            continue

    return None


def derive_center_time(
    row: pd.Series, preferred_columns: Sequence[str]
) -> Optional[pd.Timestamp]:
    for start_col, end_col in START_END_COLUMN_CANDIDATES:
        if start_col in row and end_col in row:
            start_val = parse_time_value(row.get(start_col))
            end_val = parse_time_value(row.get(end_col))
            if start_val is not None and end_val is not None:
                return start_val + (end_val - start_val) / 2
            if start_val is not None:
                return start_val

    for col in preferred_columns:
        if col in row:
            candidate = parse_time_value(row.get(col))
            if candidate is not None:
                return candidate

    for col in SINGLE_TIME_COLUMNS:
        if col in row:
            candidate = parse_time_value(row.get(col))
            if candidate is not None:
                return candidate

    filename = row.get("filename_base")
    if isinstance(filename, str):
        candidate = parse_time_from_filename(filename)
        if candidate is not None:
            return candidate

    execution_timestamp = row.get("execution_timestamp")
    if execution_timestamp is not None:
        candidate = parse_time_value(execution_timestamp)
        if candidate is not None:
            return candidate

    return None


def compute_time_edges(times: np.ndarray, fallback_seconds: float) -> np.ndarray:
    if len(times) == 1:
        delta_days = fallback_seconds / 86400 if fallback_seconds > 0 else 0.5
        return np.array([times[0] - delta_days / 2, times[0] + delta_days / 2])

    deltas = np.diff(times)
    median_delta = np.median(deltas[deltas > 0]) if np.any(deltas > 0) else 0.5
    edges = np.empty(len(times) + 1)
    edges[1:-1] = (times[:-1] + times[1:]) / 2
    edges[0] = times[0] - median_delta / 2
    edges[-1] = times[-1] + median_delta / 2
    return edges


def plot_rate_heatmap(
    pdf: PdfPages,
    times: np.ndarray,
    rate_matrix: np.ndarray,
    total_seconds: np.ndarray,
    rate_values: np.ndarray,
    title: str,
    log_color: bool,
    fallback_seconds: float,
    x_limits: Optional[Tuple[float, float]] = None,
) -> None:
    if times.size == 0:
        return

    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    rate_matrix = rate_matrix[:, sorted_idx]
    total_seconds = total_seconds[sorted_idx]

    x_edges = compute_time_edges(times, fallback_seconds)
    y_edges = np.arange(rate_values.min() - 0.5, rate_values.max() + 1.5, 1)

    denom = np.where(total_seconds > 0, total_seconds, np.nan)
    freq_matrix = rate_matrix / denom
    masked = np.ma.masked_where(freq_matrix <= 0, freq_matrix)

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.viridis
    if log_color:
        vmax = np.nanmax(freq_matrix) if np.nanmax(freq_matrix) > 0 else 1.0
        norm = LogNorm(vmin=max(1e-5, vmax / 1e5), vmax=vmax)
    else:
        norm = None

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap=cmap,
        shading="auto",
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("Center time of file")
    ax.set_ylabel("Events per second")
    ax.set_ylim(rate_values.min(), rate_values.max())
    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Occurrence rate (Hz)")

    fig.autofmt_xdate()
    pdf_save_rasterized_page(pdf, fig)
    plt.close(fig)


def plot_global_histogram(
    pdf: PdfPages,
    rate_values: np.ndarray,
    total_counts: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rate_values, total_counts, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("Events per second")
    ax.set_ylabel("Total seconds")
    ax.set_xlim(rate_values.min() - 1, rate_values.max() + 1)
    ax.grid(True, axis="y", alpha=0.3)
    pdf_save_rasterized_page(pdf, fig)
    plt.close(fig)


def plot_global_rate_trend(
    pdf: PdfPages,
    times: np.ndarray,
    rates: np.ndarray,
    title: str,
    x_limits: Optional[Tuple[float, float]] = None,
) -> None:
    if times.size == 0:
        return
    valid = np.isfinite(rates) & (rates > 0)
    if not np.any(valid):
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    times = times[valid]
    rates = rates[valid]
    sorted_idx = np.argsort(times)
    ax.plot(times[sorted_idx], rates[sorted_idx], color="#264653", linewidth=1.5)
    ax.scatter(times[sorted_idx], rates[sorted_idx], s=20, color="#2a9d8f")
    ax.set_title(title)
    ax.set_xlabel("Center time of file")
    ax.set_ylabel("Global rate (events / second)")
    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    pdf_save_rasterized_page(pdf, fig)
    plt.close(fig)


def plot_task_rate_trends(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    x_limits: Optional[Tuple[float, float]] = None,
) -> None:
    if "task" not in df.columns or df.empty:
        return

    tasks = [task for task in df["task"].dropna().unique()]
    if not tasks:
        return

    def sort_key(value: object) -> Tuple[int, str]:
        text = str(value)
        return (0, text.zfill(3)) if text.isdigit() else (1, text)

    tasks = sorted(tasks, key=sort_key)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, min(len(tasks), 10))))

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, task in enumerate(tasks):
        task_df = df[df["task"] == task].copy()
        if task_df.empty:
            continue
        task_times = mdates.date2num(task_df["center_time"].to_numpy())
        task_rates = task_df["events_per_second_global_rate"].to_numpy(dtype=float)
        valid = np.isfinite(task_rates) & (task_rates > 0)
        if not np.any(valid):
            continue
        task_times = task_times[valid]
        task_rates = task_rates[valid]
        sorted_idx = np.argsort(task_times)
        color = colors[idx % len(colors)]
        label = f"Task {task}"
        ax.plot(
            task_times[sorted_idx],
            task_rates[sorted_idx],
            color=color,
            linewidth=1.6,
            label=label,
        )
        ax.scatter(
            task_times[sorted_idx],
            task_rates[sorted_idx],
            s=18,
            color=color,
            alpha=0.75,
        )

    ax.set_title(title)
    ax.set_xlabel("Center time of file")
    ax.set_ylabel("Global rate (events / second)")
    if x_limits is not None:
        ax.set_xlim(x_limits[0], x_limits[1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    fig.autofmt_xdate()
    pdf_save_rasterized_page(pdf, fig)
    plt.close(fig)


def build_file_key(df: pd.DataFrame) -> pd.Series:
    """Return a stable per-file key for matching across tasks."""
    filename = df.get("filename_base")
    if filename is None:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

    if "station" in df.columns:
        station = df["station"].astype("string")
        return station.fillna("") + ":" + filename.astype("string").fillna("")

    return filename.astype("string").fillna("")


def filter_to_common_files_across_tasks(
    df: pd.DataFrame,
    tasks: Sequence[object],
    file_key_col: str,
) -> pd.DataFrame:
    """Keep only rows whose file key exists for *all* tasks in *tasks*."""
    if df.empty or not tasks:
        return df

    task_sets: List[set[str]] = []
    for task in tasks:
        subset = df[df["task"] == task]
        keys = subset[file_key_col].dropna().astype(str)
        task_sets.append(set(keys.tolist()))

    if not task_sets:
        return df.iloc[0:0].copy()

    common = set.intersection(*task_sets)
    if not common:
        return df.iloc[0:0].copy()

    return df[df[file_key_col].astype(str).isin(common)].copy()


def plot_matched_file_traces(
    pdf: PdfPages,
    df: pd.DataFrame,
    tasks: Sequence[object],
    value_col: str,
    title: str,
    ylabel: str,
) -> None:
    if df.empty or not tasks or value_col not in df.columns:
        return

    pivot = df.pivot_table(index="_file_key", columns="task", values=value_col, aggfunc="last")
    ordered_tasks = [task for task in tasks if task in pivot.columns]
    if not ordered_tasks:
        return
    pivot = pivot[ordered_tasks].dropna(axis=0, how="any")
    if pivot.empty:
        return

    x = np.arange(len(ordered_tasks))
    fig, ax = plt.subplots(figsize=(12, 5))
    for _, row in pivot.iterrows():
        ax.plot(
            x,
            row.to_numpy(dtype=float),
            color="0.55",
            alpha=0.25,
            linewidth=1.0,
        )

    median = pivot.median(axis=0).to_numpy(dtype=float)
    mean = pivot.mean(axis=0).to_numpy(dtype=float)
    ax.plot(x, median, color="#264653", linewidth=2.2, label="Median")
    ax.plot(x, mean, color="#2a9d8f", linewidth=1.8, linestyle="--", label="Mean")

    ax.set_title(f"{title} (matched files: {len(pivot)})")
    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in ordered_tasks])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    pdf_save_rasterized_page(pdf, fig)
    plt.close(fig)


def summarize_rate_counts(df: pd.DataFrame, rate_cols: Sequence[str]) -> np.ndarray:
    counts = df[rate_cols].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    return counts.sum(axis=0).to_numpy(dtype=float)


def sort_task_values(values: Iterable[object]) -> List[object]:
    def sort_key(value: object) -> Tuple[int, str]:
        text = str(value)
        return (0, text.zfill(3)) if text.isdigit() else (1, text)

    return sorted(values, key=sort_key)


def build_rate_arrays(df: pd.DataFrame, rate_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = mdates.date2num(df["center_time"].to_numpy())
    rate_matrix = (
        df[rate_cols]
        .fillna(0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
        .T
    )
    total_seconds = df["events_per_second_total_seconds"].to_numpy(dtype=float)
    return times, rate_matrix, total_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot events-per-second metadata histograms from stage-1 specific metadata.")
    parser.add_argument(
        "--stations",
        nargs="*",
        default=DEFAULT_STATIONS,
        help="Stations to scan (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        help="Tasks to scan (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--metadata-paths",
        nargs="*",
        default=[],
        help="Explicit metadata CSV paths to include.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--max-rate",
        type=int,
        default=DEFAULT_MAX_RATE,
        help="Maximum events-per-second bin to read (default: 100).",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Only read the last N rows from each CSV (default: 0 = all rows).",
    )
    parser.add_argument(
        "--last-columns",
        type=int,
        default=0,
        help="Only read the last N columns (plus required time/rate columns).",
    )
    parser.add_argument(
        "--time-columns",
        default="",
        help="Comma-separated list of preferred time columns (overrides defaults).",
    )
    xfb_group = parser.add_mutually_exclusive_group()
    xfb_group.add_argument(
        "--x-from-basename",
        dest="x_from_basename",
        action="store_true",
        default=True,
        help="Use timestamp parsed from filename_base for the x-axis (default: enabled).",
    )
    xfb_group.add_argument(
        "--no-x-from-basename",
        dest="x_from_basename",
        action="store_false",
        help="Do NOT use timestamp parsed from filename_base for the x-axis.",
    )
    parser.add_argument(
        "--linear-color",
        action="store_true",
        help="Use linear color scale instead of log.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preferred_time_columns = [
        col.strip() for col in args.time_columns.split(",") if col.strip()
    ]

    explicit_paths = [Path(path).expanduser() for path in args.metadata_paths]
    sources = metadata_sources(args.stations, args.tasks, explicit_paths)

    if not sources:
        print("No metadata CSVs found. Provide --metadata-paths or ensure station folders exist.")
        return 1

    frames = []
    for source in sources:
        try:
            frames.append(
                read_metadata_csv(
                    source,
                    tail_rows=args.tail,
                    last_columns=args.last_columns,
                    max_rate=args.max_rate,
                    preferred_time_columns=preferred_time_columns,
                )
            )
        except Exception as exc:
            print(f"Warning: failed to read {source.path}: {exc}")

    if not frames:
        print("No metadata rows loaded.")
        return 1

    df = pd.concat(frames, ignore_index=True)

    max_rate = max(0, args.max_rate)
    rate_cols = rate_columns(max_rate)
    df = ensure_rate_columns(df, rate_cols)
    # De-fragment after loading/column-normalization to avoid slowdowns + warnings.
    df = df.copy()

    if args.x_from_basename and "filename_base" in df.columns:
        df["center_time"] = df["filename_base"].astype(str).map(parse_time_from_filename)
        missing = df["center_time"].isna()
        if missing.any():
            df.loc[missing, "center_time"] = df.loc[missing].apply(
                lambda row: derive_center_time(row, preferred_time_columns), axis=1
            )
    else:
        df["center_time"] = df.apply(
            lambda row: derive_center_time(row, preferred_time_columns), axis=1
        )
    df = df[~df["center_time"].isna()].copy()
    if df.empty:
        print("No rows with a usable center time. Use --time-columns to specify a time column.")
        return 1

    df["center_time"] = pd.to_datetime(df["center_time"], errors="coerce")
    df = df[~df["center_time"].isna()].copy()

    df["events_per_second_total_seconds"] = pd.to_numeric(
        df.get("events_per_second_total_seconds", pd.Series([np.nan] * len(df))),
        errors="coerce",
    ).fillna(0)
    df["events_per_second_total_seconds"] = df["events_per_second_total_seconds"].astype(float)

    df["events_per_second_global_rate"] = pd.to_numeric(
        df.get("events_per_second_global_rate", pd.Series([np.nan] * len(df))),
        errors="coerce",
    )

    if df["events_per_second_global_rate"].isna().all():
        rates_per_row = (
            df[rate_cols]
            .fillna(0)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        totals = df["events_per_second_total_seconds"].replace(0, np.nan)
        total_events = (rates_per_row.mul(np.arange(max_rate + 1), axis=1)).sum(axis=1)
        df["events_per_second_global_rate"] = (total_events / totals).fillna(0)

    rate_values = np.arange(0, max_rate + 1)
    total_counts = summarize_rate_counts(df, rate_cols)
    global_times, global_rate_matrix, global_total_seconds = build_rate_arrays(df, rate_cols)
    fallback_seconds = float(df["events_per_second_total_seconds"].median())
    # Share time-axis bounds across all time-series plots for easier comparison.
    x_limits: Optional[Tuple[float, float]] = None
    if global_times.size:
        x_limits = (float(np.nanmin(global_times)), float(np.nanmax(global_times)))

    tasks = (
        sort_task_values(df["task"].dropna().unique())
        if "task" in df.columns
        else []
    )
    df_rate_dedup = df
    df_rate_matched = pd.DataFrame()
    matched_file_count = 0
    if tasks and "filename_base" in df.columns and "task" in df.columns:
        df_rate_dedup = df.copy()
        df_rate_dedup["_file_key"] = build_file_key(df_rate_dedup)
        df_rate_dedup = df_rate_dedup[
            df_rate_dedup["_file_key"].notna()
            & (df_rate_dedup["_file_key"].astype(str).str.len() > 0)
        ].copy()
        # Keep the latest entry per (task, file) from the metadata CSVs.
        df_rate_dedup = df_rate_dedup.drop_duplicates(subset=["task", "_file_key"], keep="last")
        if len(tasks) > 1:
            df_rate_matched = filter_to_common_files_across_tasks(
                df_rate_dedup, tasks=tasks, file_key_col="_file_key"
            )
            if not df_rate_matched.empty:
                matched_file_count = int(df_rate_matched["_file_key"].nunique())

    report_path = output_dir / args.output_name

    with PdfPages(report_path) as pdf:
        summary_lines = [
            f"Total files: {len(df)}",
            f"Stations: {', '.join(sorted(df['station'].dropna().unique())) if 'station' in df.columns else 'n/a'}",
            f"Tasks: {', '.join(sorted(df['task'].dropna().unique())) if 'task' in df.columns else 'n/a'}",
            f"Total seconds (sum): {int(df['events_per_second_total_seconds'].sum())}",
        ]
        if matched_file_count and len(tasks) > 1:
            summary_lines.append(
                f"Matched files across tasks ({', '.join(str(t) for t in tasks)}): {matched_file_count}"
            )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(0.02, 0.95, "Events-per-second metadata report", fontsize=14, weight="bold", va="top")
        ax.text(0.02, 0.75, "\n".join(summary_lines), fontsize=11, va="top")
        pdf_save_rasterized_page(pdf, fig)
        plt.close(fig)

        if len(tasks) > 1:
            for task in tasks:
                task_df = df[df["task"] == task].copy()
                if task_df.empty:
                    continue
                task_times, task_rate_matrix, task_total_seconds = build_rate_arrays(task_df, rate_cols)
                task_fallback = float(task_df["events_per_second_total_seconds"].median())
                plot_rate_heatmap(
                    pdf,
                    times=task_times,
                    rate_matrix=task_rate_matrix,
                    total_seconds=task_total_seconds,
                    rate_values=rate_values,
                    title=f"Events per second occurrence rate per file (Task {task})",
                    log_color=not args.linear_color,
                    fallback_seconds=task_fallback,
                    x_limits=x_limits,
                )
        else:
            plot_rate_heatmap(
                pdf,
                times=global_times,
                rate_matrix=global_rate_matrix,
                total_seconds=global_total_seconds,
                rate_values=rate_values,
                title="Events per second occurrence rate per file",
                log_color=not args.linear_color,
                fallback_seconds=fallback_seconds,
                x_limits=x_limits,
            )

        plot_global_histogram(
            pdf,
            rate_values=rate_values,
            total_counts=total_counts,
            title="Global events-per-second histogram (all files)",
        )

        plot_global_rate_trend(
            pdf,
            times=global_times,
            rates=df["events_per_second_global_rate"].to_numpy(dtype=float),
            title="Global rate per file",
            x_limits=x_limits,
        )

        plot_task_rate_trends(
            pdf,
            df=df,
            title="Global event rate per file by task",
            x_limits=x_limits,
        )

        if len(tasks) > 1 and matched_file_count == 0 and "filename_base" in df.columns and "task" in df.columns:
            fig, ax = plt.subplots(figsize=(11, 3))
            ax.axis("off")
            ax.text(
                0.02,
                0.75,
                "Matched-by-filename view",
                fontsize=13,
                weight="bold",
                va="top",
            )
            ax.text(
                0.02,
                0.45,
                "No filenames were found in common across all selected tasks.\n"
                "Try increasing --tail (or reading full files) so each task includes the same filenames,\n"
                "or restrict --tasks to those you want to compare.",
                fontsize=11,
                va="top",
            )
            pdf_save_rasterized_page(pdf, fig)
            plt.close(fig)

        if len(tasks) > 1 and not df_rate_matched.empty:
            plot_task_rate_trends(
                pdf,
                df=df_rate_matched,
                title="Global event rate per file by task (matched by filename_base)",
                x_limits=x_limits,
            )
            plot_matched_file_traces(
                pdf,
                df=df_rate_matched,
                tasks=tasks,
                value_col="events_per_second_global_rate",
                title="Global rate per file across tasks",
                ylabel="Global rate (events / second)",
            )
            plot_matched_file_traces(
                pdf,
                df=df_rate_matched,
                tasks=tasks,
                value_col="events_per_second_total_seconds",
                title="Time span per file across tasks",
                ylabel="Total seconds (start-to-end window)",
            )

    print(f"Saved report to {report_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
