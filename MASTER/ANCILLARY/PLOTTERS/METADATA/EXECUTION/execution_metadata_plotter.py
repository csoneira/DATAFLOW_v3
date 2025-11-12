#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
OUTPUT_FILENAME = "execution_metadata_report.pdf"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"
FILENAME_TIMESTAMP_PATTERN = re.compile(r"mi0\d(\d{11})$", re.IGNORECASE)
CLI_DESCRIPTION = "Generate Stage 0/1 execution metadata plots."

minutes_upper_limit = 5
point_size = 3
plot_linestyle = "None"

def extract_datetime_from_basename(basename: str) -> Optional[datetime]:
    stem = Path(basename).stem
    match = FILENAME_TIMESTAMP_PATTERN.search(stem)
    if not match:
        return None

    digits = match.group(1)
    try:
        year = 2000 + int(digits[0:2])
        day_of_year = int(digits[2:5])
        hour = int(digits[5:7])
        minute = int(digits[7:9])
        second = int(digits[9:11])
    except ValueError:
        return None

    try:
        base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    except ValueError:
        return None

    return base_date.replace(hour=hour, minute=minute, second=second)


def _load_metadata_csv_from_path(metadata_csv: Path) -> pd.DataFrame:
    """Load and normalize a metadata CSV."""
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    expected_columns = {
        "filename_base",
        "execution_timestamp",
        "data_purity_percentage",
        "total_execution_time_minutes",
    }
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        print(
            f"Warning: missing expected columns {missing_columns} in {metadata_csv}; "
            "skipping metadata entries for this task."
        )
        return pd.DataFrame()

    df = df.copy()
    df["execution_timestamp"] = pd.to_datetime(
        df["execution_timestamp"], format=TIMESTAMP_FMT, errors="coerce"
    )
    df["total_execution_time_minutes"] = pd.to_numeric(
        df["total_execution_time_minutes"], errors="coerce"
    )
    df["data_purity_percentage"] = pd.to_numeric(
        df["data_purity_percentage"], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename), errors="coerce"
    )
    df = df.dropna(
        subset=[
            "execution_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    )
    df = df.sort_values("execution_timestamp")
    return df.reset_index(drop=True)


def load_metadata_csv(station: str, task_id: int) -> pd.DataFrame:
    """Load metadata CSV for a given station/task pair."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
        / f"task_{task_id}_metadata_execution.csv"
    )
    return _load_metadata_csv_from_path(metadata_csv)


def load_step2_metadata_csv(station: str) -> pd.DataFrame:
    """Load step 2 metadata CSV for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_2"
        / "METADATA"
        / "step_2_metadata_execution.csv"
    )
    return _load_metadata_csv_from_path(metadata_csv)


def load_stage0_raw_metadata(station: str) -> pd.DataFrame:
    """Load Stage 0 raw file arrival metadata for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "NEW_FILES"
        / "METADATA"
        / "raw_files_brought.csv"
    )
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    expected_columns = {"filename", "bring_timestamp"}
    missing = expected_columns.difference(df.columns)
    if missing:
        print(
            f"Warning: missing expected columns {missing} in {metadata_csv}; "
            "skipping Stage 0 raw metadata entries."
        )
        return pd.DataFrame()

    df = df.copy()
    df["filename_base"] = df["filename"].astype(str)
    df["execution_timestamp"] = pd.to_datetime(
        df["bring_timestamp"], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["total_execution_time_minutes"] = float("nan")
    df["data_purity_percentage"] = float("nan")
    df = df.dropna(subset=["execution_timestamp"]).sort_values(
        "execution_timestamp"
    )
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def load_stage0_step1_metadata(station: str) -> pd.DataFrame:
    """Load Stage 0 reprocessing step 1 metadata for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "REPROCESSING"
        / "STEP_1"
        / "METADATA"
        / "hld_files_brought.csv"
    )
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    timestamp_col: Optional[str] = None
    for candidate in ("bring_timestamp", "bring_timesamp"):
        if candidate in df.columns:
            timestamp_col = candidate
            break

    expected_columns = {"hld_name"}
    if timestamp_col:
        expected_columns.add(timestamp_col)

    missing = expected_columns.difference(df.columns)
    if missing:
        print(
            f"Warning: missing expected columns {missing} in {metadata_csv}; "
            "skipping Stage 0 step 1 metadata entries."
        )
        return pd.DataFrame()

    if not timestamp_col:
        print(
            "Warning: Stage 0 step 1 metadata missing bring_timestamp/bring_timesamp "
            f"column in {metadata_csv}; skipping entries."
        )
        return pd.DataFrame()

    df = df.copy()
    df["filename_base"] = df["hld_name"].astype(str)
    df["execution_timestamp"] = pd.to_datetime(
        df[timestamp_col], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["total_execution_time_minutes"] = float("nan")
    df["data_purity_percentage"] = float("nan")
    df = df.dropna(subset=["execution_timestamp"]).sort_values(
        "execution_timestamp"
    )
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def load_stage0_step2_metadata(station: str) -> pd.DataFrame:
    """Load Stage 0 reprocessing step metadata for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "REPROCESSING"
        / "STEP_2"
        / "METADATA"
        / "dat_files_unpacked.csv"
    )
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    expected_columns = {
        "dat_name",
        "execution_timestamp",
        "execution_duration_s",
    }
    missing = expected_columns.difference(df.columns)
    if missing:
        print(
            f"Warning: missing expected columns {missing} in {metadata_csv}; "
            "skipping Stage 0 reprocessing metadata entries."
        )
        return pd.DataFrame()

    df = df.copy()
    df["filename_base"] = df["dat_name"].astype(str)
    df["execution_timestamp"] = pd.to_datetime(
        df["execution_timestamp"], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["total_execution_time_minutes"] = pd.to_numeric(
        df["execution_duration_s"], errors="coerce"
    ).div(60)
    df["data_purity_percentage"] = float("nan")
    df = df.dropna(subset=["execution_timestamp"]).sort_values(
        "execution_timestamp"
    )
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def ensure_output_directory(path: Path) -> None:
    """Ensure the directory for the output file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def build_station_pages(
    include_step2: bool,
    include_stage0: bool,
) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    """Collect metadata DataFrames for each station."""
    station_data: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    for station in STATIONS:
        datasets: List[Tuple[str, pd.DataFrame]] = []
        if include_stage0:
            datasets.extend(
                [
                    (
                        "STAGE0_RAW_FILES",
                        load_stage0_raw_metadata(station),
                    ),
                    (
                        "STAGE0_REPROCESS_STEP_1",
                        load_stage0_step1_metadata(station),
                    ),
                    (
                        "STAGE0_REPROCESS_STEP_2",
                        load_stage0_step2_metadata(station),
                    ),
                ]
            )

        datasets.extend(
            (f"TASK_{task_id}", load_metadata_csv(station, task_id))
            for task_id in TASK_IDS
        )
        if include_step2:
            datasets.append(("STEP_2", load_step2_metadata_csv(station)))
        station_data[station] = datasets
    return station_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-r",
        "--real-date",
        action="store_true",
        help="Plot using timestamps extracted from the metadata filenames.",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        action="store_true",
        help="Restrict the x-axis to the last hour ending at the current time.",
    )
    parser.add_argument(
        "--include-step2",
        action="store_true",
        help="Include Stage 1 Step 2 execution metadata in the plots.",
    )
    parser.add_argument(
        "--include-stage0",
        action="store_true",
        help="Prepend Stage 0 metadata (raw files and reprocessing) to each station page.",
    )
    return parser


def usage() -> str:
    """Return the CLI usage/help string."""
    return build_parser().format_help()


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.help:
        print(usage(), end="")
        raise SystemExit(0)
    return args


def compute_time_bounds(
    station_pages: Dict[str, List[Tuple[str, pd.DataFrame]]], use_real_date: bool
) -> Optional[Tuple[datetime, datetime]]:
    minima: List[datetime] = []
    maxima: List[datetime] = []
    column = "file_timestamp" if use_real_date else "execution_timestamp"

    for datasets in station_pages.values():
        for _, df in datasets:
            if df.empty or column not in df:
                continue
            series = df[column].dropna()
            if series.empty:
                continue
            minima.append(series.min())
            maxima.append(series.max())

    if not minima or not maxima:
        return None

    lower = min(minima)
    upper = max(maxima)

    if lower == upper:
        upper = lower + timedelta(minutes=1)

    return (lower, upper)


def compute_month_markers(
    bounds: Optional[Tuple[datetime, datetime]]
) -> List[datetime]:
    if not bounds:
        return []

    start, end = bounds
    if start > end:
        start, end = end, start

    markers: List[datetime] = []
    current = datetime(start.year, start.month, 1)
    if current < start and start.day != 1:
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    while current <= end:
        markers.append(current)
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return markers


def resolve_output_path(use_real_date: bool, zoom: bool) -> Path:
    filename = OUTPUT_FILENAME
    if use_real_date:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_real_time.pdf"
        else:
            filename = f"{filename}_real_time"
    if zoom:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_zoomed.pdf"
        else:
            filename = f"{filename}_zoomed"
    OUTPUT_DIR = Path.home() / "DATAFLOW_v3" / "MASTER" / "ANCILLARY" / "PLOTTERS" / "METADATA" / "EXECUTION" / "PLOTS"
    return OUTPUT_DIR / filename


def plot_station(
    station: str,
    station_datasets: Iterable[Tuple[str, pd.DataFrame]],
    pdf: PdfPages,
    use_real_date: bool,
    time_bounds: Optional[Tuple[datetime, datetime]],
    month_markers: Iterable[datetime],
    current_time: datetime,
) -> None:
    """Render a page with execution metadata subplots for one station."""
    station_datasets = list(station_datasets)
    has_stage0_data = any(label.startswith("STAGE0") for label, _ in station_datasets)
    month_markers = list(month_markers)
    current_time_str_full = current_time.strftime("%Y-%m-%d %H:%M:%S")
    current_time_str_time_only = current_time.strftime("%H:%M:%S")

    median_minutes = []
    for _, df in station_datasets:
        if df.empty:
            continue
        runtime_series = df["total_execution_time_minutes"].dropna()
        if runtime_series.empty:
            continue
        median_value = runtime_series.median()
        if pd.notna(median_value):
            median_minutes.append(median_value)
    total_median_minutes = float(sum(median_minutes))

    num_panels = len(station_datasets)
    fig_height = max(8.5, num_panels * 2.2)
    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(11, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        (
            f"MINGO0{station} – "
            f"{'Stage 0/1' if has_stage0_data else 'Stage 1'} Execution Metadata "
            f"(Total median minutes/file: {total_median_minutes:.2f}) "
            f"Current: {current_time_str_full}"
        ),
        fontsize=14,
    )

    if num_panels == 1:
        axes = [axes]  # type: ignore[list-item]

    xlim: Optional[Tuple[datetime, datetime]] = None
    if time_bounds:
        xmin, xmax = time_bounds
        if xmin == xmax:
            xmax = xmin + timedelta(minutes=1)
        xlim = (xmin, xmax)
        for axis in axes:
            axis.set_xlim(xmin, xmax)

    if xlim:
        xmin, xmax = xlim
        markers_to_use = [m for m in month_markers if xmin <= m <= xmax]
    else:
        markers_to_use = month_markers

    for ax, (label, df) in zip(axes, station_datasets):
        ax.set_title(label)
        ax.set_ylabel("Exec Time (min)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.label.set_color("tab:blue")
        ax.tick_params(axis="y", colors="tab:blue")

        now_line = ax.axvline(
            current_time,
            color="green",
            linestyle="--",
            linewidth=1.0,
            label="Current time",
        )

        for marker in markers_to_use:
            ax.axvline(
                marker,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )

        axis_upper = minutes_upper_limit

        if df.empty:
            ax.set_ylim(0, axis_upper)
            ax.text(
                0.5,
                0.5,
                "No metadata available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
            )
            ax.annotate(
                current_time_str_time_only,
                xy=(current_time, ax.get_ylim()[1]),
                xycoords=("data", "data"),
                xytext=(15, -20),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="center",
                color="green",
                fontsize=10,
            )
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        if use_real_date:
            df_plot = (
                df.dropna(subset=["file_timestamp"])
                .sort_values("file_timestamp")
                .copy()
            )
        else:
            df_plot = df.sort_values("execution_timestamp")

        if df_plot.empty:
            ax.set_ylim(0, axis_upper)
            ax.text(
                0.5,
                0.5,
                "No metadata available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
            )
            ax.annotate(
                current_time_str_time_only,
                xy=(current_time, ax.get_ylim()[1]),
                xycoords=("data", "data"),
                xytext=(15, -20),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="center",
                color="green",
                fontsize=10,
            )
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        runtime_series = df_plot["total_execution_time_minutes"]
        runtime_non_nan = runtime_series.dropna()
        has_runtime_points = not runtime_non_nan.empty

        if has_runtime_points:
            runtime_max = runtime_non_nan.max()
            if pd.notna(runtime_max):
                axis_upper = max(axis_upper, float(runtime_max) * 1.15)

        ax.set_ylim(0, axis_upper)

        x = (
            df_plot["file_timestamp"]
            if use_real_date
            else df_plot["execution_timestamp"]
        )

        placeholder_runtime = False
        if has_runtime_points:
            y_runtime = runtime_series
            runtime_label = "Execution time (min)"
        else:
            placeholder_runtime = True
            midpoint = axis_upper / 2 if axis_upper > 0 else 0.5
            y_runtime = pd.Series(midpoint, index=df_plot.index)
            runtime_label = "Arrival marker (no runtime)"

        runtime_line = None
        if not y_runtime.dropna().empty:
            (runtime_line,) = ax.plot(
                x,
                y_runtime,
                marker="o",
                markersize=point_size,
                linestyle=plot_linestyle,
                color="tab:blue",
                label=runtime_label,
                alpha=0.5,
            )

        if placeholder_runtime:
            ax.text(
                0.01,
                0.92,
                "No execution time data",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="tab:blue",
            )

        purity_line = None
        purity_series = df_plot["data_purity_percentage"]
        if purity_series.notna().any():
            ax_second = ax.twinx()
            (purity_line,) = ax_second.plot(
                x,
                purity_series,
                marker="x",
                markersize=point_size,
                linestyle=plot_linestyle,
                color="tab:red",
                label="Data purity (%)",
                alpha=0.5,
            )
            ax_second.set_ylabel("Purity (%)")
            ax_second.set_ylim(0, 105)
            ax_second.yaxis.label.set_color("tab:red")
            ax_second.tick_params(axis="y", colors="tab:red")

        handles = []
        labels = []
        if runtime_line:
            handles.append(runtime_line)
            labels.append(runtime_line.get_label())
        if purity_line:
            handles.append(purity_line)
            labels.append(purity_line.get_label())
        handles.append(now_line)
        labels.append(now_line.get_label())
        ax.legend(handles, labels, loc="upper left")

        ax.annotate(
            current_time_str_time_only,
            xy=(current_time, ax.get_ylim()[1]),
            xycoords=("data", "data"),
            xytext=(15, -20),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="center",
            color="green",
            fontsize=10,
        )

    axes[-1].set_xlabel("File timestamp" if use_real_date else "Execution timestamp")
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    )
    axes[-1].xaxis.set_tick_params(rotation=0)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    station_pages = build_station_pages(
        include_step2=args.include_step2, include_stage0=args.include_stage0
    )
    current_time = datetime.now()
    if args.zoom:
        time_bounds: Optional[Tuple[datetime, datetime]] = (
            current_time - timedelta(hours=1),
            current_time + timedelta(minutes=5),
        )
    else:
        time_bounds = compute_time_bounds(
            station_pages, use_real_date=args.real_date
        )
    month_markers = compute_month_markers(time_bounds)

    output_path = resolve_output_path(args.real_date, args.zoom)
    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        for station, datasets in station_pages.items():
            plot_station(
                station,
                datasets,
                pdf,
                use_real_date=args.real_date,
                time_bounds=time_bounds,
                month_markers=month_markers,
                current_time=current_time,
            )

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
