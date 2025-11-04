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
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
OUTPUT_FILENAME = "execution_metadata_report.pdf"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"
FILENAME_TIMESTAMP_PATTERN = re.compile(r"mi0\d(\d{11})$", re.IGNORECASE)
CLI_DESCRIPTION = "Generate Stage 1 execution metadata plots."


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


def ensure_output_directory(path: Path) -> None:
    """Ensure the directory for the output file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def build_station_pages() -> Dict[str, List[pd.DataFrame]]:
    """Collect metadata DataFrames for each station."""
    station_data: Dict[str, List[pd.DataFrame]] = {}
    for station in STATIONS:
        station_data[station] = [
            load_metadata_csv(station, task_id) for task_id in TASK_IDS
        ]
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
    station_pages: Dict[str, List[pd.DataFrame]], use_real_date: bool
) -> Optional[Tuple[datetime, datetime]]:
    minima: List[datetime] = []
    maxima: List[datetime] = []
    column = "file_timestamp" if use_real_date else "execution_timestamp"

    for dataframes in station_pages.values():
        for df in dataframes:
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


def resolve_output_path(use_real_date: bool) -> Path:
    filename = OUTPUT_FILENAME
    if use_real_date:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_real_time.pdf"
        else:
            filename = f"{filename}_real_time"
    return Path(__file__).resolve().parent / filename


def plot_station(
    station: str,
    dataframes: Iterable[pd.DataFrame],
    pdf: PdfPages,
    use_real_date: bool,
    time_bounds: Optional[Tuple[datetime, datetime]],
    month_markers: Iterable[datetime],
    current_time: datetime,
) -> None:
    """Render a page with five subplots for one station."""
    dataframes = list(dataframes)
    month_markers = list(month_markers)

    median_minutes = []
    for df in dataframes:
        if df.empty:
            continue
        median_value = df["total_execution_time_minutes"].median()
        if pd.notna(median_value):
            median_minutes.append(median_value)
    total_median_minutes = float(sum(median_minutes))

    fig, axes = plt.subplots(
        len(TASK_IDS),
        1,
        figsize=(11, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        (
            f"MINGO0{station} – Stage 1 Execution Metadata "
            f"(Total median minutes/file: {total_median_minutes:.2f})"
        ),
        fontsize=14,
    )

    if len(TASK_IDS) == 1:
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

    for ax, task_id, df in zip(axes, TASK_IDS, dataframes):
        ax.set_title(f"TASK_{task_id}")
        ax.set_ylabel("Exec Time (min)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 1.5)
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

        if df.empty:
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
            ax.set_ylim(0, 1.5)
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
            ax.set_ylim(0, 1.5)
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        x = (
            df_plot["file_timestamp"]
            if use_real_date
            else df_plot["execution_timestamp"]
        )
        runtime_line, = ax.plot(
            x,
            df_plot["total_execution_time_minutes"],
            marker="o",
            markersize=1.5,
            linestyle="-",
            color="tab:blue",
            label="Execution time (min)",
            alpha=0.5,
        )

        ax_second = ax.twinx()
        purity_line, = ax_second.plot(
            x,
            df_plot["data_purity_percentage"],
            marker="x",
            markersize=1.5,
            linestyle="--",
            color="tab:red",
            label="Data purity (%)",
            alpha=0.5,
        )
        ax_second.set_ylabel("Purity (%)")
        ax_second.set_ylim(0, 105)
        ax_second.yaxis.label.set_color("tab:red")
        ax_second.tick_params(axis="y", colors="tab:red")

        handles = [runtime_line, purity_line, now_line]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="upper left")

    axes[-1].set_xlabel("File timestamp" if use_real_date else "Execution timestamp")
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    )
    axes[-1].xaxis.set_tick_params(rotation=0)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    station_pages = build_station_pages()
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

    output_path = resolve_output_path(args.real_date)
    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        for station, dfs in station_pages.items():
            plot_station(
                station,
                dfs,
                pdf,
                use_real_date=args.real_date,
                time_bounds=time_bounds,
                month_markers=month_markers,
                current_time=current_time,
            )

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
