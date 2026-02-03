#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "MASTER").is_dir()),
    Path.home() / "DATAFLOW_v3",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.common.plot_utils import pdf_save_rasterized_page

STATIONS: Tuple[str, ...] = ("0", "1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
OUTPUT_FILENAME_BASENAME = "filter_metadata_report"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"
FILENAME_TIMESTAMP_PATTERN = re.compile(r"mi0\d(\d{11})$", re.IGNORECASE)
METADATA_FILENAME_TEMPLATE = "task_{task_id}_metadata_filter.csv"
DEFAULT_PLOTS_PER_PAGE = 6

EXCLUDED_COLUMNS: Tuple[str, ...] = (
    "filename_base",
    "execution_timestamp",
    "file_timestamp",
)


def extract_datetime_from_basename(basename: str) -> Optional[datetime]:
    stem = Path(basename).stem

    # Direct timestamp format e.g. 2025-10-31_15.40.15
    if len(stem) >= 19:
        try:
            return datetime.strptime(stem, "%Y-%m-%d_%H.%M.%S")
        except ValueError:
            pass

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


def load_metadata(station: str, task_id: int) -> pd.DataFrame:
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
        / METADATA_FILENAME_TEMPLATE.format(task_id=task_id)
    )

    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    if df.empty:
        return df

    df = df.copy()

    if "filename_base" in df.columns:
        df = df[df["filename_base"].str.startswith("mi", na=False)]

    if "execution_timestamp" in df.columns:
        df["execution_timestamp"] = pd.to_datetime(
            df["execution_timestamp"], format=TIMESTAMP_FMT, errors="coerce"
        )
    else:
        df["execution_timestamp"] = pd.NaT

    filename_series = df.get("filename_base")
    if filename_series is not None:
        file_ts = filename_series.map(
            lambda value: extract_datetime_from_basename(value)
            if isinstance(value, str)
            else None
        )
        df["file_timestamp"] = pd.to_datetime(file_ts, errors="coerce")
    else:
        df["file_timestamp"] = pd.NaT

    missing_exec = df["execution_timestamp"].isna()
    if missing_exec.any():
        df.loc[missing_exec, "execution_timestamp"] = df.loc[
            missing_exec, "file_timestamp"
        ]

    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


def metric_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and not series.dropna().empty:
            columns.append(column)
    return columns


def chunk_sequence(seq: Sequence[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [list(seq)]
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def compute_time_bounds(df: pd.DataFrame, column: str) -> Optional[Tuple[datetime, datetime]]:
    if column not in df:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    lower = series.min()
    upper = series.max()
    if lower == upper:
        upper = lower + timedelta(minutes=1)
    return (lower, upper)


def compute_month_markers(bounds: Optional[Tuple[datetime, datetime]]) -> List[datetime]:
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


def resolve_output_filename(
    requested_stations: Sequence[str],
    requested_tasks: Sequence[int],
    use_real_date: bool,
) -> Path:
    station_fragment = (
        f"_stations_{'-'.join(requested_stations)}"
        if 0 < len(requested_stations) < len(STATIONS)
        else ""
    )
    task_fragment = (
        f"_tasks_{'-'.join(str(t) for t in requested_tasks)}"
        if 0 < len(requested_tasks) < len(TASK_IDS)
        else ""
    )
    suffix = "_real_time" if use_real_date else ""
    filename = f"{OUTPUT_FILENAME_BASENAME}{station_fragment}{task_fragment}{suffix}.pdf"
    output_dir = (
        Path.home()
        / "DATAFLOW_v3"
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "FILTER"
        / "PLOTS"
    )
    return output_dir / filename


def _ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_task_metrics(
    station: str,
    task_id: int,
    df: pd.DataFrame,
    pdf: PdfPages,
    use_real_date: bool,
    plots_per_page: int,
) -> None:
    metrics = metric_columns(df)
    title_suffix = "File timestamps" if use_real_date else "Execution timestamps"

    if df.empty or not metrics:
        fig, ax = plt.subplots(figsize=(11, 3))
        fig.suptitle(
            f"MINGO0{station} – Task {task_id} Filter Metadata ({title_suffix})",
            fontsize=12,
        )
        fig.set_rasterized(True)
        ax.axis("off")
        message = (
            "No filter metadata available."
            if df.empty
            else "No numeric filter metrics found."
        )
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
        pdf_save_rasterized_page(pdf, fig, dpi=150)
        plt.close(fig)
        return

    x_column = "file_timestamp" if use_real_date else "execution_timestamp"
    df_plot = df

    if use_real_date:
        file_ts = df.get("file_timestamp")
        if file_ts is None or file_ts.dropna().empty:
            fig, ax = plt.subplots(figsize=(11, 3))
            fig.suptitle(
                f"MINGO0{station} – Task {task_id} Filter Metadata",
                fontsize=12,
            )
            fig.set_rasterized(True)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Filename-derived timestamps not available for real-date plots.",
                ha="center",
                va="center",
                fontsize=10,
            )
            pdf_save_rasterized_page(pdf, fig, dpi=150)
            plt.close(fig)
            return
        df_plot = df.sort_values(by="file_timestamp")
    else:
        series = df.get(x_column)
        if series is None or series.dropna().empty:
            alternate = df.get("file_timestamp")
            if alternate is not None and not alternate.dropna().empty:
                x_column = "file_timestamp"
                df_plot = df.sort_values(by=x_column)
        else:
            df_plot = df.sort_values(by=x_column)

    time_bounds = compute_time_bounds(df_plot, x_column)
    month_markers = compute_month_markers(time_bounds)
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    current_time = datetime.now()

    pages = chunk_sequence(metrics, plots_per_page)
    for page_index, metrics_page in enumerate(pages, start=1):
        n_rows = len(metrics_page)
        fig_height = max(3.0, n_rows * 2.4)
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(11, fig_height),
            sharex=True,
            constrained_layout=True,
        )
        fig.suptitle(
            f"MINGO0{station} – Task {task_id} Filter Metadata "
            f"(Page {page_index}, {title_suffix})",
            fontsize=12,
        )

        if n_rows == 1:
            axes = [axes]  # type: ignore[list-item]

        for ax, metric in zip(axes, metrics_page):
            subset = df_plot[[x_column, metric]].dropna(subset=[x_column, metric])
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_title(metric, fontsize=10)
            ax.set_ylabel("%")

            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="dimgray",
                )
                continue

            subset = subset.sort_values(by=x_column)
            ax.plot(
                subset[x_column],
                subset[metric],
                marker="o",
                markersize=5,
                linewidth=1.0,
                alpha=0.8,
            )
            ax.set_ylim(-0.5, 100.5)

            if time_bounds:
                xmin, xmax = time_bounds
                ax.set_xlim(xmin, xmax)

            for marker in month_markers:
                ax.axvline(marker, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

            if time_bounds and time_bounds[0] <= current_time <= time_bounds[1]:
                ax.axvline(current_time, color="green", linestyle="--", linewidth=0.8)

            ax.xaxis.set_major_formatter(formatter)

        axes[-1].set_xlabel("File timestamp" if use_real_date else "Execution timestamp")
        pdf_save_rasterized_page(pdf, fig, dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate filter-metadata plots for Stage 1 tasks.",
    )
    parser.add_argument(
        "--stations",
        default="",
        help="Comma-separated station IDs (default: all).")
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task IDs (default: all).",
    )
    parser.add_argument(
        "--real-date",
        action="store_true",
        help="Plot using filename-derived timestamps instead of execution timestamps.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both execution-time and real-date PDFs.",
    )
    parser.add_argument(
        "--plots-per-page",
        type=int,
        default=DEFAULT_PLOTS_PER_PAGE,
        help=f"Number of filter metrics per PDF page (default: {DEFAULT_PLOTS_PER_PAGE}).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output PDF path (ignored when --both is set).",
    )
    return parser.parse_args()


def _parse_station_list(raw: str) -> List[str]:
    if not raw:
        return list(STATIONS)
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return tokens if tokens else list(STATIONS)


def _parse_task_list(raw: str) -> List[int]:
    if not raw:
        return list(TASK_IDS)
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values if values else list(TASK_IDS)


def _derive_real_time_output(base: Path) -> Path:
    if base.suffix.lower() == ".pdf":
        return base.with_name(f"{base.stem}_real_time{base.suffix}")
    return base.with_name(f"{base.name}_real_time")


def main() -> None:
    args = parse_args()

    stations = _parse_station_list(args.stations)
    tasks = _parse_task_list(args.tasks)
    plots_per_page = max(1, int(args.plots_per_page))

    def run_report(use_real_date: bool, output_path: Optional[Path]) -> None:
        resolved_output = output_path or resolve_output_filename(stations, tasks, use_real_date)
        _ensure_output_directory(resolved_output)
        with PdfPages(resolved_output) as pdf:
            for station in stations:
                for task_id in tasks:
                    df = load_metadata(station, task_id)
                    plot_task_metrics(
                        station=station,
                        task_id=task_id,
                        df=df,
                        pdf=pdf,
                        use_real_date=use_real_date,
                        plots_per_page=plots_per_page,
                    )
        print(f"Saved filter metadata report: {resolved_output}")

    if args.both:
        run_report(False, None)
        run_report(True, None)
        return

    if args.output:
        run_report(args.real_date, Path(args.output))
    else:
        run_report(args.real_date, None)


if __name__ == "__main__":
    main()
