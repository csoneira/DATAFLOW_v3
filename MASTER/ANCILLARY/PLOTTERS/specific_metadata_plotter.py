#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
OUTPUT_FILENAME_BASENAME = "specific_metadata_report"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"
FILENAME_TIMESTAMP_PATTERN = re.compile(r"mi0\d(\d{11})$", re.IGNORECASE)
CLI_DESCRIPTION = "Generate plots for task-specific metadata alongside execution timelines."

METADATA_FILENAME_TEMPLATE = "task_{task_id}_metadata_specific.csv"
DEFAULT_PLOTS_PER_PAGE = 6
HEIGHT_PER_PLOT = 2.6

EXCLUDED_COLUMNS: Tuple[str, ...] = (
    "filename_base",
    "execution_timestamp",
    "file_timestamp",
    "analysis_mode",
)


@dataclass(frozen=True)
class LayoutConfig:
    plots_per_page: int = DEFAULT_PLOTS_PER_PAGE
    height_per_plot: float = HEIGHT_PER_PLOT


@dataclass(frozen=True)
class PlotGroup:
    title: str
    columns: List[str]
    ylim: Optional[Tuple[float, float]] = None


T = TypeVar("T")


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
    if "execution_timestamp" in df.columns:
        df["execution_timestamp"] = pd.to_datetime(
            df["execution_timestamp"], format=TIMESTAMP_FMT, errors="coerce"
        )
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

    df = expand_coeff_columns(df)
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols: List[str] = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().empty:
                continue
            numeric_cols.append(column)
    return numeric_cols


def expand_coeff_columns(df: pd.DataFrame) -> pd.DataFrame:
    coeff_columns = [col for col in df.columns if col.endswith("_coeffs")]
    if not coeff_columns:
        return df

    for column in coeff_columns:
        series = df[column]
        parsed = []
        max_len = 0
        for value in series:
            if pd.isna(value):
                parsed.append(None)
                continue
            if isinstance(value, (list, tuple)):
                coeffs = [float(item) for item in value]
            elif isinstance(value, str):
                text = value.strip()
                try:
                    evaluated = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed.append(None)
                    continue
                if isinstance(evaluated, (list, tuple)):
                    coeffs = [float(item) for item in evaluated]
                else:
                    parsed.append(None)
                    continue
            else:
                parsed.append(None)
                continue

            max_len = max(max_len, len(coeffs))
            parsed.append(coeffs)

        if max_len == 0:
            continue

        for idx in range(max_len):
            new_column = f"{column}_{idx + 1}"
            df[new_column] = [
                coeffs[idx] if coeffs is not None and len(coeffs) > idx else pd.NA
                for coeffs in parsed
            ]

        # ensure numeric dtype for newly created columns
        new_cols = [f"{column}_{idx + 1}" for idx in range(max_len)]
        df[new_cols] = df[new_cols].apply(pd.to_numeric, errors="coerce")

    return df


def chunk_sequence(seq: Sequence[T], size: int) -> List[List[T]]:
    pages: List[List[T]] = []
    for idx in range(0, len(seq), size):
        pages.append(list(seq[idx : idx + size]))
    return pages


def _extract_group(
    remaining: List[str],
    columns_reference: Sequence[str],
    predicate,
) -> List[str]:
    group = [col for col in columns_reference if col in remaining and predicate(col)]
    for col in group:
        remaining.remove(col)
    return group


def create_plot_groups(columns: Sequence[str], task_id: int) -> List[PlotGroup]:
    remaining = list(columns)
    groups: List[PlotGroup] = []

    def add_group(title: str, predicate, ylim: Optional[Tuple[float, float]] = None) -> None:
        group = _extract_group(remaining, columns, predicate)
        if group:
            groups.append(PlotGroup(title=title, columns=group, ylim=ylim))

    coeff_groups: "OrderedDict[int, List[str]]" = OrderedDict()
    coeff_ylims = {
        1: (-0.2, 0.5),
        2: (-0.1, 0.05),
        3: (-0.003, 0.002),
        4: (-0.00001, 0.000025),
        5: (-0.0000001, 0.0000001),
    }

    for column in list(remaining):
        match = re.search(r"_coeffs_(\d+)$", column)
        if match:
            idx = int(match.group(1))
            coeff_groups.setdefault(idx, []).append(column)
            remaining.remove(column)

    for idx, coeff_cols in sorted(coeff_groups.items()):
        coeff_cols.sort()
        groups.append(
            PlotGroup(
                title=f"Columns ending with '_coeffs_{idx}'",
                columns=coeff_cols,
                ylim=coeff_ylims.get(idx),
            )
        )

    # Shared grouping rules
    add_group("Columns starting with 'z_'", lambda c: c.startswith("z_"))
    add_group(
        "Columns starting with 'sigmoid_width_'",
        lambda c: c.startswith("sigmoid_width_"),
    )
    add_group(
        "Columns starting with 'background_slope_'",
        lambda c: c.startswith("background_slope_"),
    )

    if task_id == 2:
        add_group(
            "Columns ending with '_T_sum'",
            lambda c: c.endswith("_T_sum"),
            ylim=(-0.8, 2.3),
        )
        add_group(
            "Columns ending with '_T_dif'/'_T_diff'",
            lambda c: c.endswith("_T_dif") or c.endswith("_T_diff"),
            ylim=(-4.0, 1.1),
        )
        add_group(
            "Columns ending with '_Q_sum'",
            lambda c: c.endswith("_Q_sum"),
            ylim=(80.0, 100.0),
        )
        add_group(
            "Columns ending with '_Q_B'",
            lambda c: c.endswith("_Q_B"),
            ylim=(80.0, 100.0),
        )
        add_group(
            "Columns ending with '_Q_F'",
            lambda c: c.endswith("_Q_F"),
            ylim=(80.0, 100.0),
        )
        add_group(
            "Columns ending with '_crstlk_pedestal'/'_crostlk_pedestal'",
            lambda c: c.endswith("_crstlk_pedestal")
            or c.endswith("_crostlk_pedestal"),
            ylim=(-1.5, 1.5),
        )
        add_group(
            "Columns ending with '_crstlk_limit'/'_crostlk_limit'",
            lambda c: c.endswith("_crstlk_limit") or c.endswith("_crostlk_limit"),
            ylim=(1.0, 1.25),
        )

        entries_original = [
            col for col in list(remaining) if col.endswith("_entries_original")
        ]
        for original_col in entries_original:
            if original_col not in remaining:
                continue
            prefix = original_col[: -len("_entries_original")]
            matching = [
                col
                for col in columns
                if col in remaining
                and (
                    col.startswith(prefix)
                    and (
                        col.endswith("_entries_original")
                        or col.endswith("_entries_final")
                    )
                )
            ]
            if matching:
                for col in matching:
                    remaining.remove(col)
                groups.append(
                    PlotGroup(
                        title=f"Columns for '{prefix}' entries",
                        columns=matching,
                    )
                )

    # Fallback grouping for remaining columns
    grouped: "OrderedDict[str, List[str]]" = OrderedDict()
    for column in remaining:
        parts = column.split("_")
        prefix = "_".join(parts[: min(3, len(parts))])
        grouped.setdefault(prefix, []).append(column)

    for prefix, cols in grouped.items():
        title = f"Columns with prefix '{prefix}'" if prefix else "Remaining columns"
        groups.append(PlotGroup(title=title, columns=cols))
    return groups


def build_layout(
    columns: Sequence[str], plots_per_page: int, task_id: int
) -> List[List[PlotGroup]]:
    plot_groups = create_plot_groups(columns, task_id)
    return chunk_sequence(plot_groups, plots_per_page)


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    return Path(__file__).resolve().parent / filename


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


def plot_task_metadata(
    station: str,
    task_id: int,
    df: pd.DataFrame,
    pdf: PdfPages,
    use_real_date: bool,
    layout_cfg: LayoutConfig,
) -> None:
    numeric_cols = numeric_columns(df)
    layout = build_layout(
        numeric_cols,
        layout_cfg.plots_per_page,
        task_id,
    )

    if not layout:
        fig, ax = plt.subplots(figsize=(11, 3))
        fig.suptitle(f"MINGO0{station} – Task {task_id} Specific Metadata", fontsize=12)
        fig.set_rasterized(True)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No numeric metadata columns available.",
            ha="center",
            va="center",
            fontsize=10,
        )
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        return

    x_column = "file_timestamp" if use_real_date else "execution_timestamp"
    df_plot = df

    if use_real_date:
        file_ts = df.get("file_timestamp")
        if file_ts is None or file_ts.dropna().empty:
            fig, ax = plt.subplots(figsize=(11, 3))
            fig.suptitle(
                f"MINGO0{station} – Task {task_id} Specific Metadata",
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
            pdf.savefig(fig, dpi=150)
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
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")

    for page_index, plot_groups in enumerate(layout, start=1):
        n_rows = len(plot_groups)
        fig_height = max(layout_cfg.height_per_plot * n_rows, 3.0)
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(11, fig_height),
            sharex=True,
            constrained_layout=True,
        )

        if n_rows == 1:
            axes = [axes]  # type: ignore[list-item]

        title_suffix = "File timestamps" if use_real_date else "Execution timestamps"
        fig.suptitle(
            f"MINGO0{station} – Task {task_id} Specific Metadata (Page {page_index}, {title_suffix})",
            fontsize=12,
        )

        for ax, group in zip(axes, plot_groups):
            columns = group.columns
            plotted = False
            legend_labels = []
            for column in columns:
                if column not in df_plot:
                    continue
                subset = df_plot[[x_column, column]].dropna()
                if subset.empty:
                    continue
                subset = subset.sort_values(by=x_column)
                ax.plot(
                    subset[x_column],
                    subset[column],
                    marker="o",
                    markersize=2,
                    linewidth=1.0,
                    alpha=0.8,
                    label=column,
                )
                plotted = True
                legend_labels.append(column)

            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylabel("Value")

            if time_bounds:
                ax.set_xlim(*time_bounds)

            if group.ylim:
                ax.set_ylim(*group.ylim)

            if plotted:
                if len(legend_labels) <= 5:
                    ax.legend(loc="best", fontsize=7)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data for these columns.",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="dimgray",
                )

            ax.set_title(group.title, fontsize=9)

        axes[-1].set_xlabel("File timestamp" if use_real_date else "Execution timestamp")
        axes[-1].xaxis.set_major_formatter(formatter)
        axes[-1].xaxis.set_tick_params(rotation=0)

        fig.set_rasterized(True)

        pdf.savefig(fig)
        plt.close(fig)


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
        "-s",
        "--station",
        action="append",
        metavar="STATION_ID",
        help="Limit the report to a specific station (can be provided multiple times).",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=int,
        action="append",
        metavar="TASK_ID",
        help="Limit the report to a specific task (can be provided multiple times).",
    )
    parser.add_argument(
        "-r",
        "--real-date",
        action="store_true",
        help="Plot values against timestamps extracted from metadata filenames.",
    )
    parser.add_argument(
        "--plots-per-page",
        type=int,
        default=DEFAULT_PLOTS_PER_PAGE,
        help=f"Number of plots per PDF page (default: {DEFAULT_PLOTS_PER_PAGE}).",
    )
    parser.add_argument(
        "--height-per-plot",
        type=float,
        default=HEIGHT_PER_PLOT,
        help=f"Vertical size (in inches) allocated per subplot (default: {HEIGHT_PER_PLOT}).",
    )
    return parser


def usage() -> str:
    return build_parser().format_help()


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.help:
        print(usage(), end="")
        raise SystemExit(0)
    return args


def normalize_station_ids(requested: Optional[List[str]]) -> List[str]:
    if not requested:
        return list(STATIONS)
    normalized = []
    for station in requested:
        station_id = station.strip()
        if station_id in STATIONS:
            normalized.append(station_id)
        elif station_id.startswith("0") and station_id[1:] in STATIONS:
            normalized.append(station_id[1:])
        else:
            raise ValueError(f"Unsupported station identifier: {station}")
    return sorted(set(normalized), key=lambda s: STATIONS.index(s))


def normalize_task_ids(requested: Optional[List[int]]) -> List[int]:
    if not requested:
        return list(TASK_IDS)
    valid_tasks = []
    for task in requested:
        if task in TASK_IDS:
            valid_tasks.append(task)
        else:
            raise ValueError(f"Unsupported task identifier: {task}")
    return sorted(set(valid_tasks))


def main() -> None:
    args = parse_args()
    stations = normalize_station_ids(args.station)
    tasks = normalize_task_ids(args.task)
    layout_cfg = LayoutConfig(
        plots_per_page=max(1, args.plots_per_page),
        height_per_plot=max(1.5, args.height_per_plot),
    )

    output_path = resolve_output_filename(stations, tasks, use_real_date=args.real_date)
    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        for station in stations:
            for task_id in tasks:
                df = load_metadata(station, task_id)
                if df.empty:
                    fig, ax = plt.subplots(figsize=(11, 3))
                    fig.suptitle(
                        f"MINGO0{station} – Task {task_id} Specific Metadata",
                        fontsize=12,
                    )
                    fig.set_rasterized(True)
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "Metadata file not found or empty.",
                        ha="center",
                        va="center",
                        fontsize=10,
                    )
                    pdf.savefig(fig, dpi=150)
                    plt.close(fig)
                    continue

                plot_task_metadata(
                    station=station,
                    task_id=task_id,
                    df=df,
                    pdf=pdf,
                    use_real_date=args.real_date,
                    layout_cfg=layout_cfg,
                )

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
