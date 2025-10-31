#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
OUTPUT_FILENAME = "execution_metadata_report.pdf"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"


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
        / f"step_{task_id}_metadata_execution.csv"
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
        raise ValueError(
            f"Missing expected columns {missing_columns} in {metadata_csv}"
        )

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


def plot_station(
    station: str, dataframes: Iterable[pd.DataFrame], pdf: PdfPages
) -> None:
    """Render a page with five subplots for one station."""
    dataframes = list(dataframes)

    median_minutes = []
    for df in dataframes:
        if df.empty:
            continue
        median_value = df["total_execution_time_minutes"].median()
        if pd.notna(median_value):
            median_minutes.append(median_value)
    total_median_minutes = float(sum(median_minutes))

    current_time = datetime.now()
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

    for ax, task_id, df in zip(axes, TASK_IDS, dataframes):
        ax.set_title(f"TASK_{task_id}")
        ax.set_ylabel("Exec Time (min)")
        ax.grid(True, axis="y", alpha=0.3)

        now_line = ax.axvline(
            current_time,
            color="green",
            linestyle="--",
            linewidth=1.0,
            label="Current time",
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
            ax.set_ylim(0, 1)
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        x = df["execution_timestamp"]
        runtime_line, = ax.plot(
            x,
            df["total_execution_time_minutes"],
            marker="o",
            markersize=4,
            linestyle="-",
            color="tab:blue",
            label="Execution time (min)",
        )

        ax_second = ax.twinx()
        purity_line, = ax_second.plot(
            x,
            df["data_purity_percentage"],
            marker="x",
            markersize=4,
            linestyle="--",
            color="tab:red",
            label="Data purity (%)",
        )
        ax_second.set_ylabel("Purity (%)")

        handles = [runtime_line, purity_line, now_line]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="upper left")

    axes[-1].set_xlabel("Execution timestamp")
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    )
    axes[-1].xaxis.set_tick_params(rotation=0)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    station_pages = build_station_pages()

    output_path = Path(__file__).resolve().parent / OUTPUT_FILENAME
    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        for station, dfs in station_pages.items():
            plot_station(station, dfs, pdf)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
