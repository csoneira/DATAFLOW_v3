#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import math
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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


PLOTTER_ROOT = (
    REPO_ROOT / "MASTER" / "ANCILLARY" / "PLOTTERS" / "ONLINE_FILE_COUNT"
)
CONFIG_PATH = PLOTTER_ROOT / "CONFIGS" / "online_file_count_config.txt"
SNAPSHOT_CSV = PLOTTER_ROOT / "file_count_snapshots.csv"
OUTPUT_DIR = PLOTTER_ROOT / "PLOTS"
OUTPUT_FILENAME = "online_file_count_report.pdf"
STATIC_OUTPUT_FILENAME = "online_file_count_snapshot.pdf"
STATIONS_ROOT = REPO_ROOT / "STATIONS"
TIMESTAMP_COLUMN = "snapshot_timestamp"
SNAPSHOT_COLUMNS = [
    TIMESTAMP_COLUMN,
    "station",
    "relative_path",
    "label",
    "file_count",
    "path_exists",
]


@dataclass(frozen=True)
class DirectoryEntry:
    relative_path: Path
    label: str


@dataclass(frozen=True)
class StationEntryStat:
    station: str
    label: str
    relative_path: Path
    file_count: int
    exists: bool
    resolved_path: Path


def prettify_label(relative_path: Path) -> str:
    return " / ".join(
        segment.replace("_", " ").title() for segment in relative_path.parts
    )


def facecolor_for_path(relative_path: Path) -> Tuple[float, float, float, float]:
    """Background color with transparency based on the directory name."""
    name = str(relative_path).upper()
    if "UNPROCESSED" in name:
        return (1.0, 0.0, 0.0, 0.2)  # translucent red
    if "COMPLETED" in name:
        return (0.0, 0.3, 1.0, 0.2)  # translucent blue
    return (0.6, 0.0, 0.6, 0.2)  # translucent purple


def classify_relative_path(relative_path: Path) -> str:
    name = relative_path.name.upper()
    if "UNPROCESSED" in name:
        return "unprocessed"
    if "COMPLETED" in name:
        return "completed"
    return "other"


def load_directory_entries(config_path: Path) -> List[DirectoryEntry]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    entries: List[DirectoryEntry] = []
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            relative = Path(line)
            entries.append(
                DirectoryEntry(relative_path=relative, label=prettify_label(relative))
            )

    if not entries:
        raise ValueError(
            f"No directory entries were found in {config_path}; add at least one path."
        )
    return entries


def station_directories() -> List[Path]:
    if not STATIONS_ROOT.exists():
        return []
    stations: List[Path] = []
    for entry in STATIONS_ROOT.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.upper().startswith("MINGO"):
            continue
        stations.append(entry)
    stations.sort(key=lambda path: path.name)
    return stations


def count_files(root: Path) -> int:
    """Count every file underneath `root`, including nested year/month/day folders."""
    if not root.exists():
        return 0
    if root.is_file():
        return 1

    total = 0

    def _suppress(_error: OSError) -> None:
        return None

    for _dirpath, _dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=_suppress
    ):
        total += len(filenames)
    return total


def gather_snapshot_stats(entries: Sequence[DirectoryEntry]) -> List[List[StationEntryStat]]:
    stats_per_station: List[List[StationEntryStat]] = []
    for station_dir in station_directories():
        station_stats: List[StationEntryStat] = []
        for entry in entries:
            resolved = station_dir / entry.relative_path
            exists = resolved.exists()
            file_count = count_files(resolved) if exists else 0
            station_stats.append(
                StationEntryStat(
                    station=station_dir.name,
                    label=entry.label,
                    relative_path=entry.relative_path,
                    file_count=file_count,
                    exists=exists,
                    resolved_path=resolved,
                )
            )
        if station_stats:
            stats_per_station.append(station_stats)
    return stats_per_station


def snapshot_figure_dimensions(n_items: int) -> Tuple[float, float]:
    height = max(5.0, min(18.0, 0.6 * n_items + 2.5))
    width = 11.0
    return (width, height)


def build_snapshot_figure(stats: Sequence[StationEntryStat], timestamp: str):
    labels = [stat.label for stat in stats]
    counts = [stat.file_count for stat in stats]
    y_positions = np.arange(len(stats))

    width, height = snapshot_figure_dimensions(len(stats))

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")

    fig, ax = plt.subplots(figsize=(width, height))

    cmap = plt.cm.PuBuGn  # type: ignore[attr-defined]
    colors = cmap(np.linspace(0.35, 0.9, len(stats)))

    bars = ax.barh(
        y_positions,
        counts,
        color=colors,
        edgecolor="#1f1f1f",
        linewidth=0.6,
    )
    for bar, stat in zip(bars, stats):
        if not stat.exists:
            bar.set_hatch("//")
            bar.set_facecolor("#FFFFFF")
            bar.set_edgecolor("#7a7a7a")

    ax.set_yticks(y_positions, labels=labels)
    ax.set_xlabel("File count")
    ax.set_ylabel("Pipeline segment")
    ax.set_title(
        f"{stats[0].station} • Output directories\n"
        f"Snapshot {timestamp} • Source: {STATIONS_ROOT}"
    )
    ax.invert_yaxis()

    x_max = max(counts) if counts else 0
    label_padding = max(1, int(math.log10(x_max + 1))) if x_max else 1
    for bar, count in zip(bars, counts):
        x_offset = max(1, x_max * 0.01)
        ax.text(
            count + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=9,
            color="#111111",
        )

    ax.set_xlim(0, max(5, x_max * 1.15 + label_padding))

    legend_handles: List[mpatches.Patch] = [
        mpatches.Patch(
            facecolor="#5DADE2",
            edgecolor="#1f1f1f",
            label="Directory counted",
        )
    ]
    if any(not stat.exists for stat in stats):
        legend_handles.append(
            mpatches.Patch(
                facecolor="#FFFFFF",
                edgecolor="#7a7a7a",
                hatch="//",
                label="Directory missing",
            )
        )
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    return fig


def plot_snapshot_pages(
    station_stats: Sequence[Sequence[StationEntryStat]], output_path: Path, timestamp: str
) -> None:
    if not station_stats:
        print("No station statistics to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for stats in station_stats:
            if not stats:
                continue
            fig = build_snapshot_figure(stats, timestamp)
            pdf_save_rasterized_page(pdf, fig, bbox_inches="tight")
            plt.close(fig)


def print_snapshot_summary(
    station_stats: Sequence[Sequence[StationEntryStat]]
) -> None:
    if not station_stats:
        print("No station data available.")
        return

    for stats in station_stats:
        if not stats:
            continue
        station_name = stats[0].station
        label_width = max(len(stat.label) for stat in stats)
        print(station_name)
        print("-" * len(station_name))
        for stat in stats:
            missing_note = "" if stat.exists else " (dir missing)"
            print(f"  {stat.label.ljust(label_width)} | {stat.file_count:>10,}{missing_note}")
        print()


def run_static_snapshot(entries: Sequence[DirectoryEntry]) -> None:
    station_stats = gather_snapshot_stats(entries)
    if not station_stats:
        print(f"No station directories were found under {STATIONS_ROOT}.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output_path = OUTPUT_DIR / STATIC_OUTPUT_FILENAME
    plot_snapshot_pages(station_stats, output_path, timestamp)
    print_snapshot_summary(station_stats)
    print(f"Saved static snapshot to: {output_path}")


def capture_snapshot(entries: Sequence[DirectoryEntry]) -> List[dict]:
    timestamp = datetime.now().replace(microsecond=0).isoformat()
    rows: List[dict] = []
    stations = station_directories()
    for station_dir in stations:
        for entry in entries:
            resolved = station_dir / entry.relative_path
            exists = resolved.exists()
            file_count = count_files(resolved) if exists else 0
            rows.append(
                {
                    TIMESTAMP_COLUMN: timestamp,
                    "station": station_dir.name,
                    "relative_path": str(entry.relative_path),
                    "label": entry.label,
                    "file_count": int(file_count),
                    "path_exists": int(exists),
                }
            )
    return rows


def append_snapshot(rows: Sequence[dict]) -> None:
    if not rows:
        return
    SNAPSHOT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    header = not SNAPSHOT_CSV.exists()
    df.to_csv(SNAPSHOT_CSV, mode="a", header=header, index=False)


def load_snapshot_history() -> pd.DataFrame:
    if not SNAPSHOT_CSV.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.read_csv(SNAPSHOT_CSV)
    if TIMESTAMP_COLUMN in df:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(
            df[TIMESTAMP_COLUMN], errors="coerce"
        )
        df = df.dropna(subset=[TIMESTAMP_COLUMN])
    df["file_count"] = pd.to_numeric(df["file_count"], errors="coerce").fillna(0)
    return df


def build_station_datasets(
    history: pd.DataFrame, entries: Sequence[DirectoryEntry]
) -> Dict[str, List[Tuple[str, pd.DataFrame, Path]]]:
    station_datasets: Dict[str, List[Tuple[str, pd.DataFrame, Path]]] = {}
    if history.empty:
        return station_datasets

    for station, df_station in history.groupby("station"):
        datasets: List[Tuple[str, pd.DataFrame, Path]] = []
        for entry in entries:
            mask = df_station["relative_path"] == str(entry.relative_path)
            df_entry = df_station.loc[mask].copy()
            if df_entry.empty:
                continue
            df_entry = df_entry.sort_values(TIMESTAMP_COLUMN)
            datasets.append((entry.label, df_entry, entry.relative_path))
        if datasets:
            station_datasets[station] = datasets
    return station_datasets


def build_subplot_rows(
    datasets: Sequence[Tuple[str, pd.DataFrame, Path]]
) -> List[Tuple[Optional[Tuple[str, pd.DataFrame, Path]], Optional[Tuple[str, pd.DataFrame, Path]]]]:
    """Arrange datasets into left/right column pairs."""
    rows: List[
        Tuple[Optional[Tuple[str, pd.DataFrame, Path]], Optional[Tuple[str, pd.DataFrame, Path]]]
    ] = []
    left_rows: Dict[Path, int] = {}

    for entry in datasets:
        label, df_entry, relative_path = entry
        category = classify_relative_path(relative_path)
        parent = relative_path.parent

        if category == "completed":
            row_idx = left_rows.get(parent)
            if row_idx is not None and row_idx < len(rows):
                left_entry, right_entry = rows[row_idx]
                if right_entry is None:
                    rows[row_idx] = (left_entry, entry)
                    continue
            # No matching UNPROCESSED previously seen; show completed on right-only row
            rows.append((None, entry))
            continue

        # For unprocessed/other entries, start a new row in order
        rows.append((entry, None))
        if category == "unprocessed":
            left_rows[parent] = len(rows) - 1

    return rows


def compute_time_bounds(df: pd.DataFrame) -> Optional[Tuple[datetime, datetime]]:
    if df.empty or TIMESTAMP_COLUMN not in df:
        return None
    times = df[TIMESTAMP_COLUMN].dropna()
    if times.empty:
        return None
    lower = times.min()
    upper = times.max()
    if lower == upper:
        upper = lower + timedelta(minutes=1)
    return (lower, upper)


def resolve_output_path(zoom: bool) -> Path:
    filename = OUTPUT_FILENAME
    if zoom:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_zoomed.pdf"
        else:
            filename = f"{filename}_zoomed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename


def plot_station(
    station: str,
    datasets: Iterable[Tuple[str, pd.DataFrame, Path]],
    pdf: PdfPages,
    current_time: datetime,
    time_bounds: Optional[Tuple[datetime, datetime]],
) -> None:
    datasets = list(datasets)
    if not datasets:
        return

    subplot_rows = build_subplot_rows(datasets)
    if not subplot_rows:
        return

    nrows = len(subplot_rows)
    fig_height = max(7.0, nrows * 2.4)

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")

    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(12, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    current_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"{station} • Online file counts\nSnapshot history up to {current_str}",
        fontsize=14,
    )

    if time_bounds:
        xmin, xmax = time_bounds
        for ax in axes.ravel():
            ax.set_xlim(xmin, xmax)

    def plot_entry(ax, entry: Tuple[str, pd.DataFrame, Path] | None) -> None:
        if entry is None:
            ax.axis("off")
            return
        label, df_entry, relative_path = entry
        ax.set_title(label)
        ax.set_ylabel("File count")
        ax.set_facecolor(facecolor_for_path(relative_path))
        (count_line,) = ax.plot(
            df_entry[TIMESTAMP_COLUMN],
            df_entry["file_count"],
            marker="o",
            markersize=3,
            linestyle="-",
            color="tab:blue",
            alpha=0.7,
            label="Files",
        )
        now_line = ax.axvline(
            current_time,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            label="Now",
        )
        ax.legend(handles=[count_line, now_line], loc="upper left")
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.set_ylim(bottom=0)

    for axis_row, (left_entry, right_entry) in zip(axes, subplot_rows):
        plot_entry(axis_row[0], left_entry)
        plot_entry(axis_row[1], right_entry)

    # Expand the right-side time limit slightly so the "Now" marker is fully visible.
    all_axes = [ax for ax in axes.ravel() if ax.has_data()]
    if all_axes:
        xmin, xmax = all_axes[0].get_xlim()
        span = xmax - xmin
        if span <= 0:
            span = 1 / 86400  # fallback
        for ax in all_axes:
            ax.set_xlim(xmin, xmax + span * 0.01)

    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    for ax in axes[-1]:
        if not ax.has_data():
            continue
        ax.set_xlabel("Snapshot timestamp")
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=0)

    pdf_save_rasterized_page(pdf, fig, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and plot online file counts per station directory.",
        add_help=False,
    )
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit."
    )
    parser.add_argument(
        "-z",
        "--zoom",
        action="store_true",
        help="Restrict plots to the last hour of snapshots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Path to a config file listing relative directories (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "-s",
        "--static",
        action="store_true",
        help=(
            "Only capture the current snapshot and plot per-station bar charts "
            "similar to file_tracker_plotter.py (does not update historical CSV)."
        ),
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


def main() -> None:
    args = parse_args()
    entries = load_directory_entries(args.config)

    if args.static:
        run_static_snapshot(entries)
        return

    snapshot_rows = capture_snapshot(entries)
    append_snapshot(snapshot_rows)

    history = load_snapshot_history()
    if history.empty:
        print("No snapshot history available; nothing to plot.")
        return

    current_time = datetime.now()
    if args.zoom:
        lower = current_time - timedelta(hours=1)
        history = history[history[TIMESTAMP_COLUMN] >= lower]
        if history.empty:
            print("No snapshot history available in the last hour; nothing to plot.")
            return
        time_bounds: Optional[Tuple[datetime, datetime]] = (lower, current_time)
    else:
        time_bounds = compute_time_bounds(history)

    station_data = build_station_datasets(history, entries)
    if not station_data:
        print("No station datasets available for plotting.")
        return

    output_path = resolve_output_path(args.zoom)
    with PdfPages(output_path) as pdf:
        for station, datasets in sorted(station_data.items()):
            plot_station(station, datasets, pdf, current_time, time_bounds)

    print(f"Snapshot CSV updated: {SNAPSHOT_CSV}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
