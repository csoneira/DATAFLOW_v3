#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


PLOTTER_ROOT = (
    REPO_ROOT / "MASTER" / "ANCILLARY" / "PLOTTERS" / "ONLINE_FILE_COUNT"
)
CONFIG_PATH = PLOTTER_ROOT / "online_file_count_config.txt"
SNAPSHOT_CSV = PLOTTER_ROOT / "file_count_snapshots.csv"
OUTPUT_DIR = PLOTTER_ROOT / "PLOTS"
OUTPUT_FILENAME = "online_file_count_report.pdf"
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


def prettify_label(relative_path: Path) -> str:
    return " / ".join(
        segment.replace("_", " ").title() for segment in relative_path.parts
    )


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
) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    station_datasets: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    if history.empty:
        return station_datasets

    for station, df_station in history.groupby("station"):
        datasets: List[Tuple[str, pd.DataFrame]] = []
        for entry in entries:
            mask = df_station["relative_path"] == str(entry.relative_path)
            df_entry = df_station.loc[mask].copy()
            if df_entry.empty:
                continue
            df_entry = df_entry.sort_values(TIMESTAMP_COLUMN)
            datasets.append((entry.label, df_entry))
        if datasets:
            station_datasets[station] = datasets
    return station_datasets


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
    datasets: Iterable[Tuple[str, pd.DataFrame]],
    pdf: PdfPages,
    current_time: datetime,
    time_bounds: Optional[Tuple[datetime, datetime]],
) -> None:
    datasets = list(datasets)
    if not datasets:
        return

    num_panels = len(datasets)
    fig_height = max(7.0, num_panels * 2.2)

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(11, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    if num_panels == 1:
        axes = [axes]  # type: ignore[list-item]

    current_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"{station} • Online file counts\nSnapshot history up to {current_str}",
        fontsize=14,
    )

    if time_bounds:
        xmin, xmax = time_bounds
        for ax in axes:
            ax.set_xlim(xmin, xmax)

    for ax, (label, df_entry) in zip(axes, datasets):
        ax.set_title(label)
        ax.set_ylabel("File count")
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

    axes[-1].set_xlabel("Snapshot timestamp")
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    )
    axes[-1].xaxis.set_tick_params(rotation=0)

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
