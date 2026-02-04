#!/usr/bin/env python3
"""Count pipeline output files per station and create a page-per-station PDF."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable, List, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "MASTER").is_dir()),
    Path.home() / "DATAFLOW_v3",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.common.plot_utils import pdf_save_rasterized_page


STATIONS_ROOT = REPO_ROOT / "STATIONS"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "files_config.txt"
COMPLETE_CONFIG_PATH = Path(__file__).resolve().parent / "files_config_complete.txt"
PLOTS_DIR = Path(__file__).resolve().parent / "PLOTS"


@dataclass(frozen=True)
class PipelineEntry:
    relative_path: Path
    label: str
    order: int


@dataclass
class StationEntryStat:
    station: str
    entry: PipelineEntry
    file_count: int
    exists: bool
    resolved_path: Path


def load_pipeline_entries(config_path: Path) -> List[PipelineEntry]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    entries: List[PipelineEntry] = []
    with config_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            relative_path = Path(line)
            label = prettify_label(relative_path)
            entries.append(PipelineEntry(relative_path=relative_path, label=label, order=len(entries)))

    if not entries:
        raise ValueError(f"No entries were found in {config_path}. Add at least one path.")

    return entries


def prettify_label(relative_path: Path) -> str:
    return " / ".join(segment.replace("_", " ").title() for segment in relative_path.parts)


def station_directories(stations_root: Path) -> List[Path]:
    if not stations_root.exists():
        return []
    dirs = [path for path in stations_root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    dirs.sort(key=lambda path: path.name.lower())
    return dirs


def count_files(base_path: Path) -> int:
    """Count regular files underneath `base_path` without following directory symlinks."""
    if not base_path.exists():
        return 0
    if base_path.is_file():
        return 1

    count = 0
    stack = [base_path]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    if entry.is_file():
                        count += 1
                    continue
                if entry.is_file():
                    count += 1
                elif entry.is_dir():
                    stack.append(entry)
            except (PermissionError, FileNotFoundError):
                continue
    return count


def gather_station_stats(entries: Sequence[PipelineEntry]) -> List[List[StationEntryStat]]:
    stats_per_station: List[List[StationEntryStat]] = []
    for station_dir in station_directories(STATIONS_ROOT):
        station_stats: List[StationEntryStat] = []
        for entry in entries:
            resolved = station_dir / entry.relative_path
            exists = resolved.exists()
            file_count = count_files(resolved) if exists else 0
            station_stats.append(
                StationEntryStat(
                    station=station_dir.name,
                    entry=entry,
                    file_count=file_count,
                    exists=exists,
                    resolved_path=resolved,
                )
            )
        stats_per_station.append(station_stats)
    return stats_per_station


def figure_dimensions(n_items: int) -> tuple[float, float]:
    height = max(5.0, min(18.0, 0.6 * n_items + 2.5))
    width = 11.0
    return width, height


def plot_station_pages(station_stats: Sequence[Sequence[StationEntryStat]], output_path: Path) -> None:
    if not station_stats:
        print("No station statistics to plot.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for stats in station_stats:
            if not stats:
                continue
            fig = build_station_figure(stats, timestamp)
            pdf_save_rasterized_page(pdf, fig, bbox_inches="tight")
            plt.close(fig)


def build_station_figure(stats: Sequence[StationEntryStat], timestamp: str):
    labels = [stat.entry.label for stat in stats]
    counts = [stat.file_count for stat in stats]
    y_positions = np.arange(len(stats))

    width, height = figure_dimensions(len(stats))

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")

    fig, ax = plt.subplots(figsize=(width, height))

    cmap = plt.cm.PuBuGn
    colors = cmap(np.linspace(0.35, 0.9, len(stats)))

    bars = ax.barh(y_positions, counts, color=colors, edgecolor="#1f1f1f", linewidth=0.6)
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
    ax.invert_yaxis()  # show earliest pipeline step at the top

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
        mpatches.Patch(facecolor="#5DADE2", edgecolor="#1f1f1f", label="Directory counted")
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


def print_summary(station_stats: Sequence[Sequence[StationEntryStat]]) -> None:
    if not station_stats:
        print("No station data available.")
        return

    for stats in station_stats:
        if not stats:
            continue
        station_name = stats[0].station
        label_width = max(len(stat.entry.label) for stat in stats)
        print(station_name)
        print("-" * len(station_name))
        for stat in stats:
            missing_note = "" if stat.exists else " (dir missing)"
            print(f"  {stat.entry.label.ljust(label_width)} | {stat.file_count:>10,}{missing_note}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pipeline file count plots per station."
    )
    parser.add_argument(
        "-c",
        "--complete",
        action="store_true",
        help=(
            "Use files_config_complete.txt to inspect every pipeline directory "
            "instead of the default subset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = COMPLETE_CONFIG_PATH if args.complete else DEFAULT_CONFIG_PATH
    output_filename = (
        "station_output_file_counts_complete.pdf"
        if args.complete
        else "station_output_file_counts.pdf"
    )

    try:
        entries = load_pipeline_entries(config_path)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return

    station_stats = gather_station_stats(entries)
    if not station_stats:
        print(f"No station directories were found under {STATIONS_ROOT}.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PLOTS_DIR / output_filename

    plot_station_pages(station_stats, output_path)
    print_summary(station_stats)
    print(f"Saved PDF visualization to: {output_path}")


if __name__ == "__main__":
    main()
