#!/usr/bin/env python3
"""Interactive GUI for exploring pipeline status timelines."""

from __future__ import annotations

import csv
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATIONS: Sequence[str] = ("1", "2", "3", "4")
CSV_HEADERS: Tuple[str, ...] = (
    "basename",
    "start_date",
    "hld_remote_add_date",
    "hld_local_add_date",
    "dat_add_date",
    "list_ev_name",
    "list_ev_add_date",
    "acc_name",
    "acc_add_date",
    "merge_add_date",
)
STAGES: Tuple[Tuple[str, int, str], ...] = (
    ("hld_remote_add_date", 1, "Remote HLD"),
    ("hld_local_add_date", 2, "Local HLD"),
    ("dat_add_date", 3, "DAT ready"),
    ("list_ev_add_date", 4, "List Events"),
    ("acc_add_date", 5, "ACC"),
    ("merge_add_date", 6, "Merge"),
)

CMAP = plt.get_cmap("viridis")
NORM = Normalize(vmin=1, vmax=len(STAGES))


@dataclass
class StagePoints:
    per_stage: Dict[str, List[Tuple[datetime, int]]]
    per_row: List[Tuple[datetime, List[int]]]


def parse_start_date(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d_%H.%M.%S", "%Y-%m-%d_(%j)_%H.%M.%S", "%d-%m-%y_%H.%M.%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def load_rows(csv_path: Path) -> List[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [header for header in CSV_HEADERS if header not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing columns: {', '.join(missing)}. The expected headers are: {', '.join(CSV_HEADERS)}"
            )
        return [dict(row) for row in reader]


def collect_points(rows: Iterable[dict[str, str]]) -> StagePoints:
    per_stage: Dict[str, List[Tuple[datetime, int]]] = {field: [] for field, _, _ in STAGES}
    per_row: List[Tuple[datetime, List[int]]] = []

    for row in rows:
        start = parse_start_date(row.get("start_date", ""))
        if start is None:
            continue
        row_levels: List[int] = []
        for field, level, _ in STAGES:
            if row.get(field, "").strip():
                per_stage[field].append((start, level))
                row_levels.append(level)
        if row_levels:
            per_row.append((start, sorted(set(row_levels))))
    return StagePoints(per_stage, per_row)


def plot_stage(ax: plt.Axes, points: StagePoints, align_range: Tuple[datetime | None, datetime | None] | None = None) -> None:
    ax.clear()

    for field, level, label in STAGES:
        entries = points.per_stage[field]
        next_level = level if level == len(STAGES) else level + 1
        level_color = CMAP(NORM(next_level))
        shaded_color = to_rgba(level_color, alpha=0.08)
        bottom = 0 if level == 1 else max(level - 0.5, 0)
        ax.axhspan(bottom, level + 0.5, color=shaded_color, zorder=0)
        if not entries:
            continue
        entries.sort(key=lambda item: item[0])
        dates = [date for date, _ in entries]
        y_values = [level] * len(entries)
        ax.scatter(
            dates,
            y_values,
            s=30,
            color=CMAP(NORM(level)),
            alpha=0.9,
            edgecolors="none",
            label=label,
            zorder=2,
        )

    segments: List[List[Tuple[float, float]]] = []
    segment_colors: List[Tuple[float, float, float, float]] = []
    for start, levels in points.per_row:
        if not levels:
            continue
        unique_levels = sorted(set(levels))
        if 1 in unique_levels:
            unique_levels = [0] + unique_levels
        if len(unique_levels) < 2:
            continue
        x = mdates.date2num(start)
        for lower, upper in zip(unique_levels, unique_levels[1:]):
            segments.append([(x, lower), (x, upper)])
            color_level = upper if upper >= 1 else 1
            segment_colors.append(CMAP(NORM(color_level)))

    if segments:
        lc = LineCollection(segments, colors=segment_colors, linewidths=1.4, zorder=1.8, capstyle='round')
        ax.add_collection(lc)

    ax.set_ylim(0.5, len(STAGES) + 0.5)
    ax.set_yticks([level for _, level, _ in STAGES])
    ax.set_yticklabels([label for _, _, label in STAGES])
    ax.set_ylabel("Stage")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    if align_range and all(align_range):
        ax.set_xlim(*align_range)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)


class PipelineStatusGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("Pipeline Data Status")

        self.station_var = tk.StringVar(value=DEFAULT_STATIONS[0])
        self.align_var = tk.BooleanVar(value=True)

        side_frame = tk.Frame(master)
        side_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(side_frame, text="Station").pack(anchor=tk.W)
        station_menu = tk.OptionMenu(side_frame, self.station_var, *DEFAULT_STATIONS, command=self.reload_plot)
        station_menu.pack(fill=tk.X, pady=5)

        align_check = tk.Checkbutton(side_frame, text="Align X range across stations", variable=self.align_var, command=self.reload_plot)
        align_check.pack(anchor=tk.W, pady=5)

        refresh_button = tk.Button(side_frame, text="Refresh", command=self.reload_plot)
        refresh_button.pack(fill=tk.X, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        canvas_frame = tk.Frame(master)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.station_cache: Dict[str, StagePoints] = {}
        self.global_range: Tuple[datetime | None, datetime | None] | None = None
        self.compute_global_range()
        self.reload_plot()

    def compute_global_range(self) -> None:
        min_date: datetime | None = None
        max_date: datetime | None = None
        for station in DEFAULT_STATIONS:
            stage_points = self.load_stage_points(station)
            dates = [date for entries in stage_points.per_stage.values() for date, _ in entries]
            if not dates:
                continue
            station_min = min(dates)
            station_max = max(dates)
            if min_date is None or station_min < min_date:
                min_date = station_min
            if max_date is None or station_max > max_date:
                max_date = station_max
        self.global_range = (min_date, max_date) if min_date and max_date else None

    def load_stage_points(self, station: str) -> StagePoints:
        if station in self.station_cache:
            return self.station_cache[station]
        csv_path = REPO_ROOT / "STATIONS" / f"MINGO0{station}" / f"database_status_{station}.csv"
        rows = load_rows(csv_path)
        stage_points = collect_points(rows)
        self.station_cache[station] = stage_points
        return stage_points

    def reload_plot(self, *_: object) -> None:
        station = self.station_var.get()
        stage_points = self.load_stage_points(station)
        align = self.align_var.get()
        plot_stage(self.ax, stage_points, self.global_range if align else None)
        self.ax.set_title(f"MINGO0{station}")
        self.fig.autofmt_xdate()
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    PipelineStatusGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
