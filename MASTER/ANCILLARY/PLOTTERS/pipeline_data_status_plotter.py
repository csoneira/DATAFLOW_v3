#!/usr/bin/env python3
"""Plot pipeline processing status timelines for all stations."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, to_rgba
from matplotlib.collections import LineCollection

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
CSV_COLUMNS: Tuple[str, ...] = (
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
STAGE_MAP: Tuple[Tuple[str, int, str], ...] = (
    ("hld_remote_add_date", 1, "Remote HLD"),
    ("hld_local_add_date", 2, "Local HLD"),
    ("dat_add_date", 3, "DAT ready"),
    ("list_ev_add_date", 4, "List Events"),
    ("acc_add_date", 5, "ACC"),
    ("merge_add_date", 6, "Merge"),
)

plt.switch_backend("Agg")
CMAP = plt.get_cmap("viridis")
NORM = Normalize(vmin=1, vmax=len(STAGE_MAP))


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        # ensure expected columns to avoid KeyError downstream
        missing = [col for col in CSV_COLUMNS if col not in reader.fieldnames]
        if missing:
            return []
        return [dict(row) for row in reader]


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


def collect_stage_points(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Tuple[datetime, int]]]:
    points: Dict[str, List[Tuple[datetime, int]]] = {key: [] for key, _, _ in STAGE_MAP}
    for row in rows:
        start = parse_start_date(row.get("start_date", ""))
        if start is None:
            continue
        for field, level, _ in STAGE_MAP:
            if row.get(field, "").strip():
                points[field].append((start, level))
    return points


def plot_station(station: str, output_dir: Path, global_min: datetime | None, global_max: datetime | None) -> Tuple[plt.Figure, Path, datetime | None, datetime | None]:
    station_dir = REPO_ROOT / "STATIONS" / f"MINGO0{station}"
    csv_path = station_dir / f"database_status_{station}.csv"
    rows = read_rows(csv_path)
    stage_points = collect_stage_points(rows)

    per_row_levels: List[Tuple[datetime, List[int]]] = []
    for row in rows:
        start = parse_start_date(row.get("start_date", ""))
        if start is None:
            continue
        levels = [level for field, level, _ in STAGE_MAP if row.get(field, "").strip()]
        if levels:
            per_row_levels.append((start, sorted(set(levels))))

    fig, ax = plt.subplots(figsize=(10, 4))

    for field, level, label in STAGE_MAP:
        entries = stage_points[field]
        next_level = level if level == len(STAGE_MAP) else level + 1
        level_color = CMAP(NORM(next_level))
        shaded_color = to_rgba(level_color, alpha=0.08)
        bottom = 0 if level == 1 else max(level - 0.5, 0)
        ax.axhspan(bottom, level + 0.5, color=shaded_color, zorder=0)
        if not entries:
            continue
        entries.sort(key=lambda item: item[0])
        dates = [entry[0] for entry in entries]
        y_values = [level] * len(entries)
        ax.scatter(
            dates,
            y_values,
            s=30,
            color=level_color,
            alpha=0.9,
            edgecolors="none",
            label=label,
            zorder=2,
        )

    segments: List[List[Tuple[float, float]]] = []
    segment_colors: List[Tuple[float, float, float, float]] = []

    for start, levels in per_row_levels:
        if not levels:
            continue
        levels_unique = sorted(set(levels))
        if 1 in levels_unique:
            levels_unique = [0] + levels_unique
        if len(levels_unique) < 2:
            continue
        x = mdates.date2num(start)
        for lower, upper in zip(levels_unique, levels_unique[1:]):
            segments.append([(x, lower), (x, upper)])
            color_level = upper if upper >= 1 else 1
            segment_colors.append(CMAP(NORM(color_level)))

    if segments:
        lc = LineCollection(segments, colors=segment_colors, linewidths=1.4, zorder=1.8, capstyle='round')
        ax.add_collection(lc)

    station_min = None
    station_max = None
    all_dates: List[datetime] = []
    for entries in stage_points.values():
        all_dates.extend(date for date, _ in entries)
    if all_dates:
        station_min = min(all_dates)
        station_max = max(all_dates)

    if global_min and global_max:
        ax.set_xlim(global_min, global_max)
    elif station_min and station_max:
        ax.set_xlim(station_min, station_max)

    ax.set_title(f"MINGO0{station}")
    ax.set_ylim(0.5, len(STAGE_MAP) + 0.5)
    ax.set_yticks([level for _, level, _ in STAGE_MAP])
    ax.set_yticklabels([label for _, _, label in STAGE_MAP])
    ax.set_xlabel("Start date")
    ax.set_ylabel("Stage")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()

    if any(stage_points[field] for field, _, _ in STAGE_MAP):
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"pipeline_data_status_MINGO0{station}.png"
    return fig, png_path, station_min, station_max


def main() -> None:
    output_dir = REPO_ROOT / "STATIONS"
    pdf_path = output_dir / "pipeline_data_status.pdf"

    figures: List[plt.Figure] = []
    png_paths: List[Path] = []

    station_bounds: List[Tuple[datetime | None, datetime | None]] = []
    for station in STATIONS:
        station_bounds.append((None, None))

    global_min: datetime | None = None
    global_max: datetime | None = None

    for idx, station in enumerate(STATIONS):
        fig, png_path, station_min, station_max = plot_station(station, output_dir, None, None)
        figures.append(fig)
        png_paths.append(png_path)
        station_bounds[idx] = (station_min, station_max)
        if station_min and (global_min is None or station_min < global_min):
            global_min = station_min
        if station_max and (global_max is None or station_max > global_max):
            global_max = station_max

    if global_min and global_max:
        for idx, station in enumerate(STATIONS):
            plt.close(figures[idx])
            fig, png_path, _, _ = plot_station(station, output_dir, global_min, global_max)
            figures[idx] = fig
            png_paths[idx] = png_path

    with PdfPages(pdf_path) as pdf:
        for fig, png_path in zip(figures, png_paths):
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            try:
                img = plt.imread(png_path)
                height, width = img.shape[:2]
                display_fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(display_fig, bbox_inches="tight")
                plt.close(display_fig)
            finally:
                try:
                    png_path.unlink()
                except OSError as exc:
                    print(f"Warning: could not delete {png_path}: {exc}")

    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
