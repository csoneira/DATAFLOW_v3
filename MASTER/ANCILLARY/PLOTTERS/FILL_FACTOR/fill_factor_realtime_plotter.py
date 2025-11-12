#!/usr/bin/env python3
"""Realtime fill-factor coverage plotter.

For each station, compare the remote reprocessing inventory against the list
of basenames that reached the end of the Stage-1 pipeline (STEP_3/TASK_2).
Remote-only basenames are shown in red, while those already present in the
event-data outputs are plotted in green. The x-axis encodes the acquisition
timestamp decoded from each basename (YY DDD HH MM SS).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "STATIONS").exists() and (parent / "MASTER").exists():
            return parent
    # Fallback: assume script lives inside MASTER/...
    return current.parents[4]


REPO_ROOT = detect_repo_root()
PLOTS_DIR = Path(__file__).resolve().parent / "PLOTS"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_FILENAME = PLOTS_DIR / "fill_factor_status.pdf"

STATIONS = ("1", "2", "3", "4")


def basename_to_datetime(basename: str) -> Optional[datetime]:
    """Decode acquisition timestamp embedded in the basename."""
    if len(basename) < 5 or not basename.startswith("mi0"):
        return None

    payload = basename[4:]
    if len(payload) != 11 or not payload.isdigit():
        return None

    try:
        year = 2000 + int(payload[0:2])
        day_of_year = int(payload[2:5])
        hour = int(payload[5:7])
        minute = int(payload[7:9])
        second = int(payload[9:11])
        reference = datetime(year, 1, 1)
        return reference + timedelta(
            days=day_of_year - 1,
            hours=hour,
            minutes=minute,
            seconds=second,
        )
    except ValueError:
        return None


def read_remote_basenames(station: str) -> List[str]:
    """Return the ordered basenames from the remote metadata CSV."""
    csv_path = (
        REPO_ROOT
        / f"STATIONS/MINGO0{station}/STAGE_0/REPROCESSING/STEP_1/METADATA"
        / f"remote_database_{station}.csv"
    )
    if not csv_path.exists():
        return []

    basenames: List[str] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        header_consumed = False
        for line in handle:
            row = line.strip()
            if not row:
                continue
            if not header_consumed:
                header_consumed = True
                continue
            basenames.append(row)
    return basenames


def gather_processed_basenames(station: str) -> Set[str]:
    """Inspect Event Data outputs and extract processed basenames."""
    base_dir = (
        REPO_ROOT
        / f"STATIONS/MINGO0{station}/STAGE_1/EVENT_DATA/STEP_3/TASK_2/OUTPUT_FILES"
    )
    if not base_dir.exists():
        return set()

    processed: Set[str] = set()
    for csv_file in base_dir.glob("**/event_data_*.csv"):
        try:
            with csv_file.open("r", encoding="utf-8") as handle:
                for _ in range(4):
                    line = handle.readline()
                    if not line:
                        break
                    if line.startswith("# source_basenames="):
                        payload = line.split("=", 1)[1].strip()
                        processed.update(
                            name.strip()
                            for name in payload.split(",")
                            if name.strip()
                        )
                        break
        except OSError:
            continue
    return processed


def build_station_events(
    station: str,
) -> Tuple[List[datetime], List[float], List[str], int, int]:
    """Return timestamps, y-values, colors, and aggregate stats."""
    remote = read_remote_basenames(station)
    processed = gather_processed_basenames(station)
    remote_set = set(remote)
    processed_in_remote = len(processed & remote_set)

    xs: List[datetime] = []
    ys: List[float] = []
    colors: List[str] = []

    for basename in remote:
        timestamp = basename_to_datetime(basename)
        if timestamp is None:
            continue
        xs.append(timestamp)
        if basename in processed:
            ys.append(1.0)
            colors.append("#2ca02c")  # green
        else:
            ys.append(0.0)
            colors.append("#d62728")  # red
    return xs, ys, colors, len(remote), processed_in_remote


def plot_fill_factor(
    axes: Sequence[plt.Axes],
) -> Path:
    """Render the 2x2 subplot grid and return the saved path."""
    for idx, station in enumerate(STATIONS):
        ax = axes[idx]
        xs, ys, colors, remote_total, processed_total = build_station_events(station)
        if remote_total > 0:
            pct = 100.0 * processed_total / remote_total
            title = (
                f"Station {station} — {pct:.1f}% processed "
                f"({processed_total}/{remote_total})"
            )
        else:
            title = f"Station {station} — no remote entries"

        if xs:
            ax.scatter(xs, ys, c=colors, s=22, alpha=0.85)
            ax.set_ylim(-0.5, 1.5)
        else:
            message = (
                "No data available\n"
                "(check remote list or event outputs)"
                if remote_total == 0
                else "Remote entries lack timestamps"
            )
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )
            ax.set_ylim(-0.5, 1.5)

        ax.set_title(title)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Remote only", "Processed"])
        ax.grid(True, axis="both", linestyle="--", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    axes[-2].set_xlabel("Acquisition time")
    axes[-1].set_xlabel("Acquisition time")

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="#d62728", markersize=7, label="Remote only"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="#2ca02c", markersize=7, label="Processed"),
    ]
    axes[0].figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    output_path = PLOT_FILENAME
    axes[0].figure.tight_layout(rect=(0, 0, 1, 0.96))
    axes[0].figure.savefig(output_path, dpi=200)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate realtime fill-factor coverage plots."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in addition to saving it.",
    )
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=True)
    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    output_path = plot_fill_factor(axes)
    print(f"Fill-factor plot saved to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
