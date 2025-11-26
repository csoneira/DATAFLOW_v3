#!/usr/bin/env python3
"""Fill-factor coverage plotter based on processed vs remote basenames."""

from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator


def detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "STATIONS").exists() and (parent / "MASTER").exists():
            return parent
    return current.parents[0]


REPO_ROOT = detect_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SCRIPT_DIR / "PLOTS"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TIME_SERIES_DIR = SCRIPT_DIR / "TIME_SERIES_CSVS"
TIME_SERIES_DIR.mkdir(parents=True, exist_ok=True)

from MASTER.common.plot_utils import pdf_save_rasterized_page

STATIONS: Tuple[str, ...] = ("MINGO01", "MINGO02", "MINGO03", "MINGO04")
PROCESSED_DB_DIR = (
    REPO_ROOT
    / "MASTER"
    / "ANCILLARY"
    / "PIPELINE_OPERATIONS"
    / "UPDATE_EXECUTION_CSVS"
    / "OUTPUT_FILES"
)
SNAPSHOT_COLUMNS = [
    "executed_at",
    "total_remote",
    "total_processed",
    "processed_pct",
]
COLORS = {
    "MINGO01": "#ff0000",
    "MINGO02": "#00aa00",
    "MINGO03": "#0000ff",
    "MINGO04": "#800080",
}
TIMESERIES_PDF_PATH = PLOTS_DIR / "fill_factor_timeseries.pdf"


def history_path(station: str) -> Path:
    return TIME_SERIES_DIR / f"fill_factor_timeseries_{station.lower()}.csv"


def clean_remote_csv(station: str) -> Path:
    station_num = int(station[-2:])
    return (
        REPO_ROOT
        / "STATIONS"
        / station
        / "STAGE_0/REPROCESSING/STEP_0/OUTPUT_FILES"
        / f"clean_remote_database_{station_num}.csv"
    )


def processed_basenames_csv(station: str) -> Path:
    return PROCESSED_DB_DIR / f"{station}_processed_basenames.csv"


def read_basenames_from_csv(csv_path: Path, column: str) -> List[str]:
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            names = []
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    names.append(value)
            return names
    except OSError:
        return []


def read_clean_remote_basenames(station: str) -> List[str]:
    path = clean_remote_csv(station)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines()[1:]]
        return [line for line in lines if line]
    except OSError:
        return []


def collect_counts(station: str) -> Dict[str, float]:
    remote_bases = set(read_clean_remote_basenames(station))
    processed_bases = set(
        read_basenames_from_csv(processed_basenames_csv(station), "basename")
    )
    remote_total = len(remote_bases)
    processed_total = len(remote_bases & processed_bases)
    pct = (processed_total / remote_total * 100.0) if remote_total else 0.0
    return {
        "remote_total": remote_total,
        "processed_total": processed_total,
        "processed_pct": round(pct, 3),
    }


def append_snapshot(station: str, counts: Dict[str, float]) -> None:
    history_file = history_path(station)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    executed_at = datetime.now(timezone.utc).astimezone().isoformat()
    row = {
        "executed_at": executed_at,
        "total_remote": counts["remote_total"],
        "total_processed": counts["processed_total"],
        "processed_pct": counts["processed_pct"],
    }
    write_header = not history_file.exists() or history_file.stat().st_size == 0
    with history_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_history(station: str) -> Tuple[List[datetime], List[float], List[int]]:
    path = history_path(station)
    if not path.exists():
        return [], [], []
    timestamps: List[datetime] = []
    percents: List[float] = []
    remotes: List[int] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                executed = row.get("executed_at")
                pct = row.get("processed_pct")
                total_remote = row.get("total_remote")
                if not executed or pct is None or total_remote is None:
                    continue
                try:
                    timestamps.append(datetime.fromisoformat(executed))
                except ValueError:
                    continue
                percents.append(float(pct))
                remotes.append(int(float(total_remote)))
    except OSError:
        return [], [], []
    return timestamps, percents, remotes


def median_recent_rate(
    values: Sequence[float], timestamps: Sequence[datetime], window: int = 10
) -> float:
    if len(values) < 2 or len(timestamps) < 2:
        return 0.0
    count = min(window, len(values), len(timestamps))
    recent_values = values[-count:]
    recent_timestamps = timestamps[-count:]
    deltas: List[float] = []
    for idx in range(1, count):
        dt_days = (
            recent_timestamps[idx] - recent_timestamps[idx - 1]
        ).total_seconds() / 86400.0
        if dt_days <= 0:
            continue
        delta = (recent_values[idx] - recent_values[idx - 1]) / dt_days
        deltas.append(delta)
    if not deltas:
        return 0.0
    return statistics.median(deltas)


def plot_histories(history_payload: Dict[str, Tuple[List[datetime], List[float], List[int]]]) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    any_plotted = False
    global_max_pct: float | None = None
    now_time = datetime.now(timezone.utc).astimezone()
    for station in STATIONS:
        timestamps, percents, remotes = history_payload.get(station, ([], [], []))
        if not timestamps:
            continue
        color = COLORS.get(station, None)
        label_total = remotes[-1] if remotes else 0
        rate = median_recent_rate(percents, timestamps)
        label = f"{station} (total={label_total}, {rate:.1f} %/day)"
        ax.plot(
            timestamps,
            percents,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            color=color,
            label=label,
        )
        station_max = max(percents)
        if global_max_pct is None or station_max > global_max_pct:
            global_max_pct = station_max
        any_plotted = True

    ax.set_xlabel("Snapshot timestamp")
    ax.set_ylabel("Processed coverage (%)")
    if global_max_pct is None:
        ax.set_ylim(0, 100)
    else:
        margin = max(2.5, global_max_pct * 0.05)
        ax.set_ylim(0, global_max_pct + margin)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    now_label = now_time.strftime("%Y-%m-%d %H:%M")
    now_line = ax.axvline(
        now_time,
        color="green",
        linestyle="--",
        linewidth=1.0,
        label=f"Current time ({now_label})",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()

    if not any_plotted:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")

    fig.tight_layout()
    with PdfPages(TIMESERIES_PDF_PATH) as pdf:
        pdf_save_rasterized_page(pdf, fig)
    plt.close(fig)


def process_station(station: str) -> Dict[str, float]:
    counts = collect_counts(station)
    append_snapshot(station, counts)
    log_message = (
        f"[{station}] remote={counts['remote_total']} | processed={counts['processed_total']}"
        f" | coverage={counts['processed_pct']:.3f}%"
    )
    print(log_message)
    return counts


def main(stations: Sequence[str], skip_plot: bool) -> None:
    history_payload: Dict[str, Tuple[List[datetime], List[float], List[int]]] = {}
    for station in stations:
        process_station(station)
        history_payload[station] = load_history(station)

    if not skip_plot:
        plot_histories(history_payload)
        print(f"Combined PDF refreshed → {TIMESERIES_PDF_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot processed coverage (processed vs remote basenames)."
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Collect counts but skip the matplotlib plot",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        choices=STATIONS,
        help="Subset of stations to process (default: all)",
    )
    args = parser.parse_args()
    targets = tuple(args.stations) if args.stations else STATIONS
    main(targets, skip_plot=args.skip_plot)
