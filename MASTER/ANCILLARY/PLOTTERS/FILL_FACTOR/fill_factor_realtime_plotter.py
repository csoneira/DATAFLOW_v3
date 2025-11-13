#!/usr/bin/env python3
"""Fill-factor tracker and time-series plotter for all MINGO stations."""

from __future__ import annotations

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Set, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator


def detect_repo_root() -> Path:
    """Return the DATAFLOW repository root."""
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

from MASTER.common.plot_utils import pdf_save_rasterized_page

STATIONS = ("MINGO01", "MINGO02", "MINGO03", "MINGO04")
LEGACY_STATION = "MINGO01"

TASK_IDS = range(1, 6)
TASK_KEYS = [f"task_{idx}" for idx in TASK_IDS]
STEP_KEYS = [*TASK_KEYS, "step_2"]
REMOTE_COLUMN = "basename"
FILENAME_COLUMN = "filename_base"

MATRIX_COLUMNS = ["basename", "remote", *TASK_KEYS, "step_2"]
SNAPSHOT_COLUMNS = ["executed_at", "remote"]
for key in STEP_KEYS:
    SNAPSHOT_COLUMNS.extend([key, f"{key}_pct"])

LEGACY_PRESENCE_PATH = SCRIPT_DIR / "fill_factor_presence_matrix.csv"
LEGACY_HISTORY_PATH = SCRIPT_DIR / "fill_factor_timeseries.csv"
TIMESERIES_PDF_PATH = PLOTS_DIR / "fill_factor_timeseries.pdf"


def station_presence_path(station_name: str) -> Path:
    return SCRIPT_DIR / f"fill_factor_presence_matrix_{station_name.lower()}.csv"


def station_history_path(station_name: str) -> Path:
    return SCRIPT_DIR / f"fill_factor_timeseries_{station_name.lower()}.csv"


def read_unique_list(csv_path: Path, column_hint: str) -> List[str]:
    """Return ordered unique basenames from ``csv_path``."""
    if not csv_path.exists():
        return []

    ordered: List[str] = []
    seen: Set[str] = set()

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            column = column_hint if column_hint in fieldnames else (
                fieldnames[0] if fieldnames else column_hint
            )
            for row in reader:
                raw_value = (row.get(column) or "").strip()
                if not raw_value or raw_value in seen:
                    continue
                seen.add(raw_value)
                ordered.append(raw_value)
    except OSError:
        return []

    return ordered


def build_presence_rows(
    remote_order: Sequence[str],
    membership: Dict[str, Set[str]],
) -> List[List[Union[int, str]]]:
    """Create a matrix of basename coverage."""
    seen: Set[str] = set()
    ordered_basenames: List[str] = []

    for basename in remote_order:
        if basename in seen:
            continue
        ordered_basenames.append(basename)
        seen.add(basename)

    extra_names: Set[str] = set()
    for name_set in membership.values():
        extra_names.update(name_set)
    for extras in sorted(extra_names - seen):
        ordered_basenames.append(extras)

    rows: List[List[Union[int, str]]] = []
    for basename in ordered_basenames:
        row: List[Union[int, str]] = [basename]
        for column in MATRIX_COLUMNS[1:]:
            row.append(1 if basename in membership.get(column, set()) else 0)
        rows.append(row)
    return rows


def write_presence_matrix(
    rows: Sequence[Sequence[Union[int, str]]],
    output_path: Path,
) -> None:
    """Persist the presence matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MATRIX_COLUMNS)
        writer.writerows(rows)


def compute_totals(membership: Dict[str, Set[str]]) -> Dict[str, int]:
    """Summaries for each column."""
    return {column: len(membership.get(column, set())) for column in MATRIX_COLUMNS[1:]}


def append_snapshot(
    totals: Dict[str, int],
    history_path: Path,
    *,
    executed_at: str | None = None,
) -> Dict[str, float]:
    """Append an aggregate snapshot with execution timestamp."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    executed_at = executed_at or datetime.now(timezone.utc).astimezone().isoformat()
    remote_total = totals.get("remote", 0)
    entry: Dict[str, float | int | str] = {
        "executed_at": executed_at,
        "remote": remote_total,
    }
    for key in STEP_KEYS:
        count = totals.get(key, 0)
        pct = (count / remote_total * 100.0) if remote_total else 0.0
        entry[key] = count
        entry[f"{key}_pct"] = round(pct, 3)

    history_exists = history_path.exists() and history_path.stat().st_size > 0
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLUMNS)
        if not history_exists:
            writer.writeheader()
        writer.writerow({column: entry.get(column, "") for column in SNAPSHOT_COLUMNS})
    return entry


def parse_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def parse_int(value: str, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def median_recent_rate(
    values: Sequence[float],
    timestamps: Sequence[datetime],
    window: int = 10,
) -> float:
    """Median percent-per-day change using the last ``window`` points."""
    if len(values) < 2 or len(timestamps) < 2 or window < 2:
        return 0.0

    count = min(window, len(values), len(timestamps))
    recent_values = values[-count:]
    recent_timestamps = timestamps[-count:]

    rates: List[float] = []
    for idx in range(1, count):
        delta_time = (
            recent_timestamps[idx] - recent_timestamps[idx - 1]
        ).total_seconds() / 86400.0
        if delta_time <= 0:
            continue
        delta_value = recent_values[idx] - recent_values[idx - 1]
        rates.append(delta_value / delta_time)

    if not rates:
        return 0.0
    return statistics.median(rates)


def ensure_history_schema(history_path: Path) -> None:
    """Upgrade legacy history CSV files to the latest schema if needed."""
    if not history_path.exists():
        return

    try:
        with history_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except OSError:
        return

    if not fieldnames:
        return
    if fieldnames == SNAPSHOT_COLUMNS:
        return

    converted_rows: List[Dict[str, Union[str, int, float]]] = []
    for row in rows:
        executed_at = row.get("executed_at", "")
        remote_total = parse_int(row.get("remote", "0"))
        entry: Dict[str, Union[str, int, float]] = {
            "executed_at": executed_at,
            "remote": remote_total,
        }
        for key in STEP_KEYS:
            count = parse_int(row.get(key, "0"))
            pct = (count / remote_total * 100.0) if remote_total else 0.0
            entry[key] = count
            entry[f"{key}_pct"] = round(pct, 3)
        converted_rows.append(entry)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLUMNS)
        writer.writeheader()
        for entry in converted_rows:
            writer.writerow({column: entry.get(column, "") for column in SNAPSHOT_COLUMNS})


def plot_history(history_path: Path, station_name: str) -> Figure | None:
    """Return a Matplotlib figure for the historical fill factor, if possible."""
    if not history_path.exists():
        return None

    timestamps: List[datetime] = []
    remote_counts: List[int] = []
    pct_series: Dict[str, List[float]] = {key: [] for key in STEP_KEYS}

    with history_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            executed_at = row.get("executed_at")
            if not executed_at:
                continue
            try:
                timestamps.append(datetime.fromisoformat(executed_at))
            except ValueError:
                continue
            remote_counts.append(parse_int(row.get("remote", "0")))
            for key in STEP_KEYS:
                pct_series[key].append(
                    parse_float(row.get(f"{key}_pct", "0.0"))
                )

    if not timestamps:
        return None

    plt.style.use("default")
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    palette = (
        color_cycle.by_key().get("color", [])
        if color_cycle and hasattr(color_cycle, "by_key")
        else []
    )
    remote_color = palette[0] if palette else "#1f77b4"

    max_pct_value = max(
        (max(values) if values else 0.0) for values in pct_series.values()
    )
    if max_pct_value <= 0:
        max_pct_value = 1.0
    num_axes = 1 + len(STEP_KEYS)
    figure_height = max(12.0, num_axes * 2.4)
    fig, axes = plt.subplots(num_axes, 1, figsize=(14, figure_height), sharex=True)
    if hasattr(axes, "ravel"):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    remote_denominator = max(remote_counts) if remote_counts else 0
    remote_pct_series = (
        [count / remote_denominator * 100.0 for count in remote_counts]
        if remote_denominator
        else [0.0 for _ in remote_counts]
    )
    remote_rate = median_recent_rate(remote_pct_series, timestamps)
    axes[0].plot(
        timestamps,
        remote_counts,
        label="remote",
        color=remote_color,
        linewidth=1.6,
        marker="o",
        markersize=6,
        linestyle="--",
    )
    axes[0].set_title(
        f"{station_name} fill-factor coverage history • median Δ(<=10)={remote_rate:.1f}%/day"
    )
    axes[0].set_ylabel("remote\ncount")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, axis="both", linestyle="--", linewidth=0.9, alpha=0.9, color="#555555")

    fallback_colors = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    pct_colors = palette[1:] + fallback_colors
    for idx, key in enumerate(STEP_KEYS, start=1):
        ax = axes[idx]
        color = pct_colors[idx - 1] if idx - 1 < len(pct_colors) else "#ff9da7"
        rate = median_recent_rate(pct_series[key], timestamps)
        ax.plot(
            timestamps,
            pct_series[key],
            label=f"{key}_pct",
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=5,
            linestyle="--",
        )
        ax.set_title(f"{key} • median Δ(<=10)={rate:.1f}%/day")
        ax.set_ylabel(f"{key}\n(%)")
        ax.set_ylim(0, max_pct_value)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, axis="both", linestyle="--", linewidth=0.9, alpha=0.9, color="#555555")

    axes[-1].set_xlabel("Execution time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    fig.tight_layout()
    return fig


def collect_membership(station_name: str) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Read metadata sources and return basename membership sets."""
    station_suffix = station_name[-2:]
    station_number = str(int(station_suffix))
    base_dir = REPO_ROOT / "STATIONS" / station_name

    remote_csv = (
        base_dir
        / "STAGE_0/REPROCESSING/STEP_1/METADATA"
        / f"remote_database_{station_number}.csv"
    )
    remote_order = read_unique_list(remote_csv, REMOTE_COLUMN)
    membership: Dict[str, Set[str]] = {"remote": set(remote_order)}

    for idx in TASK_IDS:
        task_key = f"task_{idx}"
        task_csv = (
            base_dir
            / f"STAGE_1/EVENT_DATA/STEP_1/TASK_{idx}/METADATA/task_{idx}_metadata_execution.csv"
        )
        membership[task_key] = set(read_unique_list(task_csv, FILENAME_COLUMN))

    step2_csv = (
        base_dir / "STAGE_1/EVENT_DATA/STEP_2/METADATA/step_2_metadata_execution.csv"
    )
    membership["step_2"] = set(read_unique_list(step2_csv, FILENAME_COLUMN))
    return remote_order, membership


def process_station(
    station_name: str,
    skip_plot: bool,
    pdf_pages: PdfPages | None,
) -> bool:
    """Run the fill-factor tracking pipeline for a single station."""
    presence_path = station_presence_path(station_name)
    history_path = station_history_path(station_name)
    ensure_history_schema(history_path)
    if station_name == LEGACY_STATION:
        ensure_history_schema(LEGACY_HISTORY_PATH)

    remote_order, membership = collect_membership(station_name)
    rows = build_presence_rows(remote_order, membership)
    write_presence_matrix(rows, presence_path)

    legacy_notice = ""
    if station_name == LEGACY_STATION:
        write_presence_matrix(rows, LEGACY_PRESENCE_PATH)
        legacy_notice = f" (legacy copy → {LEGACY_PRESENCE_PATH.name})"

    totals = compute_totals(membership)
    executed_at = datetime.now(timezone.utc).astimezone().isoformat()
    snapshot = append_snapshot(totals, history_path, executed_at=executed_at)
    if station_name == LEGACY_STATION:
        append_snapshot(totals, LEGACY_HISTORY_PATH, executed_at=executed_at)

    figure = None if skip_plot else plot_history(history_path, station_name)
    appended_to_pdf = False
    if figure is not None:
        if pdf_pages is not None:
            pdf_save_rasterized_page(pdf_pages, figure)
            appended_to_pdf = True
        plt.close(figure)

    remote_total = totals.get("remote", 0)
    step2_total = totals.get("step_2", 0)
    fill_pct = snapshot["step_2_pct"]

    print(
        f"[{station_name}] wrote {len(rows)} matrix rows → {presence_path}"
        f"{legacy_notice}"
    )
    print(
        f"[{station_name}] snapshot: remote={remote_total} | step_2={step2_total} | "
        f"fill_factor={fill_pct:.3f}% → {history_path.name}"
    )
    pct_summary = ", ".join(
        f"{key}={snapshot[f'{key}_pct']:.3f}%"
        for key in STEP_KEYS
    )
    print(f"[{station_name}] coverage: {pct_summary}")
    if not skip_plot:
        if figure is not None and appended_to_pdf:
            print(
                f"[{station_name}] time-series plot appended → {TIMESERIES_PDF_PATH.name}"
            )
        elif figure is not None:
            print(f"[{station_name}] time-series plot generated (PDF not available).")
        else:
            print(f"[{station_name}] plot not generated (insufficient history yet).")
    else:
        print(f"[{station_name}] plot skipped by request.")

    return bool(figure is not None and appended_to_pdf)


def main(stations: Sequence[str], skip_plot: bool) -> None:
    pdf_pages: PdfPages | None = None
    plots_written = False

    if not skip_plot:
        TIMESERIES_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
        pdf_pages = PdfPages(TIMESERIES_PDF_PATH)

    try:
        for station_name in stations:
            appended = process_station(
                station_name,
                skip_plot=skip_plot,
                pdf_pages=pdf_pages,
            )
            plots_written = plots_written or appended
    finally:
        if pdf_pages is not None:
            pdf_pages.close()

    if not skip_plot:
        if plots_written:
            print(f"Combined PDF refreshed → {TIMESERIES_PDF_PATH}")
        else:
            if TIMESERIES_PDF_PATH.exists():
                TIMESERIES_PDF_PATH.unlink(missing_ok=True)
            print("No plots generated; PDF not created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track fill-factor coverage for all MINGO stations in real time."
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Update CSV outputs but skip the matplotlib time-series plot.",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        choices=STATIONS,
        help="Subset of stations to process (default: all).",
    )
    args = parser.parse_args()
    targets = tuple(args.stations) if args.stations else STATIONS
    main(stations=targets, skip_plot=args.skip_plot)
