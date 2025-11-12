#!/usr/bin/env python3
"""Combine real-time execution metadata plots with HV state shading."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "DATAFLOW_v3"
    / "MASTER"
    / "ANCILLARY"
    / "PLOTTERS"
    / "REAL_TIME_HV_AND_EXEC"
    / "PLOTS"
)
OUTPUT_FILENAME = "real_time_hv_and_execution.pdf"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"
minutes_upper_limit = 5
point_size = 3
plot_linestyle = "None"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Stage 1 execution metadata using real-time file timestamps "
            "and shade the background according to HV status."
        )
    )
    parser.add_argument(
        "--include-step2",
        action="store_true",
        help="Include Stage 1 Step 2 execution metadata.",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Restrict the x-axis to the last hour ending at now.",
    )
    parser.add_argument(
        "--hv-threshold",
        type=float,
        default=3.0,
        help="HV threshold (kV) used to classify green/orange/red segments. Default: 3.0",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        choices=STATIONS,
        default=list(STATIONS),
        help="Subset of stations to include (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / OUTPUT_FILENAME,
        help="Output PDF path (default: REAL_TIME_HV_AND_EXEC/PLOTS/real_time_hv_and_execution.pdf).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    if args.output.is_dir():
        raise SystemExit(f"--output must be a file path, not a directory: {args.output}")
    return args


def _load_metadata_csv(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    expected = {
        "filename_base",
        "execution_timestamp",
        "data_purity_percentage",
        "total_execution_time_minutes",
    }
    if not expected.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["execution_timestamp"] = pd.to_datetime(
        df["execution_timestamp"], format=TIMESTAMP_FMT, errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename), errors="coerce"
    )
    df["data_purity_percentage"] = pd.to_numeric(
        df["data_purity_percentage"], errors="coerce"
    )
    df["total_execution_time_minutes"] = pd.to_numeric(
        df["total_execution_time_minutes"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "file_timestamp",
            "execution_timestamp",
            "data_purity_percentage",
            "total_execution_time_minutes",
        ]
    )
    return df.sort_values("file_timestamp").reset_index(drop=True)


def extract_datetime_from_basename(basename: str) -> Optional[datetime]:
    """Interpret filename_base timestamps following mi0XYYDDDhhmmss."""
    stem = Path(basename).stem
    if len(stem) < 15:
        return None
    digits = stem[-11:]
    try:
        year = 2000 + int(digits[0:2])
        day_of_year = int(digits[2:5])
        hour = int(digits[5:7])
        minute = int(digits[7:9])
        second = int(digits[9:11])
    except ValueError:
        return None

    try:
        base = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    except ValueError:
        return None
    return base.replace(hour=hour, minute=minute, second=second)


def load_station_metadata(
    station: str, include_step2: bool
) -> List[Tuple[str, pd.DataFrame]]:
    datasets: List[Tuple[str, pd.DataFrame]] = []
    for task_id in TASK_IDS:
        metadata_csv = (
            BASE_PATH
            / f"MINGO0{station}"
            / "STAGE_1"
            / "EVENT_DATA"
            / "STEP_1"
            / f"TASK_{task_id}"
            / "METADATA"
            / f"task_{task_id}_metadata_execution.csv"
        )
        datasets.append((f"TASK_{task_id}", _load_metadata_csv(metadata_csv)))

    if include_step2:
        step2_csv = (
            BASE_PATH
            / f"MINGO0{station}"
            / "STAGE_1"
            / "EVENT_DATA"
            / "STEP_2"
            / "METADATA"
            / "step_2_metadata_execution.csv"
        )
        datasets.append(("STEP_2", _load_metadata_csv(step2_csv)))
    return datasets


def _load_hv_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, usecols=["Time", "hv_HVneg"], low_memory=False)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["hv_HVneg"] = pd.to_numeric(df["hv_HVneg"], errors="coerce")
    return df.dropna(subset=["Time", "hv_HVneg"])


LAB_LOG_DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


def _lab_log_date_from_name(path: Path) -> Optional[datetime]:
    match = LAB_LOG_DATE_PATTERN.search(path.stem)
    if not match:
        return None
    year, month, day = map(int, match.groups())
    return datetime(year, month, day)


def load_hv_dataframe(
    station: str, time_bounds: Optional[Tuple[datetime, datetime]]
) -> pd.DataFrame:
    base = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_1"
        / "LAB_LOGS"
    )
    big_log = base / "big_log_lab_data.csv"

    frames: List[pd.DataFrame] = []
    if big_log.exists():
        frames.append(_load_hv_csv(big_log))
    else:
        step2_root = base / "STEP_2" / "OUTPUT_FILES"
        if step2_root.exists():
            if time_bounds:
                start, end = time_bounds
                margin = timedelta(days=1)
                target_start = start - margin
                target_end = end + margin
            else:
                target_start = target_end = None

            for csv_path in sorted(step2_root.rglob("lab_logs_*.csv")):
                log_date = _lab_log_date_from_name(csv_path)
                if log_date and target_start and log_date < target_start:
                    continue
                if log_date and target_end and log_date > target_end:
                    continue
                frames.append(_load_hv_csv(csv_path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("Time").set_index("Time")

    if time_bounds:
        start, end = time_bounds
        margin = timedelta(hours=2)
        lower = start - margin
        upper = end + margin
        df = df.loc[(df.index >= lower) & (df.index <= upper)]

    return df


def hv_state_segments(
    index: Sequence[pd.Timestamp],
    hv: Sequence[float],
    thr: float,
    max_gap: Optional[int] = 20,
) -> pd.DataFrame:
    """Classify HV timeline into (start, end, colour) segments."""
    t = pd.to_datetime(index, errors="coerce")
    hv_arr = np.asarray(hv, dtype=float)

    raw_state = np.where(hv_arr >= thr, 2, 1)
    mask_nan = np.isnan(hv_arr)

    if max_gap is not None and mask_nan.any():
        nan_runs = np.flatnonzero(np.diff(np.concatenate(([0], mask_nan, [0]))))
        starts, ends = nan_runs[::2], nan_runs[1::2]
        for s, e in zip(starts, ends):
            if e - s <= max_gap:
                raw_state[s:e] = raw_state[s - 1] if s > 0 else 0
                mask_nan[s:e] = False

    state = np.where(mask_nan, 0, raw_state)
    change = np.flatnonzero(np.diff(state)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(state)]))

    codes = state[starts]
    colours = np.take(["red", "orange", "green"], codes)

    return pd.DataFrame(
        {
            "start": t[starts].to_numpy(),
            "end": t[ends - 1].to_numpy(),
            "code": codes.astype(np.uint8),
            "colour": colours,
        }
    )


def span_background_from_segments(
    ax: plt.Axes,
    seg: pd.DataFrame,
    ylow: float,
    yhigh: float,
    clip_limits: Optional[Tuple[datetime, datetime]] = None,
) -> None:
    """Paint coloured spans over the provided axis."""
    if seg.empty:
        return
    clipped = seg.copy()
    if clip_limits is not None:
        xmin, xmax = clip_limits
        mask = (clipped["end"] >= xmin) & (clipped["start"] <= xmax)
        clipped = clipped.loc[mask].copy()
        if clipped.empty:
            return
        clipped.loc[clipped["start"] < xmin, "start"] = xmin
        clipped.loc[clipped["end"] > xmax, "end"] = xmax

    xranges = [
        (mdates.date2num(row.start), max(1e-6, mdates.date2num(row.end) - mdates.date2num(row.start)))
        for row in clipped.itertuples()
    ]
    colours = clipped["colour"].tolist()
    ax.broken_barh(
        xranges,
        (ylow, yhigh - ylow),
        facecolors=colours,
        alpha=0.15,
        linewidth=0,
        zorder=0,
    )


def compute_time_bounds(
    station_pages: Dict[str, List[Tuple[str, pd.DataFrame]]]
) -> Optional[Tuple[datetime, datetime]]:
    minima: List[datetime] = []
    maxima: List[datetime] = []

    for datasets in station_pages.values():
        for _, df in datasets:
            if df.empty:
                continue
            series = df["file_timestamp"].dropna()
            if series.empty:
                continue
            minima.append(series.min())
            maxima.append(series.max())

    if not minima or not maxima:
        return None

    lower = min(minima)
    upper = max(maxima)
    if lower == upper:
        upper = lower + timedelta(minutes=1)
    return (lower, upper)


def compute_month_markers(bounds: Optional[Tuple[datetime, datetime]]) -> List[datetime]:
    if not bounds:
        return []
    start, end = bounds
    if start > end:
        start, end = end, start

    markers: List[datetime] = []
    current = datetime(start.year, start.month, 1)
    if current < start and start.day != 1:
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    while current <= end:
        markers.append(current)
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return markers


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_station(
    station: str,
    datasets: Iterable[Tuple[str, pd.DataFrame]],
    hv_segments: Optional[pd.DataFrame],
    pdf: PdfPages,
    time_bounds: Optional[Tuple[datetime, datetime]],
    month_markers: Iterable[datetime],
    current_time: datetime,
) -> None:
    datasets = list(datasets)
    month_markers = list(month_markers)
    num_panels = len(datasets)

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(11, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    if num_panels == 1:
        axes = [axes]  # type: ignore[list-item]

    current_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(
        f"MINGO0{station} – Real-time execution metadata (now: {current_str})",
        fontsize=14,
    )

    xlim: Optional[Tuple[datetime, datetime]] = None
    if time_bounds:
        xmin, xmax = time_bounds
        xlim = (xmin, xmax)
        for axis in axes:
            axis.set_xlim(xmin, xmax)

    if xlim:
        markers = [m for m in month_markers if xlim[0] <= m <= xlim[1]]
    else:
        markers = month_markers

    for ax, (label, df) in zip(axes, datasets):
        ax.set_title(label)
        ax.set_ylabel("Exec Time (min)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, minutes_upper_limit)

        if hv_segments is not None:
            span_background_from_segments(ax, hv_segments, 0, minutes_upper_limit, xlim)

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
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        df_plot = df.dropna(subset=["file_timestamp"]).sort_values("file_timestamp")
        if df_plot.empty:
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
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        x = df_plot["file_timestamp"]
        runtime_line, = ax.plot(
            x,
            df_plot["total_execution_time_minutes"],
            marker="o",
            markersize=point_size,
            linestyle=plot_linestyle,
            color="tab:blue",
            label="Execution time (min)",
            alpha=0.6,
        )

        ax_second = ax.twinx()
        purity_line, = ax_second.plot(
            x,
            df_plot["data_purity_percentage"],
            marker="x",
            markersize=point_size,
            linestyle=plot_linestyle,
            color="tab:red",
            label="Data purity (%)",
            alpha=0.6,
        )
        ax_second.set_ylabel("Purity (%)")
        ax_second.set_ylim(0, 105)

        for marker in markers:
            ax.axvline(
                marker,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
            )

        handles = [runtime_line, purity_line, now_line]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="upper left")

    axes[-1].set_xlabel("File timestamp")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_directory(args.output)

    station_pages: Dict[str, List[Tuple[str, pd.DataFrame]]] = {
        station: load_station_metadata(station, include_step2=args.include_step2)
        for station in args.stations
    }

    current_time = datetime.now()
    if args.zoom:
        time_bounds: Optional[Tuple[datetime, datetime]] = (
            current_time - timedelta(hours=1),
            current_time + timedelta(minutes=5),
        )
    else:
        time_bounds = compute_time_bounds(station_pages)

    hv_segments_by_station: Dict[str, Optional[pd.DataFrame]] = {}
    for station in args.stations:
        hv_df = load_hv_dataframe(station, time_bounds)
        if hv_df.empty:
            hv_segments_by_station[station] = None
        else:
            hv_segments_by_station[station] = hv_state_segments(
                hv_df.index, hv_df["hv_HVneg"], args.hv_threshold
            )

    month_markers = compute_month_markers(time_bounds)

    with PdfPages(args.output) as pdf:
        for station in args.stations:
            plot_station(
                station,
                station_pages[station],
                hv_segments_by_station[station],
                pdf,
                time_bounds=time_bounds,
                month_markers=month_markers,
                current_time=current_time,
            )

    print(f"Saved combined real-time HV & execution report to: {args.output}")


if __name__ == "__main__":
    main()
