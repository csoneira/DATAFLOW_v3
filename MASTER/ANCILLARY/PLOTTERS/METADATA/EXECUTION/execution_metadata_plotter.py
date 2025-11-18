#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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


STATIONS: Tuple[str, ...] = ("1", "2", "3", "4")
TASK_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)
BASE_PATH = Path.home() / "DATAFLOW_v3" / "STATIONS"
OUTPUT_FILENAME = "execution_metadata_report.pdf"
TIMESTAMP_FMT = "%Y-%m-%d_%H.%M.%S"
FILENAME_TIMESTAMP_PATTERN = re.compile(r"mi0\d(\d{11})$", re.IGNORECASE)
CLI_DESCRIPTION = "Generate Stage 0/1 execution metadata plots."
HELP_NOTES = """
Notes:
  --include-remote-db requires --include-stage0 and inserts each station's remote_database_<station>.csv as the second Stage 0 panel.
  --include-hv overlays HV status shading on every execution subplot and appends '_hv' to the PDF filename.
  --no-data adds a dedicated HV-only subplot that plots hv_HVneg values with coloured markers (requires --include-hv).
""".strip()

minutes_upper_limit = 5
DEFAULT_POINT_SIZE = 3
ZOOM_POINT_SIZE = 6
point_size = DEFAULT_POINT_SIZE
plot_linestyle = "None"
HV_DEFAULT_THRESHOLD = 4.5

LAB_LOG_DATE_PATTERN = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


def _load_hv_csv(csv_path: Path) -> pd.DataFrame:
    """Load HV CSV files with the expected columns."""
    try:
        df = pd.read_csv(csv_path, usecols=["Time", "hv_HVneg"], low_memory=False)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["hv_HVneg"] = pd.to_numeric(df["hv_HVneg"], errors="coerce")
    return df.dropna(subset=["Time", "hv_HVneg"])


def _lab_log_date_from_name(path: Path) -> Optional[datetime]:
    match = LAB_LOG_DATE_PATTERN.search(path.stem)
    if not match:
        return None
    year, month, day = map(int, match.groups())
    return datetime(year, month, day)


def load_hv_dataframe(
    station: str, time_bounds: Optional[Tuple[datetime, datetime]]
) -> pd.DataFrame:
    """Load HV telemetry for the station, optionally clipped to the provided window."""
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


def hv_value_colours(values: Sequence[float], thr: float) -> List[str]:
    arr = np.asarray(values, dtype=float)
    colours = np.full(arr.shape, "orange", dtype=object)
    colours[arr >= thr] = "green"
    colours[np.isnan(arr)] = "red"
    return colours.tolist()


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
        (
            mdates.date2num(row.start),
            max(1e-6, mdates.date2num(row.end) - mdates.date2num(row.start)),
        )
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


def plot_hv_panel(
    ax: plt.Axes,
    hv_df: Optional[pd.DataFrame],
    hv_segments: Optional[pd.DataFrame],
    current_time: datetime,
    markers: Iterable[datetime],
    xlim: Optional[Tuple[datetime, datetime]],
    hv_threshold: float,
    current_time_label: str,
) -> None:
    """Render a dedicated HV subplot."""
    ax.set_title("HV timeline (hv_HVneg)")
    ax.set_ylabel("HV (-kV)")
    ax.grid(True, alpha=0.3)
    ax.yaxis.label.set_color("tab:purple")
    ax.tick_params(axis="y", colors="tab:purple")

    now_line = ax.axvline(
        current_time,
        color="green",
        linestyle="--",
        linewidth=1.0,
        label="Current time",
    )
    for marker in markers:
        ax.axvline(
            marker,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    legend_handles = [now_line]
    legend_labels = [now_line.get_label()]

    if hv_df is None or hv_df.empty:
        ax.text(
            0.5,
            0.5,
            "No HV data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="dimgray",
        )
        ax.set_ylim(-6, 1)
    else:
        hv_series = hv_df["hv_HVneg"]
        valid = hv_series.dropna()
        if valid.empty:
            axis_lower, axis_upper = -6, 1
        else:
            axis_lower = float(valid.min()) - 0.5
            axis_upper = float(valid.max()) + 0.5
            if axis_lower == axis_upper:
                axis_upper = axis_lower + 1
        ax.set_ylim(axis_lower, axis_upper)

        if hv_segments is not None:
            span_background_from_segments(ax, hv_segments, axis_lower, axis_upper, xlim)

        mask = np.isfinite(hv_series.to_numpy())
        if mask.any():
            hv_index = hv_series.index.to_numpy()[mask]
            hv_values = hv_series.to_numpy()[mask]
            colours = np.asarray(hv_value_colours(hv_series.to_numpy(), hv_threshold), dtype=object)[mask]
            hv_line = ax.plot(
                hv_index,
                hv_values,
                color="dimgray",
                linewidth=0.8,
                alpha=0.7,
                label="hv_HVneg (kV)",
            )[0]
            ax.scatter(
                hv_index,
                hv_values,
                c=colours,
                s=8,
                alpha=0.9,
                edgecolors="none",
                zorder=3,
            )
            legend_handles.insert(0, hv_line)
            legend_labels.insert(0, hv_line.get_label())
        else:
            ax.text(
                0.5,
                0.5,
                "HV readings invalid",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="dimgray",
            )

    ax.legend(legend_handles, legend_labels, loc="upper left")
    ax.annotate(
        current_time_label,
        xy=(current_time, ax.get_ylim()[1]),
        xycoords=("data", "data"),
        xytext=(15, -20),
        textcoords="offset points",
        rotation=90,
        va="top",
        ha="center",
        color="green",
        fontsize=10,
    )

def extract_datetime_from_basename(basename: str) -> Optional[datetime]:
    """Extract a timestamp encoded as mi0?YYJJJHHMMSS from filenames with multi-part suffixes."""
    raw_name = Path(basename).name
    candidates = [raw_name]

    tmp = Path(raw_name)
    while tmp.suffix:
        tmp = Path(tmp.stem)
        candidates.append(tmp.name)

    match = None
    for candidate in candidates:
        match = FILENAME_TIMESTAMP_PATTERN.search(candidate)
        if match:
            break

    if not match:
        return None

    digits = match.group(1)
    try:
        year = 2000 + int(digits[0:2])
        day_of_year = int(digits[2:5])
        hour = int(digits[5:7])
        minute = int(digits[7:9])
        second = int(digits[9:11])
    except ValueError:
        return None

    try:
        base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    except ValueError:
        return None

    return base_date.replace(hour=hour, minute=minute, second=second)


def _load_metadata_csv_from_path(metadata_csv: Path) -> pd.DataFrame:
    """Load and normalize a metadata CSV."""
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
        print(
            f"Warning: missing expected columns {missing_columns} in {metadata_csv}; "
            "skipping metadata entries for this task."
        )
        return pd.DataFrame()

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
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename), errors="coerce"
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


def _ordered_unique(values: Iterable[str]) -> List[str]:
    """Return the input strings without duplicates while preserving order."""
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        candidate = value.strip()
        if not candidate or candidate in seen:
            continue
        ordered.append(candidate)
        seen.add(candidate)
    return ordered


def _station_token_map(station: str) -> Dict[str, str]:
    """Provide string tokens derived from the station identifier for filename templates."""
    clean = station.strip()
    tokens: Dict[str, str] = {
        "station": clean,
        "station_lower": clean.lower(),
        "station_upper": clean.upper(),
    }
    digits = "".join(ch for ch in clean if ch.isdigit())
    if digits:
        stripped = digits.lstrip("0") or "0"
        tokens["id"] = stripped
        tokens["id02"] = stripped.zfill(2)
        tokens["mingo_id"] = f"mingo{stripped}"
        tokens["mingo_id02"] = f"mingo{tokens['id02']}"
    return tokens


def _station_file_candidates(
    station: str,
    patterns: Sequence[str],
    fallback: Optional[str] = None,
) -> List[str]:
    """Generate ordered filename candidates for the station using the provided templates."""
    tokens = _station_token_map(station)
    generated: List[str] = []
    for pattern in patterns:
        try:
            generated.append(pattern.format(**tokens))
        except KeyError:
            continue
    if fallback:
        generated.append(fallback)
    return _ordered_unique(generated)


def _resolve_station_metadata_file(
    directory: Path,
    station: str,
    patterns: Sequence[str],
    fallback: Optional[str] = None,
) -> Optional[Path]:
    """Find the first existing metadata file for the station from the provided filename templates."""
    for filename in _station_file_candidates(station, patterns, fallback):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


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
        / f"task_{task_id}_metadata_execution.csv"
    )
    return _load_metadata_csv_from_path(metadata_csv)


def load_step2_metadata_csv(station: str) -> pd.DataFrame:
    """Load step 2 metadata CSV for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_2"
        / "METADATA"
        / "step_2_metadata_execution.csv"
    )
    return _load_metadata_csv_from_path(metadata_csv)


def load_stage0_raw_metadata(station: str) -> pd.DataFrame:
    """Load Stage 0 raw file arrival metadata for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "NEW_FILES"
        / "METADATA"
        / "raw_files_brought.csv"
    )
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    expected_columns = {"filename", "bring_timestamp"}
    missing = expected_columns.difference(df.columns)
    if missing:
        print(
            f"Warning: missing expected columns {missing} in {metadata_csv}; "
            "skipping Stage 0 raw metadata entries."
        )
        return pd.DataFrame()

    df = df.copy()
    df["filename_base"] = df["filename"].astype(str)
    df["execution_timestamp"] = pd.to_datetime(
        df["bring_timestamp"], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["total_execution_time_minutes"] = float("nan")
    df["data_purity_percentage"] = float("nan")
    df = df.dropna(subset=["execution_timestamp"]).sort_values(
        "execution_timestamp"
    )
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def load_stage0_step1_metadata(station: str) -> pd.DataFrame:
    """Load Stage 0 reprocessing step 1 metadata for a given station."""
    metadata_csv = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "REPROCESSING"
        / "STEP_1"
        / "METADATA"
        / "hld_files_brought.csv"
    )
    if not metadata_csv.exists():
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    timestamp_col: Optional[str] = None
    for candidate in ("bring_timestamp", "bring_timesamp"):
        if candidate in df.columns:
            timestamp_col = candidate
            break

    expected_columns = {"hld_name"}
    if timestamp_col:
        expected_columns.add(timestamp_col)

    missing = expected_columns.difference(df.columns)
    if missing:
        print(
            f"Warning: missing expected columns {missing} in {metadata_csv}; "
            "skipping Stage 0 step 1 metadata entries."
        )
        return pd.DataFrame()

    if not timestamp_col:
        print(
            "Warning: Stage 0 step 1 metadata missing bring_timestamp/bring_timesamp "
            f"column in {metadata_csv}; skipping entries."
        )
        return pd.DataFrame()

    df = df.copy()
    df["filename_base"] = df["hld_name"].astype(str)
    df["execution_timestamp"] = pd.to_datetime(
        df[timestamp_col], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["total_execution_time_minutes"] = float("nan")
    df["data_purity_percentage"] = float("nan")
    df = df.dropna(subset=["execution_timestamp"]).sort_values(
        "execution_timestamp"
    )
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def load_stage0_step2_metadata(station: str) -> pd.DataFrame:
    """Load Stage 0 reprocessing step metadata for a given station."""
    metadata_dir = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "REPROCESSING"
        / "STEP_2"
        / "METADATA"
    )
    metadata_csv = _resolve_station_metadata_file(
        metadata_dir,
        station,
        (
            "dat_files_unpacked_{station_lower}.csv",
            "dat_files_unpacked_{station}.csv",
            "dat_files_unpacked_{mingo_id02}.csv",
            "dat_files_unpacked_{mingo_id}.csv",
            "dat_files_unpacked_{id02}.csv",
            "dat_files_unpacked_{id}.csv",
        ),
        fallback="dat_files_unpacked.csv",
    )
    if metadata_csv is None:
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    expected_columns = {
        "dat_name",
        "execution_timestamp",
        "execution_duration_s",
    }
    missing = expected_columns.difference(df.columns)
    if missing:
        print(
            f"Warning: missing expected columns {missing} in {metadata_csv}; "
            "skipping Stage 0 reprocessing metadata entries."
        )
        return pd.DataFrame()

    df = df.copy()
    prefix = f"mi0{station}"
    dat_series = df["dat_name"].astype(str)
    mask = dat_series.str.lower().str.startswith(prefix.lower())
    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame()

    df["filename_base"] = df["dat_name"].astype(str)
    df["execution_timestamp"] = pd.to_datetime(
        df["execution_timestamp"], errors="coerce"
    )
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["total_execution_time_minutes"] = pd.to_numeric(
        df["execution_duration_s"], errors="coerce"
    ).div(60)
    df["data_purity_percentage"] = float("nan")
    df = df.dropna(subset=["execution_timestamp"]).sort_values(
        "execution_timestamp"
    )
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def load_stage0_remote_database(station: str) -> pd.DataFrame:
    """Load Stage 0 remote database entries for a given station."""
    metadata_dir = (
        BASE_PATH
        / f"MINGO0{station}"
        / "STAGE_0"
        / "REPROCESSING"
        / "STEP_0"
        / "OUTPUT_FILES"
    )
    metadata_csv = _resolve_station_metadata_file(
        metadata_dir,
        station,
        (
            "clean_remote_database_{id}.csv",
            "clean_remote_database_{id02}.csv",
            "clean_remote_database_{station_lower}.csv",
            "clean_remote_database_{station}.csv",
            "remote_database_{id}.csv",
            "remote_database_{id02}.csv",
            "remote_database_{station_lower}.csv",
            "remote_database_{station}.csv",
        ),
    )
    if metadata_csv is None:
        return pd.DataFrame()

    df = pd.read_csv(metadata_csv)
    if "basename" not in df.columns:
        print(
            f"Warning: missing 'basename' column in {metadata_csv}; "
            "skipping remote database entries."
        )
        return pd.DataFrame()

    df = df.copy()
    df["filename_base"] = df["basename"].astype(str)
    df["file_timestamp"] = pd.to_datetime(
        df["filename_base"].map(extract_datetime_from_basename),
        errors="coerce",
    )
    df["execution_timestamp"] = df["file_timestamp"]
    df["total_execution_time_minutes"] = float("nan")
    df["data_purity_percentage"] = float("nan")

    df = df.dropna(subset=["file_timestamp"]).sort_values("file_timestamp")
    return df[
        [
            "filename_base",
            "execution_timestamp",
            "file_timestamp",
            "total_execution_time_minutes",
            "data_purity_percentage",
        ]
    ].reset_index(drop=True)


def ensure_output_directory(path: Path) -> None:
    """Ensure the directory for the output file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def build_station_pages(
    include_step2: bool,
    include_stage0: bool,
    include_remote_db: bool,
) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    """Collect metadata DataFrames for each station."""
    station_data: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    for station in STATIONS:
        datasets: List[Tuple[str, pd.DataFrame]] = []
        if include_stage0:
            stage0_datasets: List[Tuple[str, pd.DataFrame]] = [
                (
                    "STAGE0_RAW_FILES",
                    load_stage0_raw_metadata(station),
                ),
                (
                    "STAGE0_REPROCESS_STEP_1",
                    load_stage0_step1_metadata(station),
                ),
                (
                    "STAGE0_REPROCESS_STEP_2",
                    load_stage0_step2_metadata(station),
                ),
            ]
            if include_remote_db:
                stage0_datasets.insert(
                    1,
                    (
                        f"STAGE0_REMOTE_DB_{station}",
                        load_stage0_remote_database(station),
                    ),
                )
            datasets.extend(stage0_datasets)

        datasets.extend(
            (f"TASK_{task_id}", load_metadata_csv(station, task_id))
            for task_id in TASK_IDS
        )
        if include_step2:
            datasets.append(("STEP_2", load_step2_metadata_csv(station)))
        station_data[station] = datasets
    return station_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-r",
        "--real-date",
        action="store_true",
        help="Plot using timestamps extracted from the metadata filenames.",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        action="store_true",
        help="Restrict the x-axis to the last hour ending at the current time.",
    )
    parser.add_argument(
        "--include-step2",
        action="store_true",
        help="Include Stage 1 Step 2 execution metadata in the plots.",
    )
    parser.add_argument(
        "--include-stage0",
        action="store_true",
        help="Prepend Stage 0 metadata (raw files and reprocessing) to each station page.",
    )
    parser.add_argument(
        "--include-remote-db",
        action="store_true",
        help=(
            "Include Stage 0 remote database points (remote_database_<station>.csv) directly "
            "after the raw Stage 0 plot. Requires --include-stage0."
        ),
    )
    parser.add_argument(
        "--include-hv",
        action="store_true",
        help=(
            "Overlay execution panels with HV state shading sourced from LAB logs "
            "and optionally add a dedicated HV subplot."
        ),
    )
    parser.add_argument(
        "--hv-threshold",
        type=float,
        default=HV_DEFAULT_THRESHOLD,
        help="HV threshold (kV) used to classify colour segments (default: 4.5).",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help=(
            "Add an HV-only subplot that shows hv_HVneg readings coloured by state "
            "(only meaningful with --include-hv)."
        ),
    )
    return parser


def usage() -> str:
    """Return the CLI usage/help string."""
    return f"{build_parser().format_help()}\n\n{HELP_NOTES}\n"


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.help:
        print(usage(), end="")
        raise SystemExit(0)
    if args.include_remote_db and not args.include_stage0:
        print(
            "Warning: --include-remote-db requires --include-stage0; flag ignored.",
            file=sys.stderr,
        )
        args.include_remote_db = False
    if args.no_data and not args.include_hv:
        print(
            "Warning: --no-data has no effect without --include-hv; flag ignored.",
            file=sys.stderr,
        )
        args.no_data = False
    return args


def compute_time_bounds(
    station_pages: Dict[str, List[Tuple[str, pd.DataFrame]]], use_real_date: bool
) -> Optional[Tuple[datetime, datetime]]:
    minima: List[datetime] = []
    maxima: List[datetime] = []
    column = "file_timestamp" if use_real_date else "execution_timestamp"

    for datasets in station_pages.values():
        for _, df in datasets:
            if df.empty or column not in df:
                continue
            series = df[column].dropna()
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


def compute_month_markers(
    bounds: Optional[Tuple[datetime, datetime]]
) -> List[datetime]:
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


def resolve_output_path(use_real_date: bool, zoom: bool, include_hv: bool) -> Path:
    filename = OUTPUT_FILENAME
    if use_real_date:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_real_time.pdf"
        else:
            filename = f"{filename}_real_time"
    if zoom:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_zoomed.pdf"
        else:
            filename = f"{filename}_zoomed"
    if include_hv:
        if filename.lower().endswith(".pdf"):
            filename = f"{filename[:-4]}_hv.pdf"
        else:
            filename = f"{filename}_hv"
    OUTPUT_DIR = Path.home() / "DATAFLOW_v3" / "MASTER" / "ANCILLARY" / "PLOTTERS" / "METADATA" / "EXECUTION" / "PLOTS"
    return OUTPUT_DIR / filename


def plot_station(
    station: str,
    station_datasets: Iterable[Tuple[str, pd.DataFrame]],
    pdf: PdfPages,
    use_real_date: bool,
    time_bounds: Optional[Tuple[datetime, datetime]],
    month_markers: Iterable[datetime],
    current_time: datetime,
    hv_segments: Optional[pd.DataFrame],
    hv_df: Optional[pd.DataFrame],
    include_hv: bool,
    show_hv_panel: bool,
    hv_threshold: float,
) -> None:
    """Render a page with execution metadata subplots for one station."""
    station_datasets = list(station_datasets)
    has_stage0_data = any(label.startswith("STAGE0") for label, _ in station_datasets)
    month_markers = list(month_markers)
    current_time_str_full = current_time.strftime("%Y-%m-%d %H:%M:%S")
    current_time_str_time_only = current_time.strftime("%H:%M:%S")

    median_minutes = []
    for _, df in station_datasets:
        if df.empty:
            continue
        runtime_series = df["total_execution_time_minutes"].dropna()
        if runtime_series.empty:
            continue
        median_value = runtime_series.median()
        if pd.notna(median_value):
            median_minutes.append(median_value)
    total_median_minutes = float(sum(median_minutes))

    hv_panel_count = 1 if show_hv_panel else 0
    num_panels = len(station_datasets) + hv_panel_count
    if num_panels == 0:
        num_panels = 1
    fig_height = max(8.5, num_panels * 2.2)
    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(11, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(
        (
            f"MINGO0{station} – "
            f"{'Stage 0/1' if has_stage0_data else 'Stage 1'} Execution Metadata "
            f"(Total median minutes/file: {total_median_minutes:.2f}) "
            f"Current: {current_time_str_full}"
        ),
        fontsize=14,
    )

    if num_panels == 1:
        axes = [axes]  # type: ignore[list-item]
    else:
        axes = list(axes)  # type: ignore[assignment]

    if show_hv_panel:
        hv_axis = axes[0]
        data_axes = axes[1:]
    else:
        hv_axis = None
        data_axes = axes

    xlim: Optional[Tuple[datetime, datetime]] = None
    if time_bounds:
        xmin, xmax = time_bounds
        if xmin == xmax:
            xmax = xmin + timedelta(minutes=1)
        xlim = (xmin, xmax)
        for axis in axes:
            axis.set_xlim(xmin, xmax)

    if xlim:
        xmin, xmax = xlim
        markers_to_use = [m for m in month_markers if xmin <= m <= xmax]
    else:
        markers_to_use = month_markers

    if show_hv_panel and hv_axis is not None:
        hv_segments_to_use = hv_segments if include_hv else None
        plot_hv_panel(
            hv_axis,
            hv_df,
            hv_segments_to_use,
            current_time,
            markers_to_use,
            xlim,
            hv_threshold,
            current_time_str_time_only,
        )

    for ax, (label, df) in zip(data_axes, station_datasets):
        ax.set_title(label)
        ax.set_ylabel("Exec Time (min)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.label.set_color("tab:blue")
        ax.tick_params(axis="y", colors="tab:blue")

        now_line = ax.axvline(
            current_time,
            color="green",
            linestyle="--",
            linewidth=1.0,
            label="Current time",
        )

        for marker in markers_to_use:
            ax.axvline(
                marker,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
            )

        axis_upper = minutes_upper_limit

        def apply_hv_background(upper: float) -> None:
            if include_hv and hv_segments is not None:
                span_background_from_segments(ax, hv_segments, 0, upper, xlim)

        if df.empty:
            ax.set_ylim(0, axis_upper)
            apply_hv_background(axis_upper)
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
            ax.annotate(
                current_time_str_time_only,
                xy=(current_time, ax.get_ylim()[1]),
                xycoords=("data", "data"),
                xytext=(15, -20),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="center",
                color="green",
                fontsize=10,
            )
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        if use_real_date:
            df_plot = (
                df.dropna(subset=["file_timestamp"])
                .sort_values("file_timestamp")
                .copy()
            )
        else:
            df_plot = df.sort_values("execution_timestamp")

        if df_plot.empty:
            ax.set_ylim(0, axis_upper)
            apply_hv_background(axis_upper)
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
            ax.annotate(
                current_time_str_time_only,
                xy=(current_time, ax.get_ylim()[1]),
                xycoords=("data", "data"),
                xytext=(15, -20),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="center",
                color="green",
                fontsize=10,
            )
            ax.legend([now_line], [now_line.get_label()], loc="upper left")
            continue

        runtime_series = df_plot["total_execution_time_minutes"]
        runtime_non_nan = runtime_series.dropna()
        has_runtime_points = not runtime_non_nan.empty

        if has_runtime_points:
            runtime_max = runtime_non_nan.max()
            if pd.notna(runtime_max):
                axis_upper = max(axis_upper, float(runtime_max) * 1.15)

        ax.set_ylim(0, axis_upper)
        apply_hv_background(axis_upper)

        x = (
            df_plot["file_timestamp"]
            if use_real_date
            else df_plot["execution_timestamp"]
        )

        placeholder_runtime = False
        if has_runtime_points:
            y_runtime = runtime_series
            runtime_label = "Execution time (min)"
        else:
            placeholder_runtime = True
            midpoint = axis_upper / 2 if axis_upper > 0 else 0.5
            y_runtime = pd.Series(midpoint, index=df_plot.index)
            runtime_label = "Arrival marker (no runtime)"

        runtime_line = None
        if not y_runtime.dropna().empty:
            (runtime_line,) = ax.plot(
                x,
                y_runtime,
                marker="o",
                markersize=point_size,
                linestyle=plot_linestyle,
                color="tab:blue",
                label=runtime_label,
                alpha=0.5,
            )

        if placeholder_runtime:
            ax.text(
                0.01,
                0.92,
                "No execution time data",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="tab:blue",
            )

        purity_line = None
        purity_series = df_plot["data_purity_percentage"]
        if purity_series.notna().any():
            ax_second = ax.twinx()
            (purity_line,) = ax_second.plot(
                x,
                purity_series,
                marker="x",
                markersize=point_size,
                linestyle=plot_linestyle,
                color="tab:red",
                label="Data purity (%)",
                alpha=0.5,
            )
            ax_second.set_ylabel("Purity (%)")
            ax_second.set_ylim(0, 105)
            ax_second.yaxis.label.set_color("tab:red")
            ax_second.tick_params(axis="y", colors="tab:red")

        handles = []
        labels = []
        if runtime_line:
            handles.append(runtime_line)
            labels.append(runtime_line.get_label())
        if purity_line:
            handles.append(purity_line)
            labels.append(purity_line.get_label())
        handles.append(now_line)
        labels.append(now_line.get_label())
        ax.legend(handles, labels, loc="upper left")

        ax.annotate(
            current_time_str_time_only,
            xy=(current_time, ax.get_ylim()[1]),
            xycoords=("data", "data"),
            xytext=(15, -20),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="center",
            color="green",
            fontsize=10,
        )

    target_axis = data_axes[-1] if data_axes else axes[-1]
    target_axis.set_xlabel("File timestamp" if use_real_date else "Execution timestamp")
    target_axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    )
    target_axis.xaxis.set_tick_params(rotation=0)
    pdf_save_rasterized_page(pdf, fig, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    global point_size
    point_size = ZOOM_POINT_SIZE if args.zoom else DEFAULT_POINT_SIZE
    station_pages = build_station_pages(
        include_step2=args.include_step2,
        include_stage0=args.include_stage0,
        include_remote_db=args.include_remote_db,
    )
    current_time = datetime.now()
    if args.zoom:
        time_bounds: Optional[Tuple[datetime, datetime]] = (
            current_time - timedelta(hours=1),
            current_time + timedelta(minutes=5),
        )
    else:
        time_bounds = compute_time_bounds(
            station_pages, use_real_date=args.real_date
        )
    month_markers = compute_month_markers(time_bounds)

    if args.include_hv:
        hv_segments_by_station: Dict[str, Optional[pd.DataFrame]] = {}
        hv_data_by_station: Dict[str, Optional[pd.DataFrame]] = {}
        for station in station_pages:
            hv_df = load_hv_dataframe(station, time_bounds)
            hv_data_by_station[station] = hv_df if not hv_df.empty else None
            if hv_df.empty:
                hv_segments_by_station[station] = None
            else:
                hv_segments_by_station[station] = hv_state_segments(
                    hv_df.index, hv_df["hv_HVneg"], args.hv_threshold
                )
    else:
        hv_segments_by_station = {station: None for station in station_pages}
        hv_data_by_station = {station: None for station in station_pages}

    output_path = resolve_output_path(args.real_date, args.zoom, args.include_hv)
    ensure_output_directory(output_path)

    with PdfPages(output_path) as pdf:
        for station, datasets in station_pages.items():
            plot_station(
                station,
                datasets,
                pdf,
                use_real_date=args.real_date,
                time_bounds=time_bounds,
                month_markers=month_markers,
                current_time=current_time,
                hv_segments=hv_segments_by_station.get(station),
                hv_df=hv_data_by_station.get(station),
                include_hv=args.include_hv,
                show_hv_panel=args.no_data,
                hv_threshold=args.hv_threshold,
            )

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
