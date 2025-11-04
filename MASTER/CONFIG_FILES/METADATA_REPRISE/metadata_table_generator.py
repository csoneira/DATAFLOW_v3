#!/usr/bin/env python3
"""Detect outliers in STEP-1 task metadata tables and prepare reprocessing CSVs.

For each requested station/task combination the script:

* loads the available `task_*_metadata_*.csv` files,
* extracts the requested numeric columns,
* sorts entries by the timestamp encoded in the acquisition basename,
* applies a centred rolling-median filter (horizontal median) to obtain
  smoothed values,
* flags rows whose residual (original - smoothed) lies outside the requested
  quantile thresholds, and
* writes compact CSV tables with the rows that need reprocessing.

When ``--plot`` is supplied, a PDF is produced per station/task illustrating
the original series alongside its smoothed counterpart, the residual trace,
and the residual histogram for each analysed column.

By default, task-specific column families are analysed:

* Task 2: every column matching ``P*_s*_*`` (including the ``*_coeffs`` series).
* Task 4: all ``sigmoid_width_*`` and ``background_slope_*`` columns.
* All tasks: the execution-level ``data_purity_percentage`` metric.

Reprocessing tables are emitted to ``MASTER/CONFIG_FILES/METADATA_REPRISE/REFERENCE_TABLES``
unless ``--output-dir`` is provided.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from ast import literal_eval
import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(os.getenv("DATAFLOW_PROJECT_ROOT", Path.home() / "DATAFLOW_v3")).resolve()
STATIONS_DIR = PROJECT_ROOT / "STATIONS"
CONFIG_PATH = PROJECT_ROOT / "MASTER" / "CONFIG_FILES" / "config_global.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE"
CSV_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "REFERENCE_TABLES"


def load_home_path(config_path: Path) -> Path:
    """Return ``home_path`` declared in *config_path*; fall back to ``~``."""
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as config_file:
            try:
                config_data = yaml.safe_load(config_file) or {}
            except yaml.YAMLError as config_error:
                raise RuntimeError(f"Unable to parse configuration file {config_path}") from config_error
            home_path = config_data.get("home_path")
            if home_path:
                return Path(home_path).expanduser()
    return Path.home()


HOME_PATH = load_home_path(CONFIG_PATH)


TASK_COLUMN_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "2": (
        re.compile(r"^P\d_s\d_.*"),
    ),
    "4": (
        re.compile(r"^sigmoid_width_"),
        re.compile(r"^background_slope_"),
    ),
}

GLOBAL_EXECUTION_COLUMNS: Tuple[str, ...] = (
    "data_purity_percentage",
)


def find_station_directories(selected: Optional[Sequence[str]] = None) -> List[Path]:
    """Return station directories matching ``MINGO0*`` filtered by *selected*."""
    if not STATIONS_DIR.exists():
        raise FileNotFoundError(f"Stations directory not found: {STATIONS_DIR}")

    candidates = sorted(path for path in STATIONS_DIR.iterdir() if path.is_dir() and path.name.startswith("MINGO0"))
    if selected:
        target = {f"MINGO0{station}" for station in selected}
        candidates = [path for path in candidates if path.name in target]
    return candidates


def find_task_directories(station_dir: Path, tasks: Optional[Sequence[str]] = None) -> Iterator[Tuple[str, Path]]:
    """Yield ``(task_number, metadata_dir)`` available under *station_dir*."""
    base = station_dir / "STAGE_1" / "EVENT_DATA" / "STEP_1"
    if not base.exists():
        return

    for task_dir in sorted(path for path in base.iterdir() if path.is_dir() and path.name.startswith("TASK_")):
        task_number = task_dir.name.split("_", 1)[-1]
        if tasks and task_number not in tasks:
            continue
        metadata_dir = task_dir / "METADATA"
        if metadata_dir.exists():
            yield task_number, metadata_dir


_BASENAME_PATTERN = re.compile(r"(\d{11})$")


def basename_to_datetime(basename: object) -> Optional[dt.datetime]:
    """Convert the trailing 11 digits of *basename* into a ``datetime``."""
    if basename is None:
        return None
    if isinstance(basename, float) and np.isnan(basename):
        return None
    text = str(basename).strip()
    if not text:
        return None

    match = _BASENAME_PATTERN.search(text)
    if not match:
        return None

    digits = match.group(1)
    year = 2000 + int(digits[:2])
    day_of_year = int(digits[2:5])
    hh = int(digits[5:7])
    mm = int(digits[7:9])
    ss = int(digits[9:11])

    try:
        base = dt.datetime(year, 1, 1) + dt.timedelta(days=day_of_year - 1)
        return base.replace(hour=hh, minute=mm, second=ss)
    except ValueError:
        return None


def ensure_odd_window(window: int) -> int:
    """Return *window* adjusted to the closest odd >= 1."""
    if window < 1:
        return 1
    if window % 2 == 0:
        return window + 1
    return window


@dataclass
class ColumnResult:
    """Hold intermediate artefacts for a processed column."""

    times: pd.Series
    original: pd.Series
    smoothed: pd.Series
    residuals: pd.Series
    lower_quantile: float
    upper_quantile: float


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    """Return a centred rolling-median (horizontal median filter)."""
    return series.rolling(window=window, center=True, min_periods=1).median()


def resolve_columns_for_task(
    df: pd.DataFrame,
    task: str,
    explicit_columns: Optional[Sequence[str]],
) -> List[str]:
    """Return concrete column names to analyse for *task*."""
    columns: List[str] = []
    if explicit_columns:
        columns = [column for column in explicit_columns if column in df.columns]
    else:
        patterns = TASK_COLUMN_PATTERNS.get(task)
        if patterns:
            for column in df.columns:
                if any(pattern.match(column) for pattern in patterns):
                    columns.append(column)

    for column in GLOBAL_EXECUTION_COLUMNS:
        if column in df.columns and column not in columns:
            columns.append(column)

    return columns


def _parse_sequence(value: object) -> object:
    """Safely interpret textual sequences ``[a, b, ...]`` into Python lists."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return np.nan
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except (ValueError, SyntaxError):
                return np.nan
    return value


def expand_column_series(series: pd.Series, column_name: str) -> List[Tuple[str, pd.Series]]:
    """Return one or more numeric series derived from *series*."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty and series.dtype == object:
        sample = series.dropna().astype(str)
        if not sample.empty and sample.str.strip().str.startswith("[").any():
            parsed_values = [_parse_sequence(value) for value in series]
            max_len = max((len(item) for item in parsed_values if isinstance(item, (list, tuple))), default=0)
            if max_len == 0:
                return [(column_name, numeric)]

            descriptors: List[Tuple[str, pd.Series]] = []
            for idx in range(max_len):
                values = []
                for item in parsed_values:
                    if isinstance(item, (list, tuple)) and len(item) > idx:
                        values.append(item[idx])
                    else:
                        values.append(np.nan)
                descriptors.append(
                    (f"{column_name}[{idx}]", pd.Series(values, index=series.index, dtype=float))
                )
            return descriptors

    return [(column_name, numeric)]


def process_series(
    times: pd.Series,
    values: pd.Series,
    window: int,
) -> Optional[ColumnResult]:
    """Compute smoothed series and residuals for a 1-D numeric *values* series."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return None

    smoothed = rolling_median(numeric, window)
    residuals = numeric - smoothed

    residuals_clean = residuals.dropna()
    if residuals_clean.empty:
        return None

    lower = residuals_clean.quantile(0.01)
    upper = residuals_clean.quantile(0.99)

    return ColumnResult(
        times=times,
        original=numeric,
        smoothed=smoothed,
        residuals=residuals,
        lower_quantile=float(lower),
        upper_quantile=float(upper),
    )


def plot_column(
    pdf: PdfPages,
    column: str,
    station: str,
    task: str,
    result: ColumnResult,
) -> None:
    """Append plots for *column* to *pdf*."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    ax_series = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[1, 0], sharex=ax_series)
    ax_hist = fig.add_subplot(gs[:, 1])

    ax_series.plot(
        result.times,
        result.original,
        color="#1f77b4",
        linestyle="--",
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        label="Original",
    )
    ax_series.plot(
        result.times,
        result.smoothed,
        color="#ff7f0e",
        linestyle="--",
        marker="x",
        markersize=4,
        linewidth=0.9,
        label="Smoothed",
    )
    ax_series.set_ylabel(column)
    ax_series.set_title(f"{column} — station {station} task {task}")
    ax_series.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_series.legend(loc="upper right")

    ax_residual.plot(
        result.times,
        result.residuals,
        color="#2ca02c",
        linestyle="--",
        marker=".",
        markersize=3,
        linewidth=0.8,
    )
    ax_residual.axhline(0, color="k", linewidth=0.8)
    ax_residual.set_ylabel("Residual")
    ax_residual.set_xlabel("Acquisition timestamp")
    ax_residual.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    diffs = result.residuals.dropna().to_numpy()
    ax_hist.hist(diffs, bins="auto", color="#2ca02c", alpha=0.75, orientation="horizontal")
    ax_hist.axhline(0, color="k", linewidth=0.8)
    ax_hist.axhline(result.lower_quantile, color="red", linestyle="--", linewidth=1.0, label="1% quantile")
    ax_hist.axhline(result.upper_quantile, color="purple", linestyle="--", linewidth=1.0, label="99% quantile")
    ax_hist.set_xlabel("Count")
    ax_hist.set_ylabel("Residual (original - smoothed)")
    ax_hist.legend(loc="lower right")
    ax_hist.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.autofmt_xdate()
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10, hspace=0.35, wspace=0.32)
    pdf.savefig(fig)
    plt.close(fig)


def build_outlier_tables(
    outlier_data: Mapping[Tuple[str, str], Dict[str, Dict[str, float]]],
    columns_per_key: Mapping[Tuple[str, str], "OrderedDict[str, None]"],
    basename_col: str,
    time_col: str,
    output_dir: Path,
) -> List[Path]:
    """Persist compact CSV tables from *outlier_data*."""
    generated: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for (station, task), rows in outlier_data.items():
        if not rows:
            continue

        column_order = [basename_col, time_col]
        column_order.extend(f"{column}_smoothed" for column in columns_per_key[(station, task)])

        def row_sort_key(item: Tuple[str, Dict[str, float]]) -> Tuple[Optional[dt.datetime], str]:
            basename = item[1].get(basename_col, "")
            return basename_to_datetime(basename) or dt.datetime.min, basename

        data: List[List[Optional[float]]] = []
        for _, row_values in sorted(rows.items(), key=row_sort_key):
            data.append([row_values.get(column) for column in column_order])

        output_df = pd.DataFrame(data, columns=column_order)
        output_path = output_dir / f"reprocess_files_station_{station[-2:]}_task_{task}.csv"
        output_df.to_csv(output_path, index=False)
        generated.append(output_path)

    return generated


def analyse_metadata_file(
    csv_path: Path,
    task_number: str,
    explicit_columns: Optional[Sequence[str]],
    window: int,
    basename_col: str,
    time_col: str,
    enqueue_outliers: Dict[Tuple[str, str], Dict[str, Dict[str, float]]],
    columns_per_key: Dict[Tuple[str, str], "OrderedDict[str, None]"],
    station: str,
    plotting: Optional[PdfPages],
) -> None:
    """Analyse *csv_path*, updating *enqueue_outliers* and adding plots."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        print(f"Warning: malformed rows detected in {csv_path}, skipping invalid lines.")
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    if "analysis_mode" in df.columns:
        df = df[~df["analysis_mode"].astype(str).eq("1")].reset_index(drop=True)
    missing_basename = basename_col not in df.columns
    missing_time = time_col not in df.columns
    if missing_basename or missing_time:
        missing = [col for col, flag in ((basename_col, missing_basename), (time_col, missing_time)) if flag]
        raise KeyError(f"File {csv_path} is missing required columns: {', '.join(missing)}")

    requested_columns = resolve_columns_for_task(df, task_number, explicit_columns)
    if not requested_columns:
        return

    timestamps = df[basename_col].apply(basename_to_datetime)
    df = df.assign(__timestamp=timestamps)
    df = df[df["__timestamp"].notna()].sort_values("__timestamp").reset_index(drop=True)
    if df.empty:
        return

    descriptors: List[Tuple[str, pd.Series]] = []
    for column in requested_columns:
        descriptors.extend(expand_column_series(df[column], column))
    if not descriptors:
        return

    key = (station, task_number)
    outlier_bucket = enqueue_outliers[key]
    column_registry = columns_per_key.setdefault(key, OrderedDict())

    for column_label, series in descriptors:
        result = process_series(df["__timestamp"], series, window)
        if not result:
            continue

        residuals = result.residuals
        outlier_mask = (residuals < result.lower_quantile) | (residuals > result.upper_quantile)
        if plotting:
            plot_column(plotting, column_label, station[-2:], task_number, result)
        if not outlier_mask.any():
            continue

        column_registry.setdefault(column_label, None)

        smoothed = result.smoothed
        for idx in residuals.index[outlier_mask]:
            basename_value = df.at[idx, basename_col]
            row_entry = outlier_bucket.setdefault(
                basename_value,
                {
                    basename_col: basename_value,
                    time_col: df.at[idx, time_col],
                },
            )
            row_entry[f"{column_label}_smoothed"] = float(smoothed.at[idx]) if pd.notna(smoothed.at[idx]) else np.nan


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Explicit metadata columns to analyse for outliers (overrides task defaults).",
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        default=None,
        help="Limit analysis to the given station numbers (e.g. 1 2 4).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Limit analysis to the given task numbers (e.g. 1 3 5).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=41,
        help="Rolling window size for the median filter (odd number, default: 41).",
    )
    parser.add_argument(
        "--basename-column",
        default="filename_base",
        help="Column containing the acquisition basename (default: filename_base).",
    )
    parser.add_argument(
        "--time-column",
        default="execution_timestamp",
        help="Column containing the execution timestamp (default: execution_timestamp).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CSV_OUTPUT_DIR,
        help=f"Directory for the reprocessing CSVs (default: {CSV_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Generate PDF reports with the original/smoothed series and residual histograms.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "PLOTS",
        help="Directory where PDF plots are stored when --plot is enabled.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    window = ensure_odd_window(args.window)
    station_filter = args.stations
    task_filter = args.tasks
    columns = args.columns

    enqueue_outliers: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = defaultdict(dict)
    columns_per_key: Dict[Tuple[str, str], "OrderedDict[str, None]"] = {}

    stations = find_station_directories(station_filter)
    if not stations:
        raise RuntimeError("No station directories found to analyse.")

    for station_dir in stations:
        station_name = station_dir.name
        for task_number, metadata_dir in find_task_directories(station_dir, task_filter):
            csv_files = sorted(metadata_dir.glob(f"task_{task_number}_metadata_*.csv"))
            if not csv_files:
                continue

            pdf_path = None
            pdf_writer: Optional[PdfPages] = None
            if args.plot:
                args.plot_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = args.plot_dir / f"station_{station_name[-2:]}_task_{task_number}.pdf"
                pdf_writer = PdfPages(pdf_path)

            try:
                for csv_path in csv_files:
                    analyse_metadata_file(
                        csv_path=csv_path,
                        task_number=task_number,
                        explicit_columns=columns,
                        window=window,
                        basename_col=args.basename_column,
                        time_col=args.time_column,
                        enqueue_outliers=enqueue_outliers,
                        columns_per_key=columns_per_key,
                        station=station_name,
                        plotting=pdf_writer,
                    )
            finally:
                if pdf_writer:
                    pdf_writer.close()

    generated_csvs = build_outlier_tables(
        outlier_data=enqueue_outliers,
        columns_per_key=columns_per_key,
        basename_col=args.basename_column,
        time_col=args.time_column,
        output_dir=args.output_dir,
    )

    if generated_csvs:
        print("Generated CSVs:")
        for path in generated_csvs:
            print(f"  - {path}")
    else:
        print("No outliers detected with the given configuration.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
