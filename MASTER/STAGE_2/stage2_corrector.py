# %% [markdown]
"""
# Stage 2 quicklook helper

This notebook-like script loads LAB_LOGS and EVENT_DATA daily CSV files for a
given station (MINGO0x), merges the requested days, and draws quick plots for
events, temperature, and pressure. The figure files are stored under each
station's `STAGE_2` directory so every site keeps its own quicklook archive.
Run it as a regular script or import the `quicklook` function inside a notebook
for more ad-hoc exploration.
"""

# %%
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
from typing import Iterable, List, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

pd.plotting.register_matplotlib_converters()

# %%
DATAFLOW_ROOT = Path("/home/mingo/DATAFLOW_v3")
STATIONS_ROOT = DATAFLOW_ROOT / "STATIONS"


@dataclass(frozen=True)
class StationPaths:
    station: str
    station_root: Path
    lab_logs_base: Path
    event_data_base: Path
    stage2_output_base: Path


def normalize_station_name(station: str) -> str:
    token = station.strip().upper()
    if re.fullmatch(r"[1-4]", token):
        return f"MINGO0{token}"
    if re.fullmatch(r"MINGO0?[1-4]", token):
        digits = token[-1]
        return f"MINGO0{digits}"
    raise ValueError("Station must be 1-4 or like 'MINGO01'")


def build_station_paths(station: str) -> StationPaths:
    station_name = normalize_station_name(station)
    station_root = STATIONS_ROOT / station_name
    if not station_root.exists():
        raise ValueError(f"Station directory not found: {station_root}")
    stage1_root = station_root / "STAGE_1"
    lab_logs_base = stage1_root / "LAB_LOGS" / "STEP_2" / "OUTPUT_FILES"
    event_data_base = stage1_root / "EVENT_DATA" / "STEP_3" / "TASK_2" / "OUTPUT_FILES"
    stage2_output_base = station_root / "STAGE_2" / "QUICKLOOK_OUTPUTS"
    return StationPaths(
        station=station_name,
        station_root=station_root,
        lab_logs_base=lab_logs_base,
        event_data_base=event_data_base,
        stage2_output_base=stage2_output_base,
    )

DEFAULT_EVENT_COLUMNS = ["events"]
DEFAULT_TEMP_COLUMNS = [
    "sensors_ext_Temperature_ext",
    "sensors_int_Temperature_int",
]
DEFAULT_PRESSURE_COLUMNS = [
    "sensors_ext_Pressure_ext",
    "sensors_int_Pressure_int",
]


def _running_in_ipython() -> bool:
    """Detect IPython/Jupyter so we can skip CLI parsing in interactive sessions."""
    try:
        from IPython import get_ipython
    except Exception:
        return False
    return get_ipython() is not None


# %%
def parse_date(token: str) -> dt.date:
    """Accept YYYY-MM-DD, YYYY/MM/DD, or YYYYMMDD tokens."""
    token = token.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date '{token}'")


def expand_dates(
    date_value: str | None, start_value: str | None, end_value: str | None
) -> List[dt.date]:
    if date_value:
        start = end = parse_date(date_value)
    else:
        if not start_value:
            raise ValueError("Provide --date or --start-date")
        start = parse_date(start_value)
        end = parse_date(end_value) if end_value else start
    if start > end:
        start, end = end, start
    days: List[dt.date] = []
    current = start
    while current <= end:
        days.append(current)
        current += dt.timedelta(days=1)
    return days


def lab_logs_file(paths: StationPaths, day: dt.date) -> Path:
    return (
        paths.lab_logs_base
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"lab_logs_{day:%Y_%m_%d}.csv"
    )


def event_data_file(paths: StationPaths, day: dt.date) -> Path:
    return (
        paths.event_data_base
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"event_data_{day:%Y_%m_%d}.csv"
    )


def _load_daily_series(
    paths: Iterable[Path], *, comment: str | None = None
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            print(f"[WARN] missing file: {path}")
            continue
        df = pd.read_csv(path, comment=comment, low_memory=False)
        if "Time" not in df.columns:
            raise RuntimeError(f"'Time' column absent in {path}")
        df["Time"] = pd.to_datetime(df["Time"])
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("Time", inplace=True)
    return combined.reset_index(drop=True)


def load_lab_logs(days: Sequence[dt.date], paths: StationPaths) -> pd.DataFrame:
    files = [lab_logs_file(paths, day) for day in days]
    return _load_daily_series(files)


def load_event_data(days: Sequence[dt.date], paths: StationPaths) -> pd.DataFrame:
    files = [event_data_file(paths, day) for day in days]
    df = _load_daily_series(files, comment="#")
    if df.empty:
        return df
    if "events" not in df.columns:
        numeric_cols = [
            col for col in df.select_dtypes(include="number").columns if col != "Time"
        ]
        if numeric_cols:
            df["events"] = df[numeric_cols].sum(axis=1)
    return df


# %%
def _select_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [col for col in candidates if col in df.columns]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _print_column_catalog(
    df: pd.DataFrame, *, label: str, skip: Sequence[str] = ("Time", "source_file")
) -> None:
    if df.empty:
        print(f"[INFO] {label}: no data loaded")
        return
    candidates = [col for col in df.columns if col not in skip]
    if not candidates:
        print(f"[INFO] {label}: only metadata columns available")
        return
    print(f"[INFO] {label} columns ({len(candidates)} available):")
    joined = ", ".join(candidates)
    for line in textwrap.wrap(joined, width=100):
        print(f"  {line}")


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    title: str,
    ylabel: str,
) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.4",
        )
        return
    cols = _select_columns(df, columns)
    if not cols:
        ax.text(
            0.5,
            0.5,
            "columns not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.4",
        )
        return
    subset = df[["Time", *cols]].copy()
    subset["Time"] = pd.to_datetime(subset["Time"])
    times = subset["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    plotted = False
    for col in cols:
        values = _coerce_numeric(subset[col])
        if values.notna().any():
            mask = values.notna().to_numpy()
            ax.plot(times[mask], values[mask].to_numpy(), lw=1.1, label=col)
            plotted = True
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "no numeric data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.4",
        )
        return
    ax.xaxis_date()
    ax.legend(loc="upper left", frameon=False)


def plot_overview(
    events_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    *,
    event_columns: Sequence[str] = DEFAULT_EVENT_COLUMNS,
    temperature_columns: Sequence[str] = DEFAULT_TEMP_COLUMNS,
    pressure_columns: Sequence[str] = DEFAULT_PRESSURE_COLUMNS,
    title: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    _plot_panel(
        axes[0],
        events_df,
        event_columns,
        title="Event rates",
        ylabel="events / min",
    )
    _plot_panel(
        axes[1],
        lab_df,
        temperature_columns,
        title="Temperatures",
        ylabel="°C",
    )
    _plot_panel(
        axes[2],
        lab_df,
        pressure_columns,
        title="Pressures",
        ylabel="hPa",
    )
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle(title or "Station quicklook", fontsize=14)
    fig.tight_layout()
    return fig


# %%
def quicklook(
    *,
    date: str | None = None,
    start: str | None = None,
    end: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    station: str = "MINGO01",
    output_dir: str | Path | None = None,
    auto_save: bool = True,
    event_columns: Sequence[str] = DEFAULT_EVENT_COLUMNS,
    temperature_columns: Sequence[str] = DEFAULT_TEMP_COLUMNS,
    pressure_columns: Sequence[str] = DEFAULT_PRESSURE_COLUMNS,
) -> dict[str, object]:
    station_paths = build_station_paths(station)
    days = expand_dates(date, start, end)
    events_df = load_event_data(days, station_paths)
    lab_df = load_lab_logs(days, station_paths)
    _print_column_catalog(events_df, label=f"{station_paths.station} event data")
    _print_column_catalog(lab_df, label=f"{station_paths.station} lab logs")
    fig = plot_overview(
        events_df,
        lab_df,
        event_columns=event_columns,
        temperature_columns=temperature_columns,
        pressure_columns=pressure_columns,
        title=_build_title(station_paths.station, days),
    )
    save_path = _resolve_save_path(
        save,
        auto_save=auto_save,
        output_dir=output_dir,
        station_paths=station_paths,
        days=days,
    )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return {
        "events": events_df,
        "lab_logs": lab_df,
        "figure": fig,
        "save_path": save_path,
        "station": station_paths.station,
    }


def _build_title(station: str, days: Sequence[dt.date]) -> str:
    prefix = f"{station} quicklook"
    if not days:
        return prefix
    if len(days) == 1:
        return f"{prefix} — {days[0]:%Y-%m-%d}"
    return f"{prefix} — {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}"


def _default_output_filename(station: str, days: Sequence[dt.date]) -> str:
    if not days:
        return f"{station.lower()}_quicklook.png"
    if len(days) == 1:
        return f"{station.lower()}_quicklook_{days[0]:%Y%m%d}.png"
    return (
        f"{station.lower()}_quicklook_{days[0]:%Y%m%d}_{days[-1]:%Y%m%d}.png"
    )


def _resolve_save_path(
    save: str | Path | None,
    *,
    auto_save: bool,
    output_dir: str | Path | None,
    station_paths: StationPaths,
    days: Sequence[dt.date],
) -> Path | None:
    if save:
        return Path(save)
    if not auto_save:
        return None
    base = Path(output_dir) if output_dir else station_paths.stage2_output_base
    return base / _default_output_filename(station_paths.station, days)


# %%
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot event, temperature, and pressure data for a MINGO station. "
            "Use --date for a single day or --start-date/--end-date for a range."
        )
    )
    parser.add_argument("--date", help="Single day, e.g. 2025-01-31")
    parser.add_argument("--start-date", help="Range start date (inclusive)")
    parser.add_argument("--end-date", help="Range end date (inclusive)")
    parser.add_argument("--save", help="Path to save the figure as PNG")
    parser.add_argument(
        "--station",
        default="MINGO01",
        help="Station ID (default: MINGO01)",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Custom directory for outputs. Defaults to "
            "STATIONS/<station>/STAGE_2/QUICKLOOK_OUTPUTS"
        ),
    )
    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Skip automatic saving inside the station's STAGE_2 directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show(), useful when running headless",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = quicklook(
            date=args.date,
            start=args.start_date,
            end=args.end_date,
            show=not args.no_show,
            save=args.save,
            station=args.station,
            output_dir=args.output_dir,
            auto_save=not args.no_auto_save,
        )
    except ValueError as exc:
        parser.error(str(exc))
    save_path = result.get("save_path")
    if save_path:
        print(f"Figure saved to {save_path}")
    if args.no_show:
        events_rows = (
            len(result["events"]) if isinstance(result["events"], pd.DataFrame) else 0
        )
        lab_rows = (
            len(result["lab_logs"]) if isinstance(result["lab_logs"], pd.DataFrame) else 0
        )
        print(f"Loaded {events_rows} event rows and {lab_rows} lab rows")


if __name__ == "__main__" and not _running_in_ipython():
    main()


# %%

# Playground (edit this block as needed to experiment with custom columns).
# Flip PLAYGROUND_ENABLED to True (or copy the snippet into a notebook) and
# adjust the column lists for your analysis. Keep it False for normal CLI/imports
# to avoid unexpected executions.

PLAYGROUND_ENABLED = True

if PLAYGROUND_ENABLED:  # noqa: SIM115 - manual toggle; flip to True for ad-hoc work
    
    station = "1"
    
    result = quicklook(
        start="2025-10-01",
        end="2025-11-20",
        station=station,
        show=False,
        auto_save=False,
    )
    lab_logs_df = result["lab_logs"]
    events_df = result["events"]

    if lab_logs_df.empty:
        raise RuntimeError(
            "No lab log rows were loaded. Check the earlier [INFO]/[WARN] messages "
            "to confirm files exist for the chosen dates/station."
        )
    if events_df.empty:
        raise RuntimeError(
            "No event rows were loaded. Verify the date range and station."
        )


    def _sum_existing_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
        """Sum only the columns that actually exist, returning zeros if none do."""
        available = [col for col in columns if col in df.columns]
        if not available:
            return pd.Series(0.0, index=df.index, dtype=float)
        return df[available].sum(axis=1)

    def _sum_region_combo_columns(
        df: pd.DataFrame, combos: Sequence[str], region: str
    ) -> pd.Series:
        return _sum_existing_columns(df, [f"{combo}_{region}" for combo in combos])
    
    
    # Sum in 10 minute columns, not 1 minute ones
    events_df = events_df.set_index("Time").resample("120T").sum().reset_index()
    
    # Example: sum detector regions (12, 23, 34, etc.) and re-plot using those
    # derived columns. Edit the `regions` list to fit your detector layout.
    regions = ["12", "23", "34", "13", "24", "14", "123", "234", "124", "134", "1234"]
    region_sum_columns: dict[str, pd.Series] = {}
    for region in regions:
        pattern = rf"^{region}_"
        matches = events_df.filter(regex=pattern)
        if matches.empty:
            print(f"[INFO] No columns found for region {region}; skipping.")
            continue
        region_sum_columns[region] = matches.sum(axis=1)

    if region_sum_columns:
        region_sum_df = pd.DataFrame(region_sum_columns).reindex(events_df.index)
        events_df = pd.concat([events_df, region_sum_df], axis=1)
        result["events"] = events_df
        
    fig = plot_overview(
        events_df,
        lab_logs_df,
        event_columns=regions,
        temperature_columns=DEFAULT_TEMP_COLUMNS,
        pressure_columns=DEFAULT_PRESSURE_COLUMNS,
        title="Custom quicklook with region sums",
    )
    plt.show()
    
    
    # I want only the regions plots, not the temperature/pressure ones,
    # and i want in several different plots: one for 12, 23, 34;
    # other for 13, 24, 14, other for 123, 234. other for 124, 134, other for
    # 1234. Not the temperature/pressure ones. All in the same figure.
    region_groups = [
        ["12", "23", "34"],
        ["13", "24", "14"],
        ["123", "234"],
        ["124", "134"],
        ["1234"],
    ]
    
    # I want each group in a different subplot.
    fig, axes = plt.subplots(len(region_groups), 1, figsize=(14, 3 * len(region_groups)), sharex=True)
    for ax, region_group in zip(axes, region_groups):
        _plot_panel(
            ax,
            events_df,
            region_group,
            title=f"Regions: {', '.join(region_group)}",
            ylabel="events / min",
        )
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Custom quicklook with region sums", fontsize=14)
    fig.tight_layout()
    plt.show()
    
    
    # regions: "12", "23", "34" "13", "24", "14" "123", "234" "124", "134" "1234"
    
    # Plane by plane --------------------------------------------------------------------
    # plane_definitions = {
    #     1: {
    #         "detected": ["12", "13", "14", "123", "124", "134", "1234"],
    #         "passed": ["12", "23", "34", "13", "24", "14", "123", "124", "234", "134", "1234"],
    #     },
    #     2: {
    #         "detected": ["12", "23", "24", "123", "124", "234", "1234"],
    #         "passed": ["12", "23", "13", "24", "14", "123", "124", "234", "134", "1234"],
    #     },
    #     3: {
    #         "detected": ["34", "23", "13", "123", "134", "234", "1234"],
    #         "passed": ["23", "34", "13", "24", "14", "123", "124", "234", "134", "1234"],
    #     },
    #     4: {
    #         "detected": ["34", "24", "14", "124", "234", "134", "1234"],
    #         "passed": ["12", "23", "34", "13", "24", "14", "123", "124", "234", "134", "1234"],
    #     },
    # }
    
    plane_definitions = {
        1: {
            "detected": ["1234"],
            "passed": ["234", "1234"],
        },
        2: {
            "detected": ["1234"],
            "passed": ["134", "1234"],
        },
        3: {
            "detected": ["1234"],
            "passed": ["124", "1234"],
        },
        4: {
            "detected": ["1234"],
            "passed": ["123", "1234"],
        },
    }
    
    ang_regions = ["R0.0",
               "R1.0", "R1.1", "R1.2", "R1.3", "R1.4", "R1.5", "R1.6", "R1.7",
               "R2.0", "R2.1", "R2.2", "R2.3", "R2.4", "R2.5", "R2.6", "R2.7",
               "R3.0", "R3.1", "R3.2", "R3.3", "R3.4", "R3.5", "R3.6", "R3.7",
               ]

    angular_region_sum_columns: dict[str, pd.Series] = {}
    efficiency_columns: dict[str, pd.Series] = {}
    for region in ang_regions:
        pattern = rf"{region}"
        matches = events_df.filter(regex=pattern)
        if matches.empty:
            print(f"[INFO] No columns found for region {region}; skipping.")
            continue
        angular_region_sum_columns[region] = matches.sum(axis=1)
        
        for plane_id, plane_cfg in plane_definitions.items():
            region_detected = _sum_region_combo_columns(
                events_df, plane_cfg["detected"], region
            )
            region_passed = _sum_region_combo_columns(
                events_df, plane_cfg["passed"], region
            )
            denom = region_passed.where(region_passed != 0)
            efficiency_columns[f"{region}_eff_{plane_id}"] = region_detected.divide(denom)

    if angular_region_sum_columns:
        angular_region_sum_df = (
            pd.DataFrame(angular_region_sum_columns).reindex(events_df.index)
        )
        events_df = pd.concat([events_df, angular_region_sum_df], axis=1)

    if efficiency_columns:
        efficiency_df = pd.DataFrame(efficiency_columns).reindex(events_df.index)
        events_df = pd.concat([events_df, efficiency_df], axis=1)

    result["events"] = events_df
    
    
    # Plot the ang_regions in different subplots acccording to the first number after R
    region_groups = [
        [f"R0.{i}" for i in range(8)],
        [f"R1.{i}" for i in range(8)],
        [f"R2.{i}" for i in range(8)],
        [f"R3.{i}" for i in range(8)],
    ]
    fig, axes = plt.subplots(len(region_groups), 1, figsize=(14, 3 * len(region_groups)), sharex=True)
    for ax, region_group in zip(axes, region_groups):
        _plot_panel(
            ax,
            events_df,
            region_group,
            title=f"Regions: {', '.join(region_group)}",
            ylabel="events / min",
        )
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Custom quicklook with angular regions", fontsize=14)
    fig.tight_layout()
    plt.show()


    
    
    # Plot RX_eff_Y for X in [0.0, 1.0, 1.1, 1.2, ...] and Y in 1-4 in different subplots
    # In a superplot with 4 columns (one per plane) and as many rows as needed
    # to fit all the regions. Each subplot shows the efficiency for that region and plane.
    
    region_eff_columns = [col for col in events_df.columns if "_eff_" in col and "R" in col]
    
    region_names = sorted(set(col.split("_eff_")[0] for col in region_eff_columns))
    num_planes = 4
    num_regions = len(region_names)
    num_rows = num_regions
    
    # Create a list of the different first parts of the region names
    main_regions = sorted(set(name.split(".")[0] for name in region_names))
    print(f"Main regions: {main_regions}")
    
    # Associate a colour to each main region
    import matplotlib.cm as cm
    import numpy as np
    colormap = cm.get_cmap('viridis', len(main_regions))
    region_colors = {main_region: colormap(i) for i, main_region in enumerate(main_regions)}
    print(f"Region colors: {region_colors}")
    
    import numpy as np
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plane_combinations = ["12", "23", "34", "13", "24", "14", "123", "234", "124", "134", "1234"]
    num_combinations = len(plane_combinations)
    num_rows = len(region_names)
    
    
    complete_plot = False
    
    if complete_plot:
        print(f"Plotting {num_rows} rows and {num_combinations} columns for region event counts.")

        fig, axes = plt.subplots(
            num_rows,
            num_combinations,
            figsize=(4 * num_combinations, 3 * num_rows),
            sharex=True,
            sharey=True,
        )

        # Ensure axes is always 2D: shape = (num_rows, num_combinations)
        axes = np.atleast_2d(axes)

        for row_idx, region in enumerate(region_names):
            for col_idx, plane_combo in enumerate(plane_combinations):
                ax = axes[row_idx, col_idx]

                col_name = f"{plane_combo}_{region}"
                print(f"Plotting region {region} and plane combo {plane_combo} in column {col_name}")
                
                
                
                # Determine if the column exists
                if col_name not in events_df.columns:
                    print(f"  -> Column {col_name} not found in events_df. Turning axis off.")
                    ax.set_axis_off()
                    continue

                # Extract the time series, forcing numeric dtype so pandas.NA coerces to NaN
                y = pd.to_numeric(events_df[col_name], errors="coerce")

                if y.isna().all():
                    print(f"  -> Column {col_name} is all NaN. Turning axis off.")
                    ax.set_axis_off()
                    continue

                y_max = y.max()
                if not np.isfinite(y_max) or y_max <= 0:
                    print(f"  -> Column {col_name} has non-positive or non-finite max ({y_max}). Turning axis off.")
                    ax.set_axis_off()
                    continue

                # Plot background colour for this region
                main_region = region.split(".")[0]
                bg_color = region_colors.get(main_region, (0.9, 0.9, 0.9, 1.0))

                ax.axhspan(0, y_max, color=bg_color, alpha=0.2)

                # Plot the event counts curve
                ax.plot(events_df.index, y, linewidth=1.0, color="black")

                # Titles and labels
                if col_idx == 0:
                    ax.set_ylabel(region)
                if row_idx == 0:
                    ax.set_title(f"Combo {plane_combo}")

                ax.set_ylim(0, y_max * 1.1)

        # X labels only on the last row
        for ax in axes[-1, :]:
            ax.set_xlabel("Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

        fig.suptitle("Angular Region Event Counts by Plane Combination", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()
    
    
    
    fig, axes = plt.subplots(num_rows, num_planes, figsize=(4 * num_planes, 3 * num_rows), sharex=True, sharey=True)
    for row_idx, region in enumerate(region_names):
        for plane_id in range(1, num_planes + 1):
            ax = axes[row_idx, plane_id - 1] if num_rows > 1 else axes[plane_id - 1]
            col_name = f"{region}_eff_{plane_id}"
            
            # Determine if the column exists
            if col_name not in events_df.columns:
                ax.set_axis_off()
                continue

            # Extract the time series and coerce nullable ints to floats
            y = pd.to_numeric(events_df[col_name], errors="coerce")

            # Plot background colour for this region
            main_region = region.split(".")[0]
            bg_color = region_colors.get(main_region, (0.9, 0.9, 0.9, 1.0))

            ax.axhspan(0, 1, color=bg_color, alpha=0.2)

            # Plot the efficiency curve
            ax.plot(events_df.index, y, linewidth=1.0, color='black')

            # Titles and labels
            if plane_id == 1:
                ax.set_ylabel(region)
            if row_idx == 0:
                ax.set_title(f"Plane {plane_id}")

            ax.set_ylim(0, 1)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Angular Region Efficiencies by Plane", fontsize=14)
    fig.tight_layout()
    plt.show()
    
    
    
    correlation = False
    
    if correlation:
        # Study the correlation between the rate and the efficiencies
        fig, axes = plt.subplots(num_rows, num_planes, figsize=(4 * num_planes, 3 * num_rows), sharex=True)
        for row_idx, region in enumerate(region_names):
            for plane_id in range(1, num_planes + 1):
                ax = axes[row_idx, plane_id - 1] if num_rows > 1 else axes[plane_id - 1]
                rate_col = f"{region}"
                eff_col = f"{region}_eff_{plane_id}"
                
                # Determine if the columns exist
                if rate_col not in events_df.columns or eff_col not in events_df.columns:
                    ax.set_axis_off()
                    continue

                # Extract the time series
                rate_y = pd.to_numeric(events_df[rate_col], errors="coerce")
                eff_y = pd.to_numeric(events_df[eff_col], errors="coerce")

                # Scatter plot of efficiency vs rate
                ax.scatter(rate_y, eff_y, s=5, color='black', alpha=0.7)

                # Titles and labels
                if plane_id == 1:
                    ax.set_ylabel(region)
                if row_idx == 0:
                    ax.set_title(f"Plane {plane_id}")

                ax.set_xlabel("Event Rate")
                ax.set_ylabel("Efficiency")
                ax.set_ylim(0, 1)
        fig.suptitle("Angular Region Efficiency vs Event Rate", fontsize=14)
        fig.tight_layout()
        plt.show()
    
    
    
    # Correct the rate using the efficiency: rate_corrected = rate / efficiency
    
    new_rate_columns: dict[str, pd.Series] = {}
    for region in region_names:
        rate_col = f"{region}"
        eff_cols = [f"{region}_eff_{plane_id}" for plane_id in range(1, num_planes + 1)]
        
        # Determine if the columns exist
        if rate_col not in events_df.columns or any(col not in events_df.columns for col in eff_cols):
            continue

        # Extract the time series
        rate_y = pd.to_numeric(events_df[rate_col], errors="coerce")
        eff_y = pd.Series(1.0, index=events_df.index)
        for col in eff_cols:
            col_values = pd.to_numeric(events_df[col], errors="coerce")
            eff_y = eff_y.multiply(col_values, fill_value=1.0)

        # Avoid division by zero
        eff_y = eff_y.replace(0, pd.NA)

        # Corrected rate
        corrected_rate = rate_y.divide(eff_y)
        new_rate_columns[f"{region}_original_rate"] = rate_y
        new_rate_columns[f"{region}_corrected_rate"] = corrected_rate

    if new_rate_columns:
        new_rate_df = pd.DataFrame(new_rate_columns).reindex(events_df.index)
        events_df = pd.concat([events_df, new_rate_df], axis=1)
        result["events"] = events_df
        
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows), sharex=True)
    for row_idx, region in enumerate(region_names):
        ax = axes[row_idx] if num_rows > 1 else axes
        col_name = f"{region}_corrected_rate"
        
        # Determine if the column exists
        if col_name not in events_df.columns:
            ax.set_axis_off()
            continue

        # Extract the time series
        y = pd.to_numeric(events_df[col_name], errors="coerce")

        original_col_name = f"{region}_original_rate"
        y_original = pd.to_numeric(events_df[original_col_name], errors="coerce")

        # Plot the corrected rate curve
        ax.plot(events_df.index, y, linewidth=1.0, color='black')
        ax.plot(events_df.index, y_original, linewidth=1.0, color='blue')
        # Titles and labels
        ax.set_ylabel(region)
        if row_idx == 0:
            ax.set_title("Corrected Event Rates by Region")
    for ax in axes[-1:]:
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Corrected Event Rates by Region", fontsize=14)
    fig.tight_layout()
    plt.show()
    
    
    #%%
    
    # Now i want you to do the following: calculate the average per angular region
    # of corrected rate. Plot that average as a horizontal line in each of the previous plots.
    # Then calculate the difference in % between the corrected rate and that average,
    # and plot that difference in a new figure with subplots similar to the previous ones.
    # The difference should be (corrected_rate - average) / average * 100.
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows), sharex=True)
    for row_idx, region in enumerate(region_names):
        ax = axes[row_idx] if num_rows > 1 else axes
        col_name = f"{region}_corrected_rate"
        
        # Determine if the column exists
        if col_name not in events_df.columns:
            ax.set_axis_off()
            continue

        # Extract the time series
        y = pd.to_numeric(events_df[col_name], errors="coerce")
        
        average = y[y > 0].mean()
        
        # Plot the corrected rate curve
        ax.plot(events_df.index, y, linewidth=1.0, color='black')
        ax.axhline(average, color='red', linestyle='--', label='Average')
        
        # Titles and labels
        ax.set_ylabel(region)
        if row_idx == 0:
            ax.set_title("Corrected Event Rates with Average by Region")
            ax.legend()
    for ax in axes[-1:]:
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Corrected Event Rates with Average by Region", fontsize=14)
    fig.tight_layout()
    plt.show()
    
    #%%
    
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 3 * num_rows), sharex=True)
    for row_idx, region in enumerate(region_names):
        ax = axes[row_idx] if num_rows > 1 else axes
        col_name = f"{region}_corrected_rate"
        
        # Determine if the column exists
        if col_name not in events_df.columns:
            ax.set_axis_off()
            continue

        # Extract the time series
        y = pd.to_numeric(events_df[col_name], errors="coerce")
        
        average = y[y > 0].mean()
        
        # Calculate the percentage difference
        percent_diff = (y - average) / average * 100
        
        # Plot the percentage difference curve
        ax.plot(events_df.index, percent_diff, linewidth=1.0, color='black')
        
        # Ylimit between -100% and +100%
        ax.set_ylim(-50, 50)
        
        # Titles and labels
        ax.set_ylabel(region)
        if row_idx == 0:
            ax.set_title("Percentage Difference from Average Corrected Rate by Region")
    for ax in axes[-1:]:
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Percentage Difference from Average Corrected Rate by Region", fontsize=14)
    fig.tight_layout()
    plt.show()
    
    
    #%%
    
    # Now represent in a polar plot, in a GIF, those differences in %, according to the following
    # scheme:
    
    
    print("----------------------- Drawing angular regions ----------------------")
    

    import sys
    from pathlib import Path
    save_plots = False
    show_plots = True

    CURRENT_PATH = Path(__file__).resolve()
    REPO_ROOT = None
    for parent in CURRENT_PATH.parents:
        if parent.name == "MASTER":
            REPO_ROOT = parent.parent
            break
    if REPO_ROOT is None:
        REPO_ROOT = CURRENT_PATH.parents[-1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    
    
    from MASTER.common.config_loader import update_config_with_parameters
    from MASTER.common.execution_logger import set_station, start_timer 
    import os
    
    import yaml
    from ast import literal_eval
    
    start_timer(__file__)
    user_home = os.path.expanduser("~")
    config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
    parameter_config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters.csv")
    print(f"Using config file: {config_file_path}")
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    try:
        config = update_config_with_parameters(config, parameter_config_file_path, station)
    except NameError:
        pass
    home_path = config["home_path"]
    REFERENCE_TABLES_DIR = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "CONFIG_FILES" / "METADATA_REPRISE" / "REFERENCE_TABLES"


    
    
    def _coerce_numeric_sequence(raw_value, caster):
        """Return a list of numbers parsed from *raw_value*."""
        if isinstance(raw_value, (list, tuple, np.ndarray)):
            result: List[float] = []
            for item in raw_value:
                result.extend(_coerce_numeric_sequence(item, caster))
            return result
        if isinstance(raw_value, str):
            cleaned = raw_value.strip()
            if not cleaned:
                return []
            try:
                parsed = literal_eval(cleaned)
            except (ValueError, SyntaxError):
                cleaned = cleaned.replace("[", " ").replace("]", " ")
                tokens = [tok for tok in re.split(r"[;,\\s]+", cleaned) if tok]
                result = []
                for tok in tokens:
                    try:
                        result.append(caster(tok))
                    except (ValueError, TypeError):
                        continue
                return result
            else:
                return _coerce_numeric_sequence(parsed, caster)
        if np.isscalar(raw_value):
            try:
                return [caster(raw_value)]
            except (ValueError, TypeError):
                return []
        return []
    
    theta_boundaries_raw = config.get("theta_boundaries", [])
    region_layout_raw = config.get("region_layout", [])

    theta_boundaries = _coerce_numeric_sequence(theta_boundaries_raw, float)
    theta_values = []
    for b in theta_boundaries:
        if isinstance(b, (int, float)) and np.isfinite(b):
            b_float = float(b)
            if 0 <= b_float <= 90 and b_float not in theta_values:
                theta_values.append(b_float)
    theta_boundaries = theta_values

    region_layout = _coerce_numeric_sequence(region_layout_raw, int)
    region_layout = [max(1, int(abs(n))) for n in region_layout if isinstance(n, (int, float))]

    expected_regions = len(theta_boundaries) + 1
    if not region_layout:
        region_layout = [1] * expected_regions
    elif len(region_layout) < expected_regions:
        region_layout = region_layout + [region_layout[-1]] * (expected_regions - len(region_layout))
    elif len(region_layout) > expected_regions:
        region_layout = region_layout[:expected_regions]

    if not theta_boundaries:
        theta_boundaries = []


    print(f"Theta boundaries (degrees): {theta_boundaries}")

    
    
    # Input parameters
    theta_right_limit = np.pi / 2.5
    
    
    # Compute angular boundaries
    max_deg = np.degrees(theta_right_limit)
    valid_boundaries = [b for b in theta_boundaries if b <= max_deg]
    all_bounds_deg = [0] + valid_boundaries + [max_deg]
    radii = np.radians(all_bounds_deg)
    
    # Initialize plot
    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 8))
    ax.set_facecolor(plt.cm.viridis(0.0))
    ax.set_title("Region Labels for Specified Angular Segmentation", color='white')
    ax.set_theta_zero_location('N')

    # Draw concentric θ boundaries (including outermost)
    for r in radii[1:]:
        ax.plot(np.linspace(0, 2 * np.pi, 1000), [r] * 1000,
                color='white', linestyle='--', linewidth=3)

    # Draw radial (φ) separators for each region layout
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii) - 1])):
        if n_phi > 1:
            delta_phi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi = j * delta_phi
                ax.plot([phi, phi], [r0, r1], color='white', linestyle='--', linewidth=1.5)

    # Annotate region labels
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii) - 1])):
        r_label = (r0 + r1) / 2
        if n_phi == 1:
            ax.text(0, r_label, f'R{i}.0', ha='center', va='center',
                    color='white', fontsize=10, weight='bold')
        else:
            dphi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi_label = (j + 0.5) * dphi
                ax.text(phi_label, r_label, f'R{i}.{j}', ha='center', va='center',
                        rotation=0, rotation_mode='anchor',
                        color='white', fontsize=10, weight='bold')

    # Add radius labels slightly *outside* the outermost circle for clarity
    for r_deg in all_bounds_deg[1:]:
        r_rad = np.radians(r_deg)
        ax.text(np.pi + 0.09, r_rad - 0.05, f'{int(round(r_deg))}°', ha='center', va='bottom',
                color='white', fontsize=10, alpha=0.9)

    ax.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_yticklabels([])

    # Final layout
    title = "Region Labels for Specified Angular Segmentation"
    ax.set_ylim(0, theta_right_limit)
    plt.suptitle(title, fontsize=16, color='white')
    plt.tight_layout()
    plt.show()
    
    
    
    
    #%%
    
    
    
    # Now represent in a polar plot, in a GIF, those differences in %, according to the following
    # scheme:
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Wedge
    # import imageio.v2 as imageio
    from pathlib import Path
    # import imageio.v2 as imageio
    from PIL import Image
    from pathlib import Path

    
    
    # ----------------------------------------------------------------------
    # 1) Compute and store percentage differences per region in events_df
    # ----------------------------------------------------------------------
    pct_diff_cols = []
    for region in region_names:
        col_name = f"{region}_corrected_rate"
        if col_name not in events_df.columns:
            continue

        y = pd.to_numeric(events_df[col_name], errors="coerce")
        avg = y[y > 0].mean()
        if not np.isfinite(avg) or avg == 0:
            # Skip regions with ill-defined average
            continue

        diff_col = f"{region}_pct_diff"
        events_df[diff_col] = (y - avg) / avg * 100.0
        pct_diff_cols.append(diff_col)

    if not pct_diff_cols:
        raise RuntimeError("No percentage-difference columns could be computed. "
                        "Check that *_corrected_rate columns exist and are non-zero.")

    # ----------------------------------------------------------------------
    # 2) Prepare angular segmentation for polar plotting (same as above)
    # ----------------------------------------------------------------------
    # theta_boundaries, region_layout, theta_right_limit, radii have already been
    # defined in your previous block. Here we only ensure 'radii' is available.

    max_deg = np.degrees(theta_right_limit)
    valid_boundaries = [b for b in theta_boundaries if b <= max_deg]
    all_bounds_deg = [0] + valid_boundaries + [max_deg]
    radii = np.radians(all_bounds_deg)

    # Convenience: map region name "R<i>.<j>" → (ring_index=i, sector_index=j)
    def parse_region_label(region_str: str):
        """Return (ring_index, sector_index) from a label like 'R2.3'."""
        try:
            core = region_str.lstrip("R")
            ring_str, sector_str = core.split(".")
            return int(ring_str), int(sector_str)
        except Exception:
            return None, None

    # ----------------------------------------------------------------------
    # 3) Build GIF frames: one polar heat map per time bin
    # ----------------------------------------------------------------------
    # Normalization for color scale: e.g. ±50 % around zero
    vmin, vmax = -50.0, 50.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.coolwarm

    frames = []

    # Choose where to save the GIF
    gif_output_dir = Path(home_path) / "DATAFLOW_v3" / "MASTER" / "OUTPUT"
    gif_output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_output_dir / "angular_regions_pct_difference.gif"

    for t_idx, (ts, row) in enumerate(events_df.iterrows()):
        # Create polar figure
        fig, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(6, 6))
        ax.set_theta_zero_location("N")
        ax.set_ylim(0, theta_right_limit)
        ax.set_xticks([])  # Hide angle ticks if desired
        ax.set_yticklabels([])

        # Optionally, set a neutral background
        ax.set_facecolor("black")

        # Draw each ring and sector as a colored wedge according to pct difference
        for ring_idx, (r0, r1, n_phi) in enumerate(
            zip(radii[:-1], radii[1:], region_layout[: len(radii) - 1])
        ):
            if n_phi < 1:
                continue

            dphi = 2 * np.pi / n_phi
            for sector_idx in range(n_phi):
                region_label = f"R{ring_idx}.{sector_idx}"
                diff_col = f"{region_label}_pct_diff"

                if diff_col in events_df.columns:
                    value = row[diff_col]
                else:
                    value = np.nan

                if np.isfinite(value):
                    color = cmap(norm(value))
                else:
                    # Transparent / very faint if no data
                    color = (0.3, 0.3, 0.3, 0.1)

                theta1 = np.degrees(sector_idx * dphi)
                theta2 = np.degrees((sector_idx + 1) * dphi)

                wedge = Wedge(
                    (0.0, 0.0),
                    r1,
                    theta1,
                    theta2,
                    width=(r1 - r0),
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.add_patch(wedge)

        # Add a timestamp and title
        ax.set_title(
            f"Δ rate / average [%] at {ts.strftime('%Y-%m-%d %H:%M')}",
            fontsize=10,
            color="white",
            pad=20,
        )

        # Optional color bar on the side (once per frame)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("Δ rate / average [%]")

        # Convert figure to RGB array for GIF
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(image)
        plt.close(fig)

    # ----------------------------------------------------------------------
    # 4) Save GIF
    # ----------------------------------------------------------------------
    # Duration in seconds per frame; adjust as needed
    frame_duration = 0.4
    # imageio.mimsave(gif_path, frames, duration=frame_duration)
    
    # imageio.mimsave(gif_path, frames, duration=frame_duration)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(frame_duration * 1000),
        loop=0
    )

    
    print(f"Saved angular percentage-difference GIF to: {gif_path}")

    
    
    
# %%
