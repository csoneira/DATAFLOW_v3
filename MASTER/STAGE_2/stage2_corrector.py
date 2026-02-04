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
import numpy as np
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
    if re.fullmatch(r"[0-4]", token):
        return f"MINGO0{token}"
    if re.fullmatch(r"MINGO0?[0-4]", token):
        digits = token[-1]
        return f"MINGO0{digits}"
    raise ValueError("Station must be 0-4 or like 'MINGO00'")


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
    if isinstance(series, pd.DataFrame):
        # Try to squeeze single-column frames; otherwise bail out gracefully
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            series = series.stack()
    arr = np.asarray(series)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    coerced = pd.to_numeric(arr, errors="coerce")
    idx = getattr(series, "index", None)
    if idx is not None and len(idx) == len(coerced):
        return pd.Series(coerced, index=idx)
    return pd.Series(coerced)


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
        col_values = subset[col]
        if isinstance(col_values, pd.DataFrame):
            if col_values.shape[1] == 1:
                col_values = col_values.iloc[:, 0]
            else:
                print(f"[INFO] Skipping column group {col}: expected 1D data.")
                continue
        values = _coerce_numeric(col_values)
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
    
    station = "2"
    
    result = quicklook(
        start="2025-08-10",
        end="2025-11-04",
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
    events_df = events_df.set_index("Time").resample("10T").sum().reset_index()
    
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
    

    #%%

    # Eff 1 is the ( 1 - events_df['234'] ) / events_df['1234']
    if '1234' in events_df.columns and '234' in events_df.columns:
        events_df["eff1"] = (1 - events_df['234'] / events_df['1234']).replace([np.inf, -np.inf], np.nan)
    if '1234' in events_df.columns and '134' in events_df.columns:
        events_df["eff2"] = (1 - events_df['134'] / events_df['1234']).replace([np.inf, -np.inf], np.nan)
    if '1234' in events_df.columns and '124' in events_df.columns:
        events_df["eff3"] = (1 - events_df['124'] / events_df['1234']).replace([np.inf, -np.inf], np.nan)
    if '1234' in events_df.columns and '123' in events_df.columns:
        events_df["eff4"] = (1 - events_df['123'] / events_df['1234']).replace([np.inf, -np.inf], np.nan)
    
    # Plot 5 plots, one for each efficiency and one for "events"
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    _plot_panel(
        axes[0],
        events_df,
        ["events"],
        title="Event rates",
        ylabel="events / 2h",
    )
    _plot_panel(
        axes[1],
        events_df,
        ["eff1"],
        title="Efficiency 1",
        ylabel="eff1",
    )
    _plot_panel(
        axes[2],
        events_df,
        ["eff2"],
        title="Efficiency 2",
        ylabel="eff2",
    )
    _plot_panel(
        axes[3],
        events_df,
        ["eff3"],
        title="Efficiency 3",
        ylabel="eff3",
    )
    _plot_panel(
        axes[4],
        events_df,
        ["eff4"],
        title="Efficiency 4",
        ylabel="eff4",
    )
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Station custom quicklook with efficiencies", fontsize=14)
    fig.tight_layout()
    plt.show()

    #%%

    # Now i want you to detrend events against a polynomic nth grade combination of the efficiencies
    # For example, a 2nd grade polynomial would be:
    # events ~ a0 + a1*eff1 + a2*eff2 + a3*eff3 + a4*eff4 + a5*eff1^2 + a6*eff2^2 + a7*eff3^2 + a8*eff4^2 + a9*eff1*eff2 + a10*eff1*eff3 + ...
    # You can choose the degree of the polynomial (e.g., 2 for quadratic, 3 for cubic, etc.)
    # Use numpy or scipy to perform the polynomial fitting and detrending.
    # After detrending, plot the original events and the detrended events for comparison.
    from itertools import combinations_with_replacement
    def detrend_polynomial(
        df: pd.DataFrame,
        *,
        y_col: str,
        x_cols: Sequence[str],
        degree: int = 2,
        label: str = "polynomial",
        quantile: float = 0.99,
    ) -> pd.Series:
        """Remove polynomial dependence on multiple predictors at once."""
        numeric = df[[y_col, *x_cols]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if subset.empty:
            print(f"[INFO] No data to fit {label}; returning original series.")
            return numeric[y_col]

        trimmed = subset
        for col in subset.columns:
            lo, hi = subset[col].quantile([1 - quantile, quantile])
            trimmed = trimmed[trimmed[col].between(lo, hi)]
        if trimmed.empty:
            trimmed = subset

        y_trim = trimmed[y_col].to_numpy()
        
        # Build design matrix with polynomial terms
        X_terms = []
        for deg in range(1, degree + 1):
            for combo in combinations_with_replacement(x_cols, deg):
                term = np.prod([trimmed[col].to_numpy() for col in combo], axis=0)
                X_terms.append(term)
        X_design = np.column_stack([np.ones(len(trimmed)), *X_terms])
        
        coef, _, _, _ = np.linalg.lstsq(X_design, y_trim, rcond=None)

        y_pred_trim = X_design @ coef
        ss_res = float(np.sum((y_trim - y_pred_trim) ** 2))
        ss_tot = float(np.sum((y_trim - y_trim.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        print(
            f"[INFO] {label} fit: degree={degree}, r2={r2:.3f}, n={len(trimmed)}"
        )

        corrected = numeric[y_col].copy()
        valid_mask = numeric.notna().all(axis=1)
        if valid_mask.any():
            X_all_terms = []
            for deg in range(1, degree + 1):
                for combo in combinations_with_replacement(x_cols, deg):
                    term = np.prod(
                        [numeric.loc[valid_mask, col].to_numpy() for col in combo],
                        axis=0,
                    )
                    X_all_terms.append(term)
            X_all_design = np.column_stack([np.ones(len(X_all_terms[0])), *X_all_terms])
            predicted = X_all_design @ coef
            corrected.loc[valid_mask] = numeric.loc[valid_mask, y_col] - predicted
        return corrected
    detrended_events = detrend_polynomial(
        events_df,
        y_col="events",
        x_cols=["eff1", "eff2", "eff3", "eff4"],
        degree=2,
        label="Polynomial detrending",
    )

    

    # Plot original and detrended events
    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, events_df["events"], lw=1.1, label="Original Events")
    ax.plot(times, detrended_events, lw=1.1, label="Detrended Events")
    ax.set_title("Original vs Detrended Events")
    ax.set_ylabel("events / 2h")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()

    #%%

    events_df["detrended_events"] = detrended_events

    # Scatter/correction helpers to keep the plots compact and readable
    import numpy as np
    from scipy import optimize

    def merge_events_and_logs(
        events: pd.DataFrame, lab_logs: pd.DataFrame, tolerance: str = "5min"
    ) -> pd.DataFrame:
        if events.empty or lab_logs.empty:
            print("[INFO] Skipping scatter plots: missing events or lab logs.")
            return pd.DataFrame()
        return pd.merge_asof(
            events.sort_values("Time"),
            lab_logs.sort_values("Time"),
            on="Time",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )

    def plot_scatter_pairs(
        merged: pd.DataFrame,
        y_col: str,
        pairs: list[tuple[str, str, str, str]],
        *,
        figsize: tuple[int, int] = (12, 5),
    ) -> None:
        """pairs: (x_col, xlabel, title, color)."""
        pairs = [(x, xl, tt, c) for x, xl, tt, c in pairs if x in merged.columns]
        if not pairs or y_col not in merged:
            print(f"[INFO] Skipping scatter for {y_col}: columns not found.")
            return
        fig, axes = plt.subplots(1, len(pairs), figsize=figsize)
        axes = [axes] if len(pairs) == 1 else axes
        for ax, (x_col, xlabel, title, color) in zip(axes, pairs):
            ax.scatter(merged[x_col], merged[y_col], alpha=0.5, color=color, s=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(y_col.replace("_", " ").title())
            ax.set_title(title)
        fig.tight_layout()
        plt.show()
    

    # I want you to create scatter plots of the detrended events against eff1, eff2, eff3, and eff4
    merged_df = merge_events_and_logs(events_df, lab_logs_df, tolerance="10min")
    plot_scatter_pairs(
        merged_df,
        y_col="events",
        pairs=[
            ("eff1", "Efficiency 1", "Original Events vs Eff1", "blue"),
            ("eff2", "Efficiency 2", "Original Events vs Eff2", "orange"),
            ("eff3", "Efficiency 3", "Original Events vs Eff3", "green"),
            ("eff4", "Efficiency 4", "Original Events vs Eff4", "red"),
        ],
        figsize=(16, 4),
    )
    
    plot_scatter_pairs(
        merged_df,
        y_col="detrended_events",
        pairs=[
            ("eff1", "Efficiency 1", "Detrended Events vs Eff1", "blue"),
            ("eff2", "Efficiency 2", "Detrended Events vs Eff2", "orange"),
            ("eff3", "Efficiency 3", "Detrended Events vs Eff3", "green"),
            ("eff4", "Efficiency 4", "Detrended Events vs Eff4", "red"),
        ],
        figsize=(16, 4),
    )

    #%%

    # More things: give me the variables fitted in the detrending step
    # i want the coefficients of the polynomial fit used in the detrending step
    # and their uncertainties if possible.





    #%%
    
    
    
    
    
    

    def detrend_multilinear(
        df: pd.DataFrame,
        *,
        y_col: str,
        x_cols: Sequence[str],
        label: str = "multilinear",
        quantile: float = 0.99,
    ) -> pd.Series:
        """Remove linear dependence on multiple predictors at once."""
        numeric = df[[y_col, *x_cols]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if subset.empty:
            print(f"[INFO] No data to fit {label}; returning original series.")
            return numeric[y_col]

        trimmed = subset
        for col in subset.columns:
            lo, hi = subset[col].quantile([1 - quantile, quantile])
            trimmed = trimmed[trimmed[col].between(lo, hi)]
        if trimmed.empty:
            trimmed = subset

        X_trim = trimmed[x_cols].to_numpy()
        y_trim = trimmed[y_col].to_numpy()
        X_design = np.column_stack([np.ones(len(trimmed)), X_trim])
        coef, _, _, _ = np.linalg.lstsq(X_design, y_trim, rcond=None)
        intercept = coef[0]
        slopes = coef[1:]

        y_pred_trim = X_design @ coef
        ss_res = float(np.sum((y_trim - y_pred_trim) ** 2))
        ss_tot = float(np.sum((y_trim - y_trim.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        slope_str = ", ".join(f"{name}={val:.4g}" for name, val in zip(x_cols, slopes))
        print(
            f"[INFO] {label} fit: intercept={intercept:.4g}, {slope_str}, "
            f"r2={r2:.3f}, n={len(trimmed)}"
        )

        corrected = numeric[y_col].copy()
        valid_mask = numeric.notna().all(axis=1)
        if valid_mask.any():
            X_all = numeric.loc[valid_mask, x_cols].to_numpy()
            X_all_design = np.column_stack([np.ones(len(X_all)), X_all])
            predicted = X_all_design @ coef
            corrected.loc[valid_mask] = numeric.loc[valid_mask, y_col] - predicted
        return corrected

    def detrend_temp_exp_pressure(
        df: pd.DataFrame,
        *,
        y_col: str,
        temp_col: str,
        pressure_col: str,
        label: str = "Temp + exp(Pressure)",
        quantile: float = 0.99,
    ) -> pd.Series:
        """Fit temperature linearly and pressure exponentially, return residual."""
        numeric = df[[y_col, temp_col, pressure_col]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if subset.empty:
            print(f"[INFO] No data to fit {label}; returning original series.")
            return numeric[y_col]

        trimmed = subset
        for col in subset.columns:
            lo, hi = subset[col].quantile([1 - quantile, quantile])
            trimmed = trimmed[trimmed[col].between(lo, hi)]
        if trimmed.empty:
            trimmed = subset

        y_trim = trimmed[y_col].to_numpy()
        temps = trimmed[temp_col].to_numpy()
        pressures = trimmed[pressure_col].to_numpy()
        press_center = np.median(pressures)
        temp_center = np.median(temps)

        design = np.column_stack([np.ones(len(trimmed)), temps, pressures - press_center])
        coef_lin, _, _, _ = np.linalg.lstsq(design, y_trim, rcond=None)
        intercept_init, temp_slope_init, pressure_slope_init = coef_lin
        k_init = 1.0 / max(np.std(pressures), 1.0)
        amp_init = pressure_slope_init / max(k_init, 1e-6)

        def model(X, intercept, temp_slope, amp, k):
            temp, pressure = X
            return intercept + temp_slope * temp + amp * (np.exp(k * (pressure - press_center) ) - 1.0)

        try:
            popt, _ = optimize.curve_fit(
                model,
                (temps, pressures),
                y_trim,
                p0=[intercept_init, temp_slope_init, amp_init, k_init],
                maxfev=10000,
                bounds=([-np.inf, -np.inf, -np.inf, -0.05], [np.inf, np.inf, np.inf, 0.05]),
            )
        except Exception as exc:
            print(f"[INFO] {label} exp fit failed ({exc}); falling back to linear warm start.")
            popt = [intercept_init, temp_slope_init, amp_init, k_init]

        intercept_hat, temp_slope_hat, amp_hat, k_hat = popt
        approx_pressure_slope = amp_hat * k_hat
        y_pred_trim = model((temps, pressures), *popt)
        ss_res = float(np.sum((y_trim - y_pred_trim) ** 2))
        ss_tot = float(np.sum((y_trim - y_trim.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        print(
            f"[INFO] {label} exp fit: intercept={intercept_hat:.4g}, temp_slope={temp_slope_hat:.4g}, "
            f"amp={amp_hat:.4g}, k={k_hat:.4g}, linearized_pressure_slope≈{approx_pressure_slope:.4g}, "
            f"r2={r2:.3f}, n={len(trimmed)}"
        )

        corrected = numeric[y_col].copy()
        valid_mask = numeric.notna().all(axis=1)
        if valid_mask.any():
            temps_all = numeric.loc[valid_mask, temp_col].to_numpy()
            pressures_all = numeric.loc[valid_mask, pressure_col].to_numpy()
            predicted = model((temps_all, pressures_all), *popt)
            corrected.loc[valid_mask] = numeric.loc[valid_mask, y_col] - predicted
        return corrected

    def detrend_poly3_temp_pressure(
        df: pd.DataFrame,
        *,
        y_col: str,
        temp_col: str,
        pressure_col: str,
        label: str = "Temp + Pressure poly3",
        quantile: float = 0.99,
    ) -> pd.Series:
        """Fit a full 2D polynomial (temp, pressure) up to degree 3; return residual."""
        numeric = df[[y_col, temp_col, pressure_col]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if subset.empty:
            print(f"[INFO] No data to fit {label}; returning original series.")
            return numeric[y_col]

        trimmed = subset
        for col in subset.columns:
            lo, hi = subset[col].quantile([1 - quantile, quantile])
            trimmed = trimmed[trimmed[col].between(lo, hi)]
        if trimmed.empty:
            trimmed = subset

        y_trim = trimmed[y_col].to_numpy()
        temps = trimmed[temp_col].to_numpy()
        pressures = trimmed[pressure_col].to_numpy()
        temp_center = temps.mean()
        pressure_center = pressures.mean()
        t = temps - temp_center
        p = pressures - pressure_center

        # Build polynomial features up to total degree 3
        feats: list[np.ndarray] = [np.ones(len(trimmed))]
        terms = []
        for i in range(1, 4):  # degree 1..3
            for j in range(i + 1):
                power_t = i - j
                power_p = j
                terms.append((power_t, power_p))
                feats.append((t**power_t) * (p**power_p))
        X_design = np.column_stack(feats)
        coef, _, _, _ = np.linalg.lstsq(X_design, y_trim, rcond=None)

        y_pred_trim = X_design @ coef
        ss_res = float(np.sum((y_trim - y_pred_trim) ** 2))
        ss_tot = float(np.sum((y_trim - y_trim.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        # First-order coefficients correspond to linear terms
        linear_t = coef[1] if len(coef) > 1 else float("nan")
        linear_p = coef[2] if len(coef) > 2 else float("nan")
        print(
            f"[INFO] {label} fit: intercept={coef[0]:.4g}, "
            f"dT={linear_t:.4g}, dP={linear_p:.4g}, r2={r2:.3f}, n={len(trimmed)}"
        )

        corrected = numeric[y_col].copy()
        valid_mask = numeric.notna().all(axis=1)
        if valid_mask.any():
            t_all = numeric.loc[valid_mask, temp_col].to_numpy() - temp_center
            p_all = numeric.loc[valid_mask, pressure_col].to_numpy() - pressure_center
            feats_all = [np.ones(len(t_all))]
            for power_t, power_p in terms:
                feats_all.append((t_all**power_t) * (p_all**power_p))
            X_all = np.column_stack(feats_all)
            predicted = X_all @ coef
            corrected.loc[valid_mask] = numeric.loc[valid_mask, y_col] - predicted
        return corrected

    def detrend_poly4_temp_pressure(
        df: pd.DataFrame,
        *,
        y_col: str,
        temp_col: str,
        pressure_col: str,
        label: str = "Temp + Pressure poly4",
        quantile: float = 0.99,
    ) -> pd.Series:
        """Fit a full 2D polynomial (temp, pressure) up to degree 4; return residual."""
        numeric = df[[y_col, temp_col, pressure_col]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if subset.empty:
            print(f"[INFO] No data to fit {label}; returning original series.")
            return numeric[y_col]

        trimmed = subset
        for col in subset.columns:
            lo, hi = subset[col].quantile([1 - quantile, quantile])
            trimmed = trimmed[trimmed[col].between(lo, hi)]
        if trimmed.empty:
            trimmed = subset

        y_trim = trimmed[y_col].to_numpy()
        temps = trimmed[temp_col].to_numpy()
        pressures = trimmed[pressure_col].to_numpy()
        temp_center = temps.mean()
        pressure_center = pressures.mean()
        t = temps - temp_center
        p = pressures - pressure_center

        feats: list[np.ndarray] = [np.ones(len(trimmed))]
        terms = []
        for i in range(1, 8):  # degree 1..4
            for j in range(i + 1):
                power_t = i - j
                power_p = j
                terms.append((power_t, power_p))
                feats.append((t**power_t) * (p**power_p))
        X_design = np.column_stack(feats)
        coef, _, _, _ = np.linalg.lstsq(X_design, y_trim, rcond=None)

        y_pred_trim = X_design @ coef
        ss_res = float(np.sum((y_trim - y_pred_trim) ** 2))
        ss_tot = float(np.sum((y_trim - y_trim.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

        linear_t = coef[1] if len(coef) > 1 else float("nan")
        linear_p = coef[2] if len(coef) > 2 else float("nan")
        print(
            f"[INFO] {label} fit: intercept={coef[0]:.4g}, "
            f"dT={linear_t:.4g}, dP={linear_p:.4g}, r2={r2:.3f}, n={len(trimmed)}"
        )

        corrected = numeric[y_col].copy()
        valid_mask = numeric.notna().all(axis=1)
        if valid_mask.any():
            t_all = numeric.loc[valid_mask, temp_col].to_numpy() - temp_center
            p_all = numeric.loc[valid_mask, pressure_col].to_numpy() - pressure_center
            feats_all = [np.ones(len(t_all))]
            for power_t, power_p in terms:
                feats_all.append((t_all**power_t) * (p_all**power_p))
            X_all = np.column_stack(feats_all)
            predicted = X_all @ coef
            corrected.loc[valid_mask] = numeric.loc[valid_mask, y_col] - predicted
        return corrected

    merged_df = merge_events_and_logs(events_df, lab_logs_df)
    temp_col = DEFAULT_TEMP_COLUMNS[0]
    pressure_col = DEFAULT_PRESSURE_COLUMNS[0]

    if not merged_df.empty:
        # Raw events vs temperature/pressure
        plot_scatter_pairs(
            merged_df,
            "events",
            [
                (temp_col, "Temperature (°C)", "Events vs Temperature", "blue"),
                (pressure_col, "Pressure (hPa)", "Events vs Pressure", "green"),
            ],
        )

        # Joint temperature + pressure detrending to avoid reintroducing correlations
        if {temp_col, pressure_col}.issubset(merged_df.columns):
            merged_df["events_temp_pressure_detrended"] = detrend_multilinear(
                merged_df,
                y_col="events",
                x_cols=[temp_col, pressure_col],
                label="Temp + Pressure",
            )
            plot_scatter_pairs(
                merged_df,
                "events_temp_pressure_detrended",
                [
                    (
                        temp_col,
                        "Temperature (°C)",
                        "Multi-corrected Events vs Temperature",
                        "purple",
                    ),
                    (
                        pressure_col,
                        "Pressure (hPa)",
                        "Multi-corrected Events vs Pressure",
                        "brown",
                    ),
                ],
            )

            # Second try: exponential pressure contribution (linear term is first-order expansion)
            merged_df["events_temp_exp_pressure_detrended"] = detrend_temp_exp_pressure(
                merged_df,
                y_col="events",
                temp_col=temp_col,
                pressure_col=pressure_col,
                label="Temp + exp(Pressure)",
            )
            plot_scatter_pairs(
                merged_df,
                "events_temp_exp_pressure_detrended",
                [
                    (
                        temp_col,
                        "Temperature (°C)",
                        "Exp-fit Events vs Temperature",
                        "teal",
                    ),
                    (
                        pressure_col,
                        "Pressure (hPa)",
                        "Exp-fit Events vs Pressure",
                        "darkred",
                    ),
                ],
            )

            merged_df["events_temp_pressure_poly3_detrended"] = detrend_poly3_temp_pressure(
                merged_df,
                y_col="events",
                temp_col=temp_col,
                pressure_col=pressure_col,
                label="Temp + Pressure poly3",
            )
            plot_scatter_pairs(
                merged_df,
                "events_temp_pressure_poly3_detrended",
                [
                    (
                        temp_col,
                        "Temperature (°C)",
                        "Poly3-corrected Events vs Temperature",
                        "navy",
                    ),
                    (
                        pressure_col,
                        "Pressure (hPa)",
                        "Poly3-corrected Events vs Pressure",
                        "darkorange",
                    ),
                ],
            )

            merged_df["events_temp_pressure_poly4_detrended"] = detrend_poly4_temp_pressure(
                merged_df,
                y_col="events",
                temp_col=temp_col,
                pressure_col=pressure_col,
                label="Temp + Pressure poly4",
            )
            plot_scatter_pairs(
                merged_df,
                "events_temp_pressure_poly4_detrended",
                [
                    (
                        temp_col,
                        "Temperature (°C)",
                        "Poly4-corrected Events vs Temperature",
                        "darkcyan",
                    ),
                    (
                        pressure_col,
                        "Pressure (hPa)",
                        "Poly4-corrected Events vs Pressure",
                        "firebrick",
                    ),
                ],
            )
        else:
            print("[INFO] Skipping multi-fit: temp or pressure column missing.")
        
        # Histograms to compare spread of the three detrending methods
        hist_series = {
            "Linear P": merged_df.get("events_temp_pressure_detrended"),
            "Exp P": merged_df.get("events_temp_exp_pressure_detrended"),
            "Poly3 P": merged_df.get("events_temp_pressure_poly3_detrended"),
            "Poly4 P": merged_df.get("events_temp_pressure_poly4_detrended"),
        }
        hist_series = {k: v.dropna() for k, v in hist_series.items() if isinstance(v, pd.Series)}
        if len(hist_series) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            bins = 40
            for label, series in hist_series.items():
                q05, q95 = series.quantile([0.05, 0.95])
                values = series[series.between(q05, q95)]
                axes[0].hist(
                    values,
                    bins=bins,
                    histtype="step",
                    linewidth=1.2,
                    label=label,
                    density=True,
                )
                series_centered = values - values.median()
                axes[1].hist(
                    series_centered,
                    bins=bins,
                    histtype="step",
                    linewidth=1.2,
                    label=label,
                    density=True,
                )
            axes[0].set_title("Detrended events histograms")
            axes[0].set_xlabel("events / min")
            axes[0].set_ylabel("density")
            axes[1].set_title("Centered (median) histograms")
            axes[1].set_xlabel("events / min (centered)")
            axes[0].legend(frameon=False)
            fig.tight_layout()
            plt.show()
        else:
            print("[INFO] Skipping histogram comparison: not enough detrended series.")
    
    
    
    #%%
    
    
    # Plot the original events and the corrected events (three cases) vs time
    available_cases = []
    if "events_temp_pressure_detrended" in merged_df.columns:
        available_cases.append(
            ("events_temp_pressure_detrended", "Corrected (linear P)", "red")
        )
    if "events_temp_exp_pressure_detrended" in merged_df.columns:
        available_cases.append(
            ("events_temp_exp_pressure_detrended", "Corrected (exp P)", "teal")
        )
    if "events_temp_pressure_poly3_detrended" in merged_df.columns:
        available_cases.append(
            ("events_temp_pressure_poly3_detrended", "Corrected (poly3 P)", "purple")
        )
    if "events_temp_pressure_poly4_detrended" in merged_df.columns:
        available_cases.append(
            ("events_temp_pressure_poly4_detrended", "Corrected (poly4 P)", "darkcyan")
        )

    if available_cases:
        times = merged_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
        events_center = merged_df["events"].median()
        ev_q1, ev_q3 = merged_df["events"].quantile([0.01, 0.99])
        ev_iqr_mask = merged_df["events"].between(ev_q1, ev_q3)
        events_scale = merged_df.loc[ev_iqr_mask, "events"].sub(events_center).std()
        events_z = (
            merged_df["events"].sub(events_center) / events_scale
            if events_scale and not pd.isna(events_scale)
            else merged_df["events"].sub(events_center)
        )
        fig, axes = plt.subplots(len(available_cases), 1, figsize=(14, 4.5 * len(available_cases)), sharex=True)
        if len(available_cases) == 1:
            axes = [axes]
        for ax, (col, label, color) in zip(axes, available_cases):
            corr_center = merged_df[col].median()
            c_q1, c_q3 = merged_df[col].quantile([0.01, 0.99])
            corr_mask = merged_df[col].between(c_q1, c_q3)
            corr_scale = merged_df.loc[corr_mask, col].sub(corr_center).std()
            corrected_z = (
                merged_df[col].sub(corr_center) / corr_scale
                if corr_scale and not pd.isna(corr_scale)
                else merged_df[col].sub(corr_center)
            )
            ax.plot(times, events_z, label="Original (z-score)", color="blue", alpha=0.6, lw=1.0)
            ax.plot(times, corrected_z, label=f"{label} (z-score)", color=color, alpha=0.85, lw=1.1)
            ax.set_ylabel("events (median z-score)")
            ax.set_title(f"Original vs {label} (median-centered z)")
            # Limit to ~±3 sigma to reduce outlier influence on display
            y_span = 3.0
            ax.set_ylim(-y_span, y_span)
            ax.legend(loc="upper left", frameon=False)
        axes[-1].set_xlabel("Time")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        fig.tight_layout()
        plt.show()
    
    
    
    #%%
    
    
    # WAVELET ANALYSIS OF THE EVENTS TIME SERIES
    
    from scipy import signal

    def _regularize_series(
        df: pd.DataFrame, value_col: str, freq: str = "30min"
    ) -> pd.Series:
        """Resample to a regular grid and fill gaps by interpolation."""
        series = (
            pd.to_numeric(df.set_index("Time")[value_col], errors="coerce")
            .resample(freq)
            .mean()
        )
        return series.interpolate(limit_direction="both")

    def _remove_zscore_outliers(series: pd.Series, z_thresh: float = 2.0) -> pd.Series:
        """Remove outliers anywhere in the series based on z-score and interpolate."""
        if series.empty:
            return series
        z = _zscore(series)
        mask = z.abs() > z_thresh
        if mask.any():
            cleaned = series.copy()
            cleaned.loc[mask] = pd.NA
            cleaned = cleaned.interpolate(limit_direction="both")
            removed = int(mask.sum())
            print(f"[INFO] Removed {removed} outlier(s) via interpolation (z>{z_thresh}).")
            return cleaned
        return series

    def _variance_explained(y: pd.Series, y_hat: pd.Series) -> float:
        """Compute R^2 between y and its model y_hat."""
        y_arr = np.asarray(y, dtype=float)
        y_hat_arr = np.asarray(y_hat, dtype=float)
        mask = np.isfinite(y_arr) & np.isfinite(y_hat_arr)
        if mask.sum() < 2:
            return float("nan")
        ss_res = np.sum((y_arr[mask] - y_hat_arr[mask]) ** 2)
        ss_tot = np.sum((y_arr[mask] - y_arr[mask].mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot else float("nan")

    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        return (series - series.mean()) / std if std and not pd.isna(std) else series * 0

    def _welch(series: pd.Series, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD with pandas-friendly input."""
        arr = np.asarray(series, dtype=float).ravel()
        if len(arr) < 2:
            return np.array([]), np.array([])
        nperseg = min(512, len(arr))
        return signal.welch(arr, fs=fs, nperseg=nperseg)

    def _coherence(
        series_a: pd.Series, series_b: pd.Series, fs: float
    ) -> tuple[np.ndarray, np.ndarray]:
        arr_a = np.asarray(series_a, dtype=float).ravel()
        arr_b = np.asarray(series_b, dtype=float).ravel()
        if len(arr_a) < 2 or len(arr_b) < 2:
            return np.array([]), np.array([])
        nperseg = min(512, len(arr_a), len(arr_b))
        return signal.coherence(arr_a, arr_b, fs=fs, nperseg=nperseg)

    def _stft_spectrogram(
        series: pd.Series, fs: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Short-time Fourier spectrogram (time, freq, magnitude)."""
        arr = np.asarray(series, dtype=float).ravel()
        if len(arr) < 4:
            return np.array([]), np.array([]), np.array([[]])
        nperseg = min(256, len(arr))
        freqs, times, Sxx = signal.spectrogram(
            arr,
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="spectrum",
        )
        return freqs, times, Sxx

    def _morlet_wavelet(length: int, s: float, w: float = 6.0) -> np.ndarray:
        """Simple complex Morlet wavelet (manual to avoid SciPy version issues)."""
        t = np.arange(-(length // 2), length // 2, dtype=float) / s
        wavelet = np.exp(1j * w * t) * np.exp(-(t**2) / 2)
        # Normalize energy to keep scale comparable across widths
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        return wavelet

    def _cwt_morlet(arr: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """Wavelet transform using manual Morlet to avoid morlet2 availability issues."""
        arr = np.asarray(arr, dtype=float)
        output = []
        for width in widths:
            wavelet = _morlet_wavelet(len(arr), s=width, w=6.0)
            conv = signal.fftconvolve(arr, wavelet[::-1], mode="same")
            output.append(conv)
        return np.vstack(output)

    def remove_temperature_component(
        events: pd.Series,
        temperature: pd.Series,
        *,
        smooth: int | None = 5,
    ) -> tuple[pd.Series, pd.Series, tuple[float, float]]:
        """Project events onto (smoothed) temperature and return residual + component."""
        temp_series = temperature.copy()
        if smooth and smooth > 1:
            temp_series = temp_series.rolling(smooth, center=True, min_periods=1).median()
        design = np.column_stack([np.ones(len(temp_series)), temp_series.to_numpy()])
        coef, _, _, _ = np.linalg.lstsq(design, events.to_numpy(), rcond=None)
        component = pd.Series(design @ coef, index=events.index)
        residual = events - component
        return residual, component, (coef[0], coef[1])

    def _morlet_spectrogram(
        series: pd.Series, dt_seconds: float, widths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(series, dtype=float).ravel()
        if len(arr) < 4:
            return np.array([[]]), np.array([]), np.array([])
        widths = widths or np.geomspace(1, max(2, len(arr) // 6), num=60)
        cwt_matrix = _cwt_morlet(arr, widths)
        power = np.abs(cwt_matrix) ** 2
        periods_minutes = widths * dt_seconds / 60
        times_hours = np.arange(len(arr)) * dt_seconds / 3600
        return power, periods_minutes, times_hours

    if not merged_df.empty and temp_col in merged_df.columns:
        # Regularize both series to a uniform grid
        events_regular = _regularize_series(merged_df, "events")
        temp_regular = _regularize_series(merged_df, temp_col)

        # Inspect z-score distribution to pick an outlier threshold
        events_zscore_full = _zscore(events_regular)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(events_zscore_full.dropna(), bins=180, color="steelblue", alpha=0.7)
        ax.axvline(2.0, color="red", linestyle="--", label="z=±2.0")
        ax.axvline(-2.0, color="red", linestyle="--")
        ax.set_xlabel("Events z-score")
        ax.set_ylabel("Count")
        ax.set_title("Events z-score histogram (before outlier removal)")
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

        # Remove outliers and re-interpolate
        events_regular = _remove_zscore_outliers(events_regular, z_thresh=2.0)
        temp_regular = _remove_zscore_outliers(temp_regular, z_thresh=2.0)

        # Common sampling characteristics
        dt_seconds = (
            events_regular.index.to_series().diff().median().total_seconds()
            if len(events_regular) > 1
            else 1.0
        )
        fs = 1.0 / dt_seconds if dt_seconds else 1.0

        events_z = _zscore(events_regular)
        temp_z = _zscore(temp_regular)

        # Match and remove high-frequency temperature component (scaled) from events
        hf_window = 12  # ~6 hours if 30min resample
        temp_smooth = temp_regular.rolling(hf_window, center=True, min_periods=1).median()
        events_smooth = events_regular.rolling(hf_window, center=True, min_periods=1).median()
        temp_high = temp_regular - temp_smooth
        events_high = events_regular - events_smooth
        mask_hf = temp_high.notna() & events_high.notna()
        if mask_hf.any():
            design_hf = np.column_stack(
                [np.ones(mask_hf.sum()), temp_high[mask_hf].to_numpy()]
            )
            coef_hf, _, _, _ = np.linalg.lstsq(design_hf, events_high[mask_hf].to_numpy(), rcond=None)
            intercept_hf, slope_hf = coef_hf
            events_high_filtered = events_high.copy()
            events_high_filtered.loc[mask_hf] = (
                events_high[mask_hf] - (intercept_hf + slope_hf * temp_high[mask_hf])
            )
            events_temp_clean_hf = events_smooth + events_high_filtered
            print(
                f"[INFO] HF temp->events fit: intercept={intercept_hf:.3g}, slope={slope_hf:.3g}"
            )
        else:
            events_temp_clean_hf = events_regular
            slope_hf = intercept_hf = float("nan")

        # Robust multi-component temp removal (lag + smooth + high-pass)
        lag_samples = max(1, int(round((6 * 3600) / max(dt_seconds, 1))))
        best_lag = 0
        best_corr = -np.inf
        ev_z = _zscore(events_regular).fillna(0).to_numpy()
        tp_z = _zscore(temp_regular).fillna(0).to_numpy()
        for shift in range(-lag_samples, lag_samples + 1):
            if shift < 0:
                ev_shift, tp_shift = ev_z[-shift:], tp_z[: len(tp_z) + shift]
            elif shift > 0:
                ev_shift, tp_shift = ev_z[: len(ev_z) - shift], tp_z[shift:]
            else:
                ev_shift, tp_shift = ev_z, tp_z
            if len(ev_shift) < 10:
                continue
            corr = np.corrcoef(ev_shift, tp_shift)[0, 1]
            if np.isfinite(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = shift

        temp_lagged = temp_regular.shift(best_lag)
        temp_smooth_lagged = temp_lagged.rolling(hf_window, center=True, min_periods=1).median()
        temp_high_lagged = temp_lagged - temp_smooth_lagged
        reg_df = pd.DataFrame(
            {
                "events": events_regular,
                "temp_smooth": temp_smooth_lagged,
                "temp_high": temp_high_lagged,
            }
        ).dropna()
        if not reg_df.empty:
            X = np.column_stack(
                [
                    np.ones(len(reg_df)),
                    reg_df["temp_smooth"].to_numpy(),
                    reg_df["temp_high"].to_numpy(),
                ]
            )
            y = reg_df["events"].to_numpy()
            weights = np.ones(len(reg_df))
            for _ in range(4):
                coef_robust, _, _, _ = np.linalg.lstsq(X * weights[:, None], y * weights, rcond=None)
                resid = y - X @ coef_robust
                scale = 1.4826 * np.median(np.abs(resid)) if len(resid) else 1.0
                if scale == 0 or not np.isfinite(scale):
                    break
                weights = 1 / (1 + (resid / (3 * scale)) ** 2)
            coef_robust, _, _, _ = np.linalg.lstsq(X * weights[:, None], y * weights, rcond=None)
            component = pd.Series(X @ coef_robust, index=reg_df.index)
            events_temp_clean_robust = events_regular - component.reindex(events_regular.index, fill_value=0)
            r2_robust = _variance_explained(reg_df["events"], component)
            print(
                f"[INFO] Robust temp fit (lag={best_lag} samples): coef={coef_robust.tolist()}, R^2={r2_robust:.3f}"
            )
        else:
            events_temp_clean_robust = events_regular
            r2_robust = float("nan")

        # Remove the best linear estimate of temperature from events (time-domain independence)
        design = np.column_stack([np.ones(len(temp_z)), temp_z.to_numpy()])
        coef, _, _, _ = np.linalg.lstsq(design, events_z.to_numpy(), rcond=None)
        events_temp_residual = events_z - design @ coef
        print(f"[INFO] Temp->events linear fit: intercept={coef[0]:.3g}, slope={coef[1]:.3g}")

        # Reconstruct events minus dominant temperature-driven component (in original units)
        events_clean, events_temp_component, coef_units = remove_temperature_component(
            events_regular, temp_regular, smooth=5
        )
        print(
            f"[INFO] Temp->events (raw units) fit: intercept={coef_units[0]:.3g}, "
            f"slope={coef_units[1]:.3g}"
        )
        r2_raw = _variance_explained(events_regular, events_temp_component)
        print(f"[INFO] Temp component R^2: {r2_raw:.3f}")

        # Robust, lag-aware temp removal
        cleaned_df = pd.DataFrame(
            {
                "events": events_regular,
                "temp_component": events_temp_component,
                "events_temp_clean": events_clean,
                "events_temp_clean_hf": events_temp_clean_hf,
                "events_temp_clean_robust": events_temp_clean_robust,
            }
        )
        r2_raw = _variance_explained(events_regular, events_temp_component)
        r2_clean = _variance_explained(events_regular, events_clean)
        print(f"[INFO] Temp component R^2: {r2_raw:.3f}; cleaned vs raw R^2: {r2_clean:.3f}")
        # Time-domain comparison
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(events_z.index, events_z, label="events (z)", lw=1.2)
        axes[0].plot(temp_z.index, temp_z, label="temp (z)", lw=1.0, alpha=0.8)
        axes[0].set_ylabel("z-score")
        axes[0].set_title("Normalized series")
        axes[0].legend(frameon=False)

        axes[1].plot(events_temp_residual.index, events_temp_residual, lw=1.2, color="purple")
        axes[1].set_ylabel("residual (events ⟂ temp)")
        axes[1].set_xlabel("Time")
        axes[1].set_title("Events residual after removing linear temp contribution")
        fig.tight_layout()
        plt.show()

        # Temperature component vs cleaned events (original units)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(cleaned_df.index, cleaned_df["events"], label="events", lw=1.0)
        axes[0].plot(
            cleaned_df.index,
            cleaned_df["temp_component"],
            label="temp-driven component",
            lw=1.0,
            alpha=0.8,
            color="orange",
        )
        axes[0].set_ylabel("events / min")
        axes[0].legend(frameon=False)
        axes[0].set_title("Events and fitted temperature-driven component")

        axes[1].plot(
            cleaned_df.index,
            cleaned_df["events_temp_clean"],
            label="events cleaned of temperature",
            lw=1.0,
            color="green",
        )
        axes[1].set_ylabel("events / min")
        axes[1].set_xlabel("Time")
        axes[1].legend(frameon=False)
        axes[1].set_title("Temperature-cleaned events (raw units)")
        fig.tight_layout()
        plt.show()

        # Final overlay: original vs temperature-cleaned events (mean-subtracted for clarity)
        fig, ax = plt.subplots(figsize=(12, 4))
        events_anom = cleaned_df["events"] - cleaned_df["events"].mean()
        events_clean_anom = cleaned_df["events_temp_clean"] - cleaned_df[
            "events_temp_clean"
        ].mean()
        events_clean_hf_anom = cleaned_df["events_temp_clean_hf"] - cleaned_df[
            "events_temp_clean_hf"
        ].mean()
        events_clean_robust_anom = cleaned_df["events_temp_clean_robust"] - cleaned_df[
            "events_temp_clean_robust"
        ].mean()
        r2_clean_anom = _variance_explained(events_anom, events_clean_anom)
        r2_clean_hf_anom = _variance_explained(events_anom, events_clean_hf_anom)
        r2_clean_robust_anom = _variance_explained(events_anom, events_clean_robust_anom)
        print(
            f"[INFO] Variance captured (anoms): linear={r2_clean_anom:.3f}, "
            f"HF={r2_clean_hf_anom:.3f}, robust/lagged={r2_clean_robust_anom:.3f}"
        )
        ax.plot(cleaned_df.index, events_anom, label="events (original)", lw=1.0)
        ax.plot(
            cleaned_df.index,
            events_clean_anom,
            label="events (temp-cleaned)",
            lw=1.0,
            color="green",
            alpha=0.8,
        )
        ax.plot(
            cleaned_df.index,
            events_clean_hf_anom,
            label="events (temp-cleaned + HF scaled)",
            lw=1.0,
            color="orange",
            alpha=0.7,
        )
        ax.plot(
            cleaned_df.index,
            events_clean_robust_anom,
            label="events (robust lagged clean)",
            lw=1.0,
            color="purple",
            alpha=0.8,
        )
        ax.set_ylabel("events / min (mean-subtracted)")
        ax.set_xlabel("Time")
        ax.legend(frameon=False)
        ax.set_title("Events: original vs temperature-cleaned")
        fig.tight_layout()
        plt.show()

        # Frequency-domain PSDs and coherence
        f_ev, p_ev = _welch(events_z, fs)
        f_temp, p_temp = _welch(temp_z, fs)
        f_res, p_res = _welch(events_temp_residual, fs)
        f_coh, coh_ev_temp = _coherence(events_z, temp_z, fs)
        _, coh_res_temp = _coherence(events_temp_residual, temp_z, fs)

        if len(f_ev) and len(f_temp) and len(f_res):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].loglog(f_ev, p_ev, label="events", color="blue")
            axes[0].loglog(f_temp, p_temp, label="temp", color="red", alpha=0.8)
            axes[0].loglog(f_res, p_res, label="events residual", color="purple")
            axes[0].set_xlabel("Frequency (Hz)")
            axes[0].set_ylabel("PSD")
            axes[0].set_title("Welch PSDs")
            axes[0].legend(frameon=False)

            if len(f_coh):
                axes[1].semilogx(f_coh, coh_ev_temp, label="events vs temp", color="black")
                axes[1].semilogx(f_coh, coh_res_temp, label="residual vs temp", color="purple")
                axes[1].set_ylim(0, 1)
                axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Coherence")
            axes[1].set_title("Magnitude-squared coherence")
            axes[1].legend(frameon=False)
            axes[1].set_axisbelow(True)
            fig.tight_layout()
            plt.show()
        else:
            print("[INFO] Skipping PSD/coherence plot: insufficient samples.")

        # STFT spectrograms (time vs frequency heatmaps)
        f_ev_s, t_ev_s, spec_ev = _stft_spectrogram(events_z, fs)
        f_temp_s, t_temp_s, spec_temp = _stft_spectrogram(temp_z, fs)
        f_res_s, t_res_s, spec_res = _stft_spectrogram(events_temp_residual, fs)
        if spec_ev.size and spec_temp.size and spec_res.size:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
            for ax, freqs, times, spec, title in zip(
                axes,
                [f_ev_s, f_temp_s, f_res_s],
                [t_ev_s, t_temp_s, t_res_s],
                [spec_ev, spec_temp, spec_res],
                ["Events spectrogram", "Temp spectrogram", "Residual spectrogram"],
            ):
                img = ax.pcolormesh(times * dt_seconds / 3600, freqs, spec, shading="auto")
                ax.set_xlabel("Time (hours)")
                ax.set_title(title)
                fig.colorbar(img, ax=ax, label="Power")
            axes[0].set_ylabel("Frequency (Hz)")
            fig.suptitle("STFT spectrograms (time vs frequency)", fontsize=14)
            fig.tight_layout()
            plt.show()
        else:
            print("[INFO] Skipping STFT spectrograms: insufficient samples.")

        # Wavelet spectrograms and cross power
        cwt_events, periods_min, times_hours = _morlet_spectrogram(events_z, dt_seconds)
        cwt_temp, _, _ = _morlet_spectrogram(temp_z, dt_seconds)
        cross_power = np.abs(cwt_events * np.conjugate(cwt_temp))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, power, title in zip(
            axes,
            [np.abs(cwt_events) ** 2, np.abs(cwt_temp) ** 2, cross_power],
            ["Events power", "Temperature power", "Cross power"],
        ):
            img = ax.imshow(
                power,
                origin="lower",
                aspect="auto",
                extent=(
                    times_hours[0],
                    times_hours[-1] if len(times_hours) > 1 else 1,
                    periods_min[0],
                    periods_min[-1],
                ),
                cmap="viridis",
            )
            ax.set_xlabel("Time (hours)")
            ax.set_title(title)
            fig.colorbar(img, ax=ax, label="Power")
        axes[0].set_ylabel("Period (minutes)")
        fig.suptitle("Morlet wavelet analysis", fontsize=14)
        fig.tight_layout()
        plt.show()
    else:
        print("[INFO] Wavelet analysis skipped: missing merged data or temperature column.")
    
    
    
    #%%
    
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
    
    #%%
    
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
    
    #%%
    
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


    


    #%%
    
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
    
    #%%
    
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




    #%%
    
    
    
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
    pct_dif_cols = []
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
        pct_dif_cols.append(diff_col)

    if not pct_dif_cols:
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

    
    #%%







    
# %%
