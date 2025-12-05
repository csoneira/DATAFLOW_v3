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
    
    station = "1"
    
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
        return corrected, coef, r2
    
    
    
    detrended_events, coef, r2 = detrend_polynomial(
        events_df,
        y_col="events",
        x_cols=["eff1", "eff2", "eff3", "eff4"],
        degree=3,
        label="Polynomial detrending",
    )

    #%%

    import numpy as np
    import pandas as pd

    def clip_outliers(series, sigma=3):
        """Return series with values outside mean ± sigma*std removed."""
        series = pd.Series(series)  # works with numpy arrays too
        mu = series.median()
        sigma_val = series.std()    # sample std (ddof=1, default in pandas)
        mask = (series >= mu - sigma * sigma_val) & (series <= mu + sigma * sigma_val)
        return series[mask]

    # Remove outliers in each vector
    events_clean = clip_outliers(events_df["events"], sigma=1)
    detrended_clean = clip_outliers(detrended_events, sigma=0.5)

    events_mean = events_clean.mean()
    events_std = events_clean.std()
    print(f"Events mean (no outliers): {events_mean:.3f}, std: {events_std:.3f}")

    detrended_mean = detrended_clean.mean()
    detrended_std = detrended_clean.std()
    print(f"Detrended events mean (no outliers): {detrended_mean:.3f}, std: {detrended_std:.3f}")


    # events_mean = events_df["events"].median()
    # events_std = events_df["events"].std()
    # print(f"Events mean: {events_mean:.3f}, std: {events_std:.3f}")

    # detrended_mean = detrended_events.median()
    # detrended_std = detrended_events.std()
    # print(f"Detrended events mean: {detrended_mean:.3f}, std: {detrended_std:.3f}")

    z_scores_events = (events_df["events"]  / events_mean - 1)
    z_scores_detrended = (detrended_events  / detrended_mean - 1) / 100

    # Plot original and detrended events
    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, z_scores_events, lw=1.1, label="Original Events")
    ax.plot(times, z_scores_detrended, lw=1.1, label="Detrended Events")
    ax.set_title("Original vs Detrended Events")
    ax.set_ylabel("events / 2h")
    ax.set_xlabel("Time")
    # Ylim 3 sigma
    # ax.set_ylim(-10, 10)
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

    print("Detrending polynomial coefficients:")
    for i, c in enumerate(coef):
        print(f"  a{i}: {c:.6f}")
    print(f"R² of the fit: {r2:.4f}")


    # I want the time series of the predicted values from the detrending fit
    # so i can see how well the polynomial explains the original events
    # and compare it to the detrended events.

    def compute_predicted_values(
        df: pd.DataFrame,
        *,
        x_cols: Sequence[str],
        coef: np.ndarray,
        degree: int = 2,
    ) -> pd.Series:
        """Compute predicted y values from polynomial terms and coefficients."""
        numeric = df[x_cols].apply(pd.to_numeric, errors="coerce")
        predicted = pd.Series(np.nan, index=df.index)
        valid_mask = numeric.notna().all(axis=1)
        if valid_mask.any():
            X_terms = []
            for deg in range(1, degree + 1):
                for combo in combinations_with_replacement(x_cols, deg):
                    term = np.prod(
                        [numeric.loc[valid_mask, col].to_numpy() for col in combo],
                        axis=0,
                    )
                    X_terms.append(term)
            X_design = np.column_stack([np.ones(len(X_terms[0])), *X_terms])
            predicted_values = X_design @ coef
            predicted.loc[valid_mask] = predicted_values
        return predicted
    predicted_events = compute_predicted_values(
        events_df,
        x_cols=["eff1", "eff2", "eff3", "eff4"],
        coef=coef,
        degree=3,
    )
    events_df["predicted_events"] = predicted_events

    # Plot original, predicted, and detrended events
    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, events_df["events"], lw=1.1, label="Original Events")
    ax.plot(times, events_df["predicted_events"], lw=1.1, label="Predicted Events")
    ax.plot(times, events_df["detrended_events"], lw=1.1, label="Detrended Events")
    ax.set_title("Original, Predicted, and Detrended Events")
    ax.set_ylabel("events / 2h")
    ax.set_xlabel("Time")
    # Ylim 3 sigma
    # ax.set_ylim(-10, 10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()
    


# %%
