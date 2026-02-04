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
from scipy import optimize
from scipy.special import expit

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
    
    station = "2"
    
    result = quicklook(
        # start="2025-09-25",
        # start="2025-08-10",
        start="2025-01-01",
        end="2025-11-04",
        # end="2025-11-30",
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
    

    # # Affine transform efficiencies to be roughly between 0 and 1
    # a_1 = 0.9
    # b_1 = 0.5
    # events_df["eff1"] = a_1 * ( events_df["eff1"] + b_1 )

    # a_4 = 0.5
    # b_4 = 0.5
    # events_df["eff4"] = a_4 * ( events_df["eff4"] + b_4 )

    import numpy as np

    # Reference efficiency per event: mean of eff2 and eff3
    ref_eff = 0.5 * (events_df["eff2"] + events_df["eff3"])

    def fit_affine_a_x_plus_b(source, target):
        """
        Find a, b such that a * (source + b) best fits target in least-squares sense.
        """
        x = np.asarray(source, dtype=float)
        y = np.asarray(target, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return 1.0, 0.0

        x = x[mask]
        y = y[mask]

        # Clip extreme outliers to stabilize the fit
        def _clip_extreme(arr: np.ndarray, pct: float = 99.0) -> np.ndarray:
            lo, hi = np.nanpercentile(arr, [100 - pct, pct])
            return np.clip(arr, lo, hi)

        x = _clip_extreme(x)
        y = _clip_extreme(y)

        # We fit y ≈ alpha * x + beta, then map alpha, beta -> a, b via:
        # a = alpha,  b = beta / alpha    because a*(x + b) = a*x + a*b
        X = np.vstack([x, np.ones_like(x)]).T
        try:
            alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 1.0, 0.0

        if np.isclose(alpha, 0.0):
            # Degenerate case: fall back to a=1, pure shift chosen to match means
            a = 1.0
            b = (y.mean() - a * x.mean()) / a
        else:
            a = alpha
            b = beta / a

        return a, b

    # Fit for eff1
    a_1, b_1 = fit_affine_a_x_plus_b(events_df["eff1"], ref_eff)
    events_df["eff1"] = a_1 * (events_df["eff1"] + b_1)

    # Fit for eff4
    a_4, b_4 = fit_affine_a_x_plus_b(events_df["eff4"], ref_eff)
    events_df["eff4"] = a_4 * (events_df["eff4"] + b_4)

    print(f"a_1 = {a_1:.6f}, b_1 = {b_1:.6f}")
    print(f"a_4 = {a_4:.6f}, b_4 = {b_4:.6f}")


    # Plot 5 plots, one for each efficiency and one for "events"
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
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
        ["eff1", "eff2", "eff3", "eff4"],
        title="Efficiency 1",
        ylabel="eff1",
    )
    
    # I want the y limit of all efficiency plots to be 0 to 1
    for ax in axes[1:]:
        ax.set_ylim(0, 1)
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
        """Find polynomial F(eff) that minimizes correlation of events/F with efficiencies."""
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
        x_trim = trimmed[x_cols].to_numpy()

        # Build design matrix with polynomial terms (no intercept; intercept handled separately)
        X_terms = []
        term_labels = []
        for deg in range(1, degree + 1):
            for combo in combinations_with_replacement(x_cols, deg):
                term = np.prod([trimmed[col].to_numpy() for col in combo], axis=0)
                X_terms.append(term)
                term_labels.append("*".join(combo))
        if not X_terms:
            print(f"[INFO] No polynomial terms built for {label}; returning original series.")
            return numeric[y_col], np.array([]), float("nan")
        X_design = np.column_stack(X_terms)

        def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                return 0.0
            corr = np.corrcoef(a[mask], b[mask])[0, 1]
            return float(corr) if np.isfinite(corr) else 0.0

        def objective(beta: np.ndarray) -> float:
            denom = expit(X_design @ beta)  # bound F to (0,1)
            denom = np.clip(denom, 1e-3, 1.0)
            corrected = y_trim / denom
            corrs = [_safe_corr(corrected, x_trim[:, idx]) for idx in range(x_trim.shape[1])]
            penalty_l2 = 1e-4 * np.sum(beta**2)
            penalty_close_to_one = 1e-4 * float(np.mean((1.0 - denom) ** 2))
            return float(np.sum(np.square(corrs)) + penalty_l2 + penalty_close_to_one)

        beta0 = np.zeros(X_design.shape[1], dtype=float)
        opt_result = optimize.minimize(objective, beta0, method="L-BFGS-B")
        beta_hat = opt_result.x if opt_result.success else beta0
        score = objective(beta_hat)

        print(
            f"[INFO] {label}: degree={degree}, corr_score={score:.4g}, n={len(trimmed)}, "
            f"opt_success={opt_result.success}"
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
            X_all_design = np.column_stack(X_all_terms)
            denom_all = expit(X_all_design @ beta_hat)
            denom_all = np.clip(denom_all, 1e-3, 1.0)
            corrected.loc[valid_mask] = numeric.loc[valid_mask, y_col] / denom_all
        return corrected, beta_hat, score
    
    
    
    detrended_events, coef, corr_score = detrend_polynomial(
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

    z_scores_events = (events_df["events"]  - events_mean )
    z_scores_detrended = (detrended_events  - detrended_mean )

    # Plot original and detrended events
    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, z_scores_events, lw=1.1, label="Original Events")
    ax.plot(times, z_scores_detrended, lw=1.1, label="Detrended Events")
    ax.set_title("Original vs Detrended Events")
    ax.set_ylabel("events / 2h")
    ax.set_xlabel("Time")
    # Ylim 3 sigma
    ax.set_ylim(-20000, 20000)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()

    #%%

    # Plot original and detrended events
    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, events_df["events"], lw=1.1, label="Original Events")
    ax.plot(times, detrended_events, lw=1.1, label="Detrended Events")
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
    

    # Plot the time series of the fitted polynomial F(eff)
    def compute_fitted_polynomial(
        df: pd.DataFrame,
        *,
        x_cols: Sequence[str],
        coef: np.ndarray,
        degree: int = 2,
    ) -> pd.Series:
        numeric = df[x_cols].apply(pd.to_numeric, errors="coerce")
        valid_mask = numeric.notna().all(axis=1)
        fitted = pd.Series(np.nan, index=df.index, dtype=float)
        if valid_mask.any():
            X_terms = []
            for deg in range(1, degree + 1):
                for combo in combinations_with_replacement(x_cols, deg):
                    term = np.prod(
                        [numeric.loc[valid_mask, col].to_numpy() for col in combo],
                        axis=0,
                    )
                    X_terms.append(term)
            X_design = np.column_stack(X_terms)
            fitted_values = expit(X_design @ coef)
            fitted.loc[valid_mask] = fitted_values
        return fitted
    
    fitted_polynomial = compute_fitted_polynomial(
        events_df,
        x_cols=["eff1", "eff2", "eff3", "eff4"],
        coef=coef,
        degree=3,
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, fitted_polynomial, lw=1.1, label="Fitted Polynomial F(eff)")
    ax.set_title("Fitted Polynomial F(eff) over Time")
    ax.set_ylabel("F(eff)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()

    #%%

    z_scores_detrended = (detrended_events - detrended_mean) / detrended_std

    # Histogram the zscores of the detrended events and filter outliers beyond 3 sigma, but after the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z_scores_detrended.dropna(), bins=50, color="purple", alpha=0.7)
    ax.set_title("Histogram of Detrended Events Z-Scores")
    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    plt.show()

    # Filter outliers beyond 3 sigma and do a linear interpolation to fill gaps
    filtered_detrended = detrended_events.copy()
    outlier_mask = z_scores_detrended.abs() > 3
    filtered_detrended[outlier_mask] = np.nan
    filtered_detrended = filtered_detrended.interpolate(method="linear")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    times = events_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
    ax.plot(times, detrended_events, lw=1.1, label="Detrended Events")
    ax.plot(times, filtered_detrended, lw=1.1, label="Filtered Detrended Events", color="red")
    ax.set_title("Detrended Events with Outliers Filtered")
    ax.set_ylabel("events / 2h")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    plt.show()

    events_df["filtered_detrended"] = filtered_detrended


    #%%

    # Now I want you to study the correlation betweeen the detrended events and the pressure and temperature columns
    # from the lab logs. Give me scatter plots and correlation coefficients.
    merged_df = merge_events_and_logs(events_df, lab_logs_df, tolerance="10min")

    def compute_correlations(
        df: pd.DataFrame,
        y_col: str,
        x_cols: Sequence[str],
    ) -> dict[str, float]:
        corrs = {}
        for x_col in x_cols:
            if x_col in df.columns:
                valid_mask = df[[y_col, x_col]].notna().all(axis=1)
                if valid_mask.sum() >= 3:
                    corr = np.corrcoef(
                        df.loc[valid_mask, y_col].to_numpy(),
                        df.loc[valid_mask, x_col].to_numpy(),
                    )[0, 1]
                    corrs[x_col] = corr
        return corrs
    pressure_cols = _select_columns(lab_logs_df, DEFAULT_PRESSURE_COLUMNS)
    temperature_cols = _select_columns(lab_logs_df, DEFAULT_TEMP_COLUMNS)
    detrended_corrs = compute_correlations(
        merged_df,
        y_col="filtered_detrended",
        x_cols=pressure_cols + temperature_cols,
    )
    print("Correlations between detrended events and lab log variables:")
    for var, corr in detrended_corrs.items():
        print(f"  {var}: {corr:.4f}")
    
    plot_scatter_pairs(
        merged_df,
        y_col="filtered_detrended",
        pairs=[
            (col, col.replace("_", " ").title(), f"Detrended Events vs {col}", "purple")
            for col in pressure_cols + temperature_cols
        ],
        figsize=(16, 4),
    )

    #%%

    # Now display time series of pressure and temperature variables alongside detrended events
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    _plot_panel(
        axes[0],
        merged_df,
        ["filtered_detrended"],
        title="Detrended Events",
        ylabel="events / 2h",
    )
    _plot_panel(
        axes[1],
        merged_df,
        temperature_cols,
        title="Temperatures",
        ylabel="°C",
    )
    _plot_panel(
        axes[2],
        merged_df,
        pressure_cols,
        title="Pressures",
        ylabel="hPa",
    )
    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.suptitle("Detrended Events and Lab Logs Overview", fontsize=14)
    fig.tight_layout()
    plt.show()


    #%%

    # Now calculate the pressure correction factor fitting an exponential, so you calculate the ln(I/I0) = beta * ( P - P0 )
    def fit_pressure_correction(
        df: pd.DataFrame,
        *,
        y_col: str,
        p_col: str,
    ) -> tuple[float, float]:
        numeric = df[[y_col, p_col]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if subset.empty:
            print(f"[INFO] No data to fit pressure correction; returning NaNs.")
            return float("nan"), float("nan")
        
        y_values = subset[y_col].to_numpy()
        p_values = subset[p_col].to_numpy()
        
        ln_y = np.log(y_values + 1e-6)  # avoid log(0)
        
        A = np.vstack([p_values, np.ones_like(p_values)]).T
        beta, ln_I0 = np.linalg.lstsq(A, ln_y, rcond=None)[0]
        
        return beta, np.exp(ln_I0)
    beta, I0 = fit_pressure_correction(
        merged_df,
        y_col="filtered_detrended",
        p_col=pressure_cols[0] if pressure_cols else "",
    )
    print(f"Fitted pressure correction: beta = {beta:.6f}, I0 = {I0:.3f}")

    # Plot the fit
    if pressure_cols:
        p_col = pressure_cols[0]
        numeric = merged_df[[ "filtered_detrended", p_col]].apply(pd.to_numeric, errors="coerce")
        subset = numeric.dropna()
        if not subset.empty:
            y_values = subset["filtered_detrended"].to_numpy()
            p_values = subset[p_col].to_numpy()
            ln_y = np.log(y_values + 1e-6)
            fitted_ln_y = beta * p_values + np.log(I0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(p_values, ln_y, alpha=0.5, label="Data", s=1)
            ax.plot(p_values, fitted_ln_y, color="red", label="Fitted Line")
            ax.set_xlabel(p_col.replace("_", " ").title())
            ax.set_ylabel("ln(Detrended Events)")
            ax.set_title("Pressure Correction Fit")
            ax.legend()
            fig.tight_layout()
            plt.show()
    
    #%%

    # Calculate corrected detrended events
    if not np.isnan(beta):
        pressure_series = pd.to_numeric(merged_df[pressure_cols[0]], errors="coerce")
        pressure_correction = np.exp(beta * (pressure_series - pressure_series.mean()))
        merged_df["pressure_corrected_detrended"] = merged_df["filtered_detrended"] / pressure_correction

        # Plot corrected detrended events
        fig, ax = plt.subplots(figsize=(14, 5))
        times = merged_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
        ax.plot(times, merged_df["filtered_detrended"], lw=1.1, label="Detrended Events")
        ax.plot(times, merged_df["pressure_corrected_detrended"], lw=1.1, label="Pressure Corrected Detrended Events", color="green")
        ax.set_title("Pressure Corrected Detrended Events")
        ax.set_ylabel("events / 2h")
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        plt.show()
    
    #%%

    # Now do a scatter plot of pressure corrected detrended events vs pressure 
    if not np.isnan(beta):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(merged_df[pressure_cols[0]], merged_df["pressure_corrected_detrended"], alpha=0.5, s=1)
        ax.set_xlabel(pressure_cols[0].replace("_", " ").title())
        ax.set_ylabel("Pressure Corrected Detrended Events")
        ax.set_title("Pressure Corrected Detrended Events vs Pressure")
        fig.tight_layout()
        plt.show()
    

    #%%

    # Now do a histogram of the pressure corrected detrended events
    if not np.isnan(beta):
        fig, ax = plt.subplots(figsize=(8, 5))
        v = merged_df["pressure_corrected_detrended"].dropna()

        w = ( v - v.mean() ) / v.mean() * 100 # percent deviation from mean

        ax.hist(w, bins=50, color="orange", alpha=0.7)
        ax.set_title("Histogram of Pressure Corrected Detrended Events")
        ax.set_xlabel("Percent Deviation from Mean (%)")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        plt.show()
    
    #%%

    # Now calculate the corr of the pressure corrected detrended events vs ext temperature
    if not np.isnan(beta):
        detrended_corrs_pc = compute_correlations(
            merged_df,
            y_col="pressure_corrected_detrended",
            x_cols=temperature_cols,
        )
        print("Correlations between pressure corrected detrended events and temperature variables:")
        for var, corr in detrended_corrs_pc.items():
            print(f"  {var}: {corr:.4f}")
        
    # Plot it. Consider that temperature is only the "sensors_ext_Temperature_ext" col
    plot_scatter_pairs(
        merged_df,
        y_col="pressure_corrected_detrended",
        pairs=[
            (col, col.replace("_", " ").title(), f"Pressure Corrected Detrended Events vs {col}", "brown")
            for col in temperature_cols
        ],
        figsize=(16, 4),
    )

    # Calculate the linear fit coefficients between pressure corrected detrended events and ext temperature
    if not np.isnan(beta):
        def fit_linear_relation(
            df: pd.DataFrame,
            *,
            y_col: str,
            x_col: str,
        ) -> tuple[float, float]:
            numeric = df[[y_col, x_col]].apply(pd.to_numeric, errors="coerce")
            subset = numeric.dropna()
            if subset.empty:
                print(f"[INFO] No data to fit linear relation; returning NaNs.")
                return float("nan"), float("nan")
            
            y_values = subset[y_col].to_numpy()
            x_values = subset[x_col].to_numpy()
            
            A = np.vstack([x_values, np.ones_like(x_values)]).T
            m, b = np.linalg.lstsq(A, y_values, rcond=None)[0]
            
            return m, b
        
        for temp_col in temperature_cols:
            m, b = fit_linear_relation(
                merged_df,
                y_col="pressure_corrected_detrended",
                x_col=temp_col,
            )
            print(f"Fitted linear relation for {temp_col}: slope = {m:.6f}, intercept = {b:.3f}")
    
    # Plot the fits
    if not np.isnan(beta):
        for temp_col in temperature_cols:
            numeric = merged_df[[ "pressure_corrected_detrended", temp_col]].apply(pd.to_numeric, errors="coerce")
            subset = numeric.dropna()
            if not subset.empty:
                y_values = subset["pressure_corrected_detrended"].to_numpy()
                x_values = subset[temp_col].to_numpy()
                
                m, b = fit_linear_relation(
                    merged_df,
                    y_col="pressure_corrected_detrended",
                    x_col=temp_col,
                )
                fitted_y = m * x_values + b
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(x_values, y_values, alpha=0.5, label="Data", s=1)
                ax.plot(x_values, fitted_y, color="red", label="Fitted Line")
                ax.set_xlabel(temp_col.replace("_", " ").title())
                ax.set_ylabel("Pressure Corrected Detrended Events")
                ax.set_title(f"Linear Fit: Pressure Corrected Detrended Events vs {temp_col}")
                ax.legend()
                fig.tight_layout()
                plt.show()
    
    # Define the decorrelated temperature corrected detrended events
    if not np.isnan(beta):
        decorrelated_temp_detrended = merged_df["pressure_corrected_detrended"].copy()
        for temp_col in temperature_cols:
            m, b = fit_linear_relation(
                merged_df,
                y_col="pressure_corrected_detrended",
                x_col=temp_col,
            )
            temp_series = pd.to_numeric(merged_df[temp_col], errors="coerce")
            decorrelated_temp_detrended -= m * (temp_series - temp_series.mean())
        merged_df["decorrelated_temp_detrended"] = decorrelated_temp_detrended

        # Plot decorrelated temperature corrected detrended events
        fig, ax = plt.subplots(figsize=(14, 5))
        times = merged_df["Time"].map(pd.Timestamp.to_pydatetime).to_numpy()
        ax.plot(times, merged_df["pressure_corrected_detrended"], lw=1.1, label="Pressure Corrected Detrended Events")
        ax.plot(times, merged_df["decorrelated_temp_detrended"], lw=1.1, label="Decorrelated Temp Corrected Detrended Events", color="magenta")
        ax.set_title("Decorrelated Temperature Corrected Detrended Events")
        ax.set_ylabel("events / 2h")
        ax.set_xlabel("Time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        plt.show()

    #%%

    # Load the CALMA.csv file for the station and date range
    

    #%%
    # Read CALMA.csv and compare RCORR_E with decorrelated temp-corrected detrended series
    calma_path = Path("/home/mingo/DATAFLOW_v3/MASTER/STAGE_2/CALMA.csv")
    if calma_path.exists():
        calma_df = None
        try:
            calma_df = pd.read_csv(
                calma_path,
                sep=";",
                comment="#",
                header=None,
                names=["Time", "RCORR_E"],
                skip_blank_lines=True,
                engine="python",
            )
        except Exception as exc:
            print(f"[INFO] Failed reading CALMA: {exc}")
        if calma_df is not None and "decorrelated_temp_detrended" in merged_df.columns:
            calma_df["Time"] = pd.to_datetime(calma_df["Time"], errors="coerce")
            calma_df["RCORR_E"] = pd.to_numeric(calma_df["RCORR_E"], errors="coerce")
            calma_df.dropna(subset=["Time", "RCORR_E"], inplace=True)
            calma_df.sort_values("Time", inplace=True)
            merged_df_sorted = merged_df.sort_values("Time")

            fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            axes[0].plot(
                calma_df["Time"].map(pd.Timestamp.to_pydatetime),
                calma_df["RCORR_E"],
                lw=1.1,
                label="CALMA RCORR_E",
                color="steelblue",
            )
            axes[0].set_ylabel("RCORR_E")
            axes[0].set_title("CALMA RCORR_E")

            axes[1].plot(
                merged_df_sorted["Time"].map(pd.Timestamp.to_pydatetime),
                pd.to_numeric(merged_df_sorted["decorrelated_temp_detrended"], errors="coerce"),
                lw=1.1,
                label="Decorrelated Temp Detrended",
                color="darkorange",
            )
            axes[1].set_ylabel("events / 2h")
            axes[1].set_title("Decorrelated Temp Corrected Detrended Events")

            axes[-1].set_xlabel("Time")
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            for ax in axes:
                ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            plt.show()
        else:
            print("[INFO] CALMA.csv missing required columns or decorrelated series absent; skipping plot.")
    else:
        print(f"[INFO] CALMA file not found at {calma_path}; skipping CALMA comparison.")


# %%
