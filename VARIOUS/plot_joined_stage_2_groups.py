#!/usr/bin/env python3

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "STATIONS" / "MINGO01" / "STAGE_2" / "joined_stage_2.csv"
OUTPUT_PATH = REPO_ROOT / "VARIOUS" / "joined_stage_2_groups.png"

PLOT_CONFIG = {
    "apply_mean_time_filter": True,
    "mean_time_filter_window": "60min",
    "apply_median_time_filter": False,
    "median_time_filter_window": "60min",
}

X_AXIS_CONFIG = {
    "tick_frequency": "1W",
}


def prepare_plot_df(df, plot_config):
    plot_df = df.sort_values("time").reset_index(drop=True).copy()
    if plot_df["time"].isna().any():
        return plot_df

    apply_mean_filter = plot_config.get("apply_mean_time_filter", False)
    apply_median_filter = plot_config.get("apply_median_time_filter", False)
    if apply_mean_filter and apply_median_filter:
        raise ValueError("Choose only one plotting time filter: mean or median.")
    if not apply_mean_filter and not apply_median_filter:
        return plot_df

    numeric_columns = plot_df.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        return plot_df

    plot_df = plot_df.set_index("time")
    if apply_mean_filter:
        plot_df[numeric_columns] = plot_df[numeric_columns].rolling(
            plot_config["mean_time_filter_window"],
            min_periods=1,
        ).mean()
    else:
        plot_df[numeric_columns] = plot_df[numeric_columns].rolling(
            plot_config["median_time_filter_window"],
            min_periods=1,
        ).median()
    return plot_df.reset_index()


def plot_lines(ax, df, columns, title, ylabel):
    present = [column for column in columns if column in df.columns]
    for column in present:
        ax.plot(df["time"], df[column], label=column, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if present:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)


def build_time_tick_locator(tick_frequency):
    if tick_frequency is None:
        return None

    offset = pd.tseries.frequencies.to_offset(tick_frequency)
    if isinstance(offset, pd.offsets.Minute):
        return mdates.MinuteLocator(interval=offset.n)
    if isinstance(offset, pd.offsets.Hour):
        return mdates.HourLocator(interval=offset.n)
    if isinstance(offset, pd.offsets.Day):
        return mdates.DayLocator(interval=offset.n)
    if isinstance(offset, pd.offsets.Week):
        return mdates.WeekdayLocator(interval=offset.n)

    raise ValueError(
        "Unsupported x-axis tick_frequency. Use None or values like '30min', '6h', '1D', or '1W'."
    )


def configure_time_axis(ax, tick_frequency):
    ax.set_xlabel("time")
    locator = build_time_tick_locator(tick_frequency)
    if locator is None:
        return

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    df_plot = prepare_plot_df(df, PLOT_CONFIG)

    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=True)
    axes = axes.ravel()

    ax = axes[0]
    if "temp_lab" in df_plot.columns:
        ax.plot(df_plot["time"], df_plot["temp_lab"], color="tab:red", label="temp_lab", linewidth=0.8)
    ax.set_title("Lab Temp / Pressure")
    ax.set_ylabel("Temp")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    if "pressure_lab" in df_plot.columns:
        ax2.plot(df_plot["time"], df_plot["pressure_lab"], color="tab:blue", label="pressure_lab", linewidth=0.8)
    ax2.set_ylabel("Pressure")
    lines = ax.get_lines() + ax2.get_lines()
    if lines:
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper right")

    ax = axes[1]
    if "temp_ground" in df_plot.columns:
        ax.plot(df_plot["time"], df_plot["temp_ground"], label="temp_ground", linewidth=0.8)
    if "temp_100mbar" in df_plot.columns:
        ax.plot(df_plot["time"], df_plot["temp_100mbar"], label="temp_100mbar", linewidth=0.8)
    ax.set_title("Ground / 100mbar / Height")
    ax.set_ylabel("Temp")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    if "height_100mbar" in df_plot.columns:
        ax2.plot(df_plot["time"], df_plot["height_100mbar"], color="tab:green", label="height_100mbar", linewidth=0.8)
    ax2.set_ylabel("Height")
    lines = ax.get_lines() + ax2.get_lines()
    if lines:
        ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper right")

    plot_lines(axes[2], df_plot, ["Q_event_mean"], "Q_event_mean", "Q")
    plot_lines(axes[3], df_plot, ["count"], "Count", "Count")
    plot_lines(axes[4], df_plot, ["High"], "High", "Rate")
    plot_lines(axes[5], df_plot, ["Mid-N", "Mid-NE", "Mid-E", "Mid-SE", "Mid-S", "Mid-SW", "Mid-W", "Mid-NW"], "Mids", "Rate")
    plot_lines(axes[6], df_plot, ["Low-N", "Low-E", "Low-S", "Low-W"], "Lows", "Rate")

    configure_time_axis(axes[-1], X_AXIS_CONFIG["tick_frequency"])

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
