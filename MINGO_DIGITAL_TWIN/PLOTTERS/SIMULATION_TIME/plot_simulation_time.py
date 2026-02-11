#!/usr/bin/env python3
"""Plot per-step simulation execution-time histograms and time series."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


BASE_DIR = Path("/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/PLOTTERS/SIMULATION_TIME")
DEFAULT_INPUT = BASE_DIR / "simulation_execution_times.csv"
DEFAULT_OUTPUT = BASE_DIR / "simulation_execution_time_hist.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 10 step-wise execution-time histograms from CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="Keep rows at/after this UTC datetime (e.g. 2026-02-10T14:00:00Z).",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="Keep rows at/before this UTC datetime (e.g. 2026-02-10T20:00:00Z).",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=0,
        help="Use only the last N rows after filters (default: 0 means all).",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=None,
        help="Drop entries below this elapsed time.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Drop entries above this elapsed time.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=0,
        help="Histogram bins (default: auto via Freedman-Diaconis rule).",
    )
    parser.add_argument(
        "--title",
        default="Simulation Step Execution Time Histograms",
        help="Plot title.",
    )
    return parser.parse_args()


def auto_bins(values: np.ndarray) -> int:
    n = values.size
    if n <= 1:
        return 1
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return 1
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 0:
        return max(5, min(80, int(np.sqrt(n))))
    bin_width = 2.0 * iqr / np.cbrt(n)
    if bin_width <= 0:
        return max(5, min(80, int(np.sqrt(n))))
    bins = int(np.ceil((vmax - vmin) / bin_width))
    return max(5, min(120, bins))


def load_filtered_times(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {args.input}")

    if "exec_time_s" not in df.columns:
        if "elapsed_seconds" in df.columns:
            df["exec_time_s"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
        else:
            raise ValueError("CSV missing required column: exec_time_s")

    if "step" not in df.columns:
        raise ValueError("CSV missing required column: step")

    if "timestamp_utc" not in df.columns:
        raise ValueError("CSV missing required column: timestamp_utc")

    df["exec_time_s"] = pd.to_numeric(df["exec_time_s"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df[df["exec_time_s"].notna()]
    df = df[df["step"].notna()]
    df = df[df["timestamp_utc"].notna()]
    df = df[df["exec_time_s"] >= 0]
    df["step"] = df["step"].astype(int)
    df = df[df["step"].between(1, 10)]

    if args.date_from or args.date_to:
        timestamps = df["timestamp_utc"]
        if args.date_from:
            date_from = pd.to_datetime(args.date_from, utc=True, errors="coerce")
            if pd.isna(date_from):
                raise ValueError(f"Invalid --date-from value: {args.date_from}")
            df = df[timestamps >= date_from]
            timestamps = timestamps.loc[df.index]
        if args.date_to:
            date_to = pd.to_datetime(args.date_to, utc=True, errors="coerce")
            if pd.isna(date_to):
                raise ValueError(f"Invalid --date-to value: {args.date_to}")
            df = df[timestamps <= date_to]

    if args.min_seconds is not None:
        df = df[df["exec_time_s"] >= args.min_seconds]
    if args.max_seconds is not None:
        df = df[df["exec_time_s"] <= args.max_seconds]

    if args.last_n and args.last_n > 0:
        df = df.tail(args.last_n)

    df = df.sort_values("timestamp_utc")

    if df.empty:
        raise ValueError("No rows left after filters.")

    return df


def main() -> None:
    args = parse_args()
    df = load_filtered_times(args)
    values = df["exec_time_s"].to_numpy(dtype=float)

    bins = args.bins if args.bins and args.bins > 0 else auto_bins(values)

    min_v = float(np.min(values))
    max_v = float(np.max(values))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_hist, ax_ts) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )
    cmap = plt.colormaps["tab10"]
    plotted_steps: list[int] = []
    for step in range(1, 11):
        step_vals = df.loc[df["step"] == step, "exec_time_s"].to_numpy(dtype=float)
        if step_vals.size == 0:
            continue
        plotted_steps.append(step)
        ax_hist.hist(
            step_vals,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            color=cmap(step - 1),
            label=f"Step {step} (n={step_vals.size})",
        )

    if not plotted_steps:
        raise ValueError("No rows with valid steps (1-10) left after filters.")

    ax_hist.set_title(args.title)
    ax_hist.set_xlabel("Execution time [s] (exec_time_s)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(loc="upper right", ncol=2, fontsize=9)

    # Log-scale y-axis
    ax_hist.set_yscale("log")

    # Time series panel: timestamp_utc in x, exec_time_s in y (log-scale)
    for step in plotted_steps:
        step_df = df.loc[df["step"] == step, ["timestamp_utc", "exec_time_s"]].sort_values(
            "timestamp_utc"
        )
        ax_ts.plot(
            step_df["timestamp_utc"].to_numpy(),
            step_df["exec_time_s"].to_numpy(dtype=float),
            linestyle="-",
            marker="o",
            markersize=2.2,
            linewidth=1.1,
            color=cmap(step - 1),
            label=f"Step {step}",
            alpha=0.9,
        )

    ax_ts.set_title("Execution Time Time-Series by Step")
    ax_ts.set_xlabel("timestamp_utc")
    ax_ts.set_ylabel("Execution time [s] (exec_time_s)")
    ax_ts.set_yscale("log")
    ax_ts.legend(loc="upper right", ncol=2, fontsize=9)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S", tz=mdates.UTC))
    ax_ts.tick_params(axis="x", rotation=0)

    filter_text = (
        f"N={values.size}\n"
        f"steps={','.join(str(s) for s in plotted_steps)}\n"
        f"min={min_v:.2f}s\n"
        f"max={max_v:.2f}s\n"
        f"date_from={args.date_from or '-'}\n"
        f"date_to={args.date_to or '-'}"
    )
    ax_hist.text(
        0.02,
        0.98,
        filter_text,
        transform=ax_hist.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved figure: {args.output}")
    print(f"N={values.size} min={min_v:.3f}s max={max_v:.3f}s bins={bins}")
    step_counts = df["step"].value_counts().sort_index()
    print("Rows per step:", ", ".join(f"{idx}:{cnt}" for idx, cnt in step_counts.items()))


if __name__ == "__main__":
    main()
