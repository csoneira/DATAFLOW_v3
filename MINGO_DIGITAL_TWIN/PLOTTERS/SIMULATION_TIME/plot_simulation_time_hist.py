#!/usr/bin/env python3
"""Plot histogram(s) of simulation execution times recorded by run_step.sh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/PLOTTERS/SIMULATION_TIME")
DEFAULT_INPUT = BASE_DIR / "simulation_execution_times.csv"
DEFAULT_OUTPUT = BASE_DIR / "simulation_execution_time_hist.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of run_step execution times from CSV.",
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
        default="Simulation Execution Time Histogram",
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
    if "elapsed_seconds" not in df.columns:
        raise ValueError("CSV missing required column: elapsed_seconds")

    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    df = df[df["elapsed_seconds"].notna()]
    df = df[df["elapsed_seconds"] >= 0]

    if "timestamp_utc" in df.columns and (args.date_from or args.date_to):
        timestamps = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df[timestamps.notna()].copy()
        timestamps = timestamps.loc[df.index]
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
        df = df[df["elapsed_seconds"] >= args.min_seconds]
    if args.max_seconds is not None:
        df = df[df["elapsed_seconds"] <= args.max_seconds]

    if args.last_n and args.last_n > 0:
        df = df.tail(args.last_n)

    if df.empty:
        raise ValueError("No rows left after filters.")
    return df


def main() -> None:
    args = parse_args()
    df = load_filtered_times(args)
    values = df["elapsed_seconds"].to_numpy(dtype=float)

    bins = args.bins if args.bins and args.bins > 0 else auto_bins(values)

    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    p90_v = float(np.percentile(values, 90))
    min_v = float(np.min(values))
    max_v = float(np.max(values))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")

    ax.axvline(mean_v, color="#d62728", linestyle="--", linewidth=2, label=f"mean={mean_v:.2f}s")
    ax.axvline(median_v, color="#2ca02c", linestyle="-.", linewidth=2, label=f"median={median_v:.2f}s")
    ax.axvline(p90_v, color="#ff7f0e", linestyle=":", linewidth=2, label=f"p90={p90_v:.2f}s")

    ax.set_title(args.title)
    ax.set_xlabel("Execution time [s]")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")

    # Log-scale y-axis
    ax.set_yscale("log")

    filter_text = (
        f"N={values.size}\n"
        f"min={min_v:.2f}s\n"
        f"max={max_v:.2f}s\n"
        f"date_from={args.date_from or '-'}\n"
        f"date_to={args.date_to or '-'}"
    )
    ax.text(
        0.02,
        0.98,
        filter_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved histogram: {args.output}")
    print(
        f"N={values.size} min={min_v:.3f}s max={max_v:.3f}s "
        f"mean={mean_v:.3f}s median={median_v:.3f}s p90={p90_v:.3f}s"
    )


if __name__ == "__main__":
    main()
