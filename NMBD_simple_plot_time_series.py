#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple NMDB time-series plotter.

Reads NMDB export files that include metadata/comment blocks and a semicolon-
separated table, then plots one or more station columns as a time series.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().with_name("NMBD_first_week_of_december_25.csv")


def parse_nmdb_table(path: Path) -> pd.DataFrame:
    """Parse an NMDB exported text/CSV file into a DataFrame indexed by datetime."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    first_data_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ";" not in line:
            continue
        first_field = line.split(";")[0].strip()
        if pd.notna(pd.to_datetime(first_field, errors="coerce")):
            first_data_idx = idx
            break

    if first_data_idx < 1:
        raise ValueError("Could not locate NMDB table data rows in input file.")

    header_line = lines[first_data_idx - 1].strip()
    header_parts = header_line.split()
    if not header_parts:
        raise ValueError("Could not parse station header row above NMDB table data.")
    columns = ["timestamp", *header_parts]

    records: List[List[object]] = []
    for raw in lines[first_data_idx:]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ";" not in raw:
            continue

        parts = [part.strip() for part in raw.split(";") if part.strip()]
        if len(parts) != len(columns):
            continue

        timestamp = pd.to_datetime(parts[0], errors="coerce", utc=True)
        if pd.isna(timestamp):
            continue

        row: List[object] = [timestamp]
        numeric_ok = True
        for value in parts[1:]:
            num = pd.to_numeric(value, errors="coerce")
            if pd.isna(num):
                numeric_ok = False
                break
            row.append(float(num))

        if numeric_ok:
            records.append(row)

    if not records:
        raise ValueError("No data rows could be parsed from input file.")

    df = pd.DataFrame(records, columns=columns).drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def select_columns(df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    if not requested:
        return list(df.columns)

    valid = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        missing_str = ", ".join(missing)
        raise ValueError(f"Unknown columns: {missing_str}. Available: {available}")
    return valid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot NMDB time series from a local export file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to NMDB export file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=[],
        help="Columns/stations to plot. If omitted, all columns are plotted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (e.g. plot.png). If omitted, show interactively.",
    )
    parser.add_argument(
        "--title",
        default="NMDB Time Series",
        help="Plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI when saving.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = parse_nmdb_table(args.input)
    cols = select_columns(df, args.columns)

    ax = df[cols].plot(figsize=(12, 5), linewidth=1.2)
    ax.set_title(args.title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Station")

    plt.tight_layout()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=args.dpi)
        print(f"Saved plot to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
