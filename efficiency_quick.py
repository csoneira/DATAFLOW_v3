#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quick plot of plane efficiencies from TASK_1 metadata.

By default, reads the MINGO01 TASK_1 metadata CSV and plots plane 2 and plane 3
three-plane efficiencies using raw TT rates:
  eff_2 = 1 - raw_tt_134_rate_hz / raw_tt_1234_rate_hz
  eff_3 = 1 - raw_tt_124_rate_hz / raw_tt_1234_rate_hz
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/home/mingo/DATAFLOW_v3/STATIONS/MINGO01/STAGE_1/EVENT_DATA/STEP_1/TASK_1/METADATA/task_1_metadata_specific.csv"
)
MI_FILENAME_PATTERN = re.compile(r"mi0\d(?P<digits>\d{11})$", re.IGNORECASE)


def parse_time_from_basename(value: object) -> pd.Timestamp:
    """Parse datetime from filename_base values."""
    if not isinstance(value, str):
        return pd.NaT

    stem = Path(value).stem

    # Direct format used elsewhere in this repo: YYYY-mm-dd_HH.MM.SS
    try:
        return pd.Timestamp(datetime.strptime(stem, "%Y-%m-%d_%H.%M.%S"))
    except ValueError:
        pass

    # MINGO-style: mi0<station><YY><DDD><HH><MM><SS>
    match = MI_FILENAME_PATTERN.search(stem)
    if not match:
        return pd.NaT

    digits = match.group("digits")
    try:
        year = 2000 + int(digits[0:2])
        day_of_year = int(digits[2:5])
        hour = int(digits[5:7])
        minute = int(digits[7:9])
        second = int(digits[9:11])
        base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    except ValueError:
        return pd.NaT

    return pd.Timestamp(
        base_date.replace(hour=hour, minute=minute, second=second)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot quick raw-rate efficiencies for planes 2 and 3."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input metadata CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image (png/pdf). If omitted, shows the plot.",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Clip efficiencies to [0, 1] before plotting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    needed = [
        "filename_base",
        "raw_tt_1234_rate_hz",
        "raw_tt_134_rate_hz",
        "raw_tt_124_rate_hz",
    ]

    df = pd.read_csv(args.input, usecols=needed)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {args.input}: {missing}")

    data = df.dropna(subset=needed).copy()
    data["time"] = data["filename_base"].map(parse_time_from_basename)
    data = data.dropna(subset=["time"]).sort_values("time")

    denom = data["raw_tt_1234_rate_hz"].replace(0, np.nan)
    eff_2 = 1.0 - data["raw_tt_134_rate_hz"] / denom
    eff_3 = 1.0 - data["raw_tt_124_rate_hz"] / denom

    if args.clip:
        eff_2 = eff_2.clip(0, 1)
        eff_3 = eff_3.clip(0, 1)

    med_2 = eff_2.median()
    med_3 = eff_3.median()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(data["time"], eff_2, marker="o", markersize=3, linewidth=1.2, label="Plane 2: 1 - 134/1234")
    ax.plot(data["time"], eff_3, marker="s", markersize=3, linewidth=1.2, label="Plane 3: 1 - 124/1234")

    ax.axhline(med_2, linestyle="--", linewidth=1.2, label=f"Median eff2: {med_2:.4f}")
    ax.axhline(med_3, linestyle="--", linewidth=1.2, label=f"Median eff3: {med_3:.4f}")

    ax.set_title("MINGO01 TASK_1 raw-rate efficiencies (planes 2 and 3)")
    ax.set_xlabel("Time from filename_base")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    plt.tight_layout()

    print(f"Rows used: {len(data)}")
    print(f"Median efficiency plane 2: {med_2:.6f}")
    print(f"Median efficiency plane 3: {med_3:.6f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=160)
        print(f"Saved plot to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
