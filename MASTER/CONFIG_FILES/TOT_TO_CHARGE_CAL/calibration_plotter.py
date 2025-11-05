#!/usr/bin/env python3
"""Plot TOT-to-charge calibration curves and export them as PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = THIS_DIR / "tot_to_charge_calibration.csv"
DEFAULT_OUTPUT = THIS_DIR / "tot_to_charge_calibration_plot.pdf"


@dataclass
class CalibrationData:
    width: pd.Series
    charge: pd.Series


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the TOT-to-charge calibration CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination PDF file.",
    )
    parser.add_argument(
        "--title",
        default="FEE HADES TOT-to-Charge Calibration",
        help="Custom title for the plot.",
    )
    return parser.parse_args(argv)


def load_calibration_data(csv_path: Path) -> CalibrationData:
    if not csv_path.exists():
        raise FileNotFoundError(f"Calibration CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"Width", "Fast_Charge"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_columns} in {csv_path}, found {set(df.columns)}"
        )

    df = df.copy()
    df["Width"] = pd.to_numeric(df["Width"], errors="coerce")
    df["Fast_Charge"] = pd.to_numeric(df["Fast_Charge"], errors="coerce")
    df = df.dropna(subset=["Width", "Fast_Charge"]).sort_values("Width")

    if df.empty:
        raise ValueError(f"No valid data points after cleaning {csv_path}")

    return CalibrationData(width=df["Width"], charge=df["Fast_Charge"])


def render_plot(data: CalibrationData, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    widths = data.width.to_numpy()
    charges = data.charge.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    crosstalk_x = 1.0
    streamer_x = 100.0
    crosstalk_color = "tab:purple"
    streamer_color = "tab:red"

    x_min = widths.min()

    # Calibration curve.
    ax.plot(
        widths,
        charges,
        color="tab:blue",
        linewidth=2.2,
        label="Calibration curve",
        zorder=3,
    )
    ax.scatter(
        widths,
        charges,
        color="white",
        edgecolor="tab:blue",
        s=30,
        linewidth=0.7,
        zorder=4,
        label="_nolegend_",
    )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Time-over-Threshold Width (ns)", fontsize=12)
    ax.set_ylabel("Fast Charge (fC)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    x_min_data = widths.min()
    x_max_data = widths.max()
    x_span = x_max_data - x_min_data
    if x_span <= 0:
        reference = float(abs(x_min_data) if len(widths) else 1.0)
        x_margin = 0.02 * max(reference, 1.0)
    else:
        x_margin = 0.02 * x_span
    x_axis_min = x_min_data - x_margin
    x_axis_max = x_max_data + x_margin
    ax.set_xlim(x_axis_min, x_axis_max)

    y_margin = 0.05 * (charges.max() - charges.min())
    y_bottom = charges.min() - y_margin
    y_top = charges.max() + y_margin
    ax.set_ylim(y_bottom, y_top)

    # Threshold guide lines and shaded operating zones.
    crosstalk_y = float(np.interp(crosstalk_x, widths, charges))
    streamer_y = float(np.interp(streamer_x, widths, charges))

    streamer_rect = Rectangle(
        (x_axis_min, y_bottom),
        max(streamer_x - x_axis_min, 0.0),
        max(streamer_y - y_bottom, 0.0),
        facecolor=streamer_color,
        alpha=0.15,
        edgecolor="none",
        zorder=0,
    )
    ax.add_patch(streamer_rect)

    crosstalk_rect = Rectangle(
        (x_axis_min, y_bottom),
        max(crosstalk_x - x_axis_min, 0.0),
        max(crosstalk_y - y_bottom, 0.0),
        facecolor=crosstalk_color,
        alpha=0.15,
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(crosstalk_rect)

    ax.plot(
        [x_axis_min, crosstalk_x],
        [crosstalk_y, crosstalk_y],
        color=crosstalk_color,
        linewidth=1.5,
        linestyle="--",
        label="Crosstalk threshold",
        zorder=2,
    )
    ax.plot(
        [crosstalk_x, crosstalk_x],
        [y_bottom, crosstalk_y],
        color=crosstalk_color,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label="_nolegend_",
    )

    ax.plot(
        [x_axis_min, streamer_x],
        [streamer_y, streamer_y],
        color=streamer_color,
        linewidth=1.5,
        linestyle="--",
        label="Streamer threshold",
        zorder=2,
    )
    ax.plot(
        [streamer_x, streamer_x],
        [y_bottom, streamer_y],
        color=streamer_color,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label="_nolegend_",
    )

    ax.tick_params(axis="both", which="major", labelsize=10)
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l and l != "_nolegend_"]
    if filtered:
        handles, labels = map(list, zip(*filtered))
    else:
        handles, labels = [], []
    handles.extend(
        [
            Patch(facecolor=crosstalk_color, alpha=0.12, edgecolor="none", label="Crosstalk region"),
            Patch(facecolor=streamer_color, alpha=0.08, edgecolor="none", label="Streamer region"),
        ]
    )
    labels.extend(["Crosstalk region", "Streamer region"])
    ax.legend(handles, labels, loc="upper left", frameon=True, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    data = load_calibration_data(args.input)
    render_plot(data, args.title, args.output)
    print(f"Saved calibration plot to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
