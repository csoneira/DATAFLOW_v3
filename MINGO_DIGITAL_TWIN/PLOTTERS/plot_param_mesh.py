#!/usr/bin/env python3
"""Create a PDF summary of param_mesh.csv plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot param_mesh.csv summary to PDF")
    parser.add_argument(
        "--input",
        default="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv",
        help="Path to param_mesh.csv",
    )
    parser.add_argument(
        "--output",
        default="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/PLOTTERS/param_mesh_summary.pdf",
        help="Output PDF path",
    )
    return parser.parse_args()


def add_scatter(ax, df: pd.DataFrame) -> None:
    ax.scatter(df["cos_n"], df["flux_cm2_min"], s=20, alpha=0.7)
    ax.set_xlabel("cos_n")
    ax.set_ylabel("flux_cm2_min")
    ax.set_title("cos_n vs flux_cm2_min")
    ax.grid(True, alpha=0.2)


def add_hist(ax, data, title: str) -> None:
    ax.hist(data, bins=30, color="#1f77b4", alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = [
        "cos_n",
        "flux_cm2_min",
        "eff_p1",
        "eff_p2",
        "eff_p3",
        "eff_p4",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    eff_p1 = df["eff_p1"]
    eff_p2 = df["eff_p2"]
    eff_p3 = df["eff_p3"]
    eff_p4 = df["eff_p4"]

    prod_all = eff_p1 * eff_p2 * eff_p3 * eff_p4
    prod_12 = eff_p1 * eff_p2
    prod_23 = eff_p2 * eff_p3
    prod_34 = eff_p3 * eff_p4
    prod_41 = eff_p4 * eff_p1

    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        add_scatter(ax, df)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(10, 9))
        for ax in axes.ravel():
            ax.set_visible(False)

        layout = {
            (0, 0): (prod_12, "eff_p1 * eff_p2"),
            (0, 2): (prod_23, "eff_p2 * eff_p3"),
            (2, 0): (prod_41, "eff_p4 * eff_p1"),
            (2, 2): (prod_34, "eff_p3 * eff_p4"),
            (0, 1): (eff_p1, "eff_p1"),
            (1, 2): (eff_p2, "eff_p2"),
            (2, 1): (eff_p3, "eff_p3"),
            (1, 0): (eff_p4, "eff_p4"),
            (1, 1): (prod_all, "eff_p1 * eff_p2 * eff_p3 * eff_p4"),
        }

        for (row, col), (data, title) in layout.items():
            ax = axes[row, col]
            ax.set_visible(True)
            add_hist(ax, data, title)

        fig.suptitle("Efficiency Histograms", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
