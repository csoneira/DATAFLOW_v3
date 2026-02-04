#!/usr/bin/env python3
"""Create a PDF summary of param_mesh.csv plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    parser.add_argument(
        "--n-value",
        type=float,
        default=2.0,
        help="cos_n value used for the n-specific flux histogram (default: 2).",
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


def equal_efficiency_mask(df: pd.DataFrame, atol: float = 1e-12) -> pd.Series:
    return (
        np.isclose(df["eff_p1"], df["eff_p2"], atol=atol)
        & np.isclose(df["eff_p1"], df["eff_p3"], atol=atol)
        & np.isclose(df["eff_p1"], df["eff_p4"], atol=atol)
    )


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

        n_mask = np.isclose(df["cos_n"], args.n_value, atol=1e-12)
        same_eff_df = df.loc[equal_efficiency_mask(df)].copy()

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if n_mask.any():
            add_hist(ax, df.loc[n_mask, "flux_cm2_min"], f"Flux histogram for cos_n = {args.n_value:g}")
            ax.set_xlabel("flux_cm2_min")
            ax.set_ylabel("count")
        else:
            ax.text(0.5, 0.5, f"No rows found for cos_n = {args.n_value:g}", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if same_eff_df.empty:
            ax.text(0.5, 0.5, "No rows where eff_p1 = eff_p2 = eff_p3 = eff_p4", ha="center", va="center")
            ax.set_axis_off()
        else:
            add_hist(ax, same_eff_df["flux_cm2_min"], "Flux histogram for rows with equal efficiencies")
            ax.set_xlabel("flux_cm2_min")
            ax.set_ylabel("count")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if same_eff_df.empty:
            ax.text(0.5, 0.5, "No rows where eff_p1 = eff_p2 = eff_p3 = eff_p4", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.scatter(same_eff_df["flux_cm2_min"], same_eff_df["eff_p1"], s=20, alpha=0.75)
            ax.set_xlabel("flux_cm2_min")
            ax.set_ylabel("efficiency")
            ax.set_title("Flux vs efficiency for rows with equal efficiencies")
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
