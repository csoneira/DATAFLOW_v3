#!/usr/bin/env python3
"""Create a PDF summary of simulation parameter plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation parameter summary to PDF")
    parser.add_argument(
        "--input",
        default="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/step_final_simulation_params.csv",
        help="Path to the parameters CSV to plot",
    )
    parser.add_argument(
        "--output",
        default="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/PLOTTERS/param_mesh_summary.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--completed",
        default="/home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/step_final_simulation_params.csv",
        help="Path to step_final_simulation_params.csv",
    )
    parser.add_argument(
        "--n-value",
        type=float,
        default=2.0,
        help="cos_n value used for the n-specific flux histogram (default: 2).",
    )
    return parser.parse_args()


COMPLETED_COLOR = "#2ca02c"
IN_PROCESS_COLOR = "#ff7f0e"


def add_scatter(ax, completed: pd.DataFrame, in_process: pd.DataFrame) -> None:
    if not in_process.empty:
        ax.scatter(
            in_process["cos_n"],
            in_process["flux_cm2_min"],
            s=20,
            alpha=0.6,
            color=IN_PROCESS_COLOR,
            label="In process",
        )
    if not completed.empty:
        ax.scatter(
            completed["cos_n"],
            completed["flux_cm2_min"],
            s=26,
            alpha=0.7,
            color=COMPLETED_COLOR,
            label="Completed",
        )
    ax.set_xlabel("cos_n")
    ax.set_ylabel("flux_cm2_min")
    ax.set_title("cos_n vs flux_cm2_min")
    if not completed.empty or not in_process.empty:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


def add_status_hist(ax, completed, in_process, title: str, bins: int = 30) -> None:
    plotted = False
    if len(in_process):
        ax.hist(in_process, bins=bins, color=IN_PROCESS_COLOR, alpha=0.6, label="In process")
        plotted = True
    if len(completed):
        ax.hist(completed, bins=bins, color=COMPLETED_COLOR, alpha=0.8, label="Completed")
        plotted = True
    ax.set_title(title, fontsize=9)
    if plotted:
        ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)


def add_rows_per_file_hist(ax, df: pd.DataFrame) -> None:
    if "selected_rows" in df.columns:
        counts = pd.to_numeric(df["selected_rows"], errors="coerce")
        label = "selected_rows"
    elif "requested_rows" in df.columns:
        counts = pd.to_numeric(df["requested_rows"], errors="coerce")
        label = "requested_rows"
    else:
        ax.text(0.5, 0.5, "No selected_rows/requested_rows column found", ha="center", va="center")
        ax.set_axis_off()
        return
    counts = counts.dropna()
    if counts.empty:
        ax.text(0.5, 0.5, "No row counts available to plot", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.hist(counts, bins=150, color="#1f77b4", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("rows per file")
    ax.set_ylabel("count (log scale)")
    ax.set_title(f"Rows per file histogram ({label})")
    ax.grid(True, axis="y", alpha=0.2)


def equal_efficiency_mask(df: pd.DataFrame, atol: float = 1e-12) -> pd.Series:
    return (
        np.isclose(df["eff_p1"], df["eff_p2"], atol=atol)
        & np.isclose(df["eff_p1"], df["eff_p3"], atol=atol)
        & np.isclose(df["eff_p1"], df["eff_p4"], atol=atol)
    )


def parse_efficiencies_column(df: pd.DataFrame) -> pd.DataFrame:
    if {"eff_p1", "eff_p2", "eff_p3", "eff_p4"}.issubset(df.columns):
        return df
    if "efficiencies" not in df.columns:
        raise ValueError("Missing efficiency columns (eff_p1..eff_p4 or efficiencies).")

    eff_series = df["efficiencies"]
    if eff_series.apply(lambda value: isinstance(value, (list, tuple))).any():
        eff_frame = pd.DataFrame(eff_series.tolist(), columns=["eff_p1", "eff_p2", "eff_p3", "eff_p4"])
    else:
        parsed = []
        for value in eff_series:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                parsed.append([np.nan, np.nan, np.nan, np.nan])
                continue
            if isinstance(value, (list, tuple)):
                parsed.append(list(value))
                continue
            text = str(value).strip()
            if not text:
                parsed.append([np.nan, np.nan, np.nan, np.nan])
                continue
            try:
                result = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                result = None
            if isinstance(result, (list, tuple)) and len(result) >= 4:
                parsed.append(list(result[:4]))
            else:
                parts = [part.strip() for part in text.strip("[]").split(",")]
                if len(parts) >= 4:
                    parsed.append(parts[:4])
                else:
                    parsed.append([np.nan, np.nan, np.nan, np.nan])
        eff_frame = pd.DataFrame(parsed, columns=["eff_p1", "eff_p2", "eff_p3", "eff_p4"])

    for col in ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]:
        df[col] = pd.to_numeric(eff_frame[col], errors="coerce")
    return df


def normalize_step_params(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "z_plane_1": "z_p1",
        "z_plane_2": "z_p2",
        "z_plane_3": "z_p3",
        "z_plane_4": "z_p4",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = parse_efficiencies_column(df)
    return df


def build_completed_mask(mesh_df: pd.DataFrame, completed_df: pd.DataFrame, decimals: int = 6) -> pd.Series:
    key_cols = [
        "cos_n",
        "flux_cm2_min",
        "eff_p1",
        "eff_p2",
        "eff_p3",
        "eff_p4",
        "z_p1",
        "z_p2",
        "z_p3",
        "z_p4",
    ]
    missing_mesh = [col for col in key_cols if col not in mesh_df.columns]
    if missing_mesh:
        raise ValueError(f"Missing required columns in param mesh: {', '.join(missing_mesh)}")

    missing_completed = [col for col in key_cols if col not in completed_df.columns]
    if missing_completed:
        raise ValueError(f"Missing required columns in completed params: {', '.join(missing_completed)}")

    mesh_keys = mesh_df[key_cols].round(decimals).copy()
    mesh_keys["_mesh_index"] = mesh_df.index
    completed_keys = completed_df[key_cols].round(decimals).drop_duplicates()

    merged = mesh_keys.merge(completed_keys, on=key_cols, how="left", indicator=True)
    status = merged["_merge"].eq("both").groupby(merged["_mesh_index"]).any()
    return status.reindex(mesh_df.index, fill_value=False)


def split_by_status(df: pd.DataFrame, completed_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed_df = df.loc[completed_mask]
    in_process_df = df.loc[~completed_mask]
    return completed_df, in_process_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    completed_path = Path(args.completed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = normalize_step_params(df)
    if not completed_path.exists():
        raise FileNotFoundError(f"Completed params file not found: {completed_path}")
    completed_params = pd.read_csv(completed_path)
    completed_params = normalize_step_params(completed_params)

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

    completed_mask = build_completed_mask(df, completed_params)
    completed_df, in_process_df = split_by_status(df, completed_mask)

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
        add_scatter(ax, completed_df, in_process_df)
        ax.set_title(
            f"cos_n vs flux_cm2_min (completed: {len(completed_df)}, in process: {len(in_process_df)})"
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        add_rows_per_file_hist(ax, df)
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
            comp_data = data.loc[completed_mask]
            in_process_data = data.loc[~completed_mask]
            add_status_hist(ax, comp_data, in_process_data, title)

        fig.suptitle("Efficiency Histograms", fontsize=12)
        legend_handles = []
        if len(in_process_df):
            legend_handles.append(Patch(color=IN_PROCESS_COLOR, label="In process"))
        if len(completed_df):
            legend_handles.append(Patch(color=COMPLETED_COLOR, label="Completed"))
        if legend_handles:
            fig.legend(handles=legend_handles, loc="upper right", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        n_mask = np.isclose(df["cos_n"], args.n_value, atol=1e-12)
        same_eff_df = df.loc[equal_efficiency_mask(df)].copy()

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        if n_mask.any():
            add_status_hist(
                ax,
                df.loc[completed_mask & n_mask, "flux_cm2_min"],
                df.loc[~completed_mask & n_mask, "flux_cm2_min"],
                f"Flux histogram for cos_n = {args.n_value:g}",
            )
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
            same_eff_mask = equal_efficiency_mask(df)
            add_status_hist(
                ax,
                df.loc[completed_mask & same_eff_mask, "flux_cm2_min"],
                df.loc[~completed_mask & same_eff_mask, "flux_cm2_min"],
                "Flux histogram for rows with equal efficiencies",
            )
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
            same_eff_mask = equal_efficiency_mask(df)
            same_eff_completed = df.loc[completed_mask & same_eff_mask]
            same_eff_in_process = df.loc[~completed_mask & same_eff_mask]
            if not same_eff_in_process.empty:
                ax.scatter(
                    same_eff_in_process["flux_cm2_min"],
                    same_eff_in_process["eff_p1"],
                    s=20,
                    alpha=0.6,
                    color=IN_PROCESS_COLOR,
                    label="In process",
                )
            if not same_eff_completed.empty:
                ax.scatter(
                    same_eff_completed["flux_cm2_min"],
                    same_eff_completed["eff_p1"],
                    s=24,
                    alpha=0.7,
                    color=COMPLETED_COLOR,
                    label="Completed",
                )
            ax.set_xlabel("flux_cm2_min")
            ax.set_ylabel("efficiency")
            ax.set_title("Flux vs efficiency for rows with equal efficiencies")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
