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

DT_ROOT = Path(__file__).resolve().parents[2]
PLOTTER_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation parameter summary to PDF")
    parser.add_argument(
        "--input",
        default=str(DT_ROOT / "INTERSTEPS" / "STEP_0_TO_1" / "param_mesh.csv"),
        help="Path to the proposed/base parameter mesh CSV (e.g. STEP_0_TO_1/param_mesh.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(PLOTTER_DIR / "param_mesh_summary.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--completed",
        default=str(DT_ROOT / "SIMULATED_DATA" / "step_final_simulation_params.csv"),
        help="Path to the completed simulation params CSV (step_final_simulation_params.csv).",
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


def expand_params(raw_params: list[str], merged_df: pd.DataFrame) -> list[str]:
    """Expand config tokens into actual numeric column names present in merged_df.

    Supported expansions:
    - "efficiencies" -> expands to ["eff_p1", "eff_p2", "eff_p3", "eff_p4"] if those
      columns exist and are numeric.
    - "eff_1" / "eff1" -> maps to "eff_p1" (supports indices 1..4).

    Returns a list of unique numeric column names that exist in merged_df, preserving order.
    """
    expanded: list[str] = []
    for p in raw_params:
        if isinstance(p, str) and p.lower().startswith("eff"):
            # token formats: "efficiencies", "efficiencies", "eff_1", "eff1", "eff_p1"
            if p == "efficiencies":
                eff_cols = [c for c in ["eff_p1", "eff_p2", "eff_p3", "eff_p4"]
                            if c in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[c])]
                expanded.extend(eff_cols)
                continue
            # match eff_1, eff1, eff-p1, eff_p1
            import re

            m = re.match(r"^eff[_\-]?([1-4])$", p, flags=re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                col = f"eff_p{idx}"
                if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col]):
                    expanded.append(col)
                continue
            # if user already specified eff_p1..eff_p4, we'll treat below in default branch
        # default: keep the token as-is (will be filtered later)
        expanded.append(p)

    # Final filter: only keep numeric columns that exist in merged_df and preserve order/uniqueness
    final: list[str] = []
    for p in expanded:
        if p in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[p]):
            if p not in final:
                final.append(p)
    return final


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

    if not input_path.exists():
        raise FileNotFoundError(f"Proposed/base params file not found: {input_path}")
    df = pd.read_csv(input_path)
    df = normalize_step_params(df)
    if not completed_path.exists():
        raise FileNotFoundError(f"Completed params file not found: {completed_path}")
    completed_params = pd.read_csv(completed_path)
    completed_params = normalize_step_params(completed_params)

    # Outer join on columns with matching names and dtypes only
    shared_cols = [col for col in df.columns if col in completed_params.columns and df[col].dtype == completed_params[col].dtype]
    merged_df = pd.merge(df, completed_params, on=shared_cols, how="outer", suffixes=("_input", "_completed"))
    # merged_df now contains all rows from both input and completed CSVs, matching only on columns with same dtype

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
    same_eff_mask = equal_efficiency_mask(df)
    n_mask = np.isclose(df["cos_n"], args.n_value, atol=1e-12)
    same_eff_n_mask = same_eff_mask & n_mask

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
        # --- Parameter matrix plot (pairplot style) ---
        import yaml
        config_path = PLOTTER_DIR / "plot_param_mesh_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        raw_params = config.get("params", [])
        # Expand tokens (supports "efficiencies", "eff_1" / "eff1", and "eff_pN")
        param_list = expand_params(raw_params, merged_df)
        n = len(param_list)
        if n > 0:
            fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
            for i in range(n):
                for j in range(n):
                    ax = axes[i, j] if n > 1 else axes
                    x = merged_df[param_list[j]]
                    y = merged_df[param_list[i]]
                    if i == j:
                        ax.hist(x.dropna(), bins=40, color="#1f77b4", alpha=0.8)
                        ax.set_ylabel("")
                        ax.set_xlabel(param_list[i])
                    elif i > j:
                        ax.scatter(x, y, s=10, alpha=0.5, color="#2ca02c")
                        ax.set_xlabel(param_list[j])
                        ax.set_ylabel(param_list[i])
                    else:
                        ax.axis("off")
            fig.suptitle("Parameter Matrix (Histograms & Scatter Plots)", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)



if __name__ == "__main__":
    main()
