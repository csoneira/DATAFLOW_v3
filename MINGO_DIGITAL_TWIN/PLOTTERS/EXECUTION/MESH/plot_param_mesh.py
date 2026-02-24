#!/usr/bin/env python3
"""Create a PDF summary of simulation parameter plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Use a modern, clean style
plt.style.use('seaborn-v0_8-whitegrid')
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

DT_ROOT = Path(__file__).resolve().parents[2]
PLOTTER_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot simulation parameter summary to PDF.  The "
        "completed CSV is treated as the full dataset, and the mesh file "
        "indicates which entries are currently in-process (overlay)."
    )
    parser.add_argument(
        "--mesh",
        dest="mesh",
        default=str(DT_ROOT / "INTERSTEPS" / "STEP_0_TO_1" / "param_mesh.csv"),
        help="Path to the in-process mesh CSV (e.g. STEP_0_TO_1/param_mesh.csv).",
    )
    parser.add_argument(
        "--completed",
        default=str(DT_ROOT / "SIMULATED_DATA" / "step_final_simulation_params.csv"),
        help="Path to the full/completed simulation params CSV (step_final_simulation_params.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(PLOTTER_DIR / "param_mesh_summary.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--n-value",
        type=float,
        default=2.0,
        help="cos_n value used for the n-specific flux histogram (default: 2).",
    )
    return parser.parse_args()







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
    if "selected_rows" not in df.columns and "requested_rows" in df.columns:
        df["selected_rows"] = df["requested_rows"]
    for numeric_col in ("selected_rows", "requested_rows"):
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
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
    mesh_path = Path(args.mesh)
    completed_path = Path(args.completed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load completed/full dataset first
    if not completed_path.exists():
        raise FileNotFoundError(f"Completed params file not found: {completed_path}")
    completed_df = pd.read_csv(completed_path)
    completed_df = normalize_step_params(completed_df)
    # ensure execution_time is interpreted as a true timestamp.  we
    # always plot using the *completed* dataset, so the value should
    # come from the full params file (`step_final_simulation_params.csv`).
    # previously the column was left as an object/string, which caused
    # matplotlib to treat every value as ``0`` when the series was
    # empty or unparsable; the date formatter then showed 1970–01–01.
    if "execution_time" in completed_df.columns:
        completed_df["execution_time"] = pd.to_datetime(
            completed_df["execution_time"], errors="coerce",
        )

    # load mesh (in-process) which may be empty or disjoint
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh params file not found: {mesh_path}")
    mesh_df = pd.read_csv(mesh_path)
    mesh_df = normalize_step_params(mesh_df)
    # mesh files sometimes contain an ``execution_time`` column too; convert
    # it so that merges don't cast the dtype to ``object``.  the mesh
    # value is not used for plotting but keeping the type consistent
    # avoids surprises when building ``merged_df`` later.
    if "execution_time" in mesh_df.columns:
        mesh_df["execution_time"] = pd.to_datetime(
            mesh_df["execution_time"], errors="coerce",
        )

    # determine shared columns for merging (used to detect overlap)
    # prefer explicit identifiers if available
    if "file_name" in mesh_df.columns and "file_name" in completed_df.columns:
        shared_cols = ["file_name"]
    elif "param_hash" in mesh_df.columns and "param_hash" in completed_df.columns:
        shared_cols = ["param_hash"]
    else:
        shared_cols = [
            col for col in mesh_df.columns
            if col in completed_df.columns and mesh_df[col].dtype == completed_df[col].dtype
        ]

    if mesh_df.empty:
        print("[plot_param_mesh] mesh file is empty; all rows treated as completed", flush=True)
    if not shared_cols:
        # no shared columns; overlay cannot be performed
        print("[plot_param_mesh] no shared columns between mesh and completed; overlay ignored", flush=True)
        merged_df = completed_df.copy()
        merged_df["_merge"] = "left_only"
    else:
        # perform a left merge so that the resulting DataFrame has the same
        # length and index as ``completed_df``.  we only care about marking
        # completed rows that also appear in the mesh; any mesh-only rows are
        # irrelevant for plotting and previously caused a length mismatch when
        # we tried to apply the boolean mask back to ``completed_df``.
        merged_df = pd.merge(
            completed_df, mesh_df, on=shared_cols, how="left", suffixes=("", "_mesh"), indicator=True
        )

    # rows present in mesh will have _merge == 'both'; mask is therefore
    # exactly the same length as ``completed_df`` and can be used safely.
    in_process_mask = merged_df.get("_merge") == "both"
    # the following two variables are currently unused but kept for clarity
    # and potential future diagnostics
    in_process_df = completed_df.loc[in_process_mask]
    completed_only_df = completed_df.loc[~in_process_mask]

    # we will plot using completed_df as the base
    df = completed_df

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
        # --- Parameter matrix plot (pairplot style) ---
        import yaml
        config_path = PLOTTER_DIR / "plot_param_mesh_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        raw_params = config.get("params", [])
        # Expand tokens using the *base* DataFrame (completed dataset)
        param_list = expand_params(raw_params, merged_df)
        # if execution_time column exists make sure it is included and at front
        if "execution_time" in merged_df.columns:
            if "execution_time" in param_list:
                param_list.remove("execution_time")
            param_list.insert(0, "execution_time")
        n = len(param_list)
        # determine index of exec time (may be 0 after above adjustment)
        et_idx = param_list.index("execution_time") if "execution_time" in param_list else -1
        if n > 0:
            # make first column (execution_time) twice as wide as others
            width_ratios = [3.0 if i == 0 else 1.0 for i in range(n)]
            fig, axes = plt.subplots(
                n, n, figsize=(2.8 * n, 2.8 * n), dpi=110,
                gridspec_kw={"width_ratios": width_ratios},
            )
            # Compute global axis limits for all parameters
            axis_limits = {}
            for idx, param in enumerate(param_list):
                col = merged_df[param]
                if col.dtype.kind in {'M', 'O'}:
                    continue
                vmin = col.min()
                vmax = col.max()
                if vmin == vmax:
                    vmin -= 1
                    vmax += 1
                # add 5% padding around the range so scatter points are not
                # clipped at the edges; padding factor applies to both sides
                span = vmax - vmin
                pad = span * 0.05
                axis_limits[param] = (vmin - pad, vmax + pad)
            for i in range(n):
                for j in range(n):
                    ax = axes[i, j] if n > 1 else axes
                    x = merged_df[param_list[j]]
                    y = merged_df[param_list[i]]
                    # Set identical axis limits for off-diagonal plots only.  Diagonal
                    # panels are histograms; using the raw parameter range for y makes the
                    # bars nearly invisible when counts are much smaller than the value
                    # range.  We'll let matplotlib autoscale the histogram y-axis later.
                    if param_list[j] in axis_limits:
                        ax.set_xlim(axis_limits[param_list[j]])
                    if i != j and param_list[i] in axis_limits:
                        ax.set_ylim(axis_limits[param_list[i]])
                    ax.set_aspect('auto')
                    # Clean, minimal style
                    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.3)
                    # draw a simple border around each subplot
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.7)
                        spine.set_color('black')
                    if i == j:
                        x_clean = x.dropna()
                        hist_bins = 16
                        if param_list[i] in {"selected_rows", "requested_rows"}:
                            unique_count = int(pd.Series(x_clean).nunique())
                            if unique_count > 0:
                                hist_bins = min(max(unique_count, 2), 40)
                        ax.hist(
                            x_clean,
                            bins=hist_bins,
                            color="#4f8cff",
                            alpha=0.5,
                            edgecolor='white',
                            linewidth=0.5,
                        )
                        # put variable name above histogram
                        ax.set_title(param_list[i], fontsize=10, pad=4)
                        # ensure y limits fit the bars
                        ax.autoscale_view(scalex=False, scaley=True)
                    elif i > j:
                        ax.scatter(x, y, s=7, alpha=0.4, color="#2ca02c")
                        # if this column is execution_time, connect the points to form a
                        # time series line.  sorting ensures the line follows increasing
                        # time even if the DataFrame index isn't ordered.
                        if param_list[j] == "execution_time":
                            try:
                                order = np.argsort(x.values)
                                ax.plot(x.values[order], y.values[order],
                                        color="#2ca02c", alpha=0.7, linewidth=0.8)
                            except Exception:
                                pass
                    else:
                        ax.axis("off")
                    # Remove axis labels except for outer plots and add labels
                    if i < n-1:
                        ax.set_xticklabels([])
                    else:
                        # bottom row: label x-axis with parameter
                        ax.set_xlabel(param_list[j], fontsize=10)
                    if j > 0:
                        ax.set_yticklabels([])
                    else:
                        # leftmost column: label y-axis
                        ax.set_ylabel(param_list[i], fontsize=10, rotation=90, labelpad=10)
                    ax.tick_params(axis='both', which='major', labelsize=9, width=1, length=3)
            fig.suptitle("Parameter Matrix", fontsize=14, y=1.01)
            fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.07, wspace=0.03, hspace=0.03)
            # Execution_time formatting (if present)
            if "execution_time" in param_list:
                et_idx = param_list.index("execution_time")
                date_fmt = mdates.DateFormatter("%Y-%m-%d")
                for _i in range(n):
                    for _j in range(n):
                        _ax = axes[_i, _j] if n > 1 else axes
                        if _j == et_idx:
                            _ax.xaxis.set_major_formatter(date_fmt)
                            _ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
                            # only label the lowest row
                            if _i == n-1:
                                plt.setp(_ax.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=8)
                            else:
                                _ax.set_xticklabels([])
                        if _i == et_idx and _i != _j:
                            _ax.yaxis.set_major_formatter(date_fmt)
                            _ax.yaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
                            plt.setp(_ax.yaxis.get_majorticklabels(), fontsize=8)
                now = pd.Timestamp.utcnow()
                for i in range(n):
                    for j in range(n):
                        ax = axes[i, j] if n > 1 else axes
                        if i < j or not ax.axison:
                            continue
                        if param_list[j] == "execution_time":
                            ax.axvline(now, color="red", linestyle="--", alpha=0.3, zorder=10)
                        if i > j and param_list[i] == "execution_time":
                            ax.axhline(now, color="red", linestyle="--", alpha=0.3, zorder=10)
            pdf.savefig(fig)
            plt.close(fig)





if __name__ == "__main__":
    main()
