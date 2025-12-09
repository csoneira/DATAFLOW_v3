#!/usr/bin/env python3
"""
Generate calibration plots and per-configuration medians for STEP_1 TASK_2.

Supports both CLI execution (with station/pdf flags) and notebook usage via
importing and calling `run_analysis(station=..., pdf_generate=...)`.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from qa_shared import load_metadata, print_columns

# ---------------------------------------------------------------------------
# User-tunable defaults (used when running from notebooks)
# ---------------------------------------------------------------------------
DEFAULT_STATION = 3
STEP = 1
TASK = 2
START_DATE = "2024-03-01 00:00:00"
END_DATE = "2025-11-20 00:00:00"
PDF_GENERATE_DEFAULT = False

# Variables used for calibration lookups
CALIBRATION_VARIABLES = ["T_sum", "T_dif", "Q_sum"] + [f"Q_FB_coeff_{i}" for i in range(1, 6)]

# Optional per-variable filters (min/max). Leave empty dict for no filtering.
VARIABLE_FILTERS: Dict[str, Dict[str, float]] = {
    # Example:
    "Q_sum": {"min": 75, "max": 105},
    "T_sum": {"min": -5, "max": 5},
    "T_dif": {"min": -10, "max": 10},
    "Q_FB_coeff_1": {"min": -5, "max": 5},
    "Q_FB_coeff_2": {"min": -2, "max": 2},
    "Q_FB_coeff_3": {"min": -1, "max": 1},
    "Q_FB_coeff_4": {"min": -0.5, "max": 0.5},
    "Q_FB_coeff_5": {"min": -0.2, "max": 0.2},
}

OUTPUT_DIR = Path(__file__).resolve().parent / "OUTPUT_FILES"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality assurance calibrations for STEP_1 TASK_2.")
    parser.add_argument("-s", "--station", type=int, default=DEFAULT_STATION, help="Station number (1-4).")
    parser.add_argument("--pdf", action="store_true", default=PDF_GENERATE_DEFAULT, help="Generate plots into a PDF.")
    return parser.parse_args(argv)


def load_runs_dataframe(station: int) -> pd.DataFrame:
    csv_path = Path(
        f"/home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/ONLINE_RUN_DICTIONARY/STATION_{station}/input_file_mingo0{station}.csv"
    )
    runs_df = pd.read_csv(csv_path, skiprows=1)
    runs_df["start"] = pd.to_datetime(runs_df["start"])
    runs_df["end"] = pd.to_datetime(runs_df["end"]).fillna(pd.Timestamp.now())

    print(runs_df)

    return runs_df


def apply_variable_filters(series: pd.Series, variable: str) -> pd.Series:
    config = VARIABLE_FILTERS.get(variable)
    if not config:
        return pd.to_numeric(series, errors="coerce")
    filtered = pd.to_numeric(series, errors="coerce")
    min_val = config.get("min")
    max_val = config.get("max")
    if min_val is not None:
        filtered = filtered.where(filtered >= min_val)
    if max_val is not None:
        filtered = filtered.where(filtered <= max_val)
    return filtered


def safe_literal_eval(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return value
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return np.nan
    return value


def expand_q_fb_coefficients(df: pd.DataFrame, max_coeffs: int = 5) -> None:
    for plane in range(1, 5):
        for strip in range(1, 5):
            base_col = f"P{plane}_s{strip}_Q_FB_coeffs"
            if base_col not in df.columns:
                continue
            df[base_col] = df[base_col].apply(safe_literal_eval)
            for idx in range(max_coeffs):
                new_col = f"P{plane}_s{strip}_Q_FB_coeff_{idx+1}"
                df[new_col] = df[base_col].apply(
                    lambda arr, i=idx: arr[i] if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) > i else np.nan
                )


def compute_calibration_constants(
    df: pd.DataFrame,
    runs_df: pd.DataFrame,
    variables: List[str],
    planes: Iterable[int],
    strips: Iterable[int],
    time_col: str,
) -> pd.DataFrame:
    records: List[dict] = []
    for _, row in runs_df.iterrows():
        conf = row["conf"]
        start = row["start"]
        end = row["end"]
        subset = df[(df[time_col] >= start) & (df[time_col] < end)]
        if subset.empty:
            continue
        for variable in variables:
            for plane in planes:
                for strip in strips:
                    col = f"P{plane}_s{strip}_{variable}"
                    if col not in subset.columns:
                        continue
                    series = apply_variable_filters(subset[col], variable).dropna()
                    if series.empty:
                        continue
                    median_val = series.median()
                    records.append(
                        {
                            "conf": conf,
                            "plane": plane,
                            "strip": strip,
                            "variable": variable,
                            "parameter": median_val,
                            "start": start,
                            "end": end,
                        }
                    )
    return pd.DataFrame(records)


def group_segments(calibration_df: pd.DataFrame) -> Dict[tuple, List[dict]]:
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for _, row in calibration_df.iterrows():
        key = (row["plane"], row["strip"], row["variable"])
        grouped[key].append(row.to_dict())
    return grouped


def draw_calibration_segments(ax, plane: int, strip: int, variable: str, segments_map: Dict[tuple, List[dict]]):
    key = (plane, strip, variable)
    if key not in segments_map:
        return
    used_labels = set()
    for seg in segments_map[key]:
        label = f"Conf {seg['conf']} median"
        if label in used_labels:
            label = None
        else:
            used_labels.add(label)
        ax.hlines(
            y=seg["parameter"],
            xmin=seg["start"],
            xmax=seg["end"],
            colors="tab:red",
            linewidth=1.2,
            label=label,
        )


def plot_variable(
    df: pd.DataFrame,
    runs_df: pd.DataFrame,
    variable: str,
    time_col: str,
    segments_map: Dict[tuple, List[dict]],
    marker_size: float = 1.0,
    sigma_limit: float = 3.0,
    rasterized: bool = False,
) -> plt.Figure:
    planes = range(1, 5)
    strips = range(1, 5)
    fig = plt.figure(figsize=(28, 14))
    outer_gs = gridspec.GridSpec(len(planes), len(strips), hspace=0.35, wspace=0.25)
    cmap = plt.get_cmap("turbo")
    max_conf = runs_df["conf"].max() if not runs_df.empty else 1

    # Determine consistent y-limits across all subplots
    global_min = None
    global_max = None
    for plane in planes:
        for strip in strips:
            col = f"P{plane}_s{strip}_{variable}"
            if col not in df.columns:
                continue
            filtered = apply_variable_filters(df[col], variable).dropna()
            if filtered.empty:
                continue
            local_min = filtered.min()
            local_max = filtered.max()
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)
    if global_min is None or global_max is None or np.isclose(global_min, global_max):
        global_min, global_max = -1, 1

    for i_plane, plane in enumerate(planes):
        for j_strip, strip in enumerate(strips):
            col = f"P{plane}_s{strip}_{variable}"
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[i_plane, j_strip], height_ratios=[3, 1], hspace=0.05
            )
            ax_main = fig.add_subplot(inner[0, 0])
            ax_norm = fig.add_subplot(inner[1, 0], sharex=ax_main)

            if col in df.columns:
                series = apply_variable_filters(df[col], variable)
                times = df[time_col]

                ax_main.plot(times, series, marker=".", linestyle="", markersize=marker_size, label="Parameter")
                draw_calibration_segments(ax_main, plane, strip, variable, segments_map)

                for _, run in runs_df.iterrows():
                    color = cmap(run["conf"] / max_conf if max_conf else 0)
                    ax_main.axvspan(run["start"], run["end"], color=color, alpha=0.2)
                    ax_norm.axvspan(run["start"], run["end"], color=color, alpha=0.1)

                if j_strip == 0:
                    ax_main.set_ylabel("Value")
                    ax_norm.set_ylabel("Z")
                if i_plane == len(planes) - 1:
                    ax_norm.set_xlabel("Datetime")

                ax_main.set_title(f"P{plane} S{strip}", fontsize=9)
                ax_main.set_ylim(global_min, global_max)
                ax_main.grid(True)

                clean_series = series.dropna()
                if not clean_series.empty:
                    mean_val = clean_series.mean()
                    std_val = clean_series.std()
                    if std_val > 0:
                        norm_vals = (clean_series - mean_val) / std_val
                    else:
                        norm_vals = clean_series - mean_val
                    ax_norm.plot(times.loc[clean_series.index], norm_vals, marker="o", linestyle="", markersize=marker_size)
                ax_norm.set_ylim(-sigma_limit, sigma_limit)
                ax_norm.axhline(0, linestyle="--", linewidth=0.8, color="gray")
                ax_norm.grid(True)
            else:
                ax_main.set_visible(False)
                ax_norm.set_visible(False)
                continue

            plt.setp(ax_main.get_xticklabels(), visible=False)

    fig.suptitle(f"{variable} calibration parameter (raw & normalized)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if rasterized:
        for axis in fig.axes:
            axis.set_rasterized(True)
    return fig


def run_analysis(station: int = DEFAULT_STATION, pdf_generate: bool = PDF_GENERATE_DEFAULT):
    station = int(station)
    if station < 1 or station > 4:
        raise ValueError("Station must be between 1 and 4.")

    ctx = load_metadata(f"MINGO0{station}", STEP, TASK, START_DATE, END_DATE)
    df = ctx.df.copy()
    runs_df = load_runs_dataframe(station)

    print(f"Loaded: {ctx.metadata_path}")
    print(f"Rows: {len(df)}")
    print("Columns:")
    print_columns(df)

    time_col = ctx.time_col
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col])

    expand_q_fb_coefficients(df)

    calibration_df = compute_calibration_constants(df, runs_df, CALIBRATION_VARIABLES, range(1, 5), range(1, 5), time_col)
    segments_map = group_segments(calibration_df) if not calibration_df.empty else {}

    if calibration_df.empty:
        print("No calibration constants computed for the provided ranges.")
    else:
        csv_path = OUTPUT_DIR / f"MINGO0{station}_task{TASK}_calibration_parameters.csv"
        calibration_df[["conf", "plane", "strip", "variable", "parameter"]].to_csv(csv_path, index=False)
        print(f"Saved calibration parameters to {csv_path}")

    pdf: PdfPages | None = None
    if pdf_generate:
        pdf_path = OUTPUT_DIR / f"MINGO0{station}_task{TASK}_calibrations.pdf"
        pdf = PdfPages(pdf_path)
        print(f"Generating PDF at {pdf_path}")

        for variable in CALIBRATION_VARIABLES:
            fig = plot_variable(
                df, runs_df, variable, time_col, segments_map, rasterized=bool(pdf)
            )
            if pdf:
                pdf.savefig(fig, dpi=200)
                plt.close(fig)
            else:
                plt.show()

        if pdf:
            pdf.close()
            print("PDF generation complete.")


if __name__ == "__main__":
    cli_args = parse_args()
    run_analysis(station=cli_args.station, pdf_generate=cli_args.pdf)

# %%
