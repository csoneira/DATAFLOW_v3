from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import argparse
from pathlib import Path
import shutil

# Third-party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# -----------------------------------------------------------------------------

point_size = 2

# -----------------------------------------------------------------------------#
# I/O
# -----------------------------------------------------------------------------#
def _read_csv(csv_path: Path) -> pd.DataFrame:
      """Read CSV, parse Start_Time, set as index, force numeric else."""
      df = pd.read_csv(csv_path, low_memory=False)
      df["Start_Time"] = pd.to_datetime(df["Start_Time"])
      df.set_index("Start_Time", inplace=True)
      for c in df.columns:
          if c != "End_Time":
              df[c] = pd.to_numeric(df[c], errors="coerce")
      return df


def read_station_metadata(station: int) -> tuple[pd.DataFrame, pd.DataFrame]:
      """
      Return two dataframes: (raw_to_list_metadata, event_accumulator_metadata).
      """
      if station not in (1, 2, 3, 4):
          raise ValueError("station must be 1, 2, 3 or 4")

      base = (
          Path("/home/mingo/DATAFLOW_v3")
          / "STATIONS"
          / f"MINGO0{station}"
          / "FIRST_STAGE"
          / "EVENT_DATA"
      )

      df_cal = _read_csv(base / "raw_to_list_metadata.csv")
      df_evt = _read_csv(base / "event_accumulator_metadata.csv")
      return df_cal, df_evt

# -----------------------------------------------------------------------------#
# Plot helpers
# -----------------------------------------------------------------------------#
def _apply_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(axis="x", linestyle=":", linewidth=0.4)
    ax.grid(axis="y", linestyle=":", linewidth=0.4)


def figure1(df: pd.DataFrame):
    """4×4 grid – four calibration quantities across four planes."""
    planes = ["P1", "P2", "P3", "P4"]
    vars_ = ["Q_F", "Q_B", "T_sum", "T_dif"]
    strips = ["s1", "s2", "s3", "s4"]

    fig, axs = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(18, 12),
        sharex=True,
        sharey='col',
        constrained_layout=True,
    )

    for r, p in enumerate(planes):
        for c, v in enumerate(vars_):
            ax = axs[r, c]
            for s in strips:
                col = f"{p}_{s}_{v}"
                if col in df:
                    # ax.plot(df.index, df[col], label=s)
                    ax.scatter(df.index, df[col], s=point_size)
            if r == 0:
                ax.set_title(v)
            if c == 0:
                ax.set_ylabel(p)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize="small")
            _apply_time_axis(ax)

    fig.suptitle("Calibration columns by plane")
    return fig


def figure3(df: pd.DataFrame):
    """4×4 grid – entry counters per plane/strip."""
    planes = range(1, 5)
    strips = range(1, 5)
    vars_ = ["T_F", "T_B", "Q_F", "Q_B"]
    colors = ["tab:purple", "tab:cyan", "tab:olive", "tab:pink"]

    fig, axs = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(18, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for r, p in enumerate(planes):
        for c, s in enumerate(strips):
            ax = axs[r, c]
            for v, color in zip(vars_, colors):
                col = f"T{p}_{v.split('_')[0]}_{s}_entries" if v.startswith("T") else f"Q{p}_{v.split('_')[1]}_{s}_entries"
                # Correct mapping: e.g. v = "T_F"  → "T{plane}_F_{strip}_entries"
                if v.startswith("T"):
                    col = f"T{p}_{v.split('_')[1]}_{s}_entries"
                else:
                    col = f"Q{p}_{v.split('_')[1]}_{s}_entries"
                if col in df:
                    # ax.plot(df.index, df[col], label=v, linewidth=0.9, color=color)
                    ax.scatter(df.index, df[col], color=color, s=point_size)
            if r == 0:
                ax.set_title(f"s{s}")
            if c == 0:
                ax.set_ylabel(f"P{p}")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize="small")
            _apply_time_axis(ax)

    fig.suptitle("Entries counters by plane and strip")
    return fig



def figure3_1(df: pd.DataFrame):
    """
    4 × 4 grid – entry **rates** (Hz) per plane/strip.

    The rate is computed row-wise as
        rate = entries / (End_Time − Start_Time)  [s⁻¹] .
    """
    # ------------------------------------------------------------------ #
    # Compute acquisition duration per row
    # ------------------------------------------------------------------ #
    end_time = pd.to_datetime(df["End_Time"], errors="coerce")
    duration_s = (end_time - df.index).dt.total_seconds()

    # Guard against zero or negative durations
    duration_s = duration_s.where(duration_s > 0, np.nan)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    planes = range(1, 5)
    strips = range(1, 5)
    vars_ = ["T_F", "T_B", "Q_F", "Q_B"]
    colors = ["tab:purple", "tab:cyan", "tab:olive", "tab:pink"]

    fig, axs = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(18, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for r, p in enumerate(planes):
        for c, s in enumerate(strips):
            ax = axs[r, c]
            for v, color in zip(vars_, colors):
                # Map logical name → column name in the CSV
                prefix, side = v.split("_")            # e.g. T, F
                col = f"{prefix}{p}_{side}_{s}_entries"
                # print(f"Processing column: {col}")
                if col not in df:
                    continue
                # print(f"Found column: {col}")
                # print("----------------------------------------")
                # Rate [Hz] = counts / duration
                rate = df[col] / duration_s
                # ax.plot(df.index, rate, label=v, linewidth=0.9, color=color)
                ax.scatter(df.index, rate, color=color, s=point_size)

            if r == 0:
                ax.set_title(f"s{s}")
            if c == 0:
                ax.set_ylabel(f"P{p}")
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize="small")

            # Time axis formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )
            ax.grid(axis="x", linestyle=":", linewidth=0.4)
            ax.grid(axis="y", linestyle=":", linewidth=0.4)

    fig.suptitle("Entry rates by plane and strip [Hz]")
    return fig



def figure4(df: pd.DataFrame):
    """Two-panel figure: sigmoid widths and background slopes."""
    widths = [
        "sigmoid_width_234", "sigmoid_width_123", "sigmoid_width_34",
        "sigmoid_width_1234", "sigmoid_width_23", "sigmoid_width_12",
        "sigmoid_width_124", "sigmoid_width_134", "sigmoid_width_24",
        "sigmoid_width_13", "sigmoid_width_14",
    ]
    slopes = [
        "background_slope_234", "background_slope_123", "background_slope_34",
        "background_slope_1234", "background_slope_23", "background_slope_12",
        "background_slope_124", "background_slope_134", "background_slope_24",
        "background_slope_13", "background_slope_14",
    ]

    fig, (ax_w, ax_s) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
    )

    for col in widths:
        if col in df:
            # ax_w.plot(df.index, df[col], label=col)
            ax_w.scatter(df.index, df[col], s=point_size)
    ax_w.set_title("Sigmoid widths")
    ax_w.legend(frameon=False, fontsize="x-small")
    _apply_time_axis(ax_w)

    for col in slopes:
        if col in df:
            # ax_s.plot(df.index, df[col], label=col)
            ax_s.scatter(df.index, df[col], s=point_size)
    ax_s.set_title("Background slopes")
    ax_s.legend(frameon=False, fontsize="x-small")
    _apply_time_axis(ax_s)

    return fig


def figure5(df: pd.DataFrame):
    """
    One figure with 5 vertically stacked subplots (5×1),
    each showing a group of global performance metrics.

    The column 'valid_lines_in_dat_file' is multiplied by 100 before plotting.
    """
    groups = [
        ["CRT_avg"],
        ["purity_of_data_percentage", "valid_lines_in_dat_file"],
        ["one_side_events", "time_window_filtering", "old_timing_method"],
        ["z_P1", "z_P2", "z_P3", "z_P4"],
        ["unc_y", "unc_tsum", "unc_tdif"],
    ]

    titles = [
        "CRT Average",
        "Data Quality and Valid Lines",
        "One-side Events and Timing Flags",
        "z Plane Calibration Offsets",
        "Set Uncertainties",
    ]

    fig, axs = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(14, 14),
        sharex=True,
        constrained_layout=True,
    )

    for ax, group, title in zip(axs, groups, titles):
        for col in group:
            if col not in df:
                continue
            if col == "valid_lines_in_dat_file":
                data = df[col] * 100.0  # scale to percentage
                label = col + " ×100"
            if col == "unc_y":
                data = df[col] / 300.0  # scale to percentage
                label = col + " / speed of light in mm/ns"
            else:
                data = df[col]
                label = col
            # ax.plot(df.index, data, label=label, linewidth=1.0)
            ax.scatter(df.index, data, s=point_size)
            # Invert the y z-axis for the z case
            if col.startswith("z_"):
                ax.invert_yaxis()
        ax.set_title(title)
        ax.legend(frameon=False, fontsize="small")
        _apply_time_axis(ax)

    fig.suptitle("Raw to list. Global Performance Metrics", fontsize=16)
    return fig


def figure5_1(df: pd.DataFrame):
    """
    One figure with 5 vertically stacked subplots (5×1),
    each showing a group of global performance metrics.

    The column 'valid_lines_in_dat_file' is multiplied by 100 before plotting.
    """
    groups = [
        ["outliers_removed_percentage"],
        ['correct_angle_with_lut'],
    ]

    titles = [
        "Outliers removed percentage",
        "Corrected angle with LUT",
    ]

    fig, axs = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 7),
        sharex=True,
        constrained_layout=True,
    )

    for ax, group, title in zip(axs, groups, titles):
        for col in group:
            if col not in df:
                continue
            if col == "valid_lines_in_dat_file":
                data = df[col] * 100.0  # scale to percentage
                label = col + " ×100"
            if col == "unc_y":
                data = df[col] / 300.0  # scale to percentage
                label = col + " / speed of light in mm/ns"
            else:
                data = df[col]
                label = col
            # ax.plot(df.index, data, label=label, linewidth=1.0)
            ax.scatter(df.index, data, s=point_size)
            # Invert the y z-axis for the z case
            if col.startswith("z_"):
                ax.invert_yaxis()
        ax.set_title(title)
        ax.legend(frameon=False, fontsize="small")
        _apply_time_axis(ax)

    fig.suptitle("Event Accumulator. Global Performance Metrics", fontsize=16)
    return fig


# -------------------------------------------------------------------------- #
# Figures 6–10  (event_accumulator_metadata)
# -------------------------------------------------------------------------- #
def figure6(df: pd.DataFrame):
      """
      1 × 3 grid – Coefficient-of-variation metrics in separate panels.
      Columns: coeff_variation_A | coeff_variation_beta | coeff_variation_C
      """
      cols = ["coeff_variation_A", "coeff_variation_beta", "coeff_variation_C"]

      fig, axs = plt.subplots(
          nrows=1,
          ncols=3,
          figsize=(18, 4),
          sharex=True,
          constrained_layout=True,
      )

      for ax, col in zip(axs, cols):
          if col in df:
            #   ax.plot(df.index, df[col])
              ax.scatter(df.index, df[col], s=point_size)
          ax.set_title(col)
          _apply_time_axis(ax)

      fig.suptitle("Coefficient-of-variation metrics")
      return fig


def figure7(df: pd.DataFrame):
    """Three-panel figure for eff_* parameters grouped by suffix, with outlier filtering."""
    suffix_groups = {"a": [], "n": [], "0": []}
    for col in df:
        if col.startswith("eff_"):
            *_, suffix = col.split("_")
            if suffix in suffix_groups:
                suffix_groups[suffix].append(col)

    # Define column-specific filtering thresholds: (min, max)
    thresholds = {
        # Example limits — adjust as needed
        "eff_P2_a": (-0.1, 0.1),
        "eff_P2_n": (0.0, 25.0),
        "eff_P2_0": (0.9, 1.0),
        
        "eff_P3_a": (-0.1, 0.1),
        "eff_P3_n": (0.0, 25.0),
        "eff_P3_0": (0.9, 1.0),
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, constrained_layout=True)
    for ax, (suffix, cols) in zip(axs, suffix_groups.items()):
        for col in cols:
            if col in thresholds:
                min_val, max_val = thresholds[col]
                valid_mask = (df[col] > min_val) & (df[col] < max_val)
                x_valid = df.index[valid_mask]
                y_valid = df[col][valid_mask]
            else:
                x_valid = df.index
                y_valid = df[col]
            # ax.plot(x_valid, y_valid, label=col)
            ax.scatter(x_valid, y_valid, s=point_size)
        ax.set_title(f"eff_*_{suffix}")
        ax.legend(frameon=False, fontsize="x-small")
        _apply_time_axis(ax)
    fig.suptitle(r"Efficiency model parameters ($\varepsilon(\theta) = a \cdot \theta^n + \varepsilon_0$)")
    return fig



def figure8(df: pd.DataFrame):
    metrics = [
        "nbar_over_alpha",
        "offset_over_nbar",
        "alpha_over_nbar",
        "eta_curvature",
        "width_proxy",
        "Q_mode",
    ]
    planes = ["1", "2", "3", "4"]

    fig, axs = plt.subplots(4, 6, figsize=(19, 12), sharex=True, sharey='col', constrained_layout=True)
    for r, p in enumerate(planes):
        for c, m in enumerate(metrics):
            ax = axs[r, c]
            col = f"{p}_{m}"
            if col in df:
                # ax.plot(df.index, df[col])
                ax.scatter(df.index, df[col], s=point_size)
            if r == 0:
                ax.set_title(m)
            if c == 0:
                ax.set_ylabel(f"Plane {p}")
            _apply_time_axis(ax)
    fig.suptitle("n̄/α and related proxies by plane")
    return fig


def figure9(df: pd.DataFrame):
    """
    4 × 6 grid – Matrix element rates (value / duration) for M1–M6 by plane.

    Rate is computed as:  rate = M / (End_Time − Start_Time)
    """

    planes = ["P1", "P2", "P3", "P4"]
    ms = ["M1", "M2", "M3", "M4", "M5", "M6"]

    # Compute acquisition duration per row
    end_time = pd.to_datetime(df["End_Time"], errors="coerce")
    duration_s = (end_time - df.index).dt.total_seconds()
    duration_s = duration_s.where(duration_s > 0, np.nan)

    fig, axs = plt.subplots(4, 6, figsize=(19, 12), sharex=True, sharey=True, constrained_layout=True)

    for r, p in enumerate(planes):
        for c, m in enumerate(ms):
            ax = axs[r, c]
            col = f"{p}_{m}"
            if col in df:
                rate = df[col] / duration_s
                # ax.plot(df.index, rate)
                ax.scatter(df.index, rate, s=point_size)
                ax.set_yscale("log")

            if r == 0:
                ax.set_title(m)
            if c == 0:
                ax.set_ylabel(p)

            _apply_time_axis(ax)

    fig.suptitle("Matrix element rates (value / acquisition time) by plane [counts/s]")
    return fig



def figure10(df: pd.DataFrame):
    planes = ["P1", "P2", "P3", "P4"]
    suffixes = ["x0", "k"]
    thresholds = {"x0": [700, 1000], "k": [0, 0.0175]}

    fig, axs = plt.subplots(4, 2, figsize=(12, 12), sharex=True, sharey='col', constrained_layout=True)
    for r, p in enumerate(planes):
        for c, sfx in enumerate(suffixes):
            ax = axs[r, c]
            col = f"{p}_{sfx}"
            if col in df:
                # Filter out outliers
                valid_mask = ( df[col] > thresholds[sfx][0] ) & ( df[col].abs() < thresholds[sfx][1] )
                valid_index = df.index[valid_mask]
                valid_values = df[col][valid_mask]

                # ax.plot(valid_index, valid_values)
                ax.scatter(valid_index, valid_values, s=point_size)
            if r == 0:
                ax.set_title(sfx)
            if c == 0:
                ax.set_ylabel(p)
            _apply_time_axis(ax)
    fig.suptitle("Fit parameters x0 and k")
    return fig




def merge_intervals(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Given a dataframe with index = Start_Time and column 'End_Time',
    merge overlapping or contiguous time intervals.

    Returns a list of merged (start, end) tuples representing continuous acquisition.
    """
    intervals = list(zip(df.index, pd.to_datetime(df["End_Time"], errors='coerce')))
    intervals = sorted([i for i in intervals if pd.notnull(i[1])])

    if not intervals:
        return []

    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if (start - current_end).total_seconds() <= 1:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def plot_data_coverage(df_cal: pd.DataFrame, df_evt: pd.DataFrame):
    """
    Plot merged acquisition periods (in green) for calibration and event metadata.
    """
    periods_cal = merge_intervals(df_cal)
    periods_evt = merge_intervals(df_evt)
    
    # print(periods_cal)
    # print(periods_evt)

    fig, axs = plt.subplots(2, 1, figsize=(14, 5), sharex=True, constrained_layout=True)
    now = pd.Timestamp.now()

    # Top plot: raw_to_list_metadata
    axs[0].set_title("Analyzed periods: raw_to_list_metadata")
    axs[0].set_ylim(0, 1)
    axs[0].set_yticks([])
    axs[0].set_ylabel("Analyzed")
    axs[0].set_xlim(left=df_cal.index.min(), right=now)
    for start, end in periods_cal:
        axs[0].axvspan(start, end, facecolor='green', edgecolor='none', alpha=0.5)

    # Bottom plot: event_accumulator_metadata
    axs[1].set_title("Analyzed periods: event_accumulator_metadata")
    axs[1].set_ylim(0, 1)
    axs[1].set_yticks([])
    axs[1].set_ylabel("Analyzed")
    axs[1].set_xlim(left=df_evt.index.min(), right=now)
    for start, end in periods_evt:
        axs[1].axvspan(start, end, facecolor='green', edgecolor='none', alpha=0.5)

    axs[1].set_xlabel("Time")
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(True, axis='x', linestyle=":", linewidth=0.5)

    fig.suptitle("Green = merged acquisition windows", fontsize=14)
    return fig


# -----------------------------------------------------------------------------#
# Dual execution-order colour-band figure
# -----------------------------------------------------------------------------#

def _exec_colour_dict(df: pd.DataFrame, cmap_name: str = "turbo") -> dict:
    """Return {Start_Time → RGBA} based on execution_time rank (ascending)."""
    if "execution_time" not in df.columns:
        raise KeyError("'execution_time' column not found")
    exec_time = pd.to_datetime(df["execution_time"], errors="coerce")
    order     = exec_time.sort_values()                     # keep original index
    cmap      = matplotlib.colormaps[cmap_name]             # Matplotlib ≥3.7
    norm      = Normalize(vmin=0, vmax=len(order) - 1)
    return {idx: cmap(norm(rank)) for rank, idx in enumerate(order.index)}, cmap, norm


def figure_exec_bands_dual(df_cal: pd.DataFrame,
                           df_evt: pd.DataFrame) -> plt.Figure:
    """
    Two stacked panels, one per dataframe.
    Each span coloured by execution-time rank *within that dataframe*.
    """
    # colour lookup tables for each dataframe
    colours_cal, cmap_cal, norm_cal = _exec_colour_dict(df_cal)
    colours_evt, cmap_evt, norm_evt = _exec_colour_dict(df_evt)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                            constrained_layout=True)
    now = pd.Timestamp.now()

    # ------------ top panel : raw_to_list_metadata ---------------------------
    ax = axs[0]
    ax.set_title("Execution order: raw_to_list_metadata")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Files")
    ax.set_xlim(left=df_cal.index.min(), right=now)

    for start, end in zip(df_cal.index,
                          pd.to_datetime(df_cal["End_Time"], errors="coerce")):
        if pd.isna(end):
            continue
        ax.axvspan(start, end,
                   color=colours_cal.get(start, "grey"),
                   alpha=0.9, linewidth=0)

    # colour-bar (top)
    sm = cm.ScalarMappable(cmap=cmap_cal, norm=norm_cal)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.25,
                 label="Execution-time rank (old → new)")

    _apply_time_axis(ax)

    # ------------ bottom panel : event_accumulator_metadata ------------------
    ax = axs[1]
    ax.set_title("Execution order: event_accumulator_metadata")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Files")
    ax.set_xlabel("Time")
    ax.set_xlim(left=df_evt.index.min(), right=now)

    for start, end in zip(df_evt.index,
                          pd.to_datetime(df_evt["End_Time"], errors="coerce")):
        if pd.isna(end):
            continue
        ax.axvspan(start, end,
                   color=colours_evt.get(start, "grey"),
                   alpha=0.9, linewidth=0)

    # colour-bar (bottom)
    sm = cm.ScalarMappable(cmap=cmap_evt, norm=norm_evt)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.25,
                 label="Execution-time rank (old → new)")

    _apply_time_axis(ax)

    fig.suptitle("Colour = execution-time order (separate scale per panel)",
                 fontsize=14)
    return fig




# -------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Visualize MINGO station metadata.")
    parser.add_argument("station", type=int, choices=[1, 2, 3, 4], help="Station number")
    parser.add_argument("--save", action="store_true", help="Save figures as PNG and PDF.")
    args = parser.parse_args()

    df_cal, df_evt = read_station_metadata(args.station)

    fig0 = plot_data_coverage(df_cal, df_evt)                         # existing
    fig_exec = figure_exec_bands_dual(df_cal, df_evt)      # NEW two-panel colour bands

    # Generate rest of the figures
    figs = [
        fig0,
        fig_exec,                                                     #  NEW (right after fig0)
        figure1(df_cal),
        figure3(df_cal),
        figure3_1(df_cal),
        figure4(df_cal),
        figure5(df_cal),
        figure5_1(df_cal),
        figure6(df_evt),
        figure7(df_evt),
        figure8(df_evt),
        figure9(df_evt),
        figure10(df_evt),
    ]
    

    if args.save:
        
        # Save multi-page PDF in the same directory as the CSVs
        base = (
            Path("/home/mingo/DATAFLOW_v3")
            / "STATIONS"
            / f"MINGO0{args.station}"
            / "FIRST_STAGE"
            / "EVENT_DATA"
        )
        base.mkdir(parents=True, exist_ok=True)        # ensure path exists
        pdf_path = base / f"station{args.station}_summary.pdf"
        
        # Directory for PNGs
        fig_dir = base / f"station{args.station}_figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Save individual PNGs
        for i, fig in enumerate(figs, 1):
            fig.savefig(fig_dir / f"station{args.station}_figure{i}.png", dpi=300)
        
        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                print(f"Saving figure {fig.number} to PDF...")
                pdf.savefig(fig)
                plt.close(fig)
        print(f"PDF saved to: {pdf_path.resolve()}")
        
        # Remove the outdir, also if it has files
        if fig_dir.exists():
            shutil.rmtree(fig_dir)          # recursive, ignores non-empty state
            print(f"Temporary directory {fig_dir} removed.")
        
    else:
        print("Figures will not be saved. Use --save to enable saving.")


if __name__ == "__main__":
    main()

# %%
