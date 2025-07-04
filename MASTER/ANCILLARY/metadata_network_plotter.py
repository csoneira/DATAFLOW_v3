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
      """
      Read CSV, parse Time column as datetime index, coerce all other columns to numeric.
      """
      df = pd.read_csv(csv_path, low_memory=False)
      df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
      df.set_index("Time", inplace=True)
      for col in df.columns:
          df[col] = pd.to_numeric(df[col], errors="coerce")
      return df



def read_station_metadata(station: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      """
      Load metadata for stations 1 to 4. If a file is missing, return an empty DataFrame for that station.
      """
      def safe_read(path: Path) -> pd.DataFrame:
          if path.exists():
              return _read_csv(path)
          else:
              print(f"Warning: File not found → {path}")
              return pd.DataFrame()

      base = Path("/home/mingo/DATAFLOW_v3/STATIONS")
    
      df1 = safe_read(base / "MINGO01" / "SECOND_STAGE" / "total_data_table.csv")
      df2 = safe_read(base / "MINGO02" / "SECOND_STAGE" / "total_data_table.csv")
      df3 = safe_read(base / "MINGO03" / "SECOND_STAGE" / "total_data_table.csv")
      df4 = safe_read(base / "MINGO04" / "SECOND_STAGE" / "total_data_table.csv")
      
    
      return df1, df2, df3, df4




# -----------------------------------------------------------------------------#
# Plot helpers
# -----------------------------------------------------------------------------#
def _apply_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(axis="x", linestyle=":", linewidth=0.4)
    ax.grid(axis="y", linestyle=":", linewidth=0.4)



# Plot in 4 plots the column hv_HVneg; four rows, one row per dfX
import matplotlib.pyplot as plt

def figure1(df1, df2, df3, df4):
    """
    Plot the 'hv_HVneg' column from four DataFrames in a 4-row figure.
    """
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    dataframes = [df1, df2, df3, df4]
    titles = ["Station 1", "Station 2", "Station 3", "Station 4"]

    for ax, df, title in zip(axs, dataframes, titles):
        if 'hv_HVneg' not in df.columns:
            ax.set_title(f"{title} (hv_HVneg not found)")
            ax.set_axis_off()
            continue
        df['hv_HVneg'].plot(ax=ax, lw=0, marker='.', markersize=2, label="HV-")
        ax.set_ylabel("HV- (V)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", linewidth=0.4)
        _apply_time_axis(ax)

    axs[-1].set_xlabel("Time")
    return fig

# The figure1() function is now complete and ready to be used in main().
# It expects df1, df2, df3, df4 to be passed, each containing a 'hv_HVneg' column.
# The output is a matplotlib Figure object with four aligned subplots.



def merge_intervals(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Given a dataframe with index = Time and column 'End_Time',
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
    """Return {Time → RGBA} based on execution_time rank (ascending)."""
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
    parser = argparse.ArgumentParser(description="Visualize all MINGO stations.")
    parser.add_argument("--save", action="store_true", help="Save figures as PNG and PDF.")
    args = parser.parse_args()

    # Load data from all 4 stations
    df1, df2, df3, df4 = read_station_metadata(station=1)  # station arg unused inside

    # Generate figures
    fig_hv = figure1(df1, df2, df3, df4)                 # HV voltage plots

    # You can optionally still generate these for station 1 if you want
#     fig0 = plot_data_coverage(df1, df2)
#     fig_exec = figure_exec_bands_dual(df1, df2)

    # Collect figures
    figs = [
        fig_hv,
        # Add more here
    ]

    if args.save:
        outdir = Path("/home/mingo/DATAFLOW_v3/STATIONS/")
        outdir.mkdir(parents=True, exist_ok=True)
        fig_dir = outdir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Save PNGs
        for i, fig in enumerate(figs, 1):
            fig.savefig(fig_dir / f"figure{i}.png", dpi=300)

        # Save all in one PDF
        pdf_path = outdir / "summary.pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in figs:
                print(f"Saving figure {fig.number} to PDF...")
                pdf.savefig(fig)
                plt.close(fig)

        print(f"PDF saved to: {pdf_path.resolve()}")

        # Optional cleanup
        shutil.rmtree(fig_dir)
        print(f"Temporary directory {fig_dir} removed.")
    else:
        print("Figures will not be saved. Use --save to enable saving.")
        plt.show()


if __name__ == "__main__":
    main()

# %%
