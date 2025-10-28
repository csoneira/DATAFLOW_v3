#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

"""
Created on 2025-01-01

@author: csoneira@ucm.es
"""

print("\n\n")
print("                     `. ___")
print("                    __,' __`.                _..----....____")
print("        __...--.'``;.   ,.   ;``--..__     .'    ,-._    _.-'")
print("  _..-''-------'   `'   `'   `'     O ``-''._   (,;') _,'")
print(",'________________                          \\`-._`-',")
print(" `._              ```````````------...___   '-.._'-:")
print("    ```--.._      ,.                     ````--...__\\-.")
print("            `.--. `-`                       ____    |  |`")
print("              `. `.                       ,'`````.  ;  ;`")
print("                `._`.        __________   `.      \\'__/`")
print("                   `-:._____/______/___/____`.     \\  `")
print("                               |       `._    `.    \\")
print("                               `._________`-.   `.   `.___")
print("                                             SSt  `------'`")
print("\n\n")


# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------


# Standard Library
import os
import sys
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta

# Scientific Computing
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import griddata
from scipy.stats import (
    norm,
    halfnorm,
    pearsonr
)

# Machine Learning
from sklearn.linear_model import LinearRegression

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------

CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = None
for parent in CURRENT_PATH.parents:
    if parent.name == "MASTER":
        REPO_ROOT = parent.parent
        break
if REPO_ROOT is None:
    REPO_ROOT = CURRENT_PATH.parents[-1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MASTER.common.execution_logger import set_station, start_timer

start_timer(__file__)

home_path = os.path.expanduser("~")

sta_time = datetime(2025, 5, 25)
end_time = datetime(2025, 6, 5, 14)

# ----------- Configuration and Input ------------
if len(sys.argv) != 2 or sys.argv[1] not in {'1', '2', '3', '4'}:
    print("Usage: python script.py <station_index (1–4)>")
    sys.exit(1)

station_index = sys.argv[1]
set_station(station_index)
nmdb_path = f"{home_path}/DATAFLOW_v3/MASTER/STAGE_3/nmdb_combined.csv"
corrected_path = f"{home_path}/DATAFLOW_v3/STATIONS/MINGO0{station_index}/STAGE_2/large_corrected_table.csv"
output_path = f"{home_path}/DATAFLOW_v3/STATIONS/MINGO0{station_index}/STAGE_3/third_stage_table.csv"
figure_path = f"{home_path}/DATAFLOW_v3/STATIONS/MINGO0{station_index}/STAGE_3/FIGURES/"

# City of the detector. 1: Madrid, 2: Warsaw, 3: Puebla, 4: Monterrey
city_names = {
    '1': 'Madrid',
    '2': 'Warsaw',
    '3': 'Puebla',
    '4': 'Monterrey'
}
# Get the city name based on the station index
city_name = city_names.get(station_index, 'Unknown City')


# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(figure_path), exist_ok=True)





# ----------- Parse station argument from command line ------------
if len(sys.argv) != 2 or sys.argv[1] not in ['1', '2', '3', '4']:
    raise ValueError("Usage: script.py <station> with station in {1, 2, 3, 4}")
station = sys.argv[1]



# ----------- Load NMDB Data --------------------------------------
nmdb_path = f"{home_path}/DATAFLOW_v3/MASTER/STAGE_3/nmdb_combined.csv"

with open(nmdb_path, 'r') as f:
    lines = f.readlines()

# Find start of data
data_start = next(i for i, line in enumerate(lines) if line.strip()[:4].isdigit())

# Dynamically extract station names from the header line just before the data
for i in range(data_start - 1, 0, -1):
    line = lines[i].strip()
    if line and not line.startswith("#"):
        station_line = line
        break

# Build column names (Time + station names)
columns = ["Time"] + station_line.split()

# Load data from block
nmdb_df = pd.read_csv(
    StringIO(''.join(lines[data_start:])),
    sep=';',
    header=None,
    engine='python',
    na_values=["null"]
)

# Apply extracted column names, trimming if needed
nmdb_df.columns = columns[:nmdb_df.shape[1]]

# Clean types
nmdb_df["Time"] = pd.to_datetime(nmdb_df["Time"].str.strip(), errors='coerce')
nmdb_df.iloc[:, 1:] = nmdb_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')


# ----------- Load Station Data -----------------------------------
station_path = f"{home_path}/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_2/large_corrected_table.csv"
station_df = pd.read_csv(station_path, low_memory=False)
station_df["Time"] = pd.to_datetime(station_df["Time"], errors='coerce')
station_df = station_df.apply(pd.to_numeric, errors='coerce').assign(Time=station_df["Time"])


# ----------- Time filtering --------------------------------------
nmdb_df = nmdb_df[(nmdb_df["Time"] >= sta_time) & (nmdb_df["Time"] < end_time)]
station_df = station_df[(station_df["Time"] >= sta_time) & (station_df["Time"] < end_time)]


# -------------- Merging ------------------------------------------
nmdb_df.sort_values("Time", inplace=True)
station_df.sort_values("Time", inplace=True)

# Ensure Time columns are datetime
nmdb_df["Time"] = pd.to_datetime(nmdb_df["Time"], errors='coerce')
station_df["Time"] = pd.to_datetime(station_df["Time"], errors='coerce')

# Coerce non-numeric values to NaN
nmdb_df = nmdb_df.apply(pd.to_numeric, errors='coerce').assign(Time=nmdb_df["Time"])
station_df = station_df.apply(pd.to_numeric, errors='coerce').assign(Time=station_df["Time"])

# Must be sorted before merge_asof
nmdb_df_sorted = nmdb_df.sort_values('Time')
station_df_sorted = station_df.sort_values('Time')

# Round the times to the minute
nmdb_df_sorted = nmdb_df_sorted.assign(Time=nmdb_df_sorted["Time"].dt.floor('1min'))
station_df_sorted = station_df_sorted.assign(Time=station_df_sorted["Time"].dt.floor('1min'))

# Merge nearest timestamps within 5 minutes (tune tolerance)
data_df = pd.merge_asof(
    nmdb_df_sorted,           # <-- use NMDB as the base (left)
    station_df_sorted,        # station data will be aligned to it
    on="Time",
    direction="nearest",
    tolerance=pd.Timedelta("1min")
)


print(data_df.columns.to_list())






save_plots = True
show_plots = False
fig_idx = 0


def plot_grouped_series(df, group_cols, time_col='Time', title=None, figsize=(14, 4), save_path=None):
    """
    Plot time series for multiple groups of columns. Each sublist in `group_cols` is plotted in a separate subplot.
    
    Parameters:
        df (pd.DataFrame): DataFrame with time series data.
        group_cols (list of list of str): Each sublist contains column names to overlay in one subplot.
        time_col (str): Name of the time column.
        title (str): Title for the entire figure.
        figsize (tuple): Size of each subplot.
        save_path (str): If provided, save the figure to this path.
    """
    global fig_idx
    
    n_plots = len(group_cols)
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(figsize[0], figsize[1] * n_plots))
    
    if n_plots == 1:
        axes = [axes]  # Make iterable
    
    for idx, cols in enumerate(group_cols):
        ax = axes[idx]
        for col in cols:
            if col in df.columns:
                x = df[time_col]
                y = df[col]
                
                cond = y.notna() & x.notna()
                x = x[cond]
                y = y[cond]
                
                ax.plot(x, y, label=col)
            else:
                print(f"Warning: column '{col}' not found in DataFrame")
        ax.set_ylabel(' / '.join(cols))
        ax.grid(True)
        
        # Add a watermark that says "Preliminary"
        ax.text(0.3, 0.35, 'Preliminary', fontsize=40, color='gray', alpha=0.5,
                transform=ax.transAxes, ha='center', va='center', rotation=10, weight='bold')
        
        ax.legend(loc='best')
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

    axes[-1].set_xlabel('Time')
    if title:
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    if show_plots:
        plt.show()
    elif save_plots:
        new_figure_path = figure_path + f"{fig_idx}" + "_NMDB_and_TRASGO.png"
        fig_idx += 1
        print(f"Saving figure to {new_figure_path}")
        plt.savefig(new_figure_path, format='png', dpi=300)
    plt.close()



print("nmdb_df:")
print(nmdb_df.head())
print("\n")

print("station_df:")
print(station_df.head())
print("\n")


for column in data_df.columns:
    if column != "Time":
        # Normalize all columns except 'Time'
        data_df[column] = data_df[column].astype(float)
        # Do the mean of the first 25% of the data
        first_quarter_mean = data_df[column].iloc[:len(data_df)//4].mean()
        # Normalize the column
        data_df[column] = data_df[column] / first_quarter_mean - 1


group_cols = [
    [ 'OULU', 'INVK', 'SOPO', 'CALM', 'MXCO', 'ICRB', 'ICRO' ],
    [ 'total_best_sum' ]
]
plot_grouped_series(data_df, group_cols, title=f"Station {station_index} Data")

group_cols = [
    [ 'total_best_sum', 'OULU' ]
]
plot_grouped_series(data_df, group_cols, title=f"Station {station_index} Data")


data_df["miniTRASGO"] = data_df["total_best_sum"]

group_cols = [
    [ 'miniTRASGO', 'KIEL2', 'LMKS', ]
]
plot_grouped_series(data_df, group_cols, title=f"{city_name}. Station {station_index} Corrected. Normalized rate compared with NMDB.")


data_df["miniTRASGO"] = data_df["detector_12_eff_corr_pressure_corrected"]

group_cols = [
    [ 'miniTRASGO', 'MXCO', ]
]
plot_grouped_series(data_df, group_cols, title=f"{city_name}. Station {station_index} Corrected. Normalized rate compared with NMDB.")



# ----------- Save Output ------------
data_df.to_csv(output_path, index=False)
print(f"\nMerged dataframe written to {output_path}")







print("--------------------------------------------------------------------------")
print("---------------------- Rigidity cutoff calculator ------------------------")
print("--------------------------------------------------------------------------")


# Given data (effective vertical cut-off rigidities)
data = [
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.03, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.03],
    [0.13, 0.10, 0.06, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.04, 0.07, 0.09, 0.11, 0.13, 0.14, 0.15, 0.16, 0.16, 0.17, 0.18, 0.18, 0.18, 0.16, 0.13],
    [0.36, 0.28, 0.19, 0.11, 0.06, 0.03, 0.02, 0.03, 0.05, 0.09, 0.14, 0.20, 0.26, 0.30, 0.33, 0.35, 0.35, 0.36, 0.38, 0.40, 0.42, 0.45, 0.45, 0.42, 0.36],
    [0.79, 0.62, 0.45, 0.29, 0.18, 0.12, 0.09, 0.11, 0.16, 0.25, 0.36, 0.49, 0.59, 0.65, 0.69, 0.71, 0.72, 0.73, 0.76, 0.79, 0.86, 0.91, 0.93, 0.90, 0.79],
    [1.49, 1.21, 0.91, 0.62, 0.41, 0.28, 0.24, 0.26, 0.37, 0.55, 0.78, 0.98, 1.15, 1.26, 1.29, 1.31, 1.33, 1.34, 1.38, 1.44, 1.59, 1.69, 1.75, 1.68, 1.49],
    [2.54, 2.10, 1.62, 1.16, 0.81, 0.58, 0.50, 0.54, 0.76, 1.09, 1.53, 1.87, 2.07, 2.14, 2.17, 2.18, 2.20, 2.24, 2.29, 2.44, 2.65, 2.84, 2.95, 2.84, 2.54],
    [3.93, 3.22, 2.63, 2.01, 1.43, 1.07, 0.92, 1.01, 1.37, 1.98, 2.65, 3.08, 3.33, 3.43, 3.40, 3.40, 3.42, 3.52, 3.62, 3.81, 4.15, 4.38, 4.56, 4.36, 3.93],
    [5.30, 4.66, 3.97, 3.08, 2.33, 1.77, 1.52, 1.70, 2.29, 3.21, 4.22, 4.76, 4.97, 5.02, 4.99, 4.97, 5.10, 5.20, 5.35, 5.56, 5.85, 6.13, 6.21, 5.83, 5.30],
    [7.42, 6.20, 5.29, 4.49, 3.48, 2.72, 2.36, 2.60, 3.59, 4.84, 5.94, 6.91, 7.25, 7.17, 6.95, 6.91, 7.10, 7.35, 7.59, 7.95, 8.62, 9.09, 9.04, 8.49, 7.42],
    [9.32, 8.65, 7.33, 5.81, 4.84, 3.83, 3.41, 3.72, 4.97, 6.94, 9.08, 9.75, 9.80, 9.67, 9.67, 9.88, 10.30, 10.58, 10.83, 10.95, 11.06, 11.34, 10.99, 10.17, 9.32],
    [11.42, 10.19, 9.28, 8.17, 6.42, 5.13, 4.41, 4.96, 7.06, 9.81, 11.00, 11.56, 11.72, 11.71, 11.73, 12.01, 12.44, 12.74, 13.16, 13.65, 13.73, 13.59, 13.13, 12.42, 11.42],
    [12.70, 11.97, 11.18, 10.19, 8.73, 7.00, 5.99, 6.77, 9.31, 11.55, 12.56, 13.14, 13.46, 13.65, 13.83, 14.13, 14.55, 14.95, 15.20, 15.24, 15.11, 14.77, 14.19, 13.47, 12.70],
    [13.56, 12.90, 12.24, 11.45, 10.11, 8.02, 6.65, 7.79, 11.22, 12.56, 13.40, 13.96, 14.36, 14.66, 14.91, 15.25, 15.74, 16.19, 16.43, 16.38, 16.12, 15.65, 14.99, 14.27, 13.56],
    [14.23, 13.64, 13.07, 12.41, 11.39, 9.95, 9.31, 10.50, 12.13, 13.12, 13.81, 14.33, 14.77, 15.15, 15.51, 15.96, 16.52, 17.01, 17.24, 17.14, 16.78, 16.23, 15.56, 14.87, 14.23],
    [14.73, 14.21, 13.70, 13.15, 12.40, 11.49, 11.00, 11.73, 12.61, 13.35, 13.86, 14.29, 14.75, 15.21, 15.67, 16.24, 16.89, 17.42, 17.66, 17.53, 17.11, 16.53, 15.89, 15.29, 14.73],
    [15.04, 14.58, 14.12, 13.64, 13.08, 12.42, 12.04, 12.25, 12.80, 13.29, 13.58, 13.89, 14.33, 14.84, 15.42, 16.11, 16.84, 17.42, 17.67, 17.54, 17.12, 16.55, 15.99, 15.50, 15.04],
    [15.12, 14.74, 14.33, 13.90, 13.43, 12.91, 12.48, 12.45, 12.76, 12.98, 13.02, 13.17, 13.56, 14.10, 14.78, 15.60, 16.40, 17.01, 17.28, 17.19, 16.79, 16.28, 15.82, 15.46, 15.12],
    [14.93, 14.66, 14.32, 13.94, 13.53, 13.06, 12.61, 12.44, 12.52, 12.47, 12.24, 12.21, 12.52, 13.06, 13.83, 14.74, 15.59, 16.20, 16.50, 16.45, 16.12, 15.68, 15.34, 15.14, 14.93],
    [14.45, 14.30, 14.07, 13.78, 13.43, 13.00, 12.55, 12.25, 12.10, 11.79, 11.27, 11.03, 11.25, 11.82, 12.63, 13.59, 14.43, 15.01, 15.31, 15.33, 15.07, 14.71, 14.50, 14.47, 14.45],
    [13.60, 13.65, 13.58, 13.41, 13.14, 12.77, 12.31, 11.90, 11.55, 10.97, 10.14, 9.59, 9.80, 10.31, 11.19, 12.11, 12.87, 13.42, 13.69, 13.78, 13.58, 13.30, 13.25, 13.38, 13.60],
    [12.27, 12.66, 12.82, 12.82, 12.66, 12.37, 11.92, 11.40, 10.82, 9.92, 8.86, 8.11, 8.06, 8.60, 9.35, 10.10, 10.64, 11.02, 11.34, 11.49, 11.04, 10.50, 10.58, 11.61, 12.27],
    [9.63, 10.67, 11.58, 12.00, 12.01, 11.82, 11.41, 10.74, 9.94, 8.83, 7.58, 6.72, 6.66, 7.11, 7.70, 8.17, 8.31, 8.12, 8.02, 8.11, 8.02, 7.94, 8.46, 9.26, 9.63],
    [7.63, 8.99, 8.75, 10.52, 11.16, 11.13, 10.73, 9.96, 9.02, 7.70, 6.35, 5.68, 5.35, 5.55, 5.84, 5.87, 5.69, 5.51, 5.44, 5.41, 5.45, 5.50, 5.83, 6.46, 7.63],
    [5.39, 6.27, 7.69, 7.95, 9.68, 10.28, 9.87, 9.13, 7.96, 6.52, 5.56, 4.73, 4.26, 4.20, 4.28, 4.30, 4.14, 3.90, 3.62, 3.57, 3.59, 3.73, 4.17, 4.82, 5.39],
    [3.99, 4.49, 5.34, 6.51, 7.85, 9.14, 8.88, 8.12, 6.87, 5.88, 4.75, 3.86, 3.45, 3.34, 3.35, 3.24, 3.01, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.21, 3.99],
    [3.00, 3.25, 4.09, 4.76, 5.86, 7.43, 7.88, 7.34, 6.41, 5.01, 3.88, 3.20, 2.78, 2.63, 2.44, 2.24, 2.04, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.07, 3.00],
    [2.00, 2.18, 3.03, 3.67, 4.42, 5.13, 5.90, 5.76, 4.90, 3.99, 3.21, 2.65, 2.23, 2.05, 1.80, 1.52, 1.21, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.15, 2.00],
    [1.00, 1.32, 2.03, 3.00, 3.25, 3.92, 4.31, 4.27, 3.86, 3.24, 2.65, 2.14, 1.81, 1.53, 1.29, 1.05, 0.75, 0.51, 0.33, 0.25, 0.23, 0.27, 0.38, 0.59, 1.00],
    [0.48, 0.77, 1.14, 2.00, 2.17, 3.00, 3.17, 3.32, 3.07, 2.48, 2.04, 1.65, 1.36, 1.13, 0.90, 0.68, 0.45, 0.27, 0.15, 0.09, 0.07, 0.09, 0.15, 0.27, 0.48],
    [0.23, 0.42, 0.68, 1.02, 1.40, 2.00, 2.09, 2.23, 2.13, 1.80, 1.50, 1.22, 1.00, 0.80, 0.61, 0.42, 0.26, 0.13, 0.06, 0.01, 0.00, 0.00, 0.04, 0.11, 0.23],
    [0.10, 0.21, 0.37, 0.59, 0.83, 1.07, 1.28, 1.37, 1.34, 1.23, 1.04, 0.88, 0.70, 0.55, 0.40, 0.26, 0.15, 0.06, 0.01, 0.00, 0.00, 0.00, 0.00, 0.03, 0.10],
    [0.05, 0.11, 0.20, 0.33, 0.47, 0.62, 0.73, 0.81, 0.81, 0.75, 0.67, 0.57, 0.45, 0.35, 0.25, 0.16, 0.08, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05],
    [0.03, 0.07, 0.12, 0.18, 0.25, 0.32, 0.38, 0.42, 0.44, 0.42, 0.38, 0.33, 0.27, 0.21, 0.15, 0.10, 0.06, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03],
    [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.18, 0.19, 0.20, 0.20, 0.19, 0.17, 0.15, 0.12, 0.10, 0.07, 0.05, 0.04, 0.02, 0.01, 0.00, 0.00, 0.01, 0.02, 0.04],
    [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07]
]

# Coordinates for Madrid and Coimbra
coords_madrid = (40.4168, -3.7038)
coords_coimbra = (40.2033, -8.4103)

# Convert longitudes to the table range (-180 to 180)
coords_madrid = (coords_madrid[0], coords_madrid[1])
coords_coimbra = (coords_coimbra[0], coords_coimbra[1])

# Define the grid of latitudes and longitudes
latitudes = np.array([90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70, -75, -80, -85, -90])
longitudes = np.linspace(-180, 180, 25)

# Assuming the data is already defined and loaded into the variable `data`
# Flatten the data for griddata interpolation
lat_lon_pairs = [(lat, lon) for lat in latitudes for lon in longitudes]
data_flat = np.array(data).flatten()

# Prepare the coordinates for interpolation (ensure longitude is within [-180, 180])
coords_madrid = (coords_madrid[0], coords_madrid[1] if coords_madrid[1] >= -180 else coords_madrid[1] + 360)
coords_coimbra = (coords_coimbra[0], coords_coimbra[1] if coords_coimbra[1] >= -180 else coords_coimbra[1] + 360)

# Perform interpolation
value_madrid = griddata(lat_lon_pairs, data_flat, coords_madrid, method='linear')
value_coimbra = griddata(lat_lon_pairs, data_flat, coords_coimbra, method='linear')

print(f"Interpolated value for Madrid: {value_madrid}")
print(f"Interpolated value for Coimbra: {value_coimbra}")
