#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import sys
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import sys
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import halfnorm
from scipy.stats import norm
from scipy.stats import pearsonr
import sys
import pandas as pd
from io import StringIO

sta_time = datetime(2025, 5, 25)
end_time = datetime(2025, 6, 5, 14)

# ----------- Configuration and Input ------------
if len(sys.argv) != 2 or sys.argv[1] not in {'1', '2', '3', '4'}:
    print("Usage: python script.py <station_index (1–4)>")
    sys.exit(1)

station_index = sys.argv[1]
nmdb_path = "/home/cayetano/DATAFLOW_v3/MASTER/THIRD_STAGE/nmdb_combined.csv"
corrected_path = f"/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0{station_index}/SECOND_STAGE/large_corrected_table.csv"
output_path = f"/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0{station_index}/THIRD_STAGE/third_stage_table.csv"
figure_path = f"/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0{station_index}/THIRD_STAGE/FIGURES/"

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



import sys
import pandas as pd
import numpy as np
from io import StringIO


# ----------- Parse station argument from command line ------------
if len(sys.argv) != 2 or sys.argv[1] not in ['1', '2', '3', '4']:
    raise ValueError("Usage: script.py <station> with station in {1, 2, 3, 4}")
station = sys.argv[1]
import pandas as pd
from io import StringIO


# ----------- Load NMDB Data --------------------------------------
nmdb_path = "/home/cayetano/DATAFLOW_v3/MASTER/THIRD_STAGE/nmdb_combined.csv"

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
station_path = f"/home/cayetano/DATAFLOW_v3/STATIONS/MINGO0{station}/SECOND_STAGE/large_corrected_table.csv"
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

import matplotlib.ticker as mtick

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
