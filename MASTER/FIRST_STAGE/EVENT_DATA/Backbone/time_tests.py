#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


fast_mode = False # Do not iterate TimTrack, neither save figures, etc.
debug_mode = False # Only 10000 rows with all detail
# newest_file = False
# oldest_file = False
last_file_test = False

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

# globals().clear()

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.constants import c
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.linalg as linalg
from math import sqrt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import os
from scipy.stats import poisson
import shutil
import builtins
import random
import re
import csv

# Store the current time at the start. To time the execution
start_execution_time_counting = datetime.now()

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Check if the script has an argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

# Get the station argument
# station = sys.argv[1]
station = 1
# print(f"Station: {station}")

# -----------------------------------------------------------------------------

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
home_directory = os.path.expanduser(f"~")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
raw_working_directory = os.path.join(base_directory, "RAW")
raw_to_list_working_directory = os.path.join(base_directory, "RAW_TO_LIST")

# Define directory paths relative to base_directory
base_directories = {
    
    "base_plots_directory": os.path.join(raw_to_list_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(raw_to_list_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(raw_to_list_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(raw_to_list_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    
    "full_list_events_directory": os.path.join(base_directory, "FULL_LIST_EVENTS_DIRECTORY"),

}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

# Move files from RAW to RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

full_list_events_directory = base_directories["full_list_events_directory"]

z_positions = np.array([30, 145, 290, 435])  # In mm

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Body ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Determine the file path input

# Get lists of files in the directories
unprocessed_files = sorted(os.listdir(full_list_events_directory))

print(f"Number of files in FULL_LIST_EVENTS_DIRECTORY: {len(unprocessed_files)}")

if unprocessed_files:
      unprocessed_files = sorted(unprocessed_files)
      file_name = unprocessed_files[-1]
      
      file_path = os.path.join(full_list_events_directory, file_name)
      print(f"Processing: {file_name}")

data_df = pd.read_csv(file_path)

print(data_df.columns.to_list())


# %%

# Take only the columns which have T_sum in the name, I mean, inside of the column name
# Also _final cannot be in the column name. Also the column called 'type' save it
# Extract relevant columns
data_df_times = data_df.filter(regex='T_sum')
data_df_charges = data_df.filter(regex='Q_sum')
data_df_tdiff = data_df.filter(regex='T_diff')

# Concatenate all relevant data with 'type' column
data_df_times = pd.concat([data_df['type'], data_df_charges, data_df_times, data_df_tdiff], axis=1)

# Remove columns containing '_final'
data_df_times = data_df_times.loc[:, ~data_df_times.columns.str.contains('_final')]

print(data_df_times.columns.to_list())
# %%

# Histogram all the values in data_df_times in a 4x4 subfigure plot
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle(f"Station {station} - Histogram of T_sum values", fontsize=20)

for i in range(4):
      for j in range(4):
            column_name = f"T{i+1}_T_sum_{j+1}"
            v = data_df_times[column_name]
            v = v[v != 0]
            v = v[v <= -50]
            axs[i, j].hist(v, bins=100)
            axs[i, j].set_title(column_name)


# %%

print(data_df_times['type'].unique())

# %%

case = 1234
data_case = data_df_times[data_df_times["type"] == case].copy()

# Ensure "T1_T_sum_1" exists in the dataframe
if "T1_T_sum_1" not in data_case.columns:
    raise KeyError("Column 'T1_T_sum_1' not found in data_case.")

# Perform subtraction outside the loop
data_case_subtracted = data_case.copy()
for col in data_case.columns:
    if col.startswith("T") and col != "T1_T_sum_1":  # Avoid self-subtraction
        data_case_subtracted[col] = data_case[col] - data_case["T1_T_sum_1"]

# Create figure for histograms
fig, axs = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle(f"Station {station} - Histogram of T_sum values", fontsize=20)

for i in range(4):
    for j in range(4):
        column_name = f"T{i+1}_T_sum_{j+1}"
        
        # Check if the column exists in data_case_subtracted
        if column_name in data_case_subtracted.columns:
            v = data_case_subtracted[column_name].copy()
            v = v[v != 0]  # Remove zero values
            v = v[abs(v) < 5]  # Filter out extreme values

            axs[i, j].hist(v, bins=100)
            axs[i, j].set_title(column_name)
        else:
            axs[i, j].set_title(f"{column_name} (Not Found)")
            axs[i, j].axis("off")  # Hide empty subplots

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%

import itertools
import numpy as np
import matplotlib.pyplot as plt

T_sum_cols = data_case.filter(regex='T_sum', axis=1)

# Function to calculate the desired distances, ignoring zeros
def get_distances(row):
    values = row.dropna().values  # Drop NaNs if any
    values = values[values != 0]  # Exclude zeros
    
    if len(values) < 2:
        return [np.nan] * 16  # Not enough values to compute distances

    # Compute all pairwise absolute differences
    pairwise_diffs = sorted([abs(a - b) for a, b in itertools.combinations(values, 2)])

    # Create a list of 20 max and 20 closest distances (or NaN if not enough)
    max_distances = [pairwise_diffs[-i] if len(pairwise_diffs) >= i else np.nan for i in range(1, 9)]
    closest_distances = [pairwise_diffs[i-1] if len(pairwise_diffs) >= i else np.nan for i in range(1, 9)]

    return max_distances + closest_distances

# Generate column names dynamically
max_distance_cols = [f"{i+1}_max_distance" for i in range(8)]
closest_distance_cols = [f"{i+1}_closest_distance" for i in range(8)]

# Apply function to each row
data_case[max_distance_cols + closest_distance_cols] = T_sum_cols.apply(get_distances, axis=1, result_type="expand")

# Plot histograms overlayed
plt.figure(figsize=(12, 6))

bin_number = 500
colors = [
    "blue", "red", "green", "cyan", "magenta", "orange", "purple", "brown", "pink", "yellow",
    "lime", "teal", "gold", "navy", "violet", "gray", "salmon", "indigo", "turquoise", "maroon"
]

for i in range(8):
    plt.hist(data_case[max_distance_cols[i]].dropna(), bins=bin_number, alpha=0.4, label=f"{i+1}th Max Distance", color=colors[i])
    plt.hist(data_case[closest_distance_cols[i]].dropna(), bins=bin_number, alpha=0.4, label=f"{i+1}th Closest Distance", color=colors[i], linestyle="dashed")

plt.yscale("log")
plt.title("Histogram of T_sum Distance Metrics (20 Furthest & 20 Closest, No Zeros)")
plt.legend(ncol=2)  # Display legend in two columns to save space
plt.show()

# %%


plt.figure(figsize=(12, 6))

x_limit = 500  # Pre-filtering applied before plotting

bin_number = 50
# bins = bin_number
bins = np.logspace(np.log10(1), np.log10(x_limit), bin_number)

colors = [
    "blue", "red", "green", "cyan", "magenta", "orange", "purple", "brown", "pink", "yellow",
    "lime", "teal", "gold", "navy", "violet", "gray", "salmon", "indigo", "turquoise", "maroon"
]

for i in range(8):
    # Apply limit before plotting
    max_dist_data = data_case[max_distance_cols[i]].dropna()
    max_dist_data = max_dist_data[max_dist_data <= x_limit]  # Apply limit

    closest_dist_data = data_case[closest_distance_cols[i]].dropna()
    closest_dist_data = closest_dist_data[closest_dist_data <= x_limit]  # Apply limit

    plt.hist(max_dist_data, bins=bins, alpha=0.4, label=f"{i+1}th Max Distance", color=colors[i], histtype = "step")
    plt.hist(closest_dist_data, bins=bins, alpha=0.4, label=f"{i+1}th Closest Distance", color=colors[-i + 1], histtype = "step")

plt.xscale("log")
plt.yscale("log")
plt.title("Histogram of T_sum Distance Metrics (20 Furthest & 20 Closest, No Zeros, Limit Applied)")
plt.legend(ncol=2)  # Display legend in two columns to save space
plt.show()


# %%


# case = 123
# data_case = data_df_times[data_df_times["type"] == case].copy()
data_case = data_df_times

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting relevant columns
T_sum_cols = [col for col in data_case.columns if "T_sum" in col]
Q_sum_cols = [col for col in data_case.columns if "Q_sum" in col]

# Compute all pairwise differences for T_sum and Q_sum
T_sum_diffs = {f"{col1}-{col2}": data_case[col1] - data_case[col2] for col1, col2 in itertools.combinations(T_sum_cols, 2)}
Q_sum_diffs = {f"{col1}-{col2}": data_case[col1] - data_case[col2] for col1, col2 in itertools.combinations(Q_sum_cols, 2)}

# Convert to DataFrames
T_sum_diff_df = pd.DataFrame(T_sum_diffs)
Q_sum_diff_df = pd.DataFrame(Q_sum_diffs)

# Compute total sums of T_sum and Q_sum
data_case["Total_T_sum"] = data_case[T_sum_cols].sum(axis=1)
data_case["Total_Q_sum"] = data_case[Q_sum_cols].sum(axis=1)

# Merge everything into a single DataFrame
data_analysis = pd.concat([T_sum_diff_df, Q_sum_diff_df, data_case[["Total_T_sum", "Total_Q_sum"]]], axis=1)

# %%


# Determine the actual number of available T_sum and Q_sum differences
num_available_T = len(T_sum_diff_df.columns)
num_available_Q = len(Q_sum_diff_df.columns)

if num_available_T == 0 or num_available_Q == 0:
    print("No valid T_sum or Q_sum differences to plot.")
else:
    num_plots = max(1, min(num_available_T, num_available_Q, 10))  # Ensure at least 1 plot

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))  # One column layout

    # Handle case where num_plots is 1 (plt.subplots returns an Axes object, not a list)
    if num_plots == 1:
        axes = [axes]  # Convert to a list for iteration

    for i in range(num_plots):
        t_col = list(T_sum_diff_df.columns)[i]
        q_col = list(Q_sum_diff_df.columns)[i]

        axes[i].scatter(data_analysis[t_col], data_analysis[q_col], alpha=0.5, s = 1)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-100, 100)
        # axes[i].set_ylim(-500, 500)
        axes[i].set_xlabel(t_col)
        axes[i].set_ylabel(q_col)
        axes[i].set_title(f"{t_col} vs {q_col}")

    plt.tight_layout()
    plt.show()


# %%



# case = 123
# data_case = data_df_times[data_df_times["type"] == case].copy()
data_case = data_df_times

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting relevant columns
T_sum_cols = [col for col in data_case.columns if "T_sum" in col]
Q_sum_cols = [col for col in data_case.columns if "Q_sum" in col]

# Compute all pairwise differences for T_sum and Q_sum (including all T_sum columns)
T_sum_diffs = {
    f"{col1}-{col2}": data_case[col1] - data_case[col2]
    for col1, col2 in itertools.combinations(T_sum_cols, 2)
}
Q_sum_diffs = {
    f"{col1}-{col2}": data_case[col1] - data_case[col2]
    for col1, col2 in itertools.combinations(Q_sum_cols, 2)
}

# Convert to DataFrames
T_sum_diff_df = pd.DataFrame(T_sum_diffs)
Q_sum_diff_df = pd.DataFrame(Q_sum_diffs)

# Compute total sums of T_sum and Q_sum
data_case["Total_T_sum"] = data_case[T_sum_cols].sum(axis=1)
data_case["Total_Q_sum"] = data_case[Q_sum_cols].sum(axis=1)

# Merge everything into a single DataFrame
data_analysis = pd.concat([T_sum_diff_df, Q_sum_diff_df, data_case[["Total_T_sum", "Total_Q_sum"]]], axis=1)

# %%

# Determine the actual number of available T_sum and Q_sum differences
num_available_T = len(T_sum_diff_df.columns)
num_available_Q = len(Q_sum_diff_df.columns)

if num_available_T == 0 or num_available_Q == 0:
    print("No valid T_sum or Q_sum differences to plot.")
else:
    num_plots = max(1, min(num_available_T, num_available_Q, 10))  # Ensure at least 1 plot

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))  # One column layout

    # Handle case where num_plots is 1 (plt.subplots returns an Axes object, not a list)
    if num_plots == 1:
        axes = [axes]  # Convert to a list for iteration

    for i in range(num_plots):
        if i >= len(T_sum_diff_df.columns) or i >= len(Q_sum_diff_df.columns):
            break  # Ensure we do not access out-of-range columns
        
        t_col = list(T_sum_diff_df.columns)[i]
        q_col = list(Q_sum_diff_df.columns)[i]

        axes[i].scatter(data_analysis[t_col], data_analysis[q_col], alpha=0.5, s=1)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-100, 100)
        axes[i].set_xlabel(t_col)
        axes[i].set_ylabel(q_col)
        axes[i].set_title(f"{t_col} vs {q_col}")

    plt.tight_layout()
    plt.show()


# %%


# Slewing correction with position correction etc


from scipy.constants import c

beta = 1

c_mm_ns = c/1000000
muon_speed = beta * c_mm_ns

# Y ----------------------------
def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_widths = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]  # T1-T3 and T2-T4 widths
y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_T1_and_T3 = y_widths[0]
y_width_T2_and_T4 = y_widths[1]
y_pos_T1_and_T3 = y_pos(y_width_T1_and_T3)
y_pos_T2_and_T4 = y_pos(y_width_T2_and_T4)

yz_big = np.array([[[y, z] for y in y_pos_T[i % 2]] for i, z in enumerate(z_positions)])

# First position
x_1 = "Ti_T_diff_j" * 200
yz_1 = yz_big[i, j]
xyz_1 = np.append(x_1, yz_1)

# Second position
x_2 = "Ti_T_diff_j" * 200
yz_2 = yz_big[i, j]
xyz_2 = np.append(x_2, yz_2)

# Length
dist = np.sqrt(np.sum((xyz_2 - xyz_1)**2))
travel_time = dist / muon_speed


# case = 123
# data_case = data_df_times[data_df_times["type"] == case].copy()
data_case = data_df_times

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extracting relevant columns
T_sum_cols = [col for col in data_case.columns if "T_sum" in col]
Q_sum_cols = [col for col in data_case.columns if "Q_sum" in col]

# Compute all pairwise differences for T_sum and Q_sum (including all T_sum columns)
T_sum_diffs = {
    f"{col1}-{col2}": data_case[col1] - data_case[col2] - travel_time
    for col1, col2 in itertools.combinations(T_sum_cols, 2)
}
Q_sum_diffs = {
    f"{col1}-{col2}": data_case[col1] - data_case[col2]
    for col1, col2 in itertools.combinations(Q_sum_cols, 2)
}

# Convert to DataFrames
T_sum_diff_df = pd.DataFrame(T_sum_diffs)
Q_sum_diff_df = pd.DataFrame(Q_sum_diffs)

# Compute total sums of T_sum and Q_sum
data_case["Total_T_sum"] = data_case[T_sum_cols].sum(axis=1)
data_case["Total_Q_sum"] = data_case[Q_sum_cols].sum(axis=1)

# Merge everything into a single DataFrame
data_analysis = pd.concat([T_sum_diff_df, Q_sum_diff_df, data_case[["Total_T_sum", "Total_Q_sum"]]], axis=1)

# %%

# Determine the actual number of available T_sum and Q_sum differences
num_available_T = len(T_sum_diff_df.columns)
num_available_Q = len(Q_sum_diff_df.columns)

if num_available_T == 0 or num_available_Q == 0:
    print("No valid T_sum or Q_sum differences to plot.")
else:
    num_plots = max(1, min(num_available_T, num_available_Q, 10))  # Ensure at least 1 plot

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))  # One column layout

    # Handle case where num_plots is 1 (plt.subplots returns an Axes object, not a list)
    if num_plots == 1:
        axes = [axes]  # Convert to a list for iteration

    for i in range(num_plots):
        if i >= len(T_sum_diff_df.columns) or i >= len(Q_sum_diff_df.columns):
            break  # Ensure we do not access out-of-range columns
        
        t_col = list(T_sum_diff_df.columns)[i]
        q_col = list(Q_sum_diff_df.columns)[i]

        axes[i].scatter(data_analysis[t_col], data_analysis[q_col], alpha=0.5, s=1)
        axes[i].set_xlim(-5, 5)
        axes[i].set_ylim(-100, 100)
        axes[i].set_xlabel(t_col)
        axes[i].set_ylabel(q_col)
        axes[i].set_title(f"{t_col} vs {q_col}")

    plt.tight_layout()
    plt.show()


# %%

data_case = data_df_times

import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c

##############################################################################
# 1) Define geometry and plane-strip mapping
##############################################################################

z_positions = np.array([30, 145, 290, 435])  # In mm

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

def y_pos(y_width):
    """Returns array of y-centers based on the widths of each strip."""
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

# For T1/T3 vs T2/T4
y_widths = [
    np.array([63, 63, 63, 98]),  # T1 / T3
    np.array([98, 63, 63, 63])   # T2 / T4
]

y_pos_T1_T3 = y_pos(y_widths[0])  # shape (4,)
y_pos_T2_T4 = y_pos(y_widths[1])  # shape (4,)

# yz_big[plane_index, strip_index, 0/1] = (y,z)
# plane_index in [0..3] => T1=0, T2=1, T3=2, T4=3
# strip_index in [0..3], but your data columns say 1..4 => we do minus 1
yz_big = np.zeros((4, 4, 2))

# Fill T1 -> plane_idx=0, T3 -> plane_idx=2 with y_pos_T1_T3
for strip_idx in range(4):
    yz_big[0, strip_idx, 0] = y_pos_T1_T3[strip_idx]  # y
    yz_big[0, strip_idx, 1] = z_positions[0]         # z for T1
    
    yz_big[2, strip_idx, 0] = y_pos_T1_T3[strip_idx]  # y
    yz_big[2, strip_idx, 1] = z_positions[2]         # z for T3

# Fill T2 -> plane_idx=1, T4 -> plane_idx=3 with y_pos_T2_T4
for strip_idx in range(4):
    yz_big[1, strip_idx, 0] = y_pos_T2_T4[strip_idx]  # y
    yz_big[1, strip_idx, 1] = z_positions[1]         # z for T2

    yz_big[3, strip_idx, 0] = y_pos_T2_T4[strip_idx]  # y
    yz_big[3, strip_idx, 1] = z_positions[3]         # z for T4

# Mapping T1->0, T2->1, T3->2, T4->3
plane_map = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}

def get_yz(plane_idx, strip_idx):
    """
    Return (y,z) for the given plane_idx (0..3) and strip_idx (0..3).
    We do yz_big[plane_idx, strip_idx], which is [y,z].
    """
    return yz_big[plane_idx, strip_idx, :]


##############################################################################
# 2) Define constants and speed
##############################################################################
c_mm_ns = c / 1000000
beta = 1.0
muon_speed = beta * c_mm_ns

##############################################################################
# 3) Parsing plane & strip from column names
#    (e.g. "T1_T_sum_3" => plane="T1", strip=3)
##############################################################################
def parse_plane_and_strip(col_name):
    """
    This regex expects columns like "T1_T_sum_2" or "T3_T_diff_4".
    If the column doesn't match, raises ValueError.
    """
    pattern = r"^(T[1-4])_T_(?:sum|diff)_(\d+)$"
    match = re.match(pattern, col_name)
    if not match:
        raise ValueError(f"Cannot parse plane/strip from '{col_name}'")
    plane_str = match.group(1)  # e.g. "T1"
    strip_str = match.group(2)  # e.g. "3"
    return plane_str, int(strip_str)


##############################################################################
# 4) Compute travel time function
#    - figure out plane/strip for each T_sum column
#    - find T_diff columns => x positions
#    - get (y,z) from yz_big
#    - distance / speed
##############################################################################
def compute_travel_time(row, col_sum1, col_sum2):
    plane1, strip1 = parse_plane_and_strip(col_sum1)  
    plane2, strip2 = parse_plane_and_strip(col_sum2)
    
    # T_diff columns for x
    tdiff_col1 = f"{plane1}_T_diff_{strip1}"
    tdiff_col2 = f"{plane2}_T_diff_{strip2}"
    
    # x1, x2 in mm
    x1 = row[tdiff_col1] * 200.0
    x2 = row[tdiff_col2] * 200.0
    
    # yz from yz_big
    i1 = plane_map[plane1]
    i2 = plane_map[plane2]
    j1 = strip1 - 1  # because your columns say e.g. strip=1..4
    j2 = strip2 - 1
    y1, z1 = get_yz(i1, j1)
    y2, z2 = get_yz(i2, j2)
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist / muon_speed  # time in ns


##############################################################################
# 5) Example: load or define your DataFrame with T-sum, T-diff, Q-sum columns
##############################################################################
# Suppose your actual data has columns like:
# ['type', 'Q1_Q_sum_1', 'Q1_Q_sum_2', ..., 'T1_T_sum_1', ..., 'T2_T_diff_2', ...]

# 5a) Filter out the T_sum, T_diff, Q_sum columns via regex so we don’t parse e.g. "type"
T_sum_cols = [
    col for col in data_case.columns
    if re.match(r"^T[1-4]_T_sum_[1-4]$", col)
]
T_diff_cols = [
    col for col in data_case.columns
    if re.match(r"^T[1-4]_T_diff_[1-4]$", col)
]
Q_sum_cols = [
    col for col in data_case.columns
    if re.match(r"^Q[1-4]_Q_sum_[1-4]$", col)
]

##############################################################################
# 6) Build T_sum differences minus travel time
##############################################################################
import math

T_sum_diffs = {}
# We do pairwise combinations of the T_sum columns
for col1, col2 in itertools.combinations(T_sum_cols, 2):
    diff_series = data_case.apply(
        # lambda row: row[col1] - row[col2] - compute_travel_time(row, col1, col2)
        lambda row: row[col1] - row[col2]
        if (row[col1] - row[col2]) != 0 else 0,  # Perform calculation only if difference is nonzero
        axis=1
    )
    T_sum_diffs[f"{col1}-{col2}"] = diff_series

T_sum_diff_df = pd.DataFrame(T_sum_diffs)

##############################################################################
# 7) Q_sum differences (no travel time needed)
##############################################################################
Q_sum_diffs = {}
for col1, col2 in itertools.combinations(Q_sum_cols, 2):
    Q_sum_diffs[f"{col1}-{col2}"] = data_case[col1] - data_case[col2]

Q_sum_diff_df = pd.DataFrame(Q_sum_diffs)

##############################################################################
# 8) Optionally compute total sums
##############################################################################
data_case["Total_T_sum"] = data_case[T_sum_cols].sum(axis=1)
data_case["Total_Q_sum"] = data_case[Q_sum_cols].sum(axis=1) if Q_sum_cols else np.nan

##############################################################################
# 9) Merge differences + totals
##############################################################################
data_analysis = pd.concat(
    [T_sum_diff_df, Q_sum_diff_df, data_case[["Total_T_sum", "Total_Q_sum"]]],
    axis=1
)

print("data_analysis:\n", data_analysis)

# %%

import matplotlib.pyplot as plt

# Get available T_sum and Q_sum difference columns
num_available_T = len(T_sum_diff_df.columns)
num_available_Q = len(Q_sum_diff_df.columns)

# Ensure we have data to plot
if num_available_T > 0 and num_available_Q > 0:
    # Set up multiple scatter plots
    num_plots = min(num_available_T, num_available_Q)  # Limit to the smaller of both
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 4 * num_plots))

    # If there's only one plot, make axes a list to iterate
    if num_plots == 1:
        axes = [axes]

    for i in range(num_plots):
        col_t = list(T_sum_diff_df.columns)[i]
        col_q = list(Q_sum_diff_df.columns)[i]
        
        x = data_analysis[col_t]
        y = data_analysis[col_q]
        cond = ( abs(x) <= 5 ) & ( abs(y) <= 100 )
        x = x[cond]
        y = y[cond]
        
        axes[i].scatter(x, y, alpha=0.6, s=1)
        axes[i].set_xlabel(col_t)
        axes[i].set_ylabel(col_q)
        # axes[i].set_xlim(-5, 5)
        # axes[i].set_ylim(-100, 100)
        axes[i].set_title(f"{col_t} vs. {col_q}")

    plt.tight_layout()
    plt.show()

else:
    print("No valid combination of T_sum/Q_sum differences to plot.")

# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

# Define Gaussian function for fitting
def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Get available T_sum and Q_sum difference columns
num_available_T = len(T_sum_diff_df.columns)
num_available_Q = len(Q_sum_diff_df.columns)

from scipy.stats import linregress

# Get available T_sum and Q_sum difference columns
num_available_T = len(T_sum_diff_df.columns)
num_available_Q = len(Q_sum_diff_df.columns)

# Ensure we have data to process
if num_available_T > 0 and num_available_Q > 0:
    corrected_T_sum_diff = {}

    for i in range(num_available_T):
        col_t = list(T_sum_diff_df.columns)[i]
        col_q = list(Q_sum_diff_df.columns)[i]

        x = data_analysis[col_t]
        y = data_analysis[col_q]

        # Apply condition to filter out extreme values
        cond = (abs(x) <= 5) & (abs(y) <= 100) & (y != 0) & (x != 0)
        x = x[cond]
        y = y[cond]

        # Perform linear regression using scipy.stats.linregress
        try:
            # Perform linear regression using scipy.stats.linregress
            slope, intercept, _, _, _ = linregress(y, x)
            # Correct T_sum_diff values using the regression line only if the original value is not 0
            corrected_x = np.where(x != 0, x - (slope * y + intercept), x)
            corrected_T_sum_diff[col_t] = corrected_x
        except Exception as e:
            print(f"Skipping fit for {col_t} due to error: {e}")
            corrected_T_sum_diff[col_t] = x  # Keep original values if fit fails
        
        # Plot in a scatter plot the original, the fit and the corrected values
        plt.figure(figsize=(6, 4))
        plt.scatter(y, x, alpha=0.6, s=1, label="Original")
        plt.scatter(y, slope * y + intercept, alpha=0.6, s=1, label="Fit")
        plt.scatter(y, corrected_x, alpha=0.6, s=1, label="Corrected")
        plt.xlabel(col_q)
        plt.ylabel(col_t)
        plt.title(f"{col_t} vs. {col_q}")
        plt.legend()
        plt.show()
        

    # Generate histograms before and after correction 
    fig, axes = plt.subplots(num_available_T, 1, figsize=(6, 4 * num_available_T))

    if num_available_T == 1:
        axes = [axes]

    for i in range(num_available_T):
        col_t = list(T_sum_diff_df.columns)[i]

        original_data = data_analysis[col_t]
        corrected_data = corrected_T_sum_diff[col_t]
        
        cond_og = ( original_data != 0 )
        cond_corr = ( corrected_data != 0 )
        original_data = original_data[cond_og]
        corrected_data = corrected_data[cond_corr]
        
        # Histogram bins
        bins = np.linspace(-5, 5, 100)

        # Gaussian fitting with error handling
        try:
            hist_original, bin_edges = np.histogram(original_data, bins=bins, density=True)
            hist_corrected, _ = np.histogram(corrected_data, bins=bins, density=True)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            popt_original, _ = curve_fit(gaussian, bin_centers, hist_original, p0=[0, 1, max(hist_original)])
            popt_corrected, _ = curve_fit(gaussian, bin_centers, hist_corrected, p0=[0, 1, max(hist_corrected)])

            spread_original = popt_original[1]
            spread_corrected = popt_corrected[1]

            # Plot histogram and Gaussian fits
            axes[i].hist(original_data, bins=bins, alpha=0.5, label=f"Original Spread: {spread_original:.3f}", density=True)
            axes[i].hist(corrected_data, bins=bins, alpha=0.5, label=f"Corrected Spread: {spread_corrected:.3f}", density=True)

            x_vals = np.linspace(-5, 5, 100)
            axes[i].plot(x_vals, gaussian(x_vals, *popt_original), 'r-', label="Original Fit")
            axes[i].plot(x_vals, gaussian(x_vals, *popt_corrected), 'b-', label="Corrected Fit")

            axes[i].set_xlabel(col_t)
            axes[i].set_title(f"Histogram of {col_t} (Before & After Correction)")
            axes[i].legend()
        except Exception as e:
            print(f"Skipping Gaussian fit for {col_t} due to error: {e}")

    plt.tight_layout()
    plt.show()

else:
    print("No valid combination of T_sum/Q_sum differences to plot.")

# %%
