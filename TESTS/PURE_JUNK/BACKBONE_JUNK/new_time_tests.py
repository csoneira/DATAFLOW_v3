#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


fast_mode = False # Do not iterate TimTrack, neither save figures, etc.
debug_mode = False # Only 10000 rows with all detail
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
      # unprocessed_files = sorted(unprocessed_files)
      # file_name = unprocessed_files[-1]
      
      file_name = random.choice(unprocessed_files)
      
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

#%%

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
            axs[i, j].hist(v, bins=20)
            axs[i, j].set_title(column_name)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%

fig, axs = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle(f"Station {station} - Histogram of Q_sum values", fontsize=20)

for i in range(4):
      for j in range(4):
            column_name = f"Q{i+1}_Q_sum_{j+1}"
            v = data_df_times[column_name].copy()
            v = v[v != 0]
            v = v[v <= 5]
            axs[i, j].hist(v, bins=20)
            axs[i, j].set_title(column_name)
            # axs[i, j].set_xscale('log')
            axs[i, j].set_xlim(0, 5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%

remove_crosstalk = True
remove_streamers = True

print(data_df_times.columns.to_list())
# ['type',

# 'Q1_Q_sum_1', 'Q1_Q_sum_2', 'Q1_Q_sum_3', 'Q1_Q_sum_4',
# 'Q2_Q_sum_1', 'Q2_Q_sum_2', 'Q2_Q_sum_3', 'Q2_Q_sum_4', 'Q3_Q_sum_1', 
# 'Q3_Q_sum_2', 'Q3_Q_sum_3', 'Q3_Q_sum_4', 'Q4_Q_sum_1', 'Q4_Q_sum_2',
# 'Q4_Q_sum_3', 'Q4_Q_sum_4',

# 'T1_T_sum_1', 'T1_T_sum_2', 'T1_T_sum_3', 
# 'T1_T_sum_4', 'T2_T_sum_1', 'T2_T_sum_2', 'T2_T_sum_3', 'T2_T_sum_4', 
# 'T3_T_sum_1', 'T3_T_sum_2', 'T3_T_sum_3', 'T3_T_sum_4', 'T4_T_sum_1', 
# 'T4_T_sum_2', 'T4_T_sum_3', 'T4_T_sum_4',

# 'T1_T_diff_1', 'T1_T_diff_2', 'T1_T_diff_3', 'T1_T_diff_4', 'T2_T_diff_1',
# 'T2_T_diff_2', 'T2_T_diff_3', 
# 'T2_T_diff_4', 'T3_T_diff_1', 'T3_T_diff_2', 'T3_T_diff_3', 'T3_T_diff_4', 
# 'T4_T_diff_1', 'T4_T_diff_2', 'T4_T_diff_3', 'T4_T_diff_4']
      
if remove_crosstalk or remove_streamers:
      for i in range(4):
            for j in range(4):
                  column_name = f"Q{i+1}_Q_sum_{j+1}"
                  v = data_df_times[column_name]
                  # If the row in v has a value between 0 and 2, then put to 0
                  # that Q sum values but also all the T sum, T diff qith the same i and j
                  
                  if remove_crosstalk:
                        data_df_times.loc[v.between(-100, 4), f"Q{i+1}_Q_sum_{j+1}"] = 0
                        data_df_times.loc[v.between(-100, 4), f"T{i+1}_T_sum_{j+1}"] = 0
                        data_df_times.loc[v.between(-100, 4), f"T{i+1}_T_diff_{j+1}"] = 0
                  
                  if remove_streamers:
                        data_df_times.loc[v.between(100, 1e6), f"Q{i+1}_Q_sum_{j+1}"] = 0
                        data_df_times.loc[v.between(100, 1e6), f"T{i+1}_T_sum_{j+1}"] = 0
                        data_df_times.loc[v.between(100, 1e6), f"T{i+1}_T_diff_{j+1}"] = 0
      

print(data_df_times['type'].unique())

# %%

case = 1234
data_case = data_df_times[data_df_times["type"] == case].copy()

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

x_limit = 200

# Plot histograms overlayed
plt.figure(figsize=(12, 6))

bin_number = 100
colors = [
    "blue", "red", "green", "cyan", "magenta", "orange", "purple", "brown", "pink", "yellow",
    "lime", "teal", "gold", "navy", "violet", "gray", "salmon", "indigo", "turquoise", "maroon"
]

for i in range(8):
      # Filter data before plotting
      max_filtered = data_case[max_distance_cols[i]].dropna()
      max_filtered = max_filtered[max_filtered <= x_limit]  # Apply x_limit filter

      closest_filtered = data_case[closest_distance_cols[i]].dropna()
      closest_filtered = closest_filtered[closest_filtered <= x_limit]  # Apply x_limit filter

      # Plot furthest distances with solid lines
      plt.hist(max_filtered, bins=bin_number, alpha=0.4, 
                  label=f"{i+1}th Max Distance", color=colors[i], edgecolor='black')

      # Plot closest distances with a dashed outline
      plt.hist(closest_filtered, bins=bin_number, alpha=0.4, 
                  label=f"{i+1}th Closest Distance", color=colors[i], hatch='//', edgecolor='black')


plt.yscale("log")
plt.title("Histogram of T_sum Distance Metrics (20 Furthest & 20 Closest, No Zeros)")
plt.legend(ncol=2)  # Display legend in two columns to save space
plt.xlim(0, x_limit)
plt.show()

# %%

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Assuming data_df_times is provided and contains the relevant data
case = 34
data_case = data_df_times[data_df_times["type"] == case].copy()

# Extract T_sum columns
T_sum_cols = data_case.filter(regex='T_sum', axis=1)

# Function to compute all pairwise differences and count occurrences of large differences
def compute_pairwise_differences(data, threshold=5):
    problem_counter = Counter()
    all_differences = []

    for _, row in data.iterrows():
        values = row.dropna().values  # Drop NaNs
        values = values[values != 0]  # Exclude zeros
        indices = row.dropna().index  # Keep track of column names

        if len(values) < 2:
            continue  # No pairs to process

        for i, j in itertools.combinations(range(len(values)), 2):
            diff = abs(values[i] - values[j])
            all_differences.append(diff)

            if diff > threshold:
                problem_counter[indices[i]] += 1
                problem_counter[indices[j]] += 1

    return problem_counter, all_differences

# Compute problematic T_sums and all pairwise differences
problem_counts, all_differences = compute_pairwise_differences(T_sum_cols, threshold=7)

# Convert to sorted lists for plotting
problem_values = list(problem_counts.values())

# Plot histogram of all pairwise differences
plt.figure(figsize=(10, 5))
plt.hist(all_differences, bins=50, edgecolor="black", alpha=0.7)
plt.yscale("log")  # Log scale for better visibility
plt.xlabel("Pairwise T_sum Differences (ns)")
plt.ylabel("Frequency (log scale)")
plt.title("Histogram of All Pairwise T_sum Differences")
plt.grid(True)
plt.show()

# Plot histogram of bad pairings per T_sum
plt.figure(figsize=(10, 5))
plt.hist(problem_values, bins=30, edgecolor="black", alpha=0.7)
plt.yscale("log")  # Log scale for better visibility
plt.xlabel("Number of Bad Pairings per T_sum (>5 ns)")
plt.ylabel("Frequency (log scale)")
plt.title("Histogram of T_sum Values by Number of Bad Pairings")
plt.grid(True)
plt.show()

# Identify T_sum columns to be zeroed (if they appear in 3 or more bad pairings)
flagged_T_sums = {t_sum for t_sum, count in problem_counts.items() if count >= 3}

# Debug: Print flagged T_sum columns
print("T_sums flagged for zeroing:", flagged_T_sums)

# Zero out flagged T_sum values and related columns
for col in flagged_T_sums:
    data_case[col] = 0  # Zero the T_sum values
    related_cols = [col.replace("T_sum", suffix) for suffix in ["Tdiff", "Qdiff"]]  # Adjust suffixes
    for r_col in related_cols:
        if r_col in data_case.columns:
            data_case[r_col] = 0  # Zero out related values

# Debug: Check zeroing
for col in flagged_T_sums:
    print(f"Zeroed {col}: Unique values after zeroing ->", data_case[col].unique())


# %%
