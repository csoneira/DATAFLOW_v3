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

remove_crosstalk = False
remove_streamers = False

print(data_df_times.columns.to_list())
      
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

cases = [1234, 123, 234, 124, 134, 12, 23, 34, 13, 14, 24]
cmap = get_cmap('turbo')
colors = cmap(np.linspace(0, 1, len(cases)))

# Define window widths
widths = np.linspace(1, 20, 80)

plt.figure(figsize=(10, 6))

for idx, case in enumerate(cases):
    data_case = data_df_times[data_df_times["type"] == case].copy()
    
    # Extract only the _T_sum_ columns
    t_sum_columns = [col for col in data_case.columns if "_T_sum_" in col]
    t_sum_data = data_case[t_sum_columns].values  # shape: (n_events, 16)

    counts_per_width = []
    counts_per_width_dev = []
    
    for w in widths:
        count_in_window = []

        for row in t_sum_data:
            row_no_zeros = row[row != 0]
            if len(row_no_zeros) == 0:
                count_in_window.append(0)
                continue

            stat = np.mean(row_no_zeros)
            lower = stat - w / 2
            upper = stat + w / 2
            n_in_window = np.sum((row_no_zeros >= lower) & (row_no_zeros <= upper))
            count_in_window.append(n_in_window)

        counts_per_width.append(np.mean(count_in_window))
        counts_per_width_dev.append(np.std(count_in_window))

    plt.scatter(widths, counts_per_width / np.max(counts_per_width), color=colors[idx], label=f"type {case}")
    counts_per_width = np.array(counts_per_width)
    counts_per_width_dev = np.array(counts_per_width_dev)
    plt.fill_between( widths, (counts_per_width - counts_per_width_dev) / np.max(counts_per_width), (counts_per_width + counts_per_width_dev) / np.max(counts_per_width), color=colors[idx], alpha=0.2)

plt.xlabel("Window width (ns)")
plt.ylabel("Average number of non-zero T_sum values in window")
plt.title("Counts inside statistic-centered window vs w")
plt.legend()
plt.grid(True)
# Set the background color of the plot to the first colour of the colormap
# plt.gca().set_facecolor(cmap(0))
plt.tight_layout()
plt.show()

# %%


calibrated_data = data_df_times.copy()
time_window = 7

# Calculate the mean of the T_sum values for each row, considering only non-zero values
T_sum_columns = calibrated_data.filter(regex='_T_sum_')
mean_T_sum = T_sum_columns.apply(lambda row: row[row != 0].median() if row[row != 0].size > 0 else 0, axis=1)

# Calculate the difference between each T_sum value and the mean, but only for non-zero values
diff_T_sum = T_sum_columns.sub(mean_T_sum, axis=0)

# Check if the difference is within the time window, ignoring zero values
time_window_mask = np.abs(diff_T_sum) <= time_window
time_window_mask[T_sum_columns == 0] = True  # Ignore zero values in the comparison

# Apply the mask to the data
T_sum_columns[~time_window_mask] = 0

# Calculate how many values were set to zero
num_zeroed = (~time_window_mask).values.sum()
num_total = time_window_mask.size  # total number of elements

zeroed_percentage = num_zeroed / num_total

if zeroed_percentage > 0:
    print(f"Zeroed {zeroed_percentage:.2%} of the values outside the time window.")
