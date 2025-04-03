#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:31:42 2024

@author: cayesoneira
"""

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

remove_crosstalk = True
remove_streamers = False
      
if remove_crosstalk or remove_streamers:
      for i in range(4):
            for j in range(4):
                  column_name = f"Q{i+1}_Q_sum_{j+1}"
                  v = data_df[column_name]
                  # If the row in v has a value between 0 and 2, then put to 0
                  # that Q sum values but also all the T sum, T diff qith the same i and j
                  
                  if remove_crosstalk:
                        data_df.loc[v.between(-100, 4), f"Q{i+1}_Q_sum_{j+1}"] = 0
                        data_df.loc[v.between(-100, 4), f"T{i+1}_T_sum_{j+1}"] = 0
                        data_df.loc[v.between(-100, 4), f"T{i+1}_T_diff_{j+1}"] = 0
                  
                  if remove_streamers:
                        data_df.loc[v.between(100, 1e6), f"Q{i+1}_Q_sum_{j+1}"] = 0
                        data_df.loc[v.between(100, 1e6), f"T{i+1}_T_sum_{j+1}"] = 0
                        data_df.loc[v.between(100, 1e6), f"T{i+1}_T_diff_{j+1}"] = 0



df = data_df
# filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.011)\
#                  & (df['type'] == '1234') & (df['s'] > -0.001)]
# filtered_df = df[(df['x'].abs() < 200) & (df['y'].abs() < 200) & (df['s'] < 0.031)\
                #   & (df['s'] > -0.031) & (df['charge_event'] < 500) ]

filtered_df = df

#%%

# Calculate the sum of Q4_Q_sum_1, Q4_Q_sum_2, Q4_Q_sum_3, Q4_Q_sum_4
filtered_df['Q1_total_sum'] = (filtered_df['Q1_Q_sum_1'] + 
                               filtered_df['Q1_Q_sum_2'] + 
                               filtered_df['Q1_Q_sum_3'] + 
                               filtered_df['Q1_Q_sum_4'])

filtered_df['Q2_total_sum'] = (filtered_df['Q2_Q_sum_1'] + 
                               filtered_df['Q2_Q_sum_2'] + 
                               filtered_df['Q2_Q_sum_3'] + 
                               filtered_df['Q2_Q_sum_4'])

filtered_df['Q3_total_sum'] = (filtered_df['Q3_Q_sum_1'] + 
                               filtered_df['Q3_Q_sum_2'] + 
                               filtered_df['Q3_Q_sum_3'] + 
                               filtered_df['Q3_Q_sum_4'])

filtered_df['Q4_total_sum'] = (filtered_df['Q4_Q_sum_1'] + 
                               filtered_df['Q4_Q_sum_2'] + 
                               filtered_df['Q4_Q_sum_3'] + 
                               filtered_df['Q4_Q_sum_4'])

filtered_df['Q_total_sum'] = (filtered_df['Q1_total_sum'] + 
                               filtered_df['Q2_total_sum'] + 
                               filtered_df['Q3_total_sum'] + 
                               filtered_df['Q4_total_sum'])

filtered_df = filtered_df[filtered_df['Q_total_sum'] < 300]

#%%


calibrated_data = df.copy()

filtered_df = df[(df['x'].abs() < 400) & (df['y'].abs() < 400) ]

def plot_charge_vs_tsum_diff_for_pairs(calibrated_data, set_common_ylim=False, y_lim=None, set_common_xlim=False, x_lim=None):
    # Create a figure with 4 rows and 3 columns
    fig, axs = plt.subplots(4, 6, figsize=(30, 15), constrained_layout=True)  # 4x3 grid (4 planes x 3 strip pairs)
    strip_pairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)]  # The three strip pairs

    for i_plane in range(1, 5):
        for i_pair, (strip1, strip2) in enumerate(strip_pairs):
            # Define the T_sum and Q_sum columns for the current plane and strip pair
            tsum_col_1 = f'T{i_plane}_T_sum_{strip1}'
            tsum_col_2 = f'T{i_plane}_T_sum_{strip2}'
            charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
            charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'

            # Calculate the time and charge differences for the selected strips
            tsum_diff = calibrated_data[tsum_col_1] - calibrated_data[tsum_col_2]
            charge_diff = calibrated_data[charge_col_1] - calibrated_data[charge_col_2]
            
            charge_lim = 0
            
            # Filter out rows where the differences are invalid (e.g., T_sum is zero or -100)
            valid_rows = (calibrated_data[tsum_col_1] != 0) & (calibrated_data[tsum_col_2] != 0) \
                         & (calibrated_data[charge_col_1] > charge_lim) & (calibrated_data[charge_col_2] > charge_lim)

            filtered_tsum_diff = tsum_diff[valid_rows]
            filtered_charge_diff = charge_diff[valid_rows]

            # Scatter plot of charge difference vs T_sum_diff for the current pair and plane
            axs[i_plane - 1, i_pair].scatter(filtered_charge_diff, filtered_tsum_diff, alpha=0.5, s=1)  # Thin points
            axs[i_plane - 1, i_pair].set_title(f'Plane T{i_plane}: Strips {strip1}-{strip2}')
            axs[i_plane - 1, i_pair].set_xlabel('Charge Difference')
            axs[i_plane - 1, i_pair].set_ylabel('T_sum Difference')
            axs[i_plane - 1, i_pair].grid(True)

            # Set common y-axis limits if required
            if set_common_ylim and y_lim:
                axs[i_plane - 1, i_pair].set_ylim(y_lim)

            # Set common x-axis limits if required
            if set_common_xlim and x_lim:
                axs[i_plane - 1, i_pair].set_xlim(x_lim)

    plt.suptitle(f'Charge Difference vs T_sum Difference for Different Strip Pairs and Planes, Q > {charge_lim}', fontsize=16)
    plt.show()

# Example usage:
plot_charge_vs_tsum_diff_for_pairs(calibrated_data, set_common_ylim=True, y_lim=(-2, 2), set_common_xlim=True, x_lim=(-100, 100))

# %%

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Define sigmoid function
def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

# Strip pairs to analyze
strip_pairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)]

# Copy the calibrated dataframe to add new columns
df_slew = calibrated_data.copy()

# Iterate over each plane
for i_plane in range(1, 5):
    residuals_dict = {}

    # Compute differences and fit sigmoid for each strip pair
    for strip1, strip2 in strip_pairs:
        tsum_col_1 = f'T{i_plane}_T_sum_{strip1}'
        tsum_col_2 = f'T{i_plane}_T_sum_{strip2}'
        charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
        charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'

        tsum_diff = df_slew[tsum_col_1] - df_slew[tsum_col_2]
        charge_diff = df_slew[charge_col_1] - df_slew[charge_col_2]

        charge_lim = 0
        valid_rows = (df_slew[tsum_col_1] != 0) & (df_slew[tsum_col_2] != 0) \
                     & (df_slew[charge_col_1] > charge_lim) & (df_slew[charge_col_2] > charge_lim)

        filtered_tsum_diff = tsum_diff[valid_rows]
        filtered_charge_diff = charge_diff[valid_rows]

        if len(filtered_charge_diff) > 0:
            try:
                params, _ = curve_fit(sigmoid, filtered_charge_diff, filtered_tsum_diff, p0=[2, 1, 0, 0])
                fitted_vals = sigmoid(charge_diff, *params)
                residuals_dict[f'{strip1}{strip2}'] = (tsum_diff - fitted_vals).fillna(0)
            except:
                residuals_dict[f'{strip1}{strip2}'] = pd.Series(np.zeros_like(tsum_diff), index=tsum_diff.index)
        else:
            residuals_dict[f'{strip1}{strip2}'] = pd.Series(np.zeros_like(tsum_diff), index=tsum_diff.index)

    # Create system of equations and solve for corrected T_sum values (_slew)
    # Strip 1: x, Strip 2: y, Strip 3: z, Strip 4: t
    x_prime = (residuals_dict['12'] + residuals_dict['13'] + residuals_dict['14']) / 3
    y_prime = x_prime - residuals_dict['12']
    z_prime = y_prime - residuals_dict['23']
    t_prime = x_prime - residuals_dict['14']

    df_slew[f'T{i_plane}_T_sum_1_slew'] = x_prime
    df_slew[f'T{i_plane}_T_sum_2_slew'] = y_prime
    df_slew[f'T{i_plane}_T_sum_3_slew'] = z_prime
    df_slew[f'T{i_plane}_T_sum_4_slew'] = t_prime


# %%

# Plot the crossed correlations before and after the slewing correction. I mean with the charge difference

# Create a figure with 4 rows and 3 columns
fig, axs = plt.subplots(4, 6, figsize=(30, 15), constrained_layout=True)  # 4x3 grid (4 planes x 3 strip pairs)

for i_plane in range(1, 5):
    for i_pair, (strip1, strip2) in enumerate(strip_pairs):
        charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
        charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'
        charge_diff = df_slew[charge_col_1] - df_slew[charge_col_2]
        
        tsum_col_1 = f'T{i_plane}_T_sum_{strip1}'
        tsum_col_2 = f'T{i_plane}_T_sum_{strip2}'
        tsum_diff = df_slew[tsum_col_1] - df_slew[tsum_col_2]
        
        tsum_col_1 = f'T{i_plane}_T_sum_{strip1}_slew'
        tsum_col_2 = f'T{i_plane}_T_sum_{strip2}_slew'
        tsum_diff_slew = df_slew[tsum_col_1] - df_slew[tsum_col_2]
        
        charge_lim = 0
        valid_rows = (df_slew[tsum_col_1] != 0) & (df_slew[tsum_col_2] != 0) \
                     & (df_slew[charge_col_1] > charge_lim) & (df_slew[charge_col_2] > charge_lim)

        filtered_tsum_diff = tsum_diff[valid_rows]
        filtered_tsum_diff_slew = tsum_diff_slew[valid_rows]
        filtered_charge_diff = charge_diff[valid_rows]
        
        # Scatter plot of charge difference vs T_sum_diff for the current pair and plane
        axs[i_plane - 1, i_pair].scatter(filtered_charge_diff, filtered_tsum_diff, alpha=0.5, s=1, label='Original')
        axs[i_plane - 1, i_pair].scatter(filtered_charge_diff, filtered_tsum_diff_slew, alpha=0.5, s=1, label='Slewed')
        axs[i_plane - 1, i_pair].set_title(f'Plane T{i_plane}: Strips {strip1}-{strip2}')
        axs[i_plane - 1, i_pair].set_xlabel('Charge Difference')
        axs[i_plane - 1, i_pair].set_ylabel('T_sum')
        axs[i_plane - 1, i_pair].legend()
        axs[i_plane - 1, i_pair].grid(True)
        
        set_common_xlim=True; x_lim=(-100, 100)
        set_common_ylim=True; y_lim=(-2, 2)
        
        # Set common y-axis limits if required
        if set_common_ylim and y_lim:
            axs[i_plane - 1, i_pair].set_ylim(y_lim)

        # Set common x-axis limits if required
        if set_common_xlim and x_lim:
            axs[i_plane - 1, i_pair].set_xlim(x_lim)

#%%

# Plot the prime variables in the same histogram as the original data
fig, axs = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)

for i_plane in range(1, 5):
    for i_strip in range(1, 5):
        tsum_col = f'T{i_plane}_T_sum_{i_strip}'
        tsum_col_slew = f'T{i_plane}_T_sum_{i_strip}_slew'
        
        charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
        charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'
        charge_diff = df_slew[charge_col_1] - df_slew[charge_col_2]
        
        tsum_col_1 = f'T{i_plane}_T_sum_{strip1}'
        tsum_col_2 = f'T{i_plane}_T_sum_{strip2}'
        tsum_diff = df_slew[tsum_col_1] - df_slew[tsum_col_2]
        
        tsum_col_1 = f'T{i_plane}_T_sum_{strip1}_slew'
        tsum_col_2 = f'T{i_plane}_T_sum_{strip2}_slew'
        tsum_diff_slew = df_slew[tsum_col_1] - df_slew[tsum_col_2]
        
        charge_lim = 0
        valid_rows = (df_slew[tsum_col_1] != 0) & (df_slew[tsum_col_2] != 0) \
                     & (df_slew[charge_col_1] > charge_lim) & (df_slew[charge_col_2] > charge_lim)
        
        filtered_tsum_col = df_slew[tsum_col][valid_rows]
        filtered_tsum_col_slew = df_slew[tsum_col_slew][valid_rows]
        
        axs[i_plane - 1, i_strip - 1].hist(filtered_tsum_col, bins=100, alpha=0.5, label='Original')
        axs[i_plane - 1, i_strip - 1].hist(filtered_tsum_col_slew, bins=100, alpha=0.5, label='Slewed')
        axs[i_plane - 1, i_strip - 1].set_title(f'Plane T{i_plane}, Strip {i_strip}')
        axs[i_plane - 1, i_strip - 1].legend()

#%%

def plot_charge_vs_tdif_diff_for_pairs(calibrated_data, set_common_ylim=False, y_lim=None, set_common_xlim=False, x_lim=None):
    # Create a figure with 4 rows and 3 columns
    fig, axs = plt.subplots(4, 6, figsize=(30, 15), constrained_layout=True)  # 4x3 grid (4 planes x 3 strip pairs)
    strip_pairs = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)]  # The three strip pairs

    for i_plane in range(1, 5):
        for i_pair, (strip1, strip2) in enumerate(strip_pairs):
            # Define the T_dif and Q_dif columns for the current plane and strip pair
            tdif_col_1 = f'T{i_plane}_T_diff_{strip1}'
            tdif_col_2 = f'T{i_plane}_T_diff_{strip2}'
            charge_col_1 = f'Q{i_plane}_Q_sum_{strip1}'
            charge_col_2 = f'Q{i_plane}_Q_sum_{strip2}'

            # Calculate the time and charge differences for the selected strips
            tdif_diff = calibrated_data[tdif_col_1] - calibrated_data[tdif_col_2]
            charge_diff = calibrated_data[charge_col_1] - calibrated_data[charge_col_2]
            
            # Define a threshold to filter charge (set to 0 to remove invalid data)
            charge_lim = 0
            
            # Filter out rows where the differences are invalid (e.g., T_dif is zero or -100)
            valid_rows = (calibrated_data[tdif_col_1] != 0) & (calibrated_data[tdif_col_2] != 0) \
                         & (calibrated_data[charge_col_1] > charge_lim) & (calibrated_data[charge_col_2] > charge_lim)

            filtered_tdif_diff = tdif_diff[valid_rows]
            filtered_charge_diff = charge_diff[valid_rows]

            # Scatter plot of charge difference vs T_dif_diff for the current pair and plane
            axs[i_plane - 1, i_pair].scatter(filtered_charge_diff, filtered_tdif_diff, alpha=0.5, s=1)  # Thin points
            axs[i_plane - 1, i_pair].set_title(f'Plane T{i_plane}: Strips {strip1}-{strip2}')
            axs[i_plane - 1, i_pair].set_xlabel('Charge Difference')
            axs[i_plane - 1, i_pair].set_ylabel('T_dif Difference')
            axs[i_plane - 1, i_pair].grid(True)

            # Set common y-axis limits if required
            if set_common_ylim and y_lim:
                axs[i_plane - 1, i_pair].set_ylim(y_lim)

            # Set common x-axis limits if required
            if set_common_xlim and x_lim:
                axs[i_plane - 1, i_pair].set_xlim(x_lim)

    plt.suptitle(f'Charge Difference vs T_dif Difference for Different Strip Pairs and Planes, Q > {charge_lim}', fontsize=16)
    plt.show()

# Example usage:
plot_charge_vs_tdif_diff_for_pairs(calibrated_data, set_common_ylim=True, y_lim=(-0.5, 0.5), set_common_xlim=True, x_lim=(-100, 100))

# %%
