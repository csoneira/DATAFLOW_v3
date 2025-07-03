from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import builtins
import os
import random
import shutil
import sys
from datetime import datetime

# Third-party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import poisson
from tqdm import tqdm

# -----------------------------------------------------------------------------

# If the minutes of the time of execution are between 0 and 5 then put update_big_event_file to True
# if datetime.now().minute < 5:
#     update_big_event_file = True

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Running big_event_file_joiner.py -------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

# Check if the script has an argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

# Get the station argument
station = sys.argv[1]
print(f"Station: {station}")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

fig_idx = 0
plot_list = []

station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
acc_working_directory = os.path.join(working_directory, "LIST_TO_ACC")

# Define subdirectories relative to the working directory
base_directories = {
    "list_events_directory": os.path.join(working_directory, "LIST_EVENTS_DIRECTORY"),
    
    "base_plots_directory": os.path.join(acc_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(acc_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(acc_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(acc_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "unprocessed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_UNPROCESSED"),
    "processing_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_PROCESSING"),
    "error_directory": os.path.join(acc_working_directory, "ACC_FILES/ERROR_DIRECTORY"),
    "completed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_COMPLETED"),
    
    "acc_events_directory": os.path.join(working_directory, "ACC_EVENTS_DIRECTORY"),
    "acc_rejected_directory": os.path.join(working_directory, "ACC_REJECTED"),
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

# Path to big_event_data.csv
big_event_file = os.path.join(working_directory, "big_event_data.csv")

# Erase all files in the figure_directory
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory)

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))

# --------------------------------------------------------------------------------------------
# Move small or too big files in the destination folder to a directory of rejected -----------
# --------------------------------------------------------------------------------------------

# source_dir = base_directories["acc_events_directory"]
# rejected_dir = base_directories["acc_rejected_directory"]

# for filename in os.listdir(source_dir):
#     file_path = os.path.join(source_dir, filename)
    
#     # Check if it's a file
#     if os.path.isfile(file_path):
#         # Count the number of lines in the file
#         with open(file_path, "r") as f:
#             line_count = sum(1 for _ in f)

#         # Move the file if it has < 15 or > 100 rows
#         if line_count < 10 or line_count > 300:
#             shutil.move(file_path, os.path.join(rejected_dir, filename))
#             print(f"Moved: {filename}")


list_events_directory = base_directories["list_events_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
processing_directory = base_directories["processing_directory"]
error_directory = base_directories["error_directory"]
completed_directory = base_directories["completed_directory"]

list_event_files = set(os.listdir(list_events_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

# Files to copy: in LIST but not in UNPROCESSED, PROCESSING, or COMPLETED
files_to_copy = list_event_files - unprocessed_files - processing_files - completed_files

# Copy files to UNPROCESSED
for file_name in files_to_copy:
    src_path = os.path.join(list_events_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        # Copy instead of move
        shutil.copy(src_path, dest_path)
        print(f"Copied {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")



work_big_event_file = True
update_big_event_file = False
create_big_event_file = True

def round_to_significant_digits(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    return x

def combine_duplicates(group):
    # print("-----------------------------------------------------------------")
    
    # Drop fully NaN rows (except 'Time')
    group = group.dropna(subset=[col for col in group.columns if col not in ["Time", "execution_date"]], how='all')

    # If only one row remains, return it
    if len(group) == 1:
        return group.iloc[[0]]

    # Sort by execution_date (latest first)
    group = group.sort_values("execution_date", ascending=False)

    # Identify indices that are consecutive (i.e., abs difference of 1)
    indices = group.index.to_numpy()
    consecutive_indices = {i for i in indices for j in indices if abs(i - j) == 1}

    # Keep only the two most recent rows among the consecutive ones
    consecutive_group = group.loc[group.index.intersection(consecutive_indices)].head(2)

    if len(consecutive_group) == 2:
        print(f"Aggregating two consecutive rows: {consecutive_group.index.tolist()}")
        
        time_value = consecutive_group.iloc[0]["Time"]
        
        # Fill NaNs in each row using values from the other row (fixed)
        consecutive_group.iloc[0] = consecutive_group.iloc[0].fillna(consecutive_group.iloc[1])
        consecutive_group.iloc[1] = consecutive_group.iloc[1].fillna(consecutive_group.iloc[0])

        # Extract weights (events column)
        weights = consecutive_group["events"].fillna(0)  # Ensure no NaNs in weights

        # Columns to average (weighted)
        suffixes = [
            # Summary metrics and quality flags
            'CRT_avg', 'sigmoid_width', 'background_slope',
            'one_side_events', 'purity_of_data_percentage',
            'unc_y', 'unc_tsum', 'unc_tdif',

            # Reconstruction outputs
            'x', 'y', 'theta', 'phi', 's', 'th_chi',
            'x_std', 'y_std', 'theta_std', 'phi_std', 's_std', 'th_chi_std',

            # Streamer fractions
            'streamer_percent_1', 'streamer_percent_2', 'streamer_percent_3', 'streamer_percent_4',

            # Config
            'over_P1', 'P1-P2', 'P2-P3', 'P3-P4', 'phi_north',
        ]

        avg_columns = [col for col in group.columns if any(s in col for s in suffixes)]
    
        # All other columns (except "Time") are summed
        sum_columns = [col for col in group.columns if col not in avg_columns and col not in ["Time"]]

        # Aggregation logic: Weighted mean for avg_columns, sum for sum_columns
        aggregated_result = {}

        for col in group.columns:
            if col in ["execution_date"]:
                continue
            if col in ["Time"]:
                aggregated_result[col] = time_value
                continue  # Skip these columns
            elif col in avg_columns:
                # Compute weighted mean
                aggregated_result[col] = (consecutive_group[col] * weights).sum() / weights.sum()
            elif col == "events":
                # Sum events
                aggregated_result[col] = consecutive_group[col].sum()
            else:
                # Sum other columns
                aggregated_result[col] = consecutive_group[col].sum(min_count=1)

        return pd.DataFrame([pd.Series(aggregated_result)])

    # Otherwise, return the most recent row
    return group.iloc[[0]]


if work_big_event_file:
    if update_big_event_file:
        print("Whatever.")
        
        if os.path.exists(big_event_file):
            big_event_df = pd.read_csv(big_event_file, sep=',', parse_dates=['Time'])
            print(f"Loaded existing big_event_data.csv with {len(big_event_df)} rows.")
            
            # big_event_df = pd.concat([big_event_df, resampled_df], ignore_index=True)
            
        else:
            # big_event_df = pd.DataFrame(columns=resampled_df.columns)
            print("Created new empty big_event_data.csv dataframe.")
            create_big_event_file = True

    if create_big_event_file:
        print("Creating big_event_data.csv...")
        
        big_event_df = pd.DataFrame()
        
        # Process all CSV files in ACC_EVENTS_DIRECTORY
        acc_directory = base_directories["acc_events_directory"]  # Get the directory where new CSVs are saved
        csv_files = [f for f in os.listdir(acc_directory) if f.endswith('.csv')]
        
        csv_files = sorted(csv_files)
        
        # Add a tqdm progress bar
        iterator = tqdm(csv_files, total=len(csv_files), desc="Joining CSVs")

        for csv_file in iterator:
            csv_path = os.path.join(acc_directory, csv_file)

            # print(f"Merging file: {csv_path}")
            new_data = pd.read_csv(csv_path, sep=',', parse_dates=['Time'])
            
            # Put 0s to NaN
            new_data = new_data.replace(0, np.nan)
            new_data = new_data.copy()
            
            new_data['Time'] = new_data['Time'].dt.floor('1min')  # Round to minute precision
            new_data = new_data.copy()
            
            # Add as a new column, the this_time = os.path.getmtime(csv_path)
            new_data["execution_date"] = os.path.getmtime(csv_path)
            big_event_df = pd.concat([big_event_df, new_data], ignore_index=True)
    
    # Once created or updated, we need to handle duplicates in 'Time'
    print("Grouping the CSVs by 'Time' and combining duplicates...")
    
    # Print the columns of big_event_df
    print("Columns in big_event_df:", big_event_df.columns.to_list())
    
    # Group by 'Time' to combine duplicates
    print("Combining duplicates...")
    
    tqdm.pandas()
    big_event_df = big_event_df.groupby('Time', as_index=False).progress_apply(combine_duplicates).reset_index(drop=True)
    
    # Ensure big_event_df is a DataFrame
    if not isinstance(big_event_df, pd.DataFrame):
        print("Warning: big_event_df is not a DataFrame. Converting it...")
        big_event_df = big_event_df.to_frame()  # Convert Series to DataFrame if needed
    
    big_event_df = big_event_df.sort_values(by="Time")  # Now sorting should work fine
    
    # Put every 0 to NaN, if this is cheaper in terms of memory
    print("Replacing 0s with NaNs...")
    big_event_df = big_event_df.replace(0, np.nan)
    
    # -----------------------------------------------------------------------------
    # Save the updated big_event_data.csv -----------------------------------------
    # -----------------------------------------------------------------------------
    # numeric_cols = big_event_df.select_dtypes(include=['number']).columns
    # big_event_df[numeric_cols] = big_event_df[numeric_cols].applymap(round_to_significant_digits)

    big_event_df.to_csv(big_event_file, sep=',', index=False)
    
    # # Print type of the dataframe
    # print(type(big_event_df))  # Should be <class 'pandas.DataFrame'>
    
    # # Print head of the dataframe
    # print(big_event_df.head())
    # print(big_event_df.tail())
    
    print(big_event_df.columns.to_list())
    
    print(f"Saved big_event_data.csv with {len(big_event_df)} rows.")

else:
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("big_event_data.csv not updated by configuration.")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")

