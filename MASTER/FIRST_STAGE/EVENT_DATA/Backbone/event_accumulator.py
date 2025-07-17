from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

print("\n\n")
print("__| |____________________________________________________________________________________________________| |__")
print("__   ____________________________________________________________________________________________________   __")
print("  | |                                                                                                    | |  ")
print("  | |                      _                                           _       _                         | |  ")
print("  | |  _____   _____ _ __ | |_     __ _  ___ ___ _   _ _ __ ___  _   _| | __ _| |_ ___  _ __ _ __  _   _ | |  ")
print("  | | / _ \\ \\ / / _ \\ '_ \\| __|   / _` |/ __/ __| | | | '_ ` _ \\| | | | |/ _` | __/ _ \\| '__| '_ \\| | | || |  ")
print("  | ||  __/\\ V /  __/ | | | |_   | (_| | (_| (__| |_| | | | | | | |_| | | (_| | || (_) | |_ | |_) | |_| || |  ")
print("  | | \\___| \\_/ \\___|_| |_|\\__|___\\__,_|\\___\\___|\\__,_|_| |_| |_|\\__,_|_|\\__,_|\\__\\___/|_(_)| .__/ \\__, || |  ")
print("  | |                        |_____|                                                        |_|    |___/ | |  ")
print("__| |____________________________________________________________________________________________________| |__")
print("__   ____________________________________________________________________________________________________   __")
print("  | |                                                                                                    | |  ")
print("\n\n")


# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# ----------------------------- Standard Library -----------------------------
import os
import sys
import math
import random
import shutil
import builtins
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union

# ---------------------------- Third-party Libraries --------------------------
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from tqdm import tqdm

from scipy.stats import poisson
from scipy.optimize import minimize, curve_fit, nnls
from scipy.special import gamma
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, CubicSpline, RegularGridInterpolator
from scipy.sparse import load_npz, csc_matrix


# -----------------------------------------------------------------------------

correct_angle = False
last_file_test = False
update_big_event_file = False

# If the minutes of the time of execution are between 0 and 5 then put update_big_event_file to True
# if datetime.now().minute < 5:
#     update_big_event_file = True

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("--------- Running event_accumulator.py -------------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

multiple_files = True

run_jupyter_notebook = False
if run_jupyter_notebook:
    station = "2"
else:
    # Check if the script has an argument
    if len(sys.argv) < 2:
        print("Error: No station provided.")
        print("Usage: python3 script.py <station>")
        sys.exit(1)

    # Get the station argument
    station = sys.argv[1]

print(f"Station: {station}")

if len(sys.argv) == 3:
    user_file_path = sys.argv[2]
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Store the current time at the start. To time the execution
start_execution_time_counting = datetime.now()

# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

# -----------------------------------------------------------------------------
# -------------------------- Variables of execution ---------------------------
# -----------------------------------------------------------------------------

remove_outliers = True

create_plots = False
create_essential_plots = False
create_very_essential_plots = False
save_plots = False
create_pdf = False
force_replacement = True  # Creates a new datafile even if there is already one that looks complete
show_plots = False

# Angular region selection
theta_boundaries = [5, 15, 25]
region_layout = [1, 8, 8, 8]

# Particular analysis -----------------

side_calculations = True # !!!!!!!!!!!!

eff_vs_charge = True
eff_vs_angle_and_pos = True
noise_vs_angle = False
charge_vs_angle = False
polya_fit = True
real_strip_case_study = False
multiplicity_calculations = True
crosstalk_probability = True
n_study_fit = False
topology_plots = False

global_variables = {}
global_variables['execution_time'] = execution_time
global_variables['correct_angle_with_lut'] = correct_angle


print("----------------------------------------------------------------------")
print("--------------------- Starting the directories -----------------------")
print("----------------------------------------------------------------------")

fig_idx = 0
plot_list = []

config_files_directory = os.path.expanduser(f"~/DATAFLOW_v3/CONFIG_FILES/")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
acc_working_directory = os.path.join(working_directory, "LIST_TO_ACC")

csv_path = os.path.join(working_directory, "event_accumulator_metadata.csv")

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
    # "full_acc_events_directory": os.path.join(working_directory, "FULL_ACC_EVENTS_DIRECTORY"),
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

#         # Move the file if it has < 10 or > 300 rows
#         if line_count < 2 or line_count > 10000:
#             shutil.move(file_path, os.path.join(rejected_dir, filename))
#             print(f"Moved: {filename}")


# Move files from RAW to RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

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


# -----------------------------------------------------------------------------
# Functions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def custom_mean(x):
    return x[x != 0].mean() if len(x[x != 0]) > 0 else 0

def custom_std(x):
    return x[x != 0].std() if len(x[x != 0]) > 0 else 0

def round_to_significant_digits(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    return x

# def clean_type_column(x):
#     return str(int(float(x))) if isinstance(x, (float, int, str)) and not pd.isna(x) else x


print("----------------------------------------------------------------------")
print("---------------------------- Main script -----------------------------")
print("----------------------------------------------------------------------")

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
processing_files = os.listdir(base_directories["processing_directory"])
completed_files = os.listdir(base_directories["completed_directory"])

if user_file_selection:
    processing_file_path = user_file_path
    file_name = os.path.basename(user_file_path)
else:
    if last_file_test:
        if unprocessed_files:
            unprocessed_files = sorted(unprocessed_files)
            file_name = unprocessed_files[-1]
            
            unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
            
            print(f"Processing the last file in UNPROCESSED: {unprocessed_file_path}")
            print(f"Moving '{file_name}' to PROCESSING...")
            shutil.move(unprocessed_file_path, processing_file_path)
            print(f"File moved to PROCESSING: {processing_file_path}")

        elif processing_files:
            processing_files = sorted(processing_files)
            file_name = processing_files[-1]
            
            # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
            
            print(f"Processing the last file in PROCESSING: {processing_file_path}")
            error_file_path = os.path.join(base_directories["error_directory"], file_name)
            print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
            shutil.move(processing_file_path, error_file_path)
            processing_file_path = error_file_path
            print(f"File moved to ERROR: {processing_file_path}")

        elif completed_files:
            completed_files = sorted(completed_files)
            file_name = completed_files[-1]
            
            # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
            
            print(f"Reprocessing the last file in COMPLETED: {completed_file_path}")
            print(f"Moving '{completed_file_path}' to PROCESSING...")
            shutil.move(completed_file_path, processing_file_path)
            print(f"File moved to PROCESSING: {processing_file_path}")

        else:
            sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

    else:
        if unprocessed_files:
            print("Shuffling the files in UNPROCESSED...")
            random.shuffle(unprocessed_files)
            for file_name in unprocessed_files:
                unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Moving '{file_name}' to PROCESSING...")
                shutil.move(unprocessed_file_path, processing_file_path)
                print(f"File moved to PROCESSING: {processing_file_path}")
                break

        elif processing_files:
            print("Shuffling the files in PROCESSING...")
            random.shuffle(processing_files)
            for file_name in processing_files:
                # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

                print(f"Processing a file in PROCESSING: {processing_file_path}")
                error_file_path = os.path.join(base_directories["error_directory"], file_name)
                print(f"File '{processing_file_path}' is already in PROCESSING. Moving it temporarily to ERROR for analysis...")
                shutil.move(processing_file_path, error_file_path)
                processing_file_path = error_file_path
                print(f"File moved to ERROR: {processing_file_path}")
                break

        elif completed_files:
            print("Shuffling the files in COMPLETED...")
            random.shuffle(completed_files)
            for file_name in completed_files:
                # unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
                completed_file_path = os.path.join(base_directories["completed_directory"], file_name)
                processing_file_path = os.path.join(base_directories["processing_directory"], file_name)

                print(f"Moving '{file_name}' to PROCESSING...")
                shutil.move(completed_file_path, processing_file_path)
                print(f"File moved to PROCESSING: {processing_file_path}")
                break

        else:
            sys.exit("No files to process in UNPROCESSED, PROCESSING, or COMPLETED.")

# This is for all cases
file_path = processing_file_path

now = time.time()
os.utime(processing_file_path, (now, now))

df = pd.read_csv(file_path, sep=',')
df['Time'] = pd.to_datetime(df['Time'], errors='coerce') # Added errors='coerce' to handle NaT values
print(f"Number of events in the file: {len(df)}")

min_time_original = df['Time'].min()
max_time_original = df['Time'].max()
valid_times = df['Time'].dropna()

# ------------------------------------------------------------------------------------------

# --- MULTIPLE FILES HANDLING ---
if multiple_files:
    

    # Configuration
    cluster_of_files = 7  # Adjust as needed
    time_window = timedelta(hours=10)
    time_tolerance = timedelta(minutes=10)

    # Extract timestamp from filename
    def extract_datetime_from_filename(name):
        try:
            parts = name.split('_')
            dt_str = parts[2] + '_' + parts[3].replace('.txt', '')
            return datetime.strptime(dt_str, "%Y.%m.%d_%H.%M.%S")
        except Exception:
            return None

    reference_filename = os.path.basename(file_path)
    reference_datetime = extract_datetime_from_filename(reference_filename)
    if reference_datetime is None:
        raise ValueError(f"Could not parse datetime from filename: {reference_filename}")

    # Directory of the file
    file_dir = os.path.dirname(file_path)
    all_files = sorted(os.listdir(file_dir))
    
    print("Files in the directory of the datafile:")
    for fname in all_files:
        print(fname)
    
    # Filter files within ±2h of reference time
    time_candidates = []
    for fname in all_files:
        dt = extract_datetime_from_filename(fname)
        if dt and abs(dt - reference_datetime) <= time_window:
            time_candidates.append((dt, fname))
    time_candidates.sort()
    
    print(f"Found {len(time_candidates)} files within the time window of {time_window} around {reference_datetime}.")
    print("Time candidates:")
    for dt, fname in time_candidates:
        print(f"{dt} - {fname}")
    
    # Load the closest files by timestamp
    merged_df = df.copy()
    used = 1  # already using one
    for dt, fname in time_candidates:
        if fname == reference_filename:
            continue
        if used >= cluster_of_files:
            break

        fpath = os.path.join(file_dir, fname)
        try:
            temp_df = pd.read_csv(fpath, sep=',')
            temp_df['Time'] = pd.to_datetime(temp_df['Time'], errors='coerce')
            min_t, max_t = temp_df['Time'].min(), temp_df['Time'].max()
            print(f"Processing file: {fname} with min time {min_t} and max time {max_t}")
            if not (max_t < min_time_original - time_tolerance or min_t > max_time_original + time_tolerance):
                print(f"Merging file: {fname}")
                merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
                # Update time range for rolling inclusion
                min_time_original = min(min_time_original, min_t)
                max_time_original = max(max_time_original, max_t)
                used += 1
        except Exception as e:
            print(f"Skipping file {fname}: {e}")

    df = merged_df
    print(f"Total events after merging: {len(df)}")

# ------------------------------------------------------------------------------------------

# Ensure valid time entries
valid_times = df['Time'].dropna()

if not valid_times.empty:
    min_time_valid = valid_times.min()
    max_time_valid = valid_times.max()
    
    start_time = min_time_original
    end_time = max_time_original
    
    # Check if the overall min time (possibly NaT) differs from the valid min time
    if min_time_original != min_time_valid:
        print("Notice: The minimum value from 'Time' column differs from the smallest valid datetime.")
        print("Original min value (including NaT):", min_time_original)
        print("Valid min value (ignoring NaT):", min_time_valid)
    
    print(f"Time range of dataset: {min_time_valid} to {max_time_valid}")

    first_datetime = min_time_valid
    filename_save_suffix = first_datetime.strftime('%y-%m-%d_%H.%M.%S')
else:
    sys.exit("No valid datetime values found in the 'Time' column. Exiting...")

print("Filename save suffix:", filename_save_suffix)

# full_save_filename = f"full_accumulated_events_{filename_save_suffix}.csv"
# full_save_path = os.path.join(base_directories["full_acc_events_directory"], full_save_filename)

save_filename = f"accumulated_events_{filename_save_suffix}.csv"
save_path = os.path.join(base_directories["acc_events_directory"], save_filename)

save_pdf_filename = f"pdf_{filename_save_suffix}.pdf"
save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)


print("----------------------------------------------------------------------")
print("------------------------ Input file reading --------------------------")
print("----------------------------------------------------------------------")

input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    print("Input configuration file found.")
    exists_input_file = True
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")

if exists_input_file:
    
    # Read and preprocess the input file only once
    input_file["start"] = pd.to_datetime(input_file["start"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = pd.to_datetime(input_file["end"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = input_file["end"].fillna(pd.to_datetime("now"))

    # Filter matching configurations based on start_time and end_time
    matching_confs = input_file[(input_file["start"] <= start_time) & (input_file["end"] >= end_time)]

    if not matching_confs.empty:
        if len(matching_confs) > 1:
            print(f"Warning: Multiple configurations match the date range ({start_time} to {end_time}).")

        # Assign the first matching configuration
        selected_conf = matching_confs.iloc[0]
        print(f"Selected configuration: {selected_conf['conf']}")

        # Extract z_1 to z_4 values from the selected configuration
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])

        # Update dataframe with configuration values
        if len(matching_confs) > 1:
            # Create a dictionary for new columns if multiple configurations match
            new_columns = {
                "over_P1": [],
                "P1-P2": [],
                "P2-P3": [],
                "P3-P4": [],
                "phi_north": []
            }

            # Assign values to new columns based on timestamps in df
            for timestamp in df["Time"]:
                match = input_file[(input_file["start"] <= timestamp) & (input_file["end"] >= timestamp)]
                if not match.empty:
                    selected_conf = match.iloc[0]
                    new_columns["over_P1"].append(selected_conf.get("over_P1", np.nan))
                    new_columns["P1-P2"].append(selected_conf.get("P1-P2", np.nan))
                    new_columns["P2-P3"].append(selected_conf.get("P2-P3", np.nan))
                    new_columns["P3-P4"].append(selected_conf.get("P3-P4", np.nan))
                    new_columns["phi_north"].append(selected_conf.get("phi_north", np.nan))
                else:
                    new_columns["over_P1"].append(np.nan)
                    new_columns["P1-P2"].append(np.nan)
                    new_columns["P2-P3"].append(np.nan)
                    new_columns["P3-P4"].append(np.nan)
                    new_columns["phi_north"].append(0)

            df_new_cols = pd.DataFrame(new_columns)
            df_extended = pd.concat([df, df_new_cols], axis=1)
            df_extended.fillna(method='ffill', inplace=True)
            df = df_extended

        else:
            # Single match, directly apply configuration values to df
            df["over_P1"] = selected_conf.get("over_P1", np.nan)
            df["P1-P2"] = selected_conf.get("P1-P2", np.nan)
            df["P2-P3"] = selected_conf.get("P2-P3", np.nan)
            df["P3-P4"] = selected_conf.get("P3-P4", np.nan)
            df["phi_north"] = selected_conf.get("phi_north", 0)
    else:
        print("Error: No matching configuration found for the given date range.")
        # Assign default values if no match found
        z_positions = np.array([0, 150, 300, 450])  # In mm
        df["over_P1"] = 0
        df["P1-P2"] = 0
        df["P2-P3"] = 0
        df["P3-P4"] = 0
        df["phi_north"] = 0
else:
    print("Error: No input file. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm
    # Assign default values to columns in the dataframe
    df["over_P1"] = 0
    df["P1-P2"] = 0
    df["P2-P3"] = 0
    df["P3-P4"] = 0
    df["phi_north"] = 0

# Every phi_norht that is nan, put 0
df['phi_north'] = df['phi_north'].fillna(0)

# Adjust z_positions and print
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

z1 = z_positions[0]
z2 = z_positions[1]
z3 = z_positions[2]
z4 = z_positions[3]

# Rename every column that contains Q_M... for  Q_P...
for col in df.columns:
    if "Q_M" in col:
        new_col = col.replace("Q_M", "Q_P")
        df.rename(columns={col: new_col}, inplace=True)

show_plots = True

def compute_definitive_tt(row):
    name = ''
    for plane in range(1, 5):
        this_plane = False
        for strip in range(1, 5):
            q_sum_col  = f'Q_P{plane}s{strip}'
            
            if (row[q_sum_col] != 0):
                this_plane = True
        
        if this_plane:
            name += str(plane)
            
    return int(name) if name else 0  # Return 0 if no plane is valid

df["definitive_tt"] = df.apply(compute_definitive_tt, axis=1)
original_df = df.copy()

# Print the head of the df
print("----------------------------------------------------------------------")
print("Dataframe head:")
print(df.head())


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


if side_calculations:
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")
    print("----------------------- Side calculations ----------------------------")
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------

    if eff_vs_charge:
        print("----------------------------------------------------------------------")
        print("------------------- Efficiency respect the charge --------------------")
        print("----------------------------------------------------------------------")

        def compute_angular_efficiencies(df_input, filter_value, bins, bin_centers, tt_combos, unique_tt_per_col, blurring_sigma=2):
            df_filtered = df_input.copy()

            # --- Apply filter to all Q_Pxsj (not Q_P1 etc.) ---
            for col in df_filtered.columns:
                if col.startswith("Q_P") and "s" in col:
                    df_filtered[col] = np.where(df_filtered[col] > filter_value, df_filtered[col], 0)

            # --- Recompute Q_P1..4 after thresholding ---
            for plane in range(1, 5):
                q_sum = np.zeros(len(df_filtered), dtype=float)
                for strip in range(1, 5):
                    col = f"Q_P{plane}s{strip}"
                    if col in df_filtered.columns:
                        q_sum += df_filtered[col].values
                df_filtered[f"Q_P{plane}"] = q_sum

            # --- Compute processed_tt again from filtered Q_Pxsj ---
            df_filtered["processed_tt"] = df_filtered.apply(compute_definitive_tt, axis=1)

            # --- Compute subdetector labels ---
            df_filtered["subdetector_123_tt"] = df_filtered["processed_tt"].map(map_123)
            df_filtered["subdetector_234_tt"] = df_filtered["processed_tt"].map(map_234)
            df_filtered["subdetector_1234_tt"] = df_filtered["processed_tt"]

            # --- Count θ entries ---
            counts_per_tt = {}
            for col, tt_set in unique_tt_per_col.items():
                for tt in tt_set:
                    df_tt = df_filtered[df_filtered[col] == int(tt)]
                    theta_vals = df_tt['theta'].dropna()
                    print(f"[DEBUG] filter={filter_value:.1f}, col={col}, tt={tt}, entries={len(theta_vals)}")
                    if len(theta_vals) < 10:
                        continue
                    counts, _ = np.histogram(theta_vals, bins=bins)
                    counts = gaussian_filter1d(counts, sigma=blurring_sigma, mode='nearest')
                    counts_per_tt[(col, tt)] = counts

            # --- Compute efficiencies ---
            results = []
            for num_tt, den_tt, col, label, color in tt_combos:
                n_num = counts_per_tt.get((col, num_tt), np.zeros(len(bin_centers), dtype=float))
                n_den = counts_per_tt.get((col, den_tt), np.zeros(len(bin_centers), dtype=float))
                with np.errstate(divide='ignore', invalid='ignore'):
                    eff = np.divide(n_num, n_num + n_den)
                    eff[np.isnan(eff)] = 0
                    err = np.sqrt(np.divide(
                        n_num * n_den,
                        (n_num + n_den)**3,
                        out=np.zeros_like(n_num, dtype=float),
                        where=(n_num + n_den) > 0
                    ))

                results.append((eff, err, label, color))
            return results

        # --- Setup parameters ---
        nbins = 6
        right = np.pi / 3
        bins = np.linspace(0, right, nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Subdetector mappings
        map_123 = {1234: 123, 123: 123, 234: 234, 124: 12, 134: 13, 12: 12, 23: 23, 34: 3, 13: 13, 24: 2, 14: 1}
        map_234 = {1234: 234, 123: 23, 234: 234, 124: 24, 134: 34, 12: 2, 23: 23, 34: 34, 13: 3, 24: 24, 14: 4}

        df['subdetector_123_tt'] = df['processed_tt'].map(map_123)
        df['subdetector_234_tt'] = df['processed_tt'].map(map_234)
        df['subdetector_1234_tt'] = df['processed_tt']

        # TT efficiency configurations
        tt_combos = [
            ('1234', '134', 'subdetector_1234_tt', r'3-plane eff$_2$ $= \frac{1234}{134 + 1234}$', 'blue'),
            ('123',  '13',  'subdetector_123_tt',  r'2-plane eff$_2$ $= \frac{123}{13 + 123}$',     'red'),
            ('1234', '124', 'subdetector_1234_tt', r'3-plane eff$_3$ $= \frac{1234}{124 + 1234}$', 'green'),
            ('234',  '24',  'subdetector_234_tt',  r'2-plane eff$_3$ $= \frac{234}{24 + 234}$',     'orange'),
        ]

        # Required TT values
        unique_tt_per_col = {}
        for num, den, col, _, _ in tt_combos:
            unique_tt_per_col.setdefault(col, set()).update([num, den])

        # --- Compute and store efficiency curves ---
        eff_curves_by_combo = {label: [] for _, _, _, label, _ in tt_combos}
        filter_values = np.linspace(0, 8, 5)

        for fval in filter_values:
            results = compute_angular_efficiencies(df.copy(), fval, bins, bin_centers, tt_combos, unique_tt_per_col)
            for (eff, err, label, color) in results:
                eff_curves_by_combo[label].append((fval, eff, err))

        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axs = axs.flatten()

        for idx, (label, curves) in enumerate(eff_curves_by_combo.items()):
            ax = axs[idx]
            for fval, eff, err in curves:
                ax.plot(bin_centers, eff, label=f'Thresh = {fval:.1f}')
                ax.fill_between(bin_centers, eff - err, eff + err, alpha=0.2)

            ax.set_xlim(0, right)
            ax.set_ylim(0.8, 1)
            ax.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
            ax.set_ylabel('Efficiency')
            ax.set_title(label, fontsize=11)
            ax.grid(True)
            ax.legend(fontsize='x-small')

        # Hide unused axes if fewer than 4 curves
        for j in range(len(eff_curves_by_combo), 4):
            fig.delaxes(axs[j])

        fig.suptitle('Angular Efficiency vs. Threshold', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plots:
            filename = f'{fig_idx}_eff_vs_theta_2x2_grid.png'
            fig_idx += 1
            path = os.path.join(base_directories["figure_directory"], filename)
            plot_list.append(path)
            plt.savefig(path)

        if show_plots:
            plt.show()
        plt.close()
        

        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axs = axs.flatten()

        theta0_idx = np.argmin(np.abs(bin_centers - 0))  # Closest bin center to θ = 0

        for idx, (label, curves) in enumerate(eff_curves_by_combo.items()):
            ax = axs[idx]
            x_vals = []
            y_vals = []
            y_errs = []

            for fval, eff, err in curves:
                x_vals.append(fval)
                y_vals.append(eff[theta0_idx])
                y_errs.append(err[theta0_idx])

            ax.fill_between(x_vals, np.array(y_vals) - np.array(y_errs), np.array(y_vals) + np.array(y_errs), alpha=0.3)
            ax.plot(x_vals, y_vals, 'o-', label=f'{label}')
            ax.set_xlabel('Charge Threshold')
            ax.set_ylabel(r'Efficiency at $\theta = 0$')
            ax.set_title(label, fontsize=11)
            ax.grid(True)
            ax.set_ylim(0.8, 1)

        # Hide unused axes if fewer than 4 labels
        for j in range(len(eff_curves_by_combo), 4):
            fig.delaxes(axs[j])

        fig.suptitle(r'Efficiency at $\theta = 0$ vs. Charge Threshold', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plots:
            filename = f'{fig_idx}_eff_theta0_vs_threshold.png'
            fig_idx += 1
            path = os.path.join(base_directories["figure_directory"], filename)
            plot_list.append(path)
            plt.savefig(path)

        if show_plots:
            plt.show()
        plt.close()

        # Print all the column names of df
        print("Columns in the dataframe:")
        for col in df.columns:
            print(f"- {col}")


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------

    if eff_vs_angle_and_pos:
        
        print("----------------------------------------------------------------------")
        print("---------------------- Efficiency respect theta ----------------------")
        print("----------------------------------------------------------------------")

        # Mapping definitions
        map_123 = {
            1234: 123,
            123: 123,
            234: 234,
            124: 12,
            134: 13,
            12: 12,
            23: 23,
            34: 3,
            13: 13,
            24: 2,
            14: 1
        }

        map_234 = {
            1234: 234,
            123: 23,
            234: 234,
            124: 24,
            134: 34,
            12: 2,
            23: 23,
            34: 34,
            13: 3,
            24: 24,
            14: 4
        }


        # Apply mappings to new columns
        df['subdetector_123_tt'] = df['processed_tt'].map(map_123)
        df['subdetector_234_tt'] = df['processed_tt'].map(map_234)
        df['subdetector_1234_tt'] = df['processed_tt']
        
        nbins = 12
        right = np.pi / 3
        blurring = True
        blurring_sigma = 2

        bins = np.linspace(0, right, nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        df_filtered = df.copy()

        print("Calculating angular efficiencies...")

        # TT combinations: (numerator, denominator, column_name, label, color)
        tt_combos = [
            ('1234', '134', 'subdetector_1234_tt', r'3-plane-eff_2', 'blue'),
            ('123',  '13',  'subdetector_123_tt',  r'2-plane-eff_2', 'red'),
            ('1234', '124', 'subdetector_1234_tt', r'3-plane-eff_3', 'green'),
            ('234',  '24',  'subdetector_234_tt',  r'2-plane-eff_3', 'orange'),
        ]

        # Build unified set of all TT values needed, grouped by column
        unique_tt_per_col = {}
        for num, den, col, _, _ in tt_combos:
            unique_tt_per_col.setdefault(col, set()).update([num, den])

        # Compute histograms for each TT value within each subdetector column
        counts_per_tt = {}
        for col, tt_set in unique_tt_per_col.items():
            for tt in tt_set:
                df_tt = df_filtered[df_filtered[col] == int(tt)]
                theta_vals = df_tt['theta'].dropna()
                if len(theta_vals) < 10:
                    continue
                counts, _ = np.histogram(theta_vals, bins=bins)
                if blurring:
                    counts = gaussian_filter1d(counts, sigma=blurring_sigma, mode='nearest')
                counts_per_tt[(col, tt)] = counts

        # Compute efficiencies
        eff_results = []
        for num_tt, den_tt, col, label, color in tt_combos:
            n_num = counts_per_tt.get((col, num_tt), np.zeros(len(bin_centers)))
            n_den = counts_per_tt.get((col, den_tt), np.zeros(len(bin_centers)))
            with np.errstate(divide='ignore', invalid='ignore'):
                eff = np.divide(n_num, n_num + n_den)
                eff[np.isnan(eff)] = 0
                err = np.sqrt(np.divide(n_num * n_den, (n_num + n_den)**3,
                                        out=np.zeros_like(n_num, dtype=float),
                                        where=(n_num + n_den) > 0))
            eff_results.append((eff, err, label, color))
            
            if "2-plane-eff_2" in label:
                eff_2_sd = eff
            elif "2-plane-eff_3" in label:
                eff_3_sd = eff
        
        
        print("Efficiency calculations complete.")
        
        # Plot raw angular distributions
        if create_plots or create_essential_plots:
            fig_counts, ax_counts = plt.subplots(figsize=(7, 5))
            colors = plt.cm.tab10.colors
            plotted_labels = set()

            for i, (col, tt_set) in enumerate(unique_tt_per_col.items()):
                for j, tt in enumerate(sorted(tt_set)):
                    counts = counts_per_tt.get((col, tt), None)
                    if counts is not None and tt not in plotted_labels:
                        ax_counts.hist(bin_centers, bins=bins, weights=counts,
                                    histtype='step', linewidth=1,
                                    color=colors[(i + j) % len(colors)], label=str(tt))
                        plotted_labels.add(tt)

            ax_counts.set_xlim(0, right)
            ax_counts.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
            ax_counts.set_ylabel('Counts')
            ax_counts.set_title(r'Zoomed $\theta_{\mathrm{new}}$ Distributions')
            ax_counts.grid(True)
            ax_counts.legend(title='processed_tt', fontsize='small')
            plt.tight_layout()

            if save_plots:
                filename = f'{fig_idx}_theta_zoom_counts_all.png'
                fig_idx += 1
                path = os.path.join(base_directories["figure_directory"], filename)
                plot_list.append(path)
                plt.savefig(path)

            if show_plots:
                plt.show()
            plt.close()

        # Plot efficiencies
        if create_plots or create_essential_plots:
            fig_eff, ax_eff = plt.subplots(figsize=(7, 5))

            for eff, err, label, color in eff_results:
                ax_eff.plot(bin_centers, eff, label=label, color=color)
                ax_eff.fill_between(bin_centers, eff - err, eff + err, alpha=0.3, color=color)

            ax_eff.set_xlim(0, right)
            ax_eff.set_ylim(0.5, 1)
            ax_eff.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
            ax_eff.set_ylabel('Efficiency')
            ax_eff.set_title('Angular Efficiency Estimates')
            ax_eff.grid(True)
            ax_eff.legend(fontsize='small')
            plt.tight_layout()

            if save_plots:
                filename = f'{fig_idx}_theta_efficiencies_all.png'
                fig_idx += 1
                path = os.path.join(base_directories["figure_directory"], filename)
                plot_list.append(path)
                plt.savefig(path)

            if show_plots:
                plt.show()
            plt.close()


        
        fit_params_list = []
        
        # Define convex power-law model
        def power_law(theta, a, n, eps0):
            return a * theta**n + eps0

        # ──────────────────────────────────────────────────────────────────
        # Model
        # ──────────────────────────────────────────────────────────────────
        def power_law(theta: np.ndarray, a: float, n: float, eps0: float) -> np.ndarray:
            return eps0 * (1.0 - a * np.abs(theta) ** n)


        # ──────────────────────────────────────────────────────────────────
        # 1. Fitting only
        # ──────────────────────────────────────────────────────────────────
        def fit_efficiencies(
            eff_results:  List[Tuple[np.ndarray, np.ndarray, str, str]],
            bin_centers:  np.ndarray,
        ) -> Tuple[pd.DataFrame, List[np.ndarray]]:
            """
            Returns
            -------
            df_fits      : DataFrame with columns [label, color, a, n, eps0]
            fit_curves   : list evaluated on *bin_centers*, aligned with eff_results
            """
            fit_params_list: List[Dict[str, object]] = []
            fit_curves:      List[np.ndarray]        = []

            for eff, err, label, color in eff_results:
                # mask out extreme/invalid efficiency points
                mask = (eff >= 0.5) & (eff <= 1.01)
                theta_fit      = bin_centers[mask]
                eff_fit_data   = eff[mask]

                try:
                    popt, _ = curve_fit(
                        power_law,
                        theta_fit,
                        eff_fit_data,
                        p0=[1.0, 2.0, 0.7],
                        maxfev=10000,
                    )
                    fit_params_list.append(
                        dict(label=label, color=color, a=popt[0], n=popt[1], eps0=popt[2])
                    )
                    fit_curves.append(power_law(bin_centers, *popt))
                except RuntimeError:
                    print(f"[WARN] Fit failed for: {label}")
                    # keep placeholders so indexing stays aligned
                    fit_params_list.append(dict(label=label, color=color, a=np.nan, n=np.nan, eps0=np.nan))
                    fit_curves.append(np.full_like(bin_centers, np.nan))

            df_fits = pd.DataFrame(fit_params_list)
            return df_fits, fit_curves


        # ──────────────────────────────────────────────────────────────────
        # 2. Optional plotting
        # ──────────────────────────────────────────────────────────────────
        def plot_efficiencies_with_fits(
            eff_results:       List[Tuple[np.ndarray, np.ndarray, str, str]],
            bin_centers:       np.ndarray,
            fit_curves:        List[np.ndarray],
            right:             float,
            base_directories:  Dict[str, str],
            fig_idx:           int,
            save_plots:        bool,
            show_plots:        bool,
        ) -> int:
            fig, ax = plt.subplots(figsize=(7, 5))

            for (eff, err, label, color), fit in zip(eff_results, fit_curves):
                ax.plot(bin_centers, eff, label=label, color=color)
                ax.fill_between(bin_centers, eff - err, eff + err, alpha=0.3, color=color)

                if not np.isnan(fit).all():
                    ax.plot(bin_centers, fit, '--', color=color, linewidth=1.2)

            ax.set_xlim(0, right)
            ax.set_ylim(0.85, 1.05)
            ax.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
            ax.set_ylabel('Efficiency')
            ax.set_title('Angular Efficiency Estimates with Convex Fits')
            ax.grid(True)
            ax.legend(fontsize='small')
            plt.tight_layout()

            if save_plots:
                filename = f'{fig_idx}_theta_efficiencies_all_with_fit.png'
                path = os.path.join(base_directories["figure_directory"], filename)
                plt.savefig(path)
                fig_idx += 1
            if show_plots:
                plt.show()
            plt.close(fig)
            return fig_idx


        # ──────────────────────────────────────────────────────────────────
        # 3. Driver (replace flags with your existing variables)
        # ──────────────────────────────────────────────────────────────────
        df_fits, fit_curves = fit_efficiencies(eff_results, bin_centers)
        print(df_fits)                        # always available

        # store fitted parameters back into df as before
        for _, row in df_fits.iterrows():
            if "eff_2" in row["label"] and "3-plane" in row["label"]:
                df["P2_3fold_a"]   = row["a"]
                df["P2_3fold_n"]   = row["n"]
                df["P2_3fold_eps0"] = row["eps0"]
            elif "eff_2" in row["label"] and "2-plane" in row["label"]:
                df["P2_2fold_a"]   = row["a"]
                df["P2_2fold_n"]   = row["n"]
                df["P2_2fold_eps0"] = row["eps0"]
            elif "eff_3" in row["label"] and "3-plane" in row["label"]:
                df["P3_3fold_a"]   = row["a"]
                df["P3_3fold_n"]   = row["n"]
                df["P3_3fold_eps0"] = row["eps0"]
            elif "eff_3" in row["label"] and "2-plane" in row["label"]:
                df["P4_2fold_a"]   = row["a"]
                df["P4_2fold_n"]   = row["n"]
                df["P4_2fold_eps0"] = row["eps0"]

        # plotting only if requested
        if create_plots or create_essential_plots:
            fig_idx = plot_efficiencies_with_fits(
                eff_results, bin_centers, fit_curves,
                right,
                base_directories,
                fig_idx,
                save_plots,
                show_plots,
            )

            
            
            print("----------------------------------------------------------------------")
            print("--------------------- Efficiency respect (x, y) ----------------------")
            print("----------------------------------------------------------------------")
            
            nbins_x = 10
            nbins_y = 10
            len_range = (-150, 150)
            blurring_sigma = 0.05  # Smaller for 2D smoothing

            blurring = False

            bins_theta = np.linspace(*len_range, nbins_x + 1)
            bins_phi = np.linspace(*len_range, nbins_y + 1)
            theta_centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])
            phi_centers = 0.5 * (bins_phi[:-1] + bins_phi[1:])

            # Replace 1D histograms with 2D
            counts_per_tt = {}
            for col, tt_set in unique_tt_per_col.items():
                for tt in tt_set:
                    df_tt = df_filtered[df_filtered[col] == int(tt)]
                    theta_vals = df_tt['x'].dropna()
                    phi_vals = df_tt['y'].dropna()
                    if len(theta_vals) < 10:
                        continue
                    counts, _, _ = np.histogram2d(theta_vals, phi_vals, bins=[bins_theta, bins_phi])
                    if blurring:
                        counts = gaussian_filter1d(counts, sigma=blurring_sigma, axis=0, mode='nearest')
                        counts = gaussian_filter1d(counts, sigma=blurring_sigma, axis=1, mode='nearest')
                    counts_per_tt[(col, tt)] = counts

            # Compute 2D efficiency maps
            eff_results = []
            for num_tt, den_tt, col, label, color in tt_combos:
                n_num = counts_per_tt.get((col, num_tt), np.zeros((nbins_x, nbins_y)))
                n_den = counts_per_tt.get((col, den_tt), np.zeros((nbins_x, nbins_y)))
                with np.errstate(divide='ignore', invalid='ignore'):
                    eff = np.divide(n_num, n_num + n_den)
                    eff[np.isnan(eff)] = 0
                    err = np.sqrt(np.divide(n_num * n_den, (n_num + n_den)**3,
                                            out=np.zeros_like(n_num, dtype=float),
                                            where=(n_num + n_den) > 0))
                eff_results.append((eff, err, label, color))

            # Plot 2D efficiency maps
            if create_plots or create_essential_plots:
                for i, (eff, err, label, color) in enumerate(eff_results):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(eff.T, origin='lower', aspect='auto',
                                extent=[bins_theta[0], bins_theta[-1], bins_phi[0], bins_phi[-1]],
                                interpolation='nearest', cmap='viridis', vmin=0, vmax=1.0)
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Efficiency')
                    ax.set_xlabel(r'X [mm]')
                    ax.set_ylabel(r'Y [mm]')
                    ax.set_title(label)
                    plt.tight_layout()

                    if save_plots:
                        filename = f"{fig_idx}_2D_eff_pos_{label.replace(' ', '_')}.png"
                        fig_idx += 1
                        path = os.path.join(base_directories["figure_directory"], filename)
                        plot_list.append(path)
                        plt.savefig(path)

                    if show_plots:
                        plt.show()
                    plt.close()
            
            
            print("----------------------------------------------------------------------")
            print("------------------ Efficiency respect (theta, phi) -------------------")
            print("----------------------------------------------------------------------")
            
            nbins_theta = 9
            nbins_phi = 8
            right_theta = np.pi / 3
            phi_range = (-np.pi, np.pi)
            blurring_sigma = 0.05  # Smaller for 2D smoothing

            blurring = False

            bins_theta = np.linspace(0, right_theta, nbins_theta + 1)
            bins_phi = np.linspace(*phi_range, nbins_phi + 1)
            theta_centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])
            phi_centers = 0.5 * (bins_phi[:-1] + bins_phi[1:])

            # Replace 1D histograms with 2D
            counts_per_tt = {}
            for col, tt_set in unique_tt_per_col.items():
                for tt in tt_set:
                    df_tt = df_filtered[df_filtered[col] == int(tt)]
                    theta_vals = df_tt['theta'].dropna()
                    phi_vals = df_tt['phi'].dropna()
                    if len(theta_vals) < 10:
                        continue
                    counts, _, _ = np.histogram2d(theta_vals, phi_vals, bins=[bins_theta, bins_phi])
                    if blurring:
                        counts = gaussian_filter1d(counts, sigma=blurring_sigma, axis=0, mode='nearest')
                        counts = gaussian_filter1d(counts, sigma=blurring_sigma, axis=1, mode='nearest')
                    counts_per_tt[(col, tt)] = counts

            # Compute 2D efficiency maps
            eff_results = []
            for num_tt, den_tt, col, label, color in tt_combos:
                n_num = counts_per_tt.get((col, num_tt), np.zeros((nbins_theta, nbins_phi)))
                n_den = counts_per_tt.get((col, den_tt), np.zeros((nbins_theta, nbins_phi)))
                with np.errstate(divide='ignore', invalid='ignore'):
                    eff = np.divide(n_num, n_num + n_den)
                    eff[np.isnan(eff)] = 0
                    err = np.sqrt(np.divide(n_num * n_den, (n_num + n_den)**3,
                                            out=np.zeros_like(n_num, dtype=float),
                                            where=(n_num + n_den) > 0))
                eff_results.append((eff, err, label, color))

            # Plot 2D efficiency maps
            if create_plots or create_essential_plots:
                for i, (eff, err, label, color) in enumerate(eff_results):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(eff.T, origin='lower', aspect='auto',
                                extent=[bins_theta[0], bins_theta[-1], bins_phi[0], bins_phi[-1]],
                                interpolation='nearest', cmap='viridis', vmin=0.5, vmax=1.0)
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Efficiency')
                    ax.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
                    ax.set_ylabel(r'$\phi_{\mathrm{new}}$ [rad]')
                    ax.set_title(label)
                    plt.tight_layout()

                    if save_plots:
                        filename = f"{fig_idx}_2D_eff_{label.replace(' ', '_')}.png"
                        fig_idx += 1
                        path = os.path.join(base_directories["figure_directory"], filename)
                        plot_list.append(path)
                        plt.savefig(path)

                    if show_plots:
                        plt.show()
                    plt.close()


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if noise_vs_angle:
        
        print("----------------------------------------------------------------------")
        print("----------------------- Noise respect the angle ----------------------")
        print("----------------------------------------------------------------------")

        # Mapping definitions
        map_123 = {
            1234: 123,
            123: 123,
            234: 234,
            124: 12,
            134: 13,
            12: 12,
            23: 23,
            34: 3,
            13: 13,
            24: 2,
            14: 1
        }

        map_234 = {
            1234: 234,
            123: 23,
            234: 234,
            124: 24,
            134: 34,
            12: 2,
            23: 23,
            34: 34,
            13: 3,
            24: 24,
            14: 4
        }


        # Apply mappings to new columns
        df['subdetector_123_tt'] = df['processed_tt'].map(map_123)
        df['subdetector_234_tt'] = df['processed_tt'].map(map_234)
        df['subdetector_1234_tt'] = df['processed_tt']

        
        
        # Explained / noise percentage
        def compute_noise_percentages(est, measured):
            with np.errstate(divide='ignore', invalid='ignore'):
                explained = 100 * est / measured
                noise = 100 - explained
            for arr in (explained, noise):
                arr[np.isnan(arr) | np.isinf(arr)] = 0
            explained = np.clip(explained, 0, 100)
            noise = np.clip(noise, 0, 100)
            return explained, noise
        
        right = np.pi / 3
        blurring = False
        blurring_sigma = 2

        bins = np.linspace(0, right, nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        df_filtered = df.copy()
        
        # Topologies definition
        all_topologies = ['12', '23', '34', '123', '234', '1234', '24', '13', '134', '124', '14']
        # measured_topologies = ['24', '13', '24_sd', '13_sd', '134', '124', '14']
        measured_topologies = ['24', '13', '134', '124', '14']
        measured_topologies = list(set(measured_topologies) & set(df_filtered['original_tt'].unique().astype(str)))
        inferred_topologies = ['12', '23', '34', '123', '234', '1234']
        
        print(f"Measured topologies: {measured_topologies}")
        print(f"Inferred topologies: {inferred_topologies}")
        
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        
        # ---------------------- THETA DEFINITIONS ----------------------
        theta_12   = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 12, 'theta']
        theta_23   = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 23, 'theta']
        theta_34   = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 34, 'theta']
        theta_123  = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 123, 'theta']
        theta_234  = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 234, 'theta']
        theta_1234 = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 1234, 'theta']
        theta_14   = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 14, 'theta']
        theta_124  = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 124, 'theta']
        theta_134  = df_filtered.loc[df_filtered['subdetector_1234_tt'] == 134, 'theta']

        theta_12_sd_123   = df_filtered.loc[df_filtered['subdetector_123_tt'] == 12, 'theta']
        theta_23_sd_123   = df_filtered.loc[df_filtered['subdetector_123_tt'] == 23, 'theta']
        theta_123_sd_123  = df_filtered.loc[df_filtered['subdetector_123_tt'] == 123, 'theta']
        theta_13_sd_123   = df_filtered.loc[df_filtered['subdetector_123_tt'] == 13, 'theta']

        theta_23_sd_234   = df_filtered.loc[df_filtered['subdetector_234_tt'] == 23, 'theta']
        theta_34_sd_234   = df_filtered.loc[df_filtered['subdetector_234_tt'] == 34, 'theta']
        theta_234_sd_234  = df_filtered.loc[df_filtered['subdetector_234_tt'] == 234, 'theta']
        theta_24_sd_234   = df_filtered.loc[df_filtered['subdetector_234_tt'] == 24, 'theta']

        theta_13  = df_filtered.loc[df_filtered['processed_tt'] == 13,  'theta']
        theta_24  = df_filtered.loc[df_filtered['processed_tt'] == 24,  'theta']

        # ---------------------- COUNTS COMPUTATION ----------------------
        counts_12, _   = np.histogram(theta_12, bins=bins)
        counts_23, _   = np.histogram(theta_23, bins=bins)
        counts_34, _   = np.histogram(theta_34, bins=bins)
        counts_123, _  = np.histogram(theta_123, bins=bins)
        counts_234, _  = np.histogram(theta_234, bins=bins)
        counts_1234, _ = np.histogram(theta_1234, bins=bins)
        counts_14, _   = np.histogram(theta_14, bins=bins)
        counts_124, _  = np.histogram(theta_124, bins=bins)
        counts_134, _  = np.histogram(theta_134, bins=bins)

        counts_13, _   = np.histogram(theta_13, bins=bins)

        counts_12_sd_123, _   = np.histogram(theta_12_sd_123, bins=bins)
        counts_23_sd_123, _   = np.histogram(theta_23_sd_123, bins=bins)
        counts_123_sd_123, _  = np.histogram(theta_123_sd_123, bins=bins)
        counts_123_sd_123, _  = np.histogram(theta_123_sd_123, bins=bins)  # DUPLICATE
        counts_13_sd_123, _   = np.histogram(theta_13_sd_123, bins=bins)

        counts_23_sd_234, _   = np.histogram(theta_23_sd_234, bins=bins)
        counts_34_sd_234, _   = np.histogram(theta_34_sd_234, bins=bins)
        counts_234_sd_234, _  = np.histogram(theta_234_sd_234, bins=bins)
        counts_234_sd_234, _  = np.histogram(theta_234_sd_234, bins=bins)  # DUPLICATE
        counts_24_sd_234, _   = np.histogram(theta_24_sd_234, bins=bins)

        counts_24, _  = np.histogram(theta_24, bins=bins)

        # ---------------------- GAUSSIAN BLURRING ----------------------
        if blurring:
            counts_12 = gaussian_filter1d(counts_12, sigma=blurring_sigma, mode='nearest')
            counts_23 = gaussian_filter1d(counts_23, sigma=blurring_sigma, mode='nearest')
            counts_34 = gaussian_filter1d(counts_34, sigma=blurring_sigma, mode='nearest')
            counts_123 = gaussian_filter1d(counts_123, sigma=blurring_sigma, mode='nearest')
            counts_234 = gaussian_filter1d(counts_234, sigma=blurring_sigma, mode='nearest')
            counts_1234 = gaussian_filter1d(counts_1234, sigma=blurring_sigma, mode='nearest')
            counts_14 = gaussian_filter1d(counts_14, sigma=blurring_sigma, mode='nearest')
            counts_124 = gaussian_filter1d(counts_124, sigma=blurring_sigma, mode='nearest')
            counts_134 = gaussian_filter1d(counts_134, sigma=blurring_sigma, mode='nearest')

            counts_13 = gaussian_filter1d(counts_13, sigma=blurring_sigma, mode='nearest')

            counts_12_sd_123 = gaussian_filter1d(counts_12_sd_123, sigma=blurring_sigma, mode='nearest')
            counts_23_sd_123 = gaussian_filter1d(counts_23_sd_123, sigma=blurring_sigma, mode='nearest')
            counts_123_sd_123 = gaussian_filter1d(counts_123_sd_123, sigma=blurring_sigma, mode='nearest')
            counts_13_sd_123 = gaussian_filter1d(counts_13_sd_123, sigma=blurring_sigma, mode='nearest')

            counts_23_sd_234 = gaussian_filter1d(counts_23_sd_234, sigma=blurring_sigma, mode='nearest')
            counts_34_sd_234 = gaussian_filter1d(counts_34_sd_234, sigma=blurring_sigma, mode='nearest')
            counts_234_sd_234 = gaussian_filter1d(counts_234_sd_234, sigma=blurring_sigma, mode='nearest')
            counts_24_sd_234 = gaussian_filter1d(counts_24_sd_234, sigma=blurring_sigma, mode='nearest')

            counts_24 = gaussian_filter1d(counts_24, sigma=blurring_sigma, mode='nearest')

        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        
        # --- Compute per-bin efficiencies for plane 2 and 3 -------------------------
        three_plane_eff = False
        if three_plane_eff:
            eff_2 = 1 - counts_134 / np.where(counts_1234 == 0, 1, counts_1234)
            eff_3 = 1 - counts_124 / np.where(counts_1234 == 0, 1, counts_1234)
        else:
            eff_2 = eff_2_sd
            eff_3 = eff_3_sd
        
        comp_eff_2  = (1 - eff_2)
        comp_eff_3  = (1 - eff_3)
        comp_eff_23 = comp_eff_2 * comp_eff_3
        
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        
        # --- CASE: 124 (miss plane 3) ------------------------------------------------
        est_124    = counts_1234 * comp_eff_3
        explained_124, noise_124 = compute_noise_percentages(est_124, counts_124)
        
        # --- CASE: 134 (miss plane 2) ------------------------------------------------
        est_134    = counts_1234 * comp_eff_2
        explained_134, noise_134 = compute_noise_percentages(est_134, counts_134)

        # --- CASE: 14 (miss both planes 2 & 3) ---------------------------------------
        est_14    = counts_1234 * comp_eff_23
        explained_14, noise_14 = compute_noise_percentages(est_14, counts_14)

        # subdetector_123_tt ----------------------------------------------------------------
        est_13_sd_123 = counts_123_sd_123 * comp_eff_2
        explained_13_sd_123, noise_13_sd_123 = compute_noise_percentages(est_13_sd_123, counts_13_sd_123)
        
        # subdetector_234_tt ----------------------------------------------------------------
        est_24_sd_234 = counts_234_sd_234 * comp_eff_3
        explained_24_sd_234, noise_24_sd_234 = compute_noise_percentages(est_24_sd_234, counts_24_sd_234)
        
        # three plane-two plane 123 ---------------------------------------------------------
        est_13 = counts_123 * comp_eff_2
        explained_13, noise_13 = compute_noise_percentages(est_13, counts_13)

        # three plane-two plane 234 ---------------------------------------------------------
        est_24 = counts_234 * comp_eff_3
        explained_24, noise_24 = compute_noise_percentages(est_24, counts_24)
        
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        # -----------------------------------------------------------------------------
        
        # --- PLOTS: Efficiency and residual noise summary ----------------------------
        plt.figure(figsize=(8, 5))
        plt.plot(bin_centers, eff_2, 'o-', label='Eff. 2')
        plt.plot(bin_centers, eff_3, 's-', label='Eff. 3')
        plt.plot(bin_centers, 1 - comp_eff_23, color='purple', label='Miss 2 & 3 (complementary eff)')
        plt.xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        plt.ylabel('Efficiency')
        plt.title('Efficiencies of planes 2 and 3')
        plt.grid(True)
        plt.ylim(0.8, 1.01)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        cmap = get_cmap('tab10')
        colors = {}  # ensure this is a dict

        for i, topo in enumerate(measured_topologies):
            if topo not in colors:
                colors[topo] = cmap(i % 10)

        # Initialize figure and axes
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 15), sharex=True)

        # Panel 1: Residual noise
        for topo in measured_topologies:
            ax1.plot(bin_centers, globals()[f'noise_{topo}'], color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Residual noise [%]')
        ax1.set_title('Residual Noise per Topology')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 100)

        # Panel 2: Measured counts
        for topo in measured_topologies:
            ax2.plot(bin_centers, globals()[f'counts_{topo}'], color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel('Measured counts')
        ax2.set_title('Measured Counts per Topology')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        # Panel 3: Noisy counts (noise % × counts)
        for topo in measured_topologies:
            noise = globals()[f'noise_{topo}']
            counts = globals()[f'counts_{topo}']
            ax3.plot(bin_centers, (noise / 100) * counts, color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        ax3.set_ylabel('Noisy counts')
        ax3.set_title('Noisy Counts per Topology')
        ax3.grid(True)
        ax3.legend(loc='upper right')

        # Panel 4: Corrected counts ((1 - noise%) × counts)
        for topo in measured_topologies:
            noise = globals()[f'noise_{topo}']
            counts = globals()[f'counts_{topo}']
            ax4.plot(bin_centers, (1 - noise / 100) * counts, color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax4.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        ax4.set_ylabel('Corrected counts')
        ax4.set_title('Noise-Corrected Counts per Topology')
        ax4.grid(True)
        ax4.legend(loc='upper right')

        plt.tight_layout()
        figure_name = f"residual_noise_measured_TTs"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
        
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------

        # Step 1: Compute raw noise counts from measured two-plane noise %
        k13  = (noise_13  / 100) * counts_13   # ~ λ1·λ3
        k14  = (noise_14  / 100) * counts_14   # ~ λ1·λ4
        k24  = (noise_24  / 100) * counts_24   # ~ λ2·λ4

        # Step 2: Infer missing pairwise noise products (Poisson model)
        with np.errstate(divide='ignore', invalid='ignore'):
            λ1_squared = np.where(k24 > 0, (k14 * k13) / k24, 0.0)
            λ1 = np.sqrt(np.clip(λ1_squared, a_min=0, a_max=None))
            λ2 = np.where(λ1 > 0, k24 / (λ1 * (k14 / λ1)), 0.0)  # from λ4 = k14 / λ1
            λ3 = np.where(λ1 > 0, k13 / λ1, 0.0)
            λ4 = np.where(λ1 > 0, k14 / λ1, 0.0)
        
        norm_factor = 1/10
        
        # Step 3: Estimate new noise contributions
        with np.errstate(divide='ignore', invalid='ignore'):
            noise_counts_12   = λ1 * λ2
            noise_counts_23   = λ2 * λ3
            noise_counts_34   = λ3 * λ4
            noise_counts_123  = norm_factor * λ1 * λ2 * λ3
            noise_counts_234  = norm_factor * λ2 * λ3 * λ4
            noise_counts_1234 = norm_factor**2 * λ1 * λ2 * λ3 * λ4

        # Step 4: Convert to relative noise [%]
        def safe_percent(numerator, denominator):
            return 100 * np.where(denominator > 0, numerator / denominator, 0.0)

        noise_12   = safe_percent(noise_counts_12,   counts_12)
        noise_23   = safe_percent(noise_counts_23,   counts_23)
        noise_34   = safe_percent(noise_counts_34,   counts_34)
        noise_123  = safe_percent(noise_counts_123,  counts_123)
        noise_234  = safe_percent(noise_counts_234,  counts_234)
        noise_1234 = safe_percent(noise_counts_1234, counts_1234)
        
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        
        
        cmap = get_cmap('tab10')

        for i, topo in enumerate(inferred_topologies):
            if topo not in colors:
                colors[topo] = cmap(i % 10)
        
        # Define placeholder variables
        for topo in inferred_topologies:
            try:
                globals()[f'noise_{topo}']
            except KeyError:
                globals()[f'noise_{topo}'] = np.zeros_like(bin_centers)
                colors[topo] = 'gray'  # Or assign a distinct color per topology if desired
        
        # Initialize figure and axes
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 15), sharex=True)

        # Panel 1: Residual noise
        for topo in inferred_topologies:
            ax1.plot(bin_centers, globals()[f'noise_{topo}'], color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Residual noise [%]')
        ax1.set_title('Residual Noise per Topology')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 100)

        # Panel 2: Measured counts
        for topo in inferred_topologies:
            ax2.plot(bin_centers, globals()[f'counts_{topo}'], color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel('Measured counts')
        ax2.set_title('Measured Counts per Topology')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        # Panel 3: Noisy counts (noise % × counts)
        for topo in inferred_topologies:
            noise = globals()[f'noise_{topo}']
            counts = globals()[f'counts_{topo}']
            ax3.plot(bin_centers, (noise / 100) * counts, color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        ax3.set_ylabel('Noisy counts')
        ax3.set_title('Noisy Counts per Topology')
        ax3.grid(True)
        ax3.legend(loc='upper right')

        # Panel 4: Corrected counts ((1 - noise%) × counts)
        for topo in inferred_topologies:
            noise = globals()[f'noise_{topo}']
            counts = globals()[f'counts_{topo}']
            ax4.plot(bin_centers, (1 - noise / 100) * counts, color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax4.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        ax4.set_ylabel('Corrected counts')
        ax4.set_title('Noise-Corrected Counts per Topology')
        ax4.grid(True)
        ax4.legend(loc='upper right')

        plt.tight_layout()
        figure_name = f"residual_noise_inferred_TTs"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
        
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        
        for i, topo in enumerate(all_topologies):
            if topo not in colors:
                colors[topo] = cmap(i % 10)
        
        # Initialize figure and axes
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(9, 15), sharex=True)

        # Panel 1: Residual noise
        for topo in all_topologies:
            ax1.plot(bin_centers, globals()[f'noise_{topo}'], color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Residual noise [%]')
        ax1.set_title('Residual Noise per Topology')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 100)

        # Panel 2: Measured counts
        for topo in all_topologies:
            ax2.plot(bin_centers, globals()[f'counts_{topo}'], color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_ylabel('Measured counts')
        ax2.set_title('Measured Counts per Topology')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        # Panel 3: Noisy counts (noise % × counts)
        for topo in all_topologies:
            noise = globals()[f'noise_{topo}']
            counts = globals()[f'counts_{topo}']
            ax3.plot(bin_centers, (noise / 100) * counts, color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        ax3.set_ylabel('Noisy counts')
        ax3.set_title('Noisy Counts per Topology')
        ax3.grid(True)
        ax3.legend(loc='upper right')

        # Panel 4: Corrected counts ((1 - noise%) × counts)
        for topo in all_topologies:
            noise = globals()[f'noise_{topo}']
            counts = globals()[f'counts_{topo}']
            ax4.plot(bin_centers, (1 - noise / 100) * counts, color=colors[topo], label=f'{topo.replace("_sd", " (subdetector)")}')
        ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax4.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
        ax4.set_ylabel('Corrected counts')
        ax4.set_title('Noise-Corrected Counts per Topology')
        ax4.grid(True)
        ax4.legend(loc='upper right')

        plt.tight_layout()
        figure_name = f"residual_noise_all_TTs"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
        
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------

        noise_2d = True

        if noise_2d:

            # Reuse user-defined binning
            nbins_theta = 6
            nbins_phi = 6
            right_theta = np.pi / 3
            phi_range = (-np.pi, np.pi)

            bins_theta = np.linspace(0, right_theta, nbins_theta + 1)
            bins_phi = np.linspace(*phi_range, nbins_phi + 1)
            theta_centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])
            phi_centers = 0.5 * (bins_phi[:-1] + bins_phi[1:])

            # Get theta, phi from df_filtered for each topology
            def get_2d_counts(df, topo_col, topo_val):
                df_topo = df[df[topo_col] == topo_val]
                theta_vals = df_topo["theta"].dropna()
                phi_vals = df_topo["phi"].dropna()
                counts, _, _ = np.histogram2d(theta_vals, phi_vals, bins=[bins_theta, bins_phi])
                return counts

            # Populate counts from real data
            counts_1234 = get_2d_counts(df_filtered, 'subdetector_1234_tt', 1234)
            counts_124 = get_2d_counts(df_filtered, 'subdetector_1234_tt', 124)
            counts_134 = get_2d_counts(df_filtered, 'subdetector_1234_tt', 134)
            counts_14 = get_2d_counts(df_filtered, 'subdetector_1234_tt', 14)

            # Estimate efficiencies eff_2, eff_3 from subdetector_234_tt and subdetector_123_tt
            counts_234 = get_2d_counts(df_filtered, 'subdetector_234_tt', 234)
            counts_24_sd_234 = get_2d_counts(df_filtered, 'subdetector_234_tt', 24)
            eff_3 = 1 - np.divide(counts_24_sd_234, counts_234, out=np.zeros_like(counts_234, dtype=float), where=counts_234 > 0)

            counts_123 = get_2d_counts(df_filtered, 'subdetector_123_tt', 123)
            counts_13_sd_123 = get_2d_counts(df_filtered, 'subdetector_123_tt', 13)
            eff_2 = 1 - np.divide(counts_13_sd_123, counts_123, out=np.zeros_like(counts_123, dtype=float), where=counts_123 > 0)

            # Compute complementary efficiencies
            comp_eff_2 = 1.0 - eff_2
            comp_eff_3 = 1.0 - eff_3
            comp_eff_23 = comp_eff_2 * comp_eff_3

            # Estimate counts due to inefficiency
            est_124 = counts_1234 * comp_eff_3
            est_134 = counts_1234 * comp_eff_2
            est_14 = counts_1234 * comp_eff_23

            # Compute noise percentage
            def compute_noise_percentages(est, measured):
                with np.errstate(divide='ignore', invalid='ignore'):
                    explained = 100 * est / measured
                    noise = 100 - explained
                for arr in (explained, noise):
                    arr[np.isnan(arr) | np.isinf(arr)] = 0
                explained = np.clip(explained, 0, 100)
                noise = np.clip(noise, 0, 100)
                return explained, noise

            explained_124, noise_124 = compute_noise_percentages(est_124, counts_124)
            explained_134, noise_134 = compute_noise_percentages(est_134, counts_134)
            explained_14, noise_14 = compute_noise_percentages(est_14, counts_14)

            df_result = pd.DataFrame({
                "theta_center": np.repeat(theta_centers, nbins_phi),
                "phi_center": np.tile(phi_centers, nbins_theta),
                "eff_2": eff_2.flatten(),
                "eff_3": eff_3.flatten(),
                "noise_124": noise_124.flatten(),
                "noise_134": noise_134.flatten(),
                "noise_14": noise_14.flatten(),
            })
            

            # Function to plot 2D efficiency or noise maps
            def plot_2d_map(data, title, vmin=0, vmax=100, cmap='viridis'):
                global fig_idx, plot_list
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(data.T, origin='lower', aspect='auto',
                            extent=[bins_theta[0], bins_theta[-1], bins_phi[0], bins_phi[-1]],
                            interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(title)
                ax.set_xlabel(r'$\theta_{\mathrm{new}}$ [rad]')
                ax.set_ylabel(r'$\phi_{\mathrm{new}}$ [rad]')
                ax.set_title(title)
                plt.tight_layout()
                figure_name = f"{title}"
                if save_plots:
                    name_of_file = figure_name
                    final_filename = f'{fig_idx}_{name_of_file}.png'
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    plt.savefig(save_fig_path, format='png')
                if show_plots: plt.show()
                plt.close()
                return fig, ax

            # Generate plots
            fig1, _ = plot_2d_map(eff_2, 'Efficiency of Plane 2', vmin=0.8, vmax=1.0)
            fig2, _ = plot_2d_map(eff_3, 'Efficiency of Plane 3', vmin=0.8, vmax=1.0)
            
            fig3, _ = plot_2d_map(noise_124, 'Residual Noise (Missing Plane 3)', vmin=0, vmax=100)
            fig4, _ = plot_2d_map(noise_134, 'Residual Noise (Missing Plane 2)', vmin=0, vmax=100)
            fig5, _ = plot_2d_map(noise_14, 'Residual Noise (Missing Planes 2 & 3)', vmin=0, vmax=100)
            
            v = noise_124 / 100 * counts_124
            fig6, _ = plot_2d_map(v, 'Residual Absolute Noise (Missing Plane 3)', vmin=0, vmax=np.max(v))
            
            v = noise_134 / 100 * counts_134
            fig7, _ = plot_2d_map(v, 'Residual Absolute Noise (Missing Plane 2)', vmin=0, vmax=np.max(v))
            
            v = noise_14 / 100 * counts_14
            fig8, _ = plot_2d_map(v, 'Residual Absolute Noise (Missing Planes 2 & 3)', vmin=0, vmax=np.max(v))


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    # histogram_results = True
    # if histogram_results:

    #     def gaussian(x, mu, sigma, amplitude):
    #         return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    #     columns_all = [
    #         ['x', 'theta', 's', 'y', 'phi', 'th_chi'],          # TimTrack
    #         ['alt_x', 'alt_theta', 'alt_s', 'alt_y', 'alt_phi', 'alt_th_chi'],  # Alternative
    #         ['x', 'theta', 'new_s', 'y', 'phi', 'new_th_chi']   # Averaged
    #     ]

    #     titles = ['TimTrack Results', 'Alternative Results', 'Averaged Results']
    #     color_map = {
    #         "theta": "blue", "phi": "green", "x": "darkorange", "y": "darkorange",
    #         "alt_y": "darkorange", "s": "purple", "alt_s": "purple", "th_chi": "red"
    #     }

    #     ranges = {
    #         "theta": (0, np.pi/2), "phi": (-np.pi, np.pi),
    #         "x": (-500, 500), "y": (-500, 500), "alt_y": (-500, 500),
    #         "s": (-0.01, 0.02), "alt_s": (-0.01, 0.02),
    #         "th_chi": (0, 10)
    #     }

    #     fig, axs = plt.subplots(3, 6, figsize=(24, 10), constrained_layout=True)
    #     axs = axs.flatten()
    #     idx = 0
    #     quantile = 0.99
    #     fit_gaussian = True

    #     for row, (columns, row_title) in enumerate(zip(columns_all, titles)):
    #         for col in columns:
    #             data = df[col].to_numpy()
    #             data = data[data != 0]
    #             ax = axs[idx]
    #             idx += 1

    #             if len(data) == 0:
    #                 ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', color='gray')
    #                 ax.set_title(col)
    #                 continue

    #             for key in ranges:
    #                 if key in col:
    #                     left, right = ranges[key]
    #                     break
    #             else:
    #                 left, right = np.min(data), np.max(data)

    #             color = next((v for k, v in color_map.items() if k in col), 'gray')

    #             hist_data, bin_edges, _ = ax.hist(data, bins=200, color=color, alpha=0.7)

    #             ax.set_xlim(left, right)
    #             ax.set_title(col)

    #             if fit_gaussian and len(data) > 10:
    #                 try:
    #                     qmin, qmax = np.quantile(data, [1 - quantile, quantile])
    #                     filt_data = data[(data >= qmin) & (data <= qmax)]

    #                     if len(filt_data) >= 2:
    #                         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #                         popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(filt_data), np.std(filt_data), np.max(hist_data)])
    #                         x = np.linspace(qmin, qmax, 1000)
    #                         ax.plot(x, gaussian(x, *popt), 'r-', label=f'μ={popt[0]:.2g}, σ={popt[1]:.2g}')
    #                         ax.legend()
    #                     else:
    #                         ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha='center', va='center', color='gray')
    #                 except (RuntimeError, ValueError):
    #                     ax.text(0.5, 0.5, "Fit failed", transform=ax.transAxes, ha='center', va='center', color='red')

    #     for i in range(idx, len(axs)):
    #         fig.delaxes(axs[i])

    #     fig.suptitle("Histograms with Optional Gaussian Fits", fontsize=18)
    #     plt.show()


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if charge_vs_angle:
        
        print("----------------------------------------------------------------------")
        print("-------------------- Charge respect zenith angle ---------------------")
        print("----------------------------------------------------------------------")
        
        for i in range(1, 5):
            df[f"Q_P{i}"] = 0
            for j in range(1, 5):
                # Get the column name
                col_name = f"Q_P{i}s{j}"
                df[f"Q_P{i}"] += df[col_name]
        
        num_bins = 100
        n_divisions = 4
        theta_edges = np.linspace(0, np.pi/3, n_divisions + 1)
        
        # Plotting with theta ranges
        if create_plots or create_essential_plots:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust size as needed

            for i in range(1, 5):
                row = (i - 1) // 2
                col = (i - 1) % 2
                ax = axs[row, col]

                col_name = f"Q_P{i}"
                for k in range(n_divisions):
                    mask = (df["theta"] >= theta_edges[k]) & (df["theta"] < theta_edges[k+1]) & (df["processed_tt"] > 10)
                    v = df.loc[mask, col_name]
                    v = v[v != 0]
                    label = f"{theta_edges[k]:.2f} ≤ θ < {theta_edges[k+1]:.2f}"
                    ax.hist(v, bins=num_bins, range=(0, 100), alpha=0.5, label=label, histtype='step', linewidth=1.5, density=True)

                ax.set_title(col_name)
                ax.set_xlabel("Charge")
                ax.set_ylabel("Frequency")
                ax.grid(True)
                ax.legend()

            plt.tight_layout()


            plt.tight_layout()
            figure_name = f"angular_charge_mingo0{station}"
            if save_plots:
                name_of_file = figure_name
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if polya_fit:
        
        print("----------------------------------------------------------------------")
        print("----------------------------- Polya fit ------------------------------")
        print("----------------------------------------------------------------------")
        
        print("Polya fit. WIP.")

        remove_crosstalk = False
        remove_streamer = True
        crosstalk_limit = 1
        streamer_limit = 100

        FEE_calibration = {
            "Width": [
                0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                300, 310, 320, 330, 340, 350, 360, 370, 380, 390
            ],
            "Fast Charge": [
                4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
                9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
                1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
                1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
                2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
                2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
                2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
            ]
        }

        # Create the DataFrame
        FEE_calibration = pd.DataFrame(FEE_calibration)
        # Convert to NumPy arrays for interpolation
        width_table = FEE_calibration['Width'].to_numpy()
        fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()
        # Create a cubic spline interpolator
        cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

        def interpolate_fast_charge(width):
            """
            Interpolates the Fast Charge for given Width values using cubic spline interpolation.
            Parameters:
            - width (float or np.ndarray): The Width value(s) to interpolate in ns.
            Returns:
            - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
            """
            width = np.asarray(width)  # Ensure input is a NumPy array
            # Keep zero values unchanged
            result = np.where(width == 0, 0, cs(width))
            return result

        df_list_OG = [df]  # Adjust delimiter if needed


        # NO CROSSTALK SECTION --------------------------------------------------------------------------
        # Read and concatenate all files
        df_list = df_list_OG.copy()
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)
        
        # print(merged_df.columns.to_list())
        
        merged_df = merged_df[ merged_df['processed_tt'] == 1234 ]
        
        # merged_df = df.copy()

        if remove_crosstalk or remove_streamer:    
            if remove_streamer:
                for col in merged_df.columns:
                    if "Q_" in col and "s" in col:
                        merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)

        columns_to_drop = ['Time','x', 'y', 'theta', 'phi']
        merged_df = merged_df.drop(columns=columns_to_drop)

        # For all the columns apply the calibration and not change the name of the columns
        for col in merged_df.columns:
            merged_df[col] = interpolate_fast_charge(merged_df[col])

        # For each module, calculate the total charge per event, then store them in a dataframe
        total_charge = pd.DataFrame()
        for i in range(1, 5):
            total_charge[f"Q_P{i}"] = merged_df[[f"Q_P{i}s{j}" for j in range(1, 5)]].sum(axis=1)

        

        # Constants
        q_e = 1.602e-4  # fC

        # Polya model
        def polya_induced_charge(Q, theta, nbar, alpha, A, offset):
            n = Q * alpha + offset
            norm = ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
            return A * norm * (n / nbar)**theta * np.exp(-(theta + 1) * n / nbar)

        # Prepare figure
        fig, axs = plt.subplots(
            3, 4, figsize=(17, 5), sharex='col', 
            gridspec_kw={'height_ratios': [4, 1, 1]}
        )

        for idx, module in enumerate(range(1, 5)):

            # Load and preprocess data
            data = total_charge[f"Q_P{module}"].dropna().to_numpy().flatten()
            data = data[data != 0] / q_e  # convert to e–

            # Histogram
            counts, bin_edges = np.histogram(data, bins=50, range=(0, 1.1e7))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            bin_center = bin_centers[counts >= 0.05 * max(counts)][0]
            mask = (bin_centers > bin_center) & (counts > 0)
            x_fit = bin_centers[mask]
            y_fit = counts[mask]

            # Fit theta, nbar, alpha, A, offset
            p0 = [1, 1e6, 1, max(counts), 1e6]
            bounds = ([0, 0, 0, 0, -1e16], [20, 1e16, 1,  max(counts), 1e16])
            popt, _ = curve_fit(polya_induced_charge, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
            theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt
            
            # Store fit results

            polya_results = {
                "module": module,
                "theta": theta_fit,
                "nbar": nbar_fit,
                "alpha": alpha_fit,
                "A": A_fit,
                "offset": offset_fit,

                # Effective / composite parameters
                "nbar/alpha": nbar_fit / alpha_fit,
                "offset/nbar": offset_fit / nbar_fit,
                "alpha/nbar": alpha_fit / nbar_fit,
                "eta_curvature": (theta_fit + 1) * (alpha_fit / nbar_fit),
                "width_proxy": nbar_fit / np.sqrt(theta_fit + 1),
                
                # Mode (only valid for theta > 1)
                "Q_mode": ((nbar_fit * (theta_fit - 1)) - (alpha_fit * offset_fit)) / (alpha_fit**2 * theta_fit)
                        if theta_fit > 1 else 0,
            }

            if 'polya_fit_list' not in locals():
                polya_fit_list = []
            polya_fit_list.append(polya_results)

            # Fine x for fit curve
            x_fine = np.linspace(0, 1.1e7, 300)
            y_model = polya_induced_charge(x_fine, *popt)

            # Residuals
            residuals = y_fit - polya_induced_charge(x_fit, *popt)
            residuals_norm = residuals / y_fit * 100

            # Plot index
            ax1 = axs[0, idx]
            ax2 = axs[1, idx]
            ax3 = axs[2, idx]

            # --- Fit plot ---
        #     plot_label = (
        #         rf"$\theta={theta_fit:.2f},\ \bar{{n}}={nbar_fit:.0f},\ "
        #         rf"\alpha={alpha_fit:.2f},\ A={A_fit:.2f},\ \mathrm{{off}}={offset_fit:.2f},\ "
        #         rf"\bar{{n}}/\alpha={nbar_fit / alpha_fit:.3g}$"
        #     )
            plot_label = (
                rf"$\theta={theta_fit:.2f},\ \mathrm{{off}}={offset_fit:.2f},\ "
                rf"\bar{{n}}/\alpha={nbar_fit / alpha_fit:.3g}$"
            )
            ax1.plot(x_fine, y_model, "r--", label = plot_label)
            ax1.plot(x_fit, y_fit, 'bo', markersize = 2)
            ax1.set_title(f"Module {module}")
            ax1.legend(fontsize=8)
            ax1.grid(True)
            if idx == 0:
                ax1.set_ylabel("Entries")

            # --- Residuals ---
            ax2.axhline(0, color='gray', linestyle='--')
            ax2.plot(x_fit, residuals, 'k.')
            if idx == 0:
                ax2.set_ylabel("Res.")

            ax2.grid(True)

            # --- Normalized residuals ---
            ax3.axhline(0, color='gray', linestyle='--')
            ax3.plot(x_fit, residuals_norm, 'k.')
            if idx == 0:
                ax3.set_ylabel("Res. (%)")
            ax3.set_xlabel("Induced equivalent electrons")
            ax3.set_ylim(-10, 100)
            ax3.grid(True)

        plt.tight_layout()
        figure_name = f"polya_fit_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

        df_polya_fit = pd.DataFrame(polya_fit_list)

        print("Polya fit results:")
        with pd.option_context('display.precision', 1):
            print(df_polya_fit)
        
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        
        print("Polya fit respect to the angle. WIP.")

        remove_crosstalk = False
        remove_streamer = True
        crosstalk_limit = 1
        streamer_limit = 100

        FEE_calibration = {
            "Width": [
                0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                300, 310, 320, 330, 340, 350, 360, 370, 380, 390
            ],
            "Fast Charge": [
                4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
                9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
                1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
                1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
                2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
                2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
                2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
            ]
        }

        # Create the DataFrame
        FEE_calibration = pd.DataFrame(FEE_calibration)
        # Convert to NumPy arrays for interpolation
        width_table = FEE_calibration['Width'].to_numpy()
        fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()
        # Create a cubic spline interpolator
        cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

        def interpolate_fast_charge(width):
            """
            Interpolates the Fast Charge for given Width values using cubic spline interpolation.
            Parameters:
            - width (float or np.ndarray): The Width value(s) to interpolate in ns.
            Returns:
            - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
            """
            width = np.asarray(width)  # Ensure input is a NumPy array
            # Keep zero values unchanged
            result = np.where(width == 0, 0, cs(width))
            return result

        df_list_OG = [df]  # Adjust delimiter if needed
        df_list = df_list_OG.copy()
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)
        
        # print(merged_df.columns.to_list())
        
        merged_df = merged_df[ merged_df['processed_tt'] > 10 ]
        merged_df = merged_df[ merged_df['theta'] < 0.5 ]

        if remove_crosstalk or remove_streamer:    
            if remove_streamer:
                for col in merged_df.columns:
                    if "Q_" in col and "s" in col:
                        merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)

        columns_to_drop = ['Time', 'x', 'y', 'theta', 'phi']
        merged_df = merged_df.drop(columns=columns_to_drop)

        # For all the columns apply the calibration and not change the name of the columns
        for col in merged_df.columns:
            merged_df[col] = interpolate_fast_charge(merged_df[col])

        # For each module, calculate the total charge per event, then store them in a dataframe
        total_charge = pd.DataFrame()
        for i in range(1, 5):
            total_charge[f"Q_P{i}"] = merged_df[[f"Q_P{i}s{j}" for j in range(1, 5)]].sum(axis=1)

        # Constants
        q_e = 1.602e-4  # fC

        # Polya model
        def polya_induced_charge(Q, theta, nbar, alpha, A, offset):
            n = Q * alpha + offset
            norm = ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
            return A * norm * (n / nbar)**theta * np.exp(-(theta + 1) * n / nbar)

        # Prepare figure
        fig, axs = plt.subplots(
            3, 4, figsize=(17, 5), sharex='col', 
            gridspec_kw={'height_ratios': [4, 1, 1]}
        )

        for idx, module in enumerate(range(1, 5)):

            # Load and preprocess data
            data = total_charge[f"Q_P{module}"].dropna().to_numpy().flatten()
            data = data[data != 0] / q_e  # convert to e–

            # Histogram
            counts, bin_edges = np.histogram(data, bins=50, range=(0, 1.1e7))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            bin_center = bin_centers[counts >= 0.05 * max(counts)][0]
            mask = (bin_centers > bin_center) & (counts > 0)
            x_fit = bin_centers[mask]
            y_fit = counts[mask]

            # Fit theta, nbar, alpha, A, offset
            p0 = [1, 1e6, 1, max(counts), 1e6]
            bounds = ([0, 0, 0, 0, -1e16], [20, 1e16, 1,  max(counts), 1e16])
            popt, _ = curve_fit(polya_induced_charge, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
            theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt
            
            # Store fit results

            # Fine x for fit curve
            x_fine = np.linspace(0, 1.1e7, 300)
            y_model = polya_induced_charge(x_fine, *popt)

            # Residuals
            residuals = y_fit - polya_induced_charge(x_fit, *popt)
            residuals_norm = residuals / y_fit * 100

            # Plot index
            ax1 = axs[0, idx]
            ax2 = axs[1, idx]
            ax3 = axs[2, idx]

            plot_label = (
                rf"$\theta={theta_fit:.2f},\ \mathrm{{off}}={offset_fit:.2f},\ "
                rf"\bar{{n}}/\alpha={nbar_fit / alpha_fit:.3g}$"
            )
            ax1.plot(x_fine, y_model, "r--", label = plot_label)
            ax1.plot(x_fit, y_fit, 'bo', markersize = 2)
            ax1.set_title(f"Module {module}")
            ax1.legend(fontsize=8)
            ax1.grid(True)
            if idx == 0:
                ax1.set_ylabel("Entries")

            # --- Residuals ---
            ax2.axhline(0, color='gray', linestyle='--')
            ax2.plot(x_fit, residuals, 'k.')
            if idx == 0:
                ax2.set_ylabel("Res.")

            ax2.grid(True)

            # --- Normalized residuals ---
            ax3.axhline(0, color='gray', linestyle='--')
            ax3.plot(x_fit, residuals_norm, 'k.')
            if idx == 0:
                ax3.set_ylabel("Res. (%)")
            ax3.set_xlabel("Induced equivalent electrons")
            ax3.set_ylim(-10, 100)
            ax3.grid(True)

        plt.tight_layout()
        figure_name = f"polya_fit_zenith_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

        
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        
        print("Polya fit respect to the angle. WIP.")

        df_list_OG = [df]  # Adjust delimiter if needed
        df_list = df_list_OG.copy()
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)
        
        # print(merged_df.columns.to_list())
        
        merged_df = merged_df[ merged_df['processed_tt'] > 10 ]
        merged_df = merged_df[ merged_df['theta'] > 0.5 ]

        if remove_crosstalk or remove_streamer:    
            if remove_streamer:
                for col in merged_df.columns:
                    if "Q_" in col and "s" in col:
                        merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)

        columns_to_drop = ['Time', 'x', 'y', 'theta', 'phi']
        merged_df = merged_df.drop(columns=columns_to_drop)

        # For all the columns apply the calibration and not change the name of the columns
        for col in merged_df.columns:
            merged_df[col] = interpolate_fast_charge(merged_df[col])

        # For each module, calculate the total charge per event, then store them in a dataframe
        total_charge = pd.DataFrame()
        for i in range(1, 5):
            total_charge[f"Q_P{i}"] = merged_df[[f"Q_P{i}s{j}" for j in range(1, 5)]].sum(axis=1)

        # Constants
        q_e = 1.602e-4  # fC

        # Polya model
        def polya_induced_charge(Q, theta, nbar, alpha, A, offset):
            n = Q * alpha + offset
            norm = ((theta + 1) ** (theta + 1)) / gamma(theta + 1)
            return A * norm * (n / nbar)**theta * np.exp(-(theta + 1) * n / nbar)

        # Prepare figure
        fig, axs = plt.subplots(
            3, 4, figsize=(17, 5), sharex='col', 
            gridspec_kw={'height_ratios': [4, 1, 1]}
        )

        for idx, module in enumerate(range(1, 5)):

            # Load and preprocess data
            data = total_charge[f"Q_P{module}"].dropna().to_numpy().flatten()
            data = data[data != 0] / q_e  # convert to e–

            # Histogram
            counts, bin_edges = np.histogram(data, bins=50, range=(0, 1.1e7))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            bin_center = bin_centers[counts >= 0.05 * max(counts)][0]
            mask = (bin_centers > bin_center) & (counts > 0)
            x_fit = bin_centers[mask]
            y_fit = counts[mask]

            # Fit theta, nbar, alpha, A, offset
            p0 = [1, 1e6, 1, max(counts), 1e6]
            bounds = ([0, 0, 0, 0, -1e16], [20, 1e16, 1,  max(counts), 1e16])
            popt, _ = curve_fit(polya_induced_charge, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=100000)
            theta_fit, nbar_fit, alpha_fit, A_fit, offset_fit = popt
            
            # Store fit results
            

            # Fine x for fit curve
            x_fine = np.linspace(0, 1.1e7, 300)
            y_model = polya_induced_charge(x_fine, *popt)

            # Residuals
            residuals = y_fit - polya_induced_charge(x_fit, *popt)
            residuals_norm = residuals / y_fit * 100

            # Plot index
            ax1 = axs[0, idx]
            ax2 = axs[1, idx]
            ax3 = axs[2, idx]

            plot_label = (
                rf"$\theta={theta_fit:.2f},\ \mathrm{{off}}={offset_fit:.2f},\ "
                rf"\bar{{n}}/\alpha={nbar_fit / alpha_fit:.3g}$"
            )
            ax1.plot(x_fine, y_model, "r--", label = plot_label)
            ax1.plot(x_fit, y_fit, 'bo', markersize = 2)
            ax1.set_title(f"Module {module}")
            ax1.legend(fontsize=8)
            ax1.grid(True)
            if idx == 0:
                ax1.set_ylabel("Entries")

            # --- Residuals ---
            ax2.axhline(0, color='gray', linestyle='--')
            ax2.plot(x_fit, residuals, 'k.')
            if idx == 0:
                ax2.set_ylabel("Res.")

            ax2.grid(True)

            # --- Normalized residuals ---
            ax3.axhline(0, color='gray', linestyle='--')
            ax3.plot(x_fit, residuals_norm, 'k.')
            if idx == 0:
                ax3.set_ylabel("Res. (%)")
            ax3.set_xlabel("Induced equivalent electrons")
            ax3.set_ylim(-10, 100)
            ax3.grid(True)

        plt.tight_layout()
        figure_name = f"polya_fit_zenith_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if real_strip_case_study:
        
        print("----------------------------------------------------------------------")
        print("-------------------- Real adjacent and single cases ------------------")
        print("----------------------------------------------------------------------")
        
        print("Real strip case study. WIP.")

        # Read and concatenate all files
        df_list = [df]  # Adjust delimiter if needed
        merged_df = pd.concat(df_list, ignore_index=True)

        # Drop duplicates if necessary
        merged_df.drop_duplicates(inplace=True)

        FEE_calibration = {
            "Width": [
                0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                300, 310, 320, 330, 340, 350, 360, 370, 380, 390
            ],
            "Fast Charge": [
                4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
                9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
                1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
                1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
                2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
                2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
                2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
            ]
        }

        # Create the DataFrame
        FEE_calibration = pd.DataFrame(FEE_calibration)


        # Convert to NumPy arrays for interpolation
        width_table = FEE_calibration['Width'].to_numpy()
        fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()

        # Create a cubic spline interpolator
        cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

        def interpolate_fast_charge(width):
            """
            Interpolates the Fast Charge for given Width values using cubic spline interpolation.
            Parameters:
            - width (float or np.ndarray): The Width value(s) to interpolate in ns.
            Returns:
            - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
            """
            width = np.asarray(width)  # Ensure input is a NumPy array

            # Keep zero values unchanged
            result = np.where(width == 0, 0, cs(width))

            return result

        columns_to_drop = ['Time', 'x', 'y', 'theta', 'phi']
        merged_df = merged_df.drop(columns=columns_to_drop)

        # For all the columns apply the calibration and not change the name of the columns
        # for col in merged_df.columns:
        #     merged_df[col] = interpolate_fast_charge(merged_df[col])

        # Initialize dictionaries to store charge distributions
        singles = {f'single_M{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}
        double_adj = {f'double_M{i}_s{j}{j+1}': [] for i in range(1, 5) for j in range(1, 4)}
        double_non_adj = {f'double_M{i}_s{pair[0]}{pair[1]}': [] for i in range(1, 5) for pair in [(1,3), (2,4), (1,4)]}
        triple_adj = {f'triple_M{i}_s{j}{j+1}{j+2}': [] for i in range(1, 5) for j in range(1, 3)}
        triple_non_adj = {f'triple_M{i}_s{triplet[0]}{triplet[1]}{triplet[2]}': [] for i in range(1, 5) for triplet in [(1,2,4), (1,3,4)]}
        quadruples = {f'quadruple_M{i}_s1234': [] for i in range(1, 5)}

        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                col_name = f"Q_P{i}s{j}"  # Column name
                v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0] + 1  # Get active strip indices (1-based)
                charges = charge_matrix[event_idx, nonzero_strips - 1]  # Get nonzero charges

                # Single detection
                if count == 1:
                    key = f'single_M{i}_s{nonzero_strips[0]}'
                    singles[key].append((charges[0],))

                # Double adjacent
                elif count == 2 and nonzero_strips[1] - nonzero_strips[0] == 1:
                    key = f'double_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}'
                    double_adj[key].append(tuple(charges))

                # Double non-adjacent
                elif count == 2 and nonzero_strips[1] - nonzero_strips[0] > 1:
                    key = f'double_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}'
                    if key in double_non_adj:
                        double_non_adj[key].append(tuple(charges))

                # Triple adjacent
                elif count == 3 and (nonzero_strips[2] - nonzero_strips[0] == 2):
                    key = f'triple_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}{nonzero_strips[2]}'
                    triple_adj[key].append(tuple(charges))

                # Triple non-adjacent
                elif count == 3 and (nonzero_strips[2] - nonzero_strips[0] > 2):
                    key = f'triple_M{i}_s{nonzero_strips[0]}{nonzero_strips[1]}{nonzero_strips[2]}'
                    if key in triple_non_adj:
                        triple_non_adj[key].append(tuple(charges))

                # Quadruple detection
                elif count == 4:
                    key = f'quadruple_M{i}_s1234'
                    quadruples[key].append(tuple(charges))

        # Convert results to DataFrames
        df_singles = {k: pd.DataFrame(v, columns=["Charge1"]) for k, v in singles.items()}
        df_double_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2"]) for k, v in double_adj.items()}
        df_double_non_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2"]) for k, v in double_non_adj.items()}
        df_triple_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2", "Charge3"]) for k, v in triple_adj.items()}
        df_triple_non_adj = {k: pd.DataFrame(v, columns=["Charge1", "Charge2", "Charge3"]) for k, v in triple_non_adj.items()}
        df_quadruples = {k: pd.DataFrame(v, columns=["Charge1", "Charge2", "Charge3", "Charge4"]) for k, v in quadruples.items()}

        # Singles
        single_M1_s1 = df_singles['single_M1_s1']
        single_M1_s2 = df_singles['single_M1_s2']
        single_M1_s3 = df_singles['single_M1_s3']
        single_M1_s4 = df_singles['single_M1_s4']

        single_M2_s1 = df_singles['single_M2_s1']
        single_M2_s2 = df_singles['single_M2_s2']
        single_M2_s3 = df_singles['single_M2_s3']
        single_M2_s4 = df_singles['single_M2_s4']

        single_M3_s1 = df_singles['single_M3_s1']
        single_M3_s2 = df_singles['single_M3_s2']
        single_M3_s3 = df_singles['single_M3_s3']
        single_M3_s4 = df_singles['single_M3_s4']

        single_M4_s1 = df_singles['single_M4_s1']
        single_M4_s2 = df_singles['single_M4_s2']
        single_M4_s3 = df_singles['single_M4_s3']
        single_M4_s4 = df_singles['single_M4_s4']

        # Double adjacent
        double_M1_s12 = df_double_adj['double_M1_s12']
        double_M1_s23 = df_double_adj['double_M1_s23']
        double_M1_s34 = df_double_adj['double_M1_s34']

        double_M2_s12 = df_double_adj['double_M2_s12']
        double_M2_s23 = df_double_adj['double_M2_s23']
        double_M2_s34 = df_double_adj['double_M2_s34']

        double_M3_s12 = df_double_adj['double_M3_s12']
        double_M3_s23 = df_double_adj['double_M3_s23']
        double_M3_s34 = df_double_adj['double_M3_s34']

        double_M4_s12 = df_double_adj['double_M4_s12']
        double_M4_s23 = df_double_adj['double_M4_s23']
        double_M4_s34 = df_double_adj['double_M4_s34']

        # Doubles non adjacent
        double_M1_s13 = df_double_non_adj['double_M1_s13']
        double_M1_s24 = df_double_non_adj['double_M1_s24']
        double_M1_s14 = df_double_non_adj['double_M1_s14']

        double_M2_s13 = df_double_non_adj['double_M2_s13']
        double_M2_s24 = df_double_non_adj['double_M2_s24']
        double_M2_s14 = df_double_non_adj['double_M2_s14']

        double_M3_s13 = df_double_non_adj['double_M3_s13']
        double_M3_s24 = df_double_non_adj['double_M3_s24']
        double_M3_s14 = df_double_non_adj['double_M3_s14']

        double_M4_s13 = df_double_non_adj['double_M4_s13']
        double_M4_s24 = df_double_non_adj['double_M4_s24']
        double_M4_s14 = df_double_non_adj['double_M4_s14']

        # Triple adjacent
        triple_M1_s123 = df_triple_adj['triple_M1_s123']
        triple_M1_s234 = df_triple_adj['triple_M1_s234']

        triple_M2_s123 = df_triple_adj['triple_M2_s123']
        triple_M2_s234 = df_triple_adj['triple_M2_s234']

        triple_M3_s123 = df_triple_adj['triple_M3_s123']
        triple_M3_s234 = df_triple_adj['triple_M3_s234']

        triple_M4_s123 = df_triple_adj['triple_M4_s123']
        triple_M4_s234 = df_triple_adj['triple_M4_s234']

        # Triple non adjacent
        triple_M1_s124 = df_triple_non_adj['triple_M1_s124']
        triple_M1_s134 = df_triple_non_adj['triple_M1_s134']

        triple_M2_s124 = df_triple_non_adj['triple_M2_s124']
        triple_M2_s134 = df_triple_non_adj['triple_M2_s134']

        triple_M3_s124 = df_triple_non_adj['triple_M3_s124']
        triple_M3_s134 = df_triple_non_adj['triple_M3_s134']

        triple_M4_s124 = df_triple_non_adj['triple_M4_s124']
        triple_M4_s134 = df_triple_non_adj['triple_M4_s134']

        # Quadruple
        quadruple_M1_s1234 = df_quadruples['quadruple_M1_s1234']
        quadruple_M2_s1234 = df_quadruples['quadruple_M2_s1234']
        quadruple_M3_s1234 = df_quadruples['quadruple_M3_s1234']
        quadruple_M4_s1234 = df_quadruples['quadruple_M4_s1234']

        # Helper function to rename columns based on their source dataset
        def rename_columns(df, source_name):
            return df.rename(columns={col: f"{source_name}_{col}" for col in df.columns})

        # Initialize dictionary
        real_multiplicities = {}

        # Define modules
        modules = ["M1", "M2", "M3", "M4"]

        # Loop over modules
        for module in modules:
            real_multiplicities[f"real_single_{module}_s1"] = pd.concat([
                rename_columns(globals()[f"single_{module}_s1"], f"single_{module}_s1"),
                rename_columns(globals()[f"double_{module}_s13"][['Charge1', 'Charge2']], f"double_{module}_s13"),
                rename_columns(globals()[f"double_{module}_s14"][['Charge1', 'Charge2']], f"double_{module}_s14"),
                rename_columns(globals()[f"triple_{module}_s134"][['Charge1']], f"triple_{module}_s134")
            ], axis=1)

            real_multiplicities[f"real_single_{module}_s2"] = pd.concat([
                rename_columns(globals()[f"single_{module}_s2"], f"single_{module}_s2"),
                rename_columns(globals()[f"double_{module}_s24"][['Charge1', 'Charge2']], f"double_{module}_s24")
            ], axis=1)

            real_multiplicities[f"real_single_{module}_s3"] = pd.concat([
                rename_columns(globals()[f"single_{module}_s3"], f"single_{module}_s3"),
                rename_columns(globals()[f"double_{module}_s13"][['Charge1', 'Charge2']], f"double_{module}_s13")
            ], axis=1)

            real_multiplicities[f"real_single_{module}_s4"] = pd.concat([
                rename_columns(globals()[f"single_{module}_s4"], f"single_{module}_s4"),
                rename_columns(globals()[f"double_{module}_s24"][['Charge1', 'Charge2']], f"double_{module}_s24"),
                rename_columns(globals()[f"double_{module}_s14"][['Charge1', 'Charge2']], f"double_{module}_s14"),
                rename_columns(globals()[f"triple_{module}_s124"][['Charge3']], f"triple_{module}_s124")
            ], axis=1)

            # Doubles adjacent
            real_multiplicities[f"real_double_{module}_s12"] = pd.concat([
                rename_columns(globals()[f"double_{module}_s12"], f"double_{module}_s12"),
                rename_columns(globals()[f"triple_{module}_s124"][['Charge1', 'Charge2']], f"triple_{module}_s124")
            ], axis=1)

            real_multiplicities[f"real_double_{module}_s23"] = rename_columns(globals()[f"double_{module}_s23"], f"double_{module}_s23")

            real_multiplicities[f"real_double_{module}_s34"] = pd.concat([
                rename_columns(globals()[f"double_{module}_s34"], f"double_{module}_s34"),
                rename_columns(globals()[f"triple_{module}_s134"][['Charge2', 'Charge3']], f"triple_{module}_s134")
            ], axis=1)

            # Triples adjacent
            real_multiplicities[f"real_triple_{module}_s123"] = rename_columns(globals()[f"triple_{module}_s123"], f"triple_{module}_s123")
            real_multiplicities[f"real_triple_{module}_s234"] = rename_columns(globals()[f"triple_{module}_s234"], f"triple_{module}_s234")

            # Quadruples
            real_multiplicities[f"real_quadruple_{module}_s1234"] = rename_columns(globals()[f"quadruple_{module}_s1234"], f"quadruple_{module}_s1234")

        # List the keys
        print(real_multiplicities.keys())
        cases = ["real_single", "real_double", "real_triple", "real_quadruple"]

        for case in cases:
            fig_rows = len(modules)
            fig_cols = 0

            # First, compute the max number of columns across all modules (for consistent layout)
            max_columns = 0
            all_combined_dfs = []  # Store the per-module DataFrames

            for module in modules:
                other_key = f"{case}_{module}"
                matching_keys = sorted([key for key in real_multiplicities if key.startswith(f"{other_key}_")])

                if not matching_keys:
                    print(f"No data for {other_key}")
                    all_combined_dfs.append(None)
                    continue

                # combined_df = pd.concat([real_multiplicities[key] for key in matching_keys], axis=1)
                
                seen_columns = set()
                dfs_unique = []
                for key in matching_keys:
                    df = real_multiplicities[key]
                    df_unique = df[[col for col in df.columns if col not in seen_columns]]
                    seen_columns.update(df_unique.columns)
                    dfs_unique.append(df_unique)

                combined_df = pd.concat(dfs_unique, axis=1)
                all_combined_dfs.append(combined_df)

                if combined_df.shape[1] > max_columns:
                    max_columns = combined_df.shape[1]

            # Now that we know max_columns, build the subplot grid
            fig, axs = plt.subplots(fig_rows, max_columns, figsize=(4 * max_columns, 4 * fig_rows))

            # Make axs 2D no matter what
            if fig_rows == 1:
                axs = [axs]
            if max_columns == 1:
                axs = [[ax] for ax in axs]

            for a, (module, combined_df) in enumerate(zip(modules, all_combined_dfs)):
                if combined_df is None:
                    continue  # Skip missing data

                for i, column in enumerate(combined_df.columns):
                    # axs[a][i].hist(combined_df[column], bins=70, range=(0, 1500), histtype="step", linewidth=1.5, density=False)
                    axs[a][i].hist(combined_df[column], bins=70, range=(0, 100), alpha = 0.6, linewidth=1.5, density=False)
                    axs[a][i].set_title(f"{module} - {column}")
                    axs[a][i].set_xlabel("Charge")
                    axs[a][i].set_ylabel("Frequency")
                    axs[a][i].grid(True)

                # Hide unused subplots (if any)
                for j in range(i + 1, max_columns):
                    axs[a][j].axis("off")

            plt.tight_layout()
            figure_name = f"real_multiplicities_{case}_{station}"
            if save_plots:
                name_of_file = figure_name
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots: plt.show()
            plt.close()

        modules = ["M1", "M2", "M3", "M4"]
        sum_real_multiplicities = {}

        for module in modules:
            # -- DOUBLES -------------------------------------------------
            # s12
            df_12 = real_multiplicities[f"real_double_{module}_s12"]
            sum_12 = df_12.sum(axis=1, numeric_only=True)  # Sum of all columns in that DataFrame
            sum_real_multiplicities[f"sum_real_double_{module}_s12"] = pd.DataFrame({"Charge12": sum_12})

            # s23
            df_23 = real_multiplicities[f"real_double_{module}_s23"]
            sum_23 = df_23.sum(axis=1, numeric_only=True)
            sum_real_multiplicities[f"sum_real_double_{module}_s23"] = pd.DataFrame({"Charge23": sum_23})

            # s34
            df_34 = real_multiplicities[f"real_double_{module}_s34"]
            sum_34 = df_34.sum(axis=1, numeric_only=True)
            sum_real_multiplicities[f"sum_real_double_{module}_s34"] = pd.DataFrame({"Charge34": sum_34})

            # -- TRIPLES -------------------------------------------------
            # s123
            df_123 = real_multiplicities[f"real_triple_{module}_s123"]
            sum_123 = df_123.sum(axis=1, numeric_only=True)
            sum_real_multiplicities[f"sum_real_triple_{module}_s123"] = pd.DataFrame({"Charge123": sum_123})

            # s234
            df_234 = real_multiplicities[f"real_triple_{module}_s234"]
            sum_234 = df_234.sum(axis=1, numeric_only=True)
            sum_real_multiplicities[f"sum_real_triple_{module}_s234"] = pd.DataFrame({"Charge234": sum_234})

            # -- QUADRUPLES ----------------------------------------------
            # s1234
            df_1234 = real_multiplicities[f"real_quadruple_{module}_s1234"]
            sum_1234 = df_1234.sum(axis=1, numeric_only=True)
            sum_real_multiplicities[f"sum_real_quadruple_{module}_s1234"] = pd.DataFrame({"Charge1234": sum_1234})

        sum_real_multiplicities.update({
            k: df for k, df in real_multiplicities.items() if "real_single" in k
        })

        # 1) Merge single from real_multiplicities + double/triple/quad from sum_real_multiplicities
        combined_multiplicities = {}

        # Copy the single-case DataFrames as-is from real_multiplicities
        for key, df in real_multiplicities.items():
            if key.startswith("real_single"):
                combined_multiplicities[key] = df

        # Copy (and rename) the double/triple/quad entries from sum_real_multiplicities
        for key, df in sum_real_multiplicities.items():
            # They have keys like "sum_real_double_M1_s12"
            # We rename them to match "real_double_M1_s12"
            new_key = key.replace("sum_", "")  # e.g. "sum_real_double_M1_s12" -> "real_double_M1_s12"
            combined_multiplicities[new_key] = df

        cases = ["real_single", "real_double", "real_triple", "real_quadruple"]
        modules = ["M1", "M2", "M3", "M4"]

        for case in cases:
            fig_rows = len(modules)
            max_columns = 0
            all_combined_dfs = []

            # 1) Identify & combine all DataFrames for each module
            for module in modules:
                # We'll look for dictionary keys that start like "real_single_M1_...", etc.
                # Example: "real_double_M1_s12"
                prefix = f"{case}_{module}"
                matching_keys = sorted(k for k in combined_multiplicities if k.startswith(prefix))

                if not matching_keys:
                    print(f"No data for {prefix}")
                    all_combined_dfs.append(None)
                    continue

                # Concatenate all DataFrames for this module into one big DF (columns side by side)
                # combined_df = pd.concat([combined_multiplicities[k] for k in matching_keys], axis=1)
                seen_columns = set()
                dfs_unique = []
                for key in matching_keys:
                    df = combined_multiplicities[key]
                    unique_cols = [col for col in df.columns if col not in seen_columns]
                    if unique_cols:
                        df_unique = df[unique_cols]
                        dfs_unique.append(df_unique)
                        seen_columns.update(unique_cols)

                combined_df = pd.concat(dfs_unique, axis=1)
                
                all_combined_dfs.append(combined_df)

                # Track largest number of columns (for consistent subplot layout)
                if combined_df.shape[1] > max_columns:
                    max_columns = combined_df.shape[1]

            # 2) Build the subplot grid for this case
            fig, axs = plt.subplots(fig_rows, max_columns, figsize=(4 * max_columns, 4 * fig_rows))

            # Make sure axs is 2D no matter what
            if fig_rows == 1:
                axs = [axs]  # wrap in a list so axs[a][i] won't error
            if max_columns == 1:
                axs = [[ax] for ax in axs]  # similarly wrap columns

            # 3) Plot each row (module) and column (strips or partial sums)
            for row_idx, (module, combined_df) in enumerate(zip(modules, all_combined_dfs)):
                if combined_df is None:
                    # No data for this module
                    continue

                for col_idx, column_name in enumerate(combined_df.columns):
                    # axs[row_idx][col_idx].hist( combined_df[column_name], bins=70, range=(0, 2000), histtype="step", linewidth=1.5, density=False )
                    axs[row_idx][col_idx].hist( combined_df[column_name], bins=70, range=(0, 100), alpha = 0.6, linewidth=1.5, density=False )
                    axs[row_idx][col_idx].set_title(f"{module} - {column_name}")
                    axs[row_idx][col_idx].set_xlabel("Charge")
                    axs[row_idx][col_idx].set_ylabel("Frequency")
                    axs[row_idx][col_idx].grid(True)

                # Hide any unused subplots in this row
                for hidden_col_idx in range(col_idx + 1, max_columns):
                    axs[row_idx][hidden_col_idx].axis("off")

            plt.tight_layout()
            figure_name = f"sum_{case}_{station}"
            if save_plots:
                name_of_file = figure_name
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots: plt.show()
            plt.close()

    else:
        print("Real strip case study not available yet. WIP.")


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if multiplicity_calculations:
        
        print("----------------------------------------------------------------------")
        print("----------------------- Multiplicity calculations --------------------")
        print("----------------------------------------------------------------------")

        # STEP 0 ----------------------------------------
        # We could add here a step 0 which is the calculation of the real single particle
        # spectrum, as I did in the new_charge_analysis code, but it would require to study
        # the self-trigger spectrum, in case some double non adjacent etc are remarkably
        # noisy and should not be taken into account as pure single particle events.
        
        # STEP 1 ----------------------------------------
        # Assuming a single particle spectrum for each plane (we could refine this creating
        # a single particle spectrum for each strip using the completed STEP 0, and even for
        # each trigger-type), we can fit a Polya to it and generate the sums of Polya's to
        # see how the total charge spectrum for each cluster size can be explained by the sum
        # of the single particle spectra.

        # Take the cluster size 1 charge spectrum per plane for four-plane coincidence events

        remove_crosstalk = False
        crosstalk_limit = 0.1 #2.6

        remove_streamer = True
        streamer_limit = 100

        # Read and concatenate all files
        df_list = [df]  # Adjust delimiter if needed
        merged_df = pd.concat(df_list, ignore_index=True)

        # Drop duplicates if necessary
        merged_df.drop_duplicates(inplace=True)

        FEE_calibration = {
            "Width": [
                0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                300, 310, 320, 330, 340, 350, 360, 370, 380, 390
            ],
            "Fast Charge": [
                4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
                9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
                1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
                1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
                2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
                2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
                2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
            ]
        }

        # # Create the DataFrame
        FEE_calibration = pd.DataFrame(FEE_calibration)


        # Convert to NumPy arrays for interpolation
        width_table = FEE_calibration['Width'].to_numpy()
        fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()

        # Create a cubic spline interpolator
        cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

        def interpolate_fast_charge(width):
            width = np.asarray(width)  # Ensure input is a NumPy array

            # Keep zero values unchanged
            result = np.where(width == 0, 0, cs(width))

            return result


        if remove_crosstalk or remove_streamer:
            if remove_crosstalk:
                    for col in merged_df.columns:
                        if "Q_" in col and "s" in col:
                                merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
                        
            if remove_streamer:
                    for col in merged_df.columns:
                        if "Q_" in col and "s" in col:
                                merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)     


        columns_to_drop = ['Time', 'x', 'y', 'theta', 'phi']
        merged_df = merged_df.drop(columns=columns_to_drop)

        # For all the columns apply the calibration and not change the name of the columns
        for col in merged_df.columns:
            if "processed_tt" in col:
                continue
            merged_df[col] = interpolate_fast_charge(merged_df[col])

        # Create a 4x4 subfigure
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(1, 5):
            for j in range(1, 5):
                    # Get the column name
                    col_name = f"Q_P{i}s{j}"
                    
                    # Plot the histogram
                    v = merged_df[col_name]
                    v = v[v != 0]
                    axs[i-1, j-1].hist(v, bins=100, range=(0, 1200))
                    axs[i-1, j-1].set_title(col_name)
                    axs[i-1, j-1].set_xlabel("Charge")
                    axs[i-1, j-1].set_ylabel("Frequency")
                    axs[i-1, j-1].grid(True)

        plt.tight_layout()
        figure_name = f"all_channels_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


        # Create a vector of minimum and other of maximum charge for double adjacent detections, for each module
        # Dictionaries to store min and max charge values for double-adjacent detections
        double_adjacent_P1_min, double_adjacent_P1_max = [], []
        double_adjacent_P2_min, double_adjacent_P2_max = [], []
        double_adjacent_P3_min, double_adjacent_P3_max = [], []
        double_adjacent_P4_min, double_adjacent_P4_max = [], []

        double_non_adjacent_P1_min, double_non_adjacent_P1_max = [], []
        double_non_adjacent_P2_min, double_non_adjacent_P2_max = [], []
        double_non_adjacent_P3_min, double_non_adjacent_P3_max = [], []
        double_non_adjacent_P4_min, double_non_adjacent_P4_max = [], []

        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                    col_name = f"Q_P{i}s{j}"  # Column name
                    v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                    charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                    nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
                    charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

                    if count == 2 and np.all(np.diff(nonzero_strips) == 1):  # Double adjacent
                        min_charge = np.min(charges)
                        max_charge = np.max(charges)

                        if i == 1:
                                double_adjacent_P1_min.append(min_charge)
                                double_adjacent_P1_max.append(max_charge)
                        elif i == 2:
                                double_adjacent_P2_min.append(min_charge)
                                double_adjacent_P2_max.append(max_charge)
                        elif i == 3:
                                double_adjacent_P3_min.append(min_charge)
                                double_adjacent_P3_max.append(max_charge)
                        elif i == 4:
                                double_adjacent_P4_min.append(min_charge)
                                double_adjacent_P4_max.append(max_charge)
                                
                    if count == 2 and np.all(np.diff(nonzero_strips) != 1):
                        min_charge = np.min(charges)
                        max_charge = np.max(charges)

                        if i == 1:
                                double_non_adjacent_P1_min.append(min_charge)
                                double_non_adjacent_P1_max.append(max_charge)
                        elif i == 2:
                                double_non_adjacent_P2_min.append(min_charge)
                                double_non_adjacent_P2_max.append(max_charge)
                        elif i == 3:
                                double_non_adjacent_P3_min.append(min_charge)
                                double_non_adjacent_P3_max.append(max_charge)
                        elif i == 4:
                                double_non_adjacent_P4_min.append(min_charge)
                                double_non_adjacent_P4_max.append(max_charge)
                        
                    

        # Convert lists to DataFrames for better visualization
        df_double_adj_M1 = pd.DataFrame({"Min": double_adjacent_P1_min, "Max": double_adjacent_P1_max, "Sum": np.array(double_adjacent_P1_min) + np.array(double_adjacent_P1_max)})
        df_double_adj_M2 = pd.DataFrame({"Min": double_adjacent_P2_min, "Max": double_adjacent_P2_max, "Sum": np.array(double_adjacent_P2_min) + np.array(double_adjacent_P2_max)})
        df_double_adj_M3 = pd.DataFrame({"Min": double_adjacent_P3_min, "Max": double_adjacent_P3_max, "Sum": np.array(double_adjacent_P3_min) + np.array(double_adjacent_P3_max)})
        df_double_adj_M4 = pd.DataFrame({"Min": double_adjacent_P4_min, "Max": double_adjacent_P4_max, "Sum": np.array(double_adjacent_P4_min) + np.array(double_adjacent_P4_max)})

        df_double_non_adj_M1 = pd.DataFrame({"Min": double_non_adjacent_P1_min, "Max": double_non_adjacent_P1_max, "Sum": np.array(double_non_adjacent_P1_min) + np.array(double_non_adjacent_P1_max)})
        df_double_non_adj_M2 = pd.DataFrame({"Min": double_non_adjacent_P2_min, "Max": double_non_adjacent_P2_max, "Sum": np.array(double_non_adjacent_P2_min) + np.array(double_non_adjacent_P2_max)})
        df_double_non_adj_M3 = pd.DataFrame({"Min": double_non_adjacent_P3_min, "Max": double_non_adjacent_P3_max, "Sum": np.array(double_non_adjacent_P3_min) + np.array(double_non_adjacent_P3_max)})
        df_double_non_adj_M4 = pd.DataFrame({"Min": double_non_adjacent_P4_min, "Max": double_non_adjacent_P4_max, "Sum": np.array(double_non_adjacent_P4_min) + np.array(double_non_adjacent_P4_max)})


        # Same, but for three strip cases -----------------------------------------------------------------------------------------------
        # Dictionaries to store min, mid, and max charge values for triple adjacent detections
        triple_adjacent_P1_min, triple_adjacent_P1_mid, triple_adjacent_P1_max = [], [], []
        triple_adjacent_P2_min, triple_adjacent_P2_mid, triple_adjacent_P2_max = [], [], []
        triple_adjacent_P3_min, triple_adjacent_P3_mid, triple_adjacent_P3_max = [], [], []
        triple_adjacent_P4_min, triple_adjacent_P4_mid, triple_adjacent_P4_max = [], [], []

        triple_non_adjacent_P1_min, triple_non_adjacent_P1_mid, triple_non_adjacent_P1_max = [], [], []
        triple_non_adjacent_P2_min, triple_non_adjacent_P2_mid, triple_non_adjacent_P2_max = [], [], []
        triple_non_adjacent_P3_min, triple_non_adjacent_P3_mid, triple_non_adjacent_P3_max = [], [], []
        triple_non_adjacent_P4_min, triple_non_adjacent_P4_mid, triple_non_adjacent_P4_max = [], [], []

        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                col_name = f"Q_P{i}s{j}"  # Column name
                v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
                charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

                # Triple adjacent: 3 consecutive strips
                if count == 3 and np.all(np.diff(nonzero_strips) == 1):
                    min_charge, mid_charge, max_charge = np.sort(charges)

                    if i == 1:
                        triple_adjacent_P1_min.append(min_charge)
                        triple_adjacent_P1_mid.append(mid_charge)
                        triple_adjacent_P1_max.append(max_charge)
                    elif i == 2:
                        triple_adjacent_P2_min.append(min_charge)
                        triple_adjacent_P2_mid.append(mid_charge)
                        triple_adjacent_P2_max.append(max_charge)
                    elif i == 3:
                        triple_adjacent_P3_min.append(min_charge)
                        triple_adjacent_P3_mid.append(mid_charge)
                        triple_adjacent_P3_max.append(max_charge)
                    elif i == 4:
                        triple_adjacent_P4_min.append(min_charge)
                        triple_adjacent_P4_mid.append(mid_charge)
                        triple_adjacent_P4_max.append(max_charge)

                # Triple non-adjacent: 3 non-consecutive strips
                if count == 3 and not np.all(np.diff(nonzero_strips) == 1):
                    min_charge, mid_charge, max_charge = np.sort(charges)

                    if i == 1:
                        triple_non_adjacent_P1_min.append(min_charge)
                        triple_non_adjacent_P1_mid.append(mid_charge)
                        triple_non_adjacent_P1_max.append(max_charge)
                    elif i == 2:
                        triple_non_adjacent_P2_min.append(min_charge)
                        triple_non_adjacent_P2_mid.append(mid_charge)
                        triple_non_adjacent_P2_max.append(max_charge)
                    elif i == 3:
                        triple_non_adjacent_P3_min.append(min_charge)
                        triple_non_adjacent_P3_mid.append(mid_charge)
                        triple_non_adjacent_P3_max.append(max_charge)
                    elif i == 4:
                        triple_non_adjacent_P4_min.append(min_charge)
                        triple_non_adjacent_P4_mid.append(mid_charge)
                        triple_non_adjacent_P4_max.append(max_charge)

        # Convert lists to DataFrames for better visualization
        df_triple_adj_M1 = pd.DataFrame({"Min": triple_adjacent_P1_min, "Mid": triple_adjacent_P1_mid, "Max": triple_adjacent_P1_max, "Sum": np.array(triple_adjacent_P1_min) + np.array(triple_adjacent_P1_mid) + np.array(triple_adjacent_P1_max)})
        df_triple_adj_M2 = pd.DataFrame({"Min": triple_adjacent_P2_min, "Mid": triple_adjacent_P2_mid, "Max": triple_adjacent_P2_max, "Sum": np.array(triple_adjacent_P2_min) + np.array(triple_adjacent_P2_mid) + np.array(triple_adjacent_P2_max)})
        df_triple_adj_M3 = pd.DataFrame({"Min": triple_adjacent_P3_min, "Mid": triple_adjacent_P3_mid, "Max": triple_adjacent_P3_max, "Sum": np.array(triple_adjacent_P3_min) + np.array(triple_adjacent_P3_mid) + np.array(triple_adjacent_P3_max)})
        df_triple_adj_M4 = pd.DataFrame({"Min": triple_adjacent_P4_min, "Mid": triple_adjacent_P4_mid, "Max": triple_adjacent_P4_max, "Sum": np.array(triple_adjacent_P4_min) + np.array(triple_adjacent_P4_mid) + np.array(triple_adjacent_P4_max)})

        df_triple_non_adj_M1 = pd.DataFrame({"Min": triple_non_adjacent_P1_min, "Mid": triple_non_adjacent_P1_mid, "Max": triple_non_adjacent_P1_max, "Sum": np.array(triple_non_adjacent_P1_min) + np.array(triple_non_adjacent_P1_mid) + np.array(triple_non_adjacent_P1_max)})
        df_triple_non_adj_M2 = pd.DataFrame({"Min": triple_non_adjacent_P2_min, "Mid": triple_non_adjacent_P2_mid, "Max": triple_non_adjacent_P2_max, "Sum": np.array(triple_non_adjacent_P2_min) + np.array(triple_non_adjacent_P2_mid) + np.array(triple_non_adjacent_P2_max)})
        df_triple_non_adj_M3 = pd.DataFrame({"Min": triple_non_adjacent_P3_min, "Mid": triple_non_adjacent_P3_mid, "Max": triple_non_adjacent_P3_max, "Sum": np.array(triple_non_adjacent_P3_min) + np.array(triple_non_adjacent_P3_mid) + np.array(triple_non_adjacent_P3_max)})
        df_triple_non_adj_M4 = pd.DataFrame({"Min": triple_non_adjacent_P4_min, "Mid": triple_non_adjacent_P4_mid, "Max": triple_non_adjacent_P4_max, "Sum": np.array(triple_non_adjacent_P4_min) + np.array(triple_non_adjacent_P4_mid) + np.array(triple_non_adjacent_P4_max)})

        # ---------------------------------------------------------------------------------------------------------------------------------

        # Create vectors of charge for single detection, double adjacent detections, triple adjacent detections and quadruple detections for each module
        
        # Dictionaries to store charge values for single and quadruple detections
        single_sample_M1, single_sample_M2, single_sample_M3, single_sample_M4 = [], [], [], []
        merged_sample_df = merged_df.copy()
        
        # print(merged_sample_df["processed_tt"])
        merged_sample_df = merged_sample_df[ merged_sample_df["processed_tt"] == 1234 ]
        
        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_sample_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                col_name = f"Q_P{i}s{j}"  # Column name
                v = merged_sample_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
                charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

                # Single detection: exactly 1 strip has charge
                if count == 1:
                    if i == 1:
                        single_sample_M1.append(charges[0])
                    elif i == 2:
                        single_sample_M2.append(charges[0])
                    elif i == 3:
                        single_sample_M3.append(charges[0])
                    elif i == 4:
                        single_sample_M4.append(charges[0])

        # Convert lists to DataFrames for better visualization
        df_single_sample_M1 = pd.DataFrame({"Charge": single_sample_M1})
        df_single_sample_M2 = pd.DataFrame({"Charge": single_sample_M2})
        df_single_sample_M3 = pd.DataFrame({"Charge": single_sample_M3})
        df_single_sample_M4 = pd.DataFrame({"Charge": single_sample_M4})
        
        df_single_sample_M1_sum = df_single_sample_M1["Charge"]
        df_single_sample_M2_sum = df_single_sample_M2["Charge"]
        df_single_sample_M3_sum = df_single_sample_M3["Charge"]
        df_single_sample_M4_sum = df_single_sample_M4["Charge"]
        
        df_single_sample_M1_sum = df_single_sample_M1_sum[ df_single_sample_M1_sum > 0 ]
        df_single_sample_M2_sum = df_single_sample_M2_sum[ df_single_sample_M2_sum > 0 ]
        df_single_sample_M3_sum = df_single_sample_M3_sum[ df_single_sample_M3_sum > 0 ]
        df_single_sample_M4_sum = df_single_sample_M4_sum[ df_single_sample_M4_sum > 0 ]
        
        # Dictionaries to store charge values for single and quadruple detections
        single_M1, single_M2, single_M3, single_M4 = [], [], [], []
        quadruple_M1, quadruple_M2, quadruple_M3, quadruple_M4 = [], [], [], []

        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                col_name = f"Q_P{i}s{j}"  # Column name
                v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0]  # Get active strip indices
                charges = charge_matrix[event_idx, nonzero_strips]  # Get nonzero charges

                # Single detection: exactly 1 strip has charge
                if count == 1:
                    if i == 1:
                        single_M1.append(charges[0])
                    elif i == 2:
                        single_M2.append(charges[0])
                    elif i == 3:
                        single_M3.append(charges[0])
                    elif i == 4:
                        single_M4.append(charges[0])

                # Quadruple detection: all 4 strips have charge
                if count == 4:
                    total_charge = np.sum(charges)
                    if i == 1:
                        quadruple_M1.append(total_charge)
                    elif i == 2:
                        quadruple_M2.append(total_charge)
                    elif i == 3:
                        quadruple_M3.append(total_charge)
                    elif i == 4:
                        quadruple_M4.append(total_charge)

        # Convert lists to DataFrames for better visualization
        df_single_M1 = pd.DataFrame({"Charge": single_M1})
        df_single_M2 = pd.DataFrame({"Charge": single_M2})
        df_single_M3 = pd.DataFrame({"Charge": single_M3})
        df_single_M4 = pd.DataFrame({"Charge": single_M4})

        df_quadruple_M1 = pd.DataFrame({"Total Charge": quadruple_M1})
        df_quadruple_M2 = pd.DataFrame({"Total Charge": quadruple_M2})
        df_quadruple_M3 = pd.DataFrame({"Total Charge": quadruple_M3})
        df_quadruple_M4 = pd.DataFrame({"Total Charge": quadruple_M4})

        # Now create a dataframe of double and triple adjacent detections with the sums of the charges
        df_single_M1_sum = df_single_M1["Charge"]
        df_single_M2_sum = df_single_M2["Charge"]
        df_single_M3_sum = df_single_M3["Charge"]
        df_single_M4_sum = df_single_M4["Charge"]

        df_double_adj_M1_sum = df_double_adj_M1["Sum"]
        df_double_adj_M2_sum = df_double_adj_M2["Sum"]
        df_double_adj_M3_sum = df_double_adj_M3["Sum"]
        df_double_adj_M4_sum = df_double_adj_M4["Sum"]

        df_triple_adj_M1_sum = df_triple_adj_M1["Sum"]
        df_triple_adj_M2_sum = df_triple_adj_M2["Sum"]
        df_triple_adj_M3_sum = df_triple_adj_M3["Sum"]
        df_triple_adj_M4_sum = df_triple_adj_M4["Sum"]

        df_quadruple_M1_sum = df_quadruple_M1["Total Charge"]
        df_quadruple_M2_sum = df_quadruple_M2["Total Charge"]
        df_quadruple_M3_sum = df_quadruple_M3["Total Charge"]
        df_quadruple_M4_sum = df_quadruple_M4["Total Charge"]

        df_total_M1 = pd.concat([df_single_M1_sum, df_double_adj_M1_sum, df_triple_adj_M1_sum, df_quadruple_M1_sum], axis=0)
        df_total_M2 = pd.concat([df_single_M2_sum, df_double_adj_M2_sum, df_triple_adj_M2_sum, df_quadruple_M2_sum], axis=0)
        df_total_M3 = pd.concat([df_single_M3_sum, df_double_adj_M3_sum, df_triple_adj_M3_sum, df_quadruple_M3_sum], axis=0)
        df_total_M4 = pd.concat([df_single_M4_sum, df_double_adj_M4_sum, df_triple_adj_M4_sum, df_quadruple_M4_sum], axis=0)

        df_single = pd.concat([df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum], axis=0)
        df_double_adj = pd.concat([df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum], axis=0)
        df_triple_adj = pd.concat([df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum], axis=0)
        df_quadruple = pd.concat([df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum], axis=0)
        df_total = pd.concat([df_single, df_double_adj, df_triple_adj, df_quadruple], axis=0)


        # PLOT 4. AMOUNT OF STRIPS TRIGGERED --------------------------------------------------------------------------------------------

        # Now count the number of single, double, triple and quadruple detections for each module and histogram it
        # Create vectors of counts for single, double adjacent, triple adjacent, and quadruple detections for each module

        df_single_M1_sum = df_single_M1_sum[ df_single_M1_sum > 0 ]
        df_single_M2_sum = df_single_M2_sum[ df_single_M2_sum > 0 ]
        df_single_M3_sum = df_single_M3_sum[ df_single_M3_sum > 0 ]
        df_single_M4_sum = df_single_M4_sum[ df_single_M4_sum > 0 ]

        df_double_adj_M1_sum = df_double_adj_M1_sum[ df_double_adj_M1_sum > 0 ]
        df_double_adj_M2_sum = df_double_adj_M2_sum[ df_double_adj_M2_sum > 0 ]
        df_double_adj_M3_sum = df_double_adj_M3_sum[ df_double_adj_M3_sum > 0 ]
        df_double_adj_M4_sum = df_double_adj_M4_sum[ df_double_adj_M4_sum > 0 ]

        df_triple_adj_M1_sum = df_triple_adj_M1_sum[ df_triple_adj_M1_sum > 0 ]
        df_triple_adj_M2_sum = df_triple_adj_M2_sum[ df_triple_adj_M2_sum > 0 ]
        df_triple_adj_M3_sum = df_triple_adj_M3_sum[ df_triple_adj_M3_sum > 0 ]
        df_triple_adj_M4_sum = df_triple_adj_M4_sum[ df_triple_adj_M4_sum > 0 ]

        df_quadruple_M1_sum = df_quadruple_M1_sum[ df_quadruple_M1_sum > 0 ]
        df_quadruple_M2_sum = df_quadruple_M2_sum[ df_quadruple_M2_sum > 0 ]
        df_quadruple_M3_sum = df_quadruple_M3_sum[ df_quadruple_M3_sum > 0 ]
        df_quadruple_M4_sum = df_quadruple_M4_sum[ df_quadruple_M4_sum > 0 ]

        # Compute total counts for normalization per module
        total_counts = [
            len(df_single_M1_sum) + len(df_double_adj_M1_sum) + len(df_triple_adj_M1_sum) + len(df_quadruple_M1_sum),
            len(df_single_M2_sum) + len(df_double_adj_M2_sum) + len(df_triple_adj_M2_sum) + len(df_quadruple_M2_sum),
            len(df_single_M3_sum) + len(df_double_adj_M3_sum) + len(df_triple_adj_M3_sum) + len(df_quadruple_M3_sum),
            len(df_single_M4_sum) + len(df_double_adj_M4_sum) + len(df_triple_adj_M4_sum) + len(df_quadruple_M4_sum)
        ]

        # Normalize counts relative to the total counts in each module
        single_counts = [
            len(df_single_M1_sum) / total_counts[0],
            len(df_single_M2_sum) / total_counts[1],
            len(df_single_M3_sum) / total_counts[2],
            len(df_single_M4_sum) / total_counts[3]
        ]
        double_adjacent_counts = [
            len(df_double_adj_M1_sum) / total_counts[0],
            len(df_double_adj_M2_sum) / total_counts[1],
            len(df_double_adj_M3_sum) / total_counts[2],
            len(df_double_adj_M4_sum) / total_counts[3]
        ]
        triple_adjacent_counts = [
            len(df_triple_adj_M1_sum) / total_counts[0],
            len(df_triple_adj_M2_sum) / total_counts[1],
            len(df_triple_adj_M3_sum) / total_counts[2],
            len(df_triple_adj_M4_sum) / total_counts[3]
        ]
        quadruple_counts = [
            len(df_quadruple_M1_sum) / total_counts[0],
            len(df_quadruple_M2_sum) / total_counts[1],
            len(df_quadruple_M3_sum) / total_counts[2],
            len(df_quadruple_M4_sum) / total_counts[3]
        ]

        M1 = [single_counts[0], double_adjacent_counts[0], triple_adjacent_counts[0], quadruple_counts[0]]
        M2 = [single_counts[1], double_adjacent_counts[1], triple_adjacent_counts[1], quadruple_counts[1]]
        M3 = [single_counts[2], double_adjacent_counts[2], triple_adjacent_counts[2], quadruple_counts[2]]
        M4 = [single_counts[3], double_adjacent_counts[3], triple_adjacent_counts[3], quadruple_counts[3]]

        # Define the labels for the detection types
        detection_types = ["Single", "Double\nAdjacent", "Triple\nAdjacent", "Quadruple"]

        # Define colors for each module
        module_colors = ["r", "orange", "g", "b"]  # Module 1: Red, Module 2: Green, Module 3: Blue, Module 4: Magenta

        # Create a single plot for all modules
        fig, ax = plt.subplots(figsize=(5, 4))

        # Width for each bar in the grouped bar plot
        bar_width = 0.2
        x = np.arange(len(detection_types))  # X-axis positions

        # Plot each module's normalized counts
        selected_alpha = 0.6
        ax.bar(x - 1.5 * bar_width, M1, width=bar_width, color=module_colors[0], alpha=selected_alpha, label="Plane 1")
        ax.bar(x - 0.5 * bar_width, M2, width=bar_width, color=module_colors[1], alpha=selected_alpha, label="Plane 2")
        ax.bar(x + 0.5 * bar_width, M3, width=bar_width, color=module_colors[2], alpha=selected_alpha, label="Plane 3")
        ax.bar(x + 1.5 * bar_width, M4, width=bar_width, color=module_colors[3], alpha=selected_alpha, label="Plane 4")

        # Formatting the plot
        ax.set_xticks(x)
        ax.set_xticklabels(detection_types)
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")
        # ax.set_title("Detection Type Distribution per Module (Normalized)")
        ax.legend()
        ax.grid(True, alpha=0.5, zorder=0, axis = "y")

        def custom_formatter(x, _):
            if x >= 0.01:  # 1% or higher
                return f'{x:.0%}'
            else:  # Less than 1%
                return f'{x:.1%}'

        # Apply the custom formatter
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

        plt.tight_layout()
        figure_name = f"barplot_detection_type_distribution_per_module_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

        

        # Parameters
        selected_alpha = 0.7
        bin_number = 100 # 250
        right_lim = 4500
        module_colors = ["r", "orange", "g", "b"]
        n_events = 20000
        bin_edges = np.linspace(0, right_lim, bin_number + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Define detection types and data
        detection_types = ['Total', 'Single', 'Double Adjacent', 'Triple Adjacent', 'Quadruple']
        df_data = [
            [df_total_M1, df_total_M2, df_total_M3, df_total_M4],
            [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum],
            [df_double_adj_M1_sum, df_double_adj_M2_sum, df_double_adj_M3_sum, df_double_adj_M4_sum],
            [df_triple_adj_M1_sum, df_triple_adj_M2_sum, df_triple_adj_M3_sum, df_triple_adj_M4_sum],
            [df_quadruple_M1_sum, df_quadruple_M2_sum, df_quadruple_M3_sum, df_quadruple_M4_sum],
        ]
        singles = [df_single_M1_sum, df_single_M2_sum, df_single_M3_sum, df_single_M4_sum]
        singles_sample = [df_single_sample_M1_sum, df_single_sample_M2_sum, df_single_sample_M3_sum, df_single_sample_M4_sum]
        
        # Step 1: Precompute 1–number_of_particles_bound_up single sums for each module
        hist_basis_all_modules = []  # [ [H1, H2, ..., H6] for each module ]

        number_of_particles_bound_up = 6

        for single_data in singles_sample:
            single_data = np.array(single_data)
            
            # Apply a gaussian filter to smooth the data a little bit
            # single_data = gaussian_filter1d(single_data, sigma=1)
            
            module_hists = []
            for n in range(1, number_of_particles_bound_up + 1):
                samples = np.random.choice(single_data, size=(n_events, n), replace=True).sum(axis=1)
                hist, _ = np.histogram(samples, bins=bin_edges, density=True)
                module_hists.append(hist)
            hist_basis_all_modules.append(np.stack(module_hists, axis=1))  # shape: (bins, 6)


        # Plotting parameters
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        module_labels = ['M1', 'M2', 'M3', 'M4']
        colors = plt.cm.viridis(np.linspace(0, 1, number_of_particles_bound_up))

        # Create one subplot per module
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        for idx, (module_hists, ax) in enumerate(zip(hist_basis_all_modules, axs)):
            for n in range(number_of_particles_bound_up):
                ax.plot(bin_centers, module_hists[:, n], label=f"n={n+1}", color=colors[n])
            
            ax.set_title(f"Module {module_labels[idx]} — Charge Distributions from 1 to {number_of_particles_bound_up} Particles")
            ax.set_ylabel("Normalized Density")
            ax.grid(True)
            ax.legend(fontsize=6, ncol=4, loc='upper right')

        axs[-1].set_xlabel("Summed Charge (fC)")

        plt.suptitle("Generated Histograms Used in NNLS Basis (Per Module)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figure_name = f"basis_{number_of_particles_bound_up}_singles"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

        # Step 2: Plot 5×2 grid
        fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex='col')


        coeff_tables = {dt: pd.DataFrame(index=[f"S{n}" for n in range(1, number_of_particles_bound_up + 1)],
                                        columns=["P1", "P2", "P3", "P4"])
                        for dt in detection_types}

        # Accumulate event-weighted contributions per module
        component_counts = {
            "P1": np.zeros(number_of_particles_bound_up),
            "P2": np.zeros(number_of_particles_bound_up),
            "P3": np.zeros(number_of_particles_bound_up),
            "P4": np.zeros(number_of_particles_bound_up)
        }


        for i, (detection_type, df_group) in enumerate(zip(detection_types, df_data)):
            ax_hist = axes[i, 0]   # Left column: histograms and fit
            ax_scatter = axes[i, 1]  # Right column: scatter plot

            for j, (df_in, color, module) in enumerate(zip(df_group, module_colors, ['P1', 'P2', 'P3', 'P4'])):
                # Real data histogram
                
                df_in = np.asarray(df_in)
                df_in = df_in[np.isfinite(df_in)]  # Remove NaNs and infs
                
                counts_df, _ = np.histogram(df_in, bins=bin_edges, density=False)

                # Basis matrix A for this module (bins × 6)
                A = hist_basis_all_modules[j]

                # Fit: non-negative least squares
                coeffs, _ = nnls(A, counts_df)
                coeff_tables[detection_type].loc[:, module] = coeffs
                model = A @ coeffs  # predicted density
                
                # Get total number of events for that module and detection type
                n_events = len(df_in)
                # Weighted contribution = coeff * n_events
                component_counts[module] += coeffs * n_events
                
                # Plot histogram and model
                ax_hist.plot(bin_centers, counts_df, color=color, linestyle='-', label=f'{module} data')
                ax_hist.plot(bin_centers, model, color=color, linestyle='--', label=f'{module} fit')

                # Coefficients text
                coeff_text = " + ".join([f"{a:.3f}×S{idx+1}" for idx, a in enumerate(coeffs) if a > 0.001])
                ax_hist.text(0.02, 0.95 - j * 0.08, f"{module}: {coeff_text}", transform=ax_hist.transAxes,
                            fontsize=8, color=color, verticalalignment='top')
                
                # Get positive values
                model_pos = model[model > 0]
                counts_pos = counts_df[counts_df > 0]

                # Safely compute min_val only if both arrays are non-empty
                if model_pos.size > 0 and counts_pos.size > 0:
                    min_val = max(np.min(model_pos), np.min(counts_pos))
                else:
                    min_val = 0  # or choose a sensible fallback, e.g., np.nan or skip the plot

                max_val = max(np.max(model) if model.size > 0 else 0,
                              np.max(counts_df) if counts_df.size > 0 else 0)

                # Plot only if data exists
                if model.size > 0 and counts_df.size > 0:
                    ax_scatter.scatter(model, counts_df, label=module, color=color, s=1)
                    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1,
                                    label='y = x' if j == 0 else None)

            # Format histogram panel
            ax_hist.set_title(f"{detection_type}")
            ax_hist.set_ylabel("Density")
            ax_hist.grid(True, alpha=0.5)
            ax_hist.legend(fontsize=8)

            # Format scatter panel
            ax_scatter.set_xscale("log")
            ax_scatter.set_yscale("log")
            ax_scatter.grid(True, alpha=0.5)
            ax_scatter.set_aspect('equal', 'box')
            ax_scatter.set_title("Model vs Data")
            ax_scatter.set_ylabel("Freq. (measured)")

        # Final X labels
        axes[-1, 0].set_xlabel("Charge (fC)")
        axes[-1, 1].set_xlabel("Freq. (fitted model)")

        # Layout & save
        plt.suptitle(f"Charge Distributions and Scatter Model Fit (1–{number_of_particles_bound_up} Singles)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figure_name = f"fit_and_scatter_sum_of_1_to_{number_of_particles_bound_up}_singles"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


        # Normalize the columns to the sum of each column (safely handle zero-division)
        coeff_tables_normalized = coeff_tables.copy()
        for detection_type, df_coeffs in coeff_tables.items():
            col_sums = df_coeffs.sum(axis=0).replace(0, np.nan)  # avoid division by zero
            coeff_tables_normalized[detection_type] = df_coeffs.div(col_sums, axis=1).fillna(0)


        for detection_type, df_coeffs in coeff_tables_normalized.items():
            df_coeffs = df_coeffs.astype(float)
            df_percent = (df_coeffs * 100).round(1)
            print(f"\n===== Coefficients for {detection_type} (in %) =====")
            print(df_percent.to_string())  # Forces output in all environments


        # Module colors
        module_colors = ["r", "orange", "g", "b"]

        # Create a vertical stack of plots: one per detection type
        fig, axes = plt.subplots(len(coeff_tables_normalized), 1, figsize=(8, 14), sharex=True)

        # Loop through each detection type and its coefficients
        for i, (detection_type, df_coeffs) in enumerate(coeff_tables_normalized.items()):
            ax = axes[i]
            df_coeffs = df_coeffs.astype(float)
            df_percent = (df_coeffs * 100).round(1)

            x = np.arange(len(df_percent.index))  # S1 to S6 = positions on x-axis
            width = 0.1

            for j, module in enumerate(df_percent.columns):
            #   ax.bar(x + j * width, df_percent[module], alpha = 0.7, width=width, label=module, color=module_colors[j])
                ax.plot(x + j * width, df_percent[module], alpha = 0.7, label=module, color=module_colors[j])

            ax.set_title(f"{detection_type} - Coefficient Breakdown")
            ax.set_ylabel("Percentage (%)")
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(df_percent.index)
            ax.legend(title="Module", fontsize=8)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.4)

        # Final formatting
        axes[-1].set_xlabel("Summed singles components (S1 to Sn)")
        plt.tight_layout()
        figure_name = "coefficients_barplots_per_type"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

        # Now, multiply each coefficient by the total number of events in the module and the type
        # then sum them all up asnd obtain for each module a coeff. vs total number plot

        components = [f"S{i}" for i in range(1, number_of_particles_bound_up + 1)]
        x = np.arange(len(components))
        width = 0.1
        module_colors = ["r", "orange", "g", "b"]

        fig, ax = plt.subplots(figsize=(8, 5))

        for j, module in enumerate(component_counts.keys()):
        #     ax.bar(x + j * width, component_counts[module] / np.sum( component_counts[module] ), width=width,
        #            label=module, color=module_colors[j], alpha = 0.7)
            ax.plot(x + j * width, component_counts[module] / np.sum( component_counts[module] ),
                label=module, color=module_colors[j])

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(components)
        ax.set_ylabel("Total Events (Weighted by Coefficients)")
        ax.set_title("Total Event Contributions from Sums of 1–6 Singles per Module")
        ax.legend(title="Module")
        ax.grid(True, alpha=0.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.5)
        ax.set_xlim(-0.1, 5)

        plt.tight_layout()
        figure_name = "total_event_contributions_per_component"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


        # Assume coeff_tables_normalized is your dictionary (as printed above)

        # Filter out 'Total' and stack the rest into one DataFrame
        df_mult_fit = ( pd.concat( {k: v for k, v in coeff_tables_normalized.items() if k != 'Total'},
            names=["detection_type", "multiplicity"] ).reset_index().rename(columns={"level_1": "multiplicity"}) )
        
        
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        
        print("---------- Induction section determination using the LUT -------------")
        print("Eventually this part of the code should be done directly with the binary_topology values.")
        
        df_single_M1_sum = df_single_M1_sum[ df_single_M1_sum > 0 ]
        df_single_M2_sum = df_single_M2_sum[ df_single_M2_sum > 0 ]
        df_single_M3_sum = df_single_M3_sum[ df_single_M3_sum > 0 ]
        df_single_M4_sum = df_single_M4_sum[ df_single_M4_sum > 0 ]

        df_double_adj_M1_sum = df_double_adj_M1_sum[ df_double_adj_M1_sum > 0 ]
        df_double_adj_M2_sum = df_double_adj_M2_sum[ df_double_adj_M2_sum > 0 ]
        df_double_adj_M3_sum = df_double_adj_M3_sum[ df_double_adj_M3_sum > 0 ]
        df_double_adj_M4_sum = df_double_adj_M4_sum[ df_double_adj_M4_sum > 0 ]

        df_triple_adj_M1_sum = df_triple_adj_M1_sum[ df_triple_adj_M1_sum > 0 ]
        df_triple_adj_M2_sum = df_triple_adj_M2_sum[ df_triple_adj_M2_sum > 0 ]
        df_triple_adj_M3_sum = df_triple_adj_M3_sum[ df_triple_adj_M3_sum > 0 ]
        df_triple_adj_M4_sum = df_triple_adj_M4_sum[ df_triple_adj_M4_sum > 0 ]

        df_quadruple_M1_sum = df_quadruple_M1_sum[ df_quadruple_M1_sum > 0 ]
        df_quadruple_M2_sum = df_quadruple_M2_sum[ df_quadruple_M2_sum > 0 ]
        df_quadruple_M3_sum = df_quadruple_M3_sum[ df_quadruple_M3_sum > 0 ]
        df_quadruple_M4_sum = df_quadruple_M4_sum[ df_quadruple_M4_sum > 0 ]

        # Compute total counts for normalization per module
        total_counts = [
            len(df_single_M1_sum) + len(df_double_adj_M1_sum) + len(df_triple_adj_M1_sum) + len(df_quadruple_M1_sum),
            len(df_single_M2_sum) + len(df_double_adj_M2_sum) + len(df_triple_adj_M2_sum) + len(df_quadruple_M2_sum),
            len(df_single_M3_sum) + len(df_double_adj_M3_sum) + len(df_triple_adj_M3_sum) + len(df_quadruple_M3_sum),
            len(df_single_M4_sum) + len(df_double_adj_M4_sum) + len(df_triple_adj_M4_sum) + len(df_quadruple_M4_sum)
        ]

        # Normalize counts relative to the total counts in each module
        single_counts = [
            len(df_single_M1_sum) / total_counts[0],
            len(df_single_M2_sum) / total_counts[1],
            len(df_single_M3_sum) / total_counts[2],
            len(df_single_M4_sum) / total_counts[3]
        ]
        double_adjacent_counts = [
            len(df_double_adj_M1_sum) / total_counts[0],
            len(df_double_adj_M2_sum) / total_counts[1],
            len(df_double_adj_M3_sum) / total_counts[2],
            len(df_double_adj_M4_sum) / total_counts[3]
        ]
        triple_adjacent_counts = [
            len(df_triple_adj_M1_sum) / total_counts[0],
            len(df_triple_adj_M2_sum) / total_counts[1],
            len(df_triple_adj_M3_sum) / total_counts[2],
            len(df_triple_adj_M4_sum) / total_counts[3]
        ]
        quadruple_counts = [
            len(df_quadruple_M1_sum) / total_counts[0],
            len(df_quadruple_M2_sum) / total_counts[1],
            len(df_quadruple_M3_sum) / total_counts[2],
            len(df_quadruple_M4_sum) / total_counts[3]
        ]
        
        induction_section_table = {
        "plane": ["M1", "M2", "M3", "M4"],
        "cluster_size_1": [single_counts[0], single_counts[1], single_counts[2], single_counts[3]],
        "cluster_size_2": [double_adjacent_counts[0], double_adjacent_counts[1], double_adjacent_counts[2], double_adjacent_counts[3]],
        "cluster_size_3": [triple_adjacent_counts[0], triple_adjacent_counts[1], triple_adjacent_counts[2], triple_adjacent_counts[3]],
        "cluster_size_4": [quadruple_counts[0], quadruple_counts[1], quadruple_counts[2], quadruple_counts[3]],
        }

        # Create the DataFrame
        induction_section_df = pd.DataFrame(induction_section_table)

        # Print the DataFrame
        print(induction_section_df)
        
        # Load the LUT
        lut_file = "/home/mingo/DATAFLOW_v3/MASTER/ANCILLARY/lut.csv"
        lut_df = pd.read_csv(lut_file)

        # Initialize a list to store the best induction section values for each plane
        best_induction_sections = []

        # Loop through each plane in the induction_section_df
        for _, plane_row in induction_section_df.iterrows():
            # Extract the cluster size data for the current plane
            plane_data = plane_row[["cluster_size_1", "cluster_size_2", "cluster_size_3", "cluster_size_4"]].values

            # Calculate the difference between the plane data and each row in the LUT
            differences = lut_df[["cluster_size_1", "cluster_size_2", "cluster_size_3", "cluster_size_4"]].values - plane_data

            # Compute the squared error for each row in the LUT
            squared_errors = np.sum(differences**2, axis=1)

            # Find the index of the row with the smallest squared error
            best_match_index = np.argmin(squared_errors)

            # Get the corresponding avalanche_width (induction section) from the LUT
            best_induction_section = lut_df.loc[best_match_index, "avalanche_width"]

            # Append the result to the list
            best_induction_sections.append(best_induction_section)

        # Create a new DataFrame to store the results
        df_best_induction_section = pd.DataFrame({
            "plane": induction_section_df["plane"],
            "best_induction_section": best_induction_sections
        })

        # Print the resulting DataFrame
        print(df_best_induction_section)
        
        # Create new columns called PX_induction_section with th e best induction section value
        for i in range(1, 5):
            df[f"P{i}_induction_section"] = best_induction_sections[i - 1]


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if crosstalk_probability:
        
        print("----------------------------------------------------------------------")
        print("------------ Crosstalk probability respect the charge ----------------")
        print("----------------------------------------------------------------------")
        
        n_bins = 100
        right_lim = 1400 # 1250
        crosstalk_limit = 1 #2.6
        charge_vector = np.linspace(crosstalk_limit, right_lim, n_bins)

        FEE_calibration = {
            "Width": [
                0.0000001, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
                300, 310, 320, 330, 340, 350, 360, 370, 380, 390
            ],
            "Fast Charge": [
                4.0530E+01, 2.6457E+02, 4.5081E+02, 6.0573E+02, 7.3499E+02, 8.4353E+02,
                9.3562E+02, 1.0149E+03, 1.0845E+03, 1.1471E+03, 1.2047E+03, 1.2592E+03,
                1.3118E+03, 1.3638E+03, 1.4159E+03, 1.4688E+03, 1.5227E+03, 1.5779E+03,
                1.6345E+03, 1.6926E+03, 1.7519E+03, 1.8125E+03, 1.8742E+03, 1.9368E+03,
                2.0001E+03, 2.0642E+03, 2.1288E+03, 2.1940E+03, 2.2599E+03, 2.3264E+03,
                2.3939E+03, 2.4625E+03, 2.5325E+03, 2.6044E+03, 2.6786E+03, 2.7555E+03,
                2.8356E+03, 2.9196E+03, 3.0079E+03, 3.1012E+03
            ]
        }

        # Create the DataFrame
        FEE_calibration = pd.DataFrame(FEE_calibration)
        
        # Convert to NumPy arrays for interpolation
        width_table = FEE_calibration['Width'].to_numpy()
        fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()
        # Create a cubic spline interpolator
        cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

        def interpolate_fast_charge(width):
            """
            Interpolates the Fast Charge for given Width values using cubic spline interpolation.
            Parameters:
            - width (float or np.ndarray): The Width value(s) to interpolate in ns.
            Returns:
            - float or np.ndarray: The interpolated Fast Charge value(s) in fC.
            """
            width = np.asarray(width)  # Ensure input is a NumPy array
            # Keep zero values unchanged
            result = np.where(width == 0, 0, cs(width))
            return result

        df_list_OG = [df]  # Adjust delimiter if needed


        # NO CROSSTALK SECTION --------------------------------------------------------------------------

        # Read and concatenate all files
        df_list = df_list_OG.copy()
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)

        columns_to_keep = [f"Q_P{i}s{j}" for i in range(1, 5) for j in range(1, 5)]
        merged_df = merged_df[columns_to_keep]

        # For all the columns apply the calibration and not change the name of the columns
        for col in merged_df.columns:
            merged_df[col] = interpolate_fast_charge(merged_df[col])

        # Initialize dictionaries to store charge distributions
        singles = {f'single_P{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}

        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                col_name = f"Q_P{i}s{j}"  # Column name
                v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0] + 1  # Get active strip indices (1-based)
                charges = charge_matrix[event_idx, nonzero_strips - 1]  # Get nonzero charges

                # Single detection
                if count == 1:
                    key = f'single_P{i}_s{nonzero_strips[0]}'
                    singles[key].append((charges[0],))

        # Convert results to DataFrames
        df_singles = {k: pd.DataFrame(v, columns=["Charge1"]) for k, v in singles.items()}

        # Assuming df_singles and crosstalk limit are already defined
        bin_edges = charge_vector
        histograms_no_crosstalk = {}

        print("Histograms for no crosstalk")
        for m in range(1, 5):
            for s in range(1, 5):
                key = f"P{m}_s{s}"
                data = df_singles[f"single_P{m}_s{s}"]['Charge1'].values
                hist, _ = np.histogram(data, bins=bin_edges)
                histograms_no_crosstalk[key] = hist


        # YES CROSSTALK SECTION -------------------------------------------------------------------------

        # Read and concatenate all files
        df_list = df_list_OG.copy()
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.drop_duplicates(inplace=True)

        columns_to_keep = [f"Q_P{i}s{j}_with_crstlk" for i in range(1, 5) for j in range(1, 5)]
        merged_df = merged_df[columns_to_keep]

        # For all the columns apply the calibration and not change the name of the columns
        for col in merged_df.columns:
            merged_df[col] = interpolate_fast_charge(merged_df[col])

        # Initialize dictionaries to store charge distributions
        singles = {f'single_P{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}

        # Loop over modules
        for i in range(1, 5):
            charge_matrix = np.zeros((len(merged_df), 4))  # Stores strip-wise charges for this module

            for j in range(1, 5):  # Loop over strips
                col_name = f"Q_P{i}s{j}_with_crstlk"  # Column name
                v = merged_df[col_name].fillna(0).to_numpy()  # Ensure no NaNs
                charge_matrix[:, j - 1] = v  # Store strip charge

            # Classify events based on strip charge distribution
            nonzero_counts = (charge_matrix > 0).sum(axis=1)  # Count nonzero strips per event

            for event_idx, count in enumerate(nonzero_counts):
                nonzero_strips = np.where(charge_matrix[event_idx, :] > 0)[0] + 1  # Get active strip indices (1-based)
                charges = charge_matrix[event_idx, nonzero_strips - 1]  # Get nonzero charges

                # Single detection
                if count == 1:
                    key = f'single_P{i}_s{nonzero_strips[0]}'
                    singles[key].append((charges[0],))

        # Convert results to DataFrames
        df_singles = {k: pd.DataFrame(v, columns=["Charge1"]) for k, v in singles.items()}

        # Assuming df_singles and crosstalklimit are already defined
        bin_edges = charge_vector
        histograms_yes_crosstalk = {}

        print("Histograms for yes crosstalk")
        for m in range(1, 5):
            for s in range(1, 5):
                key = f"P{m}_s{s}"
                data = df_singles[f"single_P{m}_s{s}"]['Charge1'].values
                hist, _ = np.histogram(data, bins=bin_edges)
                histograms_yes_crosstalk[key] = hist

        def compute_fraction_and_uncertainty(charge_edges, hist_no, hist_yes):
            fraction_dict = {}
            uncertainty_dict = {}

            # We remove the last edge to match the histogram "counts" length:
            x_vals = charge_edges[:-1]

            for key in hist_no:
                # Just for clarity
                Nn = hist_no[key]   # 'no crosstalk' counts
                Ny = hist_yes[key]  # 'yes crosstalk' counts
                D = Nn + Ny         # denominator

                # Fraction
                with np.errstate(divide='ignore', invalid='ignore'):
                    f = (Nn - Ny) / D
                    # If D=0 => f -> undefined => set to 0
                    f[np.isnan(f)] = 0  

                # Poisson errors for counts
                sigma_Nn = np.sqrt(Nn)
                sigma_Ny = np.sqrt(Ny)

                # Partial derivatives
                # df/dNn =  2*Ny / D^2
                # df/dNy = -2*Nn / D^2
                with np.errstate(divide='ignore', invalid='ignore'):
                    df_dNn = 2 * Ny / (D**2)
                    df_dNy = -2 * Nn / (D**2)

                    # Total variance
                    sigma_f_sq = (df_dNn**2) * (sigma_Nn**2) + (df_dNy**2) * (sigma_Ny**2)
                    # If D=0, that might lead to NaN
                    sigma_f = np.sqrt(sigma_f_sq)
                    sigma_f[np.isnan(sigma_f)] = 0

                fraction_dict[key] = f
                uncertainty_dict[key] = sigma_f

            return x_vals, fraction_dict, uncertainty_dict

        # ---------------------------------------------------------------------
        # Example usage to produce the 4x4 grid of plots:
        # ---------------------------------------------------------------------
        
        
        # # 1. Compute fraction + uncertainty
        # x_vals, fraction_hist, frac_err = compute_fraction_and_uncertainty(
        #     charge_vector,  # bin edges from your snippet
        #     histograms_no_crosstalk,
        #     histograms_yes_crosstalk
        # )

        # 2. Plot in 4x4
        # fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
        # fig.suptitle(f"Crosstalk probability with Poisson Error Bands, mingo0{station}", fontsize=14)

        # for m in range(1, 5):
        #     for s in range(1, 5):
        #         ax = axs[m-1, s-1]
        #         key = f"P{m}_s{s}"

        #         y_vals = fraction_hist[key]
        #         y_err  = frac_err[key]

        #         ax.plot(x_vals, y_vals, label=key)
        #         ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, alpha=0.3)
        #         ax.set_ylim(0, 1)         # Because fraction can be negative if N_yes > N_no
        #         ax.set_title(key)
        #         ax.grid(True)

        # # Better spacing
        # for ax in axs[-1, :]:
        #     ax.set_xlabel("Charge")
        # for ax in axs[:, 0]:
        #     ax.set_ylabel("Probability")

        # # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.tight_layout()
        # figure_name = f"crosstalk_probability_mingo0{station}"
        # if save_plots:
        #     name_of_file = figure_name
        #     final_filename = f'{fig_idx}_{name_of_file}.png'
        #     fig_idx += 1
        #     save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        #     plot_list.append(save_fig_path)
        #     plt.savefig(save_fig_path, format='png')
        # if show_plots: plt.show()
        # plt.close()


        # Fit a sigmoidal and store the fitting values to compare with temperature, etc.

        

        # --- 1. Define 3-parameter sigmoid (bounded to [0,1]) ---
        def sigmoid_3p(x, x0, k):
            exp_arg = np.clip(-k * (x - x0), -500, 500)
            return 1 / (1 + np.exp(exp_arg))
        
        # --- 2. Compute fractions and uncertainties ---
        x_vals, fraction_hist, frac_err = compute_fraction_and_uncertainty(
            charge_vector,
            histograms_no_crosstalk,
            histograms_yes_crosstalk
        )

        fit_results = []

        # --- 3. Plot and fit ---
        fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
        fig.suptitle(f"Crosstalk probability with Sigmoid Fit, mingo0{station}", fontsize=14)

        for m in range(1, 5):
            for s in range(1, 5):
                ax = axs[m-1, s-1]
                key = f"P{m}_s{s}"
                y_vals_full = fraction_hist[key]
                y_err_full  = frac_err[key]

                # Restrict to x in [200, 1300]
                domain_mask = (x_vals >= 200) & (x_vals <= 1300)
                x_domain = x_vals[domain_mask]
                y_vals = y_vals_full[domain_mask]
                y_err  = y_err_full[domain_mask]

                # Restrict further to transition region: y in [0.05, 0.95]
                trans_mask = (y_vals > 0.05) & (y_vals < 0.95)
                x_fit = x_domain[trans_mask]
                y_fit = y_vals[trans_mask]
                y_err_fit = y_err[trans_mask]

                # Skip if too little data
                if len(x_fit) < 5:
                    popt = [np.nan, np.nan]
                else:
                    # Initial guess
                    x0_guess = x_fit[np.argmin(np.abs(y_fit - 0.5))]
                    k_guess = 0.05  # shallow initial slope

                    try:
                        popt, pcov = curve_fit(sigmoid_3p, x_fit, y_fit, p0=[x0_guess, k_guess],
                                            sigma=np.where(y_err_fit == 0, 1e-6, y_err_fit),
                                            absolute_sigma=True, maxfev=10000)
                    except RuntimeError:
                        popt = [np.nan, np.nan]

                # --- 4. Store fit results ---
                fit_results.append({
                    'key': key,
                    'x0': popt[0],
                    'k': popt[1]
                })

                # --- 5. Plot raw data ---
                ax.plot(x_vals, y_vals_full, label=key)
                ax.fill_between(x_vals, y_vals_full - y_err_full, y_vals_full + y_err_full, alpha=0.3)
                ax.set_ylim(0, 1)
                ax.set_title(key)
                ax.grid(True)

                # --- 6. Plot sigmoid fit ---
                if not np.any(np.isnan(popt)):
                    x_dense = np.linspace(200, 1300, 300)
                    y_dense = sigmoid_3p(x_dense, *popt)
                    ax.plot(x_dense, y_dense, 'r--', label='Sigmoid fit')

        # --- 7. Axis labels and layout ---
        for ax in axs[-1, :]:
            ax.set_xlabel("Charge")
        for ax in axs[:, 0]:
            ax.set_ylabel("Probability")
        axs[0, 0].legend()

        plt.tight_layout()
        figure_name = f"crosstalk_probability_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()

        # --- 8. Save fit results ---
        df_cross_fit = pd.DataFrame(fit_results)
        print(df_cross_fit)


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if n_study_fit:
        
        print("----------------------------------------------------------------------")
        print("------------------------ The n study fit -----------------------------")
        print("----------------------------------------------------------------------")
        
        first_approach = True
        
        if first_approach:
            # Fit the histogram of df["theta"] values per each definitive_tt to the a function which is:
            # F_ij(theta, phi) = A_ij * cos(theta)^n_ij * R_ij(theta, phi)
            # where R_ij is the response function of a telescope made of square planes of 300 mm
            # length and at a distance of abs(z_position[i] - z_positions[j]). First simulate the R_ij for
            # the cases 13, 24, 14, 23. Note that definitive_tt = 123 has the same response as definitive_tt
            # = 13, because what count are the top and bottom planes in each case. Simulate and plot the
            # R_ij for those cases, then fit the theta to F_ij, and plot it. Obtain an A_ij and n_ij
            # per each case. Then sum all to obtain a realistic total flux crossing the detector. Go!
            
            use_only_theta = True
            if use_only_theta:
                
                
                # STEP 1: response function calculation

                # Simulation parameters
                N = 3000000  # number of events per configuration
                PLANE_SIZE = 300  # mm, square plane side
                HALF_SIZE = PLANE_SIZE / 2

                pairs = {
                    '13': (1, 3),
                    '24': (2, 4),
                    '14': (1, 4),
                    '23': (2, 3),
                    
                    '123': (1, 3),
                    '234': (2, 4),
                    '1234': (1, 4),
                }

                def simulate_telescope_response(z1, z2, N=N):
                    """Simulate direction distribution of tracks crossing square planes at z1 and z2."""
                    # Random (x, y) points on both planes
                    x1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    x2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)

                    dx = x2 - x1
                    dy = y2 - y1
                    dz = z2 - z1

                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    theta = np.arccos(np.abs(dz) / r)
                    phi = np.arctan2(dy, dx)

                    return theta, phi

                # Simulate and histogram theta density
                theta_bins = np.linspace(0, np.pi / 2, 100)
                theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
                response_histograms = {}

                for label, (i, j) in pairs.items():
                    z1, z2 = z_positions[i-1], z_positions[j-1]
                    theta, phi = simulate_telescope_response(z1, z2)
                    hist, _ = np.histogram(theta, bins=theta_bins, density=True)
                    response_histograms[label] = hist

                # Plot simulated response densities
                fig, ax = plt.subplots(figsize=(8, 6))
                for label in response_histograms:
                    ax.plot(theta_centers * 180 / np.pi, response_histograms[label], label=f'{label}')
                ax.set_xlabel(r'$\theta$ [deg]')
                ax.set_ylabel(r'Normalized density')
                ax.set_title('Simulated Angular Response $R_{ij}(\\theta)$ from Geometry')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                if save_plots:
                    final_filename = f'{fig_idx}_simulated_response_function.png'
                    fig_idx += 1
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    plt.savefig(save_fig_path, format='png')
                if show_plots:
                    plt.show()
                plt.close()


                # STEP 2: fit the theta to F_ij, and plot it

                # Assuming df is available and contains 'theta' and 'definitive_tt'

                
                
                # Reuse the existing response_histograms from earlier simulation
                fit_results_real = {}

                # Loop over response functions and fit real data per definitive_tt
                for label, response in response_histograms.items():
                    if label not in df['definitive_tt'].astype(str).unique():
                        continue

                    # Extract real theta data for this topology
                    df_tt = df[df['definitive_tt'].astype(str) == label]
                    theta_data = df_tt['theta'].dropna().values
                    if len(theta_data) < 20:
                        continue

                    # Histogram real data
                    data_hist, _ = np.histogram(theta_data, bins=theta_bins, density=False)

                    # Create interpolated response function
                    R_interp = interp1d(theta_centers, response, kind='linear', bounds_error=False, fill_value=0)

                    # Fit function: F(theta) = A * cos(theta)^n * R_ij(theta)
                    def fit_func(theta, A, n, B, C):
                        return (A * np.cos(theta) ** 2 + B * np.cos(theta) ** n + C) * R_interp(theta)

                    try:
                        popt, _ = curve_fit(fit_func, theta_centers, data_hist,\
                            p0=[1.0, 2.0, 1.0, 1.0], bounds=([0, 0, 0, 0], [np.inf, 100, np.inf, np.inf]))
                        A_fit, n_fit, B_fit, C_fit = popt
                        fit_results_real[label] = (A_fit, n_fit, B_fit, C_fit)

                        # Plot
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(theta_centers * 180 / np.pi, data_hist, label='Data', color='black')
                        ax.plot(theta_centers * 180 / np.pi, fit_func(theta_centers, *popt), '--', \
                            label=f'Fit: A={A_fit:.2f}, n={n_fit:.2f}, B={B_fit:.2f}, C={C_fit:.2f}', color='red')
                        ax.set_xlabel(r'$\theta$ [deg]')
                        ax.set_ylabel('Normalized density')
                        ax.set_title(f'Fit of $F_{{{label}}}(\\theta)$ to Real Data')
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()
                        plt.show()
                    except RuntimeError:
                        print(f"Fit failed for {label}")

                fit_results_real

            else:
                

                # Define the simulation again for theta and phi
                N = 1000000
                PLANE_SIZE = 300  # mm
                HALF_SIZE = PLANE_SIZE / 2

                pairs = {
                    '13': (1, 3),
                    '24': (2, 4),
                    '14': (1, 4),
                    '23': (2, 3),
                    
                    '123': (1, 3),
                    '234': (2, 4),
                    '1234': (1, 4),
                }

                

                # Set z_positions for the 4 planes (in mm)
                
                # Reuse definitions
                PLANE_SIZE = 300  # mm
                HALF_SIZE = PLANE_SIZE / 2
                N = 2000000

                # Define angular binning
                theta_bins = np.linspace(0, np.pi / 2.5, 20)
                phi_bins = np.linspace(-np.pi, np.pi, 20)
                theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
                phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
                THETA_MESH, PHI_MESH = np.meshgrid(theta_centers, phi_centers, indexing='ij')


                # Simulate response in 2D (theta, phi)
                def simulate_theta_phi(z1, z2, N=N):
                    x1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y1 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    x2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                    y2 = np.random.uniform(-HALF_SIZE, HALF_SIZE, N)
                
                    dx = x2 - x1
                    dy = y2 - y1
                    dz = z2 - z1
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    theta = np.arccos(np.abs(dz) / r)
                    phi = np.arctan2(dy, dx)
                    return theta, phi

                # Simulate and construct 2D histograms
                response_2d = {}
                response_functions_2d = {}
                for label, (i, j) in pairs.items():
                    z1, z2 = z_positions[i-1], z_positions[j-1]
                    theta_sim, phi_sim = simulate_theta_phi(z1, z2)
                    H, _, _ = np.histogram2d(theta_sim, phi_sim, bins=[theta_bins, phi_bins], density=True)
                    response_2d[label] = H
                    response_functions_2d[label] = H

                # Plot simulated 2D response R_ij(theta, phi)
                fig, axes = plt.subplots(2, 4, subplot_kw={'projection': 'polar'}, figsize=(18, 10))
                axes = axes.flatten()
                for idx, (label, R) in enumerate(response_2d.items()):
                    ax = axes[idx]
                    pcm = ax.pcolormesh(PHI_MESH, THETA_MESH, R, shading='auto', cmap='viridis')
                    ax.set_title(f'Response R_{{{label}}}($\\theta$, $\\phi$)')
                    fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.1)

                for ax in axes[len(response_2d):]:
                    ax.axis('off')

                plt.suptitle('Simulated 2D Response Functions $R_{ij}(\\theta, \\phi)$', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


                # Now proceed to fitting with updated response_functions_2d
                fit_results = {}

                for label, interp_R in {
                    l: RegularGridInterpolator((theta_centers, phi_centers), response_functions_2d[l], bounds_error=False, fill_value=0)
                    for l in response_functions_2d
                }.items():
                    if label not in df['definitive_tt'].astype(str).unique():
                        continue

                    df_tt = df[df['definitive_tt'].astype(str) == label]
                    theta_data = df_tt['theta'].dropna().values
                    phi_data = df_tt['phi'].dropna().values

                    if len(theta_data) < 50:
                        continue

                    data_hist, _, _ = np.histogram2d(theta_data, phi_data, bins=[theta_bins, phi_bins], density=False)
                    theta_phi_coords = np.array(np.meshgrid(theta_centers, phi_centers, indexing='ij'))
                    theta_phi_flat = np.vstack([theta_phi_coords[0].ravel(), theta_phi_coords[1].ravel()]).T
                    R_vals = interp_R(theta_phi_flat).reshape(len(theta_centers), len(phi_centers))

                    def fit_func(grid_flat, A, n, B):
                        theta_flat = grid_flat[:, 0]
                        phi_flat = grid_flat[:, 1]
                        R_eval = interp_R(np.vstack([theta_flat, phi_flat]).T)
                        return (A * np.cos(theta_flat) ** n + B) * R_eval

                    data_flat = data_hist.ravel()
                    coords_flat = theta_phi_flat
                    mask = data_flat > 0

                    try:
                        popt, _ = curve_fit(fit_func, coords_flat[mask], data_flat[mask], p0=[1.0, 2.0, 1.0], bounds=([0, 0, 0], [np.inf, 100, np.inf]))
                        fit_results[label] = popt

                        fit_vals = fit_func(coords_flat, *popt).reshape(data_hist.shape)

                        # Plotting
                        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(14, 6))
                        PHI_MESH, THETA_MESH = np.meshgrid(phi_centers, theta_centers, indexing='ij')

                        pcm1 = axes[0].pcolormesh(PHI_MESH, THETA_MESH, data_hist.T, shading='auto', cmap='viridis')
                        axes[0].set_title(f'Data: {label}')
                        fig.colorbar(pcm1, ax=axes[0])

                        pcm2 = axes[1].pcolormesh(PHI_MESH, THETA_MESH, fit_vals.T, shading='auto', cmap='viridis')
                        axes[1].set_title(f'Fit: A={popt[0]:.2f}, n={popt[1]:.2f}, B={popt[2]:.2f}')
                        fig.colorbar(pcm2, ax=axes[1])

                        plt.tight_layout()
                        plt.show()

                    except RuntimeError:
                        fit_results[label] = None
        
        else:

            # Okey, so look out: i want you to change this to take a slightly different approach: instead of 
            # calculating the response functions just considerng two planes and thats all, you are going to 
            # do the following: generate tracks in a uniform distribution in zenith and azimuth and in a plane 
            # above the first plane of the detector (z_positions[0]) but from -1000 to 1000, then you are going 
            # to check which planes the track crosses, according to its theta, phi, x, y and the z_positions of 
            # the planes, which are in x and y between -150 and 150. Then you are going to apply a efficiency, 
            # which will be 0.95 * theta**0.001, to determine if the plane is detected or not for a certain plane.
            # So in the end each track will have a crossing_tt and a measured_tt (when applying the efficiency 
            # over crossing_tt), you calculate, from the total tracks in a bin of theta, how many where detected 
            # in each combination of planes, so you get a response function but taking account efficiencies. So 
            # in this case, 123 is not the same as 13, because for an event to be 13 then it must NOT be detected in plane 2,

            

            # Simulation parameters
            N = 50_000_000
            PLANE_SIZE = 300  # mm, square plane side
            HALF_SIZE = PLANE_SIZE / 2
            EFFICIENCY_FUNC = lambda theta: 0.93 + 0.02 * theta ** 0.9  # simplified efficiency

            # Detector configuration
            n_planes = len(z_positions)

            # Define angular and spatial domain of generation
            z_gen = 0  # generation plane
            x_gen = np.random.uniform(-HALF_SIZE * 5, HALF_SIZE * 5, N)
            y_gen = np.random.uniform(-HALF_SIZE * 5, HALF_SIZE * 5, N)

            n = 2
            u = np.random.uniform(0, 1, N)
            theta_gen = np.arccos((1 - u) ** (1 / (n + 1)))
            # theta_gen = np.arccos(np.random.uniform(0, 1, N))  # uniform in cosθ

            phi_gen = np.random.uniform(-np.pi, np.pi, N)

            # Compute direction vector components
            dx = np.sin(theta_gen) * np.cos(phi_gen)
            dy = np.sin(theta_gen) * np.sin(phi_gen)
            dz = np.cos(theta_gen)

            # Track intersection with each plane
            crossing_tt = []
            measured_tt = []

            for i_plane, z_det in enumerate(z_positions):
                t = (z_det - z_gen) / dz
                x_det = x_gen + t * dx
                y_det = y_gen + t * dy
                inside = (np.abs(x_det) <= HALF_SIZE) & (np.abs(y_det) <= HALF_SIZE)
                if i_plane == 0:
                    crossing_mask = inside.astype(int)
                else:
                    crossing_mask = np.vstack([crossing_mask, inside.astype(int)])

            # Transpose to get shape (N, n_planes)
            crossing_mask = crossing_mask.T

            for i in range(N):
                # print a message each 100000 events
                if i % 100000 == 0:
                    print(f"Processing event {i} of {N}... {i/N*100:.1f} % complete")
                
                crossing_planes = [str(j + 1) for j in range(n_planes) if crossing_mask[i, j]]
                measured_planes = []
                for j in range(n_planes):
                    if crossing_mask[i, j] and np.random.rand() < EFFICIENCY_FUNC(theta_gen[i]):
                        measured_planes.append(str(j + 1))
                if crossing_planes:
                    crossing_tt.append(''.join(crossing_planes))
                else:
                    crossing_tt.append('0')
                if measured_planes:
                    measured_tt.append(''.join(measured_planes))
                else:
                    measured_tt.append('0')

            # Store by configuration
            theta_bins = np.linspace(0, np.pi / 2, 50)
            theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
            response_histograms = defaultdict(lambda: np.zeros(len(theta_centers)))

            # Build histograms for each measured_tt
            for i in range(N):
                label = measured_tt[i]
                if label != '0':
                    bin_idx = np.digitize(theta_gen[i], theta_bins) - 1
                    if 0 <= bin_idx < len(theta_centers):
                        response_histograms[label][bin_idx] += 1

            # Normalize each histogram
            for label in response_histograms:
                total = np.sum(response_histograms[label])
                if total > 0:
                    response_histograms[label] /= total


            # Convert response_histograms to a DataFrame for saving as CSV
            response_df = pd.DataFrame.from_dict(response_histograms, orient='index', columns=theta_centers)
            response_df.to_csv(os.path.join(base_directories["figure_directory"], 'response_histograms.csv'))


            read_response = True
            if read_response:
                # Load response_histograms from the csv file
                response_df = pd.read_csv(os.path.join(base_directories["figure_directory"], 'response_histograms.csv'), index_col=0)



            # Plot simulated response densities

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = cm.viridis(np.linspace(0, 1, len(response_histograms)))
            for idx, label in enumerate(sorted(response_histograms.keys())):
                if label in ['0', '', '1', '2', '3', '4', '14']:
                    continue
                ax.plot(theta_centers * 180 / np.pi, response_histograms[label], label=label, color=colors[idx])

            ax.set_xlabel(r'$\theta$ [deg]')
            ax.set_ylabel(r'Normalized density')
            ax.set_title('Simulated Angular Response $R_{tt}(\\theta)$ with Efficiency and Plane Logic')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.tight_layout()
            if save_plots:
                final_filename = f'{fig_idx}_sim_ang_fit_response_{label}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


            # STEP 2: Fit the theta distribution to F_ij(theta) model using the simulated R_ij

            # Store fit results
            fit_results_real = {}

            # Perform fitting
            for label, response in response_histograms.items():
                if label not in df['definitive_tt'].astype(str).unique():
                    continue

                df_tt = df[df['definitive_tt'].astype(str) == label]
                theta_data = df_tt['theta'].dropna().values
                if len(theta_data) < 30:
                    continue

                data_hist, _ = np.histogram(theta_data, bins=theta_bins, density=False)
                R_interp = interp1d(theta_centers, response, kind='linear', bounds_error=False, fill_value=0)
                
                # Fit model: (A * cos² + B * cos^n + C) * R(θ)
                def fit_func(theta, A, n, B, C):
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    # return (A * cos_t**2 + B * cos_t**n + C) * R_interp(theta)
                    # return (A * cos_t**n + B) * R_interp(theta) * sin_t
                    # return (A * cos_t**n) * R_interp(theta) * sin_t
                    return (A * cos_t**n) * R_interp(theta)

                try:
                    popt, _ = curve_fit(fit_func, theta_centers, data_hist, p0=[1.0, 2.0, 1.0, 1.0],
                                        bounds=([0, 0, 0, 0], [np.inf, 100, np.inf, np.inf]))
                    fit_results_real[label] = (popt, data_hist, response)
                except RuntimeError:
                    fit_results_real[label] = None

            # Plotting results in subplots
            n = len(fit_results_real)
            ncols = 3
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

            for idx, (label, result) in enumerate(fit_results_real.items()):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                if result is None:
                    ax.set_visible(False)
                    continue

                popt, data_hist, response = result
                R_interp = interp1d(theta_centers, response, kind='linear', bounds_error=False, fill_value=0)

                def fit_func(theta, A, n, B, C):
                    cos_t = np.cos(theta)
                    return (A * cos_t**n) * R_interp(theta)

                ax.plot(theta_centers * 180 / np.pi, data_hist, label='Data', color='black')
                ax.plot(theta_centers * 180 / np.pi, fit_func(theta_centers, *popt), '--', color='red',
                        label=fr'Fit: $A$={popt[0]:.2f}, $n$={popt[1]:.2f}')
                ax.set_xlabel(r'$\theta$ [deg]')
                ax.set_ylabel('Counts per bin')
                ax.set_title(f'Topology {label}')
                ax.legend()
                ax.grid(True)

            # Hide unused axes
            for i in range(n, nrows * ncols):
                row, col = divmod(i, ncols)
                axes[row][col].set_visible(False)

            plt.tight_layout()
            if save_plots:
                final_filename = f'{fig_idx}_fit_response_all.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    
    
    if topology_plots:

        for i in range(1, 5):
            cols = [f"Q_P{i}s{j}" for j in range(1, 5)]
            q = df[cols].copy()
            
            # Basic counts
            df[f"cluster_size_{i}"] = (q > 0).sum(axis=1)
            df[f"cluster_charge_{i}"] = q.sum(axis=1)
            df[f"cluster_max_q_{i}"] = q.max(axis=1)
            df[f"cluster_q_ratio_{i}"] = df[f"cluster_max_q_{i}"] / df[f"cluster_charge_{i}"].replace(0, np.nan)

            # Charge-weighted barycenter
            strip_positions = np.array([1, 2, 3, 4])
            weighted_sum = (q * strip_positions).sum(axis=1)
            df[f"cluster_barycenter_{i}"] = weighted_sum / df[f"cluster_charge_{i}"].replace(0, np.nan)

            # Charge-weighted RMS
            barycenter = df[f"cluster_barycenter_{i}"]
            squared_diff = (strip_positions.reshape(1, -1) - barycenter.values[:, None]) ** 2
            weighted_squared = q.values * squared_diff
            rms = np.sqrt( abs( weighted_squared.sum(axis=1) / df[f"cluster_charge_{i}"].replace(0, np.nan) ) )
            df[f"cluster_rms_{i}"] = rms

        # Aggregate over all modules (i = 1 to 4)
        cluster_size_cols = [f"cluster_size_{i}" for i in range(1, 5)]
        cluster_charge_cols = [f"cluster_charge_{i}" for i in range(1, 5)]
        cluster_rms_cols = [f"cluster_rms_{i}" for i in range(1, 5)]
        cluster_barycenter_cols = [f"cluster_barycenter_{i}" for i in range(1, 5)]

        # Mean cluster size
        df["mean_cluster_size"] = df[cluster_size_cols].mean(axis=1)

        # Mean cluster size weighted by module charge
        charge_sum = df[cluster_charge_cols].sum(axis=1).replace(0, np.nan)
        weighted_cluster_size = (df[cluster_size_cols].values * df[cluster_charge_cols].values).sum(axis=1)
        df["mean_cluster_size_weighted_q"] = weighted_cluster_size / charge_sum

        # Total cluster charge
        df["total_cluster_charge"] = df[cluster_charge_cols].sum(axis=1)

        # Maximum RMS
        df["max_cluster_rms"] = df[cluster_rms_cols].max(axis=1)

        # Minimum barycenter (across modules)
        df["min_cluster_barycenter"] = df[cluster_barycenter_cols].min(axis=1)

        # Charge-weighted global barycenter across modules
        numerator = np.zeros(len(df))
        for i in range(1, 5):
            q = df[f"cluster_charge_{i}"]
            bc = df[f"cluster_barycenter_{i}"]
            numerator += q * bc


        # Some plots of these calculations --------------------------------------------
        df["weighted_global_barycenter"] = numerator / charge_sum
        print(df.columns)

        # --- Collect relevant column groups ---
        per_module_cols = []
        for i in range(1, 5):
            per_module_cols += [
                f"cluster_size_{i}",
                f"cluster_charge_{i}",
                # f"cluster_max_q_{i}",
                f"cluster_q_ratio_{i}",
                f"cluster_barycenter_{i}",
                f"cluster_rms_{i}",
                # f"Q_{i}",
                # f"avalanche_{i}",
                # f"streamer_{i}",
            ]

        event_level_cols = [
            # "Q_event",
            "mean_cluster_size",
            "mean_cluster_size_weighted_q",
            "total_cluster_charge",
            "max_cluster_rms",
            "min_cluster_barycenter",
            "weighted_global_barycenter",
        ]


        all_metrics = per_module_cols
        all_metrics = sorted(all_metrics)

        ncols = 4
        nrows = (len(all_metrics) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(all_metrics):
            if col in df.columns:
                ax = axes[i]
                data = df[col]
                data = data[np.isfinite(data)]  # drop NaNs
                ax.hist(data, bins=50, alpha=0.7)
                ax.set_title(col.replace('_', ' '))
                ax.set_xlabel('Value')
                ax.set_ylabel('Entries')

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.suptitle("Histograms of Cluster and Event Metrics", fontsize=16, y=1.02)
        if save_plots:
            name_of_file = 'charge_statistics_per_module'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

        # --- Plot histograms event-wise ---

        all_metrics = event_level_cols
        all_metrics = sorted(all_metrics)

        ncols = 4
        nrows = (len(all_metrics) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for i, col in enumerate(all_metrics):
            if col in df.columns:
                ax = axes[i]
                data = df[col]
                data = data[np.isfinite(data)]  # drop NaNs
                ax.hist(data, bins=50, alpha=0.7)
                ax.set_title(col.replace('_', ' '))
                ax.set_xlabel('Value')
                ax.set_ylabel('Entries')

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.suptitle("Histograms of Cluster and Event Metrics", fontsize=16, y=1.02)
        if save_plots:
            name_of_file = 'charge_statistics_global'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


        # Topology --------------------------------------------------------------------

        df["topology"] = df[[f"cluster_size_{i}" for i in range(1, 5)]].astype(str).agg("".join, axis=1)

        topology_counts = df["topology"].value_counts(normalize=True)
        topology_filtered = topology_counts[topology_counts >= 0.001]  # keep ≥ 0.1%

        # Plot
        plt.figure(figsize=(12, 6)) 
        plt.bar(topology_filtered.index, topology_filtered.values)
        plt.xlabel("Topology (cluster sizes per plane)")
        plt.ylabel("Number of Events")
        plt.title("Event Topology Frequency Histogram")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.suptitle("Histograms of Cluster and Event Metrics", fontsize=16, y=1.02)
        if save_plots:
            name_of_file = 'topology'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


        # Topology per charges --------------------------------------------------------------------------------

        i_vals = np.arange(0, 21, 10)     # From 0 to 80 in steps of 20
        j_vals = [90]  # Only the value 100

        plt.figure(figsize=(12, 6))
        color_cycle = plt.cm.viridis(np.linspace(0, 1, len(i_vals) * len(j_vals)))

        k = 0  # color index
        for i_min in i_vals:
            for j_max in j_vals:
                # Define topology_i_j as a 4-digit string based on charge cuts
                def compute_topology(row):
                    topology_digits = []
                    for m in range(1, 5):
                        q = row[f"Q_{m}"]
                        s = row[f"cluster_size_{m}"]
                        digit = str(s) if i_min <= q <= j_max else "0"
                        topology_digits.append(digit)
                    return "".join(topology_digits)

                col_name = f"topology_{i_min}_{j_max}"
                df[col_name] = df.apply(compute_topology, axis=1)

                # Get normalized histogram
                topo_counts = df[col_name].value_counts(normalize=True)
                topo_counts = topo_counts[topo_counts >= 0.001]
                topo_counts = topo_counts[topo_counts.index != "0000"]
                topo_counts = topo_counts[topo_counts.index.map(lambda x: sum(c != '0' for c in x) > 1)]
                
                if not topo_counts.empty:
                    # Prepare integer x-axis
                    x_vals = np.arange(len(topo_counts))
                    labels = topo_counts.index  # topology strings

                    # Plot bars
                    plt.bar(
                        x_vals,
                        topo_counts.values,
                        alpha=0.25,
                        color=color_cycle[k % len(color_cycle)],
                        edgecolor='black',
                        label=f"{i_min}–{j_max}"
                    )

                    # Plot connecting lines
                    plt.plot(
                        x_vals,
                        topo_counts.values,
                        alpha=0.75,
                        color=color_cycle[k % len(color_cycle)]
                    )

                    # Set the x-axis ticks to the topology strings
                    plt.xticks(x_vals, labels, rotation=90)
                    k += 1

        plt.xlabel("Topology (cluster sizes per plane)")
        plt.ylabel("Relative Frequency")
        plt.title("Overlaid Topology Histograms for Charge Windows")
        plt.xticks(rotation=90)
        plt.legend(title="Q window (i–j)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_plots:
            name_of_file = 'topology_charge_windows'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close()


        # Binary topology --------------------------------------------------------------------------------

        df["binary_topology"] = (df[[f"cluster_size_{i}" for i in range(1, 5)]] > 0).astype(int).astype(str).agg("".join, axis=1)

        topology_counts = df["binary_topology"].value_counts(normalize=True)
        topology_filtered = topology_counts[topology_counts >= 0.00000001]  # keep ≥ 0.1%

        # Plot
        plt.figure(figsize=(12, 6)) 
        plt.bar(topology_filtered.index, topology_filtered.values)
        plt.xlabel("Topology (cluster sizes per plane)")
        plt.ylabel("Number of Events")
        plt.title("Event Topology Frequency Histogram")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.suptitle("Histograms of Cluster and Event Metrics", fontsize=16, y=1.02)
        if save_plots:
            name_of_file = 'binary_topology'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


        # Binary Topology per charges --------------------------------------------------------------------------------

        i_vals = np.arange(0, 21, 10)  # e.g., [0, 10]
        j_vals = [300]                 # e.g., [90]

        plt.figure(figsize=(12, 6))
        color_cycle = plt.cm.viridis(np.linspace(0, 1, len(i_vals) * len(j_vals)))

        k = 0  # color index
        for i_min in i_vals:
            for j_max in j_vals:
                # Define binary_topology_i_j: '1' if Q in range and cluster_size > 0, else '0'
                def compute_binary_topology(row):
                    return "".join(
                        ['1' if (i_min <= row[f"Q_{m}"] <= j_max and row[f"cluster_size_{m}"] > 0) else '0'
                        for m in range(1, 5)]
                    )

                col_name = f"binary_topology_{i_min}_{j_max}"
                df[col_name] = df.apply(compute_binary_topology, axis=1)

                # Get normalized histogram
                topo_counts = df[col_name].value_counts(normalize=False)
                # topo_counts = topo_counts[topo_counts >= 0.001]
                topo_counts = topo_counts[topo_counts.index != "0000"]
                topo_counts = topo_counts[topo_counts.index.map(lambda x: sum(c != '0' for c in x) > 1)]

                if not topo_counts.empty:
                    # Prepare integer x-axis
                    x_vals = np.arange(len(topo_counts))
                    labels = topo_counts.index  # binary topology strings

                    # Plot bars
                    plt.bar(
                        x_vals,
                        topo_counts.values,
                        alpha=0.25,
                        color=color_cycle[k % len(color_cycle)],
                        edgecolor='black',
                        label=f"{i_min}–{j_max}"
                    )

                    # Plot connecting lines
                    plt.plot(
                        x_vals,
                        topo_counts.values,
                        alpha=0.75,
                        color=color_cycle[k % len(color_cycle)]
                    )

                    # Set the x-axis ticks to the binary topology strings
                    plt.xticks(x_vals, labels, rotation=90)
                    k += 1

        plt.xlabel("Binary Topology (active modules)")
        plt.ylabel("Relative Frequency")
        plt.title("Overlaid Binary Topology Histograms for Charge Windows")
        plt.xticks(rotation=90)
        plt.legend(title="Q window (i–j)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_plots:
            name_of_file = 'binary_topology_charge_windows'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')

        if show_plots:
            plt.show()
        plt.close()
    
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------

    # This should be the definition of another table which contains all the metadata and metanalysis, and
    # in principle we are not joining them, I think.
    
    # print("----------------------------------------------------------------------")
    # print("----------------- Derived metrics pre-aggregation --------------------")
    # print("----------------------------------------------------------------------")

    # # Derived metrics
    # print("Derived metrics...")
    # for i in range(1, 5):
    #     df[f'Q_{i}'] = df[[f'Q_P{i}s{j}' for j in range(1, 5)]].sum(axis=1)
    #     df[f'count_in_{i}'] = (df[f'Q_{i}'] > 0).astype(int)
    #     df[f'streamer_{i}'] = (df[f'Q_{i}'] > 100).astype(int)

    # df[f'Q_event'] = df[[f'Q_{j}' for j in range(1, 5)]].sum(axis=1)



    # print("----------------------------------------------------------------------")
    # print("----------------- Aggregation and Poisson filtering ------------------")
    # print("----------------------------------------------------------------------")

    # df['events'] = 1

    # # Aggregation logic
    # # Start with your static aggregation dictionary
    # agg_dict = {
    #     'events': 'sum',
    #     'count_in_1': 'sum',
    #     'count_in_2': 'sum',
    #     'count_in_3': 'sum',
    #     'count_in_4': 'sum',
    #     'streamer_1': 'sum',
    #     'streamer_2': 'sum',
    #     'streamer_3': 'sum',
    #     'streamer_4': 'sum',
    
    #     'original_tt': lambda x: pd.Series(x).value_counts().to_dict(),
    #     'processed_tt': lambda x: pd.Series(x).value_counts().to_dict(),
    #     # 'tracking_tt': lambda x: pd.Series(x).value_counts().to_dict(),
    
    #     'x': [custom_mean, custom_std],
    #     'y': [custom_mean, custom_std],
    #     'theta': [custom_mean, custom_std],
    #     'phi': [custom_mean, custom_std],
    #     'new_s': [custom_mean, custom_std],
    #     'new_th_chi': [custom_mean, custom_std],
    
    #     'Q_event': [custom_mean, custom_std],
    
    #     'CRT_avg': custom_mean,
    #     'one_side_events': custom_mean,
    #     'purity_of_data_percentage': custom_mean,
    #     'unc_y': custom_mean,
    #     'unc_tsum': custom_mean,
    #     'unc_tdif': custom_mean,
    
    #     "over_P1": custom_mean,
    #     "P1-P2": custom_mean,
    #     "P2-P3": custom_mean,
    #     "P3-P4": custom_mean,
    #     "phi_north": custom_mean,
    
    #     "P1_induction_section": custom_mean,
    #     "P2_induction_section": custom_mean,
    #     "P3_induction_section": custom_mean,
    #     "P4_induction_section": custom_mean,
    
    #     # Efficiency fitting
    #     "P2_3fold_a": custom_mean,
    #     "P2_3fold_eps0": custom_mean,
    #     "P2_3fold_n": custom_mean,
    #     "P2_2fold_a": custom_mean,
    #     "P2_2fold_eps0": custom_mean,
    #     "P2_2fold_n": custom_mean,
    #     "P3_3fold_a": custom_mean,
    #     "P3_3fold_eps0": custom_mean,
    #     "P3_3fold_n": custom_mean,
    #     "P3_3fold_a": custom_mean,
    #     "P3_3fold_eps0": custom_mean,
    #     "P3_3fold_n": custom_mean,
    # }

    # # Dynamically add all sigmoid_width_XXX and background_slope_XXX columns
    # sigmoid_cols = [col for col in df.columns if col.startswith('sigmoid_width_')]
    # background_cols = [col for col in df.columns if col.startswith('background_slope_')]

    # for col in sigmoid_cols + background_cols:
    #     agg_dict[col] = custom_mean

    # # Add all new region columns with sum aggregation
    # for col in df.columns:
    #     if col.startswith("new_region_theta_"):
    #         agg_dict[col] = 'sum'


    # # Fit a Poisson distribution to the 1-second data and removed outliers based on the Poisson -------------------------
    # if remove_outliers:
    #     print("Resampling for outlier removal...")
    #     # Resampling
    #     resampled_df_test = df.resample('1min', on='Time').agg(agg_dict)
    #     # TEST 1s rates -------------------------------------------------------
    #     resampled_second_df = df.resample('1s', on='Time').agg(agg_dict)
    #     print("Resampled.")
    #     # Plot the 1 min and 1 s rates ----------------------------------------
    #     if create_plots:
    #         fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    #         ax[0].plot(resampled_df_test.index, resampled_df_test['events'], label='Q_event')
    #         ax[0].set_ylabel('Q_event (mean)')
    #         ax[0].legend()

    #         ax[1].plot(resampled_second_df.index, resampled_second_df['events'], label='Q_event')
    #         ax[1].set_ylabel('Q_event (mean)')
    #         ax[1].legend()

    #         plt.tight_layout()
    #         # Show the plot
    #         if save_plots:
    #             name_of_file = 'original_rates'
    #             final_filename = f'{fig_idx}_{name_of_file}.png'
    #             fig_idx += 1
            
    #             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
    #             plot_list.append(save_fig_path)
    #             plt.savefig(save_fig_path, format='png')
            
    #         if show_plots: plt.show()
    #         plt.close()

    #     print("Removing outliers...")
    #     # Data: Replace with your actual data
    #     data = resampled_second_df['events']

    #     # Define the negative log-likelihood function for Poisson
    #     def negative_log_likelihood(lambda_hat, data):
    #         return -np.sum(poisson.logpmf(data, lambda_hat))

    #     # Initial guess for λ (mean of data)
    #     initial_guess = np.mean(data)

    #     # Optimize λ to minimize the negative log-likelihood
    #     result = minimize(negative_log_likelihood, x0=initial_guess, args=(data,), bounds=[(1e-5, None)])
    #     lambda_fit = result.x[0]

    #     print(f"Best-fit λ: {lambda_fit:.2f}")

    #     lower_bound = poisson.ppf(0.0005, lambda_fit)  # Lower 00.05% bound
    #     upper_bound = poisson.ppf(0.9995, lambda_fit)  # Upper 99.95% bound

    #     # Overlay the fitted Poisson distribution
    #     if create_plots:
    #         # Generate Poisson probabilities for the range of your data
    #         x = np.arange(0, np.max(data) + 1)  # Range of event counts
    #         poisson_probs = poisson.pmf(x, lambda_fit) * len(data)  # Scale by sample size

    #         plt.hist(data, bins=100, alpha=0.7, label='Observed data', color='blue', density=False)
    #         plt.plot(x, poisson_probs, 'r-', lw=2, label=f'Poisson Fit ($\lambda={lambda_fit:.2f}$)')
    #         plt.axvline(lower_bound, color='r', linestyle='--', label='Poisson 0.1%')
    #         plt.axvline(upper_bound, color='r', linestyle='--', label='Poisson 99.9%')
    #         plt.xlabel('Number of events per second')
    #         plt.ylabel('Frequency')
    #         plt.title('Histogram of events per second with Poisson Fit')
    #         plt.legend()
    #         if save_plots:
    #             name_of_file = 'poisson_fit'
    #             final_filename = f'{fig_idx}_{name_of_file}.png'
    #             fig_idx += 1
    #             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
    #             plot_list.append(save_fig_path)
    #             plt.savefig(save_fig_path, format='png')
    #         if show_plots: plt.show()
    #         plt.close()
        
    #     # Now, if any value is outside of the tails of the poisson distribution, we can remove it
    #     # so obtain the extremes, obtain the seconds in which the values are outside of the distribution
    #     # and remove those seconds from a copy of the original df, so a new resampled_df can be created
    #     # with the new data
    
    #     # Ensure 'events' is a flat Series
    #     events_series = resampled_second_df['events']
    #     # Count how many values fall below or above the bounds
    #     below_count = (events_series < lower_bound).sum()
    #     above_count = (events_series > upper_bound).sum()
    #     print("-------------------------------------------------------------")
    #     print(f"Below bound: {below_count.sum()}")
    #     print(f"Above bound: {above_count.sum()}")
    #     print(f"Total outliers: {(below_count + above_count).sum()}")
    #     print("-------------------------------------------------------------")
    #     # Identify outlier indices correctly
    #     outlier_mask = (events_series < lower_bound) | (events_series > upper_bound)
    #     # Flatten outlier_mask to 1D
    #     outlier_mask = outlier_mask.values.ravel()  # Turn into a 1D array
    #     # Extract the indices of outliers
    #     outlier_indices = events_series.index[outlier_mask]
    #     # Create a temporary 'Time_sec' column floored to seconds for alignment
    #     df['Time_sec'] = df['Time'].dt.floor('1s')
    #     # Filter out rows corresponding to the outlier indices
    #     filtered_df = df[~df['Time_sec'].isin(outlier_indices)].drop(columns=['Time_sec'])
    #     # Resample the filtered data to 1-minute intervals
    #     new_resampled_df = filtered_df.resample('1min', on='Time').agg(agg_dict)

    #     # Plot the new resampled data
    #     if create_plots:
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #         ax.plot(new_resampled_df.index, new_resampled_df['events'], label='Q_event (filtered)')
    #         ax.set_ylabel('Q_event (mean)')
    #         ax.set_title('1-Minute Resampled Data After Outlier Removal')
    #         ax.legend()
    #         plt.tight_layout()
    #         if save_plots:
    #             name_of_file = 'filtered_rate'
    #             final_filename = f'{fig_idx}_{name_of_file}.png'
    #             fig_idx += 1
    #             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
    #             plot_list.append(save_fig_path)
    #             plt.savefig(save_fig_path, format='png')
    #         if show_plots: plt.show()
    #         plt.close()

    #     # Create a resampled_df with the new one
    
    #     rejected_percentage = ((below_count + above_count) / len(df)) * 100
    #     global_variables["poisson_rejected"] = rejected_percentage
    
    #     resampled_df = new_resampled_df
    # else:
    #     resampled_df = df.resample('1min', on='Time').agg(agg_dict)


    # print("----------------------------------------------------------------------")
    # print("--------------------------- Some renaming ----------------------------")
    # print("----------------------------------------------------------------------")

    # resampled_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in resampled_df.columns.values]

    # resampled_df.rename(columns=lambda x: x.replace('custom_mean', 'mean').replace('custom_std', 'std'), inplace=True)
    # resampled_df.rename(columns=lambda x: x.replace('_sum', ''), inplace=True)
    # resampled_df.rename(columns=lambda x: x.replace('_mean', ''), inplace=True)
    # resampled_df.rename(columns=lambda x: x.replace('new_', ''), inplace=True)


    # print("----------------------------------------------------------------------")
    # print("------------------------ Column aggregation --------------------------")
    # print("----------------------------------------------------------------------")

    # # Region-specific count aggregation -------------------------------------------
    # region_counts = pd.crosstab(df['Time'].dt.floor('1min'), df['region'])
    # resampled_df = resampled_df.join(region_counts, how='left').fillna(0)

    # # Hans' region-specific count aggregation -------------------------------------
    # # region_hans_counts = pd.crosstab(df['Time'].dt.floor('1min'), df['region_hans'])
    # # resampled_df = resampled_df.join(region_hans_counts, how='left').fillna(0)

    # # New region-specific count aggregation ---------------------------------------
    # # Floor time to 1-minute bins
    # df['Time_floor'] = df['Time'].dt.floor('1min')
    # # Select only the binary region columns
    # region_cols = [col for col in df.columns if col.startswith('region_theta_')]
    # # Group by floored time and sum the binary indicators
    # region_counts = df.groupby('Time_floor')[region_cols].sum()
    # # Join to resampled_df (assuming resampled_df has time index compatible with Time_floor)
    # resampled_df = resampled_df.join(region_counts, how='left').fillna(0)


    # print("----------------------------------------------------------------------")
    # print("-------------------------- Counting types ----------------------------")
    # print("----------------------------------------------------------------------")

    # # Split 'type_<lambda>' dictionary into separate columns for each type
    # # types = ["original_tt", "processed_tt", "tracking_tt"]
    # types = ["original_tt", "processed_tt"]
    # for type_key in types:
    #     if f'{type_key}_<lambda>' in resampled_df.columns:
    #         type_dict_col = resampled_df[f'{type_key}_<lambda>']
    #         for type_key_unique in df[type_key].unique():
    #             resampled_df[f'{type_key}_{type_key_unique}'] = type_dict_col.apply(lambda x: x.get(type_key_unique, 0) if isinstance(x, dict) else 0)
    #         resampled_df.drop(columns=[f'{type_key}_<lambda>'], inplace=True)

    # # Streamer percentage
    # for i in range(1, 5):
    #     resampled_df[f"streamer_percent_{i}"] = ( (resampled_df[f"streamer_{i}"] / resampled_df[f"count_in_{i}"]).fillna(0) * 100 )

    # print("----------------------------------------------------------------------")
    # print("----------------------------------------------------------------------")
    # print("----------------------- Saving and finishing -------------------------")
    # print("----------------------------------------------------------------------")
    # print("----------------------------------------------------------------------")

    # resampled_df.reset_index(inplace=True)

    # if polya_fit:

    #     print("\n\n")
    #     # Flatten df_polya_fit into wide format
    #     df_single_row = df_polya_fit.set_index('module').stack().rename('value').reset_index()
    #     df_single_row['column'] = 'polya_' + df_single_row['level_1'] + '_' + df_single_row['module'].astype(str)

    #     # Fix: pivot without setting index=None
    #     df_wide = df_single_row.pivot(columns='column', values='value')
    #     df_wide.columns.name = None  # remove column index name
    #     df_wide = df_wide.reset_index(drop=True)

    #     # Check available columns
    #     print(df_wide.columns.to_list())

    #     # Optional: restrict to specific subset of columns
    #     df_wide = df_wide[[
    #         'polya_A_1', 'polya_A_2', 'polya_A_3', 'polya_A_4',
    #         'polya_theta_1', 'polya_theta_2', 'polya_theta_3', 'polya_theta_4',
    #         'polya_nbar/alpha_1', 'polya_nbar/alpha_2', 'polya_nbar/alpha_3', 'polya_nbar/alpha_4',
    #     ]]

    #     # Repeat values to match number of rows in resampled_df
    #     df_polya_expanded = pd.concat([df_wide] * len(resampled_df), ignore_index=True)

    #     # Merge with original DataFrame
    #     resampled_df = pd.concat([resampled_df, df_polya_expanded], axis=1)
    
    #     with pd.option_context('display.precision', 1):
    #         print(df_polya_fit)
    
    #     # Optional print
    #     with pd.option_context('display.precision', 3):
    #         print(resampled_df[df_wide.columns])


    # print(df_polya_flat.columns)
    # print(df_polya_flat.columns)
    # df_polya = df_polya_flat['polya_theta', 'polya_nbar/alpha', 'polya_A']

    # print(df_polya)

    # # --- Flatten df_cross_fit ---
    # df_cross_flat = df_cross_fit.set_index('key').add_prefix('cross_')  # cross_M1_s1_x0, cross_M1_s1_k

    # # --- Flatten df_mult_fit ---
    # df_mult_long = df_mult_fit.melt(id_vars=['detection_type', 'multiplicity'], var_name='module', value_name='value')
    # df_mult_long['colname'] = (
    #     df_mult_long['detection_type'] + '_' +
    #     df_mult_long['multiplicity'] + '_' +
    #     df_mult_long['module']
    # )
    # df_mult_flat = df_mult_long.set_index('colname')['value'].to_frame().T.add_prefix('mult_')  # single row

    # # --- Combine all into one row DataFrame ---
    # flat_polya = df_polya_flat.stack().to_frame().T
    # flat_cross = df_cross_flat.stack().to_frame().T
    # flat_mult = df_mult_flat

    # print(flat_polya)

    # flat_all = pd.concat([flat_polya, flat_cross, flat_mult], axis=1)

    # # --- Repeat and merge with resampled_df ---
    # flat_all_repeated = pd.concat([flat_all] * len(resampled_df), ignore_index=True)
    # resampled_df = pd.concat([resampled_df.reset_index(drop=True), flat_all_repeated], axis=1)


    # print("----------------------------------------------------------------------")
    # print("------------------------- Saving the data ----------------------------")
    # print("----------------------------------------------------------------------")

    # # Print the columns of resampled_df
    # print("\n\n")
    # print(resampled_df.columns.to_list())
    # print("\n\n")

    # resampled_df = resampled_df.applymap(round_to_significant_digits)

    # # Save the newly created file to ACC_EVENTS_DIRECTORY --------------------------
    # resampled_df.to_csv(full_save_path, sep=',', index=False)
    # print(f"Complete datafile saved in {full_save_filename}. Path is {full_save_path}")

    # sigmoid_cols = [col for col in df.columns if col.startswith('sigmoid_width_')]
    # background_cols = [col for col in df.columns if col.startswith('background_slope_')]

    # columns_to_keep = [
    #     # Introductory
    #     'Time',
    
    #     # Columns to sum -----------------------------------------------------------
    
    #     # Basic counts
    #     'events', 'count_in_1', 'count_in_2', 'count_in_3', 'count_in_4',
    
    #     # Detection types
    #     'original_tt_123', 'original_tt_12', 'original_tt_234', 'original_tt_34', 'original_tt_23', 'original_tt_1234', 'original_tt_134', 'original_tt_124', 'original_tt_13',
    #     'processed_tt_123', 'processed_tt_12', 'processed_tt_234', 'processed_tt_34', 'processed_tt_23', 'processed_tt_1234', 'processed_tt_24', 'processed_tt_13', 'processed_tt_134', 'processed_tt_14', 'processed_tt_124',
    #     # 'tracking_tt_1234', 'tracking_tt_123', 'tracking_tt_12', 'tracking_tt_234', 'tracking_tt_34', 'tracking_tt_23',
    
    #     # Region-specific counts
    #     'High', 'N', 'S', 'E', 'W',
    
    #     # Counts to average ---------------------------------------------------------
    
    #     # Summary metrics and quality flags
    #     'CRT_avg', 'one_side_events', 'purity_of_data_percentage',
    #     'unc_y', 'unc_tsum', 'unc_tdif',

    #     # Reconstruction outputs
    #     'x', 'y', 'theta', 'phi', 's', 'th_chi',
    #     'x_std', 'y_std', 'theta_std', 'phi_std', 's_std', 'th_chi_std',

    #     'streamer_percent_1', 'streamer_percent_2', 'streamer_percent_3', 'streamer_percent_4',
    
    #     # Configuration parameters
    #     "over_P1", "P1-P2", "P2-P3", "P3-P4", "phi_north",
    
    #     # Induction section
    #     "P1_induction_section", "P2_induction_section", "P3_induction_section", "P4_induction_section",
    
    #     # Efficiency fittings
    #     "P2_3fold_a", "P2_3fold_eps0", "P2_3fold_n",
    #     "P2_2fold_a", "P2_2fold_eps0", "P2_2fold_n",
    #     "P3_3fold_a", "P3_3fold_eps0", "P3_3fold_n",
    #     "P3_3fold_a", "P3_3fold_eps0", "P3_3fold_n",
    # ]

    # sigmoid_cols = [col for col in df.columns if col.startswith('sigmoid_width_')]
    # background_cols = [col for col in df.columns if col.startswith('background_slope_')]

    # columns_to_keep.extend(sigmoid_cols + background_cols)

    # # Filter columns_to_keep to include only those present in resampled_df
    # valid_columns = [col for col in columns_to_keep if col in resampled_df.columns]

    # # Optionally, fill missing columns with NaN (or zeros) before subsetting
    # missing_columns = [col for col in columns_to_keep if col not in resampled_df.columns]
    # for col in missing_columns:
    #     resampled_df[col] = np.nan  # or 0, depending on context

    # # Now subset safely
    # reduced_df = resampled_df[columns_to_keep]

    # reduced_df = resampled_df[columns_to_keep]
    # reduced_df.to_csv(save_path, index=False, sep=',', float_format='%.5g')
    # print(f"Reduced columns datafile saved in {save_filename}. Path is {save_path}")


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

#%%

print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("------------------------- Main structure -----------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

main_df = df.copy()
main_df['Theta_fit'] = main_df['theta']
main_df['Phi_fit'] = main_df['phi']

if correct_angle:
    print("----------------------------------------------------------------------")
    print("-------- 1. Correction of the fitted angle --> predicted angle -------")
    print("----------------------------------------------------------------------")

    # ---------------------------------------------------------------
    # 1. Build absolute path and sanity-check
    # ---------------------------------------------------------------
    hdf_path = os.path.join(config_files_directory, "likelihood_matrices.h5")
    if not os.path.isfile(hdf_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf_path}")

    #%%

    # ---------------------------------------------------------------
    # 2. Load all matrices into memory
    # ---------------------------------------------------------------
    matrices = {}
    n_bins = None

    with pd.HDFStore(hdf_path, mode='r') as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"{hdf_path} contains no datasets.")

        for key in keys:                     # keys like '/P1', '/P2', …
            ttype = key.strip('/')           # remove leading slash
            # df_M = store.get(key)
            
            # Reduce the precision to float32 to not kill RAM
            df_M = store.get(key).astype(np.float16)
            
            matrices[ttype] = df_M

            # set n_bins once, based on the first matrix's shape
            if n_bins is None:
                size = df_M.shape[0]
                n_bins = int(np.sqrt(size))
                if n_bins * n_bins != size:
                    raise ValueError(f"Matrix size {size} is not a perfect square.")

            print(f"Loaded matrix for {ttype}: shape {df_M.shape}")

    print(f"n_bins detected: {n_bins}")

    # Helpers
    def flat(u_idx, v_idx, n_bins):
        return u_idx * n_bins + v_idx

    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    #%%

    with pd.HDFStore(hdf_path, 'r') as store:
        print("HDF5 keys:", store.keys())

    def sample_true_angles_nearest(
        df_fit: pd.DataFrame,
        matrices: Optional[Dict[str, pd.DataFrame]],
        n_bins: int,
        rng: Optional[np.random.Generator] = None,
        show_progress: bool = True,
        print_every: int = 10_000
        ) -> pd.DataFrame:
        
        if rng is None:
            rng = np.random.default_rng()

        matrix_cache = {t: df_m.to_numpy() for t, df_m in matrices.items()}
        
        u_edges = np.linspace(-1.0, 1.0, n_bins + 1)
        v_edges = np.linspace(-1.0, 1.0, n_bins + 1)

        u_fit = np.sin(df_fit["Theta_fit"].values) * np.sin(df_fit["Phi_fit"].values)
        v_fit = np.sin(df_fit["Theta_fit"].values) * np.cos(df_fit["Phi_fit"].values)

        iu = np.clip(np.digitize(u_fit, u_edges) - 1, 0, n_bins - 2)
        iv = np.clip(np.digitize(v_fit, v_edges) - 1, 0, n_bins - 2)

        iu += (u_fit - u_edges[iu]) > (u_edges[iu + 1] - u_fit)
        iv += (v_fit - v_edges[iv]) > (v_edges[iv + 1] - v_fit)

        flat_idx = lambda u, v: u * n_bins + v
        unflat = lambda k: divmod(k, n_bins)

        N = len(df_fit)
        theta_pred = np.empty(N, dtype=np.float32)
        phi_pred = np.empty(N, dtype=np.float32)

        iterator = tqdm(range(N), desc="Sampling true angles (nearest-bin)", unit="evt") if show_progress else range(N)

        for n in iterator:
            t_type = str(df_fit["definitive_tt"].iat[n])   # ensure string

            if t_type not in matrix_cache:
                raise ValueError(f"LUT not found for type: {t_type}")
            M = matrix_cache[t_type]

            col_idx = flat_idx(iu[n], iv[n])
            p = M[:, col_idx]
            s = p.sum()

            if s == 0:
                p = np.full_like(p, 1.0 / len(p))
            else:
                p /= s

            gen_idx = rng.choice(len(p), p=p)
            g_u_idx, g_v_idx = unflat(gen_idx)

            u_pred = rng.uniform(u_edges[g_u_idx], u_edges[g_u_idx + 1])
            v_pred = rng.uniform(v_edges[g_v_idx], v_edges[g_v_idx + 1])

            sin_theta = min(np.hypot(u_pred, v_pred), 1.0)
            theta_pred[n] = math.asin(sin_theta)
            phi_pred[n] = wrap_to_pi(math.atan2(u_pred, v_pred))

        df_out = df_fit.copy()
        df_out["Theta_pred"] = theta_pred
        df_out["Phi_pred"] = phi_pred
        return df_out

    #%%

    print(main_df.columns.to_list())

    #%%

    df_input = main_df
    df_pred = sample_true_angles_nearest(
                df_fit=df_input,
                matrices=matrices,
                n_bins=n_bins,
                rng=np.random.default_rng(),
                show_progress=True )

    df = df_pred.copy()
else:
    print("Angle correction is disabled.")
    df['Theta_pred'] = main_df['Theta_fit']
    df['Phi_pred'] = main_df['Phi_fit']


if create_very_essential_plots:    
    VALID_MEASURED_TYPES = ['1234', '123', '124', '234', '134', '12', '13', '14', '23', '24', '34']
    tt_lists = [ VALID_MEASURED_TYPES ]
    
    for tt_list in tt_lists:
          fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex='row')

          # Fourth column: Measured (θ_fit, ϕ_fit)
          axes[0, 0].hist(df['Theta_fit'], bins=theta_bins, histtype='step', color='black', label='All')
          axes[1, 0].hist(df['Phi_fit'], bins=phi_bins, histtype='step', color='black', label='All')
          for tt in tt_list:
                sel = (df['definitive_tt'] == int(tt))
                axes[0, 0].hist(df.loc[sel, 'Theta_fit'], bins=theta_bins, histtype='step', label=tt)
                axes[1, 0].hist(df.loc[sel, 'Phi_fit'], bins=phi_bins, histtype='step', label=tt)
                axes[0, 0].set_title("Measured tracks θ_fit")
                axes[1, 0].set_title("Measured tracks ϕ_fit")
      
          # Fourth column: Measured (θ_fit, ϕ_fit)
          axes[0, 1].hist(df['Theta_pred'], bins=theta_bins, histtype='step', color='black', label='All')
          axes[1, 1].hist(df['Phi_pred'], bins=phi_bins, histtype='step', color='black', label='All')
          for tt in tt_list:
                sel = (df['definitive_tt'] == int(tt))
                axes[0, 1].hist(df.loc[sel, 'Theta_pred'], bins=theta_bins, histtype='step', label=tt)
                axes[1, 1].hist(df.loc[sel, 'Phi_pred'], bins=phi_bins, histtype='step', label=tt)
                axes[0, 1].set_title("Corrected tracks θ_fit")
                axes[1, 1].set_title("Corrected tracks ϕ_fit")

          # Common settings
          for ax in axes.flat:
                ax.legend(fontsize='x-small')
                ax.grid(True)

          axes[1, 0].set_xlabel(r'$\phi$ [rad]')
          axes[0, 0].set_ylabel('Counts')
          axes[1, 0].set_ylabel('Counts')
          axes[0, 1].set_xlim(0, np.pi / 2)
          axes[1, 1].set_xlim(-np.pi, np.pi)

          fig.tight_layout()
          plt.show()


print("----------------------------------------------------------------------")
print("------------- 2. Determination of the angular sector -----------------")
print("----------------------------------------------------------------------")

#%%

draw_angular_regions = True
if draw_angular_regions:
    
    print("----------------------- Drawing angular regions ----------------------")
    
    def plot_polar_region_grid_flexible(ax, theta_boundaries, region_layout, theta_right_limit=np.pi / 2.5):

        # Only use boundaries below or equal to theta_right_limit
        max_deg = np.degrees(theta_right_filter)
        valid_boundaries = [b for b in theta_boundaries if b <= max_deg]
        all_bounds = [0] + valid_boundaries + [max_deg]
        radii = [np.radians(b) for b in all_bounds]

        # Draw concentric circles (excluding outermost edge)
        for r in radii[1:-1]:
            ax.plot(np.linspace(0, 2 * np.pi, 1000), [r] * 1000, color='white', linestyle='--', linewidth=3)

        # Draw radial lines within each ring
        for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii)-1])):
            if n_phi <= 1:
                continue
            delta_phi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi = j * delta_phi
                ax.plot([phi, phi], [r0, r1], color='white', linestyle='--', linewidth=3)


    def classify_region_flexible(row, theta_boundaries, region_layout):
        theta = row['theta'] * 180 / np.pi
        phi = (row['phi'] * 180 / np.pi + row.get('phi_north', 0)) % 360
        phi = ((phi + 180) % 360) - 180  # map to [-180, 180)

        # Build region bins: [0, t1), [t1, t2), ..., [tn, 90]
        all_bounds = [0] + theta_boundaries + [90]
        for i, (tmin, tmax) in enumerate(zip(all_bounds[:-1], all_bounds[1:])):
            if tmin <= theta < tmax or (i == len(region_layout) - 1 and theta == 90):
                n_phi = region_layout[i]
                if n_phi == 1:
                    return f'R{i}.0'
                else:
                    bin_width = 360 / n_phi
                    idx = int((phi + 180) // bin_width) % n_phi
                    return f'R{i}.{idx}'
            
        return 'None'

    # Input parameters
    theta_right_limit = np.pi / 2.5

    # Compute angular boundaries
    max_deg = np.degrees(theta_right_limit)
    valid_boundaries = [b for b in theta_boundaries if b <= max_deg]
    all_bounds_deg = [0] + valid_boundaries + [max_deg]
    radii = np.radians(all_bounds_deg)

    # Initialize plot
    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(8, 8))
    ax.set_facecolor(plt.cm.viridis(0.0))
    ax.set_title("Region Labels for Specified Angular Segmentation", color='white')
    ax.set_theta_zero_location('N')


    # Draw concentric θ boundaries (including outermost)
    for r in radii[1:]:
        ax.plot(np.linspace(0, 2 * np.pi, 1000), [r] * 1000,
                color='white', linestyle='--', linewidth=3)

    # Draw radial (φ) separators for each region layout
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii) - 1])):
        if n_phi > 1:
            delta_phi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi = j * delta_phi
                ax.plot([phi, phi], [r0, r1], color='white', linestyle='--', linewidth=1.5)

    # Annotate region labels
    for i, (r0, r1, n_phi) in enumerate(zip(radii[:-1], radii[1:], region_layout[:len(radii) - 1])):
        r_label = (r0 + r1) / 2
        if n_phi == 1:
            ax.text(0, r_label, f'R{i}.0', ha='center', va='center',
                    color='white', fontsize=10, weight='bold')
        else:
            dphi = 2 * np.pi / n_phi
            for j in range(n_phi):
                phi_label = (j + 0.5) * dphi
                ax.text(phi_label, r_label, f'R{i}.{j}', ha='center', va='center',
                        rotation=0, rotation_mode='anchor',
                        color='white', fontsize=10, weight='bold')

    # Add radius labels slightly *outside* the outermost circle for clarity
    for r_deg in all_bounds_deg[1:]:
        r_rad = np.radians(r_deg)
        ax.text(np.pi + 0.09, r_rad - 0.05, f'{int(round(r_deg))}°', ha='center', va='bottom',
                color='white', fontsize=10, alpha=0.9)

    ax.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_yticklabels([])

    # Final layout
    title = "Region Labels for Specified Angular Segmentation"
    ax.set_ylim(0, theta_right_limit)
    plt.suptitle(title, fontsize=16, color='white')
    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_{title.replace(" ", "_")}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


df['region'] = df.apply(lambda row: classify_region_flexible(row, theta_boundaries, region_layout), axis=1)
print(df['region'].value_counts())

#%%

if create_essential_plots or create_very_essential_plots:
    
    print("-------------------------- Angular plots -----------------------------")
        
    df_filtered = df.copy()
    
    # tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    tt_values = [23, 123, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 2
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        
    nbins = 50
    theta_bins = np.linspace(0, np.pi/2, nbins)
    phi_bins = np.linspace(-np.pi, np.pi, nbins)
    colors = plt.cm.viridis

    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]
            
        df_tt = df_filtered[df_filtered['processed_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue
        
        h = ax.hist2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins], cmap='viridis', norm=None, cmin=0, cmax=None)
        ax.set_title(f'processed_tt = {tt_val}')
        ax.set_xlabel(r'$\theta$ [rad]')
        ax.set_ylabel(r'$\phi$ [rad]')
        ax.grid(True)
        # Put the background color to the darkest in the colormap
        ax.set_facecolor(colors(0.0))  # darkest background in colormap

        fig.colorbar(h[3], ax=ax, label='Counts')

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each processed_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plots:
        final_filename = f'{fig_idx}_theta_phi_processed_tt_2D.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()
    
    # ---------------------------------------------------------------------------------
    
    theta_left_filter = 0
    theta_right_filter = np.pi / 2.5
        
    phi_left_filter = -np.pi
    phi_right_filter = np.pi
        
    df_filtered = df.copy()
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))
    
    # tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    tt_values = [23, 123, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 2
    nrows = (n_tt + 1) // ncols
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 70
    # theta_nbins = int(round(phi_nbins / 2) + 1)
    theta_nbins = 40
    theta_bins = np.linspace(theta_left_filter, theta_right_filter, theta_nbins )
    phi_bins = np.linspace(phi_left_filter, phi_right_filter, phi_nbins)
    colors = plt.cm.turbo

    # Select theta/phi range (optional filtering)
    theta_min, theta_max = theta_left_filter, theta_right_filter    # adjust as needed
    phi_min, phi_max     = phi_left_filter, phi_right_filter        # adjust as needed
    
    vmax_global = df_filtered.groupby('definitive_tt').apply(lambda df: np.histogram2d(df['theta'], df['phi'], bins=[theta_bins, phi_bins])[0].max()).max()
    
    for idx, tt_val in enumerate(tt_values):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        # Apply range filtering
        # Apply range filtering
        df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val].copy()
        mask = (
            (df_tt['theta'] >= theta_min) & (df_tt['theta'] <= theta_max) &
            (df_tt['phi'] >= phi_min) & (df_tt['phi'] <= phi_max)
        )
        df_tt = df_tt[mask]

        theta_vals = df_tt['theta']
        phi_vals   = df_tt['phi']

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap
        ax.set_title(f'definitive_tt = {tt_val}', fontsize=14)
            
        plot_polar_region_grid_flexible(ax, theta_boundaries, region_layout)
            
        # Limit in radius in theta_right_filter
        ax.set_ylim(0, theta_right_filter)
            
        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        # R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        R, PHI = np.meshgrid(r_edges, phi_edges, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each definitive_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_definitive_tt_2D_detail.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


#%%

print("----------------------------------------------------------------------")
print("------------------ 3. Binning per Time, tt, Region -------------------")
print("----------------------------------------------------------------------")

# I have three relevant columns: Time, definitive_tt, region. Bin the data of df_main according to the number of
# counts in each group of (Time, definitive_tt, region) and plot the results. But before grouping. round the
# the times to 1 minute

df_original = df.copy()

#%%

# Define time bin widths, from 1s to 360s with 5s steps
time_bins = [ pd.Timedelta(seconds=i) for i in range(1, 601, 1) ]

# Initialize containers
discarded_percentages = []
standard_deviations = []
normalized_curves = []
time_curves = []

for bin_width in time_bins:
    df = df_original.copy()
    df['Time'] = df['Time'].dt.floor(bin_width)

    # Group and count
    binned = df.groupby(['Time', 'definitive_tt', 'region']).size().reset_index(name='counts')
    pivoted = binned.pivot_table(index='Time', columns=['definitive_tt', 'region'], values='counts', fill_value=0)
    pivoted.columns = [f"{tt}_{region}" for tt, region in pivoted.columns]
    pivoted['events'] = pivoted.sum(axis=1)
    pivoted = pivoted.reset_index()

    # Remove borders
    n = 1
    if n > 0:
        pivoted = pivoted.iloc[n:-n]
    if pivoted.empty:
        print(f"No data available for bin width {bin_width}. Skipping...")
        continue

    data = pivoted['events'].values

    # Poisson fit
    def nll(lmbda, data): return -np.sum(poisson.logpmf(data, lmbda))
    res = minimize(nll, x0=[np.mean(data)], args=(data,), bounds=[(1e-5, None)])
    lmbda_fit = res.x[0]

    quantile = 0.5 / 100  # 1%
    lower = poisson.ppf(quantile, lmbda_fit)
    upper = poisson.ppf(1 - quantile, lmbda_fit)

    # Mask
    outlier_mask = (data < lower) | (data > upper)
    num_discarded = outlier_mask.sum()
    percent_kept = 100 - 100 * num_discarded / len(data)
    discarded_percentages.append(percent_kept)

    # Normalize
    normalized = data / np.mean(data)
    standard_deviations.append(np.std(data) / np.mean(data) * 100)

    # Store for plotting
    normalized_curves.append(normalized)
    time_curves.append(pivoted['Time'].values)

if create_essential_plots:
    # Plotting stage
    plt.figure(figsize=(12, 6))

    for t, norm, label in zip(time_curves, normalized_curves, time_bins):
        idx = time_bins.index(label)
        plt.plot(t, norm, label=f'{label} ({discarded_percentages[idx]:.1f}%)', alpha=0.6)

    plt.xlabel("Time")
    plt.ylabel("Normalized Event Rate")
    plt.title("Normalized Rates over Different Time Bins")
    plt.grid(True)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


bin_seconds = [pd.to_timedelta(tb).total_seconds() for tb in time_bins]

if create_essential_plots:
    # Plot discarded percentage vs bin width

    plt.figure(figsize=(8, 5))
    plt.plot(bin_seconds, discarded_percentages, marker='o')
    plt.xlabel("Time Bin Width [s]")
    plt.ylabel("Kept Events [%]")
    plt.title("Kept Percentage vs. Accumulation Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# These must match in length
standard_deviations = np.array(standard_deviations)

# Apply condition
cond = standard_deviations < 100
bin_seconds = np.array(bin_seconds)[cond]
standard_deviations = standard_deviations[cond]


# Power-law decay + offset model
def power_law_model(t, A, beta, C):
    return A * (t ** -beta) + C

# Fit
popt, _ = curve_fit(
    power_law_model,
    bin_seconds,
    standard_deviations,
    p0=[30, 0.5, 1],
    bounds=([0, 0, 0], [100, 3, 10]),
    maxfev=10000
)

# Print fit parameters
print(f"Fit parameters: A={popt[0]:.2f}, β={popt[1]:.2f}, C={popt[2]:.2f}")

global_variables['coeff_variation_model'] = "A * (t ** -beta) + C"
global_variables['coeff_variation_A'] = float(popt[0])
global_variables['coeff_variation_beta'] = float(popt[1])
global_variables['coeff_variation_C'] = float(popt[2])

if create_essential_plots:
    # Generate curve
    t_fit = np.linspace(min(bin_seconds), max(bin_seconds), 500)
    y_fit = power_law_model(t_fit, *popt)

    # Optional: convert to minutes for x-axis
    time_to_min = False
    if time_to_min:
        bin_seconds = bin_seconds / 60
        t_fit = t_fit / 60
        unit = "min"
    else:
        unit = "s"
    
    plt.figure(figsize=(8, 5))
    plt.scatter(bin_seconds, standard_deviations, s=1, label='Data')
    plt.plot(t_fit, y_fit, 'r-', label=f'Fit: A={popt[0]:.2f}, β={popt[1]:.2f}, C={popt[2]:.2f}')
    plt.axhline(y=5, color='green', linestyle='--', linewidth=1.5, label='5% Coefficient of Variation')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='1% Coefficient of Variation')
    plt.xlabel(f"Time Bin Width [{unit}]")
    plt.ylabel("Coefficient of variation [%]")
    plt.title("Coeff. of variation vs. Accumulation Time")
    plt.grid(True)
    plt.ylim(0, 20)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%

print("----------------------------------------------------------------------")
print("--------------------- Outlier filtering and saving --------------------")
print("----------------------------------------------------------------------")

df = df_original.copy()

# --- 1. Floor to seconds for outlier detection ---
df['Time'] = df['Time'].dt.floor('1s')

# --- 2. Count events per second per (tt, region) ---
binned = df.groupby(['Time', 'definitive_tt', 'region']).size().reset_index(name='counts')
pivoted = binned.pivot_table(index='Time', columns=['definitive_tt', 'region'], values='counts', fill_value=0)
pivoted.columns = [f"{tt}_{region}" for tt, region in pivoted.columns]
pivoted['events'] = pivoted.sum(axis=1)
pivoted = pivoted.reset_index()

# --- 3. Empirical quantile outlier detection ---
lower_q = 0.005  # 0.5%
upper_q = 0.995  # 99.5%

lower = pivoted['events'].quantile(lower_q)
upper = pivoted['events'].quantile(upper_q)

outlier_mask = (pivoted['events'] < lower) | (pivoted['events'] > upper)
n_outliers = outlier_mask.sum()

outliers_percentage = 100 * n_outliers / len(pivoted)
print(f"Outliers detected: {n_outliers} ({outliers_percentage:.2f}%) using empirical quantiles in [{lower:.2f}, {upper:.2f}].")
global_variables['outliers_removed_percentage'] = outliers_percentage

# --- 4. Remove rows from df corresponding to outlier seconds ---
outlier_times = set(pivoted.loc[outlier_mask, 'Time'])
df = df[~df['Time'].isin(outlier_times)]

# --- 5. Floor to 1-minute for accumulation ---
df['Time'] = df['Time'].dt.floor('1min')

# --- 6. Accumulate event counts per minute ---
binned = df.groupby(['Time', 'definitive_tt', 'region']).size().reset_index(name='counts')
pivoted = binned.pivot_table(index='Time', columns=['definitive_tt', 'region'], values='counts', fill_value=0)
pivoted.columns = [f"{tt}_{region}" for tt, region in pivoted.columns]
pivoted['events'] = pivoted.sum(axis=1)
pivoted = pivoted.reset_index()

# --- 7. Save to disk ---
pivoted.to_csv(save_path, index=False, sep=',', float_format='%.5g')
print(f"Accumulated columns datafile saved in {save_filename}. Path is {save_path}")

#%%

# Move the original file in file_path to completed_directory
print("Moving file to COMPLETED directory...")
shutil.move(file_path, completed_file_path)

now = time.time()
os.utime(completed_file_path, (now, now))

print(f"File moved to: {completed_file_path}")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Save the metadata, calibrations and monitoring stuff ------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if side_calculations:
    if eff_vs_angle_and_pos:
        # print(df_fits)
        # print("\n")
        
        for _, row in df_fits.iterrows():
            if not row["label"].startswith("3-plane"):
                continue                      # skip every 2-fold curve

            if "eff_2" in row["label"]:
                tag = "P2"                    # 3-plane efficiency for plane 2
            elif "eff_3" in row["label"]:
                tag = "P3"                    # 3-plane efficiency for plane 3
            else:
                continue

            global_variables[f"eff_{tag}_a"]   = float(row["a"])
            global_variables[f"eff_{tag}_n"]   = float(row["n"])
            global_variables[f"eff_{tag}_0"]   = float(row["eps0"])   # ε₀
        

    if polya_fit:
        # print(df_polya_fit)
        # print("\n")
        
        required = {
            "nbar/alpha"   : "nbar_over_alpha",
            "offset/nbar"  : "offset_over_nbar",
            "alpha/nbar"   : "alpha_over_nbar",
            "eta_curvature": "eta_curvature",
            "width_proxy"  : "width_proxy",
            "Q_mode"       : "Q_mode",
        }

        # df_polya_fit must contain a column named "plane" holding P1…P4
        for plane, grp in df_polya_fit.groupby("module"):
            assert len(grp) == 1, f"multiple rows for {plane}"
            row = grp.iloc[0]
            for col, key_suffix in required.items():
                global_variables[f"{plane}_{key_suffix}"] = float(row[col])
        
        
    if multiplicity_calculations:
        # print(df_mult_fit)
        # print("\n")
        
        for plane, arr in component_counts.items():
            for i, val in enumerate(arr, 1):  # M1, M2, ...
                global_variables[f"{plane}_M{i}"] = float(val)
        
        
    if crosstalk_probability:
        # print(df_cross_fit)
        # print("\n")
        
        eps = 1e-9

        df_cross_fit[["plane", "strip"]] = df_cross_fit["key"].str.split("_", expand=True)
        for plane, grp in df_cross_fit.groupby("plane"):
            # x0
            mean_x0 = grp["x0"].mean()
            w_x0    = 1.0 / (np.abs(grp["x0"] - mean_x0) + eps)
            x0_avg  = np.average(grp["x0"], weights=w_x0)

            # k
            mean_k  = grp["k"].mean()
            w_k     = 1.0 / (np.abs(grp["k"]  - mean_k)  + eps)
            k_avg   = np.average(grp["k"],  weights=w_k)

            # store
            global_variables[f"{plane}_x0"] = float(x0_avg)
            global_variables[f"{plane}_k"]  = float(k_avg)


# Construct the new calibration row
new_row = {'Start_Time': start_time, 'End_Time': end_time}

# Add global variables (e.g., counts, sigmoid widths, slopes)
for key, value in global_variables.items():
    new_row[key] = value

# Load or initialize metadata DataFrame
if os.path.exists(csv_path):
    metadata_df = pd.read_csv(csv_path, parse_dates=['Start_Time', 'End_Time'])
else:
    metadata_df = pd.DataFrame(columns=new_row.keys())

# Find full match in both Start_Time and End_Time
match = (
    (metadata_df['Start_Time'] == start_time) &
    (metadata_df['End_Time'] == end_time)
)
existing_row_index = metadata_df[match].index

if not existing_row_index.empty:
    metadata_df.loc[existing_row_index[0]] = new_row
    print(f"Updated existing calibration for time range: {start_time} to {end_time}")
else:
    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added new calibration for time range: {start_time} to {end_time}")

# Sort and save
metadata_df.sort_values(by='Start_Time', inplace=True)

# Put Start_Time and End_Time as first columns
metadata_df = metadata_df[['Start_Time', 'End_Time'] + [col for col in metadata_df.columns if col not in ['Start_Time', 'End_Time']]]

metadata_df.to_csv(csv_path, index=False, float_format='%.5g')
print(f'{csv_path} updated with the calibration summary.')


print("----------------------------------------------------------------------")
print("--------------------------- Saving the PDF ---------------------------")
print("----------------------------------------------------------------------")

if create_pdf:
    if len(plot_list) > 0:
        with PdfPages(save_pdf_path) as pdf:
            if plot_list:
                for png in plot_list:
                    if os.path.exists(png) == False:
                        print(f"Error: {png} does not exist.")
                        continue
                    
                    # Open the PNG file directly using PIL to get its dimensions
                    img = Image.open(png)
                    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)  # Set figsize and dpi
                    ax.imshow(img)
                    ax.axis('off')  # Hide the axes
                    pdf.savefig(fig, bbox_inches='tight')  # Save figure tightly fitting the image
                    plt.close(fig)  # Close the figure after adding it to the PDF

        # Remove PNG files after creating the PDF
        for png in plot_list:
            try:
                os.remove(png)
                # print(f"Deleted {png}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")

# Erase the figure_directory
if os.path.exists(figure_directory):
    print("Removing figure directory...")
    os.rmdir(figure_directory)

print("event_accumulator.py finished.\n\n")