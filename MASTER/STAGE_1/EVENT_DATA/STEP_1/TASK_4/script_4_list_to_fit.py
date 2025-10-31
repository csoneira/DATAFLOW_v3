#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""




task_number = 4


import sys
from pathlib import Path

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

from MASTER.common.config_loader import update_config_with_parameters
from MASTER.common.execution_logger import set_station, start_timer
from MASTER.common.plot_utils import pdf_save_rasterized_page
from MASTER.common.status_csv import append_status_row, mark_status_complete

from datetime import datetime

# import glob
# import pandas as pd
# import random
# import os
# import sys

# # Pick a random file in "/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/cleaned_<file>.h5"
# IN_PATH = glob.glob("/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_3/DONE/listed_*.h5")[random.randint(0, len(glob.glob("/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_3/DONE/listed_*.h5")) - 1)]
# KEY = "df"

# # Load dataframe
# working_df = pd.read_hdf(IN_PATH, key=KEY)
# print(f"Listed dataframe reloaded from: {IN_PATH}")

# # --- Continue your calibration or analysis code here ---
# # e.g.:
# # run_calibration(working_df)


# # Take basename of IN_PATH without extension and witouth the 'listed_' prefix
# basename_no_ext = os.path.splitext(os.path.basename(IN_PATH))[0].replace("listed_", "")
# print(f"File basename (no extension): {basename_no_ext}")


# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()


# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import os
import re
import csv
import math
import random
import gc
import shutil
import builtins
import warnings
import time
from datetime import timedelta
from collections import defaultdict
from itertools import combinations
from functools import reduce
from typing import Dict, Tuple, Iterable, List

# Scientific Computing
from math import sqrt
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, curve_fit, minimize_scalar
from scipy.special import erf
from scipy.stats import (
    norm,
    poisson,
    linregress,
    median_abs_deviation,
    skew
)

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

# Image Processing
from PIL import Image

# Progress Bar
from tqdm import tqdm

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")

import yaml

start_timer(__file__)
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
parameter_config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters.csv")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
try:
    config = update_config_with_parameters(config, parameter_config_file_path, station)
except NameError:
    pass
home_path = config["home_path"]




def save_metadata(metadata_path: str, row: Dict[str, object]) -> Path:
    """Append the execution metadata row to the per-task CSV."""
    metadata_path = Path(metadata_path)
    fieldnames = list(row.keys())
    file_exists = metadata_path.exists()
    write_header = not file_exists
    if file_exists:
        try:
            write_header = metadata_path.stat().st_size == 0
        except OSError:
            write_header = True
    with metadata_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return metadata_path


def plot_histograms_and_gaussian(df, columns, title, figure_number, quantile=0.99, fit_gaussian=False):
    global fig_idx
    nrows, ncols = (2, 3) if figure_number == 1 else (3, 4)
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
    axs = axs.flatten()
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Precompute quantiles for faster filtering
    if fit_gaussian:
        quantile_bounds = {}
        for col in columns:
            data = df[col].values
            data = data[data != 0]
            if len(data) > 0:
                quantile_bounds[col] = np.quantile(data, [(1 - quantile), quantile])

    # Plot histograms and fit Gaussian if needed
    for i, col in enumerate(columns):
        
        data = df[col].values
        data = data[data != 0]  # Filter out zero values

        if len(data) == 0:  # Skip if no data
            axs[i].text(0.5, 0.5, "No data", transform=axs[i].transAxes, ha='center', va='center', color='gray')
            continue

        # Example color map per column type
        color_map = {
            "theta": "blue",
            "phi": "green",
            "x": "darkorange",
            "y": "darkorange",
            "alt_y": "darkorange",
            "s": "purple",
            "alt_s": "purple",
            "th_chi": "red",
            "res_ystr": "teal",
            "res_tsum": "brown",
            "res_tdif": "purple",
            "t0": "black"
        }

        # Set default in case no match is found
        selected_col = 'gray'

        if "theta" in col:
            left, right = theta_left_filter, theta_right_filter
            selected_col = color_map["theta"]

        elif "phi" in col:
            left, right = phi_left_filter, phi_right_filter
            selected_col = color_map["phi"]

        elif col in ["x", "alt_x", "y", "alt_y"]:
            left, right = -pos_filter, pos_filter
            selected_col = color_map["x"]

        elif col in ["s", "alt_s"]:
            left, right = slowness_filter_left, slowness_filter_right
            selected_col = color_map["s"]

        elif "th_chi" in col:
            left, right = 0, 10
            selected_col = color_map["th_chi"]

        elif "res_ystr" in col:
            left, right = -res_ystr_filter, res_ystr_filter
            selected_col = color_map["res_ystr"]

        elif "res_tsum" in col:
            left, right = -res_tsum_filter, res_tsum_filter
            selected_col = color_map["res_tsum"]

        elif "res_tdif" in col:
            left, right = -res_tdif_filter, res_tdif_filter
            selected_col = color_map["res_tdif"]

        elif "t0" in col:
            left, right = t0_left_filter, t0_right_filter
            selected_col = color_map["t0"]

        # Plot histogram
        cond = (data > left) & (data < right)
        hist_data, bin_edges, _ = axs[i].hist(data, bins=50, alpha=0.7, label='Data', color=selected_col)

        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        
        axs[i].set_xlim([left, right])

        # Fit Gaussian if enabled and data is sufficient
        if fit_gaussian and len(data) >= 10:
            try:
                # Use precomputed quantile bounds
                if col in quantile_bounds:
                    lower_bound, upper_bound = quantile_bounds[col]
                    filt_data = data[(data >= lower_bound) & (data <= upper_bound)]

                if len(filt_data) < 2:
                    axs[i].text(0.5, 0.5, "Not enough data to fit", transform=axs[i].transAxes, ha='center', va='center', color='gray')
                    continue

                # Fit Gaussian to the histogram data
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(filt_data), np.std(filt_data), max(hist_data)])
                mu, sigma, amplitude = popt

                # Plot Gaussian fit
                x = np.linspace(lower_bound, upper_bound, 1000)
                axs[i].plot(x, gaussian(x, mu, sigma, amplitude), 'r-', label=f'Gaussian Fit\nμ={mu:.2g}, σ={sigma:.2g}')
                axs[i].legend()
            except (RuntimeError, ValueError):
                axs[i].text(0.5, 0.5, "Fit failed", transform=axs[i].transAxes, ha='center', va='center', color='red')

    # Remove unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(title, fontsize=16)
    if save_plots:
        final_filename = f'{fig_idx}_{title.replace(" ", "_")}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------


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

if station not in ["1", "2", "3", "4"]:
    print("Error: Invalid station. Please provide a valid station (1, 2, 3, or 4).")
    sys.exit(1)
# print(f"Station: {station}")

set_station(station)
config = update_config_with_parameters(config, parameter_config_file_path, station)

if len(sys.argv) == 3:
    user_file_path = sys.argv[2]
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False

print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
home_directory = os.path.expanduser(f"~")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_1/EVENT_DATA")
raw_to_list_working_directory = os.path.join(base_directory, f"STEP_1/TASK_{task_number}")

metadata_directory = os.path.join(raw_to_list_working_directory, "METADATA")

if task_number == 1:
    raw_directory = "STAGE_0_to_1"
else:
    raw_directory = f"STEP_1/TASK_{task_number - 1}/OUTPUT_FILES"
if task_number == 5:
    output_location = os.path.join(base_directory, "STEP_1_TO_2_OUTPUT")
else:
    output_location = os.path.join(raw_to_list_working_directory, "OUTPUT_FILES")
raw_working_directory = os.path.join(base_directory, raw_directory)

# /home/mingo/DATAFLOW_v3/STATIONS/MINGO01/STAGE_1/EVENT_DATA/STEP_1/TASK_1/OUTPUT_FILES
raw_working_directory = os.path.join(base_directory, "STEP_1/TASK_3/OUTPUT_FILES")

raw_to_list_working_directory = os.path.join(base_directory, "STEP_1/TASK_4/")

# Define directory paths relative to base_directory
base_directories = {
    "stratos_list_events_directory": os.path.join(home_directory, "STRATOS_XY_DIRECTORY"),
    
    "base_plots_directory": os.path.join(raw_to_list_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(raw_to_list_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(raw_to_list_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(raw_to_list_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "ancillary_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY"),
    
    "empty_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/EMPTY_FILES"),
    "rejected_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/REJECTED_FILES"),
    "temp_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/TEMP_FILES"),
    
    "unprocessed_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/UNPROCESSED_DIRECTORY"),
    "error_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/ERROR_DIRECTORY"),
    "processing_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/PROCESSING_DIRECTORY"),
    "completed_directory": os.path.join(raw_to_list_working_directory, "INPUT_FILES/COMPLETED_DIRECTORY"),
    "output_directory": os.path.join(raw_to_list_working_directory, "OUTPUT_FILES"),
    
    "raw_directory": os.path.join(raw_working_directory, "."),
    
    "metadata_directory": metadata_directory,
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

csv_path = os.path.join(metadata_directory, f"step_{task_number}_metadata_execution.csv")
csv_path_specific = os.path.join(metadata_directory, f"step_{task_number}_metadata_specific.csv")

# status_csv_path = os.path.join(base_directory, "raw_to_list_status.csv")
# status_timestamp = append_status_row(status_csv_path)

# Move files from STAGE_0_to_1 to STAGE_0_to_1_TO_LIST/STAGE_0_to_1_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

raw_directory = base_directories["raw_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
error_directory = base_directories["error_directory"]
stratos_list_events_directory = base_directories["stratos_list_events_directory"]
processing_directory = base_directories["processing_directory"]
completed_directory = base_directories["completed_directory"]
output_directory = base_directories["output_directory"]

empty_files_directory = base_directories["empty_files_directory"]
rejected_files_directory = base_directories["rejected_files_directory"]
temp_files_directory = base_directories["temp_files_directory"]

raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))


last_file_test = False


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------- Data reading and preprocessing ---------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Get lists of files in the directories
unprocessed_files = sorted(os.listdir(base_directories["unprocessed_directory"]))
processing_files = sorted(os.listdir(base_directories["processing_directory"]))
completed_files = sorted(os.listdir(base_directories["completed_directory"]))

def process_file(source_path, dest_path):
    print("Source path:", source_path)
    print("Destination path:", dest_path)
    
    if source_path == dest_path:
        return True
    
    if os.path.exists(dest_path):
        print(f"File already exists at destination (removing...)")
        os.remove(dest_path)
        # return False
    
    print("**********************************************************************")
    print(f"Moving\n'{source_path}'\nto\n'{dest_path}'...")
    print("**********************************************************************")
    
    shutil.move(source_path, dest_path)
    now = time.time()
    os.utime(dest_path, (now, now))
    return True

def get_file_path(directory, file_name):
    return os.path.join(directory, file_name)

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)




# status_csv_path = os.path.join(base_directory, "raw_to_list_status.csv")
# status_timestamp = append_status_row(status_csv_path)

# Move files from STAGE_0_to_1 to STAGE_0_to_1_TO_LIST/STAGE_0_to_1_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

raw_directory = base_directories["raw_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
error_directory = base_directories["error_directory"]
stratos_list_events_directory = base_directories["stratos_list_events_directory"]
processing_directory = base_directories["processing_directory"]
completed_directory = base_directories["completed_directory"]
output_directory = base_directories["output_directory"]

empty_files_directory = base_directories["empty_files_directory"]
rejected_files_directory = base_directories["rejected_files_directory"]
temp_files_directory = base_directories["temp_files_directory"]

raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))


# Ordered list from highest to lowest priority
LEVELS = [
    completed_directory,
    processing_directory,
    unprocessed_directory,
    raw_directory,
]

station_re = re.compile(r'^mi0(\d).*\.dat$', re.IGNORECASE)

seen = set()
for d in LEVELS:
    d = Path(d)
    if not d.exists():
        continue

    current_files = {p.name for p in d.iterdir() if p.is_file()}

    # ────────────────────────────────────────────────────────────────
    # Remove .dat files whose prefix “mi0X” does not match `station`
    # ────────────────────────────────────────────────────────────────
    mismatched = {
        fname for fname in current_files
        if (m := station_re.match(fname)) and int(m.group(1)) != int(station)
    }
    for fname in mismatched:
        fp = d / fname
        try:
            fp.unlink()
            print(f"Removed wrong-station file: {fp}")
        except FileNotFoundError:
            pass

    current_files -= mismatched

    # ────────────────────────────────────────────────────────────────
    # Remove duplicates lower in the hierarchy
    # ────────────────────────────────────────────────────────────────
    duplicates = current_files & seen
    for fname in duplicates:
        fp = d / fname
        try:
            fp.unlink()
            print(f"Removed duplicate: {fp}")
        except FileNotFoundError:
            pass

    seen |= (current_files - duplicates)


# Search in all this directories for empty files and move them to the empty_files_directory
for directory in [raw_directory, unprocessed_directory, processing_directory, completed_directory]:
    files = os.listdir(directory)
    for file in files:
        file_empty = os.path.join(directory, file)
        if os.path.getsize(file_empty) == 0:
            # Ensure the empty files directory exists
            os.makedirs(empty_files_directory, exist_ok=True)
            
            # Define the destination path for the file
            empty_destination_path = os.path.join(empty_files_directory, file)
            
            # Remove the destination file if it already exists
            if os.path.exists(empty_destination_path):
                os.remove(empty_destination_path)
            
            print("Moving empty file:", file)
            shutil.move(file_empty, empty_destination_path)
            now = time.time()
            os.utime(empty_destination_path, (now, now))


# Files to move: in STAGE_0_to_1 but not in UNPROCESSED, PROCESSING, or COMPLETED
raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

files_to_move = raw_files - unprocessed_files - processing_files - completed_files

# Move files to UNPROCESSED ---------------------------------------------------------------
for file_name in files_to_move:
    src_path = os.path.join(raw_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        shutil.move(src_path, dest_path)
        now = time.time()
        os.utime(dest_path, (now, now))
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to move {file_name}: {e}")


# Erase all files in the figure_directory -------------------------------------------------
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory)

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))

# Define input file path ------------------------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    if not input_file.empty:
        print("Input configuration file found and is not empty.")
        exists_input_file = True
    else:
        print("Input configuration file is empty.")
        exists_input_file = False
    
    # Print the head
    # print(input_file.head())
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")
    z_1 = 0
    z_2 = 150
    z_3 = 300
    z_4 = 450



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
            # file_name = unprocessed_files[-1]
            file_name = unprocessed_files[0]
            
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
            
            print(f"Processing the last file in PROCESSING:\n    {processing_file_path}")
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

                print(f"Processing the last file in PROCESSING: {processing_file_path}")
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

the_filename = os.path.basename(file_path)
print(f"File to process: {the_filename}")

basename_no_ext, file_extension = os.path.splitext(the_filename)
# Take basename of IN_PATH without extension and witouth the 'listed_' prefix
basename_no_ext = the_filename.replace("listed_", "").replace(".h5", "")

print(f"File basename (no extension): {basename_no_ext}")


analysis_date = datetime.now().strftime("%Y-%m-%d")
print(f"Analysis date and time: {analysis_date}")

# Modify the time of the processing file to the current time so it looks fresh
now = time.time()
os.utime(processing_file_path, (now, now))

# Check the station number in the datafile
try:
    file_station_number = int(basename_no_ext[3])  # 4th character (index 3)
    if file_station_number != int(station):
        print(f'File station number is: {file_station_number}, it does not match.')
        # Move the file to the ERROR directory
        error_file_path = os.path.join(base_directories["error_directory"], file_name)
        print(f"Moving file '{file_name}' to ERROR directory: {error_file_path}")
        process_file(file_path, error_file_path)
        sys.exit(f"File '{file_name}' does not belong to station {station}. Exiting.")
except ValueError:
    sys.exit(f"Invalid station number in file '{file_name}'. Exiting.")


left_limit_time = pd.to_datetime("1-1-2000", format='%d-%m-%Y')
right_limit_time = pd.to_datetime("1-1-2100", format='%d-%m-%Y')

# if limit:
#     print(f'Taking the first {limit_number} rows.')



# Read the data file into a DataFrame


import glob
import pandas as pd
import random
import os
import sys

KEY = "df"

# Load dataframe
working_df = pd.read_parquet(file_path, engine="pyarrow")
print(f"Listed dataframe reloaded from: {file_path}")


# List all names of columns

# Original number of events
original_number_of_events = len(working_df)



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
try:
    config = update_config_with_parameters(config, parameter_config_file_path, station)
except NameError:
    pass
home_path = config["home_path"]

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/INPUT_FILES/itineraries.csv"
)


def load_itineraries_from_file(file_path: Path, required: bool = True) -> list[list[str]]:
    """Return itineraries stored as comma-separated lines in *file_path*."""
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        print(f"Loading itineraries from {file_path}:")
        for line_number, raw_line in enumerate(itinerary_file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                print(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")

    return itineraries


def write_itineraries_to_file(
    file_path: Path,
    itineraries: Iterable[Iterable[str]],
) -> None:
    """Persist unique itineraries to *file_path* as comma-separated lines."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_itineraries: dict[tuple[str, ...], None] = {}

    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if not itinerary_tuple:
            continue
        unique_itineraries.setdefault(itinerary_tuple, None)

    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique_itineraries:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")



not_use_q_semisum = False

stratos_save = config["stratos_save"]
fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]
alternative_fitting = config["alternative_fitting"]

# Accessing all the variables from the configuration
crontab_execution = config["crontab_execution"]
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]
limit = config["limit"]
limit_number = config["limit_number"]
number_of_time_cal_figures = config["number_of_time_cal_figures"]
save_calibrations = config["save_calibrations"]
presentation = config["presentation"]
presentation_plots = config["presentation_plots"]
force_replacement = config["force_replacement"]
article_format = config["article_format"]

# Charge calibration to fC
calibrate_charge_ns_to_fc = config["calibrate_charge_ns_to_fc"]

# Charge front-back
charge_front_back = config["charge_front_back"]

# Slewing correction
slewing_correction = config["slewing_correction"]

# Time filtering
time_window_filtering = config["time_window_filtering"]

# Time calibration
time_calibration = config["time_calibration"]
old_timing_method = config["old_timing_method"]
brute_force_analysis_time_calibration_path_finding = config["brute_force_analysis_time_calibration_path_finding"]

# Y position
y_position_complex_method = config["y_position_complex_method"]
uniform_y_method = config["uniform_y_method"]
uniform_weighted_method = config["uniform_weighted_method"]

# RPC variables
y_new_method = config["y_new_method"]
blur_y = config["blur_y"]

# Alternative
alternative_iteration = config["alternative_iteration"]
number_of_alt_executions = config["number_of_alt_executions"]

# TimTrack
fixed_speed = config["fixed_speed"]
res_ana_removing_planes = config["res_ana_removing_planes"]
timtrack_iteration = config["timtrack_iteration"]
number_of_TT_executions = config["number_of_TT_executions"]

# Validation
validate_charge_pedestal_calibration = config["validate_charge_pedestal_calibration"]

EXPECTED_COLUMNS_config = config["EXPECTED_COLUMNS_config"]

residual_plots = config["residual_plots"]
residual_plots_fast = config["residual_plots_fast"]
residual_plots_debug = config["residual_plots_debug"]

timtrack_iteration = config["timtrack_iteration"]
timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration = config["time_calibration"]
time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back = config["charge_front_back"]
charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]



limit = config["limit"]
limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

limit_number = config["limit_number"]
limit_number_fast = config["limit_number_fast"]
limit_number_debug = config["limit_number_debug"]

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config["T_side_left_pre_cal_debug"]
T_side_right_pre_cal_debug = config["T_side_right_pre_cal_debug"]
Q_side_left_pre_cal_debug = config["Q_side_left_pre_cal_debug"]
Q_side_right_pre_cal_debug = config["Q_side_right_pre_cal_debug"]

T_side_left_pre_cal_default = config["T_side_left_pre_cal_default"]
T_side_right_pre_cal_default = config["T_side_right_pre_cal_default"]
Q_side_left_pre_cal_default = config["Q_side_left_pre_cal_default"]
Q_side_right_pre_cal_default = config["Q_side_right_pre_cal_default"]

T_side_left_pre_cal_ST = config["T_side_left_pre_cal_ST"]
T_side_right_pre_cal_ST = config["T_side_right_pre_cal_ST"]
Q_side_left_pre_cal_ST = config["Q_side_left_pre_cal_ST"]
Q_side_right_pre_cal_ST = config["Q_side_right_pre_cal_ST"]

# Pre-cal Sum & Diff
Q_left_pre_cal = config["Q_left_pre_cal"]
Q_right_pre_cal = config["Q_right_pre_cal"]
Q_diff_pre_cal_threshold = config["Q_diff_pre_cal_threshold"]
T_sum_left_pre_cal = config["T_sum_left_pre_cal"]
T_sum_right_pre_cal = config["T_sum_right_pre_cal"]
T_diff_pre_cal_threshold = config["T_diff_pre_cal_threshold"]

# Post-calibration
Q_sum_left_cal = config["Q_sum_left_cal"]
Q_sum_right_cal = config["Q_sum_right_cal"]
Q_diff_cal_threshold = config["Q_diff_cal_threshold"]
Q_diff_cal_threshold_FB = config["Q_diff_cal_threshold_FB"]
Q_diff_cal_threshold_FB_wide = config["Q_diff_cal_threshold_FB_wide"]
T_sum_left_cal = config["T_sum_left_cal"]
T_sum_right_cal = config["T_sum_right_cal"]
T_diff_cal_threshold = config["T_diff_cal_threshold"]

# Once calculated the RPC variables
T_sum_RPC_left = config["T_sum_RPC_left"]
T_sum_RPC_right = config["T_sum_RPC_right"]
T_diff_RPC_left = config["T_diff_RPC_left"]
T_diff_RPC_right = config["T_diff_RPC_right"]
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_left = config["Q_dif_RPC_left"]
Q_dif_RPC_right = config["Q_dif_RPC_right"]
Y_RPC_left = config["Y_RPC_left"]
Y_RPC_right = config["Y_RPC_right"]

# Alternative fitter filter
alt_pos_filter = config["alt_pos_filter"]
alt_theta_left_filter = config["alt_theta_left_filter"]
alt_theta_right_filter = config["alt_theta_right_filter"]
alt_phi_left_filter = config["alt_phi_left_filter"]
alt_phi_right_filter = config["alt_phi_right_filter"]
alt_slowness_filter_left = config["alt_slowness_filter_left"]
alt_slowness_filter_right = config["alt_slowness_filter_right"]

alt_res_ystr_filter = config["alt_res_ystr_filter"]
alt_res_tsum_filter = config["alt_res_tsum_filter"]
alt_res_tdif_filter = config["alt_res_tdif_filter"]

# TimTrack filter
proj_filter = config["proj_filter"]
res_ystr_filter = config["res_ystr_filter"]
res_tsum_filter = config["res_tsum_filter"]
res_tdif_filter = config["res_tdif_filter"]
ext_res_ystr_filter = config["ext_res_ystr_filter"]
ext_res_tsum_filter = config["ext_res_tsum_filter"]
ext_res_tdif_filter = config["ext_res_tdif_filter"]

# Fitting comparison
delta_s_left = config["delta_s_left"]
delta_s_right = config["delta_s_right"]

# Calibrations
CRT_gaussian_fit_quantile = config["CRT_gaussian_fit_quantile"]
coincidence_window_og_ns = config["coincidence_window_og_ns"]
coincidence_window_precal_ns = config["coincidence_window_precal_ns"]
coincidence_window_cal_ns = config["coincidence_window_cal_ns"]
coincidence_window_cal_number_of_points = config["coincidence_window_cal_number_of_points"]

# Pedestal charge calibration
pedestal_left = config["pedestal_left"]
pedestal_right = config["pedestal_right"]

# Front-back charge
distance_sum_charges_left_fit = config["distance_sum_charges_left_fit"]
distance_sum_charges_right_fit = config["distance_sum_charges_right_fit"]
distance_diff_charges_up_fit = config["distance_diff_charges_up_fit"]
distance_diff_charges_low_fit = config["distance_diff_charges_low_fit"]
distance_sum_charges_plot = config["distance_sum_charges_plot"]
front_back_fit_threshold = config["front_back_fit_threshold"]

# Variables to modify
beta = config["beta"]
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]
validate_pos_cal = config["validate_pos_cal"]

output_order = config["output_order"]
degree_of_polynomial = config["degree_of_polynomial"]

# X
strip_length = config["strip_length"]
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
d0 = config["d0"]
cocut = config["cocut"]
iter_max = config["iter_max"]
anc_sy = config["anc_sy"]
anc_sts = config["anc_sts"]
anc_std = config["anc_std"]
anc_sz = config["anc_sz"]

n_planes_timtrack = config["n_planes_timtrack"]

# Plotting options
T_clip_min_debug = config["T_clip_min_debug"]
T_clip_max_debug = config["T_clip_max_debug"]
Q_clip_min_debug = config["Q_clip_min_debug"]
Q_clip_max_debug = config["Q_clip_max_debug"]
num_bins_debug = config["num_bins_debug"]

T_clip_min_default = config["T_clip_min_default"]
T_clip_max_default = config["T_clip_max_default"]
Q_clip_min_default = config["Q_clip_min_default"]
Q_clip_max_default = config["Q_clip_max_default"]
num_bins_default = config["num_bins_default"]

T_clip_min_ST = config["T_clip_min_ST"]
T_clip_max_ST = config["T_clip_max_ST"]
Q_clip_min_ST = config["Q_clip_min_ST"]
Q_clip_max_ST = config["Q_clip_max_ST"]

log_scale = config["log_scale"]

calibrate_strip_Q_pedestal_thr_factor = config["calibrate_strip_Q_pedestal_thr_factor"]
calibrate_strip_Q_pedestal_thr_factor_2 = config["calibrate_strip_Q_pedestal_thr_factor_2"]
calibrate_strip_Q_pedestal_translate_charge_cal = config["calibrate_strip_Q_pedestal_translate_charge_cal"]

calibrate_strip_Q_pedestal_percentile = config["calibrate_strip_Q_pedestal_percentile"]
calibrate_strip_Q_pedestal_rel_th = config["calibrate_strip_Q_pedestal_rel_th"]
calibrate_strip_Q_pedestal_rel_th_cal = config["calibrate_strip_Q_pedestal_rel_th_cal"]
calibrate_strip_Q_pedestal_abs_th = config["calibrate_strip_Q_pedestal_abs_th"]
calibrate_strip_Q_pedestal_q_quantile = config["calibrate_strip_Q_pedestal_q_quantile"]

scatter_2d_and_fit_new_xlim_left = config["scatter_2d_and_fit_new_xlim_left"]
scatter_2d_and_fit_new_xlim_right = config["scatter_2d_and_fit_new_xlim_right"]
scatter_2d_and_fit_new_ylim_bottom = config["scatter_2d_and_fit_new_ylim_bottom"]
scatter_2d_and_fit_new_ylim_top = config["scatter_2d_and_fit_new_ylim_top"]

calibrate_strip_T_diff_T_rel_th = config["calibrate_strip_T_diff_T_rel_th"]
calibrate_strip_T_diff_T_abs_th = config["calibrate_strip_T_diff_T_abs_th"]

interpolate_fast_charge_Q_clip_min = config["interpolate_fast_charge_Q_clip_min"]
interpolate_fast_charge_Q_clip_max = config["interpolate_fast_charge_Q_clip_max"]
interpolate_fast_charge_num_bins = config["interpolate_fast_charge_num_bins"]
interpolate_fast_charge_log_scale = config["interpolate_fast_charge_log_scale"]

crosstalk_fitting = config["crosstalk_fitting"]
delta_t_left = config["delta_t_left"]
delta_t_right = config["delta_t_right"]
q_sum_left = config["q_sum_left"]
q_sum_right = config["q_sum_right"]
q_diff_left = config["q_diff_left"]
q_diff_right = config["q_diff_right"]

Q_sum_semidiff_left = config["Q_sum_semidiff_left"]
Q_sum_semidiff_right = config["Q_sum_semidiff_right"]
Q_sum_semisum_left = config["Q_sum_semisum_left"]
Q_sum_semisum_right = config["Q_sum_semisum_right"]
T_sum_corrected_diff_left = config["T_sum_corrected_diff_left"]
T_sum_corrected_diff_right = config["T_sum_corrected_diff_right"]
slewing_residual_range = config["slewing_residual_range"]

t_comparison_lim = config["t_comparison_lim"]
t0_time_cal_lim = config["t0_time_cal_lim"]

crosstalk_fit_mu_max = config["crosstalk_fit_mu_max"]
crosstalk_fit_sigma_min = config["crosstalk_fit_sigma_min"]
crosstalk_fit_sigma_max = config["crosstalk_fit_sigma_max"]

slewing_correction_r2_threshold = config["slewing_correction_r2_threshold"]

time_window_fitting = config["time_window_fitting"]

charge_plot_limit_left = config["charge_plot_limit_left"]
charge_plot_limit_right = config["charge_plot_limit_right"]
charge_plot_event_limit_right = config["charge_plot_event_limit_right"]


# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------
Q_sum_color = 'orange'
Q_diff_color = 'red'
T_sum_color = 'blue'
T_diff_color = 'green'

pos_filter = alt_pos_filter
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = alt_slowness_filter_left
slowness_filter_right = alt_slowness_filter_right

theta_left_filter = alt_theta_left_filter
theta_right_filter = alt_theta_right_filter
phi_left_filter = alt_phi_left_filter
phi_right_filter = alt_phi_right_filter

fig_idx = 1
plot_list = []

# Time dif calibration (time_dif_reference)
time_dif_distance = 30
time_dif_reference = np.array([
    [-0.0573, 0.031275, 1.033875, 0.761475],
    [-0.914, -0.873975, -0.19815, 0.452025],
    [0.8769, 1.2008, 1.014, 2.43915],
    [1.508825, 2.086375, 1.6876, 3.023575]
])

# Charge sum pedestal (charge_sum_reference)
charge_sum_distance = 30
charge_sum_reference = np.array([
    [89.4319, 98.19605, 95.99055, 91.83875],
    [96.55775, 94.50385, 94.9254, 91.0775],
    [92.12985, 92.23395, 90.60545, 95.5214],
    [93.75635, 93.57425, 93.07055, 89.27305]
])

# Charge dif calibration (charge_dif_reference)
charge_dif_distance = 30
charge_dif_reference = np.array([
    [4.512, 0.58715, 1.3204, -1.3918],
    [-4.50885, 0.918, -3.39445, -0.12325],
    [-3.8931, -3.28515, 3.27295, 1.0554],
    [-2.29505, 0.012, 2.49045, -2.14565]
])

# Time sum calibration (time_sum_reference)
time_sum_distance = 30
time_sum_reference = np.array([
    [0.0, -0.3886308, -0.53020947, 0.33711737],
    [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
    [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
    [0.33586385, 1.08329847, 0.91410244, 0.58815813]
])

if fast_mode:
    print('Working in fast mode.')
    residual_plots = residual_plots_fast
    timtrack_iteration = timtrack_iteration_fast
    time_calibration = time_calibration_fast
    charge_front_back = charge_front_back_fast
    create_plots = create_plots_fast
    limit = limit_fast
    limit_number = limit_number_fast

if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = timtrack_iteration_debug
    time_calibration = time_calibration_debug
    charge_front_back = charge_front_back_debug
    create_plots = create_plots_debug
    limit = limit_debug
    limit_number = limit_number_debug

if debug_mode:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

    Q_F_left_pre_cal = Q_side_left_pre_cal_debug
    Q_F_right_pre_cal = Q_side_right_pre_cal_debug

    Q_B_left_pre_cal = Q_side_left_pre_cal_debug
    Q_B_right_pre_cal = Q_side_right_pre_cal_debug
else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

    Q_F_left_pre_cal = Q_side_left_pre_cal_default
    Q_F_right_pre_cal = Q_side_right_pre_cal_default

    Q_B_left_pre_cal = Q_side_left_pre_cal_default
    Q_B_right_pre_cal = Q_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST
Q_F_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_F_right_pre_cal_ST = Q_side_right_pre_cal_ST
Q_B_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_B_right_pre_cal_ST = Q_side_right_pre_cal_ST

Q_left_side = Q_side_left_pre_cal_ST
Q_right_side = Q_side_right_pre_cal_ST



# Y ---------------------------------------------------------------------------
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

c_mm_ns = c/1000000
print(c_mm_ns)

# Miscelanous ----------------------------
muon_speed = beta * c_mm_ns
strip_speed = strip_speed_factor_of_c * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Not-Hardcoded
vc    = beta * c_mm_ns # mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
nplan = n_planes_timtrack
lenx  = strip_length
anc_sx = tdiff_to_x * anc_std # 2 cm

if debug_mode:
    T_clip_min = T_clip_min_debug
    T_clip_max = T_clip_max_debug
    Q_clip_min = Q_clip_min_debug
    Q_clip_max = Q_clip_max_debug
    num_bins = num_bins_debug
else:
    T_clip_min = T_clip_min_default
    T_clip_max = T_clip_max_default
    Q_clip_min = Q_clip_min_default
    Q_clip_max = Q_clip_max_default
    num_bins = num_bins_default

T_clip_min_ST = T_clip_min_ST
T_clip_max_ST = T_clip_max_ST
Q_clip_min_ST = Q_clip_min_ST
Q_clip_max_ST = Q_clip_max_ST

global_variables = {
    'execution_time': execution_time,
    'CRT_avg': 0,
    'one_side_events': 0,
    'purity_of_data_percentage': 0,
    'unc_y': anc_sy,
    'unc_tsum': anc_sts,
    'unc_tdif': anc_std,
    'time_window_filtering': time_window_filtering*1,
    'old_timing_method': old_timing_method*1,
}







station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")

# Define input file path ------------------------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    if not input_file.empty:
        print("Input configuration file found and is not empty.")
        exists_input_file = True
    else:
        print("Input configuration file is empty.")
        exists_input_file = False
    
    # Print the head
    # print(input_file.head())
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")
    z_1 = 0
    z_2 = 150
    z_3 = 300
    z_4 = 450


self_trigger = False







# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

# Standard Library
import os
import re
import sys
import csv
import math
import random
import gc
import shutil
import builtins
import warnings
import time
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations
from functools import reduce
from typing import Dict, Tuple, Iterable, List
from pathlib import Path

# Scientific Computing
from math import sqrt
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, curve_fit, minimize_scalar
from scipy.special import erf
from scipy.stats import (
    norm,
    poisson,
    linregress,
    median_abs_deviation,
    skew
)

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

# Image Processing
from PIL import Image

# Progress Bar
from tqdm import tqdm

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")

import yaml

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
from MASTER.common.plot_utils import pdf_save_rasterized_page
from MASTER.common.status_csv import append_status_row, mark_status_complete

start_timer(__file__)
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
try:
    config = update_config_with_parameters(config, parameter_config_file_path, station)
except NameError:
    pass
home_path = config["home_path"]




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
try:
    config = update_config_with_parameters(config, parameter_config_file_path, station)
except NameError:
    pass
home_path = config["home_path"]

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/INPUT_FILES/itineraries.csv"
)


def load_itineraries_from_file(file_path: Path, required: bool = True) -> list[list[str]]:
    """Return itineraries stored as comma-separated lines in *file_path*."""
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        print(f"Loading itineraries from {file_path}:")
        for line_number, raw_line in enumerate(itinerary_file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                print(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")

    return itineraries


def write_itineraries_to_file(
    file_path: Path,
    itineraries: Iterable[Iterable[str]],
) -> None:
    """Persist unique itineraries to *file_path* as comma-separated lines."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_itineraries: dict[tuple[str, ...], None] = {}

    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if not itinerary_tuple:
            continue
        unique_itineraries.setdefault(itinerary_tuple, None)

    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique_itineraries:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")



not_use_q_semisum = False

stratos_save = config["stratos_save"]
fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]
alternative_fitting = config["alternative_fitting"]

# Accessing all the variables from the configuration
crontab_execution = config["crontab_execution"]
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]
limit = config["limit"]
limit_number = config["limit_number"]
number_of_time_cal_figures = config["number_of_time_cal_figures"]
save_calibrations = config["save_calibrations"]
presentation = config["presentation"]
presentation_plots = config["presentation_plots"]
force_replacement = config["force_replacement"]
article_format = config["article_format"]

# Charge calibration to fC
calibrate_charge_ns_to_fc = config["calibrate_charge_ns_to_fc"]

# Charge front-back
charge_front_back = config["charge_front_back"]

# Slewing correction
slewing_correction = config["slewing_correction"]

# Time filtering
time_window_filtering = config["time_window_filtering"]

# Time calibration
time_calibration = config["time_calibration"]
old_timing_method = config["old_timing_method"]
brute_force_analysis_time_calibration_path_finding = config["brute_force_analysis_time_calibration_path_finding"]

# Y position
y_position_complex_method = config["y_position_complex_method"]
uniform_y_method = config["uniform_y_method"]
uniform_weighted_method = config["uniform_weighted_method"]

# RPC variables
y_new_method = config["y_new_method"]
blur_y = config["blur_y"]

# Alternative
alternative_iteration = config["alternative_iteration"]
number_of_alt_executions = config["number_of_alt_executions"]

# TimTrack
fixed_speed = config["fixed_speed"]
res_ana_removing_planes = config["res_ana_removing_planes"]
timtrack_iteration = config["timtrack_iteration"]
number_of_TT_executions = config["number_of_TT_executions"]

# Validation
validate_charge_pedestal_calibration = config["validate_charge_pedestal_calibration"]

EXPECTED_COLUMNS_config = config["EXPECTED_COLUMNS_config"]

residual_plots = config["residual_plots"]
residual_plots_fast = config["residual_plots_fast"]
residual_plots_debug = config["residual_plots_debug"]

timtrack_iteration = config["timtrack_iteration"]
timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration = config["time_calibration"]
time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back = config["charge_front_back"]
charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]



limit = config["limit"]
limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

limit_number = config["limit_number"]
limit_number_fast = config["limit_number_fast"]
limit_number_debug = config["limit_number_debug"]

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config["T_side_left_pre_cal_debug"]
T_side_right_pre_cal_debug = config["T_side_right_pre_cal_debug"]
Q_side_left_pre_cal_debug = config["Q_side_left_pre_cal_debug"]
Q_side_right_pre_cal_debug = config["Q_side_right_pre_cal_debug"]

T_side_left_pre_cal_default = config["T_side_left_pre_cal_default"]
T_side_right_pre_cal_default = config["T_side_right_pre_cal_default"]
Q_side_left_pre_cal_default = config["Q_side_left_pre_cal_default"]
Q_side_right_pre_cal_default = config["Q_side_right_pre_cal_default"]

T_side_left_pre_cal_ST = config["T_side_left_pre_cal_ST"]
T_side_right_pre_cal_ST = config["T_side_right_pre_cal_ST"]
Q_side_left_pre_cal_ST = config["Q_side_left_pre_cal_ST"]
Q_side_right_pre_cal_ST = config["Q_side_right_pre_cal_ST"]

# Pre-cal Sum & Diff
Q_left_pre_cal = config["Q_left_pre_cal"]
Q_right_pre_cal = config["Q_right_pre_cal"]
Q_diff_pre_cal_threshold = config["Q_diff_pre_cal_threshold"]
T_sum_left_pre_cal = config["T_sum_left_pre_cal"]
T_sum_right_pre_cal = config["T_sum_right_pre_cal"]
T_diff_pre_cal_threshold = config["T_diff_pre_cal_threshold"]

# Post-calibration
Q_sum_left_cal = config["Q_sum_left_cal"]
Q_sum_right_cal = config["Q_sum_right_cal"]
Q_diff_cal_threshold = config["Q_diff_cal_threshold"]
Q_diff_cal_threshold_FB = config["Q_diff_cal_threshold_FB"]
Q_diff_cal_threshold_FB_wide = config["Q_diff_cal_threshold_FB_wide"]
T_sum_left_cal = config["T_sum_left_cal"]
T_sum_right_cal = config["T_sum_right_cal"]
T_diff_cal_threshold = config["T_diff_cal_threshold"]

# Once calculated the RPC variables
T_sum_RPC_left = config["T_sum_RPC_left"]
T_sum_RPC_right = config["T_sum_RPC_right"]
T_diff_RPC_left = config["T_diff_RPC_left"]
T_diff_RPC_right = config["T_diff_RPC_right"]
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_left = config["Q_dif_RPC_left"]
Q_dif_RPC_right = config["Q_dif_RPC_right"]
Y_RPC_left = config["Y_RPC_left"]
Y_RPC_right = config["Y_RPC_right"]

# Alternative fitter filter
alt_pos_filter = config["alt_pos_filter"]
alt_theta_left_filter = config["alt_theta_left_filter"]
alt_theta_right_filter = config["alt_theta_right_filter"]
alt_phi_left_filter = config["alt_phi_left_filter"]
alt_phi_right_filter = config["alt_phi_right_filter"]
alt_slowness_filter_left = config["alt_slowness_filter_left"]
alt_slowness_filter_right = config["alt_slowness_filter_right"]

alt_res_ystr_filter = config["alt_res_ystr_filter"]
alt_res_tsum_filter = config["alt_res_tsum_filter"]
alt_res_tdif_filter = config["alt_res_tdif_filter"]

# TimTrack filter
proj_filter = config["proj_filter"]
res_ystr_filter = config["res_ystr_filter"]
res_tsum_filter = config["res_tsum_filter"]
res_tdif_filter = config["res_tdif_filter"]
ext_res_ystr_filter = config["ext_res_ystr_filter"]
ext_res_tsum_filter = config["ext_res_tsum_filter"]
ext_res_tdif_filter = config["ext_res_tdif_filter"]

# Fitting comparison
delta_s_left = config["delta_s_left"]
delta_s_right = config["delta_s_right"]

# Calibrations
CRT_gaussian_fit_quantile = config["CRT_gaussian_fit_quantile"]
coincidence_window_og_ns = config["coincidence_window_og_ns"]
coincidence_window_precal_ns = config["coincidence_window_precal_ns"]
coincidence_window_cal_ns = config["coincidence_window_cal_ns"]
coincidence_window_cal_number_of_points = config["coincidence_window_cal_number_of_points"]

# Pedestal charge calibration
pedestal_left = config["pedestal_left"]
pedestal_right = config["pedestal_right"]

# Front-back charge
distance_sum_charges_left_fit = config["distance_sum_charges_left_fit"]
distance_sum_charges_right_fit = config["distance_sum_charges_right_fit"]
distance_diff_charges_up_fit = config["distance_diff_charges_up_fit"]
distance_diff_charges_low_fit = config["distance_diff_charges_low_fit"]
distance_sum_charges_plot = config["distance_sum_charges_plot"]
front_back_fit_threshold = config["front_back_fit_threshold"]

# Variables to modify
beta = config["beta"]
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]
validate_pos_cal = config["validate_pos_cal"]

output_order = config["output_order"]
degree_of_polynomial = config["degree_of_polynomial"]

# X
strip_length = config["strip_length"]
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
d0 = config["d0"]
cocut = config["cocut"]
iter_max = config["iter_max"]
anc_sy = config["anc_sy"]
anc_sts = config["anc_sts"]
anc_std = config["anc_std"]
anc_sz = config["anc_sz"]

n_planes_timtrack = config["n_planes_timtrack"]

# Plotting options
T_clip_min_debug = config["T_clip_min_debug"]
T_clip_max_debug = config["T_clip_max_debug"]
Q_clip_min_debug = config["Q_clip_min_debug"]
Q_clip_max_debug = config["Q_clip_max_debug"]
num_bins_debug = config["num_bins_debug"]

T_clip_min_default = config["T_clip_min_default"]
T_clip_max_default = config["T_clip_max_default"]
Q_clip_min_default = config["Q_clip_min_default"]
Q_clip_max_default = config["Q_clip_max_default"]
num_bins_default = config["num_bins_default"]

T_clip_min_ST = config["T_clip_min_ST"]
T_clip_max_ST = config["T_clip_max_ST"]
Q_clip_min_ST = config["Q_clip_min_ST"]
Q_clip_max_ST = config["Q_clip_max_ST"]

log_scale = config["log_scale"]

calibrate_strip_Q_pedestal_thr_factor = config["calibrate_strip_Q_pedestal_thr_factor"]
calibrate_strip_Q_pedestal_thr_factor_2 = config["calibrate_strip_Q_pedestal_thr_factor_2"]
calibrate_strip_Q_pedestal_translate_charge_cal = config["calibrate_strip_Q_pedestal_translate_charge_cal"]

calibrate_strip_Q_pedestal_percentile = config["calibrate_strip_Q_pedestal_percentile"]
calibrate_strip_Q_pedestal_rel_th = config["calibrate_strip_Q_pedestal_rel_th"]
calibrate_strip_Q_pedestal_rel_th_cal = config["calibrate_strip_Q_pedestal_rel_th_cal"]
calibrate_strip_Q_pedestal_abs_th = config["calibrate_strip_Q_pedestal_abs_th"]
calibrate_strip_Q_pedestal_q_quantile = config["calibrate_strip_Q_pedestal_q_quantile"]

scatter_2d_and_fit_new_xlim_left = config["scatter_2d_and_fit_new_xlim_left"]
scatter_2d_and_fit_new_xlim_right = config["scatter_2d_and_fit_new_xlim_right"]
scatter_2d_and_fit_new_ylim_bottom = config["scatter_2d_and_fit_new_ylim_bottom"]
scatter_2d_and_fit_new_ylim_top = config["scatter_2d_and_fit_new_ylim_top"]

calibrate_strip_T_diff_T_rel_th = config["calibrate_strip_T_diff_T_rel_th"]
calibrate_strip_T_diff_T_abs_th = config["calibrate_strip_T_diff_T_abs_th"]

interpolate_fast_charge_Q_clip_min = config["interpolate_fast_charge_Q_clip_min"]
interpolate_fast_charge_Q_clip_max = config["interpolate_fast_charge_Q_clip_max"]
interpolate_fast_charge_num_bins = config["interpolate_fast_charge_num_bins"]
interpolate_fast_charge_log_scale = config["interpolate_fast_charge_log_scale"]

crosstalk_fitting = config["crosstalk_fitting"]
delta_t_left = config["delta_t_left"]
delta_t_right = config["delta_t_right"]
q_sum_left = config["q_sum_left"]
q_sum_right = config["q_sum_right"]
q_diff_left = config["q_diff_left"]
q_diff_right = config["q_diff_right"]

Q_sum_semidiff_left = config["Q_sum_semidiff_left"]
Q_sum_semidiff_right = config["Q_sum_semidiff_right"]
Q_sum_semisum_left = config["Q_sum_semisum_left"]
Q_sum_semisum_right = config["Q_sum_semisum_right"]
T_sum_corrected_diff_left = config["T_sum_corrected_diff_left"]
T_sum_corrected_diff_right = config["T_sum_corrected_diff_right"]
slewing_residual_range = config["slewing_residual_range"]

t_comparison_lim = config["t_comparison_lim"]
t0_time_cal_lim = config["t0_time_cal_lim"]

crosstalk_fit_mu_max = config["crosstalk_fit_mu_max"]
crosstalk_fit_sigma_min = config["crosstalk_fit_sigma_min"]
crosstalk_fit_sigma_max = config["crosstalk_fit_sigma_max"]

slewing_correction_r2_threshold = config["slewing_correction_r2_threshold"]

time_window_fitting = config["time_window_fitting"]

charge_plot_limit_left = config["charge_plot_limit_left"]
charge_plot_limit_right = config["charge_plot_limit_right"]
charge_plot_event_limit_right = config["charge_plot_event_limit_right"]


# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------
Q_sum_color = 'orange'
Q_diff_color = 'red'
T_sum_color = 'blue'
T_diff_color = 'green'

pos_filter = alt_pos_filter
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = alt_slowness_filter_left
slowness_filter_right = alt_slowness_filter_right

theta_left_filter = alt_theta_left_filter
theta_right_filter = alt_theta_right_filter
phi_left_filter = alt_phi_left_filter
phi_right_filter = alt_phi_right_filter

fig_idx = 1
plot_list = []

# Time dif calibration (time_dif_reference)
time_dif_distance = 30
time_dif_reference = np.array([
    [-0.0573, 0.031275, 1.033875, 0.761475],
    [-0.914, -0.873975, -0.19815, 0.452025],
    [0.8769, 1.2008, 1.014, 2.43915],
    [1.508825, 2.086375, 1.6876, 3.023575]
])

# Charge sum pedestal (charge_sum_reference)
charge_sum_distance = 30
charge_sum_reference = np.array([
    [89.4319, 98.19605, 95.99055, 91.83875],
    [96.55775, 94.50385, 94.9254, 91.0775],
    [92.12985, 92.23395, 90.60545, 95.5214],
    [93.75635, 93.57425, 93.07055, 89.27305]
])

# Charge dif calibration (charge_dif_reference)
charge_dif_distance = 30
charge_dif_reference = np.array([
    [4.512, 0.58715, 1.3204, -1.3918],
    [-4.50885, 0.918, -3.39445, -0.12325],
    [-3.8931, -3.28515, 3.27295, 1.0554],
    [-2.29505, 0.012, 2.49045, -2.14565]
])

# Time sum calibration (time_sum_reference)
time_sum_distance = 30
time_sum_reference = np.array([
    [0.0, -0.3886308, -0.53020947, 0.33711737],
    [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
    [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
    [0.33586385, 1.08329847, 0.91410244, 0.58815813]
])

if fast_mode:
    print('Working in fast mode.')
    residual_plots = residual_plots_fast
    timtrack_iteration = timtrack_iteration_fast
    time_calibration = time_calibration_fast
    charge_front_back = charge_front_back_fast
    create_plots = create_plots_fast
    limit = limit_fast
    limit_number = limit_number_fast

if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = timtrack_iteration_debug
    time_calibration = time_calibration_debug
    charge_front_back = charge_front_back_debug
    create_plots = create_plots_debug
    limit = limit_debug
    limit_number = limit_number_debug

if debug_mode:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

    Q_F_left_pre_cal = Q_side_left_pre_cal_debug
    Q_F_right_pre_cal = Q_side_right_pre_cal_debug

    Q_B_left_pre_cal = Q_side_left_pre_cal_debug
    Q_B_right_pre_cal = Q_side_right_pre_cal_debug
else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

    Q_F_left_pre_cal = Q_side_left_pre_cal_default
    Q_F_right_pre_cal = Q_side_right_pre_cal_default

    Q_B_left_pre_cal = Q_side_left_pre_cal_default
    Q_B_right_pre_cal = Q_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST
Q_F_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_F_right_pre_cal_ST = Q_side_right_pre_cal_ST
Q_B_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_B_right_pre_cal_ST = Q_side_right_pre_cal_ST

Q_left_side = Q_side_left_pre_cal_ST
Q_right_side = Q_side_right_pre_cal_ST



# Y ---------------------------------------------------------------------------
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

c_mm_ns = c/1000000
print(c_mm_ns)

# Miscelanous ----------------------------
muon_speed = beta * c_mm_ns
strip_speed = strip_speed_factor_of_c * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Not-Hardcoded
vc    = beta * c_mm_ns # mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
nplan = n_planes_timtrack
lenx  = strip_length
anc_sx = tdiff_to_x * anc_std # 2 cm

if debug_mode:
    T_clip_min = T_clip_min_debug
    T_clip_max = T_clip_max_debug
    Q_clip_min = Q_clip_min_debug
    Q_clip_max = Q_clip_max_debug
    num_bins = num_bins_debug
else:
    T_clip_min = T_clip_min_default
    T_clip_max = T_clip_max_default
    Q_clip_min = Q_clip_min_default
    Q_clip_max = Q_clip_max_default
    num_bins = num_bins_default

T_clip_min_ST = T_clip_min_ST
T_clip_max_ST = T_clip_max_ST
Q_clip_min_ST = Q_clip_min_ST
Q_clip_max_ST = Q_clip_max_ST

global_variables = {
    'execution_time': execution_time,
    'CRT_avg': 0,
    'one_side_events': 0,
    'purity_of_data_percentage': 0,
    'unc_y': anc_sy,
    'unc_tsum': anc_sts,
    'unc_tdif': anc_std,
    'time_window_filtering': time_window_filtering*1,
    'old_timing_method': old_timing_method*1,
}







# -----------------------------------------------------------------------------
# Stuff that could change between mingos --------------------------------------
# -----------------------------------------------------------------------------

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

if station not in ["1", "2", "3", "4"]:
    print("Error: Invalid station. Please provide a valid station (1, 2, 3, or 4).")
    sys.exit(1)
# print(f"Station: {station}")

set_station(station)
config = update_config_with_parameters(config, parameter_config_file_path, station)

if len(sys.argv) == 3:
    user_file_path = sys.argv[2]
    user_file_selection = True
    print("User provided file path:", user_file_path)
else:
    user_file_selection = False


station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")

# Define input file path ------------------------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    if not input_file.empty:
        print("Input configuration file found and is not empty.")
        exists_input_file = True
    else:
        print("Input configuration file is empty.")
        exists_input_file = False
    
    # Print the head
    # print(input_file.head())
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")
    z_1 = 0
    z_2 = 150
    z_3 = 300
    z_4 = 450


self_trigger = False


# Store the current time at the start. To time the execution

# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

global_variables = {
    'execution_time': execution_time,
    'CRT_avg': 0,
    'one_side_events': 0,
    'purity_of_data_percentage': 0,
    'unc_y': anc_sy,
    'unc_tsum': anc_sts,
    'unc_tdif': anc_std,
    'time_window_filtering': time_window_filtering*1,
    'old_timing_method': old_timing_method*1,
}


# Note that the middle between start and end time could also be taken. This is for calibration storage.
datetime_value = working_df['datetime'].iloc[0]
end_datetime_value = working_df['datetime'].iloc[-1]

if self_trigger:
    print(self_trigger_df)
    datetime_value_st = self_trigger_df['datetime'].iloc[0]
    end_datetime_value_st = self_trigger_df['datetime'].iloc[-1]
    datetime_str_st = str(datetime_value_st)
    save_filename_suffix_st = datetime_str_st.replace(' ', "_").replace(':', ".").replace('-', ".")

start_time = datetime_value
end_time = end_datetime_value
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")




print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print(f"------------- Starting date is {save_filename_suffix} -------------------") # This is longer so it displays nicely
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Defining the directories that will store the data
save_full_filename = f"full_list_events_{save_filename_suffix}.txt"
save_filename = f"list_events_{save_filename_suffix}.txt"
save_pdf_filename = f"pdf_{save_filename_suffix}.pdf"

if create_plots == False:
    if create_essential_plots == True:
        print("Creating essential plots, modifying the PDF filename.")
        save_pdf_filename = "essential_" + save_pdf_filename

save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)






# -------------------------------------------------------------------------------
# ------------ Input file and data managing to select configuration -------------
# -------------------------------------------------------------------------------

if exists_input_file:
    # Ensure `start` and `end` columns are in datetime format
    input_file["start"] = pd.to_datetime(input_file["start"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = pd.to_datetime(input_file["end"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = input_file["end"].fillna(pd.to_datetime('now'))
    matching_confs = input_file[ (input_file["start"] <= start_time) & (input_file["end"] >= end_time) ]
    print(matching_confs)
    
    if not matching_confs.empty:
        if len(matching_confs) > 1:
            print(f"Warning:\nMultiple configurations match the date range\n{start_time} to {end_time}.\nTaking the first one.")
        selected_conf = matching_confs.iloc[0]
        print(f"Selected configuration: {selected_conf['conf']}")
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])
        found_matching_conf = True
        print(selected_conf['conf'])
    else:
        print("Error: No matching configuration found for the given date range. Using default z_positions.")
        found_matching_conf = False
        z_positions = np.array([0, 150, 300, 450])  # In mm
else:
    print("Error: No input file. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm


# If any of the z_positions is NaN, use default values
if np.isnan(z_positions).any():
    print("Error: Incomplete z_positions in the selected configuration. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm


# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

# Save the z_positions in the metadata file
global_variables['z_P1'] =  z_positions[0]
global_variables['z_P2'] =  z_positions[1]
global_variables['z_P3'] =  z_positions[2]
global_variables['z_P4'] =  z_positions[3]


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


import os
import yaml
user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
try:
    config = update_config_with_parameters(config, parameter_config_file_path, station)
except NameError:
    pass
home_path = config["home_path"]

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/ANCILLARY/INPUT_FILES/itineraries.csv"
)


def load_itineraries_from_file(file_path: Path, required: bool = True) -> list[list[str]]:
    """Return itineraries stored as comma-separated lines in *file_path*."""
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"Cannot find itineraries file: {file_path}")
        return []

    itineraries: list[list[str]] = []
    with file_path.open("r", encoding="utf-8") as itinerary_file:
        print(f"Loading itineraries from {file_path}:")
        for line_number, raw_line in enumerate(itinerary_file, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            segments = [segment.strip() for segment in stripped_line.split(",") if segment.strip()]
            if segments:
                itineraries.append(segments)
                print(segments)

    if not itineraries and required:
        raise ValueError(f"Itineraries file {file_path} is empty.")

    return itineraries


def write_itineraries_to_file(
    file_path: Path,
    itineraries: Iterable[Iterable[str]],
) -> None:
    """Persist unique itineraries to *file_path* as comma-separated lines."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_itineraries: dict[tuple[str, ...], None] = {}

    for itinerary in itineraries:
        itinerary_tuple = tuple(itinerary)
        if not itinerary_tuple:
            continue
        unique_itineraries.setdefault(itinerary_tuple, None)

    with file_path.open("w", encoding="utf-8") as itinerary_file:
        for itinerary_tuple in unique_itineraries:
            itinerary_file.write(",".join(itinerary_tuple) + "\n")



not_use_q_semisum = False

stratos_save = config["stratos_save"]
fast_mode = config["fast_mode"]
debug_mode = config["debug_mode"]
last_file_test = config["last_file_test"]
alternative_fitting = config["alternative_fitting"]

# Accessing all the variables from the configuration
crontab_execution = config["crontab_execution"]
create_plots = config["create_plots"]
create_essential_plots = config["create_essential_plots"]
save_plots = config["save_plots"]
show_plots = config["show_plots"]
create_pdf = config["create_pdf"]
limit = config["limit"]
limit_number = config["limit_number"]
number_of_time_cal_figures = config["number_of_time_cal_figures"]
save_calibrations = config["save_calibrations"]
presentation = config["presentation"]
presentation_plots = config["presentation_plots"]
force_replacement = config["force_replacement"]
article_format = config["article_format"]

# Charge calibration to fC
calibrate_charge_ns_to_fc = config["calibrate_charge_ns_to_fc"]

# Charge front-back
charge_front_back = config["charge_front_back"]

# Slewing correction
slewing_correction = config["slewing_correction"]

# Time filtering
time_window_filtering = config["time_window_filtering"]

# Time calibration
time_calibration = config["time_calibration"]
old_timing_method = config["old_timing_method"]
brute_force_analysis_time_calibration_path_finding = config["brute_force_analysis_time_calibration_path_finding"]

# Y position
y_position_complex_method = config["y_position_complex_method"]
uniform_y_method = config["uniform_y_method"]
uniform_weighted_method = config["uniform_weighted_method"]

# RPC variables
y_new_method = config["y_new_method"]
blur_y = config["blur_y"]

# Alternative
alternative_iteration = config["alternative_iteration"]
number_of_alt_executions = config["number_of_alt_executions"]

# TimTrack
fixed_speed = config["fixed_speed"]
res_ana_removing_planes = config["res_ana_removing_planes"]
timtrack_iteration = config["timtrack_iteration"]
number_of_TT_executions = config["number_of_TT_executions"]

# Validation
validate_charge_pedestal_calibration = config["validate_charge_pedestal_calibration"]

EXPECTED_COLUMNS_config = config["EXPECTED_COLUMNS_config"]

residual_plots = config["residual_plots"]
residual_plots_fast = config["residual_plots_fast"]
residual_plots_debug = config["residual_plots_debug"]

timtrack_iteration = config["timtrack_iteration"]
timtrack_iteration_fast = config["timtrack_iteration_fast"]
timtrack_iteration_debug = config["timtrack_iteration_debug"]

time_calibration = config["time_calibration"]
time_calibration_fast = config["time_calibration_fast"]
time_calibration_debug = config["time_calibration_debug"]

charge_front_back = config["charge_front_back"]
charge_front_back_fast = config["charge_front_back_fast"]
charge_front_back_debug = config["charge_front_back_debug"]

create_plots = config["create_plots"]



limit = config["limit"]
limit_fast = config["limit_fast"]
limit_debug = config["limit_debug"]

limit_number = config["limit_number"]
limit_number_fast = config["limit_number_fast"]
limit_number_debug = config["limit_number_debug"]

# Pre-cal Front & Back
T_side_left_pre_cal_debug = config["T_side_left_pre_cal_debug"]
T_side_right_pre_cal_debug = config["T_side_right_pre_cal_debug"]
Q_side_left_pre_cal_debug = config["Q_side_left_pre_cal_debug"]
Q_side_right_pre_cal_debug = config["Q_side_right_pre_cal_debug"]

T_side_left_pre_cal_default = config["T_side_left_pre_cal_default"]
T_side_right_pre_cal_default = config["T_side_right_pre_cal_default"]
Q_side_left_pre_cal_default = config["Q_side_left_pre_cal_default"]
Q_side_right_pre_cal_default = config["Q_side_right_pre_cal_default"]

T_side_left_pre_cal_ST = config["T_side_left_pre_cal_ST"]
T_side_right_pre_cal_ST = config["T_side_right_pre_cal_ST"]
Q_side_left_pre_cal_ST = config["Q_side_left_pre_cal_ST"]
Q_side_right_pre_cal_ST = config["Q_side_right_pre_cal_ST"]

# Pre-cal Sum & Diff
Q_left_pre_cal = config["Q_left_pre_cal"]
Q_right_pre_cal = config["Q_right_pre_cal"]
Q_diff_pre_cal_threshold = config["Q_diff_pre_cal_threshold"]
T_sum_left_pre_cal = config["T_sum_left_pre_cal"]
T_sum_right_pre_cal = config["T_sum_right_pre_cal"]
T_diff_pre_cal_threshold = config["T_diff_pre_cal_threshold"]

# Post-calibration
Q_sum_left_cal = config["Q_sum_left_cal"]
Q_sum_right_cal = config["Q_sum_right_cal"]
Q_diff_cal_threshold = config["Q_diff_cal_threshold"]
Q_diff_cal_threshold_FB = config["Q_diff_cal_threshold_FB"]
Q_diff_cal_threshold_FB_wide = config["Q_diff_cal_threshold_FB_wide"]
T_sum_left_cal = config["T_sum_left_cal"]
T_sum_right_cal = config["T_sum_right_cal"]
T_diff_cal_threshold = config["T_diff_cal_threshold"]

# Once calculated the RPC variables
T_sum_RPC_left = config["T_sum_RPC_left"]
T_sum_RPC_right = config["T_sum_RPC_right"]
T_diff_RPC_left = config["T_diff_RPC_left"]
T_diff_RPC_right = config["T_diff_RPC_right"]
Q_RPC_left = config["Q_RPC_left"]
Q_RPC_right = config["Q_RPC_right"]
Q_dif_RPC_left = config["Q_dif_RPC_left"]
Q_dif_RPC_right = config["Q_dif_RPC_right"]
Y_RPC_left = config["Y_RPC_left"]
Y_RPC_right = config["Y_RPC_right"]

# Alternative fitter filter
alt_pos_filter = config["alt_pos_filter"]
alt_theta_left_filter = config["alt_theta_left_filter"]
alt_theta_right_filter = config["alt_theta_right_filter"]
alt_phi_left_filter = config["alt_phi_left_filter"]
alt_phi_right_filter = config["alt_phi_right_filter"]
alt_slowness_filter_left = config["alt_slowness_filter_left"]
alt_slowness_filter_right = config["alt_slowness_filter_right"]

alt_res_ystr_filter = config["alt_res_ystr_filter"]
alt_res_tsum_filter = config["alt_res_tsum_filter"]
alt_res_tdif_filter = config["alt_res_tdif_filter"]

# TimTrack filter
proj_filter = config["proj_filter"]
res_ystr_filter = config["res_ystr_filter"]
res_tsum_filter = config["res_tsum_filter"]
res_tdif_filter = config["res_tdif_filter"]
ext_res_ystr_filter = config["ext_res_ystr_filter"]
ext_res_tsum_filter = config["ext_res_tsum_filter"]
ext_res_tdif_filter = config["ext_res_tdif_filter"]

# Fitting comparison
delta_s_left = config["delta_s_left"]
delta_s_right = config["delta_s_right"]

# Calibrations
CRT_gaussian_fit_quantile = config["CRT_gaussian_fit_quantile"]
coincidence_window_og_ns = config["coincidence_window_og_ns"]
coincidence_window_precal_ns = config["coincidence_window_precal_ns"]
coincidence_window_cal_ns = config["coincidence_window_cal_ns"]
coincidence_window_cal_number_of_points = config["coincidence_window_cal_number_of_points"]

# Pedestal charge calibration
pedestal_left = config["pedestal_left"]
pedestal_right = config["pedestal_right"]

# Front-back charge
distance_sum_charges_left_fit = config["distance_sum_charges_left_fit"]
distance_sum_charges_right_fit = config["distance_sum_charges_right_fit"]
distance_diff_charges_up_fit = config["distance_diff_charges_up_fit"]
distance_diff_charges_low_fit = config["distance_diff_charges_low_fit"]
distance_sum_charges_plot = config["distance_sum_charges_plot"]
front_back_fit_threshold = config["front_back_fit_threshold"]

# Variables to modify
beta = config["beta"]
strip_speed_factor_of_c = config["strip_speed_factor_of_c"]
validate_pos_cal = config["validate_pos_cal"]

output_order = config["output_order"]
degree_of_polynomial = config["degree_of_polynomial"]

# X
strip_length = config["strip_length"]
narrow_strip = config["narrow_strip"]
wide_strip = config["wide_strip"]

# Timtrack parameters
d0 = config["d0"]
cocut = config["cocut"]
iter_max = config["iter_max"]
anc_sy = config["anc_sy"]
anc_sts = config["anc_sts"]
anc_std = config["anc_std"]
anc_sz = config["anc_sz"]

n_planes_timtrack = config["n_planes_timtrack"]

# Plotting options
T_clip_min_debug = config["T_clip_min_debug"]
T_clip_max_debug = config["T_clip_max_debug"]
Q_clip_min_debug = config["Q_clip_min_debug"]
Q_clip_max_debug = config["Q_clip_max_debug"]
num_bins_debug = config["num_bins_debug"]

T_clip_min_default = config["T_clip_min_default"]
T_clip_max_default = config["T_clip_max_default"]
Q_clip_min_default = config["Q_clip_min_default"]
Q_clip_max_default = config["Q_clip_max_default"]
num_bins_default = config["num_bins_default"]

T_clip_min_ST = config["T_clip_min_ST"]
T_clip_max_ST = config["T_clip_max_ST"]
Q_clip_min_ST = config["Q_clip_min_ST"]
Q_clip_max_ST = config["Q_clip_max_ST"]

log_scale = config["log_scale"]

calibrate_strip_Q_pedestal_thr_factor = config["calibrate_strip_Q_pedestal_thr_factor"]
calibrate_strip_Q_pedestal_thr_factor_2 = config["calibrate_strip_Q_pedestal_thr_factor_2"]
calibrate_strip_Q_pedestal_translate_charge_cal = config["calibrate_strip_Q_pedestal_translate_charge_cal"]

calibrate_strip_Q_pedestal_percentile = config["calibrate_strip_Q_pedestal_percentile"]
calibrate_strip_Q_pedestal_rel_th = config["calibrate_strip_Q_pedestal_rel_th"]
calibrate_strip_Q_pedestal_rel_th_cal = config["calibrate_strip_Q_pedestal_rel_th_cal"]
calibrate_strip_Q_pedestal_abs_th = config["calibrate_strip_Q_pedestal_abs_th"]
calibrate_strip_Q_pedestal_q_quantile = config["calibrate_strip_Q_pedestal_q_quantile"]

scatter_2d_and_fit_new_xlim_left = config["scatter_2d_and_fit_new_xlim_left"]
scatter_2d_and_fit_new_xlim_right = config["scatter_2d_and_fit_new_xlim_right"]
scatter_2d_and_fit_new_ylim_bottom = config["scatter_2d_and_fit_new_ylim_bottom"]
scatter_2d_and_fit_new_ylim_top = config["scatter_2d_and_fit_new_ylim_top"]

calibrate_strip_T_diff_T_rel_th = config["calibrate_strip_T_diff_T_rel_th"]
calibrate_strip_T_diff_T_abs_th = config["calibrate_strip_T_diff_T_abs_th"]

interpolate_fast_charge_Q_clip_min = config["interpolate_fast_charge_Q_clip_min"]
interpolate_fast_charge_Q_clip_max = config["interpolate_fast_charge_Q_clip_max"]
interpolate_fast_charge_num_bins = config["interpolate_fast_charge_num_bins"]
interpolate_fast_charge_log_scale = config["interpolate_fast_charge_log_scale"]

crosstalk_fitting = config["crosstalk_fitting"]
delta_t_left = config["delta_t_left"]
delta_t_right = config["delta_t_right"]
q_sum_left = config["q_sum_left"]
q_sum_right = config["q_sum_right"]
q_diff_left = config["q_diff_left"]
q_diff_right = config["q_diff_right"]

Q_sum_semidiff_left = config["Q_sum_semidiff_left"]
Q_sum_semidiff_right = config["Q_sum_semidiff_right"]
Q_sum_semisum_left = config["Q_sum_semisum_left"]
Q_sum_semisum_right = config["Q_sum_semisum_right"]
T_sum_corrected_diff_left = config["T_sum_corrected_diff_left"]
T_sum_corrected_diff_right = config["T_sum_corrected_diff_right"]
slewing_residual_range = config["slewing_residual_range"]

t_comparison_lim = config["t_comparison_lim"]
t0_time_cal_lim = config["t0_time_cal_lim"]

crosstalk_fit_mu_max = config["crosstalk_fit_mu_max"]
crosstalk_fit_sigma_min = config["crosstalk_fit_sigma_min"]
crosstalk_fit_sigma_max = config["crosstalk_fit_sigma_max"]

slewing_correction_r2_threshold = config["slewing_correction_r2_threshold"]

time_window_fitting = config["time_window_fitting"]

charge_plot_limit_left = config["charge_plot_limit_left"]
charge_plot_limit_right = config["charge_plot_limit_right"]
charge_plot_event_limit_right = config["charge_plot_event_limit_right"]


# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'purity_of_data', etc.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------
Q_sum_color = 'orange'
Q_diff_color = 'red'
T_sum_color = 'blue'
T_diff_color = 'green'

pos_filter = alt_pos_filter
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = alt_slowness_filter_left
slowness_filter_right = alt_slowness_filter_right

theta_left_filter = alt_theta_left_filter
theta_right_filter = alt_theta_right_filter
phi_left_filter = alt_phi_left_filter
phi_right_filter = alt_phi_right_filter

fig_idx = 1
plot_list = []

# Time dif calibration (time_dif_reference)
time_dif_distance = 30
time_dif_reference = np.array([
    [-0.0573, 0.031275, 1.033875, 0.761475],
    [-0.914, -0.873975, -0.19815, 0.452025],
    [0.8769, 1.2008, 1.014, 2.43915],
    [1.508825, 2.086375, 1.6876, 3.023575]
])

# Charge sum pedestal (charge_sum_reference)
charge_sum_distance = 30
charge_sum_reference = np.array([
    [89.4319, 98.19605, 95.99055, 91.83875],
    [96.55775, 94.50385, 94.9254, 91.0775],
    [92.12985, 92.23395, 90.60545, 95.5214],
    [93.75635, 93.57425, 93.07055, 89.27305]
])

# Charge dif calibration (charge_dif_reference)
charge_dif_distance = 30
charge_dif_reference = np.array([
    [4.512, 0.58715, 1.3204, -1.3918],
    [-4.50885, 0.918, -3.39445, -0.12325],
    [-3.8931, -3.28515, 3.27295, 1.0554],
    [-2.29505, 0.012, 2.49045, -2.14565]
])

# Time sum calibration (time_sum_reference)
time_sum_distance = 30
time_sum_reference = np.array([
    [0.0, -0.3886308, -0.53020947, 0.33711737],
    [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
    [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
    [0.33586385, 1.08329847, 0.91410244, 0.58815813]
])

if fast_mode:
    print('Working in fast mode.')
    residual_plots = residual_plots_fast
    timtrack_iteration = timtrack_iteration_fast
    time_calibration = time_calibration_fast
    charge_front_back = charge_front_back_fast
    create_plots = create_plots_fast
    limit = limit_fast
    limit_number = limit_number_fast

if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = timtrack_iteration_debug
    time_calibration = time_calibration_debug
    charge_front_back = charge_front_back_debug
    create_plots = create_plots_debug
    limit = limit_debug
    limit_number = limit_number_debug



if debug_mode:
    T_F_left_pre_cal = T_side_left_pre_cal_debug
    T_F_right_pre_cal = T_side_right_pre_cal_debug

    T_B_left_pre_cal = T_side_left_pre_cal_debug
    T_B_right_pre_cal = T_side_right_pre_cal_debug

    Q_F_left_pre_cal = Q_side_left_pre_cal_debug
    Q_F_right_pre_cal = Q_side_right_pre_cal_debug

    Q_B_left_pre_cal = Q_side_left_pre_cal_debug
    Q_B_right_pre_cal = Q_side_right_pre_cal_debug
else:
    T_F_left_pre_cal = T_side_left_pre_cal_default  #-130
    T_F_right_pre_cal = T_side_right_pre_cal_default

    T_B_left_pre_cal = T_side_left_pre_cal_default
    T_B_right_pre_cal = T_side_right_pre_cal_default

    Q_F_left_pre_cal = Q_side_left_pre_cal_default
    Q_F_right_pre_cal = Q_side_right_pre_cal_default

    Q_B_left_pre_cal = Q_side_left_pre_cal_default
    Q_B_right_pre_cal = Q_side_right_pre_cal_default

T_F_left_pre_cal_ST = T_side_left_pre_cal_ST  #-115
T_F_right_pre_cal_ST = T_side_right_pre_cal_ST
T_B_left_pre_cal_ST = T_side_left_pre_cal_ST
T_B_right_pre_cal_ST = T_side_right_pre_cal_ST
Q_F_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_F_right_pre_cal_ST = Q_side_right_pre_cal_ST
Q_B_left_pre_cal_ST = Q_side_left_pre_cal_ST
Q_B_right_pre_cal_ST = Q_side_right_pre_cal_ST

Q_left_side = Q_side_left_pre_cal_ST
Q_right_side = Q_side_right_pre_cal_ST



# Y ---------------------------------------------------------------------------
y_widths = [np.array([wide_strip, wide_strip, wide_strip, narrow_strip]), 
            np.array([narrow_strip, wide_strip, wide_strip, wide_strip])]

def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)
total_width = np.sum(y_width_P1_and_P3)

c_mm_ns = c/1000000
print(c_mm_ns)

# Miscelanous ----------------------------
muon_speed = beta * c_mm_ns
strip_speed = strip_speed_factor_of_c * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Not-Hardcoded
vc    = beta * c_mm_ns # mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
nplan = n_planes_timtrack
lenx  = strip_length
anc_sx = tdiff_to_x * anc_std # 2 cm

if debug_mode:
    T_clip_min = T_clip_min_debug
    T_clip_max = T_clip_max_debug
    Q_clip_min = Q_clip_min_debug
    Q_clip_max = Q_clip_max_debug
    num_bins = num_bins_debug
else:
    T_clip_min = T_clip_min_default
    T_clip_max = T_clip_max_default
    Q_clip_min = Q_clip_min_default
    Q_clip_max = Q_clip_max_default
    num_bins = num_bins_default

T_clip_min_ST = T_clip_min_ST
T_clip_max_ST = T_clip_max_ST
Q_clip_min_ST = Q_clip_min_ST
Q_clip_max_ST = Q_clip_max_ST

global_variables = {
    'execution_time': execution_time,
    'CRT_avg': 0,
    'one_side_events': 0,
    'purity_of_data_percentage': 0,
    'unc_y': anc_sy,
    'unc_tsum': anc_sts,
    'unc_tdif': anc_std,
    'time_window_filtering': time_window_filtering*1,
    'old_timing_method': old_timing_method*1,
}






raw_data_len = len(working_df)
if raw_data_len == 0 and not self_trigger:
    print("No coincidence nor self-trigger events.")
    sys.exit(1)




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -------- TASK_4: fitting ----------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("-------------- Alternative angle and slowness fitting ----------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# ---------------------------------------------------------------------------
# 1. Geometrical line fit (orthogonal-distance regression) ------------------
# ---------------------------------------------------------------------------

def fit_3d_line(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sx: float,
    sy: float,
    sz: float,
    plane_ids: Iterable[int],
    tdiff_to_x: float,
) -> Tuple[float, float, float, float, float,
           Dict[int, float], Dict[int, float]]:
    """
    Returns
    -------
    x_z0, y_z0              : intercept with z = 0
    theta, phi              : zenith (0 = down-coming) and azimuth  [rad]
    chi2                    : χ² of the ODR
    res_td_dict, res_y_dict : residuals per plane (ΔTdiff units, y units)
    """
    pts = np.column_stack((x, y, z))
    c   = pts.mean(axis=0)
    d   = np.linalg.svd(pts - c, full_matrices=False)[2][0]   # principal axis

    if d[2] < 0:                                              # enforce d_z > 0
        d = -d
    d /= np.linalg.norm(d)

    theta = np.arccos(d[2])
    phi   = np.arctan2(d[1], d[0])

    # z = 0 intercept
    t0  = -c[2] / d[2] if d[2] != 0 else np.nan
    xz0 = c[0] + t0 * d[0]
    yz0 = c[1] + t0 * d[1]

    # orthogonal residual vectors
    proj = np.outer((pts - c) @ d, d)                         # (N,3)
    res  = (pts - c) - proj

    res_td = res[:, 0] / tdiff_to_x
    res_y  = res[:, 1]

    chi2 = np.einsum('ij,ij->', res, res) / (sx**2 + sy**2 + sz**2)
    return (xz0, yz0, float(theta), float(phi), float(chi2), dict(zip(plane_ids, res_td)), dict(zip(plane_ids, res_y)))


# ---------------------------------------------------------------------------
# ---------------------------- Loop starts here -----------------------------
# ---------------------------------------------------------------------------

n = len(working_df)

# Angular definitions
fit_cols = (
    ['alt_x', 'alt_y', 'alt_theta', 'alt_phi', 'alt_chi2'] +
    [f'alt_res_tdif_{p}' for p in range(1, 5)] +
    [f'alt_res_ystr_{p}' for p in range(1, 5)]
)

# Slowness definitions
slow_cols = ['alt_s', 'alt_s_ordinate' , 'chi2_tsum_fit'] + [f'alt_res_tsum_{p}' for p in range(1, 5)]

# Alternative analysis starts -----------------------------------------------
repeat = number_of_alt_executions - 1 if alternative_iteration else 0
for alt_iteration in range(repeat + 1):
    fitted = 0
    if alternative_iteration:
        print(f"Alternative iteration {alt_iteration+1} out of {number_of_alt_executions}.")
    
    fit_res = {c: np.zeros(n, dtype=float) for c in fit_cols}
    slow_res  = {c: np.zeros(n, dtype=float) for c in slow_cols}
    
    for i, trk in enumerate(working_df.itertuples(index=False)):
        planes = [p for p in range(1, nplan + 1)
                if getattr(trk, f'P{p}_Q_sum_final') > 0]
        if len(planes) < 2:
            continue
        
        # Angular part -----------------------------------------------------------------
        x = np.array([tdiff_to_x * getattr(trk, f'P{p}_T_diff_final') for p in planes])
        y = np.array([getattr(trk, f'P{p}_Y_final') for p in planes])
        z = z_positions[np.array(planes) - 1]

        (fit_res['alt_x'][i], fit_res['alt_y'][i], fit_res['alt_theta'][i], fit_res['alt_phi'][i], fit_res['alt_chi2'][i], res_td, res_y) = fit_3d_line(x, y, z, anc_sx, anc_sy, anc_sz, planes, tdiff_to_x)

        for p in range(1, 5):
            fit_res[f'alt_res_tdif_{p}'][i] = res_td.get(p, 0.0)
            fit_res[f'alt_res_ystr_{p}'][i] = res_y .get(p, 0.0)

        # Slowness part ----------------------------------------------------------------
        tsum = np.array([getattr(trk, f'P{p}_T_sum_final') for p in planes])

        # Reconstruct fitted points using the fitted direction and z-positions
        θ, φ = fit_res['alt_theta'][i], fit_res['alt_phi'][i]
        x0, y0 = fit_res['alt_x'][i], fit_res['alt_y'][i]

        v = np.array([np.sin(θ) * np.cos(φ),
                      np.sin(θ) * np.sin(φ),
                      np.cos(θ)])
        v /= np.linalg.norm(v)

        # Compute fitted positions along z
        x_fit = x0 + v[0] * z / v[2]
        y_fit = y0 + v[1] * z / v[2]
        positions = np.stack((x_fit, y_fit, z), axis=1)

        # Distance along the fitted track (scalar projection)
        real_dist = positions @ v
        s_rel = real_dist - real_dist[0]
        t_rel = tsum - tsum[0]

        k, b = np.polyfit(s_rel, t_rel, 1)
        res  = t_rel - (k * s_rel + b)
        chi2 = np.sum((res / anc_sts) ** 2)

        slow_res['alt_s'][i]          = k
        slow_res['alt_s_ordinate'][i] = b
        slow_res['chi2_tsum_fit'][i]  = chi2
        for p, r in zip(planes, res):
            slow_res[f'alt_res_tsum_{p}'][i] = r


    # 4.  Assemble all results and join once
    all_res = {**fit_res, **slow_res}
    all_res['alt_th_chi'] = all_res['alt_chi2'] + all_res['chi2_tsum_fit']

    new_cols = pd.DataFrame(all_res, index=working_df.index)
    dupes = new_cols.columns.intersection(working_df.columns)
    working_df = working_df.drop(columns=dupes, errors='ignore')
    working_df = working_df.join(new_cols)
    working_df = working_df.copy()


    # Filter according to residual ------------------------------------------------
    alt_changed_event_count = 0
    for index, row in working_df.iterrows():
        alt_changed = False
        for i in range(1, 5):
            if abs(row[f'alt_res_tsum_{i}']) > alt_res_tsum_filter or \
                abs(row[f'alt_res_tdif_{i}']) > alt_res_tdif_filter or \
                abs(row[f'alt_res_ystr_{i}']) > alt_res_ystr_filter:
                
                alt_changed = True
                working_df.at[index, f'P{i}_Y_final'] = 0
                working_df.at[index, f'P{i}_T_sum_final'] = 0
                working_df.at[index, f'P{i}_T_diff_final'] = 0
                working_df.at[index, f'P{i}_Q_sum_final'] = 0
                working_df.at[index, f'P{i}_Q_diff_final'] = 0
        if alt_changed:
            alt_changed_event_count += 1
    print(f"--> {alt_changed_event_count} events were residual filtered.")
    
    alt_iteration += 1


# ---------------------------------------------------------------------------
# Put every value close to 0 to effectively 0 -------------------------------
# ---------------------------------------------------------------------------

# Filter the values inside the machine number window ------------------------
eps = 1e-7  # Threshold
def is_small_nonzero(x):
    return isinstance(x, (int, float)) and x != 0 and abs(x) < eps

if create_plots:

    # Flatten all numeric values except 0
    flat_values = working_df.select_dtypes(include=[np.number]).values.ravel()
    flat_values = flat_values[flat_values != 0]

    cond = abs(flat_values) < eps
    flat_values = flat_values[cond]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_values, bins=300, alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.title('Histogram of All Nonzero Values in working_df')
    plt.yscale('log')  # Optional: log scale to reveal structure
    plt.grid(True)
    plt.tight_layout()
    if save_plots:
        name_of_file = 'flat_values_histogram'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


# Filter the small values ----------------------------------------------------
mask = working_df.map(is_small_nonzero)  # Create mask of small, non-zero numeric values
nonzero_numeric_mask = working_df.map(lambda x: isinstance(x, (int, float)) and x != 0)  # Count total non-zero numeric entries
n_total = nonzero_numeric_mask.sum().sum()
n_small = mask.sum().sum()
working_df = working_df.mask(mask, 0)  # Apply the replacement
pct = 100 * n_small / n_total if n_total > 0 else 0
print(f"{n_small} out of {n_total} non-zero numeric values are below {eps} ({pct:.4f}%)")  # Report


for col in working_df.columns:
    # Alternative fitting results
    if 'alt_x' == col or 'alt_y' == col:
        cond_bound = (working_df[col] > alt_pos_filter) | (working_df[col] < -1*alt_pos_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'alt_theta' == col:
        cond_bound = (working_df[col] > alt_theta_right_filter) | (working_df[col] < alt_theta_left_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'alt_phi' == col:
        cond_bound = (working_df[col] > alt_phi_right_filter) | (working_df[col] < alt_phi_left_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'alt_s' == col:
        cond_bound = (working_df[col] > alt_slowness_filter_right) | (working_df[col] < alt_slowness_filter_left)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])

print("Alternative fitting done.")


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("------------------------- TimTrack fitting ---------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

if fixed_speed:
    print("Fixed the slowness to 1 / speed of light.")
    npar = 5
else:
    print("Slowness not fixed.")
    npar = 6


def fmgx(nvar, npar, vs, ss, zi): # G matrix for t measurements in X-axis
    mg = np.zeros([nvar, npar])
    XP = vs[1]; YP = vs[3]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = sqrt(1 + XP*XP + YP*YP)
    kzi = 1 / kz
    mg[0,2] = 1
    mg[0,3] = zi
    mg[1,1] = kzi * S0 * XP * zi
    mg[1,3] = kzi * S0 * YP * zi
    mg[1,4] = 1
    if fixed_speed == False: mg[1,5] = kz * zi
    mg[2,0] = ss
    mg[2,1] = ss * zi
    return mg

def fmwx(nvar, vsig): # Weigth matrix 
    sy = vsig[0]; sts = vsig[1]; std = vsig[2]
    mw = np.zeros([nvar, nvar])
    mw[0,0] = 1/(sy*sy)
    mw[1,1] = 1/(sts*sts)
    mw[2,2] = 1/(std*std)
    return mw

def fvmx(nvar, vs, lenx, ss, zi): # Fitting model array with X-strips
    vm = np.zeros(nvar)
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = np.sqrt(1 + XP*XP + YP*YP)
    xi = X0 + XP * zi
    yi = Y0 + YP * zi
    ti = T0 + kz * S0 * zi
    th = 0.5 * lenx * ss   # tau half
    # lxmn = -lenx/2
    vm[0] = yi
    vm[1] = th + ti
    # vm[2] = ss * (xi-lxmn) - th
    vm[2] = ss * xi
    return vm

def fmkx(nvar, npar, vs, vsig, ss, zi): # K matrix
    mk  = np.zeros([npar,npar])
    mg  = fmgx(nvar, npar, vs, ss, zi)
    mgt = mg.transpose()
    mw  = fmwx(nvar, vsig)
    mk  = mgt @ mw @ mg
    return mk

def fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi): # va vector
    va = np.zeros(npar)
    mw = fmwx(nvar, vsig)
    vm = fvmx(nvar, vs, lenx, ss, zi)
    mg = fmgx(nvar, npar, vs, ss, zi)
    vg = vm - mg @ vs
    vdmg = vdat - vg
    va = mg.transpose() @ mw @ vdmg
    return va

def fmahd(npar, vin1, vin2, merr): # Mahalanobis distance
    vdif  = np.subtract(vin1,vin2)
    vdsq  = np.power(vdif,2)
    verr  = np.diag(merr,0)
    vsig  = np.divide(vdsq,verr)
    dist  = np.sqrt(np.sum(vsig))
    return dist

def fres(vs, vdat, lenx, ss, zi):  # Residuals array
    X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]
    if fixed_speed:
        S0 = sc
    else:
        S0 = vs[5]
    kz = sqrt(1 + XP*XP + YP*YP)
    # Fitted values
    xfit  = X0 + XP * zi
    yfit  = Y0 + YP * zi
    tbfit = T0 + S0 * kz * zi + (lenx/2 + xfit) * ss
    tffit = T0 + S0 * kz * zi + (lenx/2 - xfit) * ss
    tsfit = 0.5 * (tbfit + tffit)
    tdfit = 0.5 * (tbfit - tffit)
    # Data values
    ydat  = vdat[0]
    tsdat = vdat[1]
    tddat = vdat[2]
    # Residuals
    yr   = (yfit  - ydat)
    tsr  = (tsfit - tsdat)
    tdr  = (tdfit - tddat)
    # DeltaX_tsum = abs( (tsdat - ( T0 + S0 * kz * zi ) ) / 0.5 / ss - lenx)
    vres = [yr, tsr, tdr]
    return vres

def extract_plane_data(track, iplane):
    zi  = z_positions[iplane - 1]
    yst = getattr(track, f'P{iplane}_Y_final')
    ts  = getattr(track, f'P{iplane}_T_sum_final')
    td  = getattr(track, f'P{iplane}_T_diff_final')
    return [yst, ts, td], [anc_sy, anc_sts, anc_std], zi

nvar = 3
i = 0
ntrk  = len(working_df)
if limit and limit_number < ntrk: ntrk = limit_number
print("-----------------------------")
print(f"{ntrk} events to be fitted")

timtrack_results = [ 'x', 'xp', 'y', 'yp', 't0', 's',
                'th_chi', 'res_y', 'res_ts', 'res_td', 'processed_tt',
                'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
                'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
                'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4',
                'ext_res_ystr_1', 'ext_res_ystr_2', 'ext_res_ystr_3', 'ext_res_ystr_4',
                'ext_res_tsum_1', 'ext_res_tsum_2', 'ext_res_tsum_3', 'ext_res_tsum_4',
                'ext_res_tdif_1', 'ext_res_tdif_2', 'ext_res_tdif_3', 'ext_res_tdif_4',
                'charge_1', 'charge_2', 'charge_3', 'charge_4', 'charge_event',
                "iterations", "conv_distance", 'converged']

new_columns_df = pd.DataFrame(0., index=working_df.index, columns=timtrack_results)
working_df = pd.concat([working_df, new_columns_df], axis=1)

# TimTrack starts ------------------------------------------------------
repeat = number_of_TT_executions - 1 if timtrack_iteration else 0
for iteration in range(repeat + 1):
    fitted = 0
    if timtrack_iteration:
        print(f"TimTrack iteration {iteration+1} out of {number_of_TT_executions}")
    
    n_rows = len(working_df)
    charge_arr = np.zeros((n_rows, 4), dtype=float)
    res_ystr_arr = np.zeros((n_rows, 4), dtype=float)
    res_tsum_arr = np.zeros((n_rows, 4), dtype=float)
    res_tdif_arr = np.zeros((n_rows, 4), dtype=float)
    ext_res_ystr_arr = np.zeros((n_rows, 4), dtype=float)
    ext_res_tsum_arr = np.zeros((n_rows, 4), dtype=float)
    ext_res_tdif_arr = np.zeros((n_rows, 4), dtype=float)

    charge_event_arr = np.zeros(n_rows, dtype=float)
    iterations_arr = np.zeros(n_rows, dtype=np.int32)
    conv_distance_arr = np.zeros(n_rows, dtype=float)
    converged_arr = np.zeros(n_rows, dtype=np.int8)
    processed_tt_arr = np.zeros(n_rows, dtype=np.int32)

    th_chi_arr = np.zeros(n_rows, dtype=float)
    x_arr = np.zeros(n_rows, dtype=float)
    xp_arr = np.zeros(n_rows, dtype=float)
    y_arr = np.zeros(n_rows, dtype=float)
    yp_arr = np.zeros(n_rows, dtype=float)
    t0_arr = np.zeros(n_rows, dtype=float)
    s_arr = np.zeros(n_rows, dtype=float)

    th_chi_ndf_arrays = {}

    iterator = working_df.itertuples(index=False, name='Track')
    if not crontab_execution:
        iterator = tqdm(iterator, total=working_df.shape[0], desc="Processing events")
    
    for pos, track in enumerate(iterator):
        # INTRODUCTION ------------------------------------------------------------------
        name_type_parts = []
        planes_to_iterate = []
        charge_event = 0.0
        for i_plane in range(nplan):
            plane_id = i_plane + 1
            charge_plane = getattr(track, f'P{plane_id}_Q_sum_final')
            if charge_plane != 0:
                name_type_parts.append(str(plane_id))
                planes_to_iterate.append(plane_id)
                if plane_id <= 4:
                    charge_arr[pos, plane_id - 1] = charge_plane
                charge_event += charge_plane
        
        name_type = int(''.join(name_type_parts)) if name_type_parts else 0
        processed_tt_arr[pos] = name_type
        charge_event_arr[pos] = charge_event
        
        # FITTING -----------------------------------------------------------------------
        if len(planes_to_iterate) <= 1:
            continue
        
        if fixed_speed:
            vs  = np.zeros(5, dtype=float)
        else:
            vs  = np.zeros(6, dtype=float)
            vs[5] = sc
        mk  = np.zeros([npar, npar])
        va  = np.zeros(npar)
        istp = 0   # nb. of fitting steps
        dist = d0
        while dist > cocut and istp < iter_max:
            for iplane in planes_to_iterate:
                
                # Data --------------------------------------------------------
                vdat, vsig, zi = extract_plane_data(track, iplane)
                # -------------------------------------------------------------
                
                mk += fmkx(nvar, npar, vs, vsig, ss, zi)
                va += fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
            istp = istp + 1
            vs0 = vs
            vs = np.linalg.solve(mk, va)  # Solve mk @ vs = va
            merr = np.linalg.inv(mk)      # Only compute if needed for fmahd()
            dist = fmahd(npar, vs, vs0, merr)
            
        if istp >= iter_max or dist >= cocut:
            converged_arr[pos] = 1
        iterations_arr[pos] = istp
        conv_distance_arr[pos] = dist
        
        vsf = vs       # final saeta
        fitted += 1
        
        
        # RESIDUAL ANALYSIS ----------------------------------------------------------------------------
        
        # Fit residuals
        res_ystr = res_tsum = res_tdif = ndat = 0
        
        if len(planes_to_iterate) > 1:
            for iplane in planes_to_iterate:
                
                ndat = ndat + nvar
                
                # Data --------------------------------------------------------------------------------
                vdat, vsig, zi = extract_plane_data(track, iplane)
                # -------------------------------------------------------------------------------------
                
                vres = fres(vsf, vdat, lenx, ss, zi)
                
                res_ystr  = res_ystr  + vres[0]
                res_tsum  = res_tsum  + vres[1]
                res_tdif  = res_tdif  + vres[2]
                
                if iplane <= 4:
                    res_ystr_arr[pos, iplane - 1] = vres[0]
                    res_tsum_arr[pos, iplane - 1] = vres[1]
                    res_tdif_arr[pos, iplane - 1] = vres[2]
            
            processed_tt_arr[pos] = name_type
            
            ndf  = ndat - npar    # number of degrees of freedom; was ndat - npar
            
            chi2 = ( res_ystr / anc_sy )**2 + ( res_tsum / anc_sts )**2 + ( res_tdif / anc_std )**2
            th_chi_arr[pos] = chi2
            th_chi_ndf_arrays.setdefault(ndf, np.zeros(n_rows, dtype=float))[pos] = chi2
            
            x_arr[pos] = vsf[0]
            xp_arr[pos] = vsf[1]
            y_arr[pos] = vsf[2]
            yp_arr[pos] = vsf[3]
            t0_arr[pos] = vsf[4]
            
            if fixed_speed:
                s_arr[pos] = sc
            else:
                s_arr[pos] = vsf[5]
        
        
        # ---------------------------------------------------------------------------------------------
        # Residual analysis with 4-plane tracks (hide a plane and make a fit in the 3 remaining planes)
        # ---------------------------------------------------------------------------------------------
        if len(planes_to_iterate) >= 3 and res_ana_removing_planes:
            
            # for iplane_ref, istrip_ref in zip(planes_to_iterate, istrip_list):
            for iplane_ref in planes_to_iterate:
                
                # Data ------------------------------------------------------------
                vdat_ref, _, z_ref = extract_plane_data(track, iplane_ref)
                # -----------------------------------------------------------------
                
                planes_to_iterate_short = [p for p in planes_to_iterate if p != iplane_ref]
                
                vs     = vsf  # We start with the previous 4-planes fit
                mk     = np.zeros([npar, npar])
                va     = np.zeros(npar)
                istp = 0
                dist = d0
                while dist > cocut and istp < iter_max:
                    for iplane in planes_to_iterate_short:
                    
                        # Data --------------------------------------------------------
                        vdat, vsig, zi = extract_plane_data(track, iplane)
                        zi  = zi - z_ref    
                        # -------------------------------------------------------------
                        
                        mk += fmkx(nvar, npar, vs, vsig, ss, zi)
                        va += fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                    istp = istp + 1
                    vs0 = vs
                    vs = np.linalg.solve(mk, va)  # Solve mk @ vs = va
                    merr = np.linalg.inv(mk)      # Only compute if needed for fmahd()
                    dist = fmahd(npar, vs, vs0, merr)
                    
                v_res = fres(vs, vdat_ref, lenx, ss, 0)
                
                if iplane_ref <= 4:
                    ext_res_ystr_arr[pos, iplane_ref - 1] = v_res[0]
                    ext_res_tsum_arr[pos, iplane_ref - 1] = v_res[1]
                    ext_res_tdif_arr[pos, iplane_ref - 1] = v_res[2]
    
    # Push the accumulated results back to the DataFrame in a single shot ------
    for plane_idx in range(4):
        col_suffix = plane_idx + 1
        working_df[f'charge_{col_suffix}'] = charge_arr[:, plane_idx]
        working_df[f'res_ystr_{col_suffix}'] = res_ystr_arr[:, plane_idx]
        working_df[f'res_tsum_{col_suffix}'] = res_tsum_arr[:, plane_idx]
        working_df[f'res_tdif_{col_suffix}'] = res_tdif_arr[:, plane_idx]
        working_df[f'ext_res_ystr_{col_suffix}'] = ext_res_ystr_arr[:, plane_idx]
        working_df[f'ext_res_tsum_{col_suffix}'] = ext_res_tsum_arr[:, plane_idx]
        working_df[f'ext_res_tdif_{col_suffix}'] = ext_res_tdif_arr[:, plane_idx]

    working_df['charge_event'] = charge_event_arr
    working_df['iterations'] = iterations_arr
    working_df['conv_distance'] = conv_distance_arr
    working_df['converged'] = converged_arr
    working_df['processed_tt'] = processed_tt_arr

    working_df['th_chi'] = th_chi_arr
    working_df['x'] = x_arr
    working_df['xp'] = xp_arr
    working_df['y'] = y_arr
    working_df['yp'] = yp_arr
    working_df['t0'] = t0_arr
    working_df['s'] = s_arr
    working_df[['res_y', 'res_ts', 'res_td']] = 0.0

    possible_ndf = {nvar * planes - npar for planes in range(2, nplan + 1)}
    possible_ndf = {ndf for ndf in possible_ndf if ndf >= 0}
    for ndf in possible_ndf:
        working_df[f'th_chi_{ndf}'] = th_chi_ndf_arrays.get(ndf, np.zeros(n_rows, dtype=float))
    
    # Filter according to residual ------------------------------------------------
    plane_cols = range(1, 5)
    res_tsum_abs = np.abs(working_df[[f'res_tsum_{i}' for i in plane_cols]].to_numpy())
    res_tdif_abs = np.abs(working_df[[f'res_tdif_{i}' for i in plane_cols]].to_numpy())
    res_ystr_abs = np.abs(working_df[[f'res_ystr_{i}' for i in plane_cols]].to_numpy())
    ext_res_tsum_abs = np.abs(working_df[[f'ext_res_tsum_{i}' for i in plane_cols]].to_numpy())
    ext_res_tdif_abs = np.abs(working_df[[f'ext_res_tdif_{i}' for i in plane_cols]].to_numpy())
    ext_res_ystr_abs = np.abs(working_df[[f'ext_res_ystr_{i}' for i in plane_cols]].to_numpy())

    plane_rejected = (
        (res_tsum_abs > res_tsum_filter) |
        (res_tdif_abs > res_tdif_filter) |
        (res_ystr_abs > res_ystr_filter) |
        (ext_res_tsum_abs > ext_res_tsum_filter) |
        (ext_res_tdif_abs > ext_res_tdif_filter) |
        (ext_res_ystr_abs > ext_res_ystr_filter)
    )
    plane_rejected_df = pd.DataFrame(plane_rejected, index=working_df.index, columns=list(plane_cols))

    changed_event_mask = plane_rejected_df.any(axis=1)
    changed_event_count = int(changed_event_mask.sum())

    for plane_idx in plane_cols:
        mask = plane_rejected_df[plane_idx]
        if mask.any():
            working_df.loc[mask, [f'P{plane_idx}_Y_final',
                                  f'P{plane_idx}_T_sum_final',
                                  f'P{plane_idx}_T_diff_final',
                                  f'P{plane_idx}_Q_sum_final',
                                  f'P{plane_idx}_Q_diff_final']] = 0

    print(f"--> {changed_event_count} events were residual filtered.")
    
    print(f"{len(working_df[working_df.iterations == iter_max])} reached the maximum number of iterations ({iter_max}).")
    print(f"Percentage of events that did not converge: {len(working_df[working_df.iterations == iter_max]) / len(working_df) * 100:.2f}%")
    
    # --------------------------------------------------------------------------------
    iteration += 1


# ------------------------------------------------------------------------------------
# End of TimTrack loop ---------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Set the label to integer -----------------------------------------------------------
working_df['processed_tt'] = working_df['processed_tt'].apply(builtins.int)

# Calculate angles -------------------------------------------------------------------
def calculate_angles(xproj, yproj):
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

theta, phi = calculate_angles(working_df['xp'], working_df['yp'])
new_columns_df = pd.DataFrame({'theta': theta, 'phi': phi}, index=working_df.index)
working_df = pd.concat([working_df, new_columns_df], axis=1)


print("----------------------------------------------------------------------")
print("----------------------- Timtrack results filter ----------------------")
print("----------------------------------------------------------------------")

for col in working_df.columns:
    # TimTrack results
    if 't0' == col:
        working_df.loc[:, col] = np.where((working_df[col] > t0_right_filter) | (working_df[col] < t0_left_filter), 0, working_df[col])
    if 'x' == col or 'y' == col:
        cond_bound = (working_df[col] > pos_filter) | (working_df[col] < -1*pos_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'xp' == col or 'yp' == col:
        cond_bound = (working_df[col] > proj_filter) | (working_df[col] < -1*proj_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 's' == col:
        cond_bound = (working_df[col] > slowness_filter_right) | (working_df[col] < slowness_filter_left)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'theta' == col:
        cond_bound = (working_df[col] > theta_right_filter) | (working_df[col] < theta_left_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])
    if 'phi' == col:
        cond_bound = (working_df[col] > phi_right_filter) | (working_df[col] < phi_left_filter)
        cond_zero = (working_df[col] == 0)
        working_df.loc[:, col] = np.where((cond_bound | cond_zero), 0, working_df[col])


print("----------------------------------------------------------------------")
print("------------------ TimTrack convergence comprobation -----------------")
print("----------------------------------------------------------------------")



if create_plots:

    df_filtered = working_df.copy()
    colors = plt.cm.tab10.colors
    tt_values = [12, 23, 34, 13, 124, 134, 123, 234, 1234]
    n_plots = len(tt_values)
    ncols = 3
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easier indexing
    
    for i, tt_val in enumerate(tt_values):
        ax = axes[i]
        
        df_tt = df_filtered[df_filtered['processed_tt'] == tt_val]
        x = df_tt['iterations']
        y = df_tt['conv_distance']
        # ax.scatter(df_tt['s'], residuals, s=1, color='C0', alpha=0.5)
        ax.scatter(x, y, s=2, color='C0', alpha=0.5)
        ax.axvline(x=iter_max, color='r', linestyle='--', linewidth=1.5, label = "Iteration limit set")
        ax.axhline(y=cocut, color='g', linestyle='--', linewidth=1.5, label = "Convergence cut set")
        ax.set_title(f'TT {tt_val}', fontsize=10)
        # ax.set_xlim(slowness_filter_left, slowness_filter_right)
        ax.set_ylim(0, cocut * 1.05)
        # ax.set_xlim(-1, 5)
        # ax.set_ylim(-0.15, 0.15)
        # ax.set_ylim(slowness_filter_left / 10, slowness_filter_right / 20)
        ax.grid(True)
        ax.legend()

        if i % ncols == 0:
            ax.set_ylabel(r'Iterations vs cocut')
        if i // ncols == nrows - 1:
            ax.set_xlabel(r'$Iterations$')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(r'Iteration vs distance cut in convergence per processed_tt case', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.tight_layout()
    if save_plots:
        filename = f'{fig_idx}_iterations_vs_cocut.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("------------------ Slowness residual comprobation ---------------------")
print("----------------------------------------------------------------------")

working_df['delta_s'] = working_df['alt_s'] - working_df['s']  # Calculate the difference from the speed of light



if create_plots:
    print("Plotting residuals of alt_s - s for each original_tt to processed_tt case...")
    
    df_filtered = working_df.copy()
    bins = np.linspace(delta_s_left, delta_s_right, 100)  # Adjust range and bin size as needed
    colors = plt.cm.tab10.colors

    tt_values = [12, 23, 34, 13, 124, 134, 123, 234, 1234]
    
    # Layout configuration
    n_plots = len(tt_values)
    ncols = 3
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easier indexing
    
    for i, tt_val in enumerate(tt_values):
        ax = axes[i]

        df_tt = df_filtered[df_filtered['processed_tt'] == tt_val]
        residuals = df_tt['delta_s']  # Calculate the residuals
        # residuals = 2 * ( df_tt['alt_s'] - df_tt['s'] ) / ( df_tt['alt_s'] + df_tt['s'] )  # Calculate the residuals
        # rel_sum = ( df_tt['alt_s'] + df_tt['s'] ) / 2
        rel_sum = df_tt['s']
        
        if len(residuals) < 10:
            ax.set_visible(False)
            continue

        # ax.scatter(df_tt['s'], residuals, s=1, color='C0', alpha=0.5)
        ax.scatter(rel_sum, residuals, s=0.8, color='C0', alpha=0.1)
        ax.axvline(x=sc, color='r', linestyle='--', linewidth=1.5, label = "$\\beta = 1$")  # Vertical line at x=0
        ax.axvline(x=0, color='g', linestyle='--', linewidth=1.5, label = "Zero")  # Vertical line at x=0
        ax.set_title(f'TT {tt_val}', fontsize=10)
        ax.set_xlim(slowness_filter_left, slowness_filter_right)
        # ax.set_ylim(-0.001, 0.001)
        # ax.set_xlim(-1, 5)
        # ax.set_ylim(-0.15, 0.15)
        ax.set_ylim(delta_s_left, delta_s_right)
        ax.grid(True)
        ax.legend()

        if i % ncols == 0:
            ax.set_ylabel(r'$alt_s - s$')
        if i // ncols == nrows - 1:
            ax.set_xlabel(r'$s$')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(r'Residuals: $alt_s - s$ per processed_tt case', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save or show the plot
    plt.tight_layout()
    if save_plots:
        filename = f'{fig_idx}_residuals_alt_s_minus_s_processed_tt.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("--------------------- Comparison results filter ----------------------")
print("----------------------------------------------------------------------")

for col in working_df.columns:
    # TimTrack results
    if 'delta_s' == col:
        working_df.loc[:, col] = np.where((working_df[col] > delta_s_right) | (working_df[col] < delta_s_left), 0, working_df[col])

print("----------------------------------------------------------------------")
print("-------------------------- New definitions ---------------------------")
print("----------------------------------------------------------------------")

working_df['x'] = ( working_df['x'] + working_df['alt_x'] ) / 2
working_df['y'] = ( working_df['y'] + working_df['alt_y'] ) / 2
working_df['theta'] = ( working_df['theta'] + working_df['alt_theta'] ) / 2
working_df['phi'] = ( working_df['phi'] + working_df['alt_phi'] ) / 2
working_df['s'] = ( working_df['s'] + working_df['alt_s'] ) / 2

working_df['x_err'] = ( working_df['x'] - working_df['alt_x'] ) / 2
working_df['y_err'] = ( working_df['y'] - working_df['alt_y'] ) / 2
working_df['theta_err'] = ( working_df['theta'] - working_df['alt_theta'] ) / 2
working_df['phi_err'] = ( working_df['phi'] - working_df['alt_phi'] ) / 2
working_df['s_err'] = ( working_df['s'] - working_df['alt_s'] ) / 2

working_df['chi_timtrack'] = working_df['th_chi']
working_df['chi_alternative'] = working_df['alt_th_chi']


print("----------------------------------------------------------------------")
print("-------------------- Real tracking trigger type ----------------------")
print("----------------------------------------------------------------------")

# Required constants supplied by the DAQ geometry
strip_half  = strip_length / 2.0         # x acceptance  : [-strip_half , +strip_half ]
width_half  = total_width / 2.0          # y acceptance  : [-width_half , +width_half ]
z_planes    = np.asarray(z_positions)    # shape (nplan,)

# Precompute averages of the two independent fits --------------------------
# New fitting track columns combining timtrack and the alternative method
x0_avg   = working_df['x']
y0_avg   = working_df['y']
theta_av = working_df['theta']
phi_av   = working_df['phi']

vx = np.sin(theta_av) * np.cos(phi_av)                   # direction cosines
vy = np.sin(theta_av) * np.sin(phi_av)
vz = np.cos(theta_av)

tracking_vals = np.zeros(len(working_df), dtype=int)

for idx in range(len(working_df)):
    if vz.iat[idx] <= 0.0:                                # upward track ⇒ no planes
        continue
    
    if (x0_avg.iat[idx] == 0 or y0_avg.iat[idx] == 0 or
        theta_av.iat[idx] == 0 or phi_av.iat[idx] == 0):
        continue

    planes_hit = []
    for p, z_p in enumerate(z_planes, start=1):
        t   = z_p / vz.iat[idx]                           # parameter at plane p
        x_i = x0_avg.iat[idx] + vx.iat[idx] * t
        y_i = y0_avg.iat[idx] + vy.iat[idx] * t

        if (-strip_half <= x_i <= strip_half and
            -width_half <= y_i <= width_half):
            planes_hit.append(str(p))

    if planes_hit:                                        # concatenate plane numbers
        tracking_vals[idx] = int(''.join(planes_hit))

tracking_df = pd.DataFrame({'tracking_tt': tracking_vals}, index=working_df.index)
working_df = working_df.drop(columns=tracking_df.columns.intersection(working_df.columns), errors='ignore')
working_df = working_df.join(tracking_df)
working_df = working_df.copy()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# The noise determination, if everything goes well ----------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_definitive_tt(row):
    name = ''
    for plane in range(1, 5):
        this_plane = False
        q_sum_col  = f'P{plane}_Q_sum_final'
        q_diff_col = f'P{plane}_Q_diff_final'
        t_sum_col  = f'P{plane}_T_sum_final'
        t_diff_col = f'P{plane}_T_diff_final'
        
        if (row[q_sum_col] != 0 and row[q_diff_col] != 0 and
            row[t_sum_col] != 0 and row[t_diff_col] != 0):
            this_plane = True
        
        if this_plane:
            name += str(plane)
            
    return int(name) if name else 0  # Return 0 if no plane is valid

# Apply to all rows
working_df["definitive_tt"] = working_df.apply(compute_definitive_tt, axis=1)

if time_window_fitting:
    
    print("---------------------------- Fitting loop ----------------------------")
    
    for definitive_tt in [ 234, 123, 34, 1234, 23, 12, 124, 134, 24, 13, 14 ]:
        # Create a mask for the current definitive_tt
        mask = working_df['definitive_tt'] == definitive_tt

        # Filter the DataFrame based on the mask
        filtered_df = working_df[mask]

        # Check if there are any rows in the filtered DataFrame
        if len(filtered_df) > 0:
            print(f"\nProcessing definitive_tt: {definitive_tt} with {len(filtered_df)} events.")
        T_sum_columns = filtered_df.filter(regex='_T_sum_')

        t_sum_data = T_sum_columns.values  # shape: (n_events, n_detectors)
        
        nonzero_rows = [np.any(row != 0) for row in t_sum_data]
        if not any(nonzero_rows):
            print(f"\n[Warning] Skipping definitive_tt {definitive_tt}: no non-zero T_sum data.")
            continue
        
        widths = np.linspace(0, 2 * coincidence_window_cal_ns, coincidence_window_cal_number_of_points)  # Scan range of window widths in ns
        
        counts_per_width = []
        counts_per_width_dev = []

        for w in widths:
            count_in_window = []
            for row in t_sum_data:
                row_no_zeros = row[row != 0]
                if len(row_no_zeros) == 0:
                    count_in_window.append(0)
                    continue

                stat = np.mean(row_no_zeros)  # or np.median(row_no_zeros)
                lower = stat - w / 2
                upper = stat + w / 2
                n_in_window = np.sum((row_no_zeros >= lower) & (row_no_zeros <= upper))
                count_in_window.append(n_in_window)

            counts_per_width.append(np.mean(count_in_window))
            counts_per_width_dev.append(np.std(count_in_window))

        counts_per_width = np.array(counts_per_width)
        counts_per_width_dev = np.array(counts_per_width_dev)
        # Safely compute a scalar denominator: use the maximum of counts_per_width if positive,
        # otherwise fall back to 1. This avoids passing a list mixing arrays and scalars to np.max,
        # which raises an error due to inhomogeneous shapes.
        if counts_per_width.size == 0:
            denom = 1.0
        else:
            denom = float(np.max(counts_per_width))
            if not np.isfinite(denom) or denom <= 0:
                denom = 1.0
        counts_per_width_norm = counts_per_width / denom

        # # Define model function: signal (logistic) + linear background
        # def signal_plus_background(w, S, w0, tau, B):
        #     return S / (1 + np.exp(-(w - w0) / tau)) + B * w
        
        def signal_plus_background(w, S, w0, sigma, B):
            return 0.5 * S * (1 + erf((w - w0) / (np.sqrt(2) * sigma))) + B * w

        p0 = [1.0, 1.0, 0.5, 0.02]
        
        # Convert to NumPy arrays (if not already)
        widths = np.asarray(widths)
        counts_per_width_norm = np.asarray(counts_per_width_norm)

        # Create a mask for valid (finite) values
        valid_mask = np.isfinite(widths) & np.isfinite(counts_per_width_norm)

        # Apply mask to both x and y data
        widths_clean = widths[valid_mask]
        counts_clean = counts_per_width_norm[valid_mask]
        
        if len(counts_clean) == 0 or len(widths_clean) == 0:
            print(f"[Warning] Skipping definitive_tt {definitive_tt}: no valid data.")
            continue
        
        # Then fit
        popt, pcov = curve_fit(signal_plus_background, widths_clean, counts_clean, p0=p0)
                
        S_fit, w0_fit, tau_fit, B_fit = popt
        print(f"definitive_tt {definitive_tt} - Fit parameters:\n  Signal amplitude S = {S_fit:.4f}\n  Transition center w0 = {w0_fit:.4f} ns\n  Transition width τ = {tau_fit:.4f} ns\n  Background slope B = {B_fit:.6f} per ns")

        global_variables[f'sigmoid_width_{definitive_tt}'] = tau_fit
        global_variables[f'background_slope_{definitive_tt}'] = B_fit

       
        if create_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
            ax.axvline(x=coincidence_window_cal_ns, color='red', linestyle='--', label='Time coincidence window')
            ax.set_xlabel("Window width (ns)")
            ax.set_ylabel("Normalized average # of T_sum values in window")
            ax.set_title("Fraction of hits within stat-centered window vs width")
            ax.grid(True)
            w_fit = np.linspace(min(widths), max(widths), 300)
            f_fit = signal_plus_background(w_fit, *popt)
            ax.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
            ax.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
            s_vals = S_fit / (1 + np.exp(-(w_fit - w0_fit) / tau_fit))
            b_vals = B_fit * w_fit
            f_vals = s_vals + b_vals
            P_signal = s_vals / f_vals
            P_background = b_vals / f_vals
            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
            ax_fill = fig.add_subplot(gs[0])  # Top: signal vs. background fill
            ax_main = fig.add_subplot(gs[1], sharex=ax_fill)  # Bottom: your original plot
            ax_fill.fill_between(w_fit, 0, P_signal, color='green', alpha=0.4, label='Signal')
            ax_fill.fill_between(w_fit, P_signal, 1, color='red', alpha=0.4, label='Background')
            ax_fill.set_ylabel("Fraction")
            ax_fill.set_ylim(np.min(P_signal), 1)
            # ax_fill.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_fill.legend(loc="upper right")
            ax_fill.set_title(f"Estimated Signal and Background Fractions per Window Width, definitive_tt = {definitive_tt}")
            plt.setp(ax_fill.get_xticklabels(), visible=False)
            ax_main.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
            ax_main.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
            ax_main.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
            ax_main.set_xlabel("Window width (ns)")
            ax_main.set_ylabel("Normalized average # of T_sum values in window")
            ax_main.grid(True)
            fit_summary = (f"Fit: S = {S_fit:.3f}, w₀ = {w0_fit:.3f} ns, " f"τ = {tau_fit:.3f} ns, B = {B_fit:.4f}/ns")
            ax_main.plot([], [], ' ', label=fit_summary)  # invisible handle to add text
            ax_main.legend()
            
            if save_plots:
                name_of_file = f'stat_window_accumulation_{definitive_tt}'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


# -----------------------------------------------------------------------------
# Last filterings -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Put to zero the rows with traking in only one plane, that is, put 0 if tracking_tt < 10
for index, row in working_df.iterrows():
    if row['tracking_tt'] < 10 or row['processed_tt'] < 10 or row['original_tt'] < 10 or row['definitive_tt'] < 10:
        working_df.at[index, 'x'] = 0
        working_df.at[index, 'xp'] = 0
        working_df.at[index, 'y'] = 0
        working_df.at[index, 'yp'] = 0
        working_df.at[index, 't0'] = 0
        working_df.at[index, 's'] = 0


# -----------------------------------------------------------------------------
# -------------- Correlate trigger types in different stages ------------------
# -----------------------------------------------------------------------------

def plot_tt_correlation(df, row_label, col_label, title, filename_suffix, fig_idx, base_dir, show_plots=False, save_plots=False, plot_list=None):

    analysis_data = df[[row_label, col_label]]
    counts = analysis_data.groupby([row_label, col_label]).size().unstack(fill_value=0)

    row_order = sorted(analysis_data[row_label].unique(), reverse=True)
    col_unique = analysis_data[col_label].unique()
    col_order = list(row_order) + [x for x in col_unique if x not in row_order]
    counts = counts.reindex(index=row_order, columns=col_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns)
    ax.set_yticklabels(counts.index)

    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)

    im = ax.imshow(counts, cmap='plasma', origin='lower')
    total = counts.values.sum()

    for i in range(len(counts.index)):
        for j in range(len(counts.columns)):
            value = counts.iloc[i, j]
            if value > 0:
                pct = 100 * value / total
                if pct > 1:
                    ax.text(j, i, f"{pct:.1f}%",
                            ha="center", va="center",
                            color="black" if value > counts.values.max() * 0.5 else "white")

    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_{filename_suffix}.png'
        save_fig_path = os.path.join(base_dir, final_filename)
        if plot_list is not None:
            plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

    return fig_idx + 1


if create_plots:
    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='original_tt',
        col_label='processed_tt',
        title='Event counts per (original_tt, processed_tt) combination',
        filename_suffix='trigger_types_og_and_processed',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='tracking_tt',
        col_label='processed_tt',
        title='Event counts per (tracking_tt, processed_tt) combination',
        filename_suffix='trigger_types_tracking_and_processed',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='tracking_tt',
        col_label='original_tt',
        title='Event counts per (tracking_tt, original_tt) combination',
        filename_suffix='trigger_types_tracking_and_original',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )

if create_plots or create_essential_plots:
    fig_idx = plot_tt_correlation(
        df=working_df,
        row_label='original_tt',
        col_label='definitive_tt',
        title='Event counts per (original_tt, definitive_tt) combination',
        filename_suffix='trigger_types_definitive_tt_and_original',
        fig_idx=fig_idx,
        base_dir=base_directories["figure_directory"],
        show_plots=show_plots,
        save_plots=save_plots,
        plot_list=plot_list
    )


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Define the last dataframe, the definitive one -------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

definitive_df = working_df.copy()

# Remove small, non-zero values -----------------------------------------------
mask = definitive_df.map(is_small_nonzero)
nonzero_numeric_mask = definitive_df.map(lambda x: isinstance(x, (int, float)) and x != 0)
n_total = nonzero_numeric_mask.sum().sum()
n_small = mask.sum().sum()
definitive_df = definitive_df.mask(mask, 0)
pct = 100 * n_small / n_total if n_total > 0 else 0
print(f"\nIn definitive_df {n_small} out of {n_total} non-zero numeric values are below {eps} ({pct:.4f}%)")

# Remove rows with zeros in key places ----------------------------------------
cols_to_check = ['x', 'xp', 'y', 'yp', 's', 't0', 'alt_x', 'alt_y', 'alt_theta', 'alt_phi', 'alt_s']

cond = (working_df[cols_to_check[0]] != 0)
for col in cols_to_check[1:]:
    cond &= (working_df[col] != 0)

n_before = len(definitive_df)
definitive_df = definitive_df[cond]
n_after = len(definitive_df)

# Calculate and print percentage ----------------------------------------------
percentage_retained = 100 * n_after / n_before if n_before > 0 else 0
print(f"Rows before: {n_before}")
print(f"Rows after: {n_after}")
print(f"Retained: {percentage_retained:.2f}%")

print("----------------------------------------------------------------------")
print("Unique original_tt values:", sorted(definitive_df['original_tt'].unique()))
print("Unique preprocessed_tt values:", sorted(definitive_df['preprocessed_tt'].unique()))
print("Unique processed_tt values:", sorted(definitive_df['processed_tt'].unique()))
print("Unique tracking_tt values:", sorted(definitive_df['tracking_tt'].unique()))
print("Unique definitive_tt values:", sorted(definitive_df['definitive_tt'].unique()))


print("----------------------------------------------------------------------")
print("----------------------- Calculating some stuff -----------------------")
print("----------------------------------------------------------------------")

df_plot_ancillary = definitive_df.copy()



cond = ( df_plot_ancillary['charge_1'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_2'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_3'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_4'] < charge_plot_limit_right ) &\
    ( df_plot_ancillary['charge_event'] > charge_plot_limit_left )

df_plot_ancillary = df_plot_ancillary.loc[cond].copy()


# -----------------------------------------------------------------------------------------------------------------------------

if create_plots:
    
    # Alternative method --------------------------------------------------------------------------------------------
    residual_columns = [
        'alt_res_ystr_1', 'alt_res_ystr_2', 'alt_res_ystr_3', 'alt_res_ystr_4',
        'alt_res_tsum_1', 'alt_res_tsum_2', 'alt_res_tsum_3', 'alt_res_tsum_4',
        'alt_res_tdif_1', 'alt_res_tdif_2', 'alt_res_tdif_3', 'alt_res_tdif_4'
    ]
    
    unique_types = df_plot_ancillary['definitive_tt'].unique()
    for t in unique_types:
        if t < 100:
            continue
        subset_data = df_plot_ancillary[df_plot_ancillary['definitive_tt'] == t]
        plot_histograms_and_gaussian(subset_data, residual_columns, f"Alternative fitting Residuals with Gaussian for Original Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)
        
    
    # TimTrack method --------------------------------------------------------------------------------------------
    residual_columns = [
        'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
        'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
        'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4'
    ]
    
    unique_types = df_plot_ancillary['definitive_tt'].unique()
    for t in unique_types:
        if t < 100:
            continue
        subset_data = df_plot_ancillary[df_plot_ancillary['definitive_tt'] == t]
        plot_histograms_and_gaussian(subset_data, residual_columns, f"TimTrack Residuals with Gaussian for Processed Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)
    
    
    # TimTrack method - External residues -------------------------------------------------------------------------
    residual_columns = [
        'ext_res_ystr_1', 'ext_res_ystr_2', 'ext_res_ystr_3', 'ext_res_ystr_4',
        'ext_res_tsum_1', 'ext_res_tsum_2', 'ext_res_tsum_3', 'ext_res_tsum_4',
        'ext_res_tdif_1', 'ext_res_tdif_2', 'ext_res_tdif_3', 'ext_res_tdif_4'
    ]

    unique_types = df_plot_ancillary['definitive_tt'].unique()
    for t in unique_types:
        if t < 100:
            continue
        subset_data = df_plot_ancillary[df_plot_ancillary['definitive_tt'] == t]
        plot_histograms_and_gaussian(subset_data, residual_columns, f"External Residuals with Gaussian for Processed Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)

# -----------------------------------------------------------------------------------------------------------------------------


if create_plots or create_essential_plots:
    
    df_filtered = df_plot_ancillary.copy()
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))
    
    tt_values = [13, 12, 23, 34, 123, 124, 134, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 3
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 28
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
        mask = (theta_vals >= theta_min) & (theta_vals <= theta_max) & \
               (phi_vals >= phi_min) & (phi_vals <= phi_max)
        theta_vals = theta_vals[mask]
        phi_vals   = phi_vals[mask]

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap

        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')
        # Put as title of the subplot the definitive_tt value
        ax.set_title(f'Plane combination (definitive) {tt_val}', fontsize=10)

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each definitive_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_definitive_tt_2D.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()




if create_plots:
    
    df_filtered = df_plot_ancillary.copy()
    # tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))
    
    tt_values = [12, 23, 34, 123, 234, 1234]
    
    n_tt = len(tt_values)
    ncols = 3
    nrows = (n_tt + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
    phi_nbins = 40
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

        df_tt = df_filtered[df_filtered['tracking_tt'] == tt_val]
        theta_vals = df_tt['theta'].dropna()
        phi_vals = df_tt['phi'].dropna()

        # Apply range filtering
        mask = (theta_vals >= theta_min) & (theta_vals <= theta_max) & \
               (phi_vals >= phi_min) & (phi_vals <= phi_max)
        theta_vals = theta_vals[mask]
        phi_vals   = phi_vals[mask]

        if len(theta_vals) < 10 or len(phi_vals) < 10:
            ax.set_visible(False)
            continue

        # Polar plot settings
        fig.delaxes(axes[row_idx][col_idx])  # remove the original non-polar Axes
        ax = fig.add_subplot(nrows, ncols, idx + 1, polar=True)  # add a polar Axes
        axes[row_idx][col_idx] = ax  # update reference for consistency

        ax.set_facecolor(colors(0.0))  # darkest background in colormap

        # 2D histogram: use phi as angle, theta as radius
        h, r_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        R, PHI = np.meshgrid(r_centers, phi_centers, indexing='ij')
        c = ax.pcolormesh(PHI, R, h, cmap='viridis', vmin=0, vmax=vmax_global)
        local_max = h.max()
        cb = fig.colorbar(c, ax=ax, pad=0.1)
        cb.ax.hlines(local_max, *cb.ax.get_xlim(), colors='white', linewidth=2, linestyles='dashed')

    plt.suptitle(r'2D Histogram of $\theta$ vs. $\phi$ for each tracking_tt Type', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_polar_theta_phi_tracking_tt_2D.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()

# -----------------------------------------------------------------------------------------------------------------------------



if create_plots:

    def plot_hexbin_matrix(df, columns_of_interest, filter_conditions, title, save_plots, show_plots, base_directories, fig_idx, plot_list, num_bins=40):
        
        axis_limits = {
            # Static
            'x': [-pos_filter, pos_filter],
            'y': [-pos_filter, pos_filter],
            'alt_x': [-pos_filter, pos_filter],
            'alt_y': [-pos_filter, pos_filter],
            'theta': [theta_left_filter, theta_right_filter],
            'phi': [phi_left_filter, phi_right_filter],
            'alt_theta': [alt_theta_left_filter, alt_theta_right_filter],
            'alt_phi': [alt_phi_left_filter, alt_phi_right_filter],
            'xp': [-1 * proj_filter, proj_filter],
            'yp': [-1 * proj_filter, proj_filter],
            's': [slowness_filter_left, slowness_filter_right],
            'alt_s': [alt_slowness_filter_left, alt_slowness_filter_right],
            'delta_s': [delta_s_left, delta_s_right],
            # 'th_chi': [0, 0.03],
            # 'alt_th_chi': [0, 12],
            
            # Dinamic
            'charge_event': [charge_plot_limit_left, charge_plot_event_limit_right],
            'charge_1': [charge_plot_limit_left, charge_plot_limit_right],
            'charge_2': [charge_plot_limit_left, charge_plot_limit_right],
            'charge_3': [charge_plot_limit_left, charge_plot_limit_right],
            'charge_4': [charge_plot_limit_left, charge_plot_limit_right],
            'res_ystr_1': [-res_ystr_filter, res_ystr_filter], 'res_ystr_2': [-res_ystr_filter, res_ystr_filter], 'res_ystr_3': [-res_ystr_filter, res_ystr_filter], 'res_ystr_4': [-res_ystr_filter, res_ystr_filter],
            'res_tsum_1': [-res_tsum_filter, res_tsum_filter], 'res_tsum_2': [-res_tsum_filter, res_tsum_filter], 'res_tsum_3': [-res_tsum_filter, res_tsum_filter], 'res_tsum_4': [-res_tsum_filter, res_tsum_filter],
            'res_tdif_1': [-res_tdif_filter, res_tdif_filter], 'res_tdif_2': [-res_tdif_filter, res_tdif_filter], 'res_tdif_3': [-res_tdif_filter, res_tdif_filter], 'res_tdif_4': [-res_tdif_filter, res_tdif_filter],
            'alt_res_ystr_1': [-alt_res_ystr_filter, alt_res_ystr_filter], 'alt_res_ystr_2': [-alt_res_ystr_filter, alt_res_ystr_filter], 'alt_res_ystr_3': [-alt_res_ystr_filter, alt_res_ystr_filter], 'alt_res_ystr_4': [-alt_res_ystr_filter, alt_res_ystr_filter],
            'alt_res_tsum_1': [-alt_res_tsum_filter, alt_res_tsum_filter], 'alt_res_tsum_2': [-alt_res_tsum_filter, alt_res_tsum_filter], 'alt_res_tsum_3': [-alt_res_tsum_filter, alt_res_tsum_filter], 'alt_res_tsum_4': [-alt_res_tsum_filter, alt_res_tsum_filter],
            'alt_res_tdif_1': [-alt_res_tdif_filter, alt_res_tdif_filter], 'alt_res_tdif_2': [-alt_res_tdif_filter, alt_res_tdif_filter], 'alt_res_tdif_3': [-alt_res_tdif_filter, alt_res_tdif_filter], 'alt_res_tdif_4': [-alt_res_tdif_filter, alt_res_tdif_filter],
            'ext_res_ystr_1': [-ext_res_ystr_filter, ext_res_ystr_filter], 'ext_res_ystr_2': [-ext_res_ystr_filter, ext_res_ystr_filter], 'ext_res_ystr_3': [-ext_res_ystr_filter, ext_res_ystr_filter], 'ext_res_ystr_4': [-ext_res_ystr_filter, ext_res_ystr_filter],
            'ext_res_tsum_1': [-ext_res_tsum_filter, ext_res_tsum_filter], 'ext_res_tsum_2': [-ext_res_tsum_filter, ext_res_tsum_filter], 'ext_res_tsum_3': [-ext_res_tsum_filter, ext_res_tsum_filter], 'ext_res_tsum_4': [-ext_res_tsum_filter, ext_res_tsum_filter],
            'ext_res_tdif_1': [-ext_res_tdif_filter, ext_res_tdif_filter], 'ext_res_tdif_2': [-ext_res_tdif_filter, ext_res_tdif_filter], 'ext_res_tdif_3': [-ext_res_tdif_filter, ext_res_tdif_filter], 'ext_res_tdif_4': [-ext_res_tdif_filter, ext_res_tdif_filter],
        }
        
        # Apply filters
        for col, min_val, max_val in filter_conditions:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
        
        num_var = len(columns_of_interest)
        fig, axes = plt.subplots(num_var, num_var, figsize=(15, 15))
        
        auto_limits = {}
        for col in columns_of_interest:
            if col in axis_limits:
                auto_limits[col] = axis_limits[col]
            else:
                auto_limits[col] = [df[col].min(), df[col].max()]
        
        for i in range(num_var):
            for j in range(num_var):
                ax = axes[i, j]
                x_col = columns_of_interest[j]
                y_col = columns_of_interest[i]
                
                if i < j:
                    ax.axis('off')  # Leave the lower triangle blank
                elif i == j:
                    # Diagonal: 1D histogram
                    hist_data = df[x_col]
                    # Remove nans
                    hist_data = hist_data[~np.isnan(hist_data)]
                    # Remove zeroes
                    hist_data = hist_data[hist_data != 0]
                    hist, bins = np.histogram(hist_data, bins=num_bins)
                    bin_centers = 0.5 * (bins[1:] + bins[:-1])
                    norm = plt.Normalize(hist.min(), hist.max())
                    cmap = plt.get_cmap('turbo')
                    
                    for k in range(len(hist)):
                        ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlim(auto_limits[x_col])
                    
                    # If the column is 'charge_1, 2, 3 or 4', set logscale in Y
                    if x_col.startswith('charge'):
                        ax.set_yscale('log')
                    
                else:
                    # Upper triangle: hexbin
                    x_data = df[x_col]
                    y_data = df[y_col]
                    # Remove zeroes and nans
                    cond = (x_data != 0) & (y_data != 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
                    x_data = x_data[cond]
                    y_data = y_data[cond]
                    ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
                    ax.set_facecolor(plt.cm.turbo(0))
                    
                    if "alt_s" in x_col and "s" in x_col or "s" in y_col and "alt_s" in y_col:
                        # Draw a line in the diagonal y = x
                        line_x = np.linspace(-0.01, 0.015, 100)
                        line_y = line_x
                        ax.plot(line_x, line_y, color='white', linewidth=1)  # Thin white line
                    
                    square_x = [-150, 150, 150, -150, -150]  # Closing the loop
                    square_y = [-150, -150, 150, 150, -150]
                    ax.plot(square_x, square_y, color='white', linewidth=1)  # Thin white line
                    
                    # Apply determined limits
                    ax.set_xlim(auto_limits[x_col])
                    ax.set_ylim(auto_limits[y_col])
                
                if i != num_var - 1:
                    ax.set_xticklabels([])
                if j != 0:
                    ax.set_yticklabels([])
                if i == num_var - 1:
                    ax.set_xlabel(x_col)
                if j == 0:
                    ax.set_ylabel(y_col)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.suptitle(title)
        if save_plots:
            name_of_file = 'timtrack_results_hexbin_combination_projections'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        # Show plot if enabled
        if show_plots:
            plt.show()
        plt.close()
        return fig_idx


    # df_cases_2 = [
    #     ([("processed_tt", 12, 12)], "1-2 cases"),
    #     ([("processed_tt", 23, 23)], "2-3 cases"),
    #     ([("processed_tt", 34, 34)], "3-4 cases"),
    #     ([("processed_tt", 13, 13)], "1-3 cases"),
    #     ([("processed_tt", 14, 14)], "1-4 cases"),
    #     ([("processed_tt", 123, 123)], "1-2-3 cases"),
    #     ([("processed_tt", 234, 234)], "2-3-4 cases"),
    #     ([("processed_tt", 124, 124)], "1-2-4 cases"),
    #     ([("processed_tt", 134, 134)], "1-3-4 cases"),
    #     ([("processed_tt", 1234, 1234)], "1-2-3-4 cases"),
    # ]
    
    
    # df_cases_2 = [
    #     ([("tracking_tt", 12, 12)], "1-2 cases"),
    #     ([("tracking_tt", 23, 23)], "2-3 cases"),
    #     ([("tracking_tt", 34, 34)], "3-4 cases"),
    #     ([("tracking_tt", 123, 123)], "1-2-3 cases"),
    #     ([("tracking_tt", 234, 234)], "2-3-4 cases"),
    #     ([("tracking_tt", 1234, 1234)], "1-2-3-4 cases"),
    # ]
    
    df_cases_1 = [
        ([("definitive_tt", 12, 12)], "1-2 cases"),
        ([("definitive_tt", 23, 23)], "2-3 cases"),
        ([("definitive_tt", 34, 34)], "3-4 cases"),
        ([("definitive_tt", 123, 123)], "1-2-3 cases"),
        ([("definitive_tt", 234, 234)], "2-3-4 cases"),
        ([("definitive_tt", 1234, 1234)], "1-2-3-4 cases"),
        ([("definitive_tt", 13, 13)], "1-3 cases"),
        ([("definitive_tt", 14, 14)], "1-4 cases"),
        ([("definitive_tt", 124, 124)], "1-2-4 cases"),
        ([("definitive_tt", 134, 134)], "1-3-4 cases"),
    ]
    
    df_cases_2 = [
        ([("definitive_tt", 1234, 1234)], "1-2-3-4 cases"),
        ([("definitive_tt", 123, 123)], "1-2-3 cases"),
        ([("definitive_tt", 234, 234)], "2-3-4 cases"),
        ([("definitive_tt", 124, 124)], "1-2-4 cases"),
        ([("definitive_tt", 134, 134)], "1-3-4 cases"),
    ]
    
    df_cases_3 = [
        ([("definitive_tt", 12, 12), ("iterations", 2, 2)], "1-2 cases, iterations = 2"),
        ([("definitive_tt", 12, 12), ("iterations", 3, 3)], "1-2 cases, iterations = 3"),
        ([("definitive_tt", 12, 12), ("iterations", 4, 4)], "1-2 cases, iterations = 4"),
        ([("definitive_tt", 12, 12), ("iterations", 5, 5)], "1-2 cases, iterations = 5"),
        ([("definitive_tt", 12, 12), ("iterations", 6, iter_max)], f"1-2 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 23, 23), ("iterations", 2, 2)], "2-3 cases, iterations = 2"),
        ([("definitive_tt", 23, 23), ("iterations", 3, 3)], "2-3 cases, iterations = 3"),
        ([("definitive_tt", 23, 23), ("iterations", 4, 4)], "2-3 cases, iterations = 4"),
        ([("definitive_tt", 23, 23), ("iterations", 5, 5)], "2-3 cases, iterations = 5"),
        ([("definitive_tt", 23, 23), ("iterations", 6, iter_max)], f"2-3 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 34, 34), ("iterations", 2, 2)], "3-4 cases, iterations = 2"),
        ([("definitive_tt", 34, 34), ("iterations", 3, 3)], "3-4 cases, iterations = 3"),
        ([("definitive_tt", 34, 34), ("iterations", 4, 4)], "3-4 cases, iterations = 4"),
        ([("definitive_tt", 34, 34), ("iterations", 5, 5)], "3-4 cases, iterations = 5"),
        ([("definitive_tt", 34, 34), ("iterations", 6, iter_max)], f"3-4 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 123, 123), ("iterations", 2, 2)], "1-2-3 cases, iterations = 2"),
        ([("definitive_tt", 123, 123), ("iterations", 3, 3)], "1-2-3 cases, iterations = 3"),
        ([("definitive_tt", 123, 123), ("iterations", 4, 4)], "1-2-3 cases, iterations = 4"),
        ([("definitive_tt", 123, 123), ("iterations", 5, 5)], "1-2-3 cases, iterations = 5"),
        ([("definitive_tt", 123, 123), ("iterations", 6, iter_max)], f"1-2-3 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 234, 234), ("iterations", 2, 2)], "2-3-4 cases, iterations = 2"),
        ([("definitive_tt", 234, 234), ("iterations", 3, 3)], "2-3-4 cases, iterations = 3"),
        ([("definitive_tt", 234, 234), ("iterations", 4, 4)], "2-3-4 cases, iterations = 4"),
        ([("definitive_tt", 234, 234), ("iterations", 5, 5)], "2-3-4 cases, iterations = 5"),
        ([("definitive_tt", 234, 234), ("iterations", 6, iter_max)], f"2-3-4 cases, iterations = 6 to {iter_max}"),
        
        ([("definitive_tt", 1234, 1234), ("iterations", 2, 2)], "1-2-3-4 cases, iterations = 2"),
        ([("definitive_tt", 1234, 1234), ("iterations", 3, 3)], "1-2-3-4 cases, iterations = 3"),
        ([("definitive_tt", 1234, 1234), ("iterations", 4, 4)], "1-2-3-4 cases, iterations = 4"),
        ([("definitive_tt", 1234, 1234), ("iterations", 5, 5)], "1-2-3-4 cases, iterations = 5"),
        ([("definitive_tt", 1234, 1234), ("iterations", 6, iter_max)], f"1-2-3-4 cases, iterations = 6 to {iter_max}"),
    ]
    
    # df_cases_1 = [
    #     # From original_tt = 1234
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 123, 123)], "original=1234, processed=123"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 124, 124)], "original=1234, processed=124"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 134, 134)], "original=1234, processed=134"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 234, 234)], "original=1234, processed=234"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 12, 12)],   "original=1234, processed=12"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 13, 13)],   "original=1234, processed=13"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 14, 14)],   "original=1234, processed=14"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 23, 23)],   "original=1234, processed=23"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 24, 24)],   "original=1234, processed=24"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 34, 34)],   "original=1234, processed=34"),
    #     ([("original_tt", 1234, 1234), ("definitive_tt", 1234, 1234)], "original=1234, processed=1234"),

    #     # From original_tt = 124
    #     ([("original_tt", 124, 124), ("definitive_tt", 12, 12)], "original=124, processed=12"),
    #     ([("original_tt", 124, 124), ("definitive_tt", 14, 14)], "original=124, processed=14"),
    #     ([("original_tt", 124, 124), ("definitive_tt", 24, 24)], "original=124, processed=24"),
    #     ([("original_tt", 124, 124), ("definitive_tt", 124, 124)], "original=124, processed=124"),

    #     # From original_tt = 134
    #     ([("original_tt", 134, 134), ("definitive_tt", 13, 13)], "original=134, processed=13"),
    #     ([("original_tt", 134, 134), ("definitive_tt", 14, 14)], "original=134, processed=14"),
    #     ([("original_tt", 134, 134), ("definitive_tt", 34, 34)], "original=134, processed=34"),
    #     ([("original_tt", 134, 134), ("definitive_tt", 134, 134)], "original=134, processed=134"),

    #     # From original_tt = 123
    #     ([("original_tt", 123, 123), ("definitive_tt", 12, 12)], "original=123, processed=12"),
    #     ([("original_tt", 123, 123), ("definitive_tt", 13, 13)], "original=123, processed=13"),
    #     ([("original_tt", 123, 123), ("definitive_tt", 23, 23)], "original=123, processed=23"),
    #     ([("original_tt", 123, 123), ("definitive_tt", 123, 123)], "original=123, processed=123"),

    #     # From original_tt = 234
    #     ([("original_tt", 234, 234), ("definitive_tt", 23, 23)], "original=234, processed=23"),
    #     ([("original_tt", 234, 234), ("definitive_tt", 24, 24)], "original=234, processed=24"),
    #     ([("original_tt", 234, 234), ("definitive_tt", 34, 34)], "original=234, processed=34"),
    #     ([("original_tt", 234, 234), ("definitive_tt", 234, 234)], "original=234, processed=234"),

    #     # From original_tt = 12
    #     ([("original_tt", 12, 12), ("definitive_tt", 12, 12)], "original=12, processed=12"),

    #     # From original_tt = 23
    #     ([("original_tt", 23, 23), ("definitive_tt", 23, 23)], "original=23, processed=23"),

    #     # From original_tt = 34
    #     ([("original_tt", 34, 34), ("definitive_tt", 34, 34)], "original=34, processed=34"),

    #     # From original_tt = 13
    #     ([("original_tt", 13, 13), ("definitive_tt", 13, 13)], "original=13, processed=13"),
    # ]


    # # Charge of each plane -------------------------------------------------------------------
    # for filters, title in df_cases_2:
    #     # Extract the relevant charge numbers from the title (e.g., "1-2 cases" -> [1, 2])
    #     relevant_charges = [f"charge_{n}" for n in map(int, title.split()[0].split('-'))]

    #     # Define the columns - interest dynamically
    #     # columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp'] + relevant_charges
    #     columns_of_interest = relevant_charges

    #     # Keep the original filters (if needed) and apply them
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,  # Dynamically set the columns to include relevant charges
    #         filters,  # Keep original filters
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )


    # # Residues --------------------------------------------------------------------------------------
    # for filters, title in df_cases_2:
    #     relevant_residues_tsum = [f"res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_tdif = [f"res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
    #     relevant_residues_ystr = [f"res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        
    #     columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp', 's'] + relevant_residues_tsum + relevant_residues_tdif + relevant_residues_ystr
        
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         columns_of_interest,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    for filters, title in df_cases_2:
        relevant_residues_tsum = [f"res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
        relevant_residues_alt_tsum = [f"alt_res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
        relevant_residues_ext_tsum = [f"ext_res_tsum_{n}" for n in map(int, title.split()[0].split('-'))]
        
        columns_of_interest = relevant_residues_tsum + relevant_residues_alt_tsum + relevant_residues_ext_tsum
        
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            columns_of_interest,
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    for filters, title in df_cases_2:
        relevant_residues_tdif = [f"res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
        relevant_residues_alt_tdif = [f"alt_res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
        relevant_residues_ext_tdif = [f"ext_res_tdif_{n}" for n in map(int, title.split()[0].split('-'))]
        
        columns_of_interest = relevant_residues_tdif + relevant_residues_alt_tdif + relevant_residues_ext_tdif
        
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            columns_of_interest,
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    for filters, title in df_cases_2:
        relevant_residues_ystr = [f"res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        relevant_residues_alt_ystr = [f"alt_res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        relevant_residues_ext_ystr = [f"ext_res_ystr_{n}" for n in map(int, title.split()[0].split('-'))]
        
        columns_of_interest = relevant_residues_ystr + relevant_residues_alt_ystr + relevant_residues_ext_ystr
        
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            columns_of_interest,
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    
    # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 's', 'delta_s', 'alt_s', 'alt_phi', 'alt_theta', 'alt_y', 'alt_x']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # plot_col = ['x', 'y', 'theta', 'phi', 's']
    # plot_col = ['x', 'xp', 'delta_s', 'yp', 'y']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 's', 'delta_s', 'alt_s', 'alt_phi', 'alt_theta', 'alt_y', 'alt_x']
    # for filters, title in df_cases_3:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['t0', 's', 'delta_s', 'alt_s', 'alt_s_ordinate']
    # for filters, title in df_cases_3:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    
    # Comparison with alternative fitting -------------------------------------------------------------------
    plot_col = ['theta', 'alt_theta']
    for filters, title in df_cases_2:
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            plot_col,
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    # A pure theta vs phi map
    plot_col = ['theta', 'phi']
    for filters, title in df_cases_2:
        fig_idx = plot_hexbin_matrix(
            df_plot_ancillary,
            plot_col,
            filters,
            title,
            save_plots,
            show_plots,
            base_directories,
            fig_idx,
            plot_list
        )
    
    
    # df_plot_ancillary_conv = df_plot_ancillary[df_plot_ancillary['converged'] == 1].copy()
    # # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 'delta_s']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )
    
    # df_plot_ancillary_conv = df_plot_ancillary[df_plot_ancillary['converged'] == 0].copy()
    # # Comparison with alternative fitting -------------------------------------------------------------------
    # plot_col = ['x', 'y', 'theta', 'phi', 'delta_s']
    # for filters, title in df_cases_1:
    #     fig_idx = plot_hexbin_matrix(
    #         df_plot_ancillary,
    #         plot_col,
    #         filters,
    #         title,
    #         save_plots,
    #         show_plots,
    #         base_directories,
    #         fig_idx,
    #         plot_list
    #     )

# ------------------------------------------------------------------------------------------------------



if create_plots:
    df_filtered = df_plot_ancillary.copy()
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    colors = plt.cm.tab10.colors
    bins = np.linspace(theta_left_filter, theta_right_filter, 150)
    tt_values = sorted(df_filtered['definitive_tt'].dropna().unique(), key=lambda x: int(x))

    for row_idx, (theta_col, row_label) in enumerate([('theta', r'$\theta$'), ('alt_theta', r'$\theta_{\mathrm{alt}}$')]):
        ax = axes[row_idx]
        for i, tt_val in enumerate(tt_values):
            df_tt = df_filtered[df_filtered['definitive_tt'] == tt_val]
            theta_vals = df_tt[theta_col].dropna()
            if len(theta_vals) < 10:
                continue
            label = f'{tt_val}'
            ax.hist(theta_vals, bins=bins, histtype='step', linewidth=1,
                    color=colors[i % len(colors)], label=label)
        ax.set_xlim(theta_left_filter, theta_right_filter)
        ax.set_xlabel(row_label + r' [rad]')
        ax.set_ylabel('Counts')
        ax.set_title(f'{row_label} — Zoom-in')
        ax.grid(True)
        if row_idx == 0:
            ax.legend(title='definitive_tt', fontsize='small')

    plt.suptitle(r'$\theta$ and $\theta_{\mathrm{alt}}$ (Zoom-in) by Definitive TT Type', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_theta_alt_theta_zoom_definitive_tt.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()




if create_plots:
    df_filtered = df_plot_ancillary.copy()
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    colors = plt.cm.tab10.colors
    bins = np.linspace(theta_left_filter, theta_right_filter, 150)
    tt_values = sorted(df_filtered['tracking_tt'].dropna().unique(), key=lambda x: int(x))

    for row_idx, (theta_col, row_label) in enumerate([('theta', r'$\theta$'), ('alt_theta', r'$\theta_{\mathrm{alt}}$')]):
        ax = axes[row_idx]
        for i, tt_val in enumerate(tt_values):
            df_tt = df_filtered[df_filtered['tracking_tt'] == tt_val]
            theta_vals = df_tt[theta_col].dropna()
            if len(theta_vals) < 10:
                continue
            label = f'{tt_val}'
            ax.hist(theta_vals, bins=bins, histtype='step', linewidth=1,
                    color=colors[i % len(colors)], label=label)
        ax.set_xlim(theta_left_filter, theta_right_filter)
        ax.set_xlabel(row_label + r' [rad]')
        ax.set_ylabel('Counts')
        ax.set_title(f'{row_label} — Zoom-in')
        ax.grid(True)
        if row_idx == 0:
            ax.legend(title='tracking_tt', fontsize='small')

    plt.suptitle(r'$\theta$ and $\theta_{\mathrm{alt}}$ (Zoom-in) by Tracking TT Type', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        final_filename = f'{fig_idx}_theta_alt_theta_zoom_tracking_tt.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Final data statistics ------------------------")
print("----------------------------------------------------------------------")

data_purity = len(definitive_df) / raw_data_len*100
print(f"Data purity is {data_purity:.1f}%")

global_variables['purity_of_data_percentage'] = data_purity

if create_plots:

    column_chosen = "definitive_tt"
    plot_ancillary_df = definitive_df.copy()
    
    # Ensure datetime is proper and indexed
    plot_ancillary_df['datetime'] = pd.to_datetime(plot_ancillary_df['datetime'], errors='coerce')
    plot_ancillary_df = plot_ancillary_df.set_index('datetime')

    # Prepare a container for each group: 2-plane, 3-plane, 4-plane cases
    grouped_data = {
        "Two planes": defaultdict(list),
        "Three planes": defaultdict(list),
        "Four planes": defaultdict(list)
    }

    # Classify events by number of planes in original_tt
    for tt_code in plot_ancillary_df[column_chosen].unique():
        planes = str(tt_code)
        count = len(planes)
        label = f'Case {tt_code}'
        if count == 1:
            grouped_data["One plane"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
        if count == 2:
            grouped_data["Two planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
        elif count == 3:
            grouped_data["Three planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
        elif count == 4:
            grouped_data["Four planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = plt.colormaps['tab10']

    for ax, (title, group_dict) in zip(axes, grouped_data.items()):
        for i, (label, df) in enumerate(group_dict.items()):
            df.index = pd.to_datetime(df.index, errors='coerce')
            event_times = df.index.floor('s')
            full_range = pd.date_range(start=event_times.min(), end=event_times.max(), freq='S')
            events_per_second = event_times.value_counts().reindex(full_range, fill_value=0).sort_index()
            
            hist_data = events_per_second.value_counts().sort_index()
            lambda_estimate = events_per_second.mean()
            x_values = np.arange(0, hist_data.index.max() + 1)
            poisson_pmf = poisson.pmf(x_values, lambda_estimate)
            poisson_pmf_scaled = poisson_pmf * len(events_per_second)

            ax.plot(hist_data.index, hist_data.values, label=label, alpha=0.9, color=colors(i % 10), linewidth = 3)
            ax.plot(x_values, poisson_pmf_scaled, '--', lw=1.5, color=colors(i % 10), alpha=0.6)
            ax.set_xlim(0, 8)

        ax.set_title(f'{title}')
        ax.set_xlabel('Number of Events per Second')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize='small', loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.suptitle('Event Rate Histograms by Original_tt Cardinality with Poisson Fits', fontsize=16)

    if save_plots:
        final_filename = f'{fig_idx}_events_per_second_by_plane_cardinality_definitive_tt.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots:
        plt.show()
    plt.close()


if create_plots:


    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    colors = plt.colormaps['tab10']
    tt_types = ['original_tt', 'definitive_tt']
    row_titles = ['Original TT', 'Processed TT']

    for row_idx, column_chosen in enumerate(tt_types):
        plot_ancillary_df = definitive_df.copy()

        # Ensure datetime is proper and indexed
        plot_ancillary_df['datetime'] = pd.to_datetime(plot_ancillary_df['datetime'], errors='coerce')
        plot_ancillary_df = plot_ancillary_df.set_index('datetime')

        grouped_data = {
            "Two planes": defaultdict(list),
            "Three planes": defaultdict(list),
            "Four planes": defaultdict(list)
        }

        for tt_code in plot_ancillary_df[column_chosen].dropna().unique():
            planes = str(tt_code)
            count = len(planes)
            label = f'Case {tt_code}'
            if count == 2:
                grouped_data["Two planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
            elif count == 3:
                grouped_data["Three planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]
            elif count == 4:
                grouped_data["Four planes"][label] = plot_ancillary_df[plot_ancillary_df[column_chosen] == tt_code]

        for col_idx, (title, group_dict) in enumerate(grouped_data.items()):
            ax = axes[row_idx, col_idx]
            for i, (label, df) in enumerate(group_dict.items()):
                df.index = pd.to_datetime(df.index, errors='coerce')
                event_times = df.index.floor('s')
                full_range = pd.date_range(start=event_times.min(), end=event_times.max(), freq='S')
                events_per_second = event_times.value_counts().reindex(full_range, fill_value=0).sort_index()

                hist_data = events_per_second.value_counts().sort_index()
                lambda_estimate = events_per_second.mean()
                x_values = np.arange(0, hist_data.index.max() + 1)
                poisson_pmf = poisson.pmf(x_values, lambda_estimate)
                poisson_pmf_scaled = poisson_pmf * len(events_per_second)

                ax.plot(hist_data.index, hist_data.values, label=label, alpha=0.9, color=colors(i % 10), linewidth=3)
                ax.plot(x_values, poisson_pmf_scaled, '--', lw=1.5, color=colors(i % 10), alpha=0.6)
                ax.set_xlim(0, 8)

            ax.set_title(f'{title} ({row_titles[row_idx]})')
            ax.set_xlabel('Number of Events per Second')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize='small', loc='upper right')
            ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle('Event Rate Histograms by TT Type and Plane Cardinality with Poisson Fits', fontsize=18)

    # Save and show
    if save_plots:
        final_filename = f'{fig_idx}_events_per_second_by_plane_cardinality_double_row.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("-------------------------- Save and finish ---------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

# Round to 4 significant digits -----------------------------------------------
def round_to_4_significant_digits(x):
    try:
        return builtins.float(f"{builtins.float(x):.4g}")  # Use builtins.float to avoid any overridden names
    except (builtins.ValueError, builtins.TypeError):
        return x

print("Rounding the dataframe values.") 
for col in definitive_df.select_dtypes(include=[np.floating]).columns:
    original_dtype = definitive_df[col].dtype
    rounded_series = definitive_df[col].apply(round_to_4_significant_digits)
    definitive_df.loc[:, col] = rounded_series.astype(original_dtype, copy=False)



# Save the data ---------------------------------------------------------------
# if save_full_data: # Save a full version of the data, for different studies and debugging
#     definitive_df.to_csv(save_full_path, index=False, sep=',', float_format='%.5g')
#     print(f"Datafile saved in {save_full_filename}.")

# Save the main columns, relevant for the posterior analysis ------------------
for i, module in enumerate(['1', '2', '3', '4']):
    for j in range(4):
        strip = j + 1
        definitive_df[f'Q_P{module}s{strip}'] = definitive_df[f'Q{module}_Q_sum_{strip}_no_crstlk']
        definitive_df[f'Q_P{module}s{strip}_with_crstlk'] = definitive_df[f'Q{module}_Q_sum_{strip}_with_crstlk']

if self_trigger:
    for i, module in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            strip = j + 1
            working_st_df[f'Q_P{module}s{strip}'] = working_st_df[f'Q{module}_Q_sum_{strip}']


# Charge checking --------------------------------------------------------------------------------------------------------
if self_trigger:
    if create_plots:
   
        fig, axs = plt.subplots(4, 4, figsize=(18, 12))
        for i in range(1, 5):
            for j in range(1, 5):
                # Get the column name
                col_name = f"Q_P{i}s{j}"
                col_name_2 = f"Q_P{i}s{j}_with_crstlk"
                
                # Plot the histogram
                v = definitive_df[col_name]
                v = v[v != 0]
                w = definitive_df[col_name_2]
                w = w[w != 0]
                
                # For 'no crosstalk' histogram
                counts_v, bins_v = np.histogram(v, bins=80, range=(0, 40))
                normalized_v = counts_v / max(counts_v)
                axs[i-1, j-1].stairs(normalized_v, bins_v, alpha=0.5, label='no crosstalk', color='blue', fill=True)

                # For 'with crosstalk' histogram (if uncommented)
                # counts_w, bins_w = np.histogram(w, bins=80, range=(0, 40))
                # normalized_w = counts_w / max(counts_w)
                # axs[i-1, j-1].stairs(normalized_w, bins_w, alpha=0.5, label='with crosstalk', color='orange', fill=True)

                if self_trigger:
                    x = working_st_df[col_name]
                    x = x[x != 0]
                    counts_x, bins_x = np.histogram(x, bins=40, range=(0, 40))
                    normalized_x = counts_x / max(counts_x)
                    axs[i-1, j-1].stairs(normalized_x, bins_x, alpha=0.5, label='self-trigger', color='orange', fill=True)
                
                axs[i-1, j-1].set_title(col_name)
                axs[i-1, j-1].set_xlabel("Charge / ns")
                axs[i-1, j-1].set_ylabel("Frequency")
                axs[i-1, j-1].grid(True)
                    
                if i == j == 4:
                    axs[i-1, j-1].legend(loc='upper right')
        
        plt.suptitle("Event and self trigger charge spectra comparison")
        plt.tight_layout()
        figure_name = f"all_channels_charge_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()


if self_trigger:
    if create_plots:
   
        fig, axs = plt.subplots(4, 4, figsize=(18, 12))
        for i in range(1, 5):
            for j in range(1, 5):
                # Get the column name
                col_name = f"Q_P{i}s{j}"
                col_name_2 = f"Q_P{i}s{j}_with_crstlk"
                
                plot_def_df = definitive_df.copy()
                plot_def_df = plot_def_df [ plot_def_df["definitive_tt"] == "1234" ]
                
                # Plot the histogram
                v = plot_def_df[col_name]
                v = v[v != 0]
                w = plot_def_df[col_name_2]
                w = w[w != 0]
                
                # For 'no crosstalk' histogram
                counts_v, bins_v = np.histogram(v, bins=80, range=(0, 40))
                normalized_v = counts_v / max(counts_v)
                axs[i-1, j-1].stairs(normalized_v, bins_v, alpha=0.5, label='no crosstalk', color='blue', fill=True)

                # For 'with crosstalk' histogram (if uncommented)
                # counts_w, bins_w = np.histogram(w, bins=80, range=(0, 40))
                # normalized_w = counts_w / max(counts_w)
                # axs[i-1, j-1].stairs(normalized_w, bins_w, alpha=0.5, label='with crosstalk', color='orange', fill=True)

                if self_trigger:
                    x = working_st_df[col_name]
                    x = x[x != 0]
                    counts_x, bins_x = np.histogram(x, bins=40, range=(0, 40))
                    normalized_x = counts_x / max(counts_x)
                    axs[i-1, j-1].stairs(normalized_x, bins_x, alpha=0.5, label='self-trigger', color='orange', fill=True)
                
                axs[i-1, j-1].set_title(col_name)
                axs[i-1, j-1].set_xlabel("Charge / ns")
                axs[i-1, j-1].set_ylabel("Frequency")
                axs[i-1, j-1].grid(True)
                    
                if i == j == 4:
                    axs[i-1, j-1].legend(loc='upper right')
        
        plt.suptitle("Event (4-fold) and self trigger charge spectra comparison")
        plt.tight_layout()
        figure_name = f"all_channels_charge_mingo0{station}"
        if save_plots:
            name_of_file = figure_name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
# ------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Update pipeline status CSV with list events metadata
# -----------------------------------------------------------------------------
def _pipeline_strip_suffix(name: str) -> str:
    for suffix in ('.txt', '.csv', '.dat', '.hld.tar.gz', '.hld-tar-gz', '.hld'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _pipeline_compute_start_timestamp(base: str) -> str:
    digits = base[-11:]
    if len(digits) == 11 and digits.isdigit():
        yy = int(digits[:2])
        doy = int(digits[2:5])
        hh = int(digits[5:7])
        mm = int(digits[7:9])
        ss = int(digits[9:11])
        year = 2000 + yy
        try:
            dt = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
            return dt.strftime('%Y-%m-%d_%H.%M.%S')
        except ValueError:
            return ''
    return ''


# def _update_pipeline_csv_for_list_event() -> None:
#     csv_headers = [
#         'basename',
#         'start_date',
#         'hld_remote_add_date',
#         'hld_local_add_date',
#         'dat_add_date',
#         'list_ev_name',
#         'list_ev_add_date',
#         'acc_name',
#         'acc_add_date',
#         'merge_add_date',
#     ]

#     station_dir = Path(home_path) / 'DATAFLOW_v3' / 'STATIONS' / f'MINGO0{station}'
#     csv_path = station_dir / f'database_status_{station}.csv'
#     csv_path.parent.mkdir(parents=True, exist_ok=True)
#     if not csv_path.exists():
#         with csv_path.open('w', newline='') as handle:
#             writer = csv.writer(handle)
#             writer.writerow(csv_headers)

#     base_name = _pipeline_strip_suffix(os.path.basename(the_filename))
#     list_event_name = save_filename
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     start_value = _pipeline_compute_start_timestamp(base_name)

#     rows: List[dict[str, str]] = []
#     with csv_path.open('r', newline='') as handle:
#         reader = csv.DictReader(handle)
#         rows.extend(reader)

#     found = False
#     for row in rows:
#         if row.get('basename', '') == base_name:
#             found = True
#             if not row.get('start_date') and start_value:
#                 row['start_date'] = start_value
#             row['list_ev_name'] = list_event_name
#             row['list_ev_add_date'] = timestamp
#             break

#     if not found:
#         new_row = {header: '' for header in csv_headers}
#         new_row['basename'] = base_name
#         if start_value:
#             new_row['start_date'] = start_value
#         new_row['list_ev_name'] = list_event_name
#         new_row['list_ev_add_date'] = timestamp
#         rows.append(new_row)

#     # Ensure existing list events on disk are reflected in the CSV
#     list_dir = Path(home_path) / 'DATAFLOW_v3' / 'STATIONS' / f'MINGO0{station}' / 'STAGE_1' / 'EVENT_DATA' / 'LIST_EVENTS_DIRECTORY'
#     existing_names = {row.get('list_ev_name', '') for row in rows}

#     if list_dir.exists():
#         for list_path in sorted(list_dir.glob('list_events_*.txt')):
#             list_name = list_path.name
#             if list_name in existing_names:
#                 continue

#             derived_base = _pipeline_strip_suffix(list_name)
#             derived_start = ''
#             stem = Path(list_name).stem
#             if stem.startswith('list_events_'):
#                 stamp = stem[len('list_events_'):]
#                 try:
#                     dt = datetime.strptime(stamp, '%Y.%m.%d_%H.%M.%S')
#                     derived_start = dt.strftime('%Y-%m-%d_%H.%M.%S')
#                 except ValueError:
#                     derived_start = ''

#             add_timestamp = datetime.fromtimestamp(list_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
#             filler = {header: '' for header in csv_headers}
#             filler['basename'] = derived_base
#             if derived_start:
#                 filler['start_date'] = derived_start
#             filler['list_ev_name'] = list_name
#             filler['list_ev_add_date'] = add_timestamp
#             rows.append(filler)
#             existing_names.add(list_name)

#     with csv_path.open('w', newline='') as handle:
#         writer = csv.DictWriter(handle, fieldnames=csv_headers)
#         writer.writeheader()
#         writer.writerows(rows)


# _update_pipeline_csv_for_list_event()





# -----------------------------------------------------------------------------
# Create and save the PDF -----------------------------------------------------
# -----------------------------------------------------------------------------

if create_pdf:
    print(f"Creating PDF with all plots in {save_pdf_path}...")
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
                    pdf_save_rasterized_page(pdf, fig, bbox_inches='tight')  # Save figure tightly fitting the image
                    plt.close(fig)  # Close the figure after adding it to the PDF

        # Remove PNG files after creating the PDF
        for png in plot_list:
            try:
                os.remove(png)
                # print(f"Deleted {png}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")


# Erase all files in the figure_directory -------------------------------------------------
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory)

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))

# Erase the figure_directory
if os.path.exists(figure_directory):
    print("Removing figure directory...")
    os.rmdir(figure_directory)

# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    shutil.move(file_path, completed_file_path)
    now = time.time()
    os.utime(completed_file_path, (now, now))
    print("************************************************************")
    print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
    print("************************************************************")



# Store the current time at the end
end_execution_time_counting = datetime.now()
time_taken = (end_execution_time_counting - start_execution_time_counting).total_seconds() / 60
print(f"Time taken for the whole execution: {time_taken:.2f} minutes")

# mark_status_complete(status_csv_path, status_timestamp)

print("----------------------------------------------------------------------")
print("------------------- Finished list_events creation --------------------")
print("----------------------------------------------------------------------\n\n\n")








columns_to_keep = [
    # Timestamp and identifiers
    'datetime', 'original_tt', 'processed_tt', 'tracking_tt', 'definitive_tt',

    # New definitions
    'x', 'x_err', 'y', 'y_err', 'theta', 'theta_err', 'phi', 'phi_err', 's', 's_err',
    
    # Chisqs
    'chi_timtrack', 'chi_alternative',

    # Strip-level time and charge info (ordered by plane and strip)
    *[f'Q_P{p}s{s}' for p in range(1, 5) for s in range(1, 5)],
    
    # Strip-level time and charge info with crosstalk
    *[f'Q_P{p}s{s}_with_crstlk' for p in range(1, 5) for s in range(1, 5)]
]

reduced_df = definitive_df[columns_to_keep]





# Path to save the cleaned dataframe
# Create output directory if it does not exist /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/fitted_{basename_no_ext}.h5"
KEY = "df"  # HDF5 key name

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...



# Print all column names in the dataframe
print("Columns in the reduced_df dataframe:")
for col in reduced_df.columns:
    print(col)

# Remove the columns in the form "T*_T_sum_*", "T*_T_diff_*", "Q*_Q_sum_*", "Q*_Q_diff_*", do a loop from 1 to 4
cols_to_remove = []
for i_plane in range(1, 5):
    cols_to_remove.append(f'P{i_plane}_T_sum_final')
    cols_to_remove.append(f'P{i_plane}_T_diff_final')
    cols_to_remove.append(f'P{i_plane}_Q_sum_final')
    cols_to_remove.append(f'P{i_plane}_Q_diff_final')
    cols_to_remove.append(f'P{i_plane}_Y_final')
    
    cols_to_remove.append(f'alt_res_tdif_{i_plane}')
    for strip in range(1, 5):
        cols_to_remove.append(f'T{i_plane}_T_sum_{strip}')
        cols_to_remove.append(f'T{i_plane}_T_diff_{strip}')
        cols_to_remove.append(f'Q{i_plane}_Q_sum_{strip}')
        cols_to_remove.append(f'Q{i_plane}_Q_diff_{strip}')
reduced_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')




# Print all column names in the dataframe
print("Columns in the final dataframe:")
for col in reduced_df.columns:
    print(col)
    





# def _collect_columns(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
#     """Return all column names that match *pattern*."""
#     return [name for name in columns if pattern.match(name)]

# # Pattern for P1_Q_sum_*, P2_Q_sum_*, P3_Q_sum_*, P4_Q_sum_*
# Q_SUM_PATTERN = re.compile(r'^P[1-4]_Q_sum_.*$')

# # If Q*_F_* and Q*_B_* are zero for all cases, remove the row
# Q_cols = _collect_columns(reduced_df.columns, Q_SUM_PATTERN)
# reduced_df = reduced_df[(reduced_df[Q_cols] != 0).any(axis=1)]






print(f"Original number of events in the dataframe: {original_number_of_events}")
# Final number of events
final_number_of_events = len(reduced_df)
print(f"Final number of events in the dataframe: {final_number_of_events}")

# Data purity
data_purity = final_number_of_events / original_number_of_events * 100
print(f"Data purity is {data_purity:.2f}%")
global_variables['purity_of_data_percentage'] = data_purity


# End of the execution time
end_time_execution = datetime.now()
execution_time = end_time_execution - start_execution_time_counting
# In minutes
execution_time_minutes = execution_time.total_seconds() / 60
print(f"Total execution time: {execution_time_minutes:.2f} minutes")

# To save as metadata
filename_base = basename_no_ext
execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
data_purity_percentage = data_purity
total_execution_time_minutes = execution_time_minutes


# -------------------------------------------------------------------------------
# Execution metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

print("----------\nExecution metadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"Data purity percentage: {data_purity_percentage:.2f}%")
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes\n----------")

metadata_execution_csv_path = save_metadata(
    csv_path,
    {
        "filename_base": filename_base,
        "execution_timestamp": execution_timestamp,
        "data_purity_percentage": round(float(data_purity_percentage), 4),
        "total_execution_time_minutes": round(float(total_execution_time_minutes), 4),
    },
)
print(f"Metadata (execution) CSV updated at: {metadata_execution_csv_path}")


# -------------------------------------------------------------------------------
# Specific metadata ------------------------------------------------------------
# -------------------------------------------------------------------------------

global_variables["filename_base"] = filename_base
global_variables["execution_timestamp"] = execution_timestamp

# Print completely global_variables
print("----------\nAll global variables to be saved:")
for key, value in global_variables.items():
    print(f"{key}: {value}")
print("----------\n")

print("----------\nSpecific metadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"------------- Any other variable interesting -------------")
print("\n----------")

metadata_specific_csv_path = save_metadata(
    csv_path_specific,
    global_variables,
)
print(f"Metadata (specific) CSV updated at: {metadata_specific_csv_path}")

# Save to HDF5 file

working_df = reduced_df
working_df.to_parquet(OUT_PATH, engine="pyarrow", compression="zstd", index=False)
print(f"Listed dataframe saved to: {OUT_PATH}")

# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    shutil.move(file_path, completed_file_path)
    now = time.time()
    os.utime(completed_file_path, (now, now))
    print("************************************************************")
    print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
    print("************************************************************")

